"""Extracts images from article HTML, including top image and meta image."""

import logging
import re
import urllib.parse
from copy import copy
from dataclasses import dataclass

import requests
from lxml.html import HtmlElement
from PIL import Image, ImageFile

import newspaper.extractors.defines as defines
import newspaper.parsers as parsers
from newspaper import urls
from newspaper.configuration import Configuration
from newspaper.network import is_public_url, session
from newspaper.urls import urljoin_if_valid

log = logging.getLogger(__name__)

# Values that never point to a usable image resource and are therefore
# dropped before any candidate is resolved against the page context.
_IGNORED_URL_PREFIXES = (
    "data:",
    "javascript:",
    "about:",
    "blob:",
    "mailto:",
    "#",
)

# Attributes that commonly hold the real (lazy loaded) image URL without
# containing the substring "src" in their name.
_LAZY_URL_ATTRS = frozenset(
    {
        "data-original",
        "data-lazy",
        "data-url",
        "data-image",
        "data-img",
        "data-thumb",
        "data-thumbnail",
        "data-full",
        "data-large",
    }
)


@dataclass
class ImageCandidate:
    """A single image URL candidate found in the page.

    Attributes:
        url (str): The candidate resolved to an absolute URL using the page
            context.
        element (HtmlElement): The element the candidate was found on. Used
            to order candidates by their distance to the article body.
    """

    url: str
    element: HtmlElement


def _is_srcset_attribute(name: str) -> bool:
    """Return True if the attribute holds a srcset descriptor list."""
    return name.lower().endswith("srcset")


def _is_url_attribute(name: str) -> bool:
    """Return True if the attribute may hold an image URL.

    Image sources are exposed through ``src`` and its variants
    (``data-src``, ``data-lazy-src`` ...) as well as through a handful of
    lazy loading attributes that do not mention ``src`` at all.
    """
    lowered = name.lower()
    return "src" in lowered or lowered in _LAZY_URL_ATTRS


def _is_usable_url(value: str) -> bool:
    """Return True if ``value`` may be resolved into an image URL.

    Placeholder schemes (``data:``, ``javascript:`` ...) are rejected, and so
    are attribute values that cannot be a resource reference at all (for
    example ``data-src-id="123"``); a URL always contains at least one of the
    characters below.
    """
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.lower().startswith(_IGNORED_URL_PREFIXES):
        return False
    return any(char in stripped for char in ("/", ".", "?", "&", "=", "%", ":"))


def parse_srcset(value: str) -> list[str]:
    """Split a ``srcset`` attribute into its individual URL candidates.

    The value is parsed the way browsers do: a URL is the run of
    non-whitespace characters, its optional descriptor ends at the next comma,
    and commas that belong to the URL itself (for example CNN image service
    parameters such as ``w_480,c_fill``) are therefore preserved. Trailing
    commas after a URL are treated as the candidate separator.

    Args:
        value (str): The raw value of a ``srcset`` (or ``data-srcset``)
            attribute.

    Returns:
        list[str]: The URLs in the order they appear in the attribute.
    """
    candidates: list[str] = []
    position = 0
    length = len(value)
    while position < length:
        # Skip separators and whitespace between candidates.
        while position < length and (value[position].isspace() or value[position] == ","):
            position += 1
        if position >= length:
            break

        start = position
        while position < length and not value[position].isspace():
            position += 1
        url = value[start:position].rstrip(",")

        # Skip the descriptor of this candidate; it ends at the separator.
        while position < length and value[position] != ",":
            position += 1

        url = url.strip()
        if url:
            candidates.append(url)
    return candidates


def resolve_page_base(article_url: str | None) -> str:
    """Return an absolute base URL usable for resolving page references.

    Fixtures and legacy callers sometimes pass a scheme-less article URL
    such as ``www.example.com/story``. Protocol-relative, relative and
    scheme-less references cannot be made absolute without a scheme, so a
    missing scheme defaults to ``https``.

    Args:
        article_url (str | None): The article URL as provided by the caller.

    Returns:
        str: The base URL to join page references against.
    """
    if not article_url:
        return ""
    article_url = article_url.strip()
    if article_url.startswith("//"):
        return f"https:{article_url}"
    parsed = urllib.parse.urlsplit(article_url)
    if parsed.scheme:
        return article_url
    if article_url.startswith("/"):
        # Root-relative base: nothing to make absolute without a host.
        return article_url
    host = article_url.split("/", 1)[0]
    if "." in host and " " not in host:
        # Scheme-less host such as ``www.example.com/news/story``; relative
        # references can only be resolved once the scheme is known.
        return f"https://{article_url}"
    return article_url


def page_base(doc: HtmlElement | None, article_url: str | None) -> str:
    """Return the base URL the page itself declares, resolved with the URL.

    Browsers resolve relative image references against ``<base href>`` when
    present, falling back to the document URL. Mirroring that keeps relative
    and protocol-relative candidates meaningful.

    Args:
        doc (HtmlElement | None): Root element of the HTML document.
        article_url (str | None): The article URL as provided by the caller.

    Returns:
        str: The effective base URL for the document.
    """
    base_url = resolve_page_base(article_url)
    if doc is None:
        return base_url
    try:
        base_elements = doc.xpath("//base[@href]")
    except (AttributeError, TypeError):
        return base_url
    if not base_elements:
        return base_url

    declared = (base_elements[0].get("href") or "").strip()
    if not declared:
        return base_url
    if not base_url:
        return resolve_page_base(declared)
    resolved = urljoin_if_valid(base_url, declared)
    return resolved or base_url


def node_distance(node1: HtmlElement, node2: HtmlElement) -> int:
    """Return the DOM distance between two nodes of the same document.

    Args:
        node1 (HtmlElement): First node.
        node2 (HtmlElement): Second node.

    Returns:
        int: Number of tree steps separating the nodes.
    """
    path1 = node1.getroottree().getpath(node1).split("/")
    path2 = node2.getroottree().getpath(node2).split("/")
    for i, (step1, step2) in enumerate(zip(path1, path2, strict=False)):
        if step1 != step2:
            return len(path1[i:]) + len(path2[i:])

    return abs(len(path1) - len(path2))


class ImageExtractor:
    """Extractor class for images in articles. Getting top image,
    image list, favicon, etc.
    """

    def __init__(self, config: Configuration) -> None:
        """Initialize the ImageExtractor.

        Args:
            config (Configuration): Configuration object controlling extraction behavior.
        """
        self.config = config
        self.top_image: str | None = None
        self.meta_image: str | None = None
        self.images: list[str] = []
        self.favicon: str | None = None
        self._chunksize = 1024
        # Validation results (True/False) keyed by the normalized URL. The
        # cache lives as long as the extractor, so the same address is
        # downloaded at most once even if parsing happens repeatedly or the
        # address shows up again through a different candidate source.
        self._size_cache: dict[str, bool] = {}
        # Candidate cache shared by the image list and the top image so both
        # derive from exactly the same resolution and deduplication pass.
        self._candidate_doc: HtmlElement | None = None
        self._candidate_base: str | None = None
        self._candidates: list[ImageCandidate] = []

    def parse(self, doc: HtmlElement, top_node: HtmlElement, article_url: str) -> None:
        """Main method to extract images from a document.

        All candidates (``src``, lazy loading attributes and ``srcset``
        entries) are resolved to absolute URLs with the page context, then
        deduplicated by their normalized form while keeping the first
        occurrence. The top image is selected from the same candidate set.

        Args:
            doc (HtmlElement): Root element of the full HTML document.
            top_node (HtmlElement): The top article body node.
            article_url (str): URL of the article, used to resolve relative URLs.
        """
        self._candidate_doc = None
        self._candidate_base = None
        self._candidates = []

        base_url = page_base(doc, article_url)

        self.favicon = self._get_favicon(doc)

        self.meta_image = self._get_meta_image(doc)
        if self.meta_image:
            self.meta_image = urljoin_if_valid(base_url, self.meta_image)

        candidates = self._candidates_for(doc, article_url)
        if candidates:
            self.images = [candidate.url for candidate in candidates]
        else:
            # Keep the historical behavior for documents without usable
            # image candidates (also covers callers that stub _get_images).
            self.images = [
                urljoin_if_valid(base_url, u)
                for u in self._get_images(doc)  # Tried to use top_node, but images
                # were not found in some cases (times_001.html)
                if u and u.strip()
            ]
        self.top_image = self._get_top_image(doc, top_node, article_url)

    def _get_favicon(self, doc: HtmlElement) -> str:
        """Extract the favicon from a website http://en.wikipedia.org/wiki/Favicon
        <link rel="shortcut icon" type="image/png" href="favicon.png" />
        <link rel="icon" type="image/png" href="favicon.png" />
        """
        meta = parsers.get_tags(doc, tag="link", attribs={"rel": "icon"}, attribs_match="substring")
        if meta:
            favicon = parsers.get_attribute(meta[0], "href")
            return favicon or ""
        return ""

    def _get_meta_image(self, doc: HtmlElement) -> str:
        """Extract image from the meta tags of the document."""
        candidates: list[tuple[str, int]] = []
        for elem in defines.META_IMAGE_TAGS:
            if "|" in elem["value"]:
                items = parsers.get_tags_regex(doc, tag=elem["tag"], attribs={elem["attr"]: elem["value"]})
            else:
                items = parsers.get_tags(
                    doc,
                    tag=elem["tag"],
                    attribs={elem["attr"]: elem["value"]},
                    attribs_match="exact",
                )

            candidates.extend((el.get(elem["content"]), elem["score"]) for el in items)

        candidates = [c for c in candidates if c[0]]

        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[0][0] if candidates else ""

    def _get_images(self, doc: HtmlElement) -> list[str]:
        """Return one preferred raw source per image element (legacy API).

        Args:
            doc (HtmlElement): Root element of the HTML document.

        Returns:
            list[str]: Raw, unresolved URLs, one per usable image element.
        """

        def get_src(image):
            # account for src, data-src and other attributes
            srcs = [image.attrib.get(x) for x in image.attrib if "src" in x]
            srcs = [x for x in srcs if x and not x.startswith("data:")]

            srcs.sort(key=lambda x: 0 if x.lower().startswith("http") else 1)

            return srcs[0] if srcs else None

        images = [get_src(x) for x in parsers.get_tags(doc, tag="img")]
        images = [x for x in images if x]

        return images

    @staticmethod
    def _element_urls(element: HtmlElement) -> list[str]:
        """Return the raw image URLs declared on a single element.

        Attributes are visited in document order. ``srcset`` style
        attributes are expanded into their individual candidates at the
        position of the attribute.

        Args:
            element (HtmlElement): The element to inspect.

        Returns:
            list[str]: Raw (unresolved) candidate URLs.
        """
        raw_urls: list[str] = []
        for name, value in element.attrib.items():
            if not value:
                continue
            if _is_srcset_attribute(name):
                raw_urls.extend(parse_srcset(value))
            elif _is_url_attribute(name):
                raw_urls.append(value)
        return [url for url in (raw.strip() for raw in raw_urls) if _is_usable_url(url)]

    @staticmethod
    def _normalized_key(url: str) -> str:
        """Return the deduplication key of a resolved URL."""
        return urls.normalize_url(url)

    @staticmethod
    def _image_elements(doc: HtmlElement) -> list[HtmlElement]:
        """Return the elements that contribute image candidates.

        Only ``img`` elements are considered: they carry the actual image to
        display, while sibling ``<source>`` entries are alternative encodings
        selected by the browser and are not part of the article's image list.

        Args:
            doc (HtmlElement): Root element of the HTML document.

        Returns:
            list[HtmlElement]: Image carrying elements in page order.
        """
        return parsers.get_tags(doc, tag="img")

    def _candidates_for(self, doc: HtmlElement, article_url: str) -> list[ImageCandidate]:
        """Return the cached candidates for ``doc`` and ``article_url``.

        Args:
            doc (HtmlElement): Root element of the HTML document.
            article_url (str): Page URL used to resolve relative candidates.

        Returns:
            list[ImageCandidate]: Unique absolute candidates in page order.
        """
        base_url = page_base(doc, article_url)
        if self._candidate_doc is not doc or self._candidate_base != base_url:
            self._candidates = self._collect_candidates(doc, base_url)
            self._candidate_doc = doc
            self._candidate_base = base_url
        return self._candidates

    def _collect_candidates(self, doc: HtmlElement, base_url: str) -> list[ImageCandidate]:
        """Collect page image candidates in document order.

        Every candidate is resolved against the page base URL first, and only
        then deduplicated by its normalized form. The first occurrence of a
        normalized URL wins and keeps its original position.

        Args:
            doc (HtmlElement): Root element of the HTML document.
            base_url (str): Page URL used to resolve relative candidates.

        Returns:
            list[ImageCandidate]: Unique absolute candidates in page order.
        """
        candidates: list[ImageCandidate] = []
        seen: set[str] = set()
        for element in self._image_elements(doc):
            for raw_url in self._element_urls(element):
                absolute = urljoin_if_valid(base_url, raw_url)
                if not absolute:
                    continue
                key = self._normalized_key(absolute)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(ImageCandidate(url=absolute, element=element))
        return candidates

    @staticmethod
    def _has_usable_src(element: HtmlElement) -> bool:
        """Return True if the element declares a non-placeholder ``src``."""
        src = element.get("src")
        return bool(src) and _is_usable_url(src)

    def _get_top_image(self, doc: HtmlElement, top_node: HtmlElement, article_url: str) -> str:
        """Select the top image from the shared candidate set.

        Candidates are ordered before any download happens, so a candidate
        that fails validation only moves the selection forward; it never
        reorders or deduplicates the remaining candidates.

        Args:
            doc (HtmlElement): Root element of the full HTML document.
            top_node (HtmlElement): The top article body node.
            article_url (str): URL of the article, used to resolve relative URLs.

        Returns:
            str: The selected image URL, or an empty string if none passed
            validation.
        """
        # If fetch_images is False, return meta_image without downloading for validation
        if not self.config.fetch_images:
            return self.meta_image if self.meta_image else ""

        # If fetch_images is True, validate image sizes by downloading
        if self.meta_image and self._validate_image(self.meta_image, article_url):
            return self.meta_image

        candidates = self._candidates_for(doc, article_url)
        if top_node is not None:
            # Stable sort: equal distances keep page order, and candidates of
            # the same element keep their declaration order.
            ordered = sorted(candidates, key=lambda candidate: node_distance(top_node, candidate.element))
        else:
            # Without an article body reference, elements that declare an
            # explicit source are tried first so a lazy-only placeholder does
            # not shadow a real image. Page order is preserved inside each
            # group, and lazy-only candidates remain as a fallback.
            ordered = sorted(candidates, key=lambda candidate: 0 if self._has_usable_src(candidate.element) else 1)

        for candidate in ordered:
            if self._validate_image(candidate.url, article_url):
                return candidate.url

        return ""

    def _validate_image(self, url: str, referer: str | None) -> bool:
        """Validate an image, requesting each normalized address once.

        Args:
            url (str): The absolute image URL.
            referer (str | None): Referer used for the request.

        Returns:
            bool: True if the image passes the size constraints.
        """
        key = self._normalized_key(url)
        if key not in self._size_cache:
            self._size_cache[key] = self._check_image_size(url, referer)
        return self._size_cache[key]

    def _check_image_size(self, url: str, referer: str | None) -> bool:
        img = self._fetch_image(
            url,
            referer,
        )
        if not img:
            return False

        width, height = img.size

        if self.config.top_image_settings["min_width"] > width:
            return False
        if self.config.top_image_settings["min_height"] > height:
            return False
        if self.config.top_image_settings["min_area"] > width * height:
            return False

        if (
            re.search(r"(logo|sprite)", url, re.IGNORECASE)
            and self.config.top_image_settings["min_area"] > width * height / 10
        ):
            return False

        return True

    def _fetch_image(self, url: str, referer: str | None) -> Image.Image | None:
        def clean_url(url):
            """Url quotes unicode data out of urls"""
            if not isinstance(url, str):
                return url

            url = url.encode("utf8")
            url = "".join([urllib.parse.quote(c) if ord(c) >= 127 else c for c in url.decode("utf-8")])
            return url

        requests_params = copy(self.config.requests_params)
        requests_params["headers"]["Referer"] = referer
        max_retries = self.config.top_image_settings["max_retries"]

        cur_try = 0
        url = clean_url(url)
        if not url or not url.startswith(("http://", "https://")):
            return None

        allow_private = self.config.allow_private_hosts
        requests_params.pop("allow_redirects", None)

        response = None
        while True:
            try:
                if not allow_private and not is_public_url(url):
                    log.warning("blocked non-public image url: %s", url)
                    return None
                response = session.get(
                    url,
                    stream=True,
                    allow_redirects=False,
                    **requests_params,
                )
                hops = 0
                while response.is_redirect and hops < 5:
                    target = urllib.parse.urljoin(url, response.headers.get("Location", ""))
                    response.raw.close()
                    if not allow_private and not is_public_url(target):
                        log.warning("blocked non-public image redirect: %s", target)
                        return None
                    url = target
                    hops += 1
                    response = session.get(
                        url,
                        stream=True,
                        allow_redirects=False,
                        **requests_params,
                    )
                if response.is_redirect:
                    return None

                content_type = response.headers.get("Content-Type")

                if not content_type or "image" not in content_type.lower():
                    return None

                p = ImageFile.Parser()
                new_data = response.raw.read(self._chunksize)
                while not p.image and new_data:
                    try:
                        p.feed(new_data)
                    except (OSError, ValueError) as e:
                        log.warning(
                            "error %s while fetching: %s refer: %s",
                            str(e),
                            url,
                            requests_params["headers"].get("Referer"),
                        )
                        return None
                    except Exception as e:
                        # For some favicon.ico images, the image is so small
                        # that our PIL feed() method fails a length test.
                        is_favicon = urls.url_to_filetype(url) == "ico"
                        if not is_favicon:
                            raise e
                        return None
                    new_data = response.raw.read(self._chunksize)
                return p.image
            except requests.exceptions.RequestException:
                cur_try += 1
                if cur_try >= max_retries:
                    log.warning(
                        "error while fetching: %s refer: %s",
                        url,
                        requests_params["headers"].get("Referer"),
                    )
                    return None
            finally:
                if response is not None:
                    response.raw.close()
                    if response.raw._connection:
                        response.raw._connection.close()
