"""Extracts the main article body from HTML using gravity scoring."""

import copy
import re
from functools import partial
from statistics import mean

import lxml
from lxml.html import HtmlElement

import newspaper.extractors.defines as defines
import newspaper.parsers as parsers
from newspaper.configuration import Configuration
from newspaper.text import StopWords

score_weights = {
    "bottom_negativescore_nodes": 0.25,
    "boost_score": 30,
    "parent_node": 1.0,
    "parent_parent_node": 0.4,
    "node_count_threshold": 15,
    "negative_score_threshold": 40,
    "negative_score_boost": 5.0,
    "boost_max_steps_from_node": 3,
    "boost_min_stopword_count": 5,
}

# Identifiers of the sources that can provide the article body. The value is
# stored on :any:`Article.body_strategy` and in the JSON output so that callers
# can tell where the extracted text came from.
BODY_STRATEGY_GRAVITY = "gravity_score"
BODY_STRATEGY_STRUCTURED = "structured_data"
BODY_STRATEGY_SEMANTIC = "semantic_container"

# A body with fewer words than this cannot stand as the article body: it is a
# caption, a teaser, or navigation. Structured/semantic sources are preferred.
MIN_BODY_WORDS = 50

# A scored body that is smaller than this fraction of the best alternative
# source is considered to be only a fragment of the article.
MIN_BODY_RATIO = 0.25

# JSON-LD / microdata properties that carry the article body.
_STRUCTURED_BODY_KEYS = ("articlebody", "articletext")

# Tags and ARIA roles that describe site navigation rather than article text.
_NAV_TAGS = ("nav", "aside", "footer", "header", "form", "menu", "script", "style", "noscript")
_NAV_ROLES = ("navigation", "banner", "contentinfo", "complementary", "search", "menu", "menubar")
_NAV_ATTR_RE = re.compile(
    r"(?:^|[-_\s])"
    r"(?:nav|navbar|navigation|menu|breadcrumb|breadcrumbs|footer|masthead|"
    r"sidebar|related|share|social|subscribe|search)"
    r"(?:$|[-_\s])",
    re.IGNORECASE,
)

# Block elements that can hold a paragraph of the body. Used to drop repeated
# content from fallback sources while keeping the original page order.
_BLOCK_XPATH = ".//p | .//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6 | .//li | .//blockquote"

# Elements that describe an article container semantically.
_SEMANTIC_CONTAINERS_XPATH = "//article | //main | //*[@role='main']"

get_stop_words = partial(parsers.get_attribute, attr="stop_words", type_=int, default=0)
get_word_count = partial(parsers.get_attribute, attr="word_count", type_=int, default=0)


class ArticleBodyExtractor:
    """Extracts the main article body from HTML using a gravity-score heuristic."""

    def __init__(self, config: Configuration):
        """Initialize the ArticleBodyExtractor.

        Args:
            config (Configuration): Configuration object controlling extraction behavior.
        """
        self.config = config
        self.top_node = None
        self.top_node_complemented = None
        self.stopwords: StopWords | None = None
        # Source of the body that was finally selected. One of
        # ``BODY_STRATEGY_GRAVITY``, ``BODY_STRATEGY_STRUCTURED`` or
        # ``BODY_STRATEGY_SEMANTIC``, or None if no body was found.
        self.body_strategy: str | None = None

    def parse(self, doc: HtmlElement):
        """Parse the document and identify the best article body node.

        The gravity score is the default strategy. If the scored candidate is
        missing, too short, dominated by links, or clearly smaller than a
        complete alternative, the extractor falls back to the article body
        offered by structured data (JSON-LD/microdata), then to a semantic
        article container (``<article>``, ``<main>`` or ``role="main"``).

        Args:
            doc (HtmlElement): The document root element to parse.
        """
        self.stopwords = StopWords(self.config.language)
        self.top_node = self.calculate_best_node(doc)

        if isinstance(doc, lxml.etree._Element):
            strategy, node = self.select_body_strategy(doc, self.top_node)
            self.body_strategy = strategy
            self.top_node = node if node is not None else self.top_node
        else:
            # Documents that are not element trees cannot provide fallback
            # sources; keep the scored node unchanged.
            self.body_strategy = BODY_STRATEGY_GRAVITY if self.top_node is not None else None

        self.top_node_complemented = self.complement_with_siblings(self.top_node)

    def select_body_strategy(self, doc: HtmlElement, scored_node: HtmlElement | None):
        """Pick the most complete body source available on the page.

        The gravity-scored node keeps priority as long as it is a plausible
        article body and not a small fragment of a better source. Otherwise the
        fallback candidates are compared and the most complete one wins. On equal
        length, structured data is preferred over a semantic container.

        Args:
            doc (HtmlElement): Document root element.
            scored_node (HtmlElement | None): Node selected by gravity scoring.

        Returns:
            tuple[str | None, HtmlElement | None]: The chosen strategy name and
            the corresponding node. ``(None, None)`` when nothing usable exists.
        """
        gravity_candidate = (BODY_STRATEGY_GRAVITY, scored_node) if scored_node is not None else None

        # Build the fallback candidates lazily. A cheap size estimate tells us
        # whether any alternative could beat or match the scored body; if not,
        # the default path is kept and no extra work is done.
        if (
            gravity_candidate is not None
            and self.is_plausible_body(scored_node)
            and not self.has_larger_fallback_source(doc, scored_node)
        ):
            return gravity_candidate

        candidates: list[tuple[str, HtmlElement]] = []
        if gravity_candidate is not None:
            candidates.append(gravity_candidate)

        candidates.extend(self.structured_body_candidates(doc))
        if not self.page_declares_non_article(doc):
            candidates.extend(self.semantic_body_candidates(doc))

        # A complete body always wins over a fragment. Only when no candidate
        # is complete do we fall back to the longest fragment available, which
        # is still better than returning nothing.
        complete = [item for item in candidates if self.is_plausible_body(item[1])]
        if complete:
            best_word_count = max(self.body_word_count(node) for _, node in complete)
            if gravity_candidate is not None and self.is_plausible_body(scored_node):
                if self.body_word_count(scored_node) >= best_word_count * MIN_BODY_RATIO:
                    return gravity_candidate
            for candidate in complete:
                if self.body_word_count(candidate[1]) == best_word_count:
                    return candidate
            return complete[0]

        if not candidates:
            return None, None

        # No complete source exists. Keep the longest fragment, preferring the
        # gravity-scored node on ties (the historical default).
        best_word_count = max(self.body_word_count(node) for _, node in candidates)
        for candidate in candidates:
            if self.body_word_count(candidate[1]) == best_word_count:
                return candidate
        return candidates[0]

    def has_larger_fallback_source(self, doc: HtmlElement, scored_node: HtmlElement) -> bool:
        """Return True if a fallback source can be as large as the scored body.

        This is a cheap pre-check: it only measures the raw text of structured
        bodies and semantic containers, without cloning or cleaning them. When
        every fallback source is smaller than the scored body, the scored body
        would win anyway, so the expensive fallback construction is skipped.

        Args:
            doc (HtmlElement): Document root element.
            scored_node (HtmlElement): Body selected by gravity scoring.

        Returns:
            bool: True when the fallback sources must be inspected.
        """
        scored_words = self.body_word_count(scored_node)
        # When the scored body is too short to stand as the article body, any
        # fallback source is worth inspecting. Otherwise only a larger source
        # could displace it.
        min_words = 0 if scored_words < MIN_BODY_WORDS else scored_words

        for obj in parsers.get_ld_json_object(doc):
            for text in self._extract_structured_texts(obj):
                if self._estimate_words(text) > min_words:
                    return True

        for element in doc.xpath(_SEMANTIC_CONTAINERS_XPATH):
            # Only the text of the block elements is kept by the fallback, so
            # measuring them avoids counting navigation and footer text.
            blocks_words = sum(self._estimate_words("".join(block.itertext())) for block in element.xpath(_BLOCK_XPATH))
            if blocks_words > min_words:
                return True

        return False

    def _estimate_words(self, text: str) -> int:
        """Approximate a word count without language-specific tokenization."""
        return len((text or "").split())

    def page_declares_non_article(self, doc: HtmlElement) -> bool:
        """Return True when the page identifies itself as an article listing.

        Index and section pages often expose a ``role="main"`` container full
        of teaser links. Falling back to it would mix site navigation and
        headline lists into the article text, so those pages are excluded from
        the semantic fallback. A page counts as a listing when it declares an
        ``og:type`` other than an article, or when its structured data only
        describes a collection page, and it has no article container or
        article markup anywhere on the page.

        Args:
            doc (HtmlElement): Document root element.

        Returns:
            bool: True when the semantic fallback should be skipped.
        """
        if doc.xpath("//article"):
            return False
        if self.page_has_article_structured_data(doc):
            return False

        og_types = {(meta.get("content") or "").strip().lower() for meta in parsers.get_metatags(doc, value="og:type")}
        # "website" is the og:type used by homepages and section listings. Other
        # non-article og:type values (e.g. "video.other") are left alone so a
        # legitimately odd page can still use its semantic container.
        if "website" in og_types:
            return True

        return self.page_is_collection(doc)

    def page_is_collection(self, doc: HtmlElement) -> bool:
        """Return True when JSON-LD only describes a collection page."""
        collection_types = {"collectionpage", "webpage", "website", "searchresultspage", "itemlist"}

        def visit(value) -> set[str]:
            found: set[str] = set()
            if isinstance(value, dict):
                type_value = value.get("@type")
                if isinstance(type_value, str):
                    found.add(type_value.lower())
                elif isinstance(type_value, list):
                    found.update(item.lower() for item in type_value if isinstance(item, str))
                for item in value.values():
                    found |= visit(item)
            elif isinstance(value, list):
                for item in value:
                    found |= visit(item)
            return found

        types: set[str] = set()
        for obj in parsers.get_ld_json_object(doc):
            types |= visit(obj)

        return bool(types & collection_types) and not bool(
            types & {"article", "newsarticle", "blogposting", "techarticle", "report"}
        )

    def page_has_article_structured_data(self, doc: HtmlElement) -> bool:
        """Return True if JSON-LD/microdata describes an article."""
        article_types = {
            "article",
            "newsarticle",
            "blogposting",
            "techarticle",
            "scholarlyarticle",
            "socialmediaposting",
            "report",
        }

        def visit(value) -> bool:
            if isinstance(value, dict):
                type_value = value.get("@type")
                if isinstance(type_value, str) and type_value.lower() in article_types:
                    return True
                if isinstance(type_value, list) and any(
                    isinstance(item, str) and item.lower() in article_types for item in type_value
                ):
                    return True
                return any(visit(item) for item in value.values())
            if isinstance(value, list):
                return any(visit(item) for item in value)
            return False

        if any(visit(obj) for obj in parsers.get_ld_json_object(doc)):
            return True

        return bool(
            doc.xpath(
                "//*[@itemtype][contains(translate(@itemtype, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                " 'abcdefghijklmnopqrstuvwxyz'), 'article')]"
            )
        )

    def body_word_count(self, node: HtmlElement | None) -> int:
        """Count the words of the node text using the configured tokenizer."""
        if node is None:
            return 0
        text = parsers.get_text(node)
        if not text:
            return 0
        if self.stopwords is None:
            return len(text.split())
        return len(list(self.stopwords.tokenizer(text)))

    def is_plausible_body(self, node: HtmlElement) -> bool:
        """Return True if the node can stand as the article body by itself.

        A node is rejected when it has no text, is too short, or when most of
        its text lives inside links (navigation, menus, related lists).
        """
        if node is None:
            return False
        text = parsers.get_text(node)
        if not text:
            return False
        if parsers.is_highlink_density(node, self.config.language):
            return False
        return self.body_word_count(node) >= MIN_BODY_WORDS

    def structured_body_candidates(self, doc: HtmlElement) -> list[tuple[str, HtmlElement]]:
        """Collect article bodies exposed as structured data.

        Both JSON-LD (``articleBody``/``articleText``) and microdata
        (``itemprop="articleBody"``) are supported. Navigation blocks and
        repeated paragraphs are pruned so the body stays contiguous.

        Args:
            doc (HtmlElement): Document root element.

        Returns:
            list[tuple[str, HtmlElement]]: ``(strategy, node)`` pairs, longest
            first. Nodes are flat paragraph containers owned by the extractor.
        """
        nodes: list[HtmlElement] = []

        for obj in parsers.get_ld_json_object(doc):
            for text in self._extract_structured_texts(obj):
                node = self._node_from_text(text)
                if node is not None:
                    nodes.append(node)

        microdata = doc.xpath(
            "//*[@itemprop][contains(translate(@itemprop, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            " 'abcdefghijklmnopqrstuvwxyz'), 'articlebody') or"
            " contains(translate(@itemprop, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            " 'abcdefghijklmnopqrstuvwxyz'), 'articletext')]"
        )
        for element in microdata:
            node = self._clone_article_container(element)
            if node is None:
                node = self._node_from_text(parsers.get_text(element))
            if node is not None:
                nodes.append(node)

        nodes = [node for node in nodes if self.body_word_count(node) > 0]
        nodes.sort(key=self.body_word_count, reverse=True)

        return [(BODY_STRATEGY_STRUCTURED, node) for node in nodes]

    def _extract_structured_texts(self, obj) -> list[str]:
        """Recursively pull body strings out of a JSON-LD object."""
        texts: list[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and key.lower() in _STRUCTURED_BODY_KEYS and isinstance(value, str):
                    texts.append(value)
                else:
                    texts.extend(self._extract_structured_texts(value))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(self._extract_structured_texts(item))
        return texts

    def _node_from_text(self, text: str) -> HtmlElement | None:
        """Build a flat paragraph container from a body string.

        Structured bodies may contain HTML markup or plain text with line
        breaks. The text blocks are extracted in document order and rebuilt as
        consecutive paragraphs, so the result stays contiguous and does not
        depend on the original wrapper elements.
        """
        if not text or not text.strip():
            return None

        blocks = self._text_blocks(text)
        if not blocks:
            return None

        container = parsers.fromstring("<div></div>")
        container.extend(blocks)
        return self._strip_duplicates(container)

    def _text_blocks(self, text: str) -> list[HtmlElement]:
        """Split a body string into ordered paragraph elements."""
        text = text.strip()
        if "<" in text and ">" in text:
            node = parsers.fromstring(f"<div>{text}</div>")
            if node is not None:
                blocks = [self._clone_block(item) for item in self._outermost_blocks(node.xpath(_BLOCK_XPATH))]
                blocks = [block for block in blocks if parsers.get_text(block)]
                if blocks:
                    return blocks

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) < 2:
            paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        return [parsers.create_element(tag="p", text=part) for part in paragraphs]

    def semantic_body_candidates(self, doc: HtmlElement) -> list[tuple[str, HtmlElement]]:
        """Collect semantic article containers from the page.

        ``<article>``, ``<main>`` and ``role="main"`` elements are considered.
        Navigation blocks nested inside them are pruned so that menus and
        related links do not leak into the article text.

        Args:
            doc (HtmlElement): Document root element.

        Returns:
            list[tuple[str, HtmlElement]]: ``(strategy, node)`` pairs, most
            complete first. Nodes are off-tree copies, safe to clean.
        """
        containers = doc.xpath(_SEMANTIC_CONTAINERS_XPATH)
        nodes = []
        for container in containers:
            node = self._clone_article_container(container)
            if node is None:
                continue
            if self.body_word_count(node) <= 0:
                continue
            nodes.append(node)

        nodes.sort(key=self.body_word_count, reverse=True)
        return [(BODY_STRATEGY_SEMANTIC, node) for node in nodes]

    def _clone_article_container(self, container: HtmlElement) -> HtmlElement | None:
        """Copy a semantic container, drop navigation, and flatten paragraphs.

        The container is copied first so the original document is never
        modified. Navigation blocks are removed, then the remaining text is
        rebuilt as consecutive paragraphs. This keeps the body continuous,
        avoids duplicated sections, and shields the fallback from downstream
        cleaners that use the original tree's scoring attributes.
        """
        node = copy.deepcopy(container)
        node.tail = None

        for tag in _NAV_TAGS:
            parsers.remove(parsers.get_tags(node, tag=tag))

        for bad in node.xpath("//*[@role]"):
            role = (bad.get("role") or "").strip().lower()
            if role in _NAV_ROLES:
                parsers.remove(bad)

        for bad in node.xpath("//*[@id or @class]"):
            attrs = f"{bad.get('id', '')} {bad.get('class', '')}"
            if _NAV_ATTR_RE.search(attrs) and parsers.is_highlink_density(bad, self.config.language):
                parsers.remove(bad)

        block_nodes = self._outermost_blocks(node.xpath(_BLOCK_XPATH))
        blocks = [self._clone_block(item) for item in block_nodes]
        blocks = [block for block in blocks if parsers.get_text(block)]

        if not blocks:
            # Containers without paragraph-like tags: keep the remaining text
            # as a single block so no content is lost.
            text = parsers.get_text(node)
            if not text:
                return None
            blocks = [parsers.create_element(tag="p", text=text)]

        body = parsers.fromstring("<div></div>")
        body.extend(blocks)

        return self._strip_duplicates(body)

    def _clone_block(self, node: HtmlElement) -> HtmlElement:
        """Copy a block element as a paragraph, keeping inline content.

        Inline elements (notably links) are preserved so that link density can
        still be evaluated on the fallback body, while the block is detached
        from its original parents and siblings.
        """
        block = copy.deepcopy(node)
        block.tail = ""
        if block.tag not in ("p", "blockquote", "pre", "li"):
            block.tag = "p"

        for child in block.xpath(".//*[not(self::a or self::span or self::strong or self::b or self::em or self::i)]"):
            parsers.drop_tags(child)

        for element in [block, *block.xpath(".//*")]:
            for attr in list(element.attrib):
                if attr not in ("href", "src", "alt", "title"):
                    del element.attrib[attr]

        return block

    def _outermost_blocks(self, block_nodes: list[HtmlElement]) -> list[HtmlElement]:
        """Filter out block elements nested inside another block element.

        Taking only the outermost blocks keeps each paragraph text once, in
        document order, even when a list item contains its own paragraphs.
        """
        block_ids = {id(node) for node in block_nodes}
        result = []
        for node in block_nodes:
            parent = node.getparent()
            nested = False
            while parent is not None:
                if id(parent) in block_ids:
                    nested = True
                    break
                parent = parent.getparent()
            if not nested:
                result.append(node)
        return result

    def _strip_duplicates(self, node: HtmlElement) -> HtmlElement | None:
        """Remove repeated paragraphs, keeping the first occurrence.

        Duplicate paragraphs are common in pages that render the same body in
        more than one container. Removing them keeps the extracted text
        contiguous and avoids the same text leaking in twice.

        Args:
            node (HtmlElement): Container to clean in place.

        Returns:
            HtmlElement | None: The same node, or None if it has no text left.
        """
        seen: set[str] = set()
        for candidate in node.xpath(_BLOCK_XPATH):
            key = self._paragraph_key(parsers.get_text(candidate))
            if not key:
                continue
            if key in seen:
                parsers.remove(candidate)
            else:
                seen.add(key)

        return node if parsers.get_text(node) else None

    def _paragraph_key(self, text: str) -> str:
        """Normalize a paragraph for duplicate detection."""
        text = re.sub(r"[^\w\s]", "", text or "").lower()
        return re.sub(r"\s+", " ", text).strip()

    def calculate_best_node(self, doc):
        """Calculate the node most likely to contain the article body.

        Args:
            doc: Document root element.

        Returns:
            HtmlElement | None: Best candidate node, or None if not found.
        """
        top_node = None
        self.boost_highly_likely_nodes(doc)

        nodes_with_text = self.compute_features(doc)

        # process the tree from bottom up. farthest nodes first
        nodes_with_text.sort(
            key=lambda node: parsers.get_attribute(node, "node_level", type_=int, default=0),
            reverse=True,
        )

        parent_nodes = self.compute_gravity_scores(nodes_with_text)

        if parent_nodes:
            parent_nodes.sort(key=parsers.get_node_gravity_score, reverse=True)
            for candidate in parent_nodes:
                # Do not choose a top node that will be emptied by post_cleanup
                if parsers.get_text(candidate):
                    top_node = candidate
                    break

        return top_node

    def compute_gravity_scores(self, nodes_with_text):
        """Computes the gravity score for each node in the list.
        And propagate the score to its parents and grandparents.

        Args:
            nodes_with_text (list): list of candidate nodes that have meaningful text

        Returns:
            list: list of nodes with gravity score
        """
        parent_nodes = []

        boost_discount = 1

        nodes_count = len(nodes_with_text)
        negative_scoring = 0

        bottom_negativescore_nodes = nodes_count * score_weights["bottom_negativescore_nodes"]

        for i, node in enumerate(nodes_with_text):
            boost_score = 0
            # boost score decays with distance from the top node
            if self.is_boostable(node):
                boost_score = score_weights["boost_score"] / boost_discount
                boost_discount += 1

            # nodes_number
            if nodes_count > score_weights["node_count_threshold"]:
                # higher number of possible top nodes
                dist_from_end = float(nodes_count - i)
                if dist_from_end <= bottom_negativescore_nodes:
                    booster = bottom_negativescore_nodes - dist_from_end
                    boost_score = -(booster**2)
                    negscore = abs(boost_score) + negative_scoring
                    if negscore > score_weights["negative_score_threshold"]:
                        boost_score = score_weights["negative_score_boost"]

            stop_word_count = get_stop_words(node)

            upscore = stop_word_count + boost_score

            parent_node = node.getparent()

            self.update_score(parent_node, upscore)
            self.update_node_count(parent_node, 1)

            parent_nodes.append(parent_node)

            # Parent of parent node
            parent_parent_node = parent_node.getparent() if parent_node is not None else None

            self.update_node_count(parent_parent_node, 1)
            self.update_score(parent_parent_node, upscore * score_weights["parent_parent_node"])

            parent_nodes.append(parent_parent_node)

        parent_nodes = [x for x in set(parent_nodes) if x is not None]

        return parent_nodes

    def compute_features(self, doc):
        """Compute stop-word and link-density features for all candidate nodes.

        Args:
            doc: Document root element.

        Returns:
            list: Nodes with sufficient stop-word count and low link density.
        """
        candidates = []
        nodes_to_check = self.nodes_to_check(doc)
        nodes_to_check.sort(key=parsers.get_level, reverse=True)

        for node in nodes_to_check:
            # exclude nodes that are in this list

            text_content = parsers.get_text(node)
            if not text_content:
                continue

            word_stats = self.stopwords.get_stopword_count(text_content)
            high_link_density = parsers.is_highlink_density(node, self.config.language)

            child_nodes = node.xpath(".//*[@stop_words>0]")
            children_stop_words = sum(get_stop_words(c) for c in child_nodes)
            children_word_count = sum(get_word_count(c) for c in child_nodes)

            parsers.set_attribute(node, "stop_words", word_stats.stop_word_count - children_stop_words)
            parsers.set_attribute(node, "word_count", word_stats.word_count - children_word_count)
            parsers.set_attribute(node, "is_highlink_density", 1 if high_link_density else 0)
            parsers.set_attribute(node, "node_level", parsers.get_level(node))

            if word_stats.stop_word_count > 2 and not high_link_density:
                candidates.append(node)

        return candidates

    def nodes_to_check(self, doc):
        """Returns a list of nodes we want to search
        on like paragraphs and tables
        """
        nodes_to_check = []
        for tag in ["p", "pre", "td", "article", "div"]:
            if tag == "div":
                items = []
                for attr in [
                    "articlebody",
                    "article",
                    "story",
                ]:
                    items += parsers.get_tags(
                        doc,
                        tag=tag,
                        attribs={"id": attr},
                        attribs_match="word",
                        ignore_dashes=True,
                    )
                    items += parsers.get_tags(
                        doc,
                        tag=tag,
                        attribs={"class": attr},
                        attribs_match="word",
                        ignore_dashes=True,
                    )
                for class_ in ["paragraph"]:
                    items += parsers.get_tags_regex(doc, tag=tag, attribs={"class": class_})
                if len(items) == 0 and len(nodes_to_check) < 5:
                    items = parsers.get_tags(doc, tag=tag)
                items = set(items)  # remove duplicates
            else:
                items = parsers.get_tags(doc, tag=tag)
            nodes_to_check += items

        # Do not miss some Article Bodies or Article Sections
        # for itemprop in ['articleBody', 'articlebody', 'articleText',
        #                       'articleSection']:
        #     items = [item  for item in parsers.get_tags(
        #         doc, attribs={"itemprop": itemprop}, attribs_match="word"
        #     ) if item not in nodes_to_check]
        #     nodes_to_check.extend(items)

        return nodes_to_check

    def is_boostable(self, node):
        """A lot of times the first paragraph might be the caption under an image
        so we'll want to make sure if we're going to boost a parent node that
        it should be connected to other paragraphs, at least for the first n
        paragraphs so we'll want to make sure that the next sibling is a
        paragraph and has at least some substantial weight to it.
        """
        max_stepsaway_from_node = score_weights["boost_max_steps_from_node"]

        nodes = self.walk_siblings(node)
        for current_node in nodes[:max_stepsaway_from_node]:
            if current_node.tag != node.tag:
                continue
            stop_word_count = parsers.get_attribute(current_node, "stop_words", type_=int, default=0)
            if stop_word_count > score_weights["boost_min_stopword_count"]:
                return True
        return False

    def boost_highly_likely_nodes(self, doc: HtmlElement):
        """Set a bias score for all nodes under most likely
        article containers. This way we can find articles
        that have little text.

        Args:
            doc (HtmlElement): Document to be checked
        """
        candidates = []
        for tag in ["p", "pre", "td", "article", "div"]:
            candidates.extend(parsers.get_tags(doc, tag=tag))

        for e in candidates:
            boost = self.is_highly_likely(e)
            if boost > 0:
                self.update_score(e, boost * score_weights["parent_node"])
                # for child in e.iterdescendants():
                # TODO: find an optimum value
                #     self.update_score(child, boost // 10)

    def is_highly_likely(self, node: HtmlElement) -> int:
        """Checks if the node is a well known tag + attributes combination
        for article body containers. This way we can deliver even small
        article bodies with high link density
        see: https://finance.yahoo.com/m/
            0edc9aaa-e3ba-3178-be5a-f8c16fbffff2/warren-buffett-stocks-shaky.html

        Args:
            node (HtmlElement): Node to check

        Returns:
            bool: True if node could be an article body top node
        """

        def is_tag_match(node, tag_dict):
            if node.tag != tag_dict.get("tag", node.tag):
                return False
            for k, v in tag_dict.items():
                if k in ["tag", "score_boost"]:
                    continue
                if v.startswith("re:"):
                    v = v[3:]
                    if not re.search(v, node.get(k, ""), re.IGNORECASE):
                        return False
                elif node.get(k, "").lower() != v.lower():
                    return False
            return True

        scores = []
        for tag in defines.ARTICLE_BODY_TAGS:
            if is_tag_match(node, tag):
                scores.append(tag["score_boost"])
        if scores:
            return max(scores)

        return 0

    def update_score(self, node, add_to_score):
        """Adds a score to the gravityScore Attribute we put on divs
        we'll get the current score then add the score we're passing
        in to the current.
        """
        if node is None:
            return
        new_score = parsers.get_node_gravity_score(node) + add_to_score
        parsers.set_attribute(node, "gravityScore", str(new_score))

    def update_node_count(self, node, add_to_count):
        """Stores how many decent nodes are under a parent node"""
        if node is None:
            return
        new_count = float(node.get("gravityNodes", 0)) + add_to_count
        parsers.set_attribute(node, "gravityNodes", str(new_count))

    def add_siblings(self, top_node):
        """Add preceding siblings with plausible content to a copy of the top node.

        Args:
            top_node: The detected top article node.

        Returns:
            HtmlElement: A deep copy of the top node with sibling paragraphs prepended.
        """
        res_node = copy.deepcopy(top_node)
        baseline_score = self.get_normalized_score(top_node)
        results = self.walk_siblings(top_node)
        for current_node in results:
            ps = self.get_plausible_content(current_node, baseline_score)
            for p in ps:
                res_node.insert(0, p)
        return res_node

    def get_plausible_content(self, node, baseline_score, score_weight=0.3):
        """Create list of (off-the-tree) paragraphs from a node (most likely a
        sibling of the top node) that are plausible to be part of the
        article body. We use the stop word count as a score of plausibility.
        We accept only paragraphs with a score higher than the baseline score
        weighted by the score_weight parameter.
        The baseline score is a normalized score of the top node.
        """
        if isinstance(node, lxml.etree.CommentBase | lxml.etree.EntityBase | lxml.etree.PIBase):
            return []

        if node.tag == "p" and node.text and not parsers.is_highlink_density(node, self.config.language):
            element = copy.deepcopy(node)
            element.tail = ""
            return [element]

        paragraphs = parsers.get_tags(node, tag="p")
        result = []
        if not paragraphs:
            return result

        for paragraph in paragraphs:
            stop_word_count = parsers.get_attribute(paragraph, "stop_words", type_=int, default=0)
            if stop_word_count <= 0:
                continue
            if parsers.is_highlink_density(paragraph, self.config.language):
                continue

            if stop_word_count > baseline_score * score_weight:
                text = parsers.get_text(paragraph)
                element = parsers.create_element(tag="p", text=text)
                result.append(element)

        return result

    def get_normalized_score(self, top_node):
        """We could have long articles that have tons of paragraphs
        so if we tried to calculate the base score against
        the total text score of those paragraphs it would be unfair.
        So we need to normalize the score based on the average scoring
        of the paragraphs within the top node.
        For example if our total score of 10 paragraphs was 1000
        but each had an average value of 100 then 100 should be our base.
        """
        nodes_to_check = parsers.get_tags(top_node, tag="p")

        scores = [parsers.get_node_gravity_score(node) for node in nodes_to_check]
        scores = [score for score in scores if score > 0]  # filter out 0 scores

        return mean(scores) if scores else float("inf")

    def walk_siblings(self, node):
        """Returns preceding siblings in reverse order (nearest sibling is first)"""
        return list(node.itersiblings(preceding=True))

    def complement_with_siblings(self, node: HtmlElement) -> HtmlElement:
        """Adds surrounding relevant siblings to the top node.
        Attention, it generates off-the-tree node.

        Args:
            node (HtmlElement): Top node detected

        Returns:
            HtmlElement: off the tree node complemented with siblings
        """
        if node is None:
            return node
        # TODO: test if there is a problem with siblings AFTER the top node
        # node_complemented = self.add_siblings(node)

        node_complemented = self.add_same_level_candidates(node)

        # for e in node_complemented.getchildren():
        #     if e.tag != "p":
        #         if parsers.is_highlink_density(e):
        #             parsers.remove(e)
        return node_complemented

    def add_same_level_candidates(self, node):
        """Adds any siblings that may have a decent score to this node"""
        tree = node.getroottree()

        node_level = parsers.get_level(node)
        # base_score = self.get_normalized_score(node)
        base_score = parsers.get_node_gravity_score(node)

        candidates = parsers.get_nodes_at_level(tree.getroot(), node_level)

        assert node in candidates, "node not in candidates"

        # Create the new tree as a HTML document. Otherwise it will
        # be created as Element tree and cleaners won't work
        new_tree = parsers.fromstring("<html><body></body></html>")
        new_node = new_tree.find("body")

        for n in candidates:
            if n == node:
                new_node.append(copy.deepcopy(node))
                continue

            # avoid adding nodes that do not resemble the top node
            if n.tag != node.tag:
                continue

            score = parsers.get_node_gravity_score(n)

            if score > base_score * 0.3 and not parsers.is_highlink_density(n, self.config.language):
                new_node.append(copy.deepcopy(n))
                continue

            # content_items = self.get_plausible_content(n, base_score)
            # new_node.extend(content_items)

        return new_node
