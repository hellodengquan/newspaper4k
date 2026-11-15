# ruff: noqa: D100, D103
"""Image candidate resolution, deduplication and top image selection.

The tests use the local news fixture ``tests/data/html/image_candidates.html``
which mixes scheme-relative URLs, relative paths, lazy loading attributes,
repeated ``srcset`` entries, query parameter variants and a broken image
server response. No network access is performed: the image download step is
replaced by a fake that reports the size of each requested address and
records how often each address was fetched.
"""

from types import SimpleNamespace

import pytest

from newspaper.article import Article
from newspaper.extractors.image_extractor import (
    ImageExtractor,
    page_base,
    parse_srcset,
    resolve_page_base,
)
from newspaper.configuration import Configuration
from tests.conftest import get_data

ARTICLE_URL = "https://news.example.com/2024/03/storm-cancellations"
FIXTURE = "image_candidates"

# Sizes reported by the fake image server. URLs that are absent from the map
# respond with a non-image (or broken) response, i.e. the fetch returns None.
SIZES = {
    "https://news.example.com/assets/meta/hero-meta.jpg?w=1200": (1200, 800),
    "https://cdn.example.com/photos/hero.jpg?w=1200": (1200, 800),
    "https://news.example.com/assets/images/hero.jpg?w=400": (400, 300),
    "https://news.example.com/2024/03/photos/second.jpg": (900, 600),
    "https://news.example.com/2024/03/photos/second.jpg?w=1600": (1600, 1067),
    "https://news.example.com/assets/images/lazy.jpg": (800, 600),
    "https://news.example.com/assets/images/fallback.jpg": (640, 480),
    "https://news.example.com/assets/images/fallback.jpg?utm_source=rss": (640, 480),
    "https://news.example.com/assets/images/social.png": (64, 64),
    "https://news.example.com/assets/images/logo.png": (120, 40),
}


class FakeImageServer:
    """Records image downloads and answers with a fixed size per URL."""

    def __init__(self, sizes=None):
        self.sizes = dict(SIZES if sizes is None else sizes)
        self.requests: list[str] = []

    def fetch(self, url, referer):  # noqa: ARG002 - signature of _fetch_image
        self.requests.append(url)
        size = self.sizes.get(url)
        if size is None:
            return None
        return SimpleNamespace(size=size)


def make_extractor(fetch_images=True, server=None):
    config = Configuration()
    config.fetch_images = fetch_images
    extractor = ImageExtractor(config)
    if server is not None:
        extractor._fetch_image = server.fetch
    return extractor


def parse_fixture(monkeypatch, sizes=None):
    server = FakeImageServer(sizes)
    article = Article(url=ARTICLE_URL, fetch_images=True)
    monkeypatch.setattr("newspaper.extractors.image_extractor.ImageExtractor._fetch_image", server.fetch)
    article.download(input_html=get_data(FIXTURE, "html"))
    article.parse()
    return article, server


def test_parse_srcset_keeps_commas_inside_urls():
    value = (
        "https://media.example.com/prod/x.jpg?c=16x9&q=w_480,c_fill 480w, "
        "https://media.example.com/prod/x.jpg?c=16x9&q=w_920,c_fill 920w"
    )

    assert parse_srcset(value) == [
        "https://media.example.com/prod/x.jpg?c=16x9&q=w_480,c_fill",
        "https://media.example.com/prod/x.jpg?c=16x9&q=w_920,c_fill",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/a.jpg 1x, /b.jpg 2x", ["/a.jpg", "/b.jpg"]),
        (" /a.jpg 1x ,  /b.jpg 2x ", ["/a.jpg", "/b.jpg"]),
        ("/a.jpg 1x,/b.jpg 2x", ["/a.jpg", "/b.jpg"]),
        ("/a.jpg", ["/a.jpg"]),
        ("/a.jpg,/b.jpg", ["/a.jpg,/b.jpg"]),  # commas belong to the URL
        ("", []),
    ],
)
def test_parse_srcset_edge_cases(value, expected):
    assert parse_srcset(value) == expected


def test_resolve_page_base_fills_missing_scheme():
    assert resolve_page_base("www.test.com") == "https://www.test.com"
    assert resolve_page_base("//cdn.example.com/a") == "https://cdn.example.com/a"
    assert resolve_page_base("https://example.com/a") == "https://example.com/a"
    assert resolve_page_base(None) == ""
    assert resolve_page_base("") == ""


def test_page_base_honours_base_href():
    doc = pytest.importorskip("lxml.html").fromstring('<html><head><base href="/static/"></head></html>')

    assert page_base(doc, "https://news.example.com/2024/03/story") == "https://news.example.com/static/"


def test_image_list_is_resolved_deduplicated_and_ordered(monkeypatch):
    article, _ = parse_fixture(monkeypatch)

    assert article.images == [
        "https://news.example.com/assets/images/logo.png",
        "https://cdn.example.com/photos/hero.jpg?w=1200",
        "https://news.example.com/assets/images/hero.jpg?w=400",
        "https://news.example.com/2024/03/photos/second.jpg",
        "https://news.example.com/2024/03/photos/second.jpg?w=1600",
        "https://news.example.com/assets/images/lazy.jpg",
        "https://news.example.com/assets/images/broken.jpg",
        "https://news.example.com/assets/images/fallback.jpg",
        "https://news.example.com/assets/images/fallback.jpg?utm_source=rss",
        "https://news.example.com/assets/images/social.png",
    ]


def test_meta_image_wins_when_it_passes_size_validation(monkeypatch):
    article, server = parse_fixture(monkeypatch)

    assert article.top_image == "https://news.example.com/assets/meta/hero-meta.jpg?w=1200"
    assert server.requests == ["https://news.example.com/assets/meta/hero-meta.jpg?w=1200"]


def test_top_image_falls_back_in_page_order_when_candidates_fail(monkeypatch):
    sizes = {key: value for key, value in SIZES.items() if not key.startswith("https://news.example.com/assets/meta")}
    # The hero image and the first lazy variants respond with invalid images.
    sizes.pop("https://cdn.example.com/photos/hero.jpg?w=1200", None)
    sizes.pop("https://news.example.com/assets/images/hero.jpg?w=400", None)

    article, _ = parse_fixture(monkeypatch, sizes)

    # logo -> too small, hero variants -> invalid, then the second image.
    assert article.top_image == "https://news.example.com/2024/03/photos/second.jpg"

    # The image list is unaffected by validation failures and keeps page order.
    assert article.images[1:4] == [
        "https://cdn.example.com/photos/hero.jpg?w=1200",
        "https://news.example.com/assets/images/hero.jpg?w=400",
        "https://news.example.com/2024/03/photos/second.jpg",
    ]


def test_each_normalized_address_is_downloaded_once(monkeypatch):
    article, server = parse_fixture(monkeypatch)

    assert article.top_image == "https://news.example.com/assets/meta/hero-meta.jpg?w=1200"
    duplicates = [url for url in server.requests if server.requests.count(url) > 1]
    assert duplicates == []


def test_validation_cache_survives_candidate_sources(monkeypatch):
    sizes = {key: value for key, value in SIZES.items() if not key.startswith("https://news.example.com/assets/meta")}
    article, server = parse_fixture(monkeypatch, sizes)

    # The fallback image is declared three times with two normalized forms
    # (protocol-relative duplicate merged, tracking query kept). Each form is
    # fetched at most once even though several candidates share the element.
    fallback_requests = [url for url in server.requests if "fallback.jpg" in url]
    assert len(fallback_requests) == len(set(fallback_requests))
    assert article.top_image == "https://cdn.example.com/photos/hero.jpg?w=1200"


def test_repeated_parse_yields_stable_results_and_downloads(monkeypatch):
    article, first_server = parse_fixture(monkeypatch)
    first_images = list(article.images)
    first_top = article.top_image
    first_requests = list(first_server.requests)

    second_server = FakeImageServer()
    monkeypatch.setattr("newspaper.extractors.image_extractor.ImageExtractor._fetch_image", second_server.fetch)
    article.parse()

    assert article.images == first_images
    assert article.top_image == first_top
    # The article owns one extractor; its validation cache survives the
    # re-parse, so an already validated address is never downloaded again.
    assert first_requests
    assert second_server.requests == []


def test_failed_candidate_does_not_change_later_candidates(monkeypatch):
    sizes = {key: value for key, value in SIZES.items() if not key.startswith("https://news.example.com/assets/meta")}
    sizes["https://cdn.example.com/photos/hero.jpg?w=1200"] = (100, 80)
    sizes["https://news.example.com/assets/images/hero.jpg?w=400"] = (100, 80)

    article, _ = parse_fixture(monkeypatch, sizes)
    failing = "https://cdn.example.com/photos/hero.jpg?w=1200"
    assert failing in article.images
    assert article.images.index(failing) == 1
    assert article.top_image == "https://news.example.com/2024/03/photos/second.jpg"


def test_parse_without_top_node_prefers_explicit_source_over_lazy_only(monkeypatch):
    server = FakeImageServer(
        {
            "https://example.com/lazy-only.jpg": (100, 80),
            "https://example.com/real.jpg": (800, 600),
        }
    )
    extractor = make_extractor(server=server)
    doc = pytest.importorskip("lxml.html").fromstring(
        '<html><body><img data-src="/lazy-only.jpg"><img src="/real.jpg"></body></html>'
    )

    extractor.parse(doc, None, "https://example.com/news/story")

    assert extractor.top_image == "https://example.com/real.jpg"
    # The lazy-only candidate is only requested once the explicit source
    # fails validation; here it is not needed.
    assert "https://example.com/lazy-only.jpg" not in server.requests


def test_parse_without_top_node_falls_back_to_lazy_only_candidate(monkeypatch):
    server = FakeImageServer(
        {
            "https://example.com/real.jpg": (100, 80),
            "https://example.com/lazy-only.jpg": (800, 600),
        }
    )
    extractor = make_extractor(server=server)
    doc = pytest.importorskip("lxml.html").fromstring(
        '<html><body><img src="/real.jpg" data-src="/lazy-only.jpg"></body></html>'
    )

    extractor.parse(doc, None, "https://example.com/news/story")

    assert extractor.top_image == "https://example.com/lazy-only.jpg"
    assert extractor.images == ["https://example.com/real.jpg", "https://example.com/lazy-only.jpg"]


def test_query_parameter_variants_are_not_merged(monkeypatch):
    article, _ = parse_fixture(monkeypatch)

    assert "https://news.example.com/assets/images/fallback.jpg" in article.images
    assert "https://news.example.com/assets/images/fallback.jpg?utm_source=rss" in article.images


def test_protocol_relative_duplicate_is_merged(monkeypatch):
    article, _ = parse_fixture(monkeypatch)

    # ``//news.example.com/assets/images/fallback.jpg`` resolves to the same
    # absolute URL as ``/assets/images/fallback.jpg`` and keeps the first spot.
    assert article.images.count("https://news.example.com/assets/images/fallback.jpg") == 1


def test_scheme_less_article_url_still_resolves_candidates(monkeypatch):
    """Legacy callers pass ``www.example.com`` without a scheme."""
    extractor = make_extractor(fetch_images=False)
    doc = pytest.importorskip("lxml.html").fromstring(
        '<html><body><img src="/a.jpg"><img src="//cdn.example.com/b.jpg"></body></html>'
    )

    extractor.parse(doc, None, "www.example.com")

    assert extractor.images == ["https://www.example.com/a.jpg", "https://cdn.example.com/b.jpg"]


def test_fetch_images_false_keeps_meta_image_top_image(monkeypatch):
    article = Article(url=ARTICLE_URL, fetch_images=False)
    article.download(input_html=get_data(FIXTURE, "html"))
    article.parse()

    assert article.top_image == "https://news.example.com/assets/meta/hero-meta.jpg?w=1200"
    assert len(article.images) == 10
