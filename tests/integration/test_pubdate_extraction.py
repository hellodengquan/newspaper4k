"""Integration tests for publication date extraction (source competition).

These tests run the full Article pipeline (download -> parse -> to_json)
against local HTML fixtures, so no network access is needed. They verify the
contract implemented by ``newspaper.extractors.pubdate_extractor.PubdateExtractor``:

Candidate scores (higher wins, ties keep insertion order: URL, JSON-LD,
<time> tags, meta/other tags):

- 10  date in the article URL (``/2024/03/10/``)
- 10  JSON-LD ``@graph`` item ``datePublished``
-  9  JSON-LD top-level ``datePublished``
-  8  ``<time datetime>`` whose text contains "published" / "on:"
-  7  JSON-LD top-level ``dateCreated``
-  7  known ``<meta>`` publish-date tags (6 + 1 meta boost)
-  6  known non-meta publish-date tags (e.g. ``itemprop="datePublished"``)
-  5  plain ``<time datetime>``
- -2  penalty for meta/tag candidates dated in the future
- -1  penalty for meta/tag candidates older than 25 years

Time semantics contract:

- URL dates and timezone-less strings parse to naive datetimes.
- Offset-bearing strings keep their own UTC offset; the extractor must not
  normalize them to UTC or to the local timezone.
- ``Article.to_json()`` exports ``publish_date`` via ``datetime.isoformat()``,
  so the exported string must carry the same instant *and* the same offset
  as ``article.publish_date``.
- The future-date penalty compares against ``datetime.now()`` in the
  *process-local* timezone; missing date components (e.g. "2024-03") are
  filled from the current local date by dateutil.

All expectations below are derived from that contract, not from the
implementation's current behavior on any single input.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone, tzinfo
from types import SimpleNamespace

import pytest
from dateutil.parser import parse as date_parser

import dateutil.parser._parser as dateutil_parser_module
import newspaper.extractors.pubdate_extractor as pubdate_extractor_module
from newspaper.article import Article

#: Default frozen "now": 2026-11-15T12:00:00Z. Chosen so that the local date
#: is 2026-11-15 in UTC, America/New_York and Asia/Tokyo alike.
DEFAULT_INSTANT = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

#: An instant where the local date differs across timezones:
#: 2026-11-15 in Asia/Tokyo, but still 2026-11-14 in America/New_York.
BOUNDARY_INSTANT = datetime(2026, 11, 15, 2, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def freeze_now(monkeypatch):
    """Factory fixture: freeze the current time at an aware UTC instant.

    Patches both clocks the extractor (indirectly) relies on:

    - ``newspaper.extractors.pubdate_extractor.datetime`` (scoring's
      ``datetime.now()``), and
    - ``dateutil.parser._parser.datetime`` (dateutil fills missing date
      components, e.g. the day in "2024-03", from ``datetime.now()``).

    The frozen ``now()`` returns the *process-local* time at the frozen
    instant, so changing the TZ environment variable afterwards still has a
    well-defined, controllable effect.
    """

    def _freeze(instant: datetime = DEFAULT_INSTANT) -> datetime:
        assert instant.tzinfo is not None, "instant must be timezone-aware"

        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return instant.astimezone().replace(tzinfo=None)
                return instant.astimezone(tz)

        monkeypatch.setattr(pubdate_extractor_module, "datetime", FrozenDateTime)
        monkeypatch.setattr(
            dateutil_parser_module,
            "datetime",
            SimpleNamespace(
                datetime=FrozenDateTime,
                tzinfo=tzinfo,
                timezone=timezone,
                timedelta=timedelta,
            ),
        )
        return instant

    return _freeze


@pytest.fixture(autouse=True)
def _freeze_now_by_default(freeze_now):
    """Freeze time at DEFAULT_INSTANT for every test in this module.

    Tests that need a different instant call ``freeze_now(...)`` again.
    """
    freeze_now()


@pytest.fixture
def process_tz():
    """Factory fixture: set the process timezone (TZ env var + tzset)."""
    original = os.environ.get("TZ")

    def _set(tz_name: str) -> None:
        os.environ["TZ"] = tz_name
        time.tzset()

    yield _set

    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


def make_news_page(*, json_ld=(), meta=(), times=(), body_extra=""):
    """Assemble a local news page fixture from raw HTML snippets.

    Args:
        json_ld: iterable of JSON strings for <script type="application/ld+json"> blocks.
        meta: iterable of raw <meta ...> tags.
        times: iterable of raw <time ...>...</time> tags.
        body_extra: extra raw HTML appended inside the article body.
    """
    json_ld_blocks = "\n".join(f'<script type="application/ld+json">{script}</script>' for script in json_ld)
    meta_tags = "\n".join(meta)
    time_tags = "\n".join(times)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Fixture News Site</title>
  {meta_tags}
  {json_ld_blocks}
</head>
<body>
  <article>
    <h1>Fixture headline for pubdate extraction</h1>
    {time_tags}
    {body_extra}
    <p>Fixture news article body. The quick brown fox jumps over the lazy dog.
    This paragraph only exists so that the article pipeline finds some text to
    work with while the publication date candidates compete in the head.</p>
    <p>Second fixture paragraph with a little more text for the content
    extractor, again only to keep the full pipeline happy.</p>
  </article>
</body>
</html>"""


def parse_article(url: str, html: str) -> Article:
    """Run the full Article pipeline against a local HTML fixture."""
    article = Article(url, fetch_images=False)
    article.download(input_html=html)
    article.parse()
    return article


def assert_json_export_consistent(article: Article) -> None:
    """Verify Article -> JSON export keeps instant and timezone of publish_date.

    Catches two defect classes on top of the plain attribute assertions:

    - timezone dropped during export (date string kept, offset lost), and
    - export string disagreeing with the article object's datetime.
    """
    assert article.publish_date is not None, "precondition: a publish date was extracted"
    expected = article.publish_date.isoformat()

    as_dict = article.to_json(as_string=False)
    as_str = json.loads(article.to_json(as_string=True))
    assert as_dict["publish_date"] == expected
    assert as_str["publish_date"] == expected

    roundtrip = date_parser(as_dict["publish_date"])
    assert roundtrip == article.publish_date, "JSON export changed the exported instant"
    assert (roundtrip.tzinfo is None) == (article.publish_date.tzinfo is None), (
        "JSON export lost or invented timezone information"
    )
    if article.publish_date.tzinfo is not None:
        assert roundtrip.utcoffset() == article.publish_date.utcoffset(), "JSON export changed the UTC offset"


class TestSourcePriority:
    """Candidate priority: which source wins when several dates compete."""

    def test_url_date_beats_all_other_sources(self):
        """URL date (score 10) outranks JSON-LD (9), boosted <time> (8) and meta (7)."""
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2024-03-11T08:00:00+02:00"}'],
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
            times=['<time datetime="2024-03-12T09:00:00+03:00">Published on March 12, 2024</time>'],
        )
        article = parse_article("https://news.example.com/2024/03/10/story", html)

        assert article.publish_date == datetime(2024, 3, 10)
        assert article.publish_date.tzinfo is None, "URL dates must stay naive"
        assert_json_export_consistent(article)

    def test_jsonld_graph_date_ties_with_url_and_loses_by_insertion_order(self):
        """@graph datePublished also scores 10; the URL candidate was inserted first."""
        html = make_news_page(
            json_ld=[
                '{"@context": "https://schema.org", "@graph": '
                '[{"@type": "NewsArticle", "datePublished": "2024-03-11T08:00:00+02:00"}]}'
            ],
        )
        article = parse_article("https://news.example.com/2024/03/10/story", html)

        assert article.publish_date == datetime(2024, 3, 10)
        assert_json_export_consistent(article)

    def test_jsonld_wins_when_url_has_no_date(self):
        """Without a URL date, top-level JSON-LD datePublished (9) wins."""
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2024-03-11T08:00:00+02:00"}'],
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
            times=['<time datetime="2024-03-12T09:00:00+03:00">Published on March 12, 2024</time>'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2024, 3, 11, 8, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        assert article.publish_date.utcoffset() == timedelta(hours=2)
        assert_json_export_consistent(article)

    def test_boosted_time_tag_beats_meta_and_plain_time(self):
        """A <time> tag with 'Published on' text (8) beats meta (7) and plain <time> (5)."""
        html = make_news_page(
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
            times=[
                '<time datetime="2024-03-14T11:00:00+05:00">March 14, 2024</time>',
                '<time datetime="2024-03-12T09:00:00+03:00">Published on March 12, 2024</time>',
            ],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2024, 3, 12, 9, 0, 0, tzinfo=timezone(timedelta(hours=3)))
        assert_json_export_consistent(article)

    def test_meta_beats_non_meta_tag_candidate(self):
        """Known <meta> tags (6+1) beat known non-meta tag candidates (6)."""
        html = make_news_page(
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
            body_extra='<span itemprop="datePublished" datetime="2024-03-14T11:00:00+05:00">March 14</span>',
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2024, 3, 13, 10, 0, 0, tzinfo=timezone(timedelta(hours=4)))
        assert_json_export_consistent(article)


class TestTimeSemantics:
    """Time semantics: same instant in different offsets, naive vs cross-day."""

    def test_same_instant_different_offsets_picks_higher_score_and_keeps_offset(self):
        """JSON-LD and meta carry the *same instant* with different offsets.

        JSON-LD (9) must win over meta (7). Instant equality alone cannot
        distinguish the two sources (both are 2024-03-15T00:00:00Z), so the
        UTC offset is asserted separately: a wrong-source pick with the same
        instant would show +00:00 instead of +04:00.
        """
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2024-03-15T04:00:00+04:00"}'],
            meta=['<meta property="article:published_time" content="2024-03-15T00:00:00Z">'],
        )
        article = parse_article("https://news.example.com/story/", html)

        # Same instant, however it was produced:
        assert article.publish_date == datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
        # Source identity: the JSON-LD offset must survive:
        assert article.publish_date.tzinfo is not None
        assert article.publish_date.utcoffset() == timedelta(hours=4)
        assert_json_export_consistent(article)
        assert article.to_json(as_string=False)["publish_date"] == "2024-03-15T04:00:00+04:00"

    def test_naive_date_beats_cross_day_aware_time(self):
        """Naive JSON-LD date (9) beats an aware <time> (8) that is 'tomorrow' in UTC.

        2024-03-15T23:30:00-05:00 is 2024-03-16T04:30:00Z. The naive winner
        must stay naive and keep the calendar date 2024-03-15; an
        implementation normalizing to UTC would report the 16th.
        """
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2024-03-15"}'],
            times=['<time datetime="2024-03-15T23:30:00-05:00">Published on March 15, 2024</time>'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2024, 3, 15)
        assert article.publish_date.tzinfo is None, "timezone-less input must stay naive"
        assert article.publish_date.date().isoformat() == "2024-03-15"
        assert_json_export_consistent(article)
        assert article.to_json(as_string=False)["publish_date"] == "2024-03-15T00:00:00"

    def test_aware_cross_day_time_keeps_own_offset_and_date(self):
        """An aware cross-day JSON-LD time (9) beats a naive meta date (7).

        The winner must keep its own -05:00 offset and its local calendar
        date (2024-03-15), even though the instant is 2024-03-16 in UTC.
        """
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2024-03-15T23:30:00-05:00"}'],
            meta=['<meta property="article:published_time" content="2024-03-15">'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date.tzinfo is not None, "offset-bearing input must not lose its timezone"
        assert article.publish_date.utcoffset() == timedelta(hours=-5)
        assert article.publish_date.date().isoformat() == "2024-03-15"
        assert_json_export_consistent(article)
        assert article.to_json(as_string=False)["publish_date"] == "2024-03-15T23:30:00-05:00"


class TestBrokenCandidatesFallback:
    """Broken high-priority candidates must be skipped, not fatal."""

    def test_unparseable_url_date_falls_back_to_jsonld(self):
        """URL matches the date regex (2024/13/40) but is not a real date."""
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2024-03-11T08:00:00+02:00"}'],
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
        )
        article = parse_article("https://news.example.com/2024/13/40/story", html)

        assert article.publish_date == datetime(2024, 3, 11, 8, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        assert_json_export_consistent(article)

    def test_broken_jsonld_value_falls_back_to_time_tag(self):
        """An unparseable JSON-LD datePublished is dropped; boosted <time> (8) wins."""
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "not-a-real-date"}'],
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
            times=['<time datetime="2024-03-12T09:00:00+03:00">Published on March 12, 2024</time>'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2024, 3, 12, 9, 0, 0, tzinfo=timezone(timedelta(hours=3)))
        assert_json_export_consistent(article)

    def test_malformed_jsonld_script_is_skipped(self):
        """A syntactically invalid JSON-LD block must not break extraction."""
        html = make_news_page(
            json_ld=["{not valid json,,"],
            meta=['<meta property="article:published_time" content="2024-03-13T10:00:00+04:00">'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2024, 3, 13, 10, 0, 0, tzinfo=timezone(timedelta(hours=4)))
        assert_json_export_consistent(article)


class TestPartialAndFutureDates:
    """Year-month-only candidates, future-date and old-date scoring."""

    def test_year_month_only_candidate_fills_day_from_current_date(self, freeze_now, process_tz):
        """dateutil fills the missing day of "2024-03" from *today*.

        This is the existing contract: the extracted day legitimately depends
        on the current date, so the test pins it under a frozen clock instead
        of letting the environment decide.
        """
        process_tz("UTC")
        html = make_news_page(meta=['<meta property="article:published_time" content="2024-03">'])

        freeze_now(datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc))
        article = parse_article("https://news.example.com/story/", html)
        assert article.publish_date == datetime(2024, 3, 15)
        assert_json_export_consistent(article)

        freeze_now(datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc))
        article = parse_article("https://news.example.com/story/", html)
        assert article.publish_date == datetime(2024, 3, 20)
        assert_json_export_consistent(article)

    def test_future_meta_penalized_below_past_plain_time_tag(self):
        """Future meta (7-2=5) ties a plain <time> (5); <time> was inserted first."""
        html = make_news_page(
            meta=['<meta property="article:published_time" content="2026-12-25">'],
            times=['<time datetime="2026-11-10">November 10, 2026</time>'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2026, 11, 10)
        assert_json_export_consistent(article)

    def test_future_jsonld_is_not_penalized(self):
        """The future-date penalty only applies to meta/tag candidates.

        A future JSON-LD datePublished (9, no penalty) still beats a past
        meta candidate (7). This pins the documented asymmetry of the
        scoring contract.
        """
        html = make_news_page(
            json_ld=['{"@type": "NewsArticle", "datePublished": "2027-01-05T00:00:00+00:00"}'],
            meta=['<meta property="article:published_time" content="2026-11-10">'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2027, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
        assert_json_export_consistent(article)

    def test_very_old_meta_penalized_but_still_beats_plain_time(self):
        """Meta older than 25 years scores 7-1=6, still above plain <time> (5)."""
        html = make_news_page(
            meta=['<meta property="article:published_time" content="1990-06-15">'],
            times=['<time datetime="2026-11-10">November 10, 2026</time>'],
        )
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(1990, 6, 15)
        assert_json_export_consistent(article)

    def test_only_future_candidate_is_still_returned(self):
        """A penalized future meta (5) is better than no date at all."""
        html = make_news_page(meta=['<meta property="article:published_time" content="2026-12-25">'])
        article = parse_article("https://news.example.com/story/", html)

        assert article.publish_date == datetime(2026, 12, 25)
        assert_json_export_consistent(article)


class TestEnvironmentControl:
    """Frozen instant + process timezone: drift vs. legitimate change."""

    PAGE = None  # built per-test via make_news_page

    def _page_with_today_meta_and_past_time(self):
        return make_news_page(
            meta=['<meta property="article:published_time" content="2026-11-15">'],
            times=['<time datetime="2026-11-10">November 10, 2026</time>'],
        )

    @pytest.mark.parametrize("tz_name", ["UTC", "America/New_York", "Asia/Tokyo"])
    def test_fixed_instant_same_local_date_across_timezones(self, freeze_now, process_tz, tz_name):
        """At 2026-11-15T12:00Z the local date is the 15th in all three zones.

        The meta candidate dated 2026-11-15 is 'today' (score 7) everywhere,
        so the result must be identical across timezones. Any divergence here
        is environment drift, i.e. a defect.
        """
        freeze_now(DEFAULT_INSTANT)
        process_tz(tz_name)
        article = parse_article("https://news.example.com/story/", self._page_with_today_meta_and_past_time())

        assert article.publish_date == datetime(2026, 11, 15)
        assert_json_export_consistent(article)

    @pytest.mark.parametrize(
        ("tz_name", "expected"),
        [
            # Local date is 2026-11-15: meta is 'today' (7) and wins.
            ("Asia/Tokyo", datetime(2026, 11, 15)),
            # Local date is still 2026-11-14: meta is 'future' (7-2=5),
            # ties the plain <time> (5) which was inserted first.
            ("America/New_York", datetime(2026, 11, 10)),
        ],
    )
    def test_crossing_date_boundary_changes_result_legitimately(self, freeze_now, process_tz, tz_name, expected):
        """At 2026-11-15T02:00Z the local date differs across timezones.

        Unlike drift at a fixed local date, a changed winner here is the
        legitimate consequence of the future-date penalty depending on the
        process-local calendar date.
        """
        freeze_now(BOUNDARY_INSTANT)
        process_tz(tz_name)
        article = parse_article("https://news.example.com/story/", self._page_with_today_meta_and_past_time())

        assert article.publish_date == expected
        assert_json_export_consistent(article)
