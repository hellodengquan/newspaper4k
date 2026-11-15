# ruff: noqa: D100, D101, D103
"""Integration tests for publication date extraction source competition.

These tests run the full Article pipeline (download from a local HTML
fixture -> parse -> publish_date -> to_json) against pages where several
date sources compete: URL dates, JSON-LD structured data, <time> tags and
meta tags.

Expected values are derived independently from the extraction contract
implemented by ``newspaper.extractors.pubdate_extractor.PubdateExtractor``:

* candidate scores: URL date 10, JSON-LD ``@graph`` datePublished 10,
  JSON-LD top-level datePublished 9, ``<time>`` with "published"/"on:"
  text 8, JSON-LD dateCreated 7, meta tags 6 + 1 (meta boost) = 7,
  other known tags 6, plain ``<time>`` 5;
* meta candidates dated in the future get -2, candidates older than
  25 years get -1;
* ties are broken by insertion order (URL -> JSON-LD -> <time> -> meta)
  because the candidate sort is stable;
* ``Article.to_json`` exports datetimes via ``datetime.isoformat()``.

Time control: ``freeze_now`` pins ``datetime.now()`` both in the extractor
and in dateutil's partial-date default fill, so results are reproducible.
Environment drift (process timezone changes at a fixed moment) must not
change any result, while moving the frozen moment across a date boundary
legitimately changes outcomes that depend on "now".
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone, tzinfo
from types import SimpleNamespace

import pytest

import dateutil.parser._parser as dateutil_parser
import newspaper.extractors.pubdate_extractor as pubdate_module
from newspaper.article import Article
from tests import conftest

# The fixed "current moment" used by default: all fixture dates are in the
# recent past relative to it, so no future/age penalties apply unless a
# test explicitly freezes a different moment.
FROZEN_NOW = datetime(2024, 3, 20, 12, 0, 0)

URL_WITH_DATE = "https://example.com/2024/03/15/competing-sources"
URL_WITH_BROKEN_DATE = "https://example.com/2024/02/30/broken-url-date"  # Feb 30 matches the URL date regex but is unparseable
URL_WITHOUT_DATE = "https://example.com/news/story"


def load_article(fixture_name: str, url: str) -> Article:
    """Run the full article pipeline against a local HTML fixture."""
    html_content = conftest.get_data(fixture_name, "html")
    article = Article(url=url, fetch_images=False)
    article.download(input_html=html_content)
    article.parse()
    return article


def assert_json_export_consistent(article: Article) -> None:
    """The JSON export must carry the same moment and timezone as the
    article object: no dropped timezone, no shifted date, no normalization."""
    payload = json.loads(article.to_json())
    exported = payload["publish_date"]
    assert isinstance(exported, str)
    parsed_back = datetime.fromisoformat(exported)
    assert parsed_back == article.publish_date
    assert parsed_back.utcoffset() == article.publish_date.utcoffset()
    if article.publish_date.tzinfo is None:
        # a naive datetime must not gain a timezone on export
        assert parsed_back.tzinfo is None
    # the dict export carries the same ISO string as the string export
    assert article.to_json(as_string=False)["publish_date"] == exported


@pytest.fixture
def freeze_now(monkeypatch):
    """Freeze "now" for the pubdate extractor and for dateutil.

    The extractor calls ``datetime.now()`` for the future/age penalties,
    and dateutil fills missing fields of partial dates (e.g. "2024-03")
    from the current date. Both are patched so tests are deterministic.
    """

    def _freeze(value: datetime) -> None:
        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return value
                return value.replace(tzinfo=tz)

        monkeypatch.setattr(pubdate_module, "datetime", FrozenDateTime)
        monkeypatch.setattr(
            dateutil_parser,
            "datetime",
            SimpleNamespace(datetime=FrozenDateTime, tzinfo=tzinfo),
        )

    return _freeze


@pytest.fixture
def process_timezone():
    """Change the process timezone (TZ) for the duration of a test."""

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


class TestSourceCompetition:
    """Several valid date sources on the same page: the candidate
    priority order decides, independently of the date values."""

    def test_url_date_wins_over_jsonld_time_and_meta(self, freeze_now):
        freeze_now(FROZEN_NOW)
        article = load_article("pubdate_competing_sources", URL_WITH_DATE)

        # The URL date (score 10, inserted first) wins the tie against the
        # JSON-LD @graph date (score 10) and outranks the top-level JSON-LD
        # date (9), the boosted <time> tag (8) and the meta tag (7).
        # The top-level JSON-LD candidate shares the calendar date
        # 2024-03-15 with the URL date but is timezone-aware; selecting the
        # wrong source would keep the date string while changing the moment
        # and the timezone, so the exact value AND naivety are asserted.
        assert article.publish_date == datetime(2024, 3, 15)
        assert article.publish_date.tzinfo is None

        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-15T00:00:00"
        assert_json_export_consistent(article)

    def test_same_instant_different_offsets(self, freeze_now):
        freeze_now(FROZEN_NOW)
        article = load_article("pubdate_same_instant_offsets", URL_WITHOUT_DATE)

        # JSON-LD datePublished (score 9) and the <time> tag (score 8)
        # denote the SAME instant with different offsets and different
        # local dates. The higher-scored JSON-LD representation must be
        # selected, keeping its own offset rather than normalizing to UTC.
        assert article.publish_date == datetime(2024, 3, 16, 4, 30, tzinfo=timezone.utc)
        assert article.publish_date.utcoffset() == timedelta(hours=-5)

        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-15T23:30:00-05:00"
        assert_json_export_consistent(article)

    def test_aware_crossday_time_beats_naive_date(self, freeze_now):
        freeze_now(FROZEN_NOW)
        article = load_article("pubdate_naive_date_and_crossday_time", URL_WITHOUT_DATE)

        # The boosted <time> tag (score 8) carries a timezone-aware time
        # whose UTC date (2024-03-16) crosses the day boundary relative to
        # the naive meta date (2024-03-15, score 7). The aware cross-day
        # time wins and keeps its timezone through the JSON export.
        assert article.publish_date == datetime(2024, 3, 16, 4, 30, tzinfo=timezone.utc)
        assert article.publish_date.tzinfo is not None

        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-16T04:30:00+00:00"
        assert_json_export_consistent(article)


class TestFallbackAndScoring:
    """Corrupted high-priority candidates, partial dates and future dates."""

    def test_broken_url_date_falls_back_to_jsonld(self, freeze_now):
        freeze_now(FROZEN_NOW)
        # The URL contains 2024/02/30: it matches the strict URL date
        # regex but is not a real calendar day, so it yields no candidate
        # and the JSON-LD datePublished (score 9) must take over.
        article = load_article("pubdate_broken_url_date", URL_WITH_BROKEN_DATE)

        assert article.publish_date == datetime(2024, 3, 10, 12, 0, tzinfo=timezone.utc)

        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-10T12:00:00+00:00"
        assert_json_export_consistent(article)

    def test_broken_jsonld_candidates_fall_back(self, freeze_now):
        freeze_now(FROZEN_NOW)
        # The page carries a malformed JSON-LD script (skipped) and an
        # @graph datePublished that would score 10 but is unparseable.
        # Extraction must fall through to the valid top-level JSON-LD
        # datePublished (score 9) instead of the meta tag (score 7).
        article = load_article("pubdate_broken_jsonld", URL_WITHOUT_DATE)

        assert article.publish_date == datetime(2024, 3, 11)
        assert article.publish_date.tzinfo is None

        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-11T00:00:00"
        assert_json_export_consistent(article)

    @pytest.mark.parametrize(
        ("frozen", "expected"),
        [
            (datetime(2024, 3, 20, 12, 0), datetime(2024, 3, 20)),
            (datetime(2024, 3, 31, 23, 59), datetime(2024, 3, 31)),
            # crossing the date boundary legitimately changes the filled-in
            # day (the year and month still come from the candidate string)
            (datetime(2024, 4, 1, 0, 1), datetime(2024, 3, 1)),
        ],
    )
    def test_yearmonth_only_candidate_fills_day_from_now(self, freeze_now, frozen, expected):
        freeze_now(frozen)
        article = load_article("pubdate_yearmonth_only", URL_WITHOUT_DATE)

        assert article.publish_date == expected
        assert article.publish_date.tzinfo is None
        assert_json_export_consistent(article)

    def test_future_meta_candidate_is_penalized(self, freeze_now):
        freeze_now(datetime(2024, 3, 20, 12, 0))
        # The meta candidate 2024-03-29 lies 9 days in the future:
        # score 7 - 2 = 5, so the itemprop tag (score 6) wins.
        article = load_article("pubdate_future_candidate", URL_WITHOUT_DATE)

        assert article.publish_date == datetime(2024, 3, 18)
        assert_json_export_consistent(article)

    def test_future_penalty_lifts_after_date_boundary(self, freeze_now):
        freeze_now(datetime(2024, 4, 2, 12, 0))
        # After "now" moves past 2024-03-29 the penalty no longer applies:
        # the meta candidate scores 7 and beats the itemprop tag (6).
        article = load_article("pubdate_future_candidate", URL_WITHOUT_DATE)

        assert article.publish_date == datetime(2024, 3, 29)
        assert_json_export_consistent(article)


class TestEnvironmentControl:
    """At a fixed moment, process timezone changes are environment drift
    and must not alter results; moving the frozen moment across a date
    boundary is a legitimate change (covered above)."""

    @pytest.mark.parametrize("tz_name", ["UTC", "Pacific/Kiritimati", "America/New_York"])
    def test_fixed_moment_no_timezone_drift(self, freeze_now, process_timezone, tz_name):
        process_timezone(tz_name)
        freeze_now(FROZEN_NOW)

        article = load_article("pubdate_same_instant_offsets", URL_WITHOUT_DATE)

        assert article.publish_date == datetime(2024, 3, 16, 4, 30, tzinfo=timezone.utc)
        assert article.publish_date.utcoffset() == timedelta(hours=-5)
        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-15T23:30:00-05:00"

    @pytest.mark.parametrize("tz_name", ["UTC", "Pacific/Kiritimati", "America/New_York"])
    def test_yearmonth_fill_stable_across_process_timezones(self, freeze_now, process_timezone, tz_name):
        process_timezone(tz_name)
        freeze_now(FROZEN_NOW)

        article = load_article("pubdate_yearmonth_only", URL_WITHOUT_DATE)

        assert article.publish_date == datetime(2024, 3, 20)
        payload = json.loads(article.to_json())
        assert payload["publish_date"] == "2024-03-20T00:00:00"
