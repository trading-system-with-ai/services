"""Web-search registry + stub provider contract (Catalyst research upgrade).

The registry must clone the platform's provider contract exactly (empty name
-> ProviderNotConfigured, unknown -> ValueError, no default), and the stub
must be DETERMINISTIC (no wall-clock reads) while exercising the honesty
paths downstream code depends on: a result with no publication time, and
near-duplicate URLs for the dedup layer.
"""
from datetime import datetime, timedelta, timezone

import pytest

from libs.web_search import (
    CAPABILITY_KEYS,
    RESULT_TYPE_NEWS,
    RESULT_TYPE_WEB,
    WEB_SEARCH_NOT_CONFIGURED_MESSAGE,
    ProviderNotConfigured,
    get_provider,
)

WINDOW_END = datetime(2026, 8, 20, tzinfo=timezone.utc)
WINDOW_START = WINDOW_END - timedelta(days=45)


def test_empty_name_raises_provider_not_configured():
    with pytest.raises(ProviderNotConfigured) as exc:
        get_provider("")
    assert str(exc.value) == WEB_SEARCH_NOT_CONFIGURED_MESSAGE
    with pytest.raises(ProviderNotConfigured):
        get_provider("   ")


def test_unknown_name_raises_value_error_naming_known_providers():
    with pytest.raises(ValueError) as exc:
        get_provider("bing")
    assert "bing" in str(exc.value)
    assert "stub" in str(exc.value)


def test_stub_is_deterministic_across_calls():
    a = get_provider("stub").search_news(
        "NVDA data center revenue", start_time=WINDOW_START,
        end_time=WINDOW_END, limit=5,
    )
    b = get_provider("stub").search_news(
        "NVDA data center revenue", start_time=WINDOW_START,
        end_time=WINDOW_END, limit=5,
    )
    assert a == b
    assert len(a) == 5


def test_stub_results_carry_provenance_and_stay_inside_the_window():
    results = get_provider("stub").search_web(
        "CPI shelter inflation", start_time=WINDOW_START,
        end_time=WINDOW_END, limit=5,
    )
    assert [r.rank for r in results] == list(range(len(results)))
    for r in results:
        assert r.provider == "stub"
        assert r.query == "CPI shelter inflation"
        assert r.result_type == RESULT_TYPE_WEB
        assert r.retrieved_at == WINDOW_END
        if r.published_at is not None:
            assert WINDOW_START < r.published_at < WINDOW_END


def test_stub_emits_one_result_with_no_publication_time():
    """The provider-omitted-timestamp path (§44 rule 18): downstream code must
    handle a result that cannot be placed in time, so the stub always serves
    one."""
    results = get_provider("stub").search_news(
        "GDP inventories", start_time=WINDOW_START, end_time=WINDOW_END,
        limit=5,
    )
    assert sum(1 for r in results if r.published_at is None) == 1


def test_stub_emits_near_duplicate_urls_for_the_dedup_layer():
    results = get_provider("stub").search_web(
        "FOMC rate expectations", start_time=WINDOW_START,
        end_time=WINDOW_END, limit=5,
    )
    dup_targets = [r.url for r in results if "/dup" in r.url]
    assert len(dup_targets) == 2
    # Same canonical target, different decoration — what URL normalization
    # must fold into one document.
    assert len(set(dup_targets)) == 2
    assert len({u.split("?")[0] for u in dup_targets}) == 1


def test_stub_respects_limit_and_degenerate_windows():
    stub = get_provider("stub")
    assert stub.search_web("x", limit=0) == []
    assert len(stub.search_web("x", limit=3)) == 3
    assert (
        stub.search_web(
            "x", start_time=WINDOW_END, end_time=WINDOW_START, limit=5
        )
        == []
    )


def test_stub_start_time_only_spans_forward_from_start():
    """A "results since X" call with no end bound must answer — the open end
    closes forward from the caller's start, never against a fixed past
    anchor that would make every future start_time a degenerate window."""
    results = get_provider("stub").search_web(
        "post-anchor development", start_time=WINDOW_END, limit=5
    )
    assert len(results) == 5
    for r in results:
        if r.published_at is not None:
            assert r.published_at > WINDOW_END


def test_stub_news_and_web_are_distinct_result_sets():
    news = get_provider("stub").search_news(
        "q", start_time=WINDOW_START, end_time=WINDOW_END, limit=5
    )
    web = get_provider("stub").search_web(
        "q", start_time=WINDOW_START, end_time=WINDOW_END, limit=5
    )
    assert all(r.result_type == RESULT_TYPE_NEWS for r in news)
    assert all(r.result_type == RESULT_TYPE_WEB for r in web)
    assert {r.url for r in news} != {r.url for r in web}


def test_stub_capabilities_report_every_fixed_key():
    caps = get_provider("stub").capabilities()
    assert set(caps) == set(CAPABILITY_KEYS)
    assert all(v is True for v in caps.values())
