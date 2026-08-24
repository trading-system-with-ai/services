"""Windowed news across the provider layer (Phase D evidence engine, §21-§27).

``get_news`` answers "what is on the wire right now". That is the wrong
question for an event whose window closed last week, so the evidence engine
asks a different one — ``get_news_window(tickers, start, end)`` — and these
tests pin the properties the engine depends on, in order of importance:

1. TIME IS NOT GUESSED: a naive ``start``/``end`` is REFUSED by every
   provider, identically. Assuming a zone would shift the window by 4-5 hours,
   which is the difference between an article published BEFORE a release and
   one published after it — precisely the distinction the engine reads.
2. THE WINDOW IS COMPLETE: pagination is followed to exhaustion (bounded by
   ``limit``), so a caller never mistakes a first page for the whole window.
3. ONE ARTICLE IS ONE ARTICLE: ``source_id`` de-duplication survives a page
   boundary repeating a row AND, on Massive, the per-ticker fan-out that makes
   a syndicated article arrive once per tagged ticker.
4. NO FABRICATION (§44 rule 18): a row missing a citable field (id, title,
   url, timestamp) is SKIPPED, never patched — an uncitable article cannot
   ground anything. An empty window is ``[]``, and HTTP 403 is
   :class:`CapabilityNotAvailable` naming the endpoint.
5. NEWEST FIRST, so a ``limit`` that bites keeps the most recent news.
6. ``get_news`` IS UNCHANGED: the windowed feed reuses its parsing, and the
   two normalise the same row identically — a same-article-two-ways test.

Both real adapters run over ``httpx.MockTransport``; the network is never
touched, and the wire shapes are the ones the Phase D contract documents.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from libs.market_data.alpaca import AlpacaMarketDataProvider
from libs.market_data.massive import MassiveProvider
from libs.market_data.provider import (
    CapabilityNotAvailable,
    MarketDataError,
    NewsArticle,
)
from libs.market_data.stub import StubProvider

EASTERN = ZoneInfo("America/New_York")

START = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc)


def alpaca_with(handler, **kwargs) -> AlpacaMarketDataProvider:
    return AlpacaMarketDataProvider(
        api_key_id="test-key-id",
        api_secret_key="test-secret",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def massive_with(handler, **kwargs) -> MassiveProvider:
    return MassiveProvider(
        api_key="test-key", transport=httpx.MockTransport(handler), **kwargs
    )


def _anews(article_id: int, ts: datetime, symbols=("AAPL",), **over) -> dict:
    """One Alpaca ``news[]`` row in the contract's live-verified shape."""
    row = {
        "id": article_id,
        "headline": f"AAPL headline {article_id}",
        "author": "staff",
        "created_at": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": f"summary {article_id}",
        "url": f"https://example.test/a/{article_id}",
        "symbols": list(symbols),
        "source": "benzinga",
    }
    row.update(over)
    return row


def _mnews(article_id: str, ts: datetime, tickers=("AAPL",), **over) -> dict:
    """One Massive ``results[]`` news row in the contract's shape."""
    row = {
        "id": article_id,
        "title": f"AAPL title {article_id}",
        "author": "staff",
        "published_utc": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "article_url": f"https://example.test/m/{article_id}",
        "tickers": list(tickers),
        "description": f"description {article_id}",
        "publisher": {"name": "The Motley Fool", "homepage_url": "https://x.test"},
    }
    row.update(over)
    return row


def _capture(pages):
    """A handler serving `pages` in order, recording every request URL."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        payload = pages[min(len(seen) - 1, len(pages) - 1)]
        return httpx.Response(200, json=payload)

    return handler, seen


# ----------------------------------------------------------------------
# 1. Time is not guessed — every provider refuses identically
# ----------------------------------------------------------------------

def _every_provider():
    ok_alpaca = httpx.MockTransport(lambda r: httpx.Response(200, json={"news": []}))
    ok_massive = httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))
    return [
        AlpacaMarketDataProvider(
            api_key_id="k", api_secret_key="s", transport=ok_alpaca
        ),
        MassiveProvider(api_key="k", transport=ok_massive),
        StubProvider(),
    ]


@pytest.mark.parametrize("provider", _every_provider())
def test_naive_start_rejected_by_every_provider(provider):
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.get_news_window(
            tickers=["AAPL"], start=datetime(2026, 8, 1), end=END
        )


@pytest.mark.parametrize("provider", _every_provider())
def test_naive_end_rejected_by_every_provider(provider):
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.get_news_window(
            tickers=["AAPL"], start=START, end=datetime(2026, 8, 13)
        )


@pytest.mark.parametrize("provider", _every_provider())
def test_reversed_window_rejected_by_every_provider(provider):
    with pytest.raises(ValueError, match="precedes start"):
        provider.get_news_window(tickers=["AAPL"], start=END, end=START)


@pytest.mark.parametrize("provider", _every_provider())
def test_empty_ticker_list_is_empty_not_the_whole_firehose(provider):
    # A blank `symbols`/`ticker` parameter reads as "every symbol" on both
    # real endpoints, so an empty basket must never reach the wire.
    assert provider.get_news_window(tickers=[], start=START, end=END) == []


@pytest.mark.parametrize("provider", _every_provider())
def test_non_positive_limit_returns_nothing(provider):
    assert provider.get_news_window(tickers=["AAPL"], start=START, end=END, limit=0) == []


def test_eastern_window_is_converted_to_utc_on_the_wire():
    handler, seen = _capture([{"news": [], "next_page_token": None}])
    provider = alpaca_with(handler)
    provider.get_news_window(
        tickers=["AAPL"],
        start=datetime(2026, 8, 1, 9, 30, tzinfo=EASTERN),
        end=datetime(2026, 8, 1, 16, 0, tzinfo=EASTERN),
    )
    assert seen[0].params["start"] == "2026-08-01T13:30:00Z"
    assert seen[0].params["end"] == "2026-08-01T20:00:00Z"


# ----------------------------------------------------------------------
# 2. Alpaca — request shape, pagination, bounds
# ----------------------------------------------------------------------

def test_alpaca_request_pins_the_contract_parameters():
    handler, seen = _capture([{"news": [], "next_page_token": None}])
    alpaca_with(handler).get_news_window(tickers=["AAPL", "MSFT"], start=START, end=END)
    params = seen[0].params
    assert seen[0].path == "/v1beta1/news"
    assert params["symbols"] == "AAPL,MSFT"
    assert params["start"] == "2026-08-01T00:00:00Z"
    assert params["end"] == "2026-08-13T00:00:00Z"
    assert params["sort"] == "desc"
    assert params["include_content"] == "false"


def test_alpaca_asks_for_the_basket_in_one_request_not_one_per_ticker():
    handler, seen = _capture([{"news": [], "next_page_token": None}])
    alpaca_with(handler).get_news_window(
        tickers=["AAPL", "MSFT", "NVDA"], start=START, end=END
    )
    assert len(seen) == 1


def test_alpaca_uppercases_and_dedupes_the_symbol_csv():
    handler, seen = _capture([{"news": [], "next_page_token": None}])
    alpaca_with(handler).get_news_window(
        tickers=[" aapl ", "AAPL", "msft", ""], start=START, end=END
    )
    assert seen[0].params["symbols"] == "AAPL,MSFT"


def test_alpaca_maps_one_row_field_for_field():
    ts = datetime(2026, 8, 12, 14, 5, tzinfo=timezone.utc)
    handler, _ = _capture([{"news": [_anews(7, ts, symbols=("AAPL", "MSFT"))]}])
    [article] = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert article == NewsArticle(
        source_id="alpaca:7",
        title="AAPL headline 7",
        publisher="benzinga",
        published_at=ts,
        url="https://example.test/a/7",
        tickers=("AAPL", "MSFT"),
        description="summary 7",
    )


def test_alpaca_follows_next_page_token_to_exhaustion():
    t = datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc)
    pages = [
        {"news": [_anews(1, t), _anews(2, t - timedelta(hours=1))],
         "next_page_token": "p2"},
        {"news": [_anews(3, t - timedelta(hours=2))], "next_page_token": None},
    ]
    handler, seen = _capture(pages)
    articles = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["alpaca:1", "alpaca:2", "alpaca:3"]
    assert seen[1].params["page_token"] == "p2"


def test_alpaca_empty_page_token_string_ends_pagination():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    handler, seen = _capture([{"news": [_anews(1, t)], "next_page_token": ""}])
    assert len(alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )) == 1
    assert len(seen) == 1


def test_alpaca_deduplicates_an_article_repeated_across_pages():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    pages = [
        {"news": [_anews(1, t)], "next_page_token": "p2"},
        {"news": [_anews(1, t), _anews(2, t - timedelta(hours=1))],
         "next_page_token": None},
    ]
    handler, _ = _capture(pages)
    articles = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["alpaca:1", "alpaca:2"]


def test_alpaca_first_parse_wins_on_a_duplicate_source_id():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    pages = [
        {"news": [_anews(1, t, headline="first")], "next_page_token": "p2"},
        {"news": [_anews(1, t, headline="rewritten")], "next_page_token": None},
    ]
    handler, _ = _capture(pages)
    [article] = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert article.title == "first"


def test_alpaca_limit_stops_pagination_early():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    pages = [
        {"news": [_anews(i, t - timedelta(hours=i)) for i in range(3)],
         "next_page_token": "p2"},
        {"news": [_anews(9, t - timedelta(days=1))], "next_page_token": None},
    ]
    handler, seen = _capture(pages)
    articles = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END, limit=2
    )
    assert len(articles) == 2
    assert len(seen) == 1  # the second page was never requested


def test_alpaca_limit_keeps_the_newest_articles():
    t = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    rows = [_anews(i, t - timedelta(hours=i)) for i in range(5)]
    handler, _ = _capture([{"news": rows, "next_page_token": None}])
    articles = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END, limit=2
    )
    assert [a.source_id for a in articles] == ["alpaca:0", "alpaca:1"]


def test_alpaca_returns_newest_first_even_when_the_server_does_not():
    t = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    rows = [_anews(1, t - timedelta(days=2)), _anews(2, t), _anews(3, t - timedelta(days=1))]
    handler, _ = _capture([{"news": rows, "next_page_token": None}])
    articles = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["alpaca:2", "alpaca:3", "alpaca:1"]


def test_alpaca_empty_window_is_an_empty_list_not_an_error():
    handler, _ = _capture([{"news": [], "next_page_token": None}])
    assert alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    ) == []


def test_alpaca_null_news_key_is_an_honest_absence():
    handler, _ = _capture([{"news": None, "next_page_token": None}])
    assert alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    ) == []


def test_alpaca_zero_length_window_is_allowed():
    handler, _ = _capture([{"news": [], "next_page_token": None}])
    assert alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=START
    ) == []


@pytest.mark.parametrize("field", ["id", "headline", "url", "created_at"])
def test_alpaca_skips_a_row_missing_a_citable_field(field):
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    bad = _anews(1, t)
    bad[field] = None
    handler, _ = _capture([{"news": [bad, _anews(2, t)], "next_page_token": None}])
    articles = alpaca_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["alpaca:2"]


def test_alpaca_403_is_capability_not_available_naming_the_endpoint():
    handler = lambda r: httpx.Response(403, text="news not in your plan")
    with pytest.raises(CapabilityNotAvailable, match="/v1beta1/news"):
        alpaca_with(handler).get_news_window(tickers=["AAPL"], start=START, end=END)


def test_alpaca_500_is_a_plain_error_not_a_capability_gap():
    handler = lambda r: httpx.Response(500, text="boom")
    with pytest.raises(MarketDataError) as excinfo:
        alpaca_with(handler).get_news_window(tickers=["AAPL"], start=START, end=END)
    assert not isinstance(excinfo.value, CapabilityNotAvailable)


def test_alpaca_pagination_cap_truncates_visibly_rather_than_looping(caplog):
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    counter = {"n": 0}

    def handler(request):
        counter["n"] += 1
        return httpx.Response(200, json={
            "news": [_anews(counter["n"], t - timedelta(hours=counter["n"]))],
            "next_page_token": f"p{counter['n']}",
        })

    provider = alpaca_with(handler, max_news_pages=3)
    with caplog.at_level("WARNING"):
        articles = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    assert counter["n"] == 3
    assert len(articles) == 3
    assert "truncated at 3 pages" in caplog.text


def test_alpaca_get_news_still_ignores_the_window_parameters():
    """``get_news`` is UNTOUCHED: it is still the recency feed."""
    handler, seen = _capture([{"news": [], "next_page_token": None}])
    alpaca_with(handler).get_news(limit=10)
    assert "start" not in seen[0].params
    assert "symbols" not in seen[0].params
    assert seen[0].params["limit"] == "10"


def test_alpaca_same_row_normalises_identically_through_both_methods():
    t = datetime(2026, 8, 12, 14, 5, tzinfo=timezone.utc)
    row = _anews(7, t, symbols=("AAPL", "MSFT"))
    handler, _ = _capture([{"news": [row], "next_page_token": None}])
    provider = alpaca_with(handler)
    [windowed] = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    [recent] = provider.get_news(limit=1)
    assert windowed == recent


# ----------------------------------------------------------------------
# 3. Massive — per-ticker fan-out, next_url, merge
# ----------------------------------------------------------------------

def test_massive_request_pins_the_contract_parameters():
    handler, seen = _capture([{"results": [], "next_url": None}])
    massive_with(handler).get_news_window(tickers=["AAPL"], start=START, end=END)
    params = seen[0].params
    assert seen[0].path == "/v2/reference/news"
    assert params["ticker"] == "AAPL"
    assert params["published_utc.gte"] == START.isoformat()
    assert params["published_utc.lte"] == END.isoformat()
    assert params["order"] == "desc"
    assert params["sort"] == "published_utc"


def test_massive_issues_one_request_per_ticker():
    handler, seen = _capture([{"results": [], "next_url": None}])
    massive_with(handler).get_news_window(
        tickers=["AAPL", "MSFT", "NVDA"], start=START, end=END
    )
    assert [u.params["ticker"] for u in seen] == ["AAPL", "MSFT", "NVDA"]


def test_massive_merges_and_dedupes_a_syndicated_article_tagged_twice():
    """The article both tickers carry is ONE article after the merge."""
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    shared = _mnews("shared-1", t, tickers=("AAPL", "MSFT"))
    responses = [
        {"results": [shared, _mnews("aapl-only", t - timedelta(hours=1))],
         "next_url": None},
        {"results": [shared, _mnews("msft-only", t - timedelta(hours=2),
                                    tickers=("MSFT",))], "next_url": None},
    ]

    def handler(request):
        idx = 0 if request.url.params["ticker"] == "AAPL" else 1
        return httpx.Response(200, json=responses[idx])

    articles = massive_with(handler).get_news_window(
        tickers=["AAPL", "MSFT"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["shared-1", "aapl-only", "msft-only"]


def test_massive_maps_one_row_field_for_field():
    ts = datetime(2026, 8, 12, 14, 5, tzinfo=timezone.utc)
    handler, _ = _capture([{"results": [_mnews("x1", ts, tickers=("AAPL", "MSFT"))]}])
    [article] = massive_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert article == NewsArticle(
        source_id="x1",
        title="AAPL title x1",
        publisher="The Motley Fool",
        published_at=ts,
        url="https://example.test/m/x1",
        tickers=("AAPL", "MSFT"),
        description="description x1",
    )


def test_massive_follows_next_url_to_exhaustion():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    pages = [
        {"results": [_mnews("a", t)], "next_url": "https://api.test/v2/reference/news?cursor=2"},
        {"results": [_mnews("b", t - timedelta(hours=1))], "next_url": None},
    ]
    handler, seen = _capture(pages)
    articles = massive_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["a", "b"]
    assert seen[1].params["cursor"] == "2"


def test_massive_next_url_carries_the_cursor_without_reapplied_params():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    pages = [
        {"results": [_mnews("a", t)], "next_url": "https://api.test/v2/reference/news?cursor=2"},
        {"results": [], "next_url": None},
    ]
    handler, seen = _capture(pages)
    massive_with(handler).get_news_window(tickers=["AAPL"], start=START, end=END)
    assert "ticker" not in seen[1].params


def test_massive_deduplicates_across_next_url_pages():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    pages = [
        {"results": [_mnews("a", t)], "next_url": "https://api.test/v2/reference/news?cursor=2"},
        {"results": [_mnews("a", t), _mnews("b", t - timedelta(hours=1))],
         "next_url": None},
    ]
    handler, _ = _capture(pages)
    articles = massive_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["a", "b"]


def test_massive_limit_bounds_the_merged_result_and_stops_the_fan_out():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)

    def handler(request):
        ticker = request.url.params["ticker"]
        return httpx.Response(200, json={
            "results": [_mnews(f"{ticker}-{i}", t - timedelta(hours=i))
                        for i in range(3)],
            "next_url": None,
        })

    handler_seen: list[str] = []

    def counting(request):
        handler_seen.append(request.url.params["ticker"])
        return handler(request)

    articles = massive_with(counting).get_news_window(
        tickers=["AAPL", "MSFT"], start=START, end=END, limit=2
    )
    assert len(articles) == 2
    assert handler_seen == ["AAPL"]  # MSFT never fetched — the merge was full


def test_massive_returns_newest_first_across_tickers():
    t = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)

    def handler(request):
        ticker = request.url.params["ticker"]
        offset = 0 if ticker == "AAPL" else 1
        return httpx.Response(200, json={
            "results": [_mnews(f"{ticker}", t - timedelta(hours=offset))],
            "next_url": None,
        })

    articles = massive_with(handler).get_news_window(
        tickers=["MSFT", "AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["AAPL", "MSFT"]


def test_massive_empty_results_is_an_empty_window():
    handler, _ = _capture([{"results": [], "next_url": None}])
    assert massive_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    ) == []


@pytest.mark.parametrize("field", ["id", "title", "article_url", "published_utc"])
def test_massive_skips_a_row_missing_a_citable_field(field):
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    bad = _mnews("bad", t)
    bad[field] = None
    handler, _ = _capture([{"results": [bad, _mnews("good", t)], "next_url": None}])
    articles = massive_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["good"]


def test_massive_skips_a_row_with_an_unparseable_timestamp():
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    bad = _mnews("bad", t, published_utc="not-a-timestamp")
    handler, _ = _capture([{"results": [bad, _mnews("good", t)], "next_url": None}])
    articles = massive_with(handler).get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert [a.source_id for a in articles] == ["good"]


def test_massive_403_is_capability_not_available():
    handler = lambda r: httpx.Response(403, text="news not in your plan")
    with pytest.raises(CapabilityNotAvailable):
        massive_with(handler).get_news_window(tickers=["AAPL"], start=START, end=END)


def test_massive_pagination_cap_truncates_visibly(caplog):
    t = datetime(2026, 8, 12, tzinfo=timezone.utc)
    counter = {"n": 0}

    def handler(request):
        counter["n"] += 1
        return httpx.Response(200, json={
            "results": [_mnews(f"a{counter['n']}", t - timedelta(hours=counter["n"]))],
            "next_url": f"https://api.test/v2/reference/news?cursor={counter['n']}",
        })

    provider = massive_with(handler, max_news_pages=2)
    with caplog.at_level("WARNING"):
        articles = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    assert counter["n"] == 2
    assert len(articles) == 2
    assert "truncated at 2 pages" in caplog.text


def test_massive_get_news_still_ignores_the_window_parameters():
    """``get_news`` is UNTOUCHED: no ticker filter, no upper bound."""
    handler, seen = _capture([{"results": [], "next_url": None}])
    massive_with(handler).get_news(limit=10)
    assert "ticker" not in seen[0].params
    assert "published_utc.lte" not in seen[0].params


def test_massive_same_row_normalises_identically_through_both_methods():
    t = datetime(2026, 8, 12, 14, 5, tzinfo=timezone.utc)
    row = _mnews("x1", t)
    handler, _ = _capture([{"results": [row], "next_url": None}])
    provider = massive_with(handler)
    [windowed] = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    [recent] = provider.get_news(limit=1)
    assert windowed == recent


# ----------------------------------------------------------------------
# 4. Stub — deterministic, awkward on purpose
# ----------------------------------------------------------------------

def test_stub_is_deterministic_across_instances():
    first = StubProvider().get_news_window(tickers=["AAPL"], start=START, end=END)
    second = StubProvider().get_news_window(tickers=["AAPL"], start=START, end=END)
    assert first == second


def test_stub_same_day_reads_the_same_through_any_window():
    provider = StubProvider()
    wide = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    narrow_start = datetime(2026, 8, 5, tzinfo=timezone.utc)
    narrow = provider.get_news_window(
        tickers=["AAPL"], start=narrow_start, end=narrow_start + timedelta(days=1)
    )
    wide_by_id = {a.source_id: a for a in wide}
    assert narrow  # the narrow window is not vacuously equal
    for article in narrow:
        assert wide_by_id[article.source_id] == article


def test_stub_split_windows_reassemble_into_the_whole():
    provider = StubProvider()
    mid = datetime(2026, 8, 7, tzinfo=timezone.utc)
    whole = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    left = provider.get_news_window(tickers=["AAPL"], start=START, end=mid)
    right = provider.get_news_window(
        tickers=["AAPL"], start=mid + timedelta(microseconds=1), end=END
    )
    assert {a.source_id for a in left} | {a.source_id for a in right} == {
        a.source_id for a in whole
    }


def test_stub_articles_fall_inside_the_requested_window():
    articles = StubProvider().get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert articles
    assert all(START <= a.published_at <= END for a in articles)


def test_stub_respects_a_window_edge_mid_day():
    """A window ending before the day's 13:30Z drop excludes that day."""
    provider = StubProvider()
    day = datetime(2026, 8, 5, tzinfo=timezone.utc)
    before = provider.get_news_window(
        tickers=["AAPL"], start=day, end=day.replace(hour=12)
    )
    assert before == []


def test_stub_is_newest_first():
    articles = StubProvider().get_news_window(tickers=["AAPL"], start=START, end=END)
    assert all(
        articles[i].published_at >= articles[i + 1].published_at
        for i in range(len(articles) - 1)
    )


def test_stub_limit_keeps_the_newest_articles():
    provider = StubProvider()
    full = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    capped = provider.get_news_window(tickers=["AAPL"], start=START, end=END, limit=3)
    assert capped == full[:3]


def test_stub_emits_syndicated_near_duplicates_for_the_dedup_stage():
    articles = StubProvider().get_news_window(tickers=["AAPL"], start=START, end=END)
    syndicated = [a for a in articles if a.source_id.endswith("-syndicated")]
    assert syndicated
    for copy in syndicated:
        original_id = copy.source_id.replace("-syndicated", "-0")
        [original] = [a for a in articles if a.source_id == original_id]
        assert copy.title == original.title  # same story...
        assert copy.publisher != original.publisher  # ...different newsroom
        assert copy.published_at > original.published_at


def test_stub_emits_off_topic_articles_for_the_relevance_stage():
    articles = StubProvider().get_news_window(tickers=["AAPL"], start=START, end=END)
    off_topic = [a for a in articles if a.source_id.endswith("-offtopic")]
    assert off_topic
    for article in off_topic:
        assert "AAPL" not in article.title  # tagged, but not about the company
        assert article.tickers == ("AAPL",)


def test_stub_differs_by_ticker():
    provider = StubProvider()
    aapl = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    msft = provider.get_news_window(tickers=["MSFT"], start=START, end=END)
    assert {a.source_id for a in aapl}.isdisjoint({a.source_id for a in msft})


def test_stub_merges_a_multi_ticker_basket_without_dropping_either():
    provider = StubProvider()
    aapl = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
    msft = provider.get_news_window(tickers=["MSFT"], start=START, end=END)
    both = provider.get_news_window(tickers=["AAPL", "MSFT"], start=START, end=END, limit=1000)
    assert {a.source_id for a in both} == (
        {a.source_id for a in aapl} | {a.source_id for a in msft}
    )


def test_stub_source_ids_are_unique_within_a_window():
    articles = StubProvider().get_news_window(
        tickers=["AAPL", "MSFT"], start=START, end=END, limit=1000
    )
    ids = [a.source_id for a in articles]
    assert len(ids) == len(set(ids))


def test_stub_urls_use_the_stub_scheme_so_nothing_reads_as_a_real_citation():
    articles = StubProvider().get_news_window(
        tickers=["AAPL"], start=START, end=END
    )
    assert articles
    assert all(a.url.startswith("stub://") for a in articles)
    assert all("SYNTHETIC" in a.description for a in articles)


def test_stub_covers_a_multi_day_window_with_at_least_one_article_per_day():
    articles = StubProvider().get_news_window(
        tickers=["AAPL"], start=START, end=END, limit=1000
    )
    days = {a.published_at.date() for a in articles}
    assert len(days) >= 12


def test_stub_get_news_is_untouched_by_the_windowed_feed():
    """The recency feed still serves its own universe, ids and all."""
    recent = StubProvider().get_news(limit=3)
    assert len(recent) == 3
    assert all(not a.source_id.endswith(("-0", "-syndicated", "-offtopic"))
               for a in recent)


# ----------------------------------------------------------------------
# 5. Cross-provider shape agreement
# ----------------------------------------------------------------------

def test_every_provider_returns_news_articles_newest_first():
    t = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
    alpaca_handler, _ = _capture([{
        "news": [_anews(1, t - timedelta(hours=2)), _anews(2, t)],
        "next_page_token": None,
    }])
    massive_handler, _ = _capture([{
        "results": [_mnews("a", t - timedelta(hours=2)), _mnews("b", t)],
        "next_url": None,
    }])
    for provider in (alpaca_with(alpaca_handler), massive_with(massive_handler),
                     StubProvider()):
        articles = provider.get_news_window(tickers=["AAPL"], start=START, end=END)
        assert articles
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert all(a.published_at.tzinfo is not None for a in articles)
        assert all(
            articles[i].published_at >= articles[i + 1].published_at
            for i in range(len(articles) - 1)
        )
