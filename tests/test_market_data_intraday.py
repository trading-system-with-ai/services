"""Intraday bars across the provider layer (Phase C event replay, spec §20).

Event replay places a price relative to a RELEASE INSTANT, so the intraday
adapter's job is different from the daily one's in ways these tests pin, in
order of importance:

1. TIME IS NOT GUESSED: a naive ``start``/``end`` is REFUSED by every
   provider, identically. Assuming UTC or Eastern for a naive input would
   shift a window by 4-5 hours depending on the season — the difference
   between an after-market release and the next morning's open — and produce
   plausible-looking but wrong reactions.
2. THE WINDOW IS COMPLETE: pagination is followed to exhaustion, so a caller
   never silently receives a first page and reads it as the whole event
   window. Bars repeated across a page boundary are de-duplicated rather than
   counted twice.
3. NO FABRICATION (§44 rule 18): a row missing OHLC or VOLUME is skipped, not
   zero-filled — intraday volume feeds event-replay comparisons, where an
   invented 0 reads as "nobody traded that minute". An empty window is ``[]``,
   and HTTP 403 is :class:`CapabilityNotAvailable` naming the endpoint.
4. EXTENDED HOURS SURVIVE: after-hours and pre-market bars are returned, since
   an AMC earnings release moves the stock at 16:05 ET and filtering those
   minutes away would delete the reaction being replayed.
5. The stub is deterministic PER (symbol, minute) — the same minute reads the
   same through any window — so consumer tests can be written against it.

Both real adapters run over ``httpx.MockTransport``; the network is never
touched, and the live wire shapes are the ones probed 2026-08-19.
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from libs.market_data.alpaca import AlpacaMarketDataProvider
from libs.market_data.massive import MassiveProvider
from libs.market_data.provider import (
    CapabilityNotAvailable,
    IntradayBar,
    MarketDataError,
    require_aware_utc,
)
from libs.market_data.stub import StubProvider

EASTERN = ZoneInfo("America/New_York")

# The probed live window: an AMC session whose after-hours minutes are sparse.
START = datetime(2024, 5, 2, 13, 30, tzinfo=timezone.utc)
END = datetime(2024, 5, 3, 0, 0, tzinfo=timezone.utc)


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


def _abar(ts: datetime, close: float = 100.0, volume: int = 1_000) -> dict:
    """One Alpaca ``bars[]`` row in the live-verified shape."""
    return {
        "t": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": close - 0.5,
        "h": close + 0.4,
        "l": close - 0.9,
        "c": close,
        "v": volume,
        "n": 42,
        "vw": close,
    }


def _magg(ts: datetime, close: float = 100.0, volume: int = 1_000) -> dict:
    """One Massive aggregates ``results[]`` row (t is unix MILLISECONDS)."""
    return {
        "t": int(ts.timestamp() * 1000),
        "o": close - 0.5,
        "h": close + 0.4,
        "l": close - 0.9,
        "c": close,
        "v": volume,
        "n": 42,
        "vw": close,
    }


# ---------------------------------------------------------------------------
# Aware-UTC discipline: shared by every provider, refused identically
# ---------------------------------------------------------------------------


def _providers_for_naive_check():
    ok = httpx.MockTransport(lambda r: httpx.Response(200, json={"bars": []}))
    return [
        AlpacaMarketDataProvider(
            api_key_id="k", api_secret_key="s", transport=ok
        ),
        MassiveProvider(api_key="k", transport=ok),
        StubProvider(),
    ]


@pytest.mark.parametrize("provider", _providers_for_naive_check())
def test_naive_start_rejected_by_every_provider(provider):
    """A naive datetime has no correct zone to assume — it raises, never guesses."""
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.get_intraday_bars("AAPL", datetime(2024, 5, 2, 13, 30), END)


@pytest.mark.parametrize("provider", _providers_for_naive_check())
def test_naive_end_rejected_by_every_provider(provider):
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.get_intraday_bars("AAPL", START, datetime(2024, 5, 3))


@pytest.mark.parametrize("provider", _providers_for_naive_check())
def test_reversed_window_rejected_by_every_provider(provider):
    """An empty window is start == end; a reversed one is a caller bug."""
    with pytest.raises(ValueError, match="precedes start"):
        provider.get_intraday_bars("AAPL", END, START)


def test_require_aware_utc_converts_eastern_to_utc():
    """An aware NON-UTC input is fine — only naive is refused."""
    eastern = datetime(2024, 5, 2, 9, 30, tzinfo=EASTERN)
    converted = require_aware_utc(eastern, "start")
    assert converted.tzinfo is timezone.utc
    assert converted == datetime(2024, 5, 2, 13, 30, tzinfo=timezone.utc)


def test_require_aware_utc_rejects_a_non_datetime():
    with pytest.raises(ValueError, match="must be a datetime"):
        require_aware_utc(date(2024, 5, 2), "start")  # type: ignore[arg-type]


def test_alpaca_accepts_an_eastern_window_and_sends_utc():
    """The caller may think in ET; the wire always carries UTC."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["start"] = request.url.params.get("start")
        seen["end"] = request.url.params.get("end")
        return httpx.Response(200, json={"bars": []})

    alpaca_with(handler).get_intraday_bars(
        "AAPL",
        datetime(2024, 5, 2, 9, 30, tzinfo=EASTERN),
        datetime(2024, 5, 2, 16, 0, tzinfo=EASTERN),
    )
    assert seen == {"start": "2024-05-02T13:30:00Z", "end": "2024-05-02T20:00:00Z"}


# ---------------------------------------------------------------------------
# Alpaca: wire mapping, pagination, dedup, honest absence
# ---------------------------------------------------------------------------


def test_alpaca_request_pins_the_probed_parameters():
    """feed=iex and adjustment=split are STATED, never left to the tier default.

    The default stocks feed differs by subscription, and an unstated
    adjustment would put minute closes on a different price basis from the
    daily closes a gap is computed against.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        seen["path"] = request.url.path
        return httpx.Response(200, json={"bars": []})

    alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert seen["path"] == "/v2/stocks/AAPL/bars"
    assert seen["timeframe"] == "1Min"
    assert seen["feed"] == "iex"
    assert seen["adjustment"] == "split"
    assert seen["sort"] == "asc"
    assert int(seen["limit"]) == 10_000


def test_alpaca_maps_one_bar_field_for_field():
    ts = datetime(2024, 5, 2, 13, 30, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bars": [
                    {"t": "2024-05-02T13:30:00Z", "o": 170.1, "h": 170.9,
                     "l": 169.8, "c": 170.5, "v": 12345, "n": 90, "vw": 170.3}
                ],
                "next_page_token": None,
            },
        )

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert bars == [
        IntradayBar(ts=ts, open=170.1, high=170.9, low=169.8, close=170.5,
                    volume=12345)
    ]
    assert isinstance(bars[0].volume, int)


def test_alpaca_follows_next_page_token_to_exhaustion():
    """A truncated window would be read as "the stock stopped trading"."""
    pages = {
        None: {
            "bars": [_abar(START + timedelta(minutes=i)) for i in range(3)],
            "next_page_token": "p2",
        },
        "p2": {
            "bars": [_abar(START + timedelta(minutes=i)) for i in range(3, 6)],
            "next_page_token": "p3",
        },
        "p3": {
            "bars": [_abar(START + timedelta(minutes=i)) for i in range(6, 9)],
            "next_page_token": None,
        },
    }
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("page_token")
        calls.append(token)
        return httpx.Response(200, json=pages[token])

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert calls == [None, "p2", "p3"]
    assert len(bars) == 9
    assert [b.ts for b in bars] == [START + timedelta(minutes=i) for i in range(9)]


def test_alpaca_empty_page_token_string_ends_pagination():
    """``""`` is "no more pages", not a cursor to fetch."""
    calls: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("page_token"))
        return httpx.Response(200, json={"bars": [_abar(START)], "next_page_token": ""})

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert len(calls) == 1
    assert len(bars) == 1


def test_alpaca_deduplicates_a_bar_repeated_across_pages():
    """A ts on both sides of a page boundary is ONE minute, not two."""
    overlap = _abar(START, close=101.0)
    pages = {
        None: {"bars": [_abar(START, close=101.0), _abar(START + timedelta(minutes=1))],
               "next_page_token": "p2"},
        "p2": {"bars": [overlap, _abar(START + timedelta(minutes=2))],
               "next_page_token": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[request.url.params.get("page_token")])

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert len(bars) == 3
    assert len({b.ts for b in bars}) == 3


def test_alpaca_first_parse_wins_on_a_duplicate_timestamp():
    pages = {
        None: {"bars": [_abar(START, close=101.0)], "next_page_token": "p2"},
        "p2": {"bars": [_abar(START, close=999.0)], "next_page_token": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[request.url.params.get("page_token")])

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert [b.close for b in bars] == [101.0]


def test_alpaca_returns_bars_ascending_even_when_the_server_does_not():
    out_of_order = [
        _abar(START + timedelta(minutes=5)),
        _abar(START),
        _abar(START + timedelta(minutes=2)),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": out_of_order})

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)


def test_alpaca_null_bars_is_an_empty_window_not_an_error():
    """A range before the symbol's history: honest ``[]``, never extrapolated."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": None, "next_page_token": None})

    assert alpaca_with(handler).get_intraday_bars("AAPL", START, END) == []


def test_alpaca_zero_length_window_is_allowed_and_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": []})

    assert alpaca_with(handler).get_intraday_bars("AAPL", START, START) == []


def test_alpaca_keeps_extended_hours_bars():
    """An AMC release moves the stock at 16:05 ET — those minutes ARE the data."""
    after_hours = datetime(2024, 5, 2, 20, 31, tzinfo=timezone.utc)  # 16:31 ET

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"bars": [_abar(START), _abar(after_hours, close=175.0)]}
        )

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert [b.ts for b in bars] == [START, after_hours]
    assert bars[-1].ts.astimezone(EASTERN).hour == 16


def test_alpaca_skips_a_row_missing_volume_instead_of_zero_filling():
    """A fabricated 0 would read as "nobody traded that minute"."""
    row = _abar(START)
    row.pop("v")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"bars": [row, _abar(START + timedelta(minutes=1))]}
        )

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert len(bars) == 1
    assert bars[0].ts == START + timedelta(minutes=1)


@pytest.mark.parametrize("field", ["t", "o", "h", "l", "c"])
def test_alpaca_skips_a_row_missing_any_ohlc_or_timestamp(field):
    row = _abar(START)
    row.pop(field)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": [row]})

    assert alpaca_with(handler).get_intraday_bars("AAPL", START, END) == []


def test_alpaca_skips_a_non_positive_close():
    row = _abar(START)
    row["c"] = 0.0

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": [row]})

    assert alpaca_with(handler).get_intraday_bars("AAPL", START, END) == []


def test_alpaca_skips_a_negative_volume():
    row = _abar(START, volume=-5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": [row]})

    assert alpaca_with(handler).get_intraday_bars("AAPL", START, END) == []


def test_alpaca_403_is_capability_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="subscription does not permit 1Min bars")

    with pytest.raises(CapabilityNotAvailable, match="subscription"):
        alpaca_with(handler).get_intraday_bars("AAPL", START, END)


def test_alpaca_500_is_a_plain_market_data_error_not_a_capability_gap():
    """A broken provider and an unpurchased capability are different facts."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(MarketDataError) as exc:
        alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert not isinstance(exc.value, CapabilityNotAvailable)


def test_alpaca_pagination_cap_truncates_visibly_rather_than_looping(caplog):
    """A server that never stops sending a cursor must not spin forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"bars": [_abar(START)], "next_page_token": "always-more"}
        )

    with caplog.at_level("WARNING"):
        bars = alpaca_with(handler, max_intraday_pages=3).get_intraday_bars(
            "AAPL", START, END
        )
    assert len(bars) == 1  # deduplicated across the 3 identical pages
    assert any("truncated" in r.getMessage() for r in caplog.records)


def test_alpaca_nanosecond_precision_timestamp_parses():
    """Alpaca stamps sub-microsecond fractions on some feeds."""

    def handler(request: httpx.Request) -> httpx.Response:
        row = _abar(START)
        row["t"] = "2024-05-02T13:30:00.834543155Z"
        return httpx.Response(200, json={"bars": [row]})

    bars = alpaca_with(handler).get_intraday_bars("AAPL", START, END)
    assert bars[0].ts.tzinfo is timezone.utc
    assert bars[0].ts.replace(microsecond=0) == START


# ---------------------------------------------------------------------------
# Massive: aggregates minute range with next_url pagination
# ---------------------------------------------------------------------------


def test_massive_request_uses_the_minute_aggs_range_with_ms_bounds():
    """Milliseconds, not dates: an event window is not a whole session."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"results": []})

    massive_with(handler).get_intraday_bars("AAPL", START, END)
    from_ms, to_ms = int(START.timestamp() * 1000), int(END.timestamp() * 1000)
    assert seen["path"] == f"/v2/aggs/ticker/AAPL/range/1/minute/{from_ms}/{to_ms}"
    assert seen["adjusted"] == "true"
    assert seen["sort"] == "asc"
    assert int(seen["limit"]) == 50_000


def test_massive_maps_a_millisecond_timestamp_not_a_nanosecond_one():
    """Reading ms as ns would place a 2024 bar in 1970 — wrong, not missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_magg(START, close=170.5)]})

    bars = massive_with(handler).get_intraday_bars("AAPL", START, END)
    assert bars[0].ts == START
    assert bars[0].close == 170.5
    assert isinstance(bars[0].volume, int)


def test_massive_follows_next_url_to_exhaustion():
    minute = timedelta(minutes=1)
    pages = {
        "/v2/aggs/ticker/AAPL/range/1/minute/"
        f"{int(START.timestamp() * 1000)}/{int(END.timestamp() * 1000)}": {
            "results": [_magg(START + i * minute) for i in range(2)],
            "next_url": "https://api.massive.com/v2/aggs/page2",
        },
        "/v2/aggs/page2": {
            "results": [_magg(START + i * minute) for i in range(2, 4)],
            "next_url": "https://api.massive.com/v2/aggs/page3",
        },
        "/v2/aggs/page3": {
            "results": [_magg(START + i * minute) for i in range(4, 6)],
        },
    }
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=pages[request.url.path])

    bars = massive_with(handler).get_intraday_bars("AAPL", START, END)
    assert paths == list(pages)
    assert len(bars) == 6


def test_massive_deduplicates_across_next_url_pages():
    pages = [
        {"results": [_magg(START), _magg(START + timedelta(minutes=1))],
         "next_url": "https://api.massive.com/v2/aggs/page2"},
        {"results": [_magg(START + timedelta(minutes=1)), _magg(START + timedelta(minutes=2))]},
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = pages[calls["n"]]
        calls["n"] += 1
        return httpx.Response(200, json=payload)

    bars = massive_with(handler).get_intraday_bars("AAPL", START, END)
    assert len(bars) == 3
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)


def test_massive_drops_a_bar_outside_the_requested_window():
    """A paginating server may overshoot; a bar outside the window is not ours."""
    outside = END + timedelta(minutes=30)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_magg(START), _magg(outside)]})

    bars = massive_with(handler).get_intraday_bars("AAPL", START, END)
    assert [b.ts for b in bars] == [START]


def test_massive_empty_results_is_an_empty_window():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "status": "OK"})

    assert massive_with(handler).get_intraday_bars("AAPL", START, END) == []


def test_massive_skips_a_row_missing_volume_instead_of_zero_filling():
    row = _magg(START)
    row.pop("v")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [row, _magg(START + timedelta(minutes=1))]}
        )

    bars = massive_with(handler).get_intraday_bars("AAPL", START, END)
    assert len(bars) == 1


def test_massive_403_is_capability_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="plan does not include minute aggregates")

    with pytest.raises(CapabilityNotAvailable, match="plan"):
        massive_with(handler).get_intraday_bars("AAPL", START, END)


def test_massive_unmapped_timeframe_refuses_rather_than_serving_minutes():
    """Bars at a resolution the caller did not ask for are WRONG numbers."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made for an unmapped timeframe")

    with pytest.raises(CapabilityNotAvailable, match="2Min"):
        massive_with(handler).get_intraday_bars(
            "AAPL", START, END, timeframe="2Min"
        )


def test_massive_hour_timeframe_maps_to_the_hour_timespan():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"results": []})

    massive_with(handler).get_intraday_bars("AAPL", START, END, timeframe="1Hour")
    assert "/range/1/hour/" in seen["path"]


def test_massive_pagination_cap_truncates_visibly(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [_magg(START)],
                "next_url": "https://api.massive.com/v2/aggs/next",
            },
        )

    with caplog.at_level("WARNING"):
        bars = massive_with(handler, max_intraday_pages=2).get_intraday_bars(
            "AAPL", START, END
        )
    assert len(bars) == 1
    assert any("truncated" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Stub: deterministic synthetic minutes (NOT market data)
# ---------------------------------------------------------------------------


def test_stub_is_deterministic_across_instances():
    first = StubProvider().get_intraday_bars("AAPL", START, END)
    second = StubProvider().get_intraday_bars("AAPL", START, END)
    assert first == second
    assert first  # the window does contain a session


def test_stub_same_minute_reads_the_same_through_any_window():
    """The seed is the bar's own minute, never its index in this request.

    An index-based walk would move every value whenever the window changed,
    so a consumer could not fetch overlapping windows and compare them.
    """
    provider = StubProvider()
    full = {b.ts: b for b in provider.get_intraday_bars("AAPL", START, END)}
    narrow = provider.get_intraday_bars(
        "AAPL", START + timedelta(hours=1), START + timedelta(hours=2)
    )
    assert narrow
    assert all(full[b.ts] == b for b in narrow)


def test_stub_covers_the_regular_session_at_one_minute_steps():
    provider = StubProvider()
    open_et = datetime(2024, 5, 2, 9, 30, tzinfo=EASTERN)
    close_et = datetime(2024, 5, 2, 16, 0, tzinfo=EASTERN)
    bars = provider.get_intraday_bars("AAPL", open_et, close_et - timedelta(minutes=1))
    assert len(bars) == 390  # 6.5h of minutes
    assert bars[0].ts == open_et.astimezone(timezone.utc)
    gaps = {(b.ts - a.ts).total_seconds() for a, b in zip(bars, bars[1:])}
    assert gaps == {60.0}


def test_stub_after_hours_bars_are_sparse_not_dense():
    """Real extended-hours feeds are sparse; a dense grid would leave a
    consumer's "no after-hours bars" branch permanently untested."""
    provider = StubProvider()
    close_et = datetime(2024, 5, 2, 16, 0, tzinfo=EASTERN)
    end_et = datetime(2024, 5, 2, 20, 0, tzinfo=EASTERN)
    bars = provider.get_intraday_bars("AAPL", close_et, end_et)
    assert len(bars) == 48  # 4h at 5-minute steps
    gaps = {(b.ts - a.ts).total_seconds() for a, b in zip(bars, bars[1:])}
    assert gaps == {300.0}


def test_stub_has_no_bars_on_a_weekend():
    saturday = datetime(2024, 5, 4, 12, 0, tzinfo=EASTERN)
    sunday = datetime(2024, 5, 5, 23, 0, tzinfo=EASTERN)
    assert StubProvider().get_intraday_bars("AAPL", saturday, sunday) == []


def test_stub_session_start_follows_dst_not_a_fixed_utc_offset():
    """09:30 ET is 13:30Z in summer and 14:30Z in winter."""
    provider = StubProvider()
    summer = provider.get_intraday_bars(
        "AAPL",
        datetime(2024, 7, 10, 9, 30, tzinfo=EASTERN),
        datetime(2024, 7, 10, 10, 0, tzinfo=EASTERN),
    )
    winter = provider.get_intraday_bars(
        "AAPL",
        datetime(2024, 1, 10, 9, 30, tzinfo=EASTERN),
        datetime(2024, 1, 10, 10, 0, tzinfo=EASTERN),
    )
    assert summer[0].ts.hour == 13
    assert winter[0].ts.hour == 14
    assert summer[0].ts.astimezone(EASTERN).time() == winter[0].ts.astimezone(EASTERN).time()


def test_stub_bars_are_internally_consistent_and_finite():
    for bar in StubProvider().get_intraday_bars("AAPL", START, END):
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.volume >= 0
        assert bar.ts.tzinfo is timezone.utc


def test_stub_intraday_anchors_on_the_same_days_daily_open():
    """Intraday and daily must not tell two stories about the same session."""
    provider = StubProvider()
    day = date(2024, 5, 2)
    daily = provider.get_daily_bars("AAPL", 1, end=day)
    bars = provider.get_intraday_bars(
        "AAPL",
        datetime(2024, 5, 2, 9, 30, tzinfo=EASTERN),
        datetime(2024, 5, 2, 16, 0, tzinfo=EASTERN),
    )
    assert bars[0].open == daily[0].open


def test_stub_differs_by_symbol():
    a = StubProvider().get_intraday_bars("AAPL", START, END)
    b = StubProvider().get_intraday_bars("MSFT", START, END)
    assert [x.ts for x in a] == [x.ts for x in b]
    assert [x.close for x in a] != [x.close for x in b]


def test_stub_non_minute_timeframe_returns_empty_rather_than_mislabelled_minutes():
    assert StubProvider().get_intraday_bars(
        "AAPL", START, END, timeframe="5Min"
    ) == []


def test_stub_window_before_the_walk_epoch_is_empty():
    """Pre-epoch dates do not exist — honest absence, not extrapolation."""
    old_start = datetime(2019, 5, 2, 13, 30, tzinfo=timezone.utc)
    old_end = datetime(2019, 5, 2, 20, 0, tzinfo=timezone.utc)
    assert StubProvider().get_intraday_bars("AAPL", old_start, old_end) == []


# ---------------------------------------------------------------------------
# Cross-provider shape agreement
# ---------------------------------------------------------------------------


def test_every_provider_returns_the_same_row_type():
    """A consumer must never have to ask which provider it is holding."""

    def alpaca_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": [_abar(START)]})

    def massive_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_magg(START)]})

    rows = [
        alpaca_with(alpaca_handler).get_intraday_bars("AAPL", START, END)[0],
        massive_with(massive_handler).get_intraday_bars("AAPL", START, END)[0],
        StubProvider().get_intraday_bars("AAPL", START, END)[0],
    ]
    assert all(isinstance(row, IntradayBar) for row in rows)
    assert all(row.ts.tzinfo is timezone.utc for row in rows)
    assert all(isinstance(row.volume, int) for row in rows)


def test_every_provider_exposes_get_intraday_bars():
    for cls in (AlpacaMarketDataProvider, MassiveProvider, StubProvider):
        assert callable(getattr(cls, "get_intraday_bars", None))
