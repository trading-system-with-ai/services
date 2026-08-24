"""MassiveProvider — the ONLY real market data source (guide §15/§16, §33 rule 12).

Every test runs the real adapter over ``httpx.MockTransport``; the network is
never touched. What these tests pin, in order of importance:

1. NO FABRICATION: unknown symbols are absent (never invented), unquotable or
   greek-less contracts are skipped (never zero-filled), historical chains the
   snapshot cannot serve raise (never approximated).
2. CAPABILITY HONESTY (§16): HTTP 403 is "the plan does not include this"
   (CapabilityNotAvailable), distinct from transport faults (MarketDataError),
   and an all-403 quote batch re-raises instead of reading as an empty market.
3. The exact wire mapping against the VERIFIED Massive API shapes
   (aggregates o/h/l/c/v/t unix-ms; stocks snapshot ticker.lastTrade/day/
   prevDay/todaysChangePerc; /v3/snapshot/options results with details/greeks/
   implied_volatility/open_interest/last_quote and next_url pagination).
"""
import json
from datetime import date, datetime, timedelta

import httpx
import pytest

from libs.market_data.massive import (
    CHAIN_PAGE_LIMIT,
    DEFAULT_MAX_RATE_LIMIT_RETRIES,
    EASTERN,
    RATE_LIMIT_MAX_DELAY_SECONDS,
    MassiveProvider,
    _parse_occ_ticker,
)
from libs.market_data.provider import CapabilityNotAvailable, MarketDataError

TODAY = datetime.now(EASTERN).date()


def provider_with(handler, **kwargs) -> MassiveProvider:
    return MassiveProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _json_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


# ---------------------------------------------------------------------------
# Construction & auth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_key", ["", "   "])
def test_empty_api_key_refused_at_construction(bad_key):
    with pytest.raises(MarketDataError, match="MASSIVE_API_KEY"):
        MassiveProvider(api_key=bad_key)


def test_header_auth_first_then_query_param_fallback_on_401():
    """Bearer-header auth is tried first; a 401 flips to ?apiKey= once."""
    seen: list[tuple[str | None, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        key_param = request.url.params.get("apiKey")
        seen.append((auth, key_param))
        if auth:  # reject header auth to force the fallback
            return _json_response({"message": "unauthorized"}, status=401)
        assert key_param == "test-key"
        return _json_response({"results": [], "status": "OK"})

    bars = provider_with(handler).get_daily_bars("SPY", 5)
    assert bars == []
    assert seen[0][0] == "Bearer test-key" and seen[0][1] is None
    assert seen[1][0] is None and seen[1][1] == "test-key"


def test_401_on_both_auth_forms_names_the_key_setting():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"message": "unauthorized"}, status=401)

    with pytest.raises(MarketDataError, match="MASSIVE_API_KEY"):
        provider_with(handler).get_daily_bars("SPY", 5)


def test_key_never_appears_in_logs(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"results": [], "status": "OK"})

    import logging

    with caplog.at_level(logging.DEBUG, logger="libs.market_data.massive"):
        provider_with(handler).get_daily_bars("SPY", 5)
    assert "test-key" not in caplog.text


# ---------------------------------------------------------------------------
# Failure taxonomy (§16 / §28: fail closed, never fabricate)
# ---------------------------------------------------------------------------


def test_403_is_capability_not_available_naming_the_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"message": "plan does not include this"}, status=403)

    with pytest.raises(CapabilityNotAvailable, match="plan"):
        provider_with(handler).get_daily_bars("SPY", 5)


def test_429_retries_the_bounded_ladder_honoring_retry_after_then_raises(monkeypatch):
    """UPDATED from the original "exactly one retry" policy.

    A single retry was not enough in production: a history backfill fires four
    option-bar fetches per event for eight events back-to-back, and Massive's
    rate-limit window stayed shut across several of them, so events silently
    ended with no bars. The adapter now retries ``max_rate_limit_retries``
    times, still BOUNDED (a request handler that sleeps forever is a hang),
    and still honours Retry-After when the server sends one.
    """
    sleeps: list[float] = []
    monkeypatch.setattr("libs.market_data.massive.time.sleep", sleeps.append)
    calls = {"n": 0}

    def always_limited(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429, headers={"Retry-After": "1"}, json={"message": "slow down"}
        )

    with pytest.raises(MarketDataError, match="429"):
        provider_with(always_limited).get_daily_bars("SPY", 5)
    # original + the full retry budget, then the honest error.
    assert calls["n"] == 1 + DEFAULT_MAX_RATE_LIMIT_RETRIES
    # Retry-After honored on EVERY attempt, not just the first.
    assert sleeps == [1.0] * DEFAULT_MAX_RATE_LIMIT_RETRIES


def test_429_without_retry_after_walks_the_documented_backoff_ladder(monkeypatch):
    """No Retry-After -> 2s, 5s, 12s, 25s. Growing, bounded, and stated here
    so a change to the ladder has to change this test on purpose."""
    sleeps: list[float] = []
    monkeypatch.setattr("libs.market_data.massive.time.sleep", sleeps.append)

    def always_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "slow down"})

    with pytest.raises(MarketDataError, match="persisted after 4 retries"):
        provider_with(always_limited).get_daily_bars("SPY", 5)
    assert sleeps == [2.0, 5.0, 12.0, 25.0]


def test_429_retry_after_is_capped_so_a_wild_header_cannot_park_the_request(
    monkeypatch,
):
    sleeps: list[float] = []
    monkeypatch.setattr("libs.market_data.massive.time.sleep", sleeps.append)

    def always_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "3600"}, json={"message": "slow down"}
        )

    with pytest.raises(MarketDataError, match="429"):
        provider_with(always_limited).get_daily_bars("SPY", 5)
    assert sleeps == [RATE_LIMIT_MAX_DELAY_SECONDS] * DEFAULT_MAX_RATE_LIMIT_RETRIES


def test_429_recovers_on_the_last_allowed_retry(monkeypatch):
    """N 429s then a 200 succeeds when N == the retry budget; N+1 raises."""
    monkeypatch.setattr("libs.market_data.massive.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= DEFAULT_MAX_RATE_LIMIT_RETRIES:
            return httpx.Response(429, json={"message": "slow down"})
        return _json_response({"results": [], "status": "OK"})

    assert provider_with(handler).get_daily_bars("SPY", 5) == []
    assert calls["n"] == DEFAULT_MAX_RATE_LIMIT_RETRIES + 1


def test_429_retry_budget_is_a_constructor_parameter(monkeypatch):
    """A caller on a thinner plan can widen or narrow the ladder without
    editing the module (§6.2) — pinned at 1 retry, the OLD policy."""
    sleeps: list[float] = []
    monkeypatch.setattr("libs.market_data.massive.time.sleep", sleeps.append)
    calls = {"n": 0}

    def always_limited(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"message": "slow down"})

    provider = MassiveProvider(
        api_key="k",
        transport=httpx.MockTransport(always_limited),
        max_rate_limit_retries=1,
    )
    with pytest.raises(MarketDataError, match="persisted after 1 retries"):
        provider.get_daily_bars("SPY", 5)
    assert calls["n"] == 2
    assert sleeps == [2.0]


def test_429_then_success_recovers(monkeypatch):
    monkeypatch.setattr("libs.market_data.massive.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"message": "slow down"})
        return _json_response({"results": [], "status": "OK"})

    assert provider_with(handler).get_daily_bars("SPY", 5) == []


def test_network_error_is_market_data_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(MarketDataError, match="request failed"):
        provider_with(handler).get_daily_bars("SPY", 5)


# ---------------------------------------------------------------------------
# Daily bars (aggregates)
# ---------------------------------------------------------------------------


def _agg_row(day: date, o: float, h: float, low: float, c: float, v: float) -> dict:
    ts = datetime(day.year, day.month, day.day, 9, 30, tzinfo=EASTERN)
    return {
        "t": int(ts.timestamp() * 1000),
        "o": o,
        "h": h,
        "l": low,
        "c": c,
        "v": v,
        "vw": (h + low) / 2,
        "n": 100,
    }


def test_daily_bars_parse_oldest_first_and_trim_to_days():
    days = [TODAY - timedelta(days=n) for n in range(6, 0, -1)]
    rows = [_agg_row(d, 10, 11, 9, 10.5, 1000) for d in days]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v2/aggs/ticker/SPY/range/1/day/" in str(request.url.path)
        assert request.url.params.get("adjusted") == "true"
        return _json_response(
            {"ticker": "SPY", "status": "OK", "results": rows}
        )

    bars = provider_with(handler).get_daily_bars("SPY", 4)
    assert len(bars) == 4  # trimmed to the LAST `days`
    assert [b.ts for b in bars] == days[-4:]  # oldest first
    assert bars[0].open == 10 and bars[0].high == 11
    assert bars[0].low == 9 and bars[0].close == 10.5 and bars[0].volume == 1000


def test_daily_bars_empty_results_is_honestly_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"ticker": "XQZT", "status": "OK", "queryCount": 0})

    assert provider_with(handler).get_daily_bars("XQZT", 30) == []


def test_daily_bars_malformed_rows_skipped_never_guessed():
    good = _agg_row(TODAY - timedelta(days=1), 10, 11, 9, 10.5, 500)
    rows = [good, {"o": 1, "h": 2}, "not-a-dict", {**good, "c": None}]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"status": "OK", "results": rows})

    bars = provider_with(handler).get_daily_bars("SPY", 10)
    assert len(bars) == 1 and bars[0].close == 10.5


# ---------------------------------------------------------------------------
# Quotes: stocks snapshot + indices snapshot, honest absences
# ---------------------------------------------------------------------------


def _stock_snapshot(price: float, change_pct: float, prev_close: float) -> dict:
    return {
        "status": "OK",
        "ticker": {
            "ticker": "SPY",
            "todaysChangePerc": change_pct,
            "updated": 1_700_000_000_000_000_000,
            "lastTrade": {"p": price, "t": 1_700_000_000_000_000_000},
            "day": {"c": price},
            "prevDay": {"c": prev_close},
        },
    }


def test_stock_quote_maps_snapshot_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/snapshot/locale/us/markets/stocks/tickers/SPY"
        return _json_response(_stock_snapshot(560.25, 1.2345, 553.4))

    quotes = provider_with(handler).get_quotes(["SPY"])
    assert len(quotes) == 1
    q = quotes[0]
    assert q.symbol == "SPY" and q.price == 560.25
    assert q.change_pct == pytest.approx(1.2345)


def test_stock_quote_change_derived_from_prev_close_when_field_absent():
    body = _stock_snapshot(110.0, 0.0, 100.0)
    del body["ticker"]["todaysChangePerc"]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(body)

    q = provider_with(handler).get_quotes(["SPY"])[0]
    assert q.change_pct == pytest.approx(10.0)  # (110/100 - 1) * 100


def test_unknown_symbol_is_absent_with_warning_never_invented(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if "NOSUCH" in request.url.path:
            return _json_response({"message": "not found"}, status=404)
        return _json_response(_stock_snapshot(560.0, 0.5, 557.2))

    import logging

    with caplog.at_level(logging.WARNING, logger="libs.market_data.massive"):
        quotes = provider_with(handler).get_quotes(["SPY", "NOSUCH"])
    assert [q.symbol for q in quotes] == ["SPY"]
    assert "NOSUCH" in caplog.text and "absent" in caplog.text


def test_vix_routed_to_indices_snapshot_as_i_vix():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.url.path == "/v3/snapshot/indices"
        assert request.url.params.get("ticker") == "I:VIX"
        return _json_response(
            {
                "status": "OK",
                "results": [
                    {
                        "ticker": "I:VIX",
                        "value": 16.42,
                        "session": {"change_percent": -3.1, "previous_close": 16.95},
                        "last_updated": 1_700_000_000_000_000_000,
                    }
                ],
            }
        )

    q = provider_with(handler).get_quotes(["VIX"])[0]
    assert q.symbol == "VIX" and q.price == pytest.approx(16.42)
    assert q.change_pct == pytest.approx(-3.1)
    assert paths == ["/v3/snapshot/indices"]  # never the stocks snapshot


def test_all_403_batch_reraises_capability_instead_of_empty_market():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"message": "not in plan"}, status=403)

    with pytest.raises(CapabilityNotAvailable):
        provider_with(handler).get_quotes(["SPY", "QQQ"])


def test_partial_403_still_returns_the_real_quotes():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/snapshot/indices":
            return _json_response({"message": "indices not in plan"}, status=403)
        return _json_response(_stock_snapshot(480.0, 0.2, 479.0))

    quotes = provider_with(handler).get_quotes(["QQQ", "VIX"])
    assert [q.symbol for q in quotes] == ["QQQ"]  # VIX absent, QQQ real


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------


def _chain_row(
    expiry: date,
    strike: float,
    right: str,
    *,
    bid: float | None = 4.0,
    ask: float | None = 4.4,
    greeks: dict | None = None,
    iv: float | None = 0.32,
    ticker: str | None = None,
) -> dict:
    contract_type = "call" if right == "C" else "put"
    occ = ticker or (
        f"O:SPY{expiry.strftime('%y%m%d')}{right}{int(round(strike * 1000)):08d}"
    )
    row: dict = {
        "details": {
            "ticker": occ,
            "expiration_date": expiry.isoformat(),
            "contract_type": contract_type,
            "strike_price": strike,
        },
        "greeks": greeks
        if greeks is not None
        else {"delta": 0.55, "gamma": 0.02, "theta": -0.05, "vega": 0.11},
        "implied_volatility": iv,
        "open_interest": 1200,
        "day": {"volume": 340},
        "last_trade": {"price": 4.15},
        "last_quote": {},
        "underlying_asset": {"price": 561.11},
    }
    if bid is not None:
        row["last_quote"]["bid"] = bid
    if ask is not None:
        row["last_quote"]["ask"] = ask
    return row


def test_chain_maps_contract_fields_and_prefers_snapshot_underlying():
    expiry = TODAY + timedelta(days=45)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v3/snapshot/options/SPY"
        assert request.url.params.get("limit") == str(CHAIN_PAGE_LIMIT)
        return _json_response(
            {"status": "OK", "results": [_chain_row(expiry, 560.0, "C")]}
        )

    chain = provider_with(handler).get_option_chain("SPY", 559.0, TODAY)
    assert len(chain) == 1
    c = chain[0]
    assert (c.expiry, c.strike, c.right) == (expiry, 560.0, "C")
    assert c.dte == 45
    assert c.mid == pytest.approx(4.2)
    assert c.spread_pct == pytest.approx((4.4 - 4.0) / 4.2)
    assert (c.delta, c.iv, c.open_interest) == (0.55, 0.32, 1200)


def test_chain_historical_as_of_raises_never_approximates():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made for a historical as_of")

    with pytest.raises(MarketDataError, match="current-state only"):
        provider_with(handler).get_option_chain(
            "SPY", 560.0, TODAY - timedelta(days=1)
        )


def test_chain_follows_next_url_pagination_and_caps_pages():
    expiry = TODAY + timedelta(days=30)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        # Every page returns one contract and points at another page, forever.
        strike = 500.0 + len(calls)
        return _json_response(
            {
                "status": "OK",
                "results": [_chain_row(expiry, strike, "C")],
                "next_url": "https://api.massive.com/v3/snapshot/options/SPY?cursor=n",
            }
        )

    provider = provider_with(handler, max_chain_pages=3)
    chain = provider.get_option_chain("SPY", 560.0, TODAY)
    assert len(calls) == 3  # page cap enforced
    assert len(chain) == 3
    assert "cursor=n" in calls[1]  # pagination genuinely followed


def test_chain_skips_unquotable_greekless_expired_and_mismatched_rows():
    expiry = TODAY + timedelta(days=30)
    rows = [
        _chain_row(expiry, 100.0, "C"),  # good
        _chain_row(expiry, 105.0, "C", bid=None, ask=None),  # unquotable
        _chain_row(expiry, 110.0, "C", greeks={"delta": 0.5}),  # greeks missing
        _chain_row(expiry, 115.0, "C", iv=None),  # IV missing
        _chain_row(TODAY - timedelta(days=3), 120.0, "C"),  # expired
        # OCC ticker disagrees with details -> refuse to guess
        _chain_row(
            expiry,
            125.0,
            "C",
            ticker=f"O:SPY{expiry.strftime('%y%m%d')}P{int(99 * 1000):08d}",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"status": "OK", "results": rows})

    chain = provider_with(handler).get_option_chain("SPY", 560.0, TODAY)
    assert [c.strike for c in chain] == [100.0]


def test_chain_midpoint_only_quote_uses_conservative_spread():
    expiry = TODAY + timedelta(days=30)
    row = _chain_row(expiry, 100.0, "P", bid=None, ask=None)
    row["last_quote"]["midpoint"] = 3.3

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"status": "OK", "results": [row]})

    c = provider_with(handler).get_option_chain("SPY", 560.0, TODAY)[0]
    assert c.mid == pytest.approx(3.3)
    assert c.spread_pct >= 1.0  # worst-case spread: can only ever REJECT in §9


def test_chain_quoteless_plan_falls_back_to_fresh_day_close():
    """A quotes-less options tier serves NO last_quote block at all (observed
    live): the row prices from the DAY bar's close — a real traded session
    price — with the worst-case spread, so the chain is visible with real
    greeks/IV/OI while unknown bid/ask quality can only REJECT in §9. A
    STALE day bar (expired session) is refused, never a usable price."""
    import time as _time

    expiry = TODAY + timedelta(days=30)
    fresh = _chain_row(expiry, 100.0, "C", bid=None, ask=None)
    del fresh["last_quote"]  # the plan omits the block entirely
    fresh["day"] = {
        "close": 2.85,
        "volume": 340,
        "last_updated": int(_time.time() * 1e9),  # this session, in ns
    }

    stale = _chain_row(expiry, 105.0, "C", bid=None, ask=None)
    del stale["last_quote"]
    stale["day"] = {
        "close": 1.10,
        "volume": 5,
        "last_updated": int((_time.time() - 30 * 86400) * 1e9),  # a month old
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"status": "OK", "results": [fresh, stale]})

    chain = provider_with(handler).get_option_chain("SPY", 560.0, TODAY)
    assert len(chain) == 1  # the stale-session row is refused
    c = chain[0]
    assert c.strike == 100.0
    assert c.mid == pytest.approx(2.85)
    assert c.bid == 0.0 and c.ask == 0.0  # no NBBO: honest zeros, not guesses
    assert c.spread_pct >= 1.0  # worst case: can only ever REJECT in §9
    assert c.iv == pytest.approx(0.32)  # real greeks/IV ride along untouched


def test_chain_sorted_deterministically():
    e1, e2 = TODAY + timedelta(days=30), TODAY + timedelta(days=60)
    rows = [
        _chain_row(e2, 110.0, "P"),
        _chain_row(e1, 120.0, "C"),
        _chain_row(e1, 110.0, "P"),
        _chain_row(e1, 110.0, "C"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"status": "OK", "results": rows})

    chain = provider_with(handler).get_option_chain("SPY", 560.0, TODAY)
    assert [(c.expiry, c.strike, c.right) for c in chain] == [
        (e1, 110.0, "C"),
        (e1, 110.0, "P"),
        (e1, 120.0, "C"),
        (e2, 110.0, "P"),
    ]


# ---------------------------------------------------------------------------
# OCC ticker parsing
# ---------------------------------------------------------------------------


def test_occ_ticker_parse_round_trip():
    parsed = _parse_occ_ticker("O:AAPL211022C000150000")
    assert parsed == (date(2021, 10, 22), "C", 150.0)
    assert _parse_occ_ticker("O:F260116P00012500") == (date(2026, 1, 16), "P", 12.5)
    assert _parse_occ_ticker("garbage") is None


# ---------------------------------------------------------------------------
# Capability probe (§16)
# ---------------------------------------------------------------------------


def test_probe_reports_per_capability_truth():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/v2/aggs/"):
            return _json_response({"status": "OK", "results": []})
        if path.startswith("/v2/snapshot/"):
            return _json_response({"message": "plan does not include"}, status=403)
        if path.startswith("/v3/snapshot/options/"):
            return _json_response({"status": "OK", "results": []})
        if path == "/v3/reference/options/contracts":
            # The Basic-tier reference endpoint answers even when the chain
            # snapshot is plan-gated — probed SEPARATELY.
            return _json_response({"status": "OK", "results": []})
        if path == "/v2/reference/news":
            return _json_response({"status": "OK", "results": []})
        if path == "/vX/reference/financials":
            return _json_response({"status": "OK", "results": []})
        raise AssertionError(f"unexpected probe path {path}")

    report = provider_with(handler).probe_capabilities()
    assert report["stock_history"] is True
    assert report["stock_realtime"] is False  # 403 -> plan-gated, not an error
    assert report["option_chain"] is True
    assert report["option_contracts"] is True
    assert report["news"] is True
    assert report["financials"] is True  # the ONE fundamentals endpoint (§11.3)


def test_probe_records_error_text_on_faults():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    report = provider_with(handler).probe_capabilities()
    assert all(isinstance(v, str) and "failed" in v for v in report.values())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_massive_with_key_and_refuses_blank(monkeypatch):
    from libs.common.config import get_settings
    from libs.market_data import ProviderNotConfigured, get_provider

    monkeypatch.setenv("MASSIVE_API_KEY", "reg-test-key")
    get_settings.cache_clear()
    try:
        provider = get_provider("massive")
        assert isinstance(provider, MassiveProvider)
        with pytest.raises(ProviderNotConfigured):
            get_provider("")
    finally:
        get_settings.cache_clear()


def test_registry_massive_without_key_fails_loudly(monkeypatch):
    from libs.common.config import get_settings
    from libs.market_data import get_provider

    monkeypatch.setenv("MASSIVE_API_KEY", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(MarketDataError, match="MASSIVE_API_KEY"):
            get_provider("massive")
    finally:
        get_settings.cache_clear()
