"""AlpacaMarketDataProvider — the authoritative market-data source
(data_source.md §1/§2). Every test runs the real adapter over
``httpx.MockTransport``; the network is never touched.

Pinned, in order of importance:

1. NO FABRICATION: symbols Alpaca cannot serve (VIX — no index feed) are
   honestly absent; contracts without greeks/IV are skipped, never
   zero-filled; historical chains raise, never approximate.
2. CAPABILITY HONESTY (§16): HTTP 403 -> CapabilityNotAvailable, distinct
   from transport faults; keyless construction refused naming the env vars.
3. The exact wire mapping against LIVE-VERIFIED Alpaca shapes (2026-08-13,
   real Algo Trader Plus account): bars t RFC-3339 at 04:00Z with o/h/l/c/
   v/n/vw; stock snapshots keyed by symbol with latestTrade/latestQuote/
   prevDailyBar; option chain snapshots keyed by BARE OCC symbol with
   latestQuote bp/ap/bs/as, greeks incl. rho, impliedVolatility, dailyBar;
   Trading-API contracts with STRINGIFIED numerics (open_interest,
   strike_price, close_price); news with integer ids.
"""
from datetime import date, datetime, timedelta

import httpx
import pytest

from libs.market_data.alpaca import (
    EASTERN,
    AlpacaMarketDataProvider,
    _parse_bare_occ,
)
from libs.market_data.provider import CapabilityNotAvailable, MarketDataError

TODAY = datetime.now(EASTERN).date()


def provider_with(handler, **kwargs) -> AlpacaMarketDataProvider:
    return AlpacaMarketDataProvider(
        api_key_id="test-key-id",
        api_secret_key="test-secret",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _json(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _occ(root: str, expiry: date, right: str, strike: float) -> str:
    return f"{root}{expiry.strftime('%y%m%d')}{right}{int(round(strike * 1000)):08d}"


# ---------------------------------------------------------------------------
# Construction, auth headers, taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["api_key_id", "api_secret_key"])
@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_credentials_refused_at_construction(field, bad):
    kwargs = {"api_key_id": "k", "api_secret_key": "s"}
    kwargs[field] = bad
    with pytest.raises(MarketDataError, match="ALPACA_API"):
        AlpacaMarketDataProvider(**kwargs)


def test_auth_headers_sent_and_key_never_in_errors():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("APCA-API-KEY-ID")
        seen["secret"] = request.headers.get("APCA-API-SECRET-KEY")
        return httpx.Response(500, text="boom")

    p = provider_with(handler)
    with pytest.raises(MarketDataError) as exc:
        p.get_daily_bars("SPY", 5)
    assert seen == {"key": "test-key-id", "secret": "test-secret"}
    assert "test-key-id" not in str(exc.value)
    assert "test-secret" not in str(exc.value)


def test_403_is_capability_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="subscription does not permit")

    with pytest.raises(CapabilityNotAvailable, match="subscription"):
        provider_with(handler).get_daily_bars("SPY", 5)


def test_registry_resolves_alpaca_and_refuses_keyless(monkeypatch):
    from libs.common.config import get_settings
    from libs.market_data import get_provider

    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    get_settings.cache_clear()
    from libs.market_data.alpaca import AlpacaMarketDataProvider as Cls

    assert isinstance(get_provider("alpaca"), Cls)

    monkeypatch.setenv("ALPACA_API_KEY_ID", "")
    get_settings.cache_clear()
    with pytest.raises(MarketDataError, match="ALPACA_API_KEY_ID"):
        get_provider("alpaca")
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Daily bars
# ---------------------------------------------------------------------------


def test_daily_bars_map_the_verified_wire_shape():
    """t is RFC-3339 at 04:00Z (midnight ET) — the trading DATE is Eastern."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/stocks/SPY/bars"
        q = dict(request.url.params)
        assert q["timeframe"] == "1Day"
        assert q["adjustment"] == "split"
        return _json({
            "symbol": "SPY",
            "next_page_token": None,
            "bars": [
                {"t": "2026-08-12T04:00:00Z", "o": 774.71, "h": 774.9,
                 "l": 771.28, "c": 772.49, "v": 33536044, "n": 496091,
                 "vw": 772.716635},
                {"t": "2026-08-13T04:00:00Z", "o": 774.87, "h": 779.37,
                 "l": 774.111, "c": 777.88, "v": 35713979, "n": 499516,
                 "vw": 777.278433},
            ],
        })

    bars = provider_with(handler).get_daily_bars("SPY", 2)
    assert [b.ts for b in bars] == [date(2026, 8, 12), date(2026, 8, 13)]
    assert bars[-1].close == pytest.approx(777.88)
    assert bars[-1].volume == pytest.approx(35713979)


def test_daily_bars_follow_next_page_token_and_trim_to_days():
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("page_token")
        calls.append(token)
        if token is None:
            return _json({
                "bars": [{"t": "2026-08-11T04:00:00Z", "o": 1, "h": 1,
                          "l": 1, "c": 1.0, "v": 10}],
                "next_page_token": "tok-2",
            })
        return _json({
            "bars": [
                {"t": "2026-08-12T04:00:00Z", "o": 2, "h": 2, "l": 2,
                 "c": 2.0, "v": 20},
                {"t": "2026-08-13T04:00:00Z", "o": 3, "h": 3, "l": 3,
                 "c": 3.0, "v": 30},
            ],
            "next_page_token": None,
        })

    bars = provider_with(handler).get_daily_bars("SPY", 2)
    assert calls == [None, "tok-2"]
    assert [b.close for b in bars] == [2.0, 3.0]  # trimmed to the last 2


# ---------------------------------------------------------------------------
# Quotes (multi-symbol snapshot)
# ---------------------------------------------------------------------------


def _spy_snapshot() -> dict:
    return {
        "latestTrade": {"p": 777.88, "t": "2026-08-14T00:00:00.044518299Z"},
        "latestQuote": {"bp": 778.05, "ap": 778.12},
        "prevDailyBar": {"c": 772.49},
    }


def test_quotes_one_call_price_and_change_from_snapshot():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/stocks/snapshots"
        assert dict(request.url.params)["symbols"] == "SPY,QQQ"
        return _json({"SPY": _spy_snapshot(),
                      "QQQ": {"latestTrade": {"p": 732.29,
                                              "t": "2026-08-13T23:59:35Z"},
                              "prevDailyBar": {"c": 725.0}}})

    quotes = provider_with(handler).get_quotes(["SPY", "QQQ"])
    assert [q.symbol for q in quotes] == ["SPY", "QQQ"]
    assert quotes[0].price == pytest.approx(777.88)
    assert quotes[0].change_pct == pytest.approx((777.88 / 772.49 - 1) * 100)
    # Nanosecond timestamp parsed (trimmed to microseconds, UTC-aware).
    assert quotes[0].ts.tzinfo is not None


def test_vix_is_an_honest_absence_never_a_proxy():
    """Alpaca serves no index feed: VIX is skipped BEFORE the request and the
    request only names servable symbols."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["symbols"] = dict(request.url.params)["symbols"]
        return _json({"SPY": _spy_snapshot()})

    quotes = provider_with(handler).get_quotes(["SPY", "VIX"])
    assert seen["symbols"] == "SPY"  # VIX never requested
    assert [q.symbol for q in quotes] == ["SPY"]


def test_all_unservable_symbols_returns_empty_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made")

    assert provider_with(handler).get_quotes(["VIX"]) == []


# ---------------------------------------------------------------------------
# Option chain (snapshots + OI merge)
# ---------------------------------------------------------------------------

EXPIRY = TODAY + timedelta(days=8)


def _chain_snapshot_row(*, with_quote=True, with_greeks=True) -> dict:
    row: dict = {
        "dailyBar": {"c": 0.57, "v": 183, "o": 0.55, "h": 1.0, "l": 0.55,
                     "t": "2026-08-13T04:00:00Z", "vw": 0.712},
        "latestTrade": {"p": 0.57, "s": 8,
                        "t": "2026-08-13T19:58:53.384368925Z"},
    }
    if with_quote:
        row["latestQuote"] = {"bp": 0.5, "ap": 0.6, "bs": 2023, "as": 114,
                              "t": "2026-08-13T19:59:59.998758551Z"}
    if with_greeks:
        row["greeks"] = {"delta": 0.4586, "gamma": 0.2335, "theta": -0.0427,
                         "vega": 0.0078, "rho": 0.0012}
        row["impliedVolatility"] = 0.8697
    return row


def _contracts_payload(symbols_oi: dict[str, int]) -> dict:
    return {
        "next_page_token": None,
        "option_contracts": [
            {
                "symbol": occ,
                "type": "call" if occ[-9] == "C" else "put",
                "strike_price": str(int(occ[-8:]) / 1000.0),
                "expiration_date": f"20{occ[-15:-13]}-{occ[-13:-11]}-{occ[-11:-9]}",
                "size": "100",
                "open_interest": str(oi),
                "close_price": "0.8",
                "close_price_date": "2026-08-12",
                "status": "active",
                "tradable": True,
            }
            for occ, oi in symbols_oi.items()
        ],
    }


def test_chain_merges_real_nbbo_greeks_and_contract_open_interest():
    occ = _occ("RDW", EXPIRY, "C", 13.5)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/options/contracts":
            assert request.url.host == "paper-api.alpaca.markets"
            return _json(_contracts_payload({occ: 224}))
        assert request.url.path == "/v1beta1/options/snapshots/RDW"
        assert dict(request.url.params)["feed"] == "opra"
        return _json({"next_page_token": None,
                      "snapshots": {occ: _chain_snapshot_row()}})

    chain = provider_with(handler).get_option_chain("RDW", 13.49, TODAY)
    assert len(chain) == 1
    c = chain[0]
    assert (c.expiry, c.right, c.strike) == (EXPIRY, "C", 13.5)
    assert c.bid == pytest.approx(0.5)
    assert c.ask == pytest.approx(0.6)
    assert c.mid == pytest.approx(0.55)
    assert c.price_basis == "quote"
    assert c.spread_pct == pytest.approx(0.1 / 0.55)
    assert c.iv == pytest.approx(0.8697)
    assert c.delta == pytest.approx(0.4586)
    assert c.open_interest == 224  # merged from the Trading API contracts
    assert c.volume == 183  # dailyBar volume
    assert c.last == pytest.approx(0.57)


def test_chain_keeps_greekless_rows_with_honest_nulls_and_day_close_pricing():
    """A contract with a REAL quote but no greeks is real chain data the user
    should SEE — kept with iv/greeks None (never zero-filled); the §9
    selector rejects it with a named reason downstream."""
    occ_ok = _occ("RDW", EXPIRY, "C", 13.0)
    occ_no_greeks = _occ("RDW", EXPIRY, "P", 13.0)
    occ_no_quote = _occ("RDW", EXPIRY, "C", 14.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/options/contracts":
            return _json(_contracts_payload({}))
        return _json({
            "next_page_token": None,
            "snapshots": {
                occ_ok: _chain_snapshot_row(),
                occ_no_greeks: _chain_snapshot_row(with_greeks=False),
                occ_no_quote: _chain_snapshot_row(with_quote=False),
            },
        })

    chain = provider_with(handler).get_option_chain("RDW", 13.49, TODAY)
    by = {(c.strike, c.right): c for c in chain}
    greekless = by[(13.0, "P")]  # kept — completeness with honest nulls
    assert greekless.iv is None
    assert greekless.delta is None and greekless.theta is None
    assert greekless.bid == pytest.approx(0.5)  # its REAL quote rides along
    assert by[(13.0, "C")].price_basis == "quote"
    quoteless = by[(14.0, "C")]
    assert quoteless.price_basis == "day_close"  # real session close
    assert quoteless.mid == pytest.approx(0.57)
    assert quoteless.bid == 0.0 and quoteless.ask == 0.0
    assert quoteless.spread_pct >= 1.0  # worst case: §9 can only REJECT
    assert by[(13.0, "C")].open_interest == 0  # no OI reported -> honest 0

    # And the §9 selector names the rejection instead of guessing.
    from libs.trading_core.contracts import select_contracts

    verdict = next(
        s for s in select_contracts(chain, "BEAR")
        if s.contract.right == "P" and s.contract.strike == 13.0
    )
    assert not verdict.eligible
    assert any("greeks/IV not provided" in r for r in verdict.fail_reasons)


def test_chain_one_sided_nbbo_keeps_the_real_ask():
    """OPRA reports bid 0 when NO BID exists — a real market state (observed
    live on deep OTM wings). The real ask must be KEPT, not discarded to the
    day-close fallback: bid renders as the reported 0, mid is the canonical
    no-bid midpoint (ask/2), and the worst-case spread lets §9 only reject."""
    occ = _occ("RDW", EXPIRY, "P", 12.5)
    row = _chain_snapshot_row(with_greeks=False)
    row["latestQuote"] = {"bp": 0, "bs": 0, "ap": 0.05, "as": 9,
                          "t": "2026-08-13T19:51:05.65612264Z"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/options/contracts":
            return _json(_contracts_payload({}))
        return _json({"next_page_token": None, "snapshots": {occ: row}})

    chain = provider_with(handler).get_option_chain("RDW", 13.49, TODAY)
    assert len(chain) == 1
    c = chain[0]
    assert c.price_basis == "quote"  # it IS the NBBO, one-sided
    assert c.bid == 0.0  # the reported no-bid, not an unknown
    assert c.ask == pytest.approx(0.05)  # the REAL offer, preserved
    assert c.mid == pytest.approx(0.025)
    assert c.spread_pct >= 1.0  # worst case: §9 can only ever REJECT


def test_chain_historical_as_of_raises_never_approximates():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made")

    with pytest.raises(MarketDataError, match="current-state only"):
        provider_with(handler).get_option_chain(
            "RDW", 13.49, TODAY - timedelta(days=1)
        )


def test_chain_oi_merge_failure_degrades_to_zero_never_blocks_the_chain():
    occ = _occ("RDW", EXPIRY, "C", 13.5)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/options/contracts":
            return httpx.Response(403, text="not permitted")
        return _json({"next_page_token": None,
                      "snapshots": {occ: _chain_snapshot_row()}})

    chain = provider_with(handler).get_option_chain("RDW", 13.49, TODAY)
    assert len(chain) == 1
    assert chain[0].open_interest == 0  # "none reported", never invented


# ---------------------------------------------------------------------------
# Contracts + prev bar (EOD surface)
# ---------------------------------------------------------------------------


def test_option_contracts_parse_stringified_numerics():
    occ = _occ("RDW", EXPIRY, "C", 1.0)

    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        assert q["underlying_symbols"] == "RDW"
        assert q["status"] == "active"
        return _json(_contracts_payload({occ: 34}))

    rows = provider_with(handler).get_option_contracts("RDW")
    assert rows == [{
        "ticker": occ,
        "contract_type": "call",
        "strike_price": 1.0,
        "expiration_date": EXPIRY,
        "shares_per_contract": 100.0,
        "open_interest": 34.0,
        "close_price": 0.8,
        "close_price_date": "2026-08-12",
    }]


def test_option_prev_bar_reads_prev_daily_bar_from_snapshot():
    occ = _occ("RDW", EXPIRY, "C", 13.5)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta1/options/snapshots"
        assert dict(request.url.params)["symbols"] == occ
        return _json({"snapshots": {occ: {
            "prevDailyBar": {"c": 0.8, "o": 0.83, "h": 1.05, "l": 0.75,
                             "v": 102, "vw": 0.855,
                             "t": "2026-08-12T04:00:00Z"},
        }}})

    bar = provider_with(handler).get_option_prev_bar(occ)
    assert bar["close"] == pytest.approx(0.8)
    assert bar["volume"] == pytest.approx(102)
    assert bar["date"] == "2026-08-12"


def test_option_prev_bar_missing_is_honest_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json({"snapshots": {}})

    assert provider_with(handler).get_option_prev_bar("RDW260821C00013500") is None


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def test_news_maps_verbatim_with_prefixed_ids_and_skips_uncitable():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta1/news"
        return _json({"news": [
            {"id": 61197943, "headline": "Tariff proclamation",
             "author": "Benzinga Newsdesk", "source": "benzinga",
             "created_at": "2026-08-13T21:37:18Z",
             "url": "https://example.com/a", "summary": "sum",
             "symbols": ["RDW", "AVAV"]},
            {"id": 2, "headline": "", "url": "https://example.com/b",
             "created_at": "2026-08-13T21:00:00Z"},  # no headline -> skipped
        ]})

    articles = provider_with(handler).get_news(limit=10)
    assert len(articles) == 1
    a = articles[0]
    assert a.source_id == "alpaca:61197943"  # id-space can never collide
    assert a.title == "Tariff proclamation"
    assert a.publisher == "benzinga"
    assert a.url == "https://example.com/a"
    assert a.tickers == ("RDW", "AVAV")
    assert a.published_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


def test_probe_reports_the_exact_platform_capability_keys():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/stocks/SPY/bars":
            return _json({"bars": []})
        if path == "/v2/stocks/snapshots":
            return _json({})
        if path == "/v1beta1/options/snapshots/SPY":
            return httpx.Response(403, text="not in subscription")
        if path == "/v2/options/contracts":
            return _json({"option_contracts": []})
        if path == "/v1beta1/news":
            return _json({"news": []})
        raise AssertionError(f"unexpected probe path {path}")

    report = provider_with(handler).probe_capabilities()
    assert report == {
        "stock_history": True,
        "stock_realtime": True,
        "option_chain": False,  # 403 -> plan-gated, not an error
        "option_contracts": True,
        "news": True,
        # Alpaca sells no fundamentals at any tier — a constant False, not a
        # probe, so both providers answer the SAME key set (§16).
        "financials": False,
    }


# ---------------------------------------------------------------------------
# OCC parsing
# ---------------------------------------------------------------------------


def test_bare_occ_parser_round_trips():
    assert _parse_bare_occ("RDW260814C00013500") == (date(2026, 8, 14), "C", 13.5)
    assert _parse_bare_occ("BRK.B270115P01234500") == (date(2027, 1, 15), "P", 1234.5)
    assert _parse_bare_occ("O:RDW260814C00013500") is None  # Massive form
    assert _parse_bare_occ("garbage") is None
