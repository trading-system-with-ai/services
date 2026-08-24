"""Alpaca market data stream layer (data_source.md §5).

The connection loop is not exercised here (no sockets in tests); what IS
pinned is everything deterministic: protocol parsing, the quote cache's
freshness contract (stale = absent, never a stale number served as fresh),
subscription message shapes, the status surface, and the overview's
stream-over-REST override with provenance-preserving change_pct rebasing.
"""
from datetime import datetime, timedelta, timezone

import pytest

from libs.market_data.alpaca_stream import (
    QuoteCache,
    auth_message,
    parse_stream_payload,
    subscribe_message,
)

NOW = datetime(2026, 8, 13, 20, 0, 0, tzinfo=timezone.utc)


def test_parse_stream_payload_arrays_objects_and_garbage():
    assert parse_stream_payload('[{"T":"q","S":"SPY"}]') == [{"T": "q", "S": "SPY"}]
    assert parse_stream_payload('{"T":"success","msg":"connected"}') == [
        {"T": "success", "msg": "connected"}
    ]
    assert parse_stream_payload("not json") == []
    assert parse_stream_payload('"a string"') == []
    assert parse_stream_payload('[1, {"T":"t","S":"QQQ"}]') == [{"T": "t", "S": "QQQ"}]


def test_cache_applies_quotes_and_trades_verbatim():
    cache = QuoteCache()
    cache.apply(
        [
            {"T": "q", "S": "SPY", "bp": 778.05, "ap": 778.12, "bs": 4, "as": 8,
             "t": "2026-08-13T19:59:59.817519220Z"},
            {"T": "t", "S": "SPY", "p": 778.10, "s": 100,
             "t": "2026-08-13T19:59:59.9Z"},
            {"T": "success", "msg": "authenticated"},  # non-data: ignored
            {"T": "t", "S": "QQQ", "p": 0, "s": 1, "t": "2026-08-13T19:59:59Z"},
        ],
        now=NOW,
    )
    spy = cache.get("SPY")
    assert spy.bid == pytest.approx(778.05)
    assert spy.ask == pytest.approx(778.12)
    assert spy.last_price == pytest.approx(778.10)
    assert spy.quote_ts is not None and spy.trade_ts is not None
    assert cache.messages_applied == 2  # the zero-price QQQ trade is refused
    assert cache.get("QQQ") is None or cache.get("QQQ").last_price is None


def test_cache_freshness_contract_stale_is_absent():
    cache = QuoteCache()
    cache.apply(
        [{"T": "t", "S": "SPY", "p": 778.0, "s": 1, "t": "2026-08-13T19:59:59Z"}],
        now=NOW,
    )
    assert cache.fresh("SPY", 30.0, now=NOW + timedelta(seconds=29)) is not None
    assert cache.fresh("SPY", 30.0, now=NOW + timedelta(seconds=31)) is None
    assert cache.fresh("QQQ", 30.0, now=NOW) is None


def test_protocol_messages_shapes():
    import json

    auth = json.loads(auth_message("k", "s"))
    assert auth == {"action": "auth", "key": "k", "secret": "s"}
    sub = json.loads(subscribe_message(["SPY", "RDW"]))
    assert sub == {"action": "subscribe", "trades": ["SPY", "RDW"],
                   "quotes": ["SPY", "RDW"]}


async def test_stream_status_endpoint_reports_disabled_under_stub(client):
    """With the stub provider the supervisor self-disables; the endpoint
    reports honest facts, never a fabricated connection."""
    r = await client.get("/api/market/stream/status")
    assert r.status_code == 200
    body = r.json()
    assert {"state", "subscribed", "messages_applied", "cached_symbols",
            "fresh_window_seconds"} <= set(body)
    # ASGI test transport runs no lifespan, so the supervisor never ran:
    # state stays "starting" (or "disabled" if a prior test cycled it) —
    # either way it must NOT claim a connection.
    assert body["state"] in ("starting", "disabled")


async def test_chain_cache_bounds_provider_calls_and_execution_bypasses(client, monkeypatch):
    """§5 economy + §21/§42 honesty: polling read surfaces reuse a
    seconds-old chain build; max_age_seconds=0 always hits the provider."""
    from apps.gateway.routers import options as options_router

    calls = {"n": 0}
    real_get_provider = options_router.get_provider

    def counting_get_provider(name):
        provider = real_get_provider(name)
        original = provider.get_option_chain

        def counted(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        provider.get_option_chain = counted
        return provider

    monkeypatch.setattr(options_router, "get_provider", counting_get_provider)
    options_router._chain_cache.clear()

    as_of1, chain1 = options_router.build_option_chain("VZ", 25.0)
    as_of2, chain2 = options_router.build_option_chain("VZ", 25.0)
    assert calls["n"] == 1  # second read served from the short cache
    assert (as_of1, chain1) == (as_of2, chain2)

    options_router.build_option_chain("VZ", 25.0, max_age_seconds=0.0)
    assert calls["n"] == 2  # execution-style read always hits the provider


async def test_overview_prefers_fresh_stream_price_with_rebased_change(client):
    """§5 override: a FRESH streamed trade supersedes the REST snapshot and
    the day change is rebased on the SAME previous close (provenance keeps
    one baseline, two transports)."""
    from apps.gateway import market_stream

    r = await client.get("/api/market/overview")
    assert r.status_code == 200
    rest = {i["symbol"]: i for i in r.json()["indices"]}
    assert all(i["transport"] == "rest" for i in rest.values())

    spy_rest = rest["SPY"]
    prev_close = spy_rest["price"] / (1 + spy_rest["change_pct"] / 100.0)
    streamed_price = spy_rest["price"] * 1.01  # +1% on a streamed trade
    market_stream.CACHE.apply(
        [{
            "T": "t", "S": "SPY", "p": streamed_price, "s": 10,
            # Any instant at/after the REST quote's ts qualifies.
            "t": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        }]
    )
    try:
        r = await client.get("/api/market/overview")
        spy = {i["symbol"]: i for i in r.json()["indices"]}["SPY"]
        assert spy["transport"] == "stream"
        assert spy["price"] == pytest.approx(streamed_price)
        assert spy["change_pct"] == pytest.approx(
            (streamed_price / prev_close - 1) * 100.0
        )
        # Other symbols stay on the REST path — no invented stream data.
        assert {i["symbol"]: i for i in r.json()["indices"]}["QQQ"]["transport"] == "rest"
    finally:
        market_stream.CACHE.clear()
