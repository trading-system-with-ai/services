from datetime import datetime, timezone

from libs.market_data.stub import StubProvider
from libs.trading_core.models import MarketRegime
from libs.trading_core.signals import classify_regime


async def test_market_overview_shape(client):
    r = await client.get("/api/market/overview")
    assert r.status_code == 200
    body = r.json()

    assert body["provider"] == "stub"
    assert body["stale"] is False
    datetime.fromisoformat(body["as_of"])  # valid ISO-8601
    assert body["market_regime"] in {m.value for m in MarketRegime}

    indices = body["indices"]
    assert len(indices) == 3
    symbols = {i["symbol"] for i in indices}
    assert symbols == {"SPY", "QQQ", "VIX"}
    for idx in indices:
        assert idx["price"] > 0
        assert isinstance(idx["change_pct"], float)
        datetime.fromisoformat(idx["ts"])  # valid ISO-8601


async def test_market_regime_is_computed_from_spy_bars(client):
    """The regime is now COMPUTED by classify_regime over SPY daily bars
    (plan §6.1), not the old NEUTRAL_RANGE placeholder."""
    r = await client.get("/api/market/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["market_regime"] in {m.value for m in MarketRegime}

    # The stub is deterministic per symbol and end date, so recomputing the
    # regime through the shared library must give the served value exactly.
    # The platform stores COMPLETE trading days only (today's bar is still
    # forming — ensure_daily_bars drops it and fetches one extra), so the
    # expected series applies the same trim.
    from apps.gateway.routers.analysis import _complete_days_only

    bars = _complete_days_only(StubProvider().get_daily_bars("SPY", 602))[-600:]
    expected = classify_regime(
        [b.close for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
    )
    assert body["market_regime"] == expected.classification.value


async def test_spy_backfill_is_system_audited_and_runs_once(client):
    """SPY is a system reference symbol (ADR-005): the first overview request
    lazily backfills its bars with a SYSTEM DATA_BACKFILL audit event; the
    second request must not backfill again."""
    await client.get("/api/market/overview")
    await client.get("/api/market/overview")

    r = await client.get("/api/audit", params={"entity_id": "SPY"})
    backfills = [e for e in r.json() if e["action"] == "DATA_BACKFILL"]
    assert len(backfills) == 1
    assert backfills[0]["actor_type"] == "SYSTEM"
    assert backfills[0]["details"]["provider"] == "stub"


def test_stub_provider_deterministic_within_minute():
    provider = StubProvider()
    now = datetime(2026, 8, 10, 14, 30, 12, tzinfo=timezone.utc)
    later_same_minute = now.replace(second=45)
    a = provider.get_quotes(["SPY", "QQQ", "VIX"], now=now)
    b = provider.get_quotes(["SPY", "QQQ", "VIX"], now=later_same_minute)
    assert [(q.symbol, q.price, q.change_pct) for q in a] == [
        (q.symbol, q.price, q.change_pct) for q in b
    ]


# ---------------------------------------------------------------------------
# GET /api/market/capabilities (guide §16) — probed, cached, honest.
# ---------------------------------------------------------------------------


def _reset_capabilities_cache():
    from apps.gateway.routers import market

    market._capabilities_cache.update(
        {"at": None, "provider": None, "payload": None}
    )


async def test_capabilities_503_when_unconfigured(unconfigured_client):
    _reset_capabilities_cache()
    r = await unconfigured_client.get("/api/market/capabilities")
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "MARKET_DATA_NOT_CONFIGURED"


async def test_capabilities_honest_null_for_probeless_provider(client):
    """The stub has no plan to detect: capabilities must be null with a
    message — never a fabricated all-true verdict."""
    _reset_capabilities_cache()
    r = await client.get("/api/market/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "stub"
    assert body["capabilities"] is None
    assert "does not support capability probing" in body["message"]


async def test_capabilities_probed_and_cached(client, monkeypatch):
    """A probing provider's verdicts pass through verbatim; the probe runs
    once per TTL window and again on ?refresh=true."""
    _reset_capabilities_cache()
    calls = {"n": 0}

    class Probing:
        def probe_capabilities(self):
            calls["n"] += 1
            return {
                "stock_history": True,
                "stock_realtime": True,
                "option_chain": False,
            }

    monkeypatch.setattr(
        "apps.gateway.routers.market.get_provider", lambda name: Probing()
    )

    r = await client.get("/api/market/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["capabilities"] == {
        "stock_history": True,
        "stock_realtime": True,
        "option_chain": False,
    }
    assert body["message"] is None
    assert calls["n"] == 1

    # Within the TTL the cached verdict is served — no second probe.
    await client.get("/api/market/capabilities")
    assert calls["n"] == 1

    # refresh=true re-probes (e.g. right after a plan upgrade).
    await client.get("/api/market/capabilities", params={"refresh": "true"})
    assert calls["n"] == 2
    _reset_capabilities_cache()
