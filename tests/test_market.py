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
    bars = StubProvider().get_daily_bars("SPY", 600)
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
