"""Tests for GET /api/watchlist/{ticker}/analysis (plan §4.2, §6)."""
from datetime import date, datetime

from libs.trading_core.models import DirectionalBias, MarketRegime

CONTRACT_KEYS = {
    "ticker",
    "as_of",
    "source",
    "bars",
    "price",
    "indicators",
    "regime",
    "signal",
    "series",
}
INDICATOR_KEYS = {
    "sma20",
    "sma50",
    "sma200",
    "rsi14",
    "atr14",
    "atr_pct",
    "macd",
    "macd_signal",
    "macd_histogram",
    "realized_vol20",
}
COMPONENT_KEYS = {"name", "side", "triggered", "weight", "detail"}


async def test_analysis_404_for_non_watchlist_ticker(client):
    """Historical data may exist ONLY for Watchlist symbols (plan §4.2)."""
    r = await client.get("/api/watchlist/NVDA/analysis")
    assert r.status_code == 404
    assert "not on the watchlist" in r.json()["detail"]


async def test_analysis_contract(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.get("/api/watchlist/NVDA/analysis")
    assert r.status_code == 200
    body = r.json()

    assert set(body) == CONTRACT_KEYS
    assert body["ticker"] == "NVDA"
    assert body["source"] == "stub"
    datetime.fromisoformat(body["as_of"])  # valid ISO-8601

    bars = body["bars"]
    assert set(bars) == {"count", "first", "last"}
    assert isinstance(bars["count"], int) and bars["count"] >= 250
    assert date.fromisoformat(bars["first"]) < date.fromisoformat(bars["last"])

    assert isinstance(body["price"], float) and body["price"] > 0

    indicators = body["indicators"]
    assert set(indicators) == INDICATOR_KEYS
    # 600 backfilled bars cover every warmup, so all indicators are defined.
    for key, value in indicators.items():
        assert isinstance(value, float), f"indicator {key} should be a float"
    assert indicators["atr_pct"] > 0

    regime = body["regime"]
    assert regime["classification"] in {m.value for m in MarketRegime}
    assert isinstance(regime["features"], dict) and regime["features"]

    signal = body["signal"]
    assert 0.0 <= signal["bull_score"] <= 100.0
    assert 0.0 <= signal["bear_score"] <= 100.0
    assert signal["directional_edge"] == signal["bull_score"] - signal["bear_score"]
    assert signal["bias"] in {b.value for b in DirectionalBias}
    components = signal["components"]
    assert components  # non-empty
    for c in components:
        assert set(c) == COMPONENT_KEYS
        assert c["side"] in {"bull", "bear"}
        assert isinstance(c["triggered"], bool)
        assert isinstance(c["detail"], str) and c["detail"]

    series = body["series"]
    assert set(series) == {"dates", "close", "sma20", "sma50"}
    lengths = {len(series[k]) for k in ("dates", "close", "sma20", "sma50")}
    assert len(lengths) == 1  # aligned arrays
    assert lengths.pop() <= 250
    for d in series["dates"]:
        date.fromisoformat(d)
    assert all(isinstance(v, float) for v in series["close"])


async def test_backfill_happens_once_and_is_system_audited(client):
    """Lazy backfill runs exactly once, SYSTEM-attributed (rule 12, ADR-003)."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})

    r1 = await client.get("/api/watchlist/NVDA/analysis")
    r2 = await client.get("/api/watchlist/NVDA/analysis")
    assert r1.status_code == r2.status_code == 200
    # Stored bar count unchanged between consecutive GETs: no double backfill.
    assert r1.json()["bars"]["count"] == r2.json()["bars"]["count"]

    r = await client.get("/api/audit", params={"entity_id": "NVDA"})
    backfills = [e for e in r.json() if e["action"] == "DATA_BACKFILL"]
    assert len(backfills) == 1  # exactly one, despite two GETs
    event = backfills[0]
    assert event["actor_type"] == "SYSTEM"
    details = event["details"]
    assert details["bars"] == r1.json()["bars"]["count"]
    assert details["provider"] == "stub"
    assert details["first"] == r1.json()["bars"]["first"]
    assert details["last"] == r1.json()["bars"]["last"]
