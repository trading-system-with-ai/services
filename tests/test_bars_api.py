"""Tests for GET /api/watchlist/{ticker}/bars (plan §33, Price tab)."""
from datetime import date

BAR_KEYS = {"date", "open", "high", "low", "close", "volume"}


async def test_bars_open_for_non_watchlist_ticker(client):
    """2026-08-20 (§4.2 amended): bars serve any ticker (lazy backfill)."""
    r = await client.get("/api/watchlist/NVDA/bars")
    assert r.status_code == 200
    assert len(r.json()["bars"]) > 0


async def test_bars_contract_oldest_first_and_ohlc_sanity(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.get("/api/watchlist/NVDA/bars")
    assert r.status_code == 200
    body = r.json()

    assert set(body) == {"ticker", "source", "bars"}
    assert body["ticker"] == "NVDA"
    assert body["source"] == "stub"

    bars = body["bars"]
    assert len(bars) == 250  # default limit
    dates = [date.fromisoformat(b["date"]) for b in bars]
    assert dates == sorted(dates)  # oldest first
    assert len(set(dates)) == len(dates)  # no duplicate days

    for b in bars:
        assert set(b) == BAR_KEYS
        for key in ("open", "high", "low", "close", "volume"):
            assert isinstance(b[key], float), f"{key} should be a float"
        # OHLC sanity: the range must contain both open and close.
        assert b["low"] <= b["open"] <= b["high"]
        assert b["low"] <= b["close"] <= b["high"]
        assert b["volume"] > 0


async def test_bars_limit_respected_and_validated(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})

    r = await client.get("/api/watchlist/NVDA/bars", params={"limit": 60})
    assert r.status_code == 200
    sixty = r.json()["bars"]
    assert len(sixty) == 60

    r = await client.get("/api/watchlist/NVDA/bars", params={"limit": 250})
    assert len(r.json()["bars"]) == 250
    # The 60-bar view is the tail of the 250-bar view: most recent bars, oldest first.
    assert r.json()["bars"][-60:] == sixty

    # Bounds are enforced, not clamped: 10 <= limit <= 600.
    r = await client.get("/api/watchlist/NVDA/bars", params={"limit": 5})
    assert r.status_code == 422
    r = await client.get("/api/watchlist/NVDA/bars", params={"limit": 1000})
    assert r.status_code == 422


async def test_bars_second_call_triggers_no_second_backfill(client):
    """Lazy backfill runs exactly once (rule 12, ADR-003)."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})

    r1 = await client.get("/api/watchlist/NVDA/bars")
    r2 = await client.get("/api/watchlist/NVDA/bars")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()  # stored bars, not re-fetched ones

    r = await client.get("/api/audit", params={"entity_id": "NVDA"})
    backfills = [e for e in r.json() if e["action"] == "DATA_BACKFILL"]
    assert len(backfills) == 1  # exactly one, despite two GETs
    assert backfills[0]["actor_type"] == "SYSTEM"
