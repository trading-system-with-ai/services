"""Tests for GET /api/watchlist/overview (plan §31): one dashboard row per
Watchlist symbol with regime/directional read, v0 opportunity status, and the
latest backtest outcome."""
import pytest

from libs.trading_core.models import DirectionalBias, MarketRegime, OpportunityStatus

ROW_KEYS = {
    "ticker",
    "price",
    "regime",
    "bull_score",
    "bear_score",
    "directional_edge",
    "bias",
    "opportunity_status",
    "backtest_status",
    "last_backtest_id",
}


async def test_empty_watchlist_gives_empty_overview(client):
    r = await client.get("/api/watchlist/overview")
    assert r.status_code == 200
    assert r.json() == []


async def test_overview_rows_for_two_tickers(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    await client.post("/api/watchlist", json={"ticker": "AAPL"})

    r = await client.get("/api/watchlist/overview")
    assert r.status_code == 200
    rows = r.json()
    assert [row["ticker"] for row in rows] == ["AAPL", "NVDA"]  # ticker-ordered

    for row in rows:
        assert set(row) == ROW_KEYS
        assert isinstance(row["price"], float) and row["price"] > 0
        assert row["regime"] in {m.value for m in MarketRegime}
        assert 0.0 <= row["bull_score"] <= 100.0
        assert 0.0 <= row["bear_score"] <= 100.0
        assert row["directional_edge"] == pytest.approx(
            row["bull_score"] - row["bear_score"]
        )
        assert row["bias"] in {b.value for b in DirectionalBias}
        assert row["opportunity_status"] in {s.value for s in OpportunityStatus}
        # 600 backfilled bars cover the 200-bar slow SMA: never DATA_ISSUE here.
        assert row["opportunity_status"] != "DATA_ISSUE"
        # No backtest has been run yet.
        assert row["backtest_status"] == "NONE"
        assert row["last_backtest_id"] is None


async def test_backtest_status_flips_to_completed_after_a_run(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    await client.post("/api/watchlist", json={"ticker": "AAPL"})

    posted = await client.post("/api/backtests", json={"ticker": "NVDA"})
    assert posted.status_code == 200
    backtest_id = posted.json()["id"]

    rows = (await client.get("/api/watchlist/overview")).json()
    by_ticker = {row["ticker"]: row for row in rows}

    assert by_ticker["NVDA"]["backtest_status"] == "COMPLETED"
    assert by_ticker["NVDA"]["last_backtest_id"] == backtest_id
    # The other symbol is untouched.
    assert by_ticker["AAPL"]["backtest_status"] == "NONE"
    assert by_ticker["AAPL"]["last_backtest_id"] is None
