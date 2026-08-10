"""Tests for the backtest API (plan §20): POST /api/backtests, the list and
detail GETs, the Watchlist-only gate (§4.2), the param 422 paths, and the
same-transaction audit trail (rule 12)."""
import math

CONTRACT_KEYS = {
    "id",
    "ticker",
    "created_at",
    "status",
    "params",
    "error",
    "oos_start_date",
    "metrics",
    "trades",
    "equity_curve",
}
PARAM_KEYS = {
    "position_pct",
    "commission_per_share",
    "slippage_bps",
    "entry_edge_threshold",
    "exit_edge_threshold",
    "atr_trail_k",
    "time_stop_bars",
    "min_move_atr",
    "oos_split",
    "warmup_bars",
}
SEG_KEYS = {
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "win_rate",
    "profit_factor",
    "expectancy_pct",
    "avg_trade_pct",
    "avg_hold_bars",
    "num_trades",
    "exposure_pct",
}
TRADE_KEYS = {
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "bars_held",
    "return_pct",
    "entry_reason",
    "exit_reason",
}
SUMMARY_KEYS = {
    "id",
    "ticker",
    "created_at",
    "status",
    "num_trades",
    "total_return_pct",
    "profit_factor",
    "oos_start_date",
}


def walk_assert_finite(obj, path="$") -> None:
    """Every float anywhere in the body must be finite — the contract allows
    null for undefined metrics, never NaN/Infinity (plan §44 rule 18)."""
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float at {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk_assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_assert_finite(v, f"{path}[{i}]")


async def run_backtest_for(client, ticker: str, params: dict | None = None):
    """Watchlist the ticker (idempotent for tests) and POST a backtest."""
    await client.post("/api/watchlist", json={"ticker": ticker})
    return await client.post(
        "/api/backtests", json={"ticker": ticker, "params": params or {}}
    )


async def test_post_404_for_non_watchlist_ticker(client):
    """Only Watchlist symbols may be backtested (plan §20, §4.2)."""
    r = await client.post("/api/backtests", json={"ticker": "NVDA"})
    assert r.status_code == 404
    assert "not on the watchlist" in r.json()["detail"]


async def test_post_completed_record_contract(client):
    r = await run_backtest_for(client, "NVDA", {"warmup_bars": 100})
    assert r.status_code == 200
    body = r.json()  # parsing succeeded => Starlette emitted strict JSON (no NaN)

    assert set(body) == CONTRACT_KEYS
    assert isinstance(body["id"], int)
    assert body["ticker"] == "NVDA"
    assert body["status"] == "COMPLETED"
    assert body["error"] is None

    # Resolved params: full BacktestParams shape, override applied.
    assert set(body["params"]) == PARAM_KEYS
    assert body["params"]["warmup_bars"] == 100
    assert body["params"]["oos_split"] == 0.7  # default preserved

    # Metrics: all three segments, each with the exact SEG keys.
    assert set(body["metrics"]) == {"full", "in_sample", "out_of_sample"}
    for seg in body["metrics"].values():
        assert set(seg) == SEG_KEYS
        assert isinstance(seg["num_trades"], int)

    # OOS boundary is reported (600 stored bars * 0.7 lands inside the series).
    assert body["oos_start_date"] is not None

    # Trades: contract keys only. (entry_reason is only asserted to be a str:
    # engine V1 currently drops the entry reason on fill — engine-side issue,
    # not part of this API's file set.)
    for trade in body["trades"]:
        assert set(trade) == TRADE_KEYS
        assert isinstance(trade["entry_reason"], str)
        assert isinstance(trade["exit_reason"], str) and trade["exit_reason"]
    assert body["metrics"]["full"]["num_trades"] == len(body["trades"])

    # Equity curve: aligned per-bar arrays over the whole series.
    curve = body["equity_curve"]
    assert set(curve) == {"dates", "equity", "drawdown"}
    assert len(curve["dates"]) == len(curve["equity"]) == len(curve["drawdown"])
    assert len(curve["dates"]) >= 250  # full stored history, not a tail

    walk_assert_finite(body)


async def test_post_unknown_param_key_422(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post(
        "/api/backtests", json={"ticker": "NVDA", "params": {"bogus_knob": 1}}
    )
    assert r.status_code == 422
    assert "bogus_knob" in r.json()["detail"]


async def test_post_invalid_param_value_422_surfaces_engine_message(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post(
        "/api/backtests", json={"ticker": "NVDA", "params": {"oos_split": 1.5}}
    )
    assert r.status_code == 422
    # The engine's own ValueError message is surfaced verbatim (plan §6.2).
    assert "oos_split must be in (0, 1)" in r.json()["detail"]


async def test_audit_trail_started_and_completed(client):
    """USER BACKTEST_STARTED + SYSTEM BACKTEST_COMPLETED, both present (rule 12)."""
    r = await run_backtest_for(client, "NVDA")
    body = r.json()

    events = (await client.get("/api/audit", params={"entity_id": "NVDA"})).json()
    started = [e for e in events if e["action"] == "BACKTEST_STARTED"]
    completed = [e for e in events if e["action"] == "BACKTEST_COMPLETED"]
    assert len(started) == 1 and len(completed) == 1

    assert started[0]["actor_type"] == "USER"
    assert set(started[0]["details"]["params"]) == PARAM_KEYS

    assert completed[0]["actor_type"] == "SYSTEM"
    details = completed[0]["details"]
    assert details["backtest_id"] == body["id"]
    assert details["num_trades"] == body["metrics"]["full"]["num_trades"]
    assert details["profit_factor"] == body["metrics"]["full"]["profit_factor"]
    assert (
        details["oos_total_return_pct"]
        == body["metrics"]["out_of_sample"]["total_return_pct"]
    )
    assert not [e for e in events if e["action"] == "BACKTEST_FAILED"]


async def test_list_newest_first_summaries(client):
    id_nvda = (await run_backtest_for(client, "NVDA")).json()["id"]
    id_aapl = (await run_backtest_for(client, "AAPL")).json()["id"]
    assert id_aapl > id_nvda

    r = await client.get("/api/backtests")
    assert r.status_code == 200
    rows = r.json()
    assert [row["id"] for row in rows] == [id_aapl, id_nvda]  # newest first
    for row in rows:
        assert set(row) == SUMMARY_KEYS
        assert row["status"] == "COMPLETED"
        assert isinstance(row["num_trades"], int)

    # Ticker filter.
    filtered = (await client.get("/api/backtests", params={"ticker": "aapl"})).json()
    assert [row["id"] for row in filtered] == [id_aapl]
    assert filtered[0]["ticker"] == "AAPL"


async def test_get_by_id_roundtrip_and_404(client):
    posted = (await run_backtest_for(client, "NVDA")).json()

    r = await client.get(f"/api/backtests/{posted['id']}")
    assert r.status_code == 200
    assert r.json() == posted  # stored record echoes the POST response exactly

    r = await client.get("/api/backtests/999999")
    assert r.status_code == 404
