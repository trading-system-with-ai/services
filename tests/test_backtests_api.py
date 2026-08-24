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
    "warmup_bars",
    "fill_model",
    "worst_slippage_bps",
    "instrument",
    "target_dte_min",
    "target_dte_max",
    "strike_otm_pct",
    "option_premium_pct",
    "commission_per_contract",
    "option_slippage_bps",
    "worst_option_slippage_bps",
    "spread_width_pct",
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
    "fill_model",
    "instrument",
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

    # Metrics: ONE flat full-period object (IS/OOS segmentation removed
    # 2026-08-16 — manual-only tuning until ML-driven search exists).
    assert set(body["metrics"]) == SEG_KEYS
    assert isinstance(body["metrics"]["num_trades"], int)

    # Trades: contract keys only. (entry_reason is only asserted to be a str:
    # engine V1 currently drops the entry reason on fill — engine-side issue,
    # not part of this API's file set.)
    for trade in body["trades"]:
        assert set(trade) == TRADE_KEYS
        assert isinstance(trade["entry_reason"], str)
        assert isinstance(trade["exit_reason"], str) and trade["exit_reason"]
    assert body["metrics"]["num_trades"] == len(body["trades"])

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
        "/api/backtests", json={"ticker": "NVDA", "params": {"atr_trail_k": 0.0}}
    )
    assert r.status_code == 422
    # The engine's own ValueError message is surfaced verbatim (plan §6.2).
    assert "atr_trail_k must be > 0" in r.json()["detail"]


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
    assert details["num_trades"] == body["metrics"]["num_trades"]
    assert details["profit_factor"] == body["metrics"]["profit_factor"]
    assert details["total_return_pct"] == body["metrics"]["total_return_pct"]
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


# ---------------------------------------------------------------------------
# Fill-model variants (plan §20.2): round-trip, summaries, defaults, 422s.
# ---------------------------------------------------------------------------


async def test_fill_model_round_trips_through_post_and_get(client):
    """POST with fill_model WORST -> the stored resolved params echo it, the
    detail GET returns it, and the list summary chips it (plan §20.2)."""
    posted = (
        await run_backtest_for(client, "NVDA", {"fill_model": "WORST"})
    ).json()
    assert posted["status"] == "COMPLETED"
    assert posted["params"]["fill_model"] == "WORST"
    assert posted["params"]["worst_slippage_bps"] == 25.0  # default preserved

    detail = (await client.get(f"/api/backtests/{posted['id']}")).json()
    assert detail["params"]["fill_model"] == "WORST"

    rows = (await client.get("/api/backtests")).json()
    row = next(r for r in rows if r["id"] == posted["id"])
    assert set(row) == SUMMARY_KEYS
    assert row["fill_model"] == "WORST"


async def test_fill_model_defaults_to_conservative(client):
    """A run posted without fill-model params resolves to CONSERVATIVE — the
    pre-change engine behavior (plan §20.2) — in both the stored params and
    the list summary."""
    posted = (await run_backtest_for(client, "NVDA")).json()
    assert set(posted["params"]) == PARAM_KEYS
    assert posted["params"]["fill_model"] == "CONSERVATIVE"
    assert posted["params"]["worst_slippage_bps"] == 25.0

    rows = (await client.get("/api/backtests")).json()
    assert rows[0]["fill_model"] == "CONSERVATIVE"


async def test_fill_model_worst_never_beats_conservative_via_api(client):
    """§20.2 monotonicity holds through the API: the same symbol backtested
    under WORST cannot out-return the CONSERVATIVE default run."""
    conservative = (await run_backtest_for(client, "NVDA")).json()
    worst = (
        await run_backtest_for(client, "NVDA", {"fill_model": "WORST"})
    ).json()
    assert conservative["status"] == worst["status"] == "COMPLETED"
    assert (
        worst["metrics"]["total_return_pct"]
        <= conservative["metrics"]["total_return_pct"]
    )


async def test_fill_model_invalid_values_422_with_engine_message(client):
    """Bad fill_model / negative worst_slippage_bps 422 with the engine's own
    ValueError message (plan §6.2), before any state or audit write."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})

    r = await client.post(
        "/api/backtests",
        json={"ticker": "NVDA", "params": {"fill_model": "MIDPOINT"}},
    )
    assert r.status_code == 422
    assert "fill_model must be one of" in r.json()["detail"]

    r = await client.post(
        "/api/backtests",
        json={"ticker": "NVDA", "params": {"worst_slippage_bps": -1.0}},
    )
    assert r.status_code == 422
    assert "worst_slippage_bps must be >= 0" in r.json()["detail"]

    # 422s happen before any record is written.
    assert (await client.get("/api/backtests")).json() == []


# ---------------------------------------------------------------------------
# LONG_CALL leg (user mandate 2026-08-17): options join the backtest over the
# provider's contract/bars history (stub: deterministic synthetic universe).
# ---------------------------------------------------------------------------


async def test_long_call_backtest_end_to_end(client):
    r = await run_backtest_for(client, "NVDA", {"instrument": "LONG_CALL"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["params"]["instrument"] == "LONG_CALL"
    assert set(body["metrics"]) == SEG_KEYS
    # Option trades carry contract identity; premiums are per share.
    for tr in body["trades"]:
        assert tr["contract_symbol"]
        assert tr["strike"] > 0
        assert tr["contracts"] >= 1
        assert tr["contract_expiry"] is not None
        assert tr["entry_reason"].startswith("edge ")
        assert "LONG_CALL" in tr["entry_reason"]
    walk_assert_finite(body)

    # History summary chips the instrument.
    rows = (await client.get("/api/backtests")).json()
    row = next(x for x in rows if x["id"] == body["id"])
    assert row["instrument"] == "LONG_CALL"


async def test_long_call_backtest_respects_permissions(client):
    """allow_long_call=false refuses the LONG_CALL leg (and does NOT touch
    the stock leg); allow_long_stock=false refuses stock but NOT calls —
    backtest and live gating read the same permissions factory."""
    await client.put("/api/config/providers", json={"allow_long_call": "false"})
    r = await run_backtest_for(client, "NVDA", {"instrument": "LONG_CALL"})
    assert r.status_code == 422
    assert "allow_long_call=false" in r.json()["detail"]
    r = await run_backtest_for(client, "NVDA", {})
    assert r.status_code == 200  # stock unaffected

    await client.put(
        "/api/config/providers",
        json={"allow_long_call": "true", "allow_long_stock": "false"},
    )
    r = await run_backtest_for(client, "NVDA", {"instrument": "LONG_CALL"})
    assert r.status_code == 200  # calls unaffected by the stock flag
    r = await run_backtest_for(client, "NVDA", {})
    assert r.status_code == 422
    await client.put("/api/config/providers", json={"allow_long_stock": "true"})


async def test_long_call_option_param_validation(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post(
        "/api/backtests",
        json={
            "ticker": "NVDA",
            "params": {"instrument": "LONG_CALL", "option_premium_pct": 0.0},
        },
    )
    assert r.status_code == 422
    assert "option_premium_pct" in r.json()["detail"]


async def test_bull_call_spread_backtest_end_to_end(client):
    """Spread leg via the API over the stub's deterministic grid, gated by
    allow_defined_risk_spreads (research+backtest scope, roadmap Phase 1)."""
    # Gate: off by default -> honest 422 naming the flag and the scope.
    r = await run_backtest_for(client, "NVDA", {"instrument": "BULL_CALL_SPREAD"})
    assert r.status_code == 422
    assert "allow_defined_risk_spreads=false" in r.json()["detail"]

    # Enable -> runs end to end.
    r = await client.put(
        "/api/config/providers", json={"allow_defined_risk_spreads": "true"}
    )
    assert r.status_code == 200
    r = await run_backtest_for(client, "NVDA", {"instrument": "BULL_CALL_SPREAD"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["params"]["instrument"] == "BULL_CALL_SPREAD"
    for tr in body["trades"]:
        # Both legs identified; net debit strictly inside (0, width).
        assert tr["contract_symbol"] and tr["short_symbol"]
        width = tr["short_strike"] - tr["strike"]
        assert width > 0
        assert 0 < tr["entry_price"] < width
        assert "BULL_CALL_SPREAD" in tr["entry_reason"]
    walk_assert_finite(body)

    rows = (await client.get("/api/backtests")).json()
    row = next(x for x in rows if x["id"] == body["id"])
    assert row["instrument"] == "BULL_CALL_SPREAD"

    # Disable again -> gate closes.
    await client.put(
        "/api/config/providers", json={"allow_defined_risk_spreads": "false"}
    )
    r = await run_backtest_for(client, "NVDA", {"instrument": "BULL_CALL_SPREAD"})
    assert r.status_code == 422


async def test_long_put_and_bear_put_spread_backtests_end_to_end(client):
    """Bear-side mirror (2026-08-17): PLTR reads BEAR at the anchored stub
    date, so the put legs produce real replays over the stub's put grid;
    permission gates mirror the call legs."""
    r = await run_backtest_for(client, "PLTR", {"instrument": "LONG_PUT"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["params"]["instrument"] == "LONG_PUT"
    # Non-vacuous: PLTR's synthetic path carries BEAR stretches, so the put
    # leg must actually replay trades (audit §8 item 6 made this explicit).
    assert body["trades"], "LONG_PUT on PLTR must produce trades"
    for tr in body["trades"]:
        assert "LONG_PUT" in tr["entry_reason"]
        assert "bias BEAR" in tr["entry_reason"]
    walk_assert_finite(body)

    await client.put(
        "/api/config/providers", json={"allow_defined_risk_spreads": "true"}
    )
    r = await run_backtest_for(client, "PLTR", {"instrument": "BEAR_PUT_SPREAD"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["params"]["instrument"] == "BEAR_PUT_SPREAD"
    # Audit §8 item 6: this used to pass on an EMPTY trade list because the
    # engine ran the bull evaluator and demanded short > long while the
    # resolver returns short < long. Now the resolver's put-vertical geometry
    # and the engine's bear gate agree, so the same fixture that opens
    # LONG_PUT trades must open bear put spreads too.
    assert body["trades"], "BEAR_PUT_SPREAD on PLTR must produce trades"
    for tr in body["trades"]:
        # Put vertical: the short strike sits BELOW the long strike.
        assert tr["short_strike"] < tr["strike"]
        width = tr["strike"] - tr["short_strike"]
        assert 0 < tr["entry_price"] < width
        assert tr["contract_symbol"] and tr["short_symbol"]
        assert "BEAR_PUT_SPREAD" in tr["entry_reason"]
        assert "bias BEAR" in tr["entry_reason"]
    walk_assert_finite(body)

    # Permission mirror: puts off -> LONG_PUT refused, calls unaffected.
    await client.put("/api/config/providers", json={"allow_long_put": "false"})
    r = await run_backtest_for(client, "PLTR", {"instrument": "LONG_PUT"})
    assert r.status_code == 422
    assert "allow_long_put=false" in r.json()["detail"]
    await client.put("/api/config/providers", json={"allow_long_put": "true"})


async def test_income_backtests_end_to_end_with_permission_gates(client, monkeypatch):
    """COVERED_CALL / CASH_SECURED_PUT backtests over the stub grid, gated
    by their (still locked) permissions — tests drive the gate through the
    permissions factory seam."""
    # Locked by default -> honest 422 naming Phase 2.
    r = await run_backtest_for(client, "NVDA", {"instrument": "COVERED_CALL"})
    assert r.status_code == 422
    assert "Phase 2" in r.json()["detail"]

    # Unlock via the factory seam (the §33 set shrinks when Phase 2 lands).
    import dataclasses as _dc

    from apps.gateway.routers import backtests as bt_router

    real_factory = bt_router.account_permissions_from_settings

    class _Perms:
        def __getattr__(self, name):
            if name in ("covered_call", "cash_secured_put"):
                return True
            return getattr(real_factory(), name)

    monkeypatch.setattr(
        bt_router, "account_permissions_from_settings", lambda: _Perms()
    )

    for instrument in ("COVERED_CALL", "CASH_SECURED_PUT"):
        r = await run_backtest_for(client, "NVDA", {"instrument": instrument})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "COMPLETED"
        assert body["params"]["instrument"] == instrument
        walk_assert_finite(body)
        rows = (await client.get("/api/backtests")).json()
        row = next(x for x in rows if x["id"] == body["id"])
        assert row["instrument"] == instrument


async def test_short_stock_backtest_end_to_end_with_permission_gate(client):
    """SHORT_STOCK backtest (roadmap Phase 3): locked by default with BOTH
    flags named; unlocked through the REAL runtime-config path (the same
    toggles the Settings UI drives) it replays the bear mirror engine."""
    r = await run_backtest_for(client, "PLTR", {"instrument": "SHORT_STOCK"})
    assert r.status_code == 422
    assert "allow_short_stock" in r.json()["detail"]
    assert "allow_margin" in r.json()["detail"]

    # One flag alone is not enough — margin exists to support shorting.
    await client.put("/api/config/providers", json={"allow_short_stock": "true"})
    r = await run_backtest_for(client, "PLTR", {"instrument": "SHORT_STOCK"})
    assert r.status_code == 422

    await client.put("/api/config/providers", json={"allow_margin": "true"})
    r = await run_backtest_for(client, "PLTR", {"instrument": "SHORT_STOCK"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["params"]["instrument"] == "SHORT_STOCK"
    walk_assert_finite(body)
    # Mirrored arithmetic: a short's return is positive exactly when the
    # cover printed BELOW the entry (commissions cost a few bps of edge, so
    # only clear moves are sign-checked).
    for t in body["trades"]:
        move_pct = (t["entry_price"] - t["exit_price"]) / t["entry_price"]
        if move_pct > 0.005:
            assert t["return_pct"] > 0.0, t
        elif move_pct < -0.005:
            assert t["return_pct"] < 0.0, t
    rows = (await client.get("/api/backtests")).json()
    row = next(x for x in rows if x["id"] == body["id"])
    assert row["instrument"] == "SHORT_STOCK"


# ---------------------------------------------------------------- AUTO (Phase B)


async def test_auto_backtest_runs_and_stores_decision_trail(client):
    """AUTO: the §8 matrix picks the instrument daily; the record carries
    the auto_decisions audit trail (docs/auto-strategy-portfolio-design.md)."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post(
        "/api/backtests",
        json={"ticker": "NVDA", "params": {"instrument": "AUTO"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["params"]["instrument"] == "AUTO"
    assert "auto_decisions" in body["metrics"]
    for d in body["metrics"]["auto_decisions"]:
        assert set(d) == {"date", "edge", "tier", "vol_regime", "instrument", "rationale"}


async def test_auto_multiselect_restricts_but_never_exceeds(client):
    """The instruments multi-select may RESTRICT the account permissions,
    never exceed them; unsupported selections 422 by name."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    # restrict to stock only -> run completes; no option trade rows possible
    r = await client.post(
        "/api/backtests",
        json={
            "ticker": "NVDA",
            "params": {"instrument": "AUTO"},
            "instruments": ["LONG_STOCK"],
        },
    )
    assert r.status_code == 200, r.text
    assert all("contract_symbol" not in t for t in r.json()["trades"])

    # spreads under AUTO -> refused by name (Phase D)
    r = await client.post(
        "/api/backtests",
        json={
            "ticker": "NVDA",
            "params": {"instrument": "AUTO"},
            "instruments": ["BULL_CALL_SPREAD"],
        },
    )
    assert r.status_code == 422
    assert "Phase D" in r.json()["detail"]

    # SHORT_STOCK selected while margin is off (default) -> 422, never silent
    r = await client.post(
        "/api/backtests",
        json={
            "ticker": "NVDA",
            "params": {"instrument": "AUTO"},
            "instruments": ["SHORT_STOCK"],
        },
    )
    assert r.status_code == 422
    assert "allow_short_stock" in r.json()["detail"]

    # instruments without AUTO -> 422
    r = await client.post(
        "/api/backtests",
        json={
            "ticker": "NVDA",
            "params": {"instrument": "LONG_STOCK"},
            "instruments": ["LONG_STOCK"],
        },
    )
    assert r.status_code == 422
    assert "only valid with instrument='AUTO'" in r.json()["detail"]


async def test_auto_still_watchlist_gated(client):
    """AUTO obeys the one member-only surface: backtests (§4.2 amended)."""
    r = await client.post(
        "/api/backtests",
        json={"ticker": "ZZZQ", "params": {"instrument": "AUTO"}},
    )
    assert r.status_code == 404
    assert "not on the watchlist" in r.json()["detail"]


# ------------------------------------------------------------- portfolio (C2)


async def test_portfolio_backtest_whole_watchlist(client):
    """POST /api/backtests/portfolio replays every watchlist symbol against
    one shared ledger and stores the per-day allocation table."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    await client.post("/api/watchlist", json={"ticker": "MSFT"})
    r = await client.post("/api/backtests/portfolio", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert sorted(body["tickers"]) == ["MSFT", "NVDA"]
    alloc = body["allocations"]
    assert set(alloc) == {"dates", "by_symbol", "cash_pct"}
    n = len(alloc["dates"])
    assert n == len(alloc["by_symbol"]) == len(alloc["cash_pct"]) >= 2
    # accounting identity on every bar
    for t in range(n):
        total = alloc["cash_pct"][t] + sum(alloc["by_symbol"][t].values())
        assert abs(total - 100.0) < 1e-6
    for d in body["decisions"]:
        assert d["ticker"] in ("NVDA", "MSFT")
    # round-trips through GET
    r = await client.get(f"/api/backtests/portfolio/{body['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == body["id"]
    r = await client.get("/api/backtests/portfolio")
    assert any(row["id"] == body["id"] for row in r.json())


async def test_portfolio_subset_and_membership_gate(client):
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    # subset with a non-member -> 404 naming it
    r = await client.post(
        "/api/backtests/portfolio", json={"tickers": ["NVDA", "ZZZQ"]}
    )
    assert r.status_code == 404
    assert "ZZZQ" in r.json()["detail"]
    # explicit instrument other than AUTO -> 422
    r = await client.post(
        "/api/backtests/portfolio",
        json={"tickers": ["NVDA"], "params": {"instrument": "LONG_STOCK"}},
    )
    assert r.status_code == 422
    assert "AUTO" in r.json()["detail"]


async def test_portfolio_dedupes_and_normalizes_tickers(client):
    """Verifier catch: duplicate slots against one ledger broke the
    allocation identity. Dupes/whitespace/case normalize to one slot."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post(
        "/api/backtests/portfolio", json={"tickers": ["NVDA", " nvda ", "NVDA"]}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tickers"] == ["NVDA"]
    alloc = body["allocations"]
    for t in range(len(alloc["dates"])):
        total = alloc["cash_pct"][t] + sum(alloc["by_symbol"][t].values())
        assert abs(total - 100.0) < 1e-6
    r = await client.post("/api/backtests/portfolio", json={"tickers": ["BAD TICKER!"]})
    assert r.status_code == 422


async def test_portfolio_journal_and_advice_are_stored(client):
    """Explainability mandate 2026-08-20: the record carries the rebalance
    journal (ENTER events with the full sizing arithmetic) and the
    risk-model advice (each item evidenced with a rationale)."""
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post("/api/backtests/portfolio", json={"tickers": ["NVDA"]})
    assert r.status_code == 200, r.text
    body = r.json()
    journal = body["journal"]
    trades = body["trades"]
    enters = [e for e in journal if e["action"] == "ENTER"]
    exits = [e for e in journal if e["action"] == "EXIT"]
    assert len(enters) == len(trades) == len(exits)
    for e in enters:
        assert "budget" in e["sizing"] and "÷ stop" in e["sizing"] or "premium budget" in e["sizing"]
        assert e["quantity"] > 0 and e["price"] is not None
    for e in exits:
        assert e["reason"]  # the exit engine's rule verbatim
    advice = body["advice"]
    assert advice, "a completed replay always yields at least the tail item"
    for a in advice:
        assert a["severity"] in ("INFO", "SUGGESTION", "WARNING")
        assert a["finding"] and a["rationale"] and isinstance(a["evidence"], dict)
