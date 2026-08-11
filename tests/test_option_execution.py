"""Option execution tests — preview / approve / positions / close /
check-exits for long options (plan §8, §9, §11.3, §11.7, §12.1).

Determinism (same style as the other execution tests): the stub provider is
seeded by the symbol name, so both the bar path and the option chain are
reproducible. ``CALL_TICKER`` ("GW") deterministically yields a BULL bias in
a bull regime with a VERY_STRONG edge and a LOW §7 vol regime -> the §8
matrix picks LONG_CALL with an eligible §9 candidate — a natural BULL/LOW
combination, so no threshold seams need overriding here (the seams
``orders.VOL_REGIME_PARAMS`` / ``orders.SELECTOR_PARAMS`` exist for when the
stub stops cooperating). GOOGL (BULL, NORMAL vol — the stock tests' ticker)
exercises the BEAR direction override. Forced exits drive PREMIUM_HARD_STOP
/ DTE_EXIT purely through direct UPDATEs of the stored Position row, so no
exit depends on signal luck.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from apps.gateway.db import Order, Position, SessionLocal
from libs.common.config import get_settings
from libs.trading_core.risk import RiskLimits

# Deterministic stub verdicts (see module docstring):
# GW -> MILD_BULL regime, BULL bias, |edge| 88.9 (VERY_STRONG), vol LOW
#       -> LONG_CALL with a rank-1 §9 candidate.
CALL_TICKER = "GW"
# GOOGL -> MILD_BULL regime, BULL bias, vol NORMAL: the stock ticker; with
# direction=BEAR override the §8 BEAR/STRONG/NORMAL cell degrades the Bear
# Put Spread to LONG_PUT (§5: no spreads).
STOCK_TICKER = "GOOGL"

TIER_BUDGETS = {
    "WEAK": RiskLimits().budget_weak,
    "MODERATE": RiskLimits().budget_moderate,
    "STRONG": RiskLimits().budget_strong,
    "VERY_STRONG": RiskLimits().budget_very_strong,
}

OPTION_EXIT_RULES = {
    "PREMIUM_HARD_STOP",
    "DTE_EXIT",
    "SIGNAL_FLIP",
    "SIGNAL_DECAY",
    "ATR_TRAIL",
    "TIME_STOP",
}


async def authorize(client, ticker):
    """Watchlist -> Trading Pool -> per-symbol enable -> global resume."""
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    # acknowledge_risks: the ticker has no stored history/backtest at promote
    # time, so the §4.3 promotion checks fail and need an explicit override.
    r = await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(f"/api/trading-pool/{ticker}/trading", json={"enabled": True})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200


async def preview(client, ticker, direction=None, quantity=None):
    body = {"ticker": ticker}
    if direction is not None:
        body["direction"] = direction
    if quantity is not None:
        body["quantity"] = quantity
    r = await client.post("/api/orders/preview", json=body)
    assert r.status_code == 200, r.text
    return r.json()


async def approve(client, ticker, direction=None, quantity=None):
    body = {"ticker": ticker}
    if direction is not None:
        body["direction"] = direction
    if quantity is not None:
        body["quantity"] = quantity
    return await client.post("/api/orders/approve", json=body)


async def cash(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    return r.json()["cash"]


async def db_orders():
    async with SessionLocal() as s:
        return list((await s.execute(select(Order).order_by(Order.id))).scalars().all())


async def db_position(position_id):
    async with SessionLocal() as s:
        return await s.get(Position, position_id)


def gate(body, name):
    return next(g for g in body["gates"] if g["name"] == name)


# ---------------------------------------------------------------------------
# Preview: §8 matrix + §9 contract + §12.1 contract sizing
# ---------------------------------------------------------------------------


async def test_preview_bull_low_vol_selects_long_call_with_contract_sizing(client):
    """BULL + LOW vol + strong edge -> LONG_CALL with the §9 top candidate
    proposed and §12.1 CONTRACT sizing: entry = stop = mid*100 (premium fully
    at risk), so approved_quantity counts contracts within the re-derived
    tier budget."""
    settings = get_settings()
    await authorize(client, CALL_TICKER)
    body = await preview(client, CALL_TICKER)

    proposed = body["proposed"]
    assert proposed["instrument"] == "LONG_CALL", (
        "stub no longer yields BULL/LOW for GW — re-pick the ticker or drive "
        "the vol seam (orders.VOL_REGIME_PARAMS); NEVER skip the sizing "
        f"assertions. Preview: {proposed}"
    )
    assert proposed["vol_regime"] == "LOW"
    assert proposed["instrument_rationale"]  # §8 cell + degradations spelled out

    # Gate semantics: real VOLATILITY classification, §8 INSTRUMENT verdict,
    # §9 CONTRACT_SELECTION with the top-ranked candidate.
    assert gate(body, "VOLATILITY")["status"] == "PASS"
    assert "LOW" in gate(body, "VOLATILITY")["detail"]
    assert gate(body, "INSTRUMENT")["status"] == "PASS"
    assert "§8" in gate(body, "INSTRUMENT")["detail"]
    assert gate(body, "CONTRACT_SELECTION")["status"] == "PASS"
    assert "top-ranked" in gate(body, "CONTRACT_SELECTION")["detail"]

    contract = proposed["contract"]
    assert contract is not None
    assert set(contract) == {
        "expiry",
        "dte",
        "strike",
        "right",
        "mid",
        "delta",
        "iv",
        "multiplier",
        "max_loss_per_contract",
    }
    assert contract["right"] == "C"
    assert contract["multiplier"] == 100
    assert contract["mid"] > 0
    assert contract["max_loss_per_contract"] == pytest.approx(contract["mid"] * 100)
    # §9.1 defaults: the candidate sits inside the DTE window.
    assert 30 <= contract["dte"] <= 90

    # §12.1: contract-level risk units — entry and stop are BOTH mid*100.
    assert proposed["entry_price"] == pytest.approx(contract["mid"] * 100)
    assert proposed["stop_distance"] == pytest.approx(contract["mid"] * 100)

    # Re-derive the §12.2 tier budget and check the §12.1 contract count:
    # fresh portfolio -> nav == paper_initial_cash, no open positions.
    risk = body["risk"]
    assert risk is not None
    assert risk["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE"}
    nav = settings.paper_initial_cash
    limits = RiskLimits()
    budget = min(TIER_BUDGETS[risk["signal_strength"]], limits.abs_max_trade_risk)
    per_contract_risk = contract["mid"] * 100
    qty = risk["approved_quantity"]
    assert qty >= 1
    assert qty * per_contract_risk <= nav * budget + 1e-6
    # The floor is tight: one more contract would blow the tier budget.
    assert (qty + 1) * per_contract_risk > nav * budget
    assert qty == math.floor(nav * budget / per_contract_risk + 1e-9)
    assert risk["trade_risk_usd"] == pytest.approx(qty * per_contract_risk)


async def test_preview_bear_direction_override_selects_put_or_no_trade(client):
    """direction=BEAR override -> the §8 BEAR column: LONG_PUT with a §9 put
    candidate, or a legitimate §8 NO_TRADE with the matrix rationale in the
    INSTRUMENT gate detail."""
    await authorize(client, STOCK_TICKER)
    body = await preview(client, STOCK_TICKER, direction="BEAR")

    instrument_gate = gate(body, "INSTRUMENT")
    proposed = body["proposed"]
    if proposed["instrument"] == "LONG_PUT":
        assert instrument_gate["status"] == "PASS"
        assert "§8" in instrument_gate["detail"]
        contract = proposed["contract"]
        assert contract is not None and contract["right"] == "P"
        assert proposed["entry_price"] == pytest.approx(contract["mid"] * 100)
    else:
        # Legitimate §8 NO_TRADE (e.g. weak edge or vol cell) — the verdict
        # and its matrix rationale must be explicit (§37).
        assert proposed["instrument"] == "NO_TRADE"
        failed = next(g for g in body["gates"] if g["status"] == "FAIL")
        assert failed["name"] in {"VOLATILITY", "INSTRUMENT"}
        assert "§8" in failed["detail"]
        assert body["why_not_trade"]
    # The override is reported honestly against the live signal.
    assert "override" in gate(body, "DIRECTIONAL_SIGNAL")["detail"]


async def test_preview_vol_caused_no_trade_fails_volatility_gate(client):
    """VOLATILITY FAILs ONLY when vol alone turns the §8 cell into NO_TRADE:
    "GE" deterministically reads BEAR/MODERATE with HIGH vol — the
    'Higher-delta Long Put / No Trade' cell — while the same
    direction/strength under NORMAL vol would trade (LONG_PUT)."""
    await authorize(client, "GE")
    body = await preview(client, "GE")

    vol_gate = gate(body, "VOLATILITY")
    assert vol_gate["status"] == "FAIL", (
        "stub no longer yields the BEAR/MODERATE/HIGH vol-caused NO_TRADE "
        f"cell for GE — re-pick the ticker. Preview: {body['proposed']}, "
        f"gate: {vol_gate}"
    )
    assert "NO_TRADE" in vol_gate["detail"]
    assert "§8" in vol_gate["detail"]
    # The matrix verdict is still reported honestly, with the §8 rationale.
    assert body["proposed"]["instrument"] == "NO_TRADE"
    assert body["proposed"]["vol_regime"] == "HIGH"
    assert body["proposed"]["instrument_rationale"]
    # First FAIL stops the chain: everything later is SKIPPED, risk is null.
    assert gate(body, "INSTRUMENT")["status"] == "SKIPPED"
    assert gate(body, "CONTRACT_SELECTION")["status"] == "SKIPPED"
    assert gate(body, "RISK_APPROVAL")["status"] == "SKIPPED"
    assert body["risk"] is None
    assert body["why_not_trade"]

    # §42: the vetoed cell may never fill.
    r = await approve(client, "GE")
    assert r.status_code == 422
    assert await db_orders() == []


# ---------------------------------------------------------------------------
# Approve: option fill model + persisted rows
# ---------------------------------------------------------------------------


async def test_approve_option_fill_and_rows_hand_computed(client):
    """Approving an option preview fills at mid*(1+slippage) per share with a
    hand-computed cash debit (qty*fill*100 + 0.65*qty) and persists Order +
    Position rows carrying the opt_* contract identity and multiplier 100."""
    settings = get_settings()
    await authorize(client, CALL_TICKER)
    cash_before = await cash(client)

    body_preview = await preview(client, CALL_TICKER)
    mid = body_preview["proposed"]["contract"]["mid"]

    r = await approve(client, CALL_TICKER)
    assert r.status_code == 200, r.text
    body = r.json()

    order = body["order"]
    assert order["side"] == "BUY_TO_OPEN"
    assert order["instrument"] == "LONG_CALL"
    qty = order["quantity"]
    assert qty == body["preview"]["risk"]["approved_quantity"]

    slip = settings.paper_slippage_bps / 10000.0
    expected_fill = mid * (1.0 + slip)  # per share, against the trader
    expected_commission = settings.paper_commission_per_contract * qty
    assert order["fill_price"] == pytest.approx(expected_fill)
    assert order["commission"] == pytest.approx(expected_commission)

    # Cash debit = qty * fill * 100 + per-contract commission (§11, §12.1).
    cash_after = await cash(client)
    assert cash_after == pytest.approx(
        cash_before - (qty * expected_fill * 100 + expected_commission)
    )

    # Response position: premium basis + full premium at risk.
    position = body["position"]
    assert position["instrument"] == "LONG_CALL"
    assert position["avg_price"] == pytest.approx(expected_fill)
    assert position["max_loss"] == pytest.approx(qty * expected_fill * 100)
    assert position["contract"]["multiplier"] == 100

    # Persisted rows carry the contract identity.
    preview_contract = body_preview["proposed"]["contract"]
    rows = await db_orders()
    assert len(rows) == 1
    assert rows[0].instrument == "LONG_CALL"
    assert rows[0].opt_expiry == preview_contract["expiry"]
    assert rows[0].opt_strike == pytest.approx(preview_contract["strike"])
    assert rows[0].opt_right == "C"

    db_pos = await db_position(position["id"])
    assert db_pos.instrument == "LONG_CALL"
    assert db_pos.multiplier == 100
    assert db_pos.opt_expiry == preview_contract["expiry"]
    assert db_pos.opt_strike == pytest.approx(preview_contract["strike"])
    assert db_pos.opt_right == "C"
    # stop_distance stores the per-share fill premium — the §11.3 PREMIUM
    # stop basis, not an underlying stop.
    assert db_pos.stop_distance == pytest.approx(expected_fill)
    assert db_pos.max_loss == pytest.approx(qty * expected_fill * 100)


# ---------------------------------------------------------------------------
# GET /api/positions: option row contract block + option exit families
# ---------------------------------------------------------------------------


async def test_positions_row_shows_contract_block_and_option_exit_rules(client):
    await authorize(client, CALL_TICKER)
    r = await approve(client, CALL_TICKER)
    assert r.status_code == 200, r.text
    fill = r.json()["order"]["fill_price"]
    qty = r.json()["order"]["quantity"]
    preview_contract = r.json()["preview"]["proposed"]["contract"]

    r = await client.get("/api/positions")
    rows = r.json()
    assert len(rows) == 1
    row = rows[0]

    assert row["instrument"] == "LONG_CALL"
    assert row["quantity"] == qty  # contracts
    assert row["avg_price"] == pytest.approx(fill)  # entry premium PER SHARE
    assert row["max_loss"] == pytest.approx(qty * fill * 100)  # premium paid

    contract = row["contract"]
    assert contract is not None
    assert contract["expiry"] == preview_contract["expiry"]
    assert contract["strike"] == pytest.approx(preview_contract["strike"])
    assert contract["right"] == "C"
    assert contract["multiplier"] == 100
    # Same-day chain regeneration is deterministic: the current mid is the
    # preview mid, and dte matches.
    assert contract["current_mid"] == pytest.approx(preview_contract["mid"])
    assert contract["dte"] == preview_contract["dte"]
    assert contract["premium_pnl_pct"] == pytest.approx(
        preview_contract["mid"] / fill - 1.0
    )
    assert row["market_value"] == pytest.approx(
        qty * preview_contract["mid"] * 100
    )

    # The option exit families are evaluated alongside the shared
    # underlying-driven rules (§11.3 / §11.7), every rule reported (§37).
    named = {reason.split(":")[1].strip() for reason in row["exit_reasons"]}
    assert named == OPTION_EXIT_RULES
    # Freshly entered at ~zero premium PnL, far from expiry: it must HOLD.
    assert row["exit_status"] == "HOLD"
    # stop_price is the §11.3 PREMIUM stop per share.
    assert row["stop_price"] == pytest.approx(fill * 0.55)


# ---------------------------------------------------------------------------
# Close: hand-checked realized PnL arithmetic
# ---------------------------------------------------------------------------


async def test_close_option_realized_pnl_hand_checked(client):
    settings = get_settings()
    slip = settings.paper_slippage_bps / 10000.0
    per_contract = settings.paper_commission_per_contract

    await authorize(client, CALL_TICKER)
    initial_cash = await cash(client)
    r = await approve(client, CALL_TICKER)
    assert r.status_code == 200, r.text
    buy_fill = r.json()["order"]["fill_price"]
    qty = r.json()["order"]["quantity"]
    mid = r.json()["preview"]["proposed"]["contract"]["mid"]
    buy_cost = qty * buy_fill * 100 + per_contract * qty

    r = await client.post("/api/orders/close", json={"ticker": CALL_TICKER})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["order"]["side"] == "SELL_TO_CLOSE"  # options: the ONLY close (§5)
    assert body["order"]["instrument"] == "LONG_CALL"
    # Same-day chain -> same contract mid; sell slips AGAINST the trader.
    sell_fill = mid * (1.0 - slip)
    sell_commission = per_contract * qty
    assert body["order"]["fill_price"] == pytest.approx(sell_fill)
    assert body["order"]["commission"] == pytest.approx(sell_commission)

    expected_realized = (sell_fill - buy_fill) * qty * 100 - sell_commission
    assert body["realized_pnl"] == pytest.approx(expected_realized)
    assert body["position"]["status"] == "CLOSED"
    assert body["position"]["quantity"] == 0
    assert body["position"]["realized_pnl"] == pytest.approx(expected_realized)

    # Cash conservation: initial - buy cost + net sell proceeds.
    proceeds = qty * sell_fill * 100 - sell_commission
    assert await cash(client) == pytest.approx(initial_cash - buy_cost + proceeds)


# ---------------------------------------------------------------------------
# check-exits: forced §11.3 PREMIUM_HARD_STOP and §11.7 DTE_EXIT
# ---------------------------------------------------------------------------


async def test_check_exits_forced_premium_hard_stop(client):
    """Force PREMIUM_HARD_STOP by raising the stored entry premium so the
    (unchanged) current mid sits below entry * (1 - 45%): check-exits must
    close the option position with PREMIUM_HARD_STOP audited."""
    await authorize(client, CALL_TICKER)
    r = await approve(client, CALL_TICKER)
    assert r.status_code == 200, r.text
    position_id = r.json()["position"]["id"]
    fill = r.json()["order"]["fill_price"]

    # Direct UPDATE (test-only): entry premium 3x the market -> current mid
    # is ~33% of entry, below the 55% premium stop (plan §11.3).
    async with SessionLocal() as s:
        await s.execute(
            update(Position)
            .where(Position.id == position_id)
            .values(avg_price=fill * 3.0)
        )
        await s.commit()

    r = await client.post("/api/positions/check-exits")
    assert r.status_code == 200
    body = r.json()
    assert body["held"] == []
    assert len(body["exits_triggered"]) == 1
    hit = body["exits_triggered"][0]
    assert hit["ticker"] == CALL_TICKER
    assert hit["rule"] == "PREMIUM_HARD_STOP"

    r = await client.get("/api/positions", params={"status": "CLOSED"})
    rows = r.json()
    assert len(rows) == 1 and rows[0]["id"] == position_id

    # Audited: EXIT_GENERATED (SYSTEM) with the rule, plus a SYSTEM-requested
    # SELL_TO_CLOSE order for the option (rule 12).
    r = await client.get("/api/audit", params={"entity_id": str(position_id)})
    exit_events = [e for e in r.json() if e["action"] == "EXIT_GENERATED"]
    assert len(exit_events) == 1
    assert exit_events[0]["actor_type"] == "SYSTEM"
    assert exit_events[0]["details"]["rule"] == "PREMIUM_HARD_STOP"
    sells = [o for o in await db_orders() if o.side == "SELL_TO_CLOSE"]
    assert len(sells) == 1 and sells[0].instrument == "LONG_CALL"


async def test_check_exits_forced_dte_exit(client):
    """Force DTE_EXIT (§11.7) by moving opt_expiry to 10 days out (<= the 21
    DTE threshold). The entry premium is simultaneously dropped to a token
    value so the §11.3 premium stop cannot fire first, whether or not the
    rewritten expiry still exists in today's chain."""
    await authorize(client, CALL_TICKER)
    r = await approve(client, CALL_TICKER)
    assert r.status_code == 200, r.text
    position_id = r.json()["position"]["id"]

    near_expiry = (datetime.now(timezone.utc).date() + timedelta(days=10)).isoformat()
    async with SessionLocal() as s:
        await s.execute(
            update(Position)
            .where(Position.id == position_id)
            .values(opt_expiry=near_expiry, avg_price=0.10)
        )
        await s.commit()

    r = await client.post("/api/positions/check-exits")
    assert r.status_code == 200
    body = r.json()
    assert len(body["exits_triggered"]) == 1
    assert body["exits_triggered"][0]["rule"] == "DTE_EXIT"

    r = await client.get("/api/audit", params={"entity_id": str(position_id)})
    exit_events = [e for e in r.json() if e["action"] == "EXIT_GENERATED"]
    assert len(exit_events) == 1
    assert exit_events[0]["details"]["rule"] == "DTE_EXIT"

    r = await client.get("/api/positions", params={"status": "CLOSED"})
    rows = r.json()
    assert len(rows) == 1 and rows[0]["id"] == position_id


# ---------------------------------------------------------------------------
# Stock wiring stays intact (deep coverage lives in the stock test modules)
# ---------------------------------------------------------------------------


async def test_stock_flow_keeps_default_instrument_columns(client):
    await authorize(client, STOCK_TICKER)
    r = await approve(client, STOCK_TICKER)
    assert r.status_code == 200, r.text
    assert r.json()["order"]["instrument"] == "LONG_STOCK"
    assert r.json()["order"]["contract"] is None

    rows = await db_orders()
    assert rows[0].opt_expiry is None
    assert rows[0].opt_strike is None
    assert rows[0].opt_right is None

    db_pos = await db_position(r.json()["position"]["id"])
    assert db_pos.instrument == "LONG_STOCK"
    assert db_pos.multiplier == 1
    assert db_pos.opt_expiry is None

    r = await client.get("/api/positions")
    row = r.json()[0]
    assert row["instrument"] == "LONG_STOCK"
    assert row["contract"] is None
