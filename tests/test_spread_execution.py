"""End-to-end BULL_CALL_SPREAD execution (roadmap Phase 1 — FULL UNLOCK).

Simulated venue over the deterministic stub: GW reads BULL / VERY_STRONG at
the anchored date; the documented VOL_REGIME_PARAMS seam forces NORMAL vol
(low_iv tiny, high_iv huge), which lands the §8 "Bull Call Spread" cell.
With allow_defined_risk_spreads ON the whole chain runs: §9-S selection ->
net-debit risk sizing -> atomic paper fill -> spread Position row (short
leg columns) -> net-mid close with 2-leg commission -> §18 reconciliation
counting the short leg as OURS (negative quantity).
"""
import pytest

from apps.gateway.execution import gate_chain

from libs.trading_core.volatility import VolRegimeParams


async def authorize(client, ticker):
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    r = await client.post(
        "/api/trading-pool", json={"ticker": ticker, "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(
        f"/api/trading-pool/{ticker}/trading", json={"enabled": True}
    )
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200


def gate(body, name):
    return next(g for g in body["gates"] if g["name"] == name)


@pytest.fixture
async def spread_env(client, monkeypatch):
    """Spreads permitted + NORMAL vol forced through the documented seam."""
    from apps.gateway.routers import orders as orders_router

    monkeypatch.setattr(gate_chain, "VOL_REGIME_PARAMS",
        VolRegimeParams(low_iv=0.0001, high_iv=99.0, extreme_iv=999.0),
    )
    r = await client.put(
        "/api/config/providers", json={"allow_defined_risk_spreads": "true"}
    )
    assert r.status_code == 200
    await authorize(client, "GW")
    return client


async def test_preview_selects_bull_call_spread_with_net_sizing(spread_env):
    client = spread_env
    r = await client.post("/api/orders/preview", json={"ticker": "GW"})
    assert r.status_code == 200, r.text
    body = r.json()
    proposed = body["proposed"]
    assert proposed["instrument"] == "BULL_CALL_SPREAD", (
        "stub no longer yields BULL/STRONG/NORMAL for GW — re-pick the "
        f"ticker or the vol seam. proposed={proposed}"
    )
    spread = proposed["spread"]
    assert spread is not None and proposed["contract"] is None
    # Both legs identified server-side; defined-risk arithmetic is coherent.
    assert spread["long_symbol"] and spread["short_symbol"]
    assert spread["short_strike"] > spread["long_strike"]  # call vertical
    assert 0 < spread["net_debit"] < spread["width"]
    assert spread["max_loss_per_spread"] == pytest.approx(
        spread["net_debit"] * 100
    )
    assert spread["max_profit_per_spread"] == pytest.approx(
        (spread["width"] - spread["net_debit"]) * 100
    )
    assert spread["breakeven"] == pytest.approx(
        spread["long_strike"] + spread["net_debit"]
    )
    # §12.1: risk basis is the NET debit per spread.
    assert proposed["entry_price"] == pytest.approx(spread["net_debit"] * 100)
    assert gate(body, "CONTRACT_SELECTION")["status"] == "PASS"
    assert gate(body, "INSTRUMENT")["status"] == "PASS"


async def test_approve_close_round_trip_and_reconciliation(spread_env):
    client = spread_env
    # --- Approve: atomic paper fill at NET debit * (1 + slippage). ---------
    r = await client.post("/api/orders/approve", json={"ticker": "GW"})
    assert r.status_code == 200, r.text
    body = r.json()
    order = body["order"]
    position = body["position"]
    assert order["status"] == "FILLED"
    assert order["instrument"] == "BULL_CALL_SPREAD"
    assert position["instrument"] == "BULL_CALL_SPREAD"

    # The stored row carries BOTH legs and net-basis risk numbers.
    from apps.gateway.db import Position
    from sqlalchemy import select
    from apps.gateway.db import SessionLocal  # type: ignore[attr-defined]

    r = await client.get("/api/positions")
    rows = r.json()
    row = next(p for p in rows if p["ticker"] == "GW")
    assert row["contract"]["short_symbol"], row
    assert row["contract"]["short_strike"] > 0

    # §18: the reconciliation ledger counts the short leg as OURS, negative.
    from apps.gateway.routers.broker import _local_open_quantities
    from apps.gateway.routers.orders import _open_position

    async with SessionLocal() as s:
        pos = (
            (await s.execute(select(Position).where(Position.ticker == "GW")))
            .scalars()
            .first()
        )
        assert pos.short_occ_symbol
        quantities = _local_open_quantities([pos])
        long_sym = [k for k in quantities if k != pos.short_occ_symbol][0]
        assert quantities[long_sym] == pos.quantity
        assert quantities[pos.short_occ_symbol] == -pos.quantity
        # Defined risk: max_loss = qty * net fill * 100.
        assert pos.max_loss == pytest.approx(
            pos.quantity * pos.avg_price * 100
        )

    # --- Close: net-mid reference, 2-leg commission, realized PnL. ---------
    r = await client.post("/api/orders/close", json={"ticker": "GW"})
    assert r.status_code == 200, r.text
    closed = r.json()
    assert closed["position"]["status"] == "CLOSED"
    assert closed["realized_pnl"] is not None
    sell = closed["order"]
    assert sell["side"] == "SELL_TO_CLOSE"
    # Commission covered BOTH legs (per-contract × qty × 2).
    from libs.common.config import get_settings

    per_contract = get_settings().paper_commission_per_contract
    assert sell["commission"] == pytest.approx(
        per_contract * sell["quantity"] * 2
    )


async def test_spread_disabled_degrades_to_single_leg_not_dead_end(client, monkeypatch):
    """With the permission OFF the same signal degrades inside the matrix
    (stock for the bull cell) and trades — never a dead end."""
    from apps.gateway.routers import orders as orders_router

    monkeypatch.setattr(gate_chain, "VOL_REGIME_PARAMS",
        VolRegimeParams(low_iv=0.0001, high_iv=99.0, extreme_iv=999.0),
    )
    await authorize(client, "GW")
    r = await client.post("/api/orders/preview", json={"ticker": "GW"})
    body = r.json()
    assert body["proposed"]["instrument"] == "LONG_STOCK"
    assert body["proposed"]["spread"] is None
    assert any(
        "spreads not permitted" in line
        for line in body["proposed"]["instrument_rationale"]
    )
