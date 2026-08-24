"""End-to-end SHORT_STOCK execution (roadmap Phase 3 — FULL UNLOCK).

Simulated venue over the deterministic stub: "PLTR" reads BEAR/MODERATE at
the anchored date; the documented VOL_REGIME_PARAMS seam forces HIGH vol,
landing the §8 BEAR/MODERATE/HIGH cell — premium too expensive to buy, so
with allow_short_stock AND allow_margin ON the matrix emits SHORT_STOCK.
The chain then runs end to end: gap-inflated §12.1 risk sizing -> paper
SELL_TO_OPEN with proceeds CREDITED -> liability position (negative market
value, delta −1, stop ABOVE entry) -> BUY_TO_CLOSE cover with mirrored
P&L -> §18 reconciliation claiming the ticker at NEGATIVE quantity.
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
async def short_env(client, monkeypatch):
    """short_stock + margin permitted through the REAL runtime-config path
    (the same toggles the Settings UI drives) + HIGH vol forced through the
    documented seam."""
    from apps.gateway.routers import orders as orders_router

    monkeypatch.setattr(gate_chain, "VOL_REGIME_PARAMS",
        VolRegimeParams(low_iv=0.001, high_iv=0.01, extreme_iv=99.0),
    )
    r = await client.put(
        "/api/config/providers",
        json={"allow_short_stock": "true", "allow_margin": "true"},
    )
    assert r.status_code == 200
    await authorize(client, "PLTR")
    return client


async def test_preview_selects_short_stock_with_gap_inflated_risk(short_env):
    client = short_env
    r = await client.post("/api/orders/preview", json={"ticker": "PLTR"})
    assert r.status_code == 200, r.text
    body = r.json()
    proposed = body["proposed"]
    assert proposed["instrument"] == "SHORT_STOCK", (
        "stub no longer yields BEAR/MODERATE for PLTR — re-pick the ticker "
        f"or the vol seam. proposed={proposed}"
    )
    assert proposed["vol_regime"] == "HIGH"
    assert proposed["contract"] is None and proposed["spread"] is None
    assert gate(body, "CONTRACT_SELECTION")["status"] == "SKIPPED"
    assert gate(body, "RISK_APPROVAL")["status"] == "PASS"
    # The short's stop sits ABOVE entry in the §24 exit plan.
    hard_stop = body["exit_plan"]["hard_stop"]
    assert "+" in hard_stop and "2×ATR14" in hard_stop
    entry = proposed["entry_price"]
    stop_distance = proposed["stop_distance"]
    assert entry is not None and stop_distance is not None and stop_distance > 0


async def test_short_stock_requires_both_flags_to_trade(client, monkeypatch):
    """With only ONE of the two flags the same cell stays NO_TRADE — margin
    exists to support shorting and neither flag alone opens the chain."""
    from apps.gateway.routers import orders as orders_router

    monkeypatch.setattr(gate_chain, "VOL_REGIME_PARAMS",
        VolRegimeParams(low_iv=0.001, high_iv=0.01, extreme_iv=99.0),
    )
    r = await client.put(
        "/api/config/providers", json={"allow_short_stock": "true"}
    )
    assert r.status_code == 200
    await authorize(client, "PLTR")
    r = await client.post("/api/orders/preview", json={"ticker": "PLTR"})
    assert r.status_code == 200
    body = r.json()
    assert body["proposed"]["instrument"] == "NO_TRADE"
    assert any(
        "short stock is not enabled" in line
        for line in body["proposed"]["instrument_rationale"]
    )


async def test_short_open_cover_round_trip_cash_and_reconciliation(short_env):
    client = short_env
    from sqlalchemy import select

    from apps.gateway.db import Position, SessionLocal
    from apps.gateway.db import get_or_create_portfolio
    from apps.gateway.routers.broker import _local_open_quantities

    async with SessionLocal() as s:
        cash_before = (await get_or_create_portfolio(s)).cash
        await s.commit()

    # --- Approve: paper SELL_TO_OPEN, proceeds credited. -------------------
    r = await client.post("/api/orders/approve", json={"ticker": "PLTR"})
    assert r.status_code == 200, r.text
    body = r.json()
    order = body["order"]
    position = body["position"]
    assert order["status"] == "FILLED"
    assert order["side"] == "SELL_TO_OPEN"
    assert order["instrument"] == "SHORT_STOCK"
    assert position["instrument"] == "SHORT_STOCK"
    qty = position["quantity"]
    fill = position["avg_price"]
    assert qty > 0 and fill > 0
    # The mirrored stop is reported ABOVE entry.
    assert position["stop_price"] > fill

    from libs.common.config import get_settings

    settings = get_settings()
    commission = settings.paper_commission_per_share * qty
    async with SessionLocal() as s:
        cash_after_open = (await get_or_create_portfolio(s)).cash
        rows = (
            (await s.execute(select(Position).where(Position.status == "OPEN")))
            .scalars()
            .all()
        )
        await s.commit()
    assert cash_after_open == pytest.approx(cash_before + qty * fill - commission)

    # §18: the reconciliation ledger claims the ticker at NEGATIVE quantity.
    quantities = _local_open_quantities(rows)
    assert quantities["PLTR"] == -qty

    # The positions view carries the liability honestly.
    r = await client.get("/api/positions")
    row = next(p for p in r.json() if p["ticker"] == "PLTR")
    assert row["market_value"] is not None and row["market_value"] < 0
    assert row["stop_price"] > fill

    # --- Close: BUY_TO_CLOSE cover, mirrored P&L. --------------------------
    r = await client.post("/api/orders/close", json={"ticker": "PLTR"})
    assert r.status_code == 200, r.text
    closed = r.json()
    assert closed["position"]["status"] == "CLOSED"
    assert closed["order"]["side"] == "BUY_TO_CLOSE"
    cover = closed["order"]["fill_price"]
    assert closed["realized_pnl"] == pytest.approx(
        (fill - cover) * qty - commission
    )
    async with SessionLocal() as s:
        cash_final = (await get_or_create_portfolio(s)).cash
        await s.commit()
    # Cash identity: realized excludes the OPEN-side commission (it was
    # netted from the credited proceeds at open, mirroring the long path).
    assert cash_final == pytest.approx(
        cash_before + closed["realized_pnl"] - commission
    )


async def test_kill_switch_blocks_short_opens_but_never_covers(short_env):
    client = short_env
    r = await client.post("/api/orders/approve", json={"ticker": "PLTR"})
    assert r.status_code == 200, r.text

    r = await client.post("/api/trading/pause", json={"reason": "test"})
    assert r.status_code == 200
    # A second short (same or another ticker) is blocked by the gate chain.
    r = await client.post("/api/orders/approve", json={"ticker": "PLTR"})
    assert r.status_code in (409, 422)
    # Covering REDUCES risk — always allowed under the pause (§18).
    r = await client.post("/api/orders/close", json={"ticker": "PLTR"})
    assert r.status_code == 200, r.text
    assert r.json()["position"]["status"] == "CLOSED"
