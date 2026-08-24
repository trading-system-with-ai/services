"""Income opens go through Tier 0 (risk-engine audit §8 item 3, §10 Phase
B0; spec risk_engine.md §2, §72 "the user must not accidentally bypass hard
limits").

Covered: Trading Pool authorization is gate 1 for income opens; a risk-engine
REJECT answers 422 with reason codes and creates NO position/order; every
open (approved or not) writes exactly ONE SYSTEM RISK_DECISION with the
order path's key shape; APPROVE_WITH_RESIZE fills the APPROVED contracts and
reserves collateral for those only; the portfolio risk view reports the
pledged CSP collateral (``cash_reserved_usd``, additive).
"""
import math
import pytest
from sqlalchemy import select

from tests.test_income_api import (  # noqa: F401  (fixtures + helpers)
    authorize,
    fund_paper_account,
    income_unlocked,
    open_stock,
)

RISK_KEYS = {
    "decision",
    "mode",
    "instrument",
    "reason_codes",
    "explanations",
    "quantity_requested",
    "approved_quantity",
    "heat_before_pct",
    "heat_after_pct",
    "cash_after_pct",
    "greeks_checked",
}


async def income_risk_decisions(client, ticker):
    """SYSTEM RISK_DECISION events written by the INCOME opens for `ticker`
    (entity_type income_open — the stock order path writes its own with
    entity_type order_preview)."""
    r = await client.get(
        "/api/audit", params={"entity_id": ticker, "action": "RISK_DECISION"}
    )
    assert r.status_code == 200
    events = [e for e in r.json() if e["entity_type"] == "income_open"]
    for e in events:
        assert e["actor_type"] == "SYSTEM"
    return events


async def open_rows(instrument):
    from apps.gateway.db import Order, Position, SessionLocal

    async with SessionLocal() as s:
        positions = (
            (
                await s.execute(
                    select(Position).where(Position.instrument == instrument)
                )
            )
            .scalars()
            .all()
        )
        orders = (
            (await s.execute(select(Order).where(Order.instrument == instrument)))
            .scalars()
            .all()
        )
        return positions, orders


async def test_pool_unauthorized_income_open_is_refused_and_audited(
    client, income_unlocked
):
    # In the Watchlist only, NOT the Trading Pool -> gate 1 veto (§21/§32),
    # named, 422 like the order path's approve denial; the veto is a
    # RISK_DECISION (decision VETOED) so it stays auditable.
    r = await client.post("/api/watchlist", json={"ticker": "MSFT"})
    assert r.status_code == 201
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200
    r = await client.post(
        "/api/income/cash-secured-put", json={"ticker": "MSFT", "contracts": 1}
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["veto_gate"] == "TRADING_POOL_AUTHORIZATION"
    assert detail["reason_codes"] == ["VETO_TRADING_POOL_AUTHORIZATION"]
    assert "not in the Trading Pool" in detail["explanations"][0]
    events = await income_risk_decisions(client, "MSFT")
    assert len(events) == 1
    assert events[0]["details"]["decision"] == "VETOED"
    assert events[0]["details"]["veto_gate"] == "TRADING_POOL_AUTHORIZATION"
    assert events[0]["details"]["instrument"] == "CASH_SECURED_PUT"

    # In the pool but per-symbol trading DISABLED -> same gate, other reason.
    r = await client.post(
        "/api/trading-pool", json={"ticker": "MSFT", "acknowledge_risks": True}
    )
    assert r.status_code == 201
    r = await client.post(
        "/api/income/cash-secured-put", json={"ticker": "MSFT", "contracts": 1}
    )
    assert r.status_code == 422
    assert "not enabled for MSFT" in r.json()["detail"]["explanations"][0]
    positions, orders = await open_rows("CASH_SECURED_PUT")
    assert positions == [] and orders == []


async def test_csp_reject_returns_422_with_reasons_and_creates_nothing(
    client, income_unlocked
):
    # Default $100,000 paper account: one CSP risks (strike − credit) × 100
    # (stock to zero) — far above the 1.5%-of-NAV absolute per-trade ceiling
    # ($1,500), so the risk engine REJECTs with ABS_TRADE_RISK_CAP.
    await authorize(client, "GOOGL")
    r = await client.post("/api/orders/preview", json={"ticker": "GOOGL"})
    assert r.status_code == 200  # backfills the stored bars (spot reference)
    r = await client.post(
        "/api/income/cash-secured-put", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["veto_gate"] == "RISK_APPROVAL"
    assert detail["reason_codes"] == ["ABS_TRADE_RISK_CAP"]
    assert "1.50% of NAV" in detail["explanations"][0]
    assert detail["risk"]["decision"] == "REJECT"
    assert detail["risk"]["approved_quantity"] == 0
    assert detail["risk"]["quantity_requested"] == 1

    positions, orders = await open_rows("CASH_SECURED_PUT")
    assert positions == [] and orders == []
    # The REJECT is a decision on a write path: exactly one audit row.
    events = await income_risk_decisions(client, "GOOGL")
    assert len(events) == 1
    d = events[0]["details"]
    assert RISK_KEYS <= set(d)
    assert d["decision"] == "REJECT"
    assert d["veto_gate"] == "RISK_APPROVAL"
    assert d["reason_codes"] == ["ABS_TRADE_RISK_CAP"]
    assert d["approved_quantity"] == 0


async def test_each_successful_open_writes_one_risk_decision(
    client, income_unlocked
):
    stock = await open_stock(client, "GOOGL")
    r = await client.post(
        "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk"]["decision"] == "APPROVE"
    assert body["risk"]["quantity_requested"] == 1
    assert body["risk"]["approved_quantity"] == 1
    assert body["position"]["collateral_position_id"] == stock["id"]

    events = await income_risk_decisions(client, "GOOGL")
    assert len(events) == 1
    d = events[0]["details"]
    assert RISK_KEYS <= set(d)
    assert d["decision"] == "APPROVE"
    assert d["mode"] == "execution"
    assert d["instrument"] == "COVERED_CALL"
    assert d["veto_gate"] is None
    assert d["reason_codes"] == []
    assert d["quantity_requested"] == 1 and d["approved_quantity"] == 1
    # Covered call: zero risk / capital basis -> heat unchanged by the open.
    assert d["heat_after_pct"] == pytest.approx(d["heat_before_pct"])
    assert d["greeks_checked"] is True  # the stub chain reports greeks

    # A second open -> a second event (one per open, never batched).
    if stock["quantity"] >= 200:
        r = await client.post(
            "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
        )
        assert r.status_code == 200, r.text
        assert len(await income_risk_decisions(client, "GOOGL")) == 2


async def test_resize_opens_the_approved_contracts_only(client, income_unlocked):
    # NAV $600,000: abs cap 1.5% = $9,000; one CSP risks ~$7.2k on the stub
    # GOOGL chain -> 3 requested contracts resize to 1 (RESIZED_BY_ABS_
    # TRADE_RISK_CAP); the position and its cash reservation carry the
    # APPROVED count, never the requested one.
    await authorize(client, "GOOGL")
    await fund_paper_account(600_000.0)
    r = await client.post("/api/orders/preview", json={"ticker": "GOOGL"})
    assert r.status_code == 200
    r = await client.post(
        "/api/income/cash-secured-put", json={"ticker": "GOOGL", "contracts": 3}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk"]["decision"] == "APPROVE_WITH_RESIZE"
    assert body["risk"]["quantity_requested"] == 3
    assert body["risk"]["approved_quantity"] == 1
    assert body["risk"]["reason_codes"] == ["RESIZED_BY_ABS_TRADE_RISK_CAP"]
    pos = body["position"]
    assert pos["contracts"] == 1
    strike = float(pos["occ_symbol"][-8:]) / 1000.0  # OCC strike * 1000
    assert pos["cash_reserved"] == pytest.approx(strike * 100 * 1)

    positions, orders = await open_rows("CASH_SECURED_PUT")
    assert len(positions) == 1 and positions[0].quantity == 1
    assert len(orders) == 1 and orders[0].quantity == 1
    events = await income_risk_decisions(client, "GOOGL")
    assert len(events) == 1
    d = events[0]["details"]
    assert d["decision"] == "APPROVE_WITH_RESIZE"
    assert d["quantity_requested"] == 3 and d["approved_quantity"] == 1

    # The portfolio risk view reports the pledged collateral (ADDITIVE field;
    # cash keeps its account-cash semantics).
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    view = r.json()
    assert view["cash_reserved_usd"] == pytest.approx(pos["cash_reserved"])
    assert view["cash"] is not None and view["cash"] > view["cash_reserved_usd"]

    # A second CSP now sizes against the NETTED usable cash / NAV: the risk
    # decision names the reserved total it netted out.
    r = await client.post(
        "/api/income/cash-secured-put", json={"ticker": "GOOGL", "contracts": 1}
    )
    events = await income_risk_decisions(client, "GOOGL")
    assert len(events) == 2
    latest = max(events, key=lambda e: e["id"])
    assert latest["details"]["cash_reserved_total"] == pytest.approx(
        pos["cash_reserved"]
    )
    assert latest["details"]["usable_cash"] == pytest.approx(
        view["cash"] - pos["cash_reserved"]
    )


async def test_risk_view_cash_reserved_is_zero_with_no_open_puts(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    assert r.json()["cash_reserved_usd"] == 0.0


async def test_risk_view_cash_reserved_is_null_without_an_account(
    unconfigured_client,
):
    r = await unconfigured_client.get("/api/portfolio/risk")
    assert r.status_code == 200
    body = r.json()
    assert body["cash"] is None
    assert body["cash_reserved_usd"] is None


async def test_snapshot_nav_is_unnetted_but_deployable_cash_nets_csp_collateral(
    client, income_unlocked
):
    """QA follow-up (2026-08-17): pledged CSP collateral is still the
    account's asset, so the write-path NAV equals the risk view's NAV; only
    the DEPLOYABLE cash the §13 floor measures is netted."""
    from apps.gateway.db import Position, SessionLocal
    from apps.gateway.risk_inputs import build_portfolio_snapshot

    async with SessionLocal() as session:
        session.add(
            Position(
                ticker="XOM",
                instrument="CASH_SECURED_PUT",
                quantity=-1,
                avg_price=2.0,
                max_loss=9_800.0,
                status="OPEN",
                stop_distance=2.0,
                multiplier=100,
                opt_expiry="2030-01-17",
                opt_strike=100.0,
                opt_right="P",
                cash_reserved=10_000.0,
            )
        )
        await session.commit()
        inputs = await build_portfolio_snapshot(
            session, cash=50_000.0, trading_enabled=True
        )
        assert inputs.cash_reserved_total == 10_000.0
        assert inputs.usable_cash == 40_000.0
        assert inputs.snapshot.cash == 40_000.0
        # NAV = cash + Σ market values (short put MV = −qty×credit×100 = −200
        # when the ticker has bars; with no stored bars the value is None ->
        # counted 0). Either way NAV is built on the UN-netted 50,000.
        mv = math.fsum(v for v in inputs.values if v is not None)
        assert inputs.snapshot.nav == 50_000.0 + mv
        assert inputs.snapshot.nav >= 49_000.0
