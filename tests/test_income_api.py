"""Income-strategies API e2e (Phase 2): covered call + CSP over the stub,
simulated venue. THE COLLATERAL LAW is the test target: no short leg
without named, LOCKED backing; pinned shares refuse to sell; buyback
releases everything. The permission gate itself stays LOCKED until the
Phase 2 chain completes — tests bypass it via the single seam
(_require_income_permission) to exercise the chain underneath.
"""
import pytest

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


async def fund_paper_account(cash: float) -> None:
    """Set the SIMULATED ledger's cash (a fixture knob, like
    ``paper_initial_cash``): a cash-secured put risks (strike − credit) × 100
    per contract — stock to zero — and the Tier 0 per-trade ceiling
    (``abs_max_trade_risk`` 1.5% of NAV, audit §10 B0) needs a NAV that can
    carry that before ONE contract is approvable."""
    from apps.gateway.db import SessionLocal, get_or_create_portfolio, utcnow

    async with SessionLocal() as s:
        portfolio = await get_or_create_portfolio(s)
        portfolio.cash = cash
        portfolio.updated_at = utcnow()
        await s.commit()


async def open_stock(client, ticker="GOOGL"):
    await authorize(client, ticker)
    r = await client.post("/api/orders/approve", json={"ticker": ticker})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["position"]["instrument"] == "LONG_STOCK"
    return body["position"]


@pytest.fixture
async def income_unlocked(client):
    """Phase 2 UNLOCKED: enable both income permissions through the REAL
    runtime-config path — the same toggle the Settings UI drives."""
    r = await client.put(
        "/api/config/providers",
        json={"allow_covered_call": "true", "allow_cash_secured_put": "true"},
    )
    assert r.status_code == 200


async def test_permission_gate_is_locked_by_default(client):
    r = await client.post("/api/income/covered-call", json={"ticker": "GOOGL"})
    assert r.status_code == 403
    assert "disabled in account permissions" in r.json()["detail"]


async def test_covered_call_full_cycle_with_collateral_law(
    client, income_unlocked
):
    stock = await open_stock(client, "GOOGL")
    shares = stock["quantity"]
    assert shares >= 100, "stub sizing must afford at least one contract"

    # No stock -> no covered call (different ticker). AAPL is promoted into
    # the Trading Pool first: income opens now pass gate 1 (audit §10 B0),
    # so the COLLATERAL refusal — the thing under test here — is reached.
    await authorize(client, "AAPL")
    r = await client.post("/api/income/covered-call", json={"ticker": "AAPL"})
    assert r.status_code == 422
    assert "no OPEN LONG_STOCK" in r.json()["detail"]

    # Open 1 covered call against the holding.
    r = await client.post(
        "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 200, r.text
    cc = r.json()["position"]
    assert cc["instrument"] == "COVERED_CALL"
    assert cc["collateral_position_id"] == stock["id"]
    assert cc["credit_per_share"] > 0
    assert cc["occ_symbol"].startswith("GOOGL")

    # Collateral law: selling the pinned shares is refused, named.
    r = await client.post("/api/orders/close", json={"ticker": "GOOGL"})
    assert r.status_code == 422
    assert "pinned as covered-call collateral" in r.json()["detail"]

    # Over-pinning: more contracts than free shares -> refused with counts.
    too_many = shares // 100 + 1
    r = await client.post(
        "/api/income/covered-call",
        json={"ticker": "GOOGL", "contracts": too_many},
    )
    assert r.status_code == 422
    assert "free shares" in r.json()["detail"]

    # Buyback closes the short leg and RELEASES the pin.
    r = await client.post(f"/api/income/{cc['id']}/buyback", json={})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "CLOSED"
    r = await client.post("/api/orders/close", json={"ticker": "GOOGL"})
    assert r.status_code == 200, r.text  # shares free again


async def test_csp_reserves_cash_and_respects_the_reserve(
    client, income_unlocked
):
    await authorize(client, "GOOGL")
    # A NAV that can carry one CSP under the 1.5%-of-NAV per-trade ceiling
    # (risk basis (strike − credit) × 100, audit §10 B0) but NOT the cash
    # collateral for 100 contracts (strike × 100 × 100 > $600k).
    await fund_paper_account(600_000.0)
    # Backfill stored bars (the CSP path needs a spot reference).
    r = await client.post("/api/orders/preview", json={"ticker": "GOOGL"})
    assert r.status_code == 200
    r = await client.post(
        "/api/income/cash-secured-put", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 200, r.text
    csp = r.json()["position"]
    assert r.json()["risk"]["decision"] == "APPROVE"
    assert r.json()["risk"]["approved_quantity"] == 1
    assert csp["instrument"] == "CASH_SECURED_PUT"
    assert csp["cash_reserved"] > 0
    assert csp["collateral_position_id"] is None

    # Reserve is respected: an absurd size cannot be secured.
    r = await client.post(
        "/api/income/cash-secured-put",
        json={"ticker": "GOOGL", "contracts": 100},
    )
    assert r.status_code == 422
    assert "insufficient free cash" in r.json()["detail"]
    assert "already reserved" in r.json()["detail"]

    # Buyback releases the reservation.
    r = await client.post(f"/api/income/{csp['id']}/buyback", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "CLOSED"


async def test_kill_switch_blocks_opens_but_never_buybacks(
    client, income_unlocked
):
    await open_stock(client, "GOOGL")
    r = await client.post(
        "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 200
    cc_id = r.json()["position"]["id"]

    r = await client.post("/api/trading/pause", json={"reason": "test"})
    assert r.status_code == 200
    r = await client.post(
        "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 409
    assert "kill switch" in r.json()["detail"]
    # Risk-reducing buyback still allowed while paused (§18 priority).
    r = await client.post(f"/api/income/{cc_id}/buyback", json={})
    assert r.status_code == 200


async def test_reconciliation_claims_income_short_legs_negative(
    client, income_unlocked
):
    await open_stock(client, "GOOGL")
    r = await client.post(
        "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
    )
    assert r.status_code == 200
    from sqlalchemy import select

    from apps.gateway.db import Position, SessionLocal
    from apps.gateway.routers.broker import _local_open_quantities

    async with SessionLocal() as s:
        rows = (
            (
                await s.execute(
                    select(Position).where(Position.status == "OPEN")
                )
            )
            .scalars()
            .all()
        )
        quantities = _local_open_quantities(rows)
        cc = next(p for p in rows if p.instrument == "COVERED_CALL")
        from datetime import date as _date

        from libs.broker.alpaca import occ_option_symbol

        occ = occ_option_symbol(
            cc.ticker,
            _date.fromisoformat(cc.opt_expiry),
            cc.opt_strike,
            cc.opt_right,
        )
        assert quantities[occ] == -cc.quantity  # OURS, negative
        assert quantities["GOOGL"] > 0  # the stock collateral, positive


async def test_income_broker_path_settles_at_the_brokers_credit(
    client, income_unlocked, monkeypatch
):
    """Real-broker short open (T1/T2): the position opens at the BROKER's
    credit, no local cash mutation; buyback closes at the broker's fill."""
    import contextlib

    import tests.test_broker_execution as bx

    await open_stock(client, "GOOGL")

    fake = bx.FakeAlpaca(
        bx._order_payload_factory(
            side="sell", status="filled", filled_qty="1", filled_avg_price="2.40"
        )
        if hasattr(bx, "_order_payload_factory")
        else {
            "id": "ord-sto-1",
            "client_order_id": "x",
            "symbol": "GOOGL",
            "side": "sell",
            "status": "filled",
            "qty": "1",
            "filled_qty": "1",
            "filled_avg_price": "2.40",
            "submitted_at": "2026-08-17T14:30:00Z",
        }
    )
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca_paper")
    from libs.common.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(
        "apps.gateway.deps.get_broker", lambda name: fake.broker()
    )
    try:
        r = await client.post(
            "/api/income/covered-call", json={"ticker": "GOOGL", "contracts": 1}
        )
        assert r.status_code == 200, r.text
        pos = r.json()["position"]
        assert pos is not None
        assert pos["credit_per_share"] == 2.40  # THE BROKER'S number
        # The wire carried sell_to_open with the collateral attestation gate
        # passed (the adapter would have raised otherwise).
        posted = [b for b in fake.posted if b.get("position_intent") == "sell_to_open"]
        assert posted and posted[0]["side"] == "sell"

        # Buyback at the broker.
        fake.order_payload = {
            **fake.order_payload,
            "id": "ord-btc-1",  # a DIFFERENT broker order id (unique column)
            "side": "buy",
            "filled_avg_price": "1.10",
        }
        r = await client.post(f"/api/income/{pos['id']}/buyback", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "CLOSED"
        assert body["buyback_price"] == 1.10
        assert body["realized_pnl"] == pytest.approx((2.40 - 1.10) * 100)
    finally:
        get_settings.cache_clear()
