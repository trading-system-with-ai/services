"""Paper order execution tests — approve + close (plan §11, §42, §5, §18).

Covers: §42 "no rejected ticker may produce an order" (422 with the fresh
preview embedded, zero Order rows), the documented paper fill model
hand-computed against Settings, the ORDER_REQUESTED -> ORDER_SUBMITTED ->
ORDER_FILLED audit chain, client_order_id idempotency (§42), the V1
no-pyramiding 409, close partial/full realized-PnL arithmetic with cash
conservation, closing while paused (§18 risk-priority), and the close error
paths (404 / 422).
"""
import pytest
from sqlalchemy import select

from apps.gateway.db import Order, SessionLocal
from libs.common.config import get_settings

# Deterministic stub (RNG seeded by the symbol): GOOGL yields a bull regime
# with a BULL bias, so the §10 chain approves — the same property
# tests/test_order_preview.py relies on.
BULL_TICKER = "GOOGL"


async def authorize(client, ticker, *, enable_symbol=True, resume=True):
    """Watchlist -> Trading Pool (-> per-symbol enable) (-> global resume)."""
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    r = await client.post("/api/trading-pool", json={"ticker": ticker})
    assert r.status_code == 201
    if enable_symbol:
        r = await client.post(
            f"/api/trading-pool/{ticker}/trading", json={"enabled": True}
        )
        assert r.status_code == 200
    if resume:
        r = await client.post("/api/trading/resume", json={})
        assert r.status_code == 200


async def approve(client, ticker, quantity=None, client_order_id=None):
    body = {"ticker": ticker}
    if quantity is not None:
        body["quantity"] = quantity
    if client_order_id is not None:
        body["client_order_id"] = client_order_id
    return await client.post("/api/orders/approve", json=body)


async def cash(client):
    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    return r.json()["cash"]


async def order_rows():
    """All Order rows, read directly from the shared in-memory test DB."""
    async with SessionLocal() as s:
        return list((await s.execute(select(Order).order_by(Order.id))).scalars().all())


async def audit_actions_for(client, entity_id):
    """Audit actions for one entity, oldest first."""
    r = await client.get("/api/audit", params={"entity_id": str(entity_id)})
    assert r.status_code == 200
    return list(reversed(r.json()))


# ---------------------------------------------------------------------------
# /approve
# ---------------------------------------------------------------------------


async def test_approve_rejected_ticker_422_with_preview_and_no_order(client):
    """§42: a symbol failing any gate answers 422 with the fresh gate-chain
    preview embedded, and NO Order row may exist afterwards."""
    # Watchlist only — never promoted to the pool, so gate 1 FAILs.
    r = await client.post("/api/watchlist", json={"ticker": "NVDA"})
    assert r.status_code == 201

    r = await approve(client, "NVDA")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert set(detail) == {"message", "preview"}
    assert "no rejected ticker may produce an order" in detail["message"]

    preview = detail["preview"]
    gates = preview["gates"]
    assert gates[0]["name"] == "TRADING_POOL_AUTHORIZATION"
    assert gates[0]["status"] == "FAIL"
    assert preview["why_not_trade"]

    # "No rejected ticker may produce an order" (§42): zero Order rows.
    assert await order_rows() == []
    r = await client.get("/api/positions")
    assert r.json() == []

    # The veto is still auditable (rule 12): the chain's RISK_DECISION landed.
    events = await audit_actions_for(client, "NVDA")
    decisions = [e for e in events if e["action"] == "RISK_DECISION"]
    assert len(decisions) == 1
    assert decisions[0]["details"]["decision"] == "VETOED"


async def test_approve_happy_path_fill_model_and_audit_chain(client):
    """Fully authorized approve: FILLED order, position with stop/entry_edge,
    cash decremented by exactly qty*fill + commission (hand-computed against
    the documented paper fill model), and the full audit chain present."""
    settings = get_settings()
    await authorize(client, BULL_TICKER)
    cash_before = await cash(client)
    assert cash_before == settings.paper_initial_cash

    r = await approve(client, BULL_TICKER)
    assert r.status_code == 200
    body = r.json()

    order = body["order"]
    assert order["side"] == "BUY_TO_OPEN"  # the ONLY opening side (§5)
    assert order["status"] == "FILLED"
    qty = order["quantity"]
    assert qty > 0

    # Approve embeds the same §10 gate-chain response shape preview returns.
    preview = body["preview"]
    assert [g["name"] for g in preview["gates"]][0] == "TRADING_POOL_AUTHORIZATION"
    assert preview["risk"]["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE"}
    # No quantity requested -> the fill takes exactly the approved quantity.
    assert qty == preview["risk"]["approved_quantity"]

    # Hand-computed paper fill model (plan §11): BUY fills ABOVE the last
    # stored close by paper_slippage_bps; commission per share both ways.
    last_close = preview["proposed"]["entry_price"]
    expected_fill = last_close * (1.0 + settings.paper_slippage_bps / 10000.0)
    expected_commission = settings.paper_commission_per_share * qty
    assert order["fill_price"] == pytest.approx(expected_fill)
    assert order["commission"] == pytest.approx(expected_commission)

    position = body["position"]
    assert position["avg_price"] == pytest.approx(expected_fill)
    assert position["stop_price"] == pytest.approx(
        expected_fill - preview["proposed"]["stop_distance"]
    )
    assert position["max_loss"] == pytest.approx(
        qty * preview["proposed"]["stop_distance"]
    )

    # Cash decreased by exactly qty * fill + commission.
    cash_after = await cash(client)
    assert cash_after == pytest.approx(
        cash_before - (qty * expected_fill + expected_commission)
    )

    # Position row carries the entry-time exit state.
    r = await client.get("/api/positions")
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["entry_edge"] == pytest.approx(preview["signal"]["edge"])
    assert rows[0]["stop_price"] == pytest.approx(position["stop_price"])

    # Audit chain ORDER_REQUESTED -> ORDER_SUBMITTED -> ORDER_FILLED, in
    # order, in the same transaction as the fill (rule 12).
    events = await audit_actions_for(client, order["id"])
    assert [e["action"] for e in events] == [
        "ORDER_REQUESTED",
        "ORDER_SUBMITTED",
        "ORDER_FILLED",
    ]
    assert events[0]["actor_type"] == "USER"
    assert events[1]["actor_type"] == "SYSTEM"
    assert events[2]["actor_type"] == "SYSTEM"
    assert events[2]["details"]["position_id"] == position["id"]


async def test_duplicate_client_order_id_is_idempotent(client):
    """§42: replaying the same client_order_id returns the EXISTING order —
    one Order row, no second fill, cash untouched by the second call."""
    await authorize(client, BULL_TICKER)

    r1 = await approve(client, BULL_TICKER, client_order_id="dup-key-1")
    assert r1.status_code == 200
    cash_after_first = await cash(client)

    r2 = await approve(client, BULL_TICKER, client_order_id="dup-key-1")
    assert r2.status_code == 200
    assert r2.json()["order"]["id"] == r1.json()["order"]["id"]

    rows = await order_rows()
    assert len(rows) == 1  # ONE fill, ever
    assert await cash(client) == pytest.approx(cash_after_first)


async def test_second_approve_same_ticker_409_no_pyramiding(client):
    """An existing OPEN position in the ticker -> 409 (no pyramiding in V1)."""
    await authorize(client, BULL_TICKER)
    r = await approve(client, BULL_TICKER)
    assert r.status_code == 200

    r = await approve(client, BULL_TICKER)
    assert r.status_code == 409
    assert len(await order_rows()) == 1


# ---------------------------------------------------------------------------
# /close
# ---------------------------------------------------------------------------


async def test_close_partial_then_full_pnl_and_cash_conservation(client):
    """Partial then full close: realized_pnl hand-checked against the fill
    model (buy-side commission NOT re-charged — it left cash at open), and
    final cash == initial - buy_cost + all sell proceeds."""
    settings = get_settings()
    slip = settings.paper_slippage_bps / 10000.0
    per_share = settings.paper_commission_per_share

    await authorize(client, BULL_TICKER)
    initial_cash = await cash(client)

    r = await approve(client, BULL_TICKER, quantity=10)
    assert r.status_code == 200
    order = r.json()["order"]
    qty = order["quantity"]
    assert qty == 10  # risk budget easily allows 10 shares at V1 defaults
    avg = order["fill_price"]
    buy_cost = qty * avg + order["commission"]

    # The sell reference price is the last STORED close.
    r = await client.get("/api/positions")
    last_close = r.json()[0]["current_price"]
    sell_fill = last_close * (1.0 - slip)

    # Partial close: 4 of 10.
    r = await client.post(
        "/api/orders/close", json={"ticker": BULL_TICKER, "quantity": 4}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["order"]["side"] == "SELL_TO_CLOSE"  # the ONLY closing side (§5)
    commission_1 = per_share * 4
    realized_1 = (sell_fill - avg) * 4 - commission_1
    assert body["realized_pnl"] == pytest.approx(realized_1)
    assert body["position"]["status"] == "OPEN"
    assert body["position"]["quantity"] == 6
    assert body["position"]["realized_pnl"] == pytest.approx(realized_1)

    # Full close of the remaining 6 (quantity omitted -> full).
    r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
    assert r.status_code == 200
    body = r.json()
    commission_2 = per_share * 6
    realized_2 = (sell_fill - avg) * 6 - commission_2
    assert body["realized_pnl"] == pytest.approx(realized_2)
    assert body["position"]["status"] == "CLOSED"
    assert body["position"]["quantity"] == 0
    assert body["position"]["closed_at"] is not None
    assert body["position"]["realized_pnl"] == pytest.approx(realized_1 + realized_2)

    # Cash conservation: initial - buy_cost + sell proceeds (both sells).
    proceeds = (4 * sell_fill - commission_1) + (6 * sell_fill - commission_2)
    assert await cash(client) == pytest.approx(initial_cash - buy_cost + proceeds)

    # The CLOSED row carries the accumulated realized_pnl.
    r = await client.get("/api/positions", params={"status": "CLOSED"})
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["realized_pnl"] == pytest.approx(realized_1 + realized_2)


async def test_close_allowed_while_trading_paused(client):
    """§18 risk-priority: the kill switch blocks NEW risk, never a close —
    closing reduces risk, so it succeeds while trading is paused."""
    await authorize(client, BULL_TICKER)
    r = await approve(client, BULL_TICKER)
    assert r.status_code == 200

    r = await client.post("/api/trading/pause", json={"reason": "test pause"})
    assert r.status_code == 200

    r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
    assert r.status_code == 200
    assert r.json()["position"]["status"] == "CLOSED"


async def test_close_quantity_exceeding_open_is_422(client):
    await authorize(client, BULL_TICKER)
    r = await approve(client, BULL_TICKER, quantity=5)
    assert r.status_code == 200
    open_qty = r.json()["position"]["quantity"]

    r = await client.post(
        "/api/orders/close", json={"ticker": BULL_TICKER, "quantity": open_qty + 1}
    )
    assert r.status_code == 422


async def test_close_with_no_open_position_is_404(client):
    r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Concurrency: the paper-execution lock (§42; V1 no-pyramiding)
# ---------------------------------------------------------------------------


async def test_execution_paths_serialize_on_the_shared_lock(client):
    """Regression (§42 + V1 no-pyramiding): approve, close and check-exits all
    run their check-then-act sections inside the SHARED paper-execution lock.

    Without it, two rapid approves could both pass the open-position check
    before either committed (double fill = pyramiding, forbidden in V1), a
    duplicate client_order_id could crash into the UNIQUE constraint with a
    500 instead of the §42 replay, and two rapid closes could double-credit
    cash. Holding the lock here must therefore park all three endpoints; on
    release they proceed one at a time against committed state.
    """
    import asyncio

    from apps.gateway.routers.orders import execution_lock

    await authorize(client, BULL_TICKER)
    lock = execution_lock()

    # --- approve waits for the lock; nothing fills while it is held -------
    async with lock:
        task = asyncio.create_task(
            approve(client, BULL_TICKER, client_order_id="lock-key-1")
        )
        await asyncio.sleep(0.05)
        assert not task.done(), "approve must park on the execution lock"
        assert len(await order_rows()) == 0, "no fill may land while locked"
    r = await task
    assert r.status_code == 200
    assert r.json()["order"]["status"] == "FILLED"
    assert len(await order_rows()) == 1

    # --- close and check-exits park on the SAME lock ----------------------
    async with lock:
        t_close = asyncio.create_task(
            client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        )
        t_exits = asyncio.create_task(client.post("/api/positions/check-exits"))
        await asyncio.sleep(0.05)
        assert not t_close.done(), "close must park on the execution lock"
        assert not t_exits.done(), "check-exits must park on the execution lock"
        assert len(await order_rows()) == 1, "no sell may land while locked"
    r_close, r_exits = await t_close, await t_exits
    # A freshly opened position always HOLDs on its entry bar (close is far
    # above stop/trail, bias just passed the BULL entry gate, bars_held 0),
    # so check-exits held it and the manual close then closed it — in either
    # completion order exactly ONE sell fills.
    assert r_exits.status_code == 200
    assert r_close.status_code == 200
    assert r_close.json()["position"]["status"] == "CLOSED"
    rows = await order_rows()
    assert [o.side for o in rows] == ["BUY_TO_OPEN", "SELL_TO_CLOSE"]


async def test_rapid_duplicate_client_order_id_fills_once(client):
    """§42: two RAPID approves with the same client_order_id yield exactly one
    Order row; the second answers 200 with the EXISTING order, never a 500
    and never a second fill."""
    import asyncio

    await authorize(client, BULL_TICKER)
    r1, r2 = await asyncio.gather(
        approve(client, BULL_TICKER, client_order_id="rapid-dup"),
        approve(client, BULL_TICKER, client_order_id="rapid-dup"),
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["order"]["id"] == r2.json()["order"]["id"]
    rows = await order_rows()
    assert len(rows) == 1
    assert rows[0].client_order_id == "rapid-dup"
    assert rows[0].side == "BUY_TO_OPEN"
