"""Position monitor tests — GET /api/positions + check-exits (plan §11, §37).

NOTE on determinism (per task spec): positions are opened through the real
POST /api/orders/approve chain using GOOGL — the stub provider is seeded by
the symbol name and deterministically yields a BULL bias in a bull regime
(the same property tests/test_order_preview.py relies on), so these tests do
NOT depend on signal luck for entry. The forced-exit test then drives
HARD_STOP purely through a direct UPDATE of the stored Position row
(sky-high entry, tiny stop), so no EXIT depends on signal luck either. If
the stub ever stops producing an approvable chain for GOOGL, seed the
Portfolio/Position rows directly here instead of approving.
"""
import pytest
from sqlalchemy import select, update

from apps.gateway.db import Order, Position, SessionLocal
from libs.common.config import get_settings

BULL_TICKER = "GOOGL"

EXIT_RULES = {"HARD_STOP", "SIGNAL_FLIP", "SIGNAL_DECAY", "ATR_TRAIL", "TIME_STOP"}


async def authorize(client, ticker):
    r = await client.post("/api/watchlist", json={"ticker": ticker})
    assert r.status_code == 201
    r = await client.post("/api/trading-pool", json={"ticker": ticker})
    assert r.status_code == 201
    r = await client.post(f"/api/trading-pool/{ticker}/trading", json={"enabled": True})
    assert r.status_code == 200
    r = await client.post("/api/trading/resume", json={})
    assert r.status_code == 200


async def open_position(client, ticker=BULL_TICKER):
    await authorize(client, ticker)
    r = await client.post("/api/orders/approve", json={"ticker": ticker})
    assert r.status_code == 200, r.text
    return r.json()


async def cash(client):
    r = await client.get("/api/portfolio/risk")
    return r.json()["cash"]


async def count_orders(side=None):
    async with SessionLocal() as s:
        rows = list((await s.execute(select(Order))).scalars().all())
    return len([o for o in rows if side is None or o.side == side])


async def test_open_row_carries_full_exit_read(client):
    """§37: an OPEN row must show stop_price, entry_edge, an exit_status and
    a NON-EMPTY reason list — "OK:"-prefixed for every rule that is not
    firing — so the user always sees why the system is still holding."""
    body = await open_position(client)

    r = await client.get("/api/positions")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    row = rows[0]

    assert row["status"] == "OPEN"
    assert row["stop_price"] == pytest.approx(body["position"]["stop_price"])
    assert row["stop_price"] > 0
    assert row["entry_edge"] > 0
    assert row["current_price"] is not None
    assert row["market_value"] == pytest.approx(
        row["quantity"] * row["current_price"]
    )
    assert row["bars_held"] == 0  # entered on the latest stored bar (bar 0)

    # Just entered on the current bar: every §11 rule holds, and EVERY rule
    # is reported with real numbers, "OK:"-prefixed (§37/§38).
    assert row["exit_status"] == "HOLD"
    reasons = row["exit_reasons"]
    assert len(reasons) == len(EXIT_RULES)
    assert all(r.startswith("OK: ") for r in reasons)
    named = {r.split(":")[1].strip() for r in reasons}
    assert named == EXIT_RULES


async def test_closed_rows_carry_realized_pnl_and_honest_nulls(client):
    await open_position(client)
    r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
    assert r.status_code == 200

    r = await client.get("/api/positions", params={"status": "CLOSED"})
    rows = r.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "CLOSED"
    assert row["realized_pnl"] is not None
    assert row["closed_at"] is not None
    # Honest nulls (§44 rule 18): no live read on a closed position.
    for key in (
        "current_price",
        "market_value",
        "unrealized_pnl",
        "current_edge",
        "exit_status",
    ):
        assert row[key] is None
    assert row["exit_reasons"] == []
    # The default filter (OPEN) no longer lists it.
    r = await client.get("/api/positions")
    assert r.json() == []


async def test_check_exits_healthy_position_holds_with_reasons(client):
    """A healthy just-entered position is HELD with the full reason list and
    no sell order is created."""
    await open_position(client)
    assert await count_orders() == 1  # just the BUY_TO_OPEN

    r = await client.post("/api/positions/check-exits")
    assert r.status_code == 200
    body = r.json()
    assert body["checked"] == 1
    assert body["exits_triggered"] == []
    assert len(body["held"]) == 1
    assert body["held"][0]["ticker"] == BULL_TICKER
    assert body["held"][0]["reasons"]  # §37: why the system is still holding
    assert any(reason.startswith("OK:") for reason in body["held"][0]["reasons"])

    assert await count_orders() == 1  # no order was produced


async def test_check_exits_forced_hard_stop_closes_position(client):
    """Force HARD_STOP by rewriting the stored position (sky-high entry with
    a tiny stop -> close is far below the stop): check-exits must create a
    SELL_TO_CLOSE order, CLOSE the position, audit EXIT_GENERATED and
    ORDER_FILLED, and credit cash — while trading is PAUSED, because
    mechanical exits are never blocked by the kill switch (§18)."""
    body = await open_position(client)
    position_id = body["position"]["id"]
    qty = body["position"]["quantity"]

    r = await client.get("/api/positions")
    last_close = r.json()[0]["current_price"]
    cash_before = await cash(client)

    # Direct UPDATE (test-only): entry far above the market with a tiny stop
    # distance guarantees close <= entry - stop -> HARD_STOP fires.
    async with SessionLocal() as s:
        await s.execute(
            update(Position)
            .where(Position.id == position_id)
            .values(avg_price=1_000_000.0, stop_distance=1.0)
        )
        await s.commit()

    # Kill switch engaged: mechanical exits must still run (§18 risk-priority).
    r = await client.post("/api/trading/pause", json={"reason": "forced exit test"})
    assert r.status_code == 200

    r = await client.post("/api/positions/check-exits")
    assert r.status_code == 200
    body = r.json()
    assert body["checked"] == 1
    assert body["held"] == []
    assert len(body["exits_triggered"]) == 1
    hit = body["exits_triggered"][0]
    assert hit["ticker"] == BULL_TICKER
    assert hit["rule"] == "HARD_STOP"
    order_id = hit["order_id"]

    # The position is CLOSED and the sell order exists.
    r = await client.get("/api/positions", params={"status": "CLOSED"})
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == position_id
    assert rows[0]["quantity"] == 0
    assert await count_orders(side="SELL_TO_CLOSE") == 1

    # Audit: EXIT_GENERATED (SYSTEM, with rule + reasons) on the position,
    # and the SYSTEM-requested order chain through ORDER_FILLED (rule 12).
    r = await client.get("/api/audit", params={"entity_id": str(position_id)})
    exit_events = [e for e in r.json() if e["action"] == "EXIT_GENERATED"]
    assert len(exit_events) == 1
    assert exit_events[0]["actor_type"] == "SYSTEM"
    assert exit_events[0]["details"]["rule"] == "HARD_STOP"
    assert exit_events[0]["details"]["reasons"]

    r = await client.get("/api/audit", params={"entity_id": str(order_id)})
    actions = [e["action"] for e in reversed(r.json())]
    assert actions == ["ORDER_REQUESTED", "ORDER_SUBMITTED", "ORDER_FILLED"]
    requested = [e for e in r.json() if e["action"] == "ORDER_REQUESTED"][0]
    assert requested["actor_type"] == "SYSTEM"  # system_generated exit
    assert requested["details"]["system_generated"] is True

    # Cash credited with the sell proceeds (paper fill model, §11).
    settings = get_settings()
    sell_fill = last_close * (1.0 - settings.paper_slippage_bps / 10000.0)
    commission = settings.paper_commission_per_share * qty
    assert await cash(client) == pytest.approx(
        cash_before + qty * sell_fill - commission
    )
