"""Automated position monitor (plan §26): the shared sweep core + status API.

The tests drive the app through httpx ASGITransport, which does NOT run the
FastAPI lifespan — the background monitor task therefore NEVER starts here.
That is deliberate: the sweep core (``run_exit_sweep`` /
``run_sweep_and_update``) is tested DIRECTLY, and GET /api/positions/monitor
is tested for honest reporting (enabled=false, null sweep fields) when the
loop is not running (§44 rule 18).

Determinism notes mirror tests/test_position_monitor.py: GOOGL's stub series
yields an approvable BULL entry, and the forced HARD_STOP rewrites the stored
Position row directly (sky-high entry, tiny stop) so no exit depends on
signal luck.
"""
import pytest
from sqlalchemy import select, update

from apps.gateway import monitor
from apps.gateway.db import Order, Position, SessionLocal
from apps.gateway.routers.positions import run_exit_sweep
from libs.common.config import get_settings

BULL_TICKER = "GOOGL"


@pytest.fixture(autouse=True)
def reset_monitor_state():
    """monitor.STATE is a process-global singleton; reset it per test so one
    test's manual sweep cannot leak into another's honest-null assertions."""
    monitor.STATE.enabled = False
    monitor.STATE.interval_seconds = 0
    monitor.STATE.last_sweep_at = None
    monitor.STATE.sweeps_total = 0
    monitor.STATE.last_result = None
    yield


async def authorize(client, ticker):
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


async def open_position(client, ticker=BULL_TICKER):
    await authorize(client, ticker)
    r = await client.post("/api/orders/approve", json={"ticker": ticker})
    assert r.status_code == 200, r.text
    return r.json()


async def sweep_directly():
    """Run the shared sweep core exactly as the background loop does: its own
    session, no HTTP request in flight."""
    async with SessionLocal() as session:
        return await run_exit_sweep(session)


async def test_run_exit_sweep_matches_post_endpoint_result(client):
    """The direct core call and POST /check-exits are the SAME sweep: on a
    static holding book they return identical results in the contract shape."""
    await open_position(client)

    direct = await sweep_directly()
    assert set(direct) == {"checked", "exits_triggered", "held"}
    assert direct["checked"] == 1
    assert direct["exits_triggered"] == []
    assert len(direct["held"]) == 1
    assert direct["held"][0]["ticker"] == BULL_TICKER
    assert any(r.startswith("OK:") for r in direct["held"][0]["reasons"])

    r = await client.post("/api/positions/check-exits")
    assert r.status_code == 200
    assert r.json() == direct  # one sweep implementation, two triggers

    # Holding sweeps create no orders (only the entry BUY exists).
    async with SessionLocal() as s:
        orders = list((await s.execute(select(Order))).scalars().all())
    assert len(orders) == 1 and orders[0].side == "BUY_TO_OPEN"


async def test_run_exit_sweep_forced_hard_stop_full_audit_chain(client):
    """A forced HARD_STOP closed by the DIRECT core call produces the same
    audit chain the POST endpoint produces: EXIT_GENERATED (SYSTEM, rule +
    reasons) and ORDER_REQUESTED (SYSTEM, system_generated) -> SUBMITTED ->
    FILLED, with the position CLOSED."""
    body = await open_position(client)
    position_id = body["position"]["id"]

    # Sky-high entry with a tiny stop guarantees close <= entry - stop.
    async with SessionLocal() as s:
        await s.execute(
            update(Position)
            .where(Position.id == position_id)
            .values(avg_price=1_000_000.0, stop_distance=1.0)
        )
        await s.commit()

    result = await sweep_directly()
    assert result["checked"] == 1
    assert result["held"] == []
    assert len(result["exits_triggered"]) == 1
    hit = result["exits_triggered"][0]
    assert hit["ticker"] == BULL_TICKER
    assert hit["rule"] == "HARD_STOP"
    order_id = hit["order_id"]

    # Position CLOSED, sell order persisted.
    async with SessionLocal() as s:
        pos = await s.get(Position, position_id)
        assert pos.status == "CLOSED" and pos.quantity == 0
        sells = [
            o
            for o in (await s.execute(select(Order))).scalars().all()
            if o.side == "SELL_TO_CLOSE"
        ]
    assert [o.id for o in sells] == [order_id]

    # Same audit chain as the POST path (rule 12).
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
    assert requested["actor_type"] == "SYSTEM"
    assert requested["details"]["system_generated"] is True


async def test_monitor_status_honest_when_lifespan_never_ran(client):
    """ASGITransport never runs lifespan, so no task started: the endpoint
    must report enabled=false with null/zero sweep fields even though the
    configured interval is > 0 (§44 rule 18 — no pretending)."""
    r = await client.get("/api/positions/monitor")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "enabled": False,  # interval > 0 but the task never started
        "interval_seconds": get_settings().position_monitor_interval_seconds,
        "last_sweep_at": None,
        "sweeps_total": 0,
        "last_result": None,
    }
    assert body["interval_seconds"] > 0  # default config would start the loop


async def test_run_sweep_and_update_updates_state_and_telemetry(client):
    """One manual call of the loop's sweep helper records the sweep: STATE,
    the monitor status endpoint, and the telemetry counter all advance."""
    await open_position(client)
    sweeps_before = monitor.POSITION_MONITOR_SWEEPS_TOTAL.value()

    result = await monitor.run_sweep_and_update()
    assert result["checked"] == 1

    assert monitor.STATE.sweeps_total == 1
    assert monitor.STATE.last_sweep_at is not None
    assert monitor.STATE.last_result == {"checked": 1, "exits_triggered": 0}
    assert monitor.POSITION_MONITOR_SWEEPS_TOTAL.value() == sweeps_before + 1

    # The status endpoint serves the updated state — but still reports
    # enabled=false: a manual sweep is not a running background task.
    body = (await client.get("/api/positions/monitor")).json()
    assert body["enabled"] is False
    assert body["sweeps_total"] == 1
    assert body["last_sweep_at"] is not None
    assert body["last_result"] == {"checked": 1, "exits_triggered": 0}
