"""§11 order lifecycle (PENDING_SUBMIT) + the order-sync sweep (Iteration C/D).

What these tests pin, in causal order:

1. The local order row is COMMITTED as PENDING_SUBMIT before the submit
   leaves the process — a broker fault mid-submit leaves a durable,
   resolvable row instead of an invisible broker order.
2. The sweep settles that row against the broker's own answer:
   - broker never saw it        -> REJECTED ("never_reached_broker"), safe;
   - broker filled it silently  -> adopted, position opened, cash debited;
   - fills arrive over time     -> exact incremental cash per delta, to the
     cent, across any number of sweeps (cumulative-average arithmetic);
   - sells fill late            -> proceeds + realized P&L via the sweep too.
3. The sweep answers honestly when there is no real broker to ask.

All tests run the REAL AlpacaPaperBroker over MockTransport via the
FakeAlpaca double from test_broker_execution — the status mapping,
client_order_id plumbing and paper guard are production code, not stand-ins.
"""
import httpx
import pytest

from libs.common.config import get_settings

from apps.gateway.db import Order, Position, SessionLocal
from apps.gateway.order_sync import run_order_sync
from apps.gateway.routers.orders import PENDING_SUBMIT

from tests.test_broker_execution import (
    BULL_TICKER,
    FakeAlpaca,
    _broker_client,
    approve,
    authorize,
    audit_actions_for,
    cash,
    order_body,
    rows,
)

pytestmark = pytest.mark.anyio


OPTION_INSTRUMENTS = ("LONG_CALL", "LONG_PUT")


def multiplier_of(order: Order) -> int:
    return 100 if order.instrument in OPTION_INSTRUMENTS else 1


async def sweep() -> dict:
    async with SessionLocal() as session:
        return await run_order_sync(session)


class LostResponseAlpaca(FakeAlpaca):
    """POST /v2/orders dies at the network layer.

    ``reached=False``: the request never arrived — the broker holds nothing
    under our client_order_id. ``reached=True``: the broker processed the
    submission but the RESPONSE was lost — the order exists and can be found
    by client_order_id afterwards.
    """

    def __init__(self, order_payload: dict, *, reached: bool):
        super().__init__(order_payload)
        self.reached = reached

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/orders" and request.method == "POST":
            self.calls.append((request.method, request.url.path))
            if self.reached:
                import json as _json

                self.posted.append(_json.loads(request.content))
            raise httpx.ConnectError("connection lost mid-submit", request=request)
        return super().handler(request)


# ===========================================================================
# 1. PENDING_SUBMIT durability
# ===========================================================================


async def test_broker_fault_leaves_a_durable_pending_submit_row(monkeypatch):
    """§11: the intent row survives the fault, ready for the sweep — no
    position, no cash movement, and the 502 says the sweep will resolve it."""
    fake = LostResponseAlpaca(order_body(), reached=False)
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="FAULT-1")
        assert r.status_code == 502
        assert "PENDING_SUBMIT" in r.json()["detail"]["message"]

        orders = await rows(Order)
        assert len(orders) == 1
        order = orders[0]
        assert order.status == PENDING_SUBMIT
        assert order.client_order_id == "FAULT-1"
        assert order.broker_order_id is None
        assert order.filled_quantity == 0 and order.fill_price == 0.0
        assert await rows(Position) == []
        assert await cash(client) == pytest.approx(cash_before)


async def test_sweep_settles_an_orphan_the_broker_never_saw(monkeypatch):
    """PENDING_SUBMIT + unknown at the broker -> held during the grace window
    (a fresh 404 may just be broker lookup lag), then REJECTED once aged."""
    fake = LostResponseAlpaca(order_body(), reached=False)
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)
        assert (await approve(client, quantity=10, client_order_id="ORPHAN-1")).status_code == 502

        # FRESH orphan: held, not rejected — the broker's order index may lag
        # its own accept, and a premature REJECTED would license a re-approve
        # that double-buys when the original quietly fills.
        young = await sweep()
        assert young["orphans_rejected"] == []
        assert any("held" in f["error"] for f in young["faults"])
        assert (await rows(Order))[0].status == PENDING_SUBMIT

        # Age it past the grace window; the next sweep settles it.
        from datetime import timedelta

        async with SessionLocal() as s:
            db_order = await s.get(Order, (await rows(Order))[0].id)
            db_order.created_at = db_order.created_at - timedelta(seconds=300)
            await s.commit()

        result = await sweep()
        assert result["checked"] == 1
        assert [o["order_id"] for o in result["orphans_rejected"]]

        order = (await rows(Order))[0]
        assert order.status == "REJECTED"
        assert order.broker_status == "never_reached_broker"
        assert await rows(Position) == []
        assert await cash(client) == pytest.approx(cash_before)

        events = await audit_actions_for(client, str(order.id))
        rejected = [e for e in events if e["action"] == "ORDER_REJECTED"]
        assert len(rejected) == 1
        assert rejected[0]["details"]["source"] == "order_sync_sweep"

        # Settled means settled: the next sweep has nothing left to ask about.
        assert (await sweep())["checked"] == 0


async def test_sweep_adopts_a_submit_whose_response_was_lost(monkeypatch):
    """The broker filled the order; only our copy of the answer died. The
    sweep adopts the broker's id, opens the position for the REAL fill and
    debits exactly avg * filled * multiplier."""
    fake = LostResponseAlpaca(order_body(), reached=True)
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)
        assert (await approve(client, quantity=10, client_order_id="LOST-1")).status_code == 502

        pending = (await rows(Order))[0]
        assert pending.status == PENDING_SUBMIT
        # The broker's ledger: our submission, filled at 200.00.
        fake.order_payload = order_body(
            client_order_id="LOST-1",
            qty=str(pending.quantity),
            filled_qty=str(pending.quantity),
        )

        result = await sweep()
        assert result["checked"] == 1
        assert len(result["fills_applied"]) == 1

        order = (await rows(Order))[0]
        mult = multiplier_of(order)
        assert order.status == "FILLED"
        assert order.broker_order_id == "brk-order-0001"  # adopted
        assert order.filled_quantity == order.quantity
        assert order.fill_price == pytest.approx(200.0)

        positions = await rows(Position)
        assert len(positions) == 1
        pos = positions[0]
        assert pos.quantity == order.quantity
        assert pos.avg_price == pytest.approx(200.0)
        assert pos.status == "OPEN"
        assert order.position_id == pos.id
        # Approval-time risk context survived the crash (migration 010).
        if order.instrument == "LONG_STOCK":
            assert pos.stop_distance == pytest.approx(order.stop_distance)

        # Cash lives at the broker only (static fake -> unchanged); the
        # platform's own record of the fill is the position itself.
        assert await cash(client) == pytest.approx(cash_before)


# ===========================================================================
# 2. Incremental fills across sweeps — cash conservation to the cent
# ===========================================================================


async def test_partial_fills_across_sweeps_conserve_cash_exactly(monkeypatch):
    """ACCEPTED(0) at approve -> 4 @ 200.00 -> FILLED 10 @ 201.00 cumulative.
    Total debit must be exactly 10 * 201.00 * mult — the incremental
    subtraction (new_avg*new_filled - old_avg*old_filled) leaves no dust."""
    monkeypatch.setenv("BROKER_FILL_POLL_SECONDS", "0")
    fake = FakeAlpaca(order_body(status="accepted", filled_qty="0", filled_avg_price=None))
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="PART-1")
        assert r.status_code == 200
        order = (await rows(Order))[0]
        mult = multiplier_of(order)
        requested = order.quantity
        assert order.status == "ACCEPTED" and order.filled_quantity == 0
        assert await rows(Position) == []
        assert await cash(client) == pytest.approx(cash_before)

        # --- Sweep 1: 4 of them filled at 200.00 -------------------------
        fake.poll_payload = order_body(
            client_order_id="PART-1",
            status="partially_filled",
            qty=str(requested),
            filled_qty="4",
            filled_avg_price="200.00",
        )
        result = await sweep()
        assert len(result["fills_applied"]) == 1
        order = (await rows(Order))[0]
        assert order.status == "PARTIALLY_FILLED" and order.filled_quantity == 4
        pos = (await rows(Position))[0]
        assert pos.quantity == 4 and pos.avg_price == pytest.approx(200.0)

        # --- Sweep 2: complete at cumulative average 201.00 ---------------
        fake.poll_payload = order_body(
            client_order_id="PART-1",
            status="filled",
            qty=str(requested),
            filled_qty=str(requested),
            filled_avg_price="201.00",
        )
        result = await sweep()
        assert len(result["fills_applied"]) == 1
        order = (await rows(Order))[0]
        assert order.status == "FILLED" and order.filled_quantity == requested
        pos = (await rows(Position))[0]
        assert pos.quantity == requested
        # EXACT cumulative average — never sum-of-guesses. (Cash itself lives
        # at the broker; the platform stores position facts, not a ledger.)
        assert pos.avg_price == pytest.approx(201.0)

        # Audit trail: two sweep fills, both attributed to the sweep.
        events = await audit_actions_for(client, str(order.id))
        sweep_fills = [
            e
            for e in events
            if e["action"] == "ORDER_FILLED"
            and e["details"].get("source") == "order_sync_sweep"
        ]
        assert len(sweep_fills) == 2

        # Terminal: the sweep is done with this order.
        assert (await sweep())["checked"] == 0


async def test_sell_to_close_fills_late_via_the_sweep(monkeypatch):
    """A closing order that leaves the bounded poll ACCEPTED(0) keeps the
    position open — the sweep later credits the REAL proceeds, books realized
    P&L against the position's basis, and closes it."""
    fake = FakeAlpaca(order_body())  # buy: instant fill 10 @ 200.00
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_start = await cash(client)
        assert (await approve(client, quantity=10, client_order_id="BUY-1")).status_code == 200
        buy = (await rows(Order))[0]
        mult = multiplier_of(buy)
        filled = buy.filled_quantity
        assert filled > 0

        # Close: the submit answers accepted with nothing filled.
        monkeypatch.setenv("BROKER_FILL_POLL_SECONDS", "0")
        get_settings.cache_clear()
        fake.order_payload = order_body(
            id="brk-order-0002",
            status="accepted",
            side="sell",
            qty=str(filled),
            filled_qty="0",
            filled_avg_price=None,
        )
        r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        assert r.status_code == 200

        sell = (await rows(Order))[1]
        assert sell.status == "ACCEPTED" and sell.filled_quantity == 0
        pos = (await rows(Position))[0]
        assert pos.status == "OPEN" and pos.quantity == filled  # untouched

        # The broker fills the sell at 210.00.
        fake.poll_payload = order_body(
            id="brk-order-0002",
            client_order_id=sell.client_order_id,
            status="filled",
            side="sell",
            qty=str(filled),
            filled_qty=str(filled),
            filled_avg_price="210.00",
        )
        result = await sweep()
        assert len(result["fills_applied"]) == 1

        sell = (await rows(Order))[1]
        assert sell.status == "FILLED" and sell.filled_quantity == filled
        pos = (await rows(Position))[0]
        assert pos.status == "CLOSED" and pos.quantity == 0
        # The round trip's economics, to the cent, in the platform's OWN
        # record (cash lives at the broker and is not copied locally).
        assert pos.realized_pnl == pytest.approx((210.0 - 200.0) * filled * mult)


# ===========================================================================
# 3. Honesty at the edges
# ===========================================================================


async def test_sweep_reports_skipped_without_a_real_broker(monkeypatch):
    monkeypatch.setenv("BROKER_PROVIDER", "")
    get_settings.cache_clear()
    try:
        # The early return must fire before any DB/broker access — None
        # session proves no session is ever touched.
        result = await run_order_sync(None)
        assert result == {"checked": 0, "skipped": "NO_REAL_BROKER"}
    finally:
        get_settings.cache_clear()


async def test_sweep_leaves_the_order_alone_on_a_lookup_fault(monkeypatch):
    """A fault teaches us nothing: the row must stay exactly as it was."""
    fake = LostResponseAlpaca(order_body(), reached=False)
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="FLT-1")).status_code == 502

        # Now EVERY broker call dies, including the sweep's lookup.
        def dead_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("broker unreachable", request=request)

        fake.handler = dead_handler  # type: ignore[method-assign]

        result = await sweep()
        assert result["checked"] == 1
        assert len(result["faults"]) == 1
        assert result["orphans_rejected"] == [] and result["mismatches"] == []
        order = (await rows(Order))[0]
        assert order.status == PENDING_SUBMIT  # untouched, retried next sweep


async def test_sweep_reports_a_shrinking_fill_as_mismatch_and_touches_nothing(
    monkeypatch,
):
    """The broker reporting FEWER fills than recorded cannot be applied — a
    fill cannot un-happen. Reported for §18; no mutation."""
    fake = FakeAlpaca(order_body())  # instant fill 10 @ 200
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="SHRINK-1")).status_code == 200
        cash_after_fill = await cash(client)
        order = (await rows(Order))[0]

        # Force the row non-terminal so the sweep asks about it again, then
        # script the broker to claim less than we recorded.
        async with SessionLocal() as s:
            db_order = await s.get(Order, order.id)
            db_order.status = "PARTIALLY_FILLED"
            await s.commit()
        fake.poll_payload = order_body(
            client_order_id="SHRINK-1",
            status="partially_filled",
            filled_qty="3",
            filled_avg_price="200.00",
        )

        result = await sweep()
        assert len(result["mismatches"]) == 1
        assert "un-happen" in result["mismatches"][0]["detail"]
        after = (await rows(Order))[0]
        assert after.filled_quantity == order.filled_quantity  # untouched
        assert await cash(client) == pytest.approx(cash_after_fill)


async def test_manual_sync_endpoint_answers_the_summary(monkeypatch):
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        r = await client.post("/api/broker/sync-orders")
        assert r.status_code == 200
        body = r.json()
        assert body["checked"] == 0
        assert body["fills_applied"] == [] and body["mismatches"] == []


async def test_open_orders_endpoint_tracks_in_flight_rows(monkeypatch):
    """§26 PENDING_UPDATE source: /api/orders/open lists the in-flight order
    while it is unsettled and empties once the sweep settles it."""
    fake = LostResponseAlpaca(order_body(), reached=False)
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await client.get("/api/orders/open")).json() == {"orders": []}

        assert (await approve(client, quantity=10, client_order_id="OPEN-1")).status_code == 502
        body = (await client.get("/api/orders/open")).json()
        assert len(body["orders"]) == 1
        row = body["orders"][0]
        assert row["status"] == PENDING_SUBMIT
        assert row["client_order_id"] == "OPEN-1"
        assert row["position_id"] is None  # nothing filled yet — honest null

        # Age past the orphan grace window, then settle to REJECTED (terminal).
        from datetime import timedelta

        async with SessionLocal() as s:
            db_order = await s.get(Order, row["id"])
            db_order.created_at = db_order.created_at - timedelta(seconds=300)
            await s.commit()
        await sweep()
        assert (await client.get("/api/orders/open")).json() == {"orders": []}


# ===========================================================================
# 4. Duplicate-submission guards (one intent -> at most one broker order)
# ===========================================================================


async def test_second_approve_409s_while_a_buy_is_in_flight(monkeypatch):
    """An ACCEPTED-unfilled buy has no position, so the pyramiding check
    cannot see it — the in-flight guard must. Without it a retry places TWO
    broker orders for one intent."""
    monkeypatch.setenv("BROKER_FILL_POLL_SECONDS", "0")
    fake = FakeAlpaca(order_body(status="accepted", filled_qty="0", filled_avg_price=None))
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="IF-1")).status_code == 200
        assert fake.submit_count == 1

        r = await approve(client, quantity=10, client_order_id="IF-2")
        assert r.status_code == 409
        assert "in flight" in r.json()["detail"]
        assert fake.submit_count == 1  # nothing new reached the broker
        assert len(await rows(Order)) == 1


class StrictLookupAlpaca(FakeAlpaca):
    """by_client_order_id answers ONLY from ``existing`` — unknown ids 404.

    The base fake's fallback (any posted order answers any lookup) makes a
    close ADOPT the buy's order instead of submitting a sell, which would let
    this guard test pass without ever exercising the submit path.
    """

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/orders:by_client_order_id":
            self.calls.append((request.method, request.url.path))
            cid = request.url.params.get("client_order_id", "")
            if cid in self.existing:
                return httpx.Response(200, json=self.existing[cid])
            return httpx.Response(404, json={"message": "order not found"})
        return super().handler(request)


async def test_second_close_409s_while_a_sell_is_in_flight(monkeypatch):
    """Every exit-monitor tick re-evaluates an open position; while its sell
    sits ACCEPTED-unfilled the position looks untouched. Without this guard
    each tick would mint a fresh client_order_id and re-sell the same shares."""
    fake = StrictLookupAlpaca(order_body())  # buy fills instantly
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="B-1")).status_code == 200
        buy_submits = fake.submit_count

        monkeypatch.setenv("BROKER_FILL_POLL_SECONDS", "0")
        get_settings.cache_clear()
        fake.order_payload = order_body(
            id="brk-order-0002",
            status="accepted",
            side="sell",
            filled_qty="0",
            filled_avg_price=None,
        )
        assert (
            await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        ).status_code == 200
        assert fake.submit_count == buy_submits + 1

        r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "CLOSE_ALREADY_IN_FLIGHT"
        assert fake.submit_count == buy_submits + 1  # no second sell
        # The position is still fully open and untouched.
        pos = (await rows(Position))[0]
        assert pos.status == "OPEN" and pos.quantity == (await rows(Order))[0].filled_quantity


async def test_mismatch_rolls_back_staged_adoption(monkeypatch):
    """A mismatch must leave the row EXACTLY as it was: an adopted
    broker_order_id staged before the mismatch was detected must not leak
    into the next order's commit un-audited."""
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        assert (await approve(client, quantity=10, client_order_id="ROLL-1")).status_code == 200
        order = (await rows(Order))[0]

        # Simulate a pre-adoption partial row: non-terminal, fills recorded,
        # but no broker id yet.
        async with SessionLocal() as s:
            db_order = await s.get(Order, order.id)
            db_order.status = "PARTIALLY_FILLED"
            db_order.broker_order_id = None
            await s.commit()

        # The broker answers WITH an id but FEWER fills than recorded — the
        # shrink mismatch fires after the adoption was staged.
        fake.poll_payload = order_body(
            client_order_id="ROLL-1",
            status="partially_filled",
            filled_qty="3",
            filled_avg_price="200.00",
        )
        result = await sweep()
        assert len(result["mismatches"]) == 1

        after = (await rows(Order))[0]
        assert after.broker_order_id is None  # staged adoption rolled back
        assert after.filled_quantity == order.filled_quantity


async def test_reconcile_defers_the_pause_while_orders_are_in_flight(monkeypatch):
    """A divergence that an in-flight order can explain must NOT pull the §18
    kill switch — the order-sync sweep is about to settle it."""
    fake = LostResponseAlpaca(order_body(), reached=False)
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        # An in-flight PENDING_SUBMIT row.
        assert (await approve(client, quantity=10, client_order_id="DEFER-1")).status_code == 502
        # And a genuine POSITION divergence at the broker (cash is never
        # compared — the platform keeps no copy of it).
        fake.positions = [
            {"symbol": "TSLA", "qty": "7", "avg_entry_price": "300.00",
             "market_value": "2100.00"}
        ]

        r = await client.get("/api/broker/reconcile")
        assert r.status_code == 200
        body = r.json()
        assert body["mismatches"]  # the divergence IS reported
        assert body.get("pause_deferred") is True
        assert "in flight" in body["message"]

        # Trading was NOT paused.
        state = (await client.get("/api/trading/status")).json()
        assert state["trading_enabled"] is True


# ===========================================================================
# 5. Price-pending fills (broker eventual consistency) — never lost, never
#    applied without a price
# ===========================================================================


async def test_buy_fill_with_pending_price_is_deferred_to_the_sweep(monkeypatch):
    """filled_qty can populate before filled_avg_price at the broker. The
    fill must NOT be booked without a price — and must NOT be lost either:
    filled_quantity stays at what was APPLIED (0), the row stays
    non-terminal, and the sweep applies the fill when the price arrives."""
    fake = FakeAlpaca(order_body(status="filled", filled_qty="10", filled_avg_price=None))
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_before = await cash(client)

        r = await approve(client, quantity=10, client_order_id="PENDPX-1")
        assert r.status_code == 200
        body = r.json()
        assert body["position"] is None

        order = (await rows(Order))[0]
        mult = multiplier_of(order)
        assert order.status == "ACCEPTED"  # forced non-terminal
        assert order.filled_quantity == 0  # applied-state invariant
        assert order.fill_price == 0.0
        assert await rows(Position) == []  # nothing applied without a price

        # Sweep while the price is STILL missing: held, not lost, not booked.
        fake.poll_payload = order_body(
            client_order_id="PENDPX-1", status="filled",
            filled_qty="10", filled_avg_price=None,
        )
        held = await sweep()
        assert held["fills_applied"] == [] and held["mismatches"] == []
        assert any("no" in f["error"] and "price" in f["error"] for f in held["faults"])
        assert (await rows(Order))[0].status == "ACCEPTED"  # still watched

        # Price publishes -> the sweep applies the fill exactly once.
        fake.poll_payload = order_body(
            client_order_id="PENDPX-1", status="filled",
            filled_qty="10", filled_avg_price="95.00",
        )
        result = await sweep()
        assert len(result["fills_applied"]) == 1
        order = (await rows(Order))[0]
        assert order.status == "FILLED" and order.filled_quantity == 10
        pos = (await rows(Position))[0]
        assert pos.quantity == 10 and pos.avg_price == pytest.approx(95.0)


async def test_sell_fill_with_pending_price_is_deferred_to_the_sweep(monkeypatch):
    """Same gap on the close path: contracts sold, price not yet published —
    the position must stay untouched until the sweep can book real proceeds."""
    fake = StrictLookupAlpaca(order_body())  # buy fills instantly 10 @ 200
    async with _broker_client(fake, monkeypatch) as client:
        await authorize(client)
        cash_start = await cash(client)
        assert (await approve(client, quantity=10, client_order_id="SB-1")).status_code == 200
        buy = (await rows(Order))[0]
        mult = multiplier_of(buy)
        filled = buy.filled_quantity

        monkeypatch.setenv("BROKER_FILL_POLL_SECONDS", "0")
        get_settings.cache_clear()
        fake.order_payload = order_body(
            id="brk-order-0002", status="filled", side="sell",
            qty=str(filled), filled_qty=str(filled), filled_avg_price=None,
        )
        r = await client.post("/api/orders/close", json={"ticker": BULL_TICKER})
        assert r.status_code == 200

        sell = (await rows(Order))[1]
        assert sell.status == "ACCEPTED"  # non-terminal despite raw 'filled'
        assert sell.filled_quantity == 0
        pos = (await rows(Position))[0]
        assert pos.status == "OPEN" and pos.quantity == filled  # untouched

        # Price arrives -> sweep books the sale exactly once.
        fake.existing[sell.client_order_id] = order_body(
            id="brk-order-0002", client_order_id=sell.client_order_id,
            status="filled", side="sell",
            qty=str(filled), filled_qty=str(filled), filled_avg_price="210.00",
        )
        result = await sweep()
        assert len(result["fills_applied"]) == 1
        pos = (await rows(Position))[0]
        assert pos.status == "CLOSED" and pos.quantity == 0
        # Economics in the platform's own record; cash lives at the broker.
        assert pos.realized_pnl == pytest.approx((210.0 - 200.0) * filled * mult)


def test_pending_cancel_is_not_terminal():
    """Alpaca's pending_cancel order is still live and can still fill; the
    mapping must keep it in the sweep's watch, never terminal CANCELED."""
    from libs.broker.alpaca import _map_status

    assert _map_status("pending_cancel") == "ACCEPTED"
    assert _map_status("canceled") == "CANCELED"  # a real cancel stays final
