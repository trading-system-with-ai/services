"""Order-sync sweep — settle non-terminal broker orders (guide §11/§12, Iter C).

WHY THIS EXISTS. Three windows leave a local order row that no longer matches
the broker:

1. ``PENDING_SUBMIT`` with no broker id — the process (or the broker call)
   died between committing the intent row and learning the submit's outcome.
   The order may or may not exist at the broker; only the broker knows.
2. ``ACCEPTED`` — the bounded post-submit poll ended before anything filled.
   Whatever fills later is invisible until someone asks again.
3. ``PARTIALLY_FILLED`` — same, for the remainder.

This module is the "someone asks again": every sweep looks each such order up
at the broker BY OUR ``client_order_id`` (idempotency key, §42) and applies
exactly what the broker reports — fill deltas move real position quantity
(cash moves only in the REAL account at the broker; the platform keeps no
copy of it), a vanished order is settled REJECTED, and a broker that
reports LESS than we already recorded is a mismatch handed to the §18
reconciliation, never "fixed" by guessing.

HONESTY RULES (§28):
- A broker FAULT during lookup teaches us nothing: the order is left exactly
  as it was and retried next sweep. Faults are counted, not acted on.
- Fill deltas use the broker's cumulative ``filled_avg_price`` — incremental
  cost/proceeds are derived exactly, so cash conservation holds to the cent
  across any number of partial-fill sweeps.
- The sweep NEVER creates orders and NEVER cancels anything. It records.

CONCURRENCY: the sweep takes the shared paper-execution lock, so it can never
race a concurrent approve/close/exit-sweep into a double-applied fill. Two
deliberate tradeoffs, documented rather than hidden:

- The lock is a PER-PROCESS asyncio lock (§42 note in routers/orders.py) — the
  deployment is one gateway process by design, and the UNIQUE constraints on
  client_order_id / broker_order_id are the multi-process backstop. Scaling to
  multiple gateway processes requires a DB-level advisory lock here first.
- The lock is held across one broker round-trip per non-terminal order, so a
  protective exit can wait ~N×RTT behind a busy sweep. Correctness wins: the
  alternative (lookups outside the lock, apply inside) reintroduces exactly
  the stale-read/double-apply race this lock exists to close. N is small by
  construction — the in-flight guards allow at most one open order per
  ticker/position.

Runs two ways: the background loop in main.py (real-broker mode only) and the
manual ``POST /api/broker/sync-orders``. Both call :func:`run_order_sync`.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select

from libs.broker.provider import BrokerError, BrokerOrder
from libs.common.config import get_settings
from libs.trading_core.models import ActorType, AuditAction, InstrumentType

from . import audit
from .db import Order, Position, SessionLocal, utcnow
from .deps import broker_configured, resolve_broker, simulated_broker_mode
from .execution.gate_chain import (
    PENDING_SUBMIT,
    POSITION_CLOSED,
    POSITION_OPEN,
    SELL_TO_CLOSE,
    execution_lock,
)

logger = logging.getLogger(__name__)

# Statuses the sweep still has questions about. Everything else is terminal:
# the broker has given its final answer and the row already records it.
NON_TERMINAL_STATUSES = (PENDING_SUBMIT, "ACCEPTED", "PARTIALLY_FILLED")

# How old a PENDING_SUBMIT row must be before "the broker does not know this
# client_order_id" is believed to mean "the submit never happened" rather
# than "the broker's order index is lagging its own accept". Two sweep
# cadences (default 30s) plus slack: transient lookup lag is gone by then,
# and holding an orphan a couple of minutes costs nothing — the in-flight
# guard in the approve path blocks a duplicate submission either way.
ORPHAN_GRACE_SECONDS = 120.0

_OPTION_INSTRUMENTS = (InstrumentType.LONG_CALL.value, InstrumentType.LONG_PUT.value)


@dataclass
class OrderSyncState:
    """Observability for the background loop (mirrors monitor.STATE)."""

    enabled: bool = False
    interval_seconds: float = 0.0
    sweeps_total: int = 0
    last_sweep_at: datetime | None = None
    last_result: dict = field(default_factory=dict)


STATE = OrderSyncState()


async def run_order_sync(session) -> dict:
    """One order-sync sweep. Returns a truthful summary dict.

    ``{"checked", "settled", "fills_applied", "orphans_rejected",
    "mismatches", "faults"}`` — plus ``"skipped"`` when there is no real
    broker to ask (unset or simulated: simulated fills settle synchronously
    and never leave a non-terminal row with a client_order_id).

    Commits after EACH order it settles: every order's update is
    self-contained (row + position/cash + audits, rule 12), and a fault on
    order N must not roll back the truth already learned about orders 1..N-1.
    """
    if not broker_configured() or simulated_broker_mode():
        return {"checked": 0, "skipped": "NO_REAL_BROKER"}

    broker = resolve_broker()
    async with execution_lock():
        rows = (
            (
                await session.execute(
                    select(Order)
                    .where(
                        Order.status.in_(NON_TERMINAL_STATUSES),
                        Order.client_order_id.is_not(None),
                    )
                    .order_by(Order.id)
                )
            )
            .scalars()
            .all()
        )

        settled: list[dict] = []
        fills_applied: list[dict] = []
        orphans_rejected: list[dict] = []
        mismatches: list[dict] = []
        faults: list[dict] = []

        for order in rows:
            try:
                broker_order = await asyncio.to_thread(
                    broker.get_order, order.client_order_id
                )
            except BrokerError as exc:
                # A fault teaches us NOTHING about the order. Leave it be.
                logger.warning(
                    "order_sync_lookup_failed",
                    extra={
                        "extra_fields": {
                            "order_id": order.id,
                            "client_order_id": order.client_order_id,
                            "error": str(exc),
                        }
                    },
                )
                faults.append({"order_id": order.id, "error": str(exc)})
                continue

            if broker_order is None:
                if order.status == PENDING_SUBMIT and order.broker_order_id is None:
                    # The submit never reached the broker — PROBABLY. A lookup
                    # made moments after a lost-response submit can also 404
                    # transiently (broker-side eventual consistency), and
                    # declaring REJECTED then would license a re-approve while
                    # the original order quietly fills: the exact double-buy
                    # this module exists to prevent. So an orphan must AGE
                    # past the grace window (over multiple sweeps) before it
                    # is settled; until then it is left PENDING_SUBMIT.
                    created_at = order.created_at
                    if created_at.tzinfo is None:
                        # SQLite (tests) drops tzinfo; the column is UTC.
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    age = (
                        datetime.now(timezone.utc) - created_at
                    ).total_seconds()
                    if age < ORPHAN_GRACE_SECONDS:
                        faults.append(
                            {
                                "order_id": order.id,
                                "error": (
                                    f"orphan candidate is {age:.0f}s old — "
                                    f"held until {ORPHAN_GRACE_SECONDS:.0f}s "
                                    "in case the broker lookup is lagging "
                                    "its own submit"
                                ),
                            }
                        )
                        continue
                    # Aged past the grace window and STILL unknown at the
                    # broker: the submit never happened. REJECTED is the
                    # honest terminal state — nothing was ever live.
                    await _settle_orphan(session, order)
                    orphans_rejected.append(
                        {"order_id": order.id, "ticker": order.ticker}
                    )
                    await session.commit()
                else:
                    # We HAVE seen this order at the broker before (it has a
                    # broker id or got past submit) and now the broker claims
                    # no such client_order_id. That is a ledger disagreement,
                    # §18's territory — report, never guess.
                    mismatches.append(
                        {
                            "order_id": order.id,
                            "detail": (
                                f"order {order.id} ({order.client_order_id}) is "
                                f"{order.status} locally with broker id "
                                f"{order.broker_order_id!r}, but the broker no "
                                "longer returns it by client_order_id"
                            ),
                        }
                    )
                continue

            outcome = await _apply_broker_state(session, order, broker_order)
            if outcome.get("mismatch") or outcome.get("retry"):
                if outcome.get("mismatch"):
                    mismatches.append(outcome["mismatch"])
                else:
                    faults.append(outcome["retry"])
                # Discard anything staged before the early return (e.g. an
                # adopted broker_order_id): without this rollback the NEXT
                # order's commit would silently carry it, un-audited.
                await session.rollback()
                continue
            if outcome.get("fill"):
                fills_applied.append(outcome["fill"])
            if outcome.get("settled"):
                settled.append(outcome["settled"])
            if outcome.get("changed"):
                await session.commit()

    result = {
        "checked": len(rows),
        "settled": settled,
        "fills_applied": fills_applied,
        "orphans_rejected": orphans_rejected,
        "mismatches": mismatches,
        "faults": faults,
    }
    if mismatches:
        # The sweep itself does not pause trading — reconciliation owns the
        # §18 kill switch and compares WHOLE ledgers, not single orders. But a
        # mismatch is never silent.
        logger.error(
            "order_sync_mismatches",
            extra={"extra_fields": {"mismatches": mismatches}},
        )
    return result


async def _settle_orphan(session, order: Order) -> None:
    """PENDING_SUBMIT + unknown at the broker -> REJECTED (never submitted)."""
    order.status = "REJECTED"
    order.broker_status = "never_reached_broker"
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_REJECTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "source": "order_sync_sweep",
            "ticker": order.ticker,
            "client_order_id": order.client_order_id,
            "reason": (
                "the broker holds no order under this client_order_id: the "
                "submit never reached it. Settled REJECTED — no position, no "
                "cash movement, safe to re-approve."
            ),
            "rejected_by": "order_sync_sweep",
        },
    )


async def _apply_broker_state(
    session, order: Order, broker_order: BrokerOrder
) -> dict:
    """Apply one broker order snapshot to its local row (+position/cash).

    Returns ``{"changed": bool, "fill": ..., "settled": ..., "mismatch": ...}``
    (keys present only when that thing happened). On mismatch NOTHING is
    mutated.
    """
    new_filled = broker_order.filled_quantity
    delta = new_filled - order.filled_quantity

    if delta < 0:
        return {
            "mismatch": {
                "order_id": order.id,
                "detail": (
                    f"order {order.id}: broker reports {new_filled} filled but "
                    f"{order.filled_quantity} is already recorded locally — a "
                    "fill cannot un-happen; left for §18 reconciliation"
                ),
            }
        }
    if delta > 0 and broker_order.filled_avg_price is None:
        # Not a mismatch — the broker's filled_qty routinely populates before
        # its filled_avg_price. Nothing can be applied without a price;
        # retried next sweep, when the price has usually arrived.
        return {
            "retry": {
                "order_id": order.id,
                "error": (
                    f"broker reports {new_filled} filled but no "
                    "filled_avg_price yet — fill held until the price "
                    "publishes (retried next sweep)"
                ),
            }
        }

    changed = False
    outcome: dict = {}

    # Adoption: a PENDING_SUBMIT row that DID reach the broker gains its
    # broker id and real status here (the crashed submit's missing half).
    if order.broker_order_id is None and broker_order.broker_order_id:
        order.broker_order_id = broker_order.broker_order_id
        changed = True

    if delta > 0:
        fill_detail = await _apply_fill_delta(session, order, broker_order, delta)
        if "mismatch" in fill_detail:
            return {"mismatch": fill_detail["mismatch"]}
        outcome["fill"] = fill_detail
        changed = True

    raw = broker_order.raw_status[:24] if broker_order.raw_status else None
    if (
        order.status != broker_order.status
        or order.broker_status != raw
        or order.filled_quantity != new_filled
    ):
        prior_status = order.status
        order.status = broker_order.status
        order.broker_status = raw
        order.filled_quantity = new_filled
        if broker_order.filled_avg_price is not None:
            order.fill_price = broker_order.filled_avg_price
        changed = True
        if broker_order.status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
            outcome["settled"] = {
                "order_id": order.id,
                "from": prior_status,
                "to": broker_order.status,
            }
            await audit.record(
                session,
                actor_type=ActorType.SYSTEM,
                action=(
                    AuditAction.ORDER_REJECTED
                    if broker_order.status == "REJECTED"
                    else AuditAction.ORDER_SUBMITTED
                ),
                entity_type="order",
                entity_id=str(order.id),
                details={
                    "source": "order_sync_sweep",
                    "transition": f"{prior_status} -> {broker_order.status}",
                    "broker_status_raw": broker_order.raw_status,
                    "filled_quantity": new_filled,
                    "note": (
                        "settled by the order-sync sweep from the broker's own "
                        "state — fills already applied stay applied (they were "
                        "real); no cash moved beyond them"
                    ),
                },
            )

    outcome["changed"] = changed
    return outcome


async def _apply_fill_delta(
    session, order: Order, broker_order: BrokerOrder, delta: int
) -> dict:
    """Move real cash and real position quantity for `delta` new fills.

    The broker reports CUMULATIVE filled quantity and average price, so the
    incremental cash for this delta is derived exactly:

        incremental = new_avg * new_filled - old_avg * old_filled

    (per share, x multiplier) — cash conservation holds to the cent across
    any number of sweeps. ``order.fill_price`` always stores the cumulative
    average recorded so far, which is what makes the subtraction correct.
    """
    is_option = order.instrument in _OPTION_INSTRUMENTS
    multiplier = 100 if is_option else 1
    new_avg = float(broker_order.filled_avg_price)  # checked non-None by caller
    old_avg = order.fill_price or 0.0
    old_filled = order.filled_quantity
    incremental_per_share = new_avg * (old_filled + delta) - old_avg * old_filled
    incremental_cash = incremental_per_share * multiplier

    # NO LOCAL CASH LEDGER: the sweep runs against a REAL broker only, and
    # the cash for these fills moved in the real account — the platform
    # stores no copy. Incremental cash is still computed exactly (audited
    # below) so every fill's economics stay on the record.
    if order.side == SELL_TO_CLOSE:
        position = await session.get(Position, order.position_id or -1)
        if position is None or position.status != POSITION_OPEN:
            return {
                "mismatch": {
                    "order_id": order.id,
                    "detail": (
                        f"sell order {order.id} filled {delta} more at the "
                        "broker but its local position is "
                        f"{'missing' if position is None else position.status} "
                        "— left for §18 reconciliation"
                    ),
                }
            }
        if delta > position.quantity:
            return {
                "mismatch": {
                    "order_id": order.id,
                    "detail": (
                        f"sell order {order.id}: broker filled {delta} more but "
                        f"only {position.quantity} remain open locally — "
                        "left for §18 reconciliation"
                    ),
                }
            }
        proceeds = incremental_cash
        realized = proceeds - delta * position.avg_price * multiplier
        remaining = position.quantity - delta
        position.max_loss = (
            position.max_loss * remaining / position.quantity
            if position.quantity > 0
            else 0.0
        )
        position.quantity = remaining
        position.realized_pnl = (position.realized_pnl or 0.0) + realized
        if remaining <= 0:
            position.status = POSITION_CLOSED
            position.closed_at = utcnow()
            position.max_loss = 0.0
        detail = {
            "order_id": order.id,
            "side": order.side,
            "delta": delta,
            "cash_credited": proceeds,
            "realized_pnl": realized,
            "position_id": position.id,
        }
    else:  # BUY_TO_OPEN
        position = (
            await session.get(Position, order.position_id)
            if order.position_id is not None
            else None
        )
        if position is not None and position.status != POSITION_OPEN:
            # The position this order opened has since been CLOSED (the user
            # or an exit sold the earlier fills) and MORE of the buy filled at
            # the broker. Mutating a closed row would corrupt it, and opening
            # a second row would double-count — this is a genuine ledger
            # divergence for §18 reconciliation, not something to guess at.
            return {
                "mismatch": {
                    "order_id": order.id,
                    "detail": (
                        f"buy order {order.id} filled {delta} more at the "
                        f"broker but its position {position.id} is already "
                        f"{position.status} — left for §18 reconciliation"
                    ),
                }
            }
        cost = incremental_cash
        if position is None:
            # The order's first fill arrived AFTER the approve request
            # returned. The position opens NOW, with the §10 chain's own risk
            # context captured on the order row at approval (migration 010).
            # Pre-010 rows have honest None there; 0.0 is then recorded and
            # flagged in the audit rather than invented.
            stop_distance = (
                new_avg if is_option else (order.stop_distance or 0.0)
            )
            position = Position(
                ticker=order.ticker,
                instrument=order.instrument,
                quantity=delta,
                avg_price=new_avg,
                multiplier=multiplier,
                opt_expiry=order.opt_expiry,
                opt_strike=order.opt_strike,
                opt_right=order.opt_right,
                max_loss=(
                    new_avg * delta * multiplier
                    if is_option
                    else delta * stop_distance
                ),
                stop_distance=stop_distance,
                entry_edge=order.entry_edge or 0.0,
                entry_bar_date=order.entry_bar_date,
                status=POSITION_OPEN,
            )
            session.add(position)
            await session.flush()
            order.position_id = position.id
        else:
            # More of the SAME order filled. Pyramiding is forbidden (§42),
            # so this position holds exactly this order's fills: the broker's
            # cumulative average IS the position's true average.
            position.quantity += delta
            position.avg_price = new_avg
            position.max_loss = (
                new_avg * position.quantity * multiplier
                if is_option
                else position.quantity * (position.stop_distance or 0.0)
            )
        detail = {
            "order_id": order.id,
            "side": order.side,
            "delta": delta,
            "cash_debited": cost,
            "position_id": position.id,
        }
        if not is_option and order.stop_distance is None:
            detail["risk_context"] = (
                "order predates migration 010: stop_distance unknown, "
                "recorded 0.0 — max_loss for this position is UNDERSTATED"
            )

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "source": "order_sync_sweep",
            "fill_price": new_avg,
            "filled_quantity": order.filled_quantity + delta,
            "requested_quantity": order.quantity,
            "partial": (order.filled_quantity + delta) < order.quantity,
            **detail,
        },
    )
    return detail


async def run_sync_and_update() -> dict:
    """One background sweep with STATE bookkeeping (mirrors monitor.py)."""
    async with SessionLocal() as session:
        result = await run_order_sync(session)
    if "skipped" in result:
        return result
    STATE.last_sweep_at = datetime.now(timezone.utc)
    STATE.sweeps_total += 1
    STATE.last_result = {
        "checked": result["checked"],
        "fills_applied": len(result["fills_applied"]),
        "settled": len(result["settled"]),
        "orphans_rejected": len(result["orphans_rejected"]),
        "mismatches": len(result["mismatches"]),
        "faults": len(result["faults"]),
    }
    if result["checked"]:
        logger.info(
            "order_sync_sweep",
            extra={"extra_fields": STATE.last_result},
        )
    return result


async def order_sync_loop() -> None:
    """Sleep -> sweep forever. Started/cancelled by the gateway lifespan.

    Cheap no-op ticks when no real broker is configured — the sweep itself
    answers ``skipped`` before touching the database's order table.
    """
    interval = get_settings().order_sync_interval_seconds
    STATE.interval_seconds = interval
    STATE.enabled = True
    logger.info(
        "order_sync_started",
        extra={"extra_fields": {"interval_seconds": interval}},
    )
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await run_sync_and_update()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("order_sync_sweep_failed")
    except asyncio.CancelledError:
        STATE.enabled = False
        logger.info("order_sync_stopped")
        raise
