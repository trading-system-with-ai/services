"""Broker submission mechanics shared by every execution path (plan §11).

ONE implementation of "get an order to the broker and find out what actually
happened", used by BOTH the manual paths (POST /api/orders/approve,
POST /api/orders/close) and the mechanical ones (POST /api/positions/check-exits
and the background monitor). A mechanical exit that only moved local rows while
the broker still held the position is precisely the reconciliation failure §18
warns about, so there is deliberately no second, simpler path for exits.

THE CONTRACT: this module NEVER assumes a fill.

    submit -> poll -> record whatever state was actually reached

:func:`submit_and_poll` returns the broker's own :class:`BrokerOrder` and the
callers translate it into rows. An ACCEPTED order with zero filled quantity is
a perfectly normal outcome and produces NO position and NO cash movement — the
order row simply records that the broker is working it. Nothing here ever
invents a fill price, a fill quantity or a terminal state.

IDEMPOTENCY (plan §42, extended across the network boundary). Our
``client_order_id`` is passed to the broker as ITS ``client_order_id``, so the
same key identifies the same order on both sides. Before submitting we ASK the
broker whether it already has that key: if it does, we ADOPT that order instead
of submitting a second time. This closes the window the local duplicate guard
cannot see — a submission whose HTTP response was lost still reached the broker,
and a blind retry would double the position. The local guard stays as well; the
two protect different failure modes (a replayed API call vs. a lost response).

POLLING is bounded and honest: ``settings.broker_fill_poll_seconds`` is the
TOTAL budget (default 2.0s, a documented §6.2 parameter; 0 disables polling),
spent in a short exponential backoff. We stop as soon as the order reaches a
terminal state OR shows any fill, and when the budget runs out we record the
last state we actually saw. Waiting longer would be pleasanter; pretending is
not an option.
"""
import asyncio
import logging
import uuid

from libs.broker import BrokerError, BrokerOrder, BrokerProvider, BrokerRejected
from libs.common.config import get_settings

logger = logging.getLogger(__name__)

# Poll backoff: first sleep, then double, capped — all inside the TOTAL budget.
_FIRST_POLL_DELAY_SECONDS = 0.1
_MAX_POLL_DELAY_SECONDS = 0.5

# Statuses that mean "there is nothing more to wait for".
TERMINAL_STATUSES = frozenset({"FILLED", "REJECTED", "CANCELED", "EXPIRED"})


def new_client_order_id(prefix: str) -> str:
    """A fresh idempotency key for an order we are about to place.

    Used when the caller did not supply one (mechanical exits never do). The
    key must be stable for the lifetime of the submission because it is the
    ONLY thing that lets us recognise our own order at the broker after a lost
    response, so it is generated once, here, and then written to the order row.
    """
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def is_terminal(order: BrokerOrder) -> bool:
    """True when the broker order can no longer change state."""
    return order.status in TERMINAL_STATUSES


def _settled(order: BrokerOrder) -> bool:
    """True when polling has nothing more to learn: terminal, or partly filled.

    A PARTIALLY_FILLED order is NOT terminal — more may fill later — but it is
    actionable now, and this platform records the partial honestly rather than
    holding a request open waiting for a completion that may never come.
    """
    return is_terminal(order) or order.filled_quantity > 0


async def _poll_for_settlement(
    broker: BrokerProvider, client_order_id: str, initial: BrokerOrder
) -> BrokerOrder:
    """Poll until settled or the total budget expires; return the last state seen.

    Every state returned here was actually READ FROM THE BROKER. A transport
    fault mid-poll is logged and ends the polling with the newest good state we
    have — the order still exists at the broker either way, and reconciliation
    (GET /api/broker/reconcile) is the backstop for the gap.
    """
    budget = get_settings().broker_fill_poll_seconds
    if budget <= 0 or _settled(initial):
        return initial

    latest = initial
    spent = 0.0
    delay = _FIRST_POLL_DELAY_SECONDS
    while spent < budget:
        sleep_for = min(delay, budget - spent)
        await asyncio.sleep(sleep_for)
        spent += sleep_for
        delay = min(delay * 2, _MAX_POLL_DELAY_SECONDS)
        try:
            polled = await asyncio.to_thread(broker.get_order, client_order_id)
        except BrokerError as exc:
            # A fault does NOT mean absence and does not mean failure to fill.
            # Keep the last state we genuinely observed and stop polling.
            logger.warning(
                "broker poll failed for %s: %r — recording the last observed "
                "state (%s) rather than guessing",
                client_order_id, exc, latest.status,
            )
            return latest
        if polled is None:
            # The broker no longer knows an order it just accepted: surprising
            # enough to log, but not something to invent a resolution for.
            logger.warning(
                "broker no longer reports order %s; keeping last observed "
                "state %s", client_order_id, latest.status,
            )
            return latest
        latest = polled
        if _settled(latest):
            return latest
    return latest


async def submit_and_poll(
    broker: BrokerProvider,
    client_order_id: str,
    symbol: str,
    side: str,
    quantity: int,
) -> tuple[BrokerOrder, bool]:
    """Place (or adopt) one order and return ``(order, adopted)``.

    ``adopted`` is True when the broker ALREADY had our ``client_order_id`` and
    no new order was submitted — the network-level idempotency guarantee
    described in the module docstring. The order is returned in whatever state
    polling actually observed.

    Raises :class:`BrokerRejected` when the broker refuses the submission and
    :class:`BrokerError` on a transport/HTTP fault; both are the caller's to
    translate into an HTTP response and an audit event. Every broker call runs
    in a worker thread (the adapters are synchronous httpx) so the event loop —
    and the shared execution lock the caller holds — is never blocked.
    """
    existing = await asyncio.to_thread(broker.get_order, client_order_id)
    if existing is not None:
        # ADOPT: the broker already has this exact order. Submitting again
        # would place a second one for the same intent.
        logger.info(
            "adopting existing broker order %s for client_order_id %s "
            "(status %s) — not submitting again",
            existing.broker_order_id, client_order_id, existing.raw_status,
        )
        return await _poll_for_settlement(broker, client_order_id, existing), True

    submitted = await asyncio.to_thread(
        broker.submit_order, client_order_id, symbol, side, quantity
    )
    return await _poll_for_settlement(broker, client_order_id, submitted), False


def broker_order_details(order: BrokerOrder, *, adopted: bool = False) -> dict:
    """The audit ``details`` block describing one broker order (§38, rule 12).

    Every broker interaction is auditable with the broker's OWN identifiers and
    its OWN status word — never only our normalised view of them, so a
    disagreement with the broker can always be investigated from the audit log.
    """
    return {
        "broker_order_id": order.broker_order_id,
        "client_order_id": order.client_order_id,
        "broker_status": order.raw_status,
        "status": order.status,
        "requested_quantity": order.requested_quantity,
        "filled_quantity": order.filled_quantity,
        "filled_avg_price": order.filled_avg_price,
        "submitted_at": order.submitted_at.isoformat(),
        # True when this order was already at the broker under our
        # client_order_id and we adopted it instead of submitting twice (§42).
        "adopted_existing": adopted,
    }


__all__ = [
    "TERMINAL_STATUSES",
    "BrokerError",
    "BrokerRejected",
    "broker_order_details",
    "is_terminal",
    "new_client_order_id",
    "submit_and_poll",
]
