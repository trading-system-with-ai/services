"""Alerts feed API — a severity-graded read of the audit trail (§18/§29/§38).

``GET /api/alerts`` answers the newest-first alert view of recent audit
events. It is strictly READ-ONLY — no audit writes, no state changes: the
audit trail is the single event source and this endpoint only classifies it
through the declarative ``ALERT_RULES`` table (apps/gateway/alerts.py).

Query shape: one SELECT over the newest ``limit`` audit rows whose action is
in the rules table (SQL IN filter). Classification may still drop fetched
rows — approving RISK_DECISIONs are routine, not alerts — so a page can
honestly come back shorter than ``limit``.

Enrichment: ORDER_FILLED / ORDER_REJECTED audit rows identify their order
only by entity_id (the order row id) — their details carry no ticker/side —
so the referenced Order rows are batch-fetched and their ticker/side/
quantity/fill_price merged UNDER the audit details (the audit record wins on
collisions) before classifying, letting the title read
"Order filled: BUY_TO_OPEN 12 NVDA @ 187.42".
"""
from dataclasses import asdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..alerts import ALERT_ACTIONS, classify
from ..db import AuditEvent, Order, get_session

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

# Audit entity_type whose entity_id references an Order row (see the
# ORDER_FILLED / ORDER_REJECTED writers in routers/orders.py).
_ORDER_ENTITY_TYPE = "order"


def _order_row_id(row: AuditEvent) -> int | None:
    """The referenced Order id for an order-entity audit row, else None."""
    if row.entity_type == _ORDER_ENTITY_TYPE and row.entity_id.isdigit():
        return int(row.entity_id)
    return None


@router.get("")
async def list_alerts(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    """Newest-first alerts from the audit trail (read-only, §18/§29/§38)."""
    rows = (
        (
            await session.execute(
                select(AuditEvent)
                .where(AuditEvent.action.in_(ALERT_ACTIONS))
                .order_by(AuditEvent.id.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    # Batch-fetch the Order rows the order-scoped events reference so their
    # titles can name side/quantity/ticker (one IN query, not N lookups).
    order_ids = {oid for row in rows if (oid := _order_row_id(row)) is not None}
    orders_by_id: dict[int, Order] = {}
    if order_ids:
        orders = (
            (await session.execute(select(Order).where(Order.id.in_(order_ids))))
            .scalars()
            .all()
        )
        orders_by_id = {order.id: order for order in orders}

    alerts: list[dict] = []
    for row in rows:
        extra = None
        oid = _order_row_id(row)
        if oid is not None and (order := orders_by_id.get(oid)) is not None:
            extra = {
                "ticker": order.ticker,
                "side": order.side,
                "quantity": order.quantity,
                "fill_price": order.fill_price,
            }
        alert = classify(row, extra)
        if alert is not None:  # predicate drops (e.g. approving previews)
            alerts.append(asdict(alert))
    return alerts
