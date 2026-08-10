"""Read-only audit trail API — no black-box state transitions (plan §38)."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.models import ActorType, AuditAction

from ..db import AuditEvent, get_session
from ..schemas import AuditEventOut

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("", response_model=list[AuditEventOut])
async def list_audit_events(
    session: AsyncSession = Depends(get_session),
    entity_id: str | None = Query(default=None, description="filter by ticker/entity"),
    action: AuditAction | None = Query(
        default=None, description="filter by exact audit action"
    ),
    actor_type: ActorType | None = Query(
        default=None, description="filter by actor type (USER | SYSTEM | LLM)"
    ),
    limit: int = Query(default=100, le=1000),
):
    # All provided filters combine with AND semantics.
    stmt = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(limit)
    if entity_id:
        stmt = stmt.where(AuditEvent.entity_id == entity_id.upper())
    if action is not None:
        stmt = stmt.where(AuditEvent.action == action.value)
    if actor_type is not None:
        stmt = stmt.where(AuditEvent.actor_type == actor_type.value)
    rows = await session.execute(stmt)
    return rows.scalars().all()


@router.get("/actions", response_model=list[str])
async def list_audit_actions(session: AsyncSession = Depends(get_session)):
    """Sorted DISTINCT action values present in the audit table.

    Lets clients build filter chips from what actually exists instead of
    hardcoding the AuditAction enum.
    """
    rows = await session.execute(
        select(AuditEvent.action).distinct().order_by(AuditEvent.action)
    )
    return list(rows.scalars().all())
