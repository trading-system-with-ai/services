"""Read-only audit trail API — no black-box state transitions (plan §38)."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import AuditEvent, get_session
from ..schemas import AuditEventOut

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("", response_model=list[AuditEventOut])
async def list_audit_events(
    session: AsyncSession = Depends(get_session),
    entity_id: str | None = Query(default=None, description="filter by ticker/entity"),
    limit: int = Query(default=100, le=1000),
):
    stmt = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(limit)
    if entity_id:
        stmt = stmt.where(AuditEvent.entity_id == entity_id.upper())
    rows = await session.execute(stmt)
    return rows.scalars().all()
