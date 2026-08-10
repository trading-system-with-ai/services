"""Audit writer — every state-changing decision must produce an audit record (rule 12).

Writes are part of the same DB transaction as the state change they describe,
so state and audit trail cannot diverge.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.models import ActorType, AuditAction

from .db import AuditEvent


async def record(
    session: AsyncSession,
    *,
    actor_type: ActorType,
    action: AuditAction,
    entity_type: str,
    entity_id: str,
    actor_id: str = "",
    details: dict | None = None,
    correlation_id: str = "",
) -> AuditEvent:
    event = AuditEvent(
        actor_type=actor_type.value,
        actor_id=actor_id,
        action=action.value,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details or {},
        correlation_id=correlation_id,
    )
    session.add(event)
    return event
