"""Audit writer — every state-changing decision must produce an audit record (rule 12).

Writes are part of the same DB transaction as the state change they describe,
so state and audit trail cannot diverge.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.telemetry import request_id_var
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
    """Add one audit row to the caller's transaction (rule 12).

    Correlation closure (plan §38 + §41): when the caller does not pass a
    ``correlation_id``, it is filled from the current request's ID bound by
    the gateway's request-ID middleware (:data:`request_id_var`), so every
    audit row written while serving an HTTP request is traceable to that
    exact request. Outside a request (scripts, startup) the contextvar's
    default keeps it an honest empty string. An explicitly passed ID always
    wins.
    """
    if not correlation_id:
        correlation_id = request_id_var.get()
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
