"""Global kill switch API (development plan §18 — mandatory).

Trading is disabled by default and the switch's state lives in the singleton
system_state row, so it survives restarts. Pausing always requires an explicit
reason; both pause and resume are USER actions and write their audit event in
the same transaction as the state change (rule 12).
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import SystemState, get_or_create_system_state, get_session, utcnow

router = APIRouter(prefix="/api/trading", tags=["trading-control"])

# Single-user V1: a fixed user identity until auth-service lands.
CURRENT_USER = "local-user"

# Resume takes no body, so the recorded reason is this fixed string.
RESUME_REASON = "trading resumed by user"


class PauseRequest(BaseModel):
    reason: str

    @field_validator("reason")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("reason must be a non-empty string")
        return v


def _status_payload(state: SystemState) -> dict:
    return {
        "trading_enabled": state.trading_enabled,
        "reason": state.reason,
        "updated_by": state.updated_by,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
    }


@router.get("/status")
async def trading_status(session: AsyncSession = Depends(get_session)) -> dict:
    state = await get_or_create_system_state(session)
    return _status_payload(state)


@router.post("/pause")
async def pause_trading(
    req: PauseRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    state = await get_or_create_system_state(session)
    state.trading_enabled = False
    state.reason = req.reason
    state.updated_by = CURRENT_USER
    state.updated_at = utcnow()
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.TRADING_PAUSED,
        entity_type="system_state",
        entity_id="global",
        details={"reason": req.reason},
    )
    await session.commit()
    return _status_payload(state)


@router.post("/resume")
async def resume_trading(session: AsyncSession = Depends(get_session)) -> dict:
    state = await get_or_create_system_state(session)
    state.trading_enabled = True
    state.reason = RESUME_REASON
    state.updated_by = CURRENT_USER
    state.updated_at = utcnow()
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.TRADING_RESUMED,
        entity_type="system_state",
        entity_id="global",
        details={"reason": RESUME_REASON},
    )
    await session.commit()
    return _status_payload(state)
