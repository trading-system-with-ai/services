"""Health endpoints: process liveness/readiness + Strategy Health Monitor v0.

GET /api/health/strategy (plan §19) is a READ-ONLY report over the realized
PnLs of CLOSED positions — it recommends (status PAUSE_RECOMMENDED at worst)
but never acts: no state change, no audit event, and certainly no automatic
pause. Thresholds live in libs.trading_core.health.HealthParams (parameters,
never hardcoded truths, plan §6.2).
"""
from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.health import HealthParams, compute_health

from ..db import Position, get_session

router = APIRouter(tags=["health"])

# Strategy-health thresholds (plan §19). One module-level instance so the
# API's parameterization is explicit and overridable in one place.
HEALTH_PARAMS = HealthParams()


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(session: AsyncSession = Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"status": "ready", "database": "ok"}


@router.get("/api/health/strategy")
async def strategy_health(session: AsyncSession = Depends(get_session)):
    """Strategy Health Monitor v0 report (plan §19).

    Inputs: realized_pnl of CLOSED positions in closed_at order (nulls
    excluded — honest nulls, plan §44 rule 18). Undefined statistics are
    null, never NaN/Infinity. Read-only: no audit event is written.
    """
    rows = await session.execute(
        select(Position.realized_pnl)
        .where(Position.status == "CLOSED", Position.realized_pnl.is_not(None))
        .order_by(Position.closed_at)
    )
    realized_pnls = [float(pnl) for pnl in rows.scalars().all()]
    return asdict(compute_health(realized_pnls, HEALTH_PARAMS))
