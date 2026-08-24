"""Research trade plan lifecycle API (upgrade 2026-08-12 §19/§40/§41 — Phase D).

The §18 human-review workflow's persistence layer:

    Generate (research chain, §16) -> user review -> Apply (§19) -> ACTIVE
                                                   -> Trading Pool (execution
                                                      still DISABLED)

- ``POST /api/plans/generate`` — run the RESEARCH gate chain (no Trading
  Pool prerequisite, §15) for a Watchlist symbol and persist the complete
  preview as a GENERATED plan with §41 version metadata. A plan whose
  research verdict is NO TRADE is still a valid, useful plan (§17).
- ``POST /api/plans/{id}/apply`` — the user's explicit approval (§19):
  supersede any previous ACTIVE plan for the symbol, promote the symbol to
  the Trading Pool if absent (trading stays DISABLED — §20 research
  approval ≠ execution approval), mark the plan ACTIVE, audit the user
  action. NEVER places an order; execution still requires explicit
  enablement AND the live execution gate chain (§21).
- ``POST /api/plans/{id}/cancel`` — retire a plan without applying it.

Promotion safety (§4.3) is preserved: applying a plan for a symbol not yet
in the pool runs the SAME promotion checks the direct promote endpoint
runs, and failed checks 422 with the full list unless the user explicitly
acknowledges — the override is permanently audited either way.
"""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Literal

from libs.trading_core.models import ActorType, AuditAction, PlanStatus
from libs.trading_core.signals import DirectionalParams, EdgeClassificationParams
from libs.trading_core.tradeability import TradeabilityParams

from .. import audit
from .. import event_risk  # Phase K §65 event-risk seam, SHADOW ONLY
from ..db import (
    TradePlanRow,
    TradingPoolItem,
    WatchlistItem,
    get_or_create_system_state,
    get_session,
)
from ..deps import require_market_data_provider
from ..schemas import TickerRequest
from .analysis import EASTERN, _last_expected_trading_date, _stale_trading_days
from .orders import run_gate_chain
from .trading_pool import CURRENT_USER, promotion_checks

router = APIRouter(prefix="/api/plans", tags=["plans"])

#: §42 configurable tolerance: how many TRADING DAYS a plan's market data may
#: lag the last expected trading day before applying it requires
#: revalidation. Research parameter, not a truth (tests raise it to isolate
#: the frozen stub universe from the real clock, mirroring MAX_BAR_AGE_DAYS).
PLAN_STALENESS_TOLERANCE_TRADING_DAYS = 1


def _current_versions() -> dict:
    """The §41 configuration identifiers active RIGHT NOW — compared against
    a stored plan's versions to detect drift (§42)."""
    return {
        "score_weight_version": DirectionalParams().weights_version,
        "edge_classification_version": EdgeClassificationParams().version,
        "tradeability_version": TradeabilityParams().version,
    }


def _revalidation_state(row: TradePlanRow) -> dict:
    """§42: compare plan-generation state vs current state — never silently.

    - ``stale_market_data``: the plan was generated on bars older than the
      last expected trading day (same Mon–Fri arithmetic as the freshness
      gate; unknown as-of counts as stale — fail closed).
    - ``config_changed``: any §41 configuration version differs from the
      currently active ones (a formula change invalidates reproducibility).
    - ``revalidation_required``: either. Research-plan level surface only —
      EXECUTION never trusts a plan regardless: the §21 chain re-runs live.
    """
    today_eastern = datetime.now(EASTERN).date()
    expected = _last_expected_trading_date(today_eastern)
    if row.market_data_as_of is None:
        stale = True
    else:
        try:
            as_of = date.fromisoformat(row.market_data_as_of)
            stale = (
                _stale_trading_days(as_of, today_eastern)
                > PLAN_STALENESS_TOLERANCE_TRADING_DAYS
            )
        except ValueError:
            stale = True  # unparsable as-of: fail closed, demand revalidation
    current = _current_versions()
    changed = {
        key: {"plan": (row.versions or {}).get(key), "current": value}
        for key, value in current.items()
        if (row.versions or {}).get(key) != value
    }
    return {
        "revalidation_required": stale or bool(changed),
        "stale_market_data": stale,
        "market_data_as_of": row.market_data_as_of,
        "last_expected_trading_date": expected.isoformat(),
        "config_changed": changed,
    }


class PlanGenerateRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)
    direction: Literal["AUTO", "BULL", "BEAR"] = "AUTO"


class PlanApplyRequest(BaseModel):
    acknowledge_risks: bool = False


async def _plan_payload(session: AsyncSession, row: TradePlanRow) -> dict:
    """One stored plan as the API renders it.

    ``async`` because two of its keys are computed FRESH ON EVERY READ and one
    of them needs the database. A stored plan is a snapshot of a moment; the
    two things that decay fastest — whether its inputs are still current
    (§42) and what the stock is about to walk into (§65) — must never be
    served from the frozen ``preview`` blob, because a stale countdown is not
    a weaker statement than no countdown, it is a false one.
    """
    return {
        "id": row.id,
        "ticker": row.ticker,
        "status": row.status,
        "direction": row.direction,
        "quantity_requested": row.quantity_requested,
        "preview": row.preview,
        "versions": row.versions,
        "market_data_as_of": row.market_data_as_of,
        "generated_at": row.generated_at.isoformat() if row.generated_at else None,
        "applied_at": row.applied_at.isoformat() if row.applied_at else None,
        "superseded_by": row.superseded_by,
        "created_by": row.created_by,
        # §42 — computed fresh on EVERY read: a stored plan can never present
        # itself as current without saying so.
        "revalidation": _revalidation_state(row),
        # §65 EVENT RISK, SHADOW — computed fresh on every read for the same
        # reason and never stored: "earnings in 1.3 days" is true for about
        # thirty hours. `None` when the ticker has no print inside the
        # horizon, so the panel simply does not render rather than rendering
        # an empty one. NOTHING here changes the plan's decision — that is
        # Tier 0's, sitting untouched in `preview`.
        "event_risk": await _plan_event_risk(session, row.ticker),
    }


async def _plan_event_risk(session: AsyncSession, ticker: str) -> dict | None:
    """The §65 panel block, or ``None`` — and never an exception.

    A plan read must not 500 because the event registry is empty, mid-migration
    or holding a row this seam cannot classify. The block is additive research
    context; its failure mode is its own absence, carried with the reason so a
    silent null and a broken seam stay distinguishable.
    """
    try:
        return await event_risk.plan_event_risk(session, ticker)
    except Exception as exc:  # noqa: BLE001 — SHADOW context never breaks a read
        return {"error": f"{type(exc).__name__}: {exc}"}


async def _get_plan(session: AsyncSession, plan_id: int) -> TradePlanRow:
    row = (
        await session.execute(select(TradePlanRow).where(TradePlanRow.id == plan_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"plan {plan_id} does not exist")
    return row


@router.post("/generate", status_code=201)
async def generate_plan(
    req: PlanGenerateRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Generate and persist a research trade plan (§16 chain, §41 versions).

    OPEN to any ticker (2026-08-20, §4.2 amended) and NOT
    Trading-Pool-gated (§15) — execution stays behind the §10 chain and the
    explicit per-symbol pool enable. The stored plan is the exact preview the user
    reviews; NO TRADE verdicts are stored too (§17: research is useful
    without execution).
    """
    require_market_data_provider()
    ticker = req.ticker
    # 2026-08-20 user decision (DEVLOG 40): research surfaces are OPEN to any
    # ticker — watchlist membership now means continuous tracking + backtest
    # eligibility, NOT read access. ensure_daily_bars lazily backfills for any
    # symbol; only backtests remain member-only.

    chain = await run_gate_chain(
        session, ticker, req.quantity, req.direction, mode="research"
    )
    row = TradePlanRow(
        ticker=ticker,
        status=PlanStatus.GENERATED.value,
        direction=req.direction,
        quantity_requested=req.quantity,
        preview=chain.preview,
        versions={
            # §41 — the configuration identifiers active at generation time.
            # Only versions that EXIST are recorded; nothing is invented.
            "score_weight_version": DirectionalParams().weights_version,
            "edge_classification_version": EdgeClassificationParams().version,
            "tradeability_version": TradeabilityParams().version,
        },
        market_data_as_of=chain.last_bar_date,
        created_by=CURRENT_USER,
    )
    session.add(row)
    await session.flush()  # assign row.id for the audit record
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.PLAN_GENERATED,
        entity_type="trade_plan",
        entity_id=str(row.id),
        details={
            "ticker": ticker,
            "direction": req.direction,
            "quantity_requested": req.quantity,
            "versions": row.versions,
            "market_data_as_of": row.market_data_as_of,
            "veto_gate": chain.veto_gate,
            "instrument": chain.instrument,
        },
    )
    await session.commit()
    await session.refresh(row)
    return await _plan_payload(session, row)


@router.get("")
async def list_plans(
    ticker: str | None = None, session: AsyncSession = Depends(get_session)
) -> list[dict]:
    """All plans, newest first; optionally filtered by ticker."""
    q = select(TradePlanRow).order_by(TradePlanRow.id.desc())
    if ticker:
        q = q.where(TradePlanRow.ticker == ticker.upper())
    rows = (await session.execute(q)).scalars().all()
    return [await _plan_payload(session, r) for r in rows]


@router.get("/{plan_id}")
async def get_plan(plan_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    return await _plan_payload(session, await _get_plan(session, plan_id))


@router.post("/{plan_id}/apply")
async def apply_plan(
    plan_id: int,
    req: PlanApplyRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The §19 Apply: user approval that makes a plan ACTIVE — never an order.

    In ONE transaction: supersede the previous ACTIVE plan for the symbol
    (audited), promote the symbol to the Trading Pool if absent (§4.3 checks
    enforced, trading stays DISABLED), mark this plan ACTIVE, record the
    user's approval. Afterwards: Trading Pool YES · Plan ACTIVE · Trading
    Enabled NO (§19) — enabling execution is a separate explicit action
    (§20), and any order still re-runs the live execution chain (§21).
    """
    row = await _get_plan(session, plan_id)
    if row.status not in (PlanStatus.GENERATED.value, PlanStatus.REVIEWED.value):
        raise HTTPException(
            status_code=409,
            detail=f"plan {plan_id} is {row.status}; only GENERATED or "
            "REVIEWED plans can be applied",
        )
    # §42: do not let stale research become the ACTIVE plan — recompute first.
    revalidation = _revalidation_state(row)
    if revalidation["revalidation_required"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "PLAN_REVALIDATION_REQUIRED",
                "message": (
                    f"plan {plan_id} was generated on stale market data or "
                    "outdated configuration — revalidate to get a fresh plan, "
                    "then apply that one"
                ),
                "revalidation": revalidation,
            },
        )
    ticker = row.ticker

    # Supersede the previous ACTIVE plan for this symbol (§40).
    superseded_id = None
    active = (
        await session.execute(
            select(TradePlanRow).where(
                TradePlanRow.ticker == ticker,
                TradePlanRow.status == PlanStatus.ACTIVE.value,
            )
        )
    ).scalar_one_or_none()
    if active is not None:
        active.status = PlanStatus.SUPERSEDED.value
        active.superseded_by = row.id
        superseded_id = active.id
        await audit.record(
            session,
            actor_type=ActorType.USER,
            actor_id=CURRENT_USER,
            action=AuditAction.PLAN_SUPERSEDED,
            entity_type="trade_plan",
            entity_id=str(active.id),
            details={"ticker": ticker, "superseded_by": row.id},
        )

    # Promote to the Trading Pool if absent (§19 step 2) — the SAME
    # preconditions as the direct promote endpoint: watchlist membership is a
    # hard 422 that acknowledge_risks cannot bypass (found+fixed 2026-08-20:
    # with plan GENERATION open to any ticker, apply was the one path that
    # could pool a non-watchlist symbol), then the §4.3 readiness checks;
    # trading stays DISABLED (§20).
    pool_row = (
        await session.execute(
            select(TradingPoolItem).where(TradingPoolItem.ticker == ticker)
        )
    ).scalar_one_or_none()
    promoted = False
    checks = None
    if pool_row is None:
        watch = (
            await session.execute(
                select(WatchlistItem).where(WatchlistItem.ticker == ticker)
            )
        ).scalar_one_or_none()
        if watch is None:
            raise HTTPException(
                status_code=422,
                detail=f"{ticker} is not on the Watchlist; only Watchlist "
                "symbols may be promoted",
            )
        checks = await promotion_checks(session, ticker)
        if not all(c["passed"] for c in checks) and not req.acknowledge_risks:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "promotion checks failed — review and "
                    "acknowledge to apply this plan",
                    "checks": checks,
                },
            )
        pool_row = TradingPoolItem(
            ticker=ticker,
            trading_enabled=False,  # §19: applying is authorization, never an order
            allowed_strategies=[],
            promoted_by=CURRENT_USER,
        )
        session.add(pool_row)
        promoted = True
        await audit.record(
            session,
            actor_type=ActorType.USER,
            actor_id=CURRENT_USER,
            action=AuditAction.TRADING_POOL_ADD,
            entity_type="trading_pool",
            entity_id=ticker,
            details={
                "via": "plan_apply",
                "plan_id": row.id,
                "promotion_checks": checks,
                "risks_acknowledged": req.acknowledge_risks,
            },
        )

    row.status = PlanStatus.ACTIVE.value
    row.applied_at = datetime.now(timezone.utc)
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.PLAN_APPLIED,
        entity_type="trade_plan",
        entity_id=str(row.id),
        details={
            "ticker": ticker,
            "superseded_plan_id": superseded_id,
            "promoted_to_pool": promoted,
            "versions": row.versions,
        },
    )
    await session.commit()
    await session.refresh(row)

    # §19 step 5 — re-run execution eligibility (the authorization facts,
    # freshly read). The FULL execution chain still runs at order time (§21).
    state = await get_or_create_system_state(session)
    return {
        "plan": await _plan_payload(session, row),
        "trading_pool": True,
        "trading_enabled": pool_row.trading_enabled,
        "global_trading_enabled": state.trading_enabled,
        "superseded_plan_id": superseded_id,
        "order_placed": False,  # §19: applying a plan NEVER places an order
    }


@router.post("/{plan_id}/revalidate", status_code=201)
async def revalidate_plan(
    plan_id: int, session: AsyncSession = Depends(get_session)
) -> dict:
    """§42 "Recompute": re-run the research chain NOW for this plan's exact
    parameters and persist the result as a NEW GENERATED plan.

    The old plan is never mutated — the new plan stands beside it (linked via
    ``revalidated_from`` in its audit record); applying the new one walks the
    normal §19 path and supersedes whatever is ACTIVE. Works for any
    non-terminal plan whose evidence has gone stale.
    """
    require_market_data_provider()
    row = await _get_plan(session, plan_id)

    chain = await run_gate_chain(
        session, row.ticker, row.quantity_requested, row.direction, mode="research"
    )
    fresh = TradePlanRow(
        ticker=row.ticker,
        status=PlanStatus.GENERATED.value,
        direction=row.direction,
        quantity_requested=row.quantity_requested,
        preview=chain.preview,
        versions=_current_versions(),
        market_data_as_of=chain.last_bar_date,
        created_by=CURRENT_USER,
    )
    session.add(fresh)
    await session.flush()
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.PLAN_GENERATED,
        entity_type="trade_plan",
        entity_id=str(fresh.id),
        details={
            "ticker": row.ticker,
            "revalidated_from": row.id,
            "direction": row.direction,
            "quantity_requested": row.quantity_requested,
            "versions": fresh.versions,
            "market_data_as_of": fresh.market_data_as_of,
            "veto_gate": chain.veto_gate,
            "instrument": chain.instrument,
        },
    )
    await session.commit()
    await session.refresh(fresh)
    return {
        "plan": await _plan_payload(session, fresh),
        "revalidated_from": row.id,
        # §42 comparison surface: what the old plan said vs what fresh data says.
        "previous": {
            "id": row.id,
            "status": row.status,
            "market_data_as_of": row.market_data_as_of,
            "instrument": (row.preview or {}).get("proposed", {}).get("instrument"),
            "veto_gates": [
                g["name"]
                for g in (row.preview or {}).get("gates", [])
                if g.get("status") == "FAIL"
            ],
        },
    }


@router.post("/{plan_id}/cancel")
async def cancel_plan(plan_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Retire a plan that should not be applied (§40 CANCELLED)."""
    row = await _get_plan(session, plan_id)
    if row.status in (
        PlanStatus.SUPERSEDED.value,
        PlanStatus.CANCELLED.value,
        PlanStatus.EXPIRED.value,
    ):
        raise HTTPException(
            status_code=409, detail=f"plan {plan_id} is already {row.status}"
        )
    was = row.status
    row.status = PlanStatus.CANCELLED.value
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.PLAN_CANCELLED,
        entity_type="trade_plan",
        entity_id=str(row.id),
        details={"ticker": row.ticker, "previous_status": was},
    )
    await session.commit()
    await session.refresh(row)
    return await _plan_payload(session, row)
