"""LLM recommendations API (development plan §4.1).

CENTRAL SAFETY RULE (plan §4.1, §44 rule 5, §46): the LLM proposes, the user
curates. LLM output is an information feature, NOT an order signal. The
refresh endpoint performs NO watchlist / trading-pool / order / position
writes — it only inserts recommendation rows (status PENDING) plus their
LLM-attributed audit events. The ONLY path from a recommendation to the
Watchlist is POST /api/recommendations/{id}/promote — an explicit USER API
action that reuses the exact insertion semantics of POST /api/watchlist
(routers.watchlist.add_ticker_to_watchlist), so the two paths cannot diverge.
Recommendations carry zero execution authority.
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.llm import ProviderError, get_recommendation_provider
from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import Recommendation, WatchlistItem, get_session, utcnow
from .watchlist import CURRENT_USER, add_ticker_to_watchlist

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

# How many candidate drafts one refresh asks the provider for (plan §4.1).
# A parameter, never a hardcoded truth (plan §6.2).
DEFAULT_REFRESH_LIMIT = 5

# Recommendation lifecycle states (plan §4.1).
STATUS_PENDING = "PENDING"
STATUS_DISMISSED = "DISMISSED"
STATUS_PROMOTED = "PROMOTED"


def _serialize(rec: Recommendation) -> dict:
    """Recommendation row -> the API response contract (plan §4.1)."""
    return {
        "id": rec.id,
        "ts": rec.ts.isoformat(),
        "ticker": rec.ticker,
        "company": rec.company,
        "sentiment": rec.sentiment,
        "impact": rec.impact,
        "novelty": rec.novelty,
        "source_reliability": rec.source_reliability,
        "horizon": rec.horizon,
        "catalyst_type": rec.catalyst_type,
        "reason_codes": rec.reason_codes,
        "summary": rec.summary,
        "evidence": rec.evidence,
        "status": rec.status,
    }


async def _get_recommendation(session: AsyncSession, rec_id: int) -> Recommendation:
    rec = await session.get(Recommendation, rec_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"recommendation {rec_id} not found")
    return rec


@router.post("/refresh")
async def refresh_recommendations(session: AsyncSession = Depends(get_session)):
    """Generate new PENDING recommendations from the configured provider.

    Exclusions (plan §4.1): tickers already on the Watchlist (the user already
    curates them) and tickers that already have a PENDING recommendation (no
    duplicate pending proposals) are never recommended — they are reported in
    ``skipped`` with their reason. Every created row is audited
    RECOMMENDATION_CREATED as ActorType.LLM (actor_id = provider name) in the
    SAME transaction (rule 12).

    SAFETY (plan §4.1, §44 rule 5): this route writes recommendation rows and
    audit events ONLY — never the Watchlist, Trading Pool, orders or
    positions. LLM output carries zero execution authority.
    """
    settings = get_settings()
    provider_name = settings.llm_provider
    try:
        provider = get_recommendation_provider(provider_name)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    watch_rows = await session.execute(select(WatchlistItem.ticker))
    watchlist_tickers = set(watch_rows.scalars().all())
    pending_rows = await session.execute(
        select(Recommendation.ticker).where(Recommendation.status == STATUS_PENDING)
    )
    pending_tickers = set(pending_rows.scalars().all())
    exclude = watchlist_tickers | pending_tickers

    skipped = [
        {"ticker": t, "reason": "already on watchlist"}
        for t in sorted(watchlist_tickers)
    ] + [
        {"ticker": t, "reason": "already has a PENDING recommendation"}
        for t in sorted(pending_tickers - watchlist_tickers)
    ]

    as_of = utcnow()
    try:
        drafts = provider.generate(exclude_tickers=exclude, as_of=as_of, limit=DEFAULT_REFRESH_LIMIT)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=f"recommendation provider failed: {exc}") from exc

    created: list[dict] = []
    seen: set[str] = set()
    for draft in drafts:
        # Defense in depth: never trust the provider to have honored the
        # exclusions (plan §44 rule 5) — drop violations instead of storing.
        if draft.ticker in exclude or draft.ticker in seen:
            skipped.append({"ticker": draft.ticker, "reason": "provider returned an excluded ticker"})
            continue
        seen.add(draft.ticker)

        rec = Recommendation(
            ts=as_of,
            ticker=draft.ticker,
            company=draft.company,
            sentiment=draft.sentiment,
            impact=draft.impact,
            novelty=draft.novelty,
            source_reliability=draft.source_reliability,
            horizon=draft.horizon,
            catalyst_type=draft.catalyst_type,
            reason_codes=list(draft.reason_codes),
            summary=draft.summary,
            evidence=list(draft.evidence),
            status=STATUS_PENDING,
        )
        session.add(rec)
        await session.flush()  # assign rec.id for the audit entity_id
        await audit.record(
            session,
            actor_type=ActorType.LLM,
            actor_id=provider_name,
            action=AuditAction.RECOMMENDATION_CREATED,
            entity_type="recommendation",
            entity_id=str(rec.id),
            details={
                # Full §4.1 score schema + provider name.
                "provider": provider_name,
                "ticker": draft.ticker,
                "company": draft.company,
                "sentiment": draft.sentiment,
                "impact": draft.impact,
                "novelty": draft.novelty,
                "source_reliability": draft.source_reliability,
                "horizon": draft.horizon,
                "catalyst_type": draft.catalyst_type,
                "reason_codes": list(draft.reason_codes),
                "summary": draft.summary,
                "evidence": list(draft.evidence),
            },
        )
        created.append(_serialize(rec))

    await session.commit()
    return {"created": created, "skipped": skipped}


@router.get("")
async def list_recommendations(
    status: Literal["PENDING", "DISMISSED", "PROMOTED", "ALL"] = Query(default="PENDING"),
    session: AsyncSession = Depends(get_session),
):
    """List recommendations, newest first, filtered by lifecycle status."""
    stmt = select(Recommendation).order_by(Recommendation.ts.desc(), Recommendation.id.desc())
    if status != "ALL":
        stmt = stmt.where(Recommendation.status == status)
    rows = await session.execute(stmt)
    return [_serialize(rec) for rec in rows.scalars().all()]


@router.post("/{rec_id}/dismiss")
async def dismiss_recommendation(rec_id: int, session: AsyncSession = Depends(get_session)):
    """USER dismisses a PENDING recommendation (plan §4.1)."""
    rec = await _get_recommendation(session, rec_id)
    if rec.status != STATUS_PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"recommendation {rec_id} is {rec.status}, not PENDING",
        )

    rec.status = STATUS_DISMISSED
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.RECOMMENDATION_DISMISSED,
        entity_type="recommendation",
        entity_id=str(rec.id),
        details={"ticker": rec.ticker},
    )
    await session.commit()
    return _serialize(rec)


@router.post("/{rec_id}/promote")
async def promote_recommendation(rec_id: int, session: AsyncSession = Depends(get_session)):
    """USER promotes a PENDING recommendation onto the Watchlist.

    THE ONLY recommendation -> watchlist path (plan §4.1, §44 rule 5): an
    explicit USER API action. Internally reuses
    routers.watchlist.add_ticker_to_watchlist — the exact insertion semantics
    of POST /api/watchlist (409 on duplicate, WATCHLIST_ADD audited as
    ActorType.USER) — then marks the row PROMOTED with a
    RECOMMENDATION_PROMOTED audit, all in ONE transaction (rule 12).

    409 when the ticker is already on the Watchlist (the row stays PENDING —
    nothing commits) and when the row is not PENDING.
    """
    rec = await _get_recommendation(session, rec_id)
    if rec.status != STATUS_PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"recommendation {rec_id} is {rec.status}, not PENDING",
        )

    # Raises 409 if the ticker is already watchlisted; since nothing has been
    # committed, the recommendation row stays PENDING.
    await add_ticker_to_watchlist(
        session, rec.ticker, note=f"promoted from recommendation #{rec.id}"
    )

    rec.status = STATUS_PROMOTED
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.RECOMMENDATION_PROMOTED,
        entity_type="recommendation",
        entity_id=str(rec.id),
        details={"ticker": rec.ticker},
    )
    await session.commit()
    return {"recommendation": _serialize(rec), "watchlist_ticker": rec.ticker}
