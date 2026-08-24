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
from datetime import datetime, timedelta, timezone
import asyncio
import weakref
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.llm import ProviderError, get_recommendation_provider
from libs.llm.provider import GroundingArticle
from libs.market_data import get_provider as get_market_data_provider
from libs.market_data.provider import CapabilityNotAvailable, MarketDataError
from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import (
    NewsArticleRow,
    Recommendation,
    WatchlistItem,
    get_session,
    utcnow,
)
from ..deps import require_llm_provider, require_market_data_provider
from .watchlist import CURRENT_USER, add_ticker_to_watchlist

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

# How many candidate drafts one refresh asks the provider for (plan §4.1).
# A parameter, never a hardcoded truth (plan §6.2).
DEFAULT_REFRESH_LIMIT = 5

# Phase 8 news ingestion: how many of the newest articles one refresh pulls
# from the market data provider, and how many (newest-first) are handed to
# the LLM as grounding material. Parameters, never truths (§6.2).
NEWS_FETCH_LIMIT = 50
NEWS_ENRICH_LIMIT = 20

# One refresh at a time (per event loop — mirrors orders.execution_lock):
# two concurrent refreshes would both pass the existing-ids check and then
# collide on the news_articles UNIQUE constraint. The IntegrityError handler
# below stays as the cross-process backstop.
_REFRESH_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _refresh_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _REFRESH_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _REFRESH_LOCKS[loop] = lock
    return lock


# Recommendation lifecycle states (plan §4.1).
STATUS_PENDING = "PENDING"
STATUS_EXPIRED = "EXPIRED"
# Catalyst proposals die of old age (user decision 2026-08-20: a week-old
# recommendation is meaningless). PENDING rows older than this are marked
# EXPIRED on the next refresh — clearing the no-duplicate-PENDING block so
# the ticker can be re-proposed, with the transition audited. §6.2: a
# parameter, never a truth.
EXPIRE_AFTER_DAYS = 7
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
        # §38: "" on pre-upgrade rows — honest unknown, never backfilled.
        "llm_model": rec.llm_model,
    }


async def _get_recommendation(session: AsyncSession, rec_id: int) -> Recommendation:
    rec = await session.get(Recommendation, rec_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"recommendation {rec_id} not found")
    return rec


@router.post("/refresh")
async def refresh_recommendations(session: AsyncSession = Depends(get_session)):
    """Generate new PENDING recommendations from the configured provider.

    FIRST, stale PENDING rows (older than ``EXPIRE_AFTER_DAYS``) are marked
    EXPIRED — a week-old catalyst proposal is dead (user decision
    2026-08-20) — clearing their tickers for re-proposal; the transition is
    audited. THEN exclusions (plan §4.1): tickers already on the Watchlist
    (the user already curates them) and tickers that STILL have a live
    PENDING recommendation (no duplicate pending proposals) are never
    recommended — they are reported in ``skipped`` with their reason. Every created row is audited
    RECOMMENDATION_CREATED as ActorType.LLM (actor_id = provider name) in the
    SAME transaction (rule 12).

    SAFETY (plan §4.1, §44 rule 5): this route writes recommendation rows and
    audit events ONLY — never the Watchlist, Trading Pool, orders or
    positions. LLM output carries zero execution authority.

    PHASE 8 GROUNDING (news ingestion → dedup → enrichment): real news is
    fetched from the MARKET DATA provider (Massive — the only data source),
    deduplicated into ``news_articles`` by the provider's own article id,
    and the LLM is handed ONLY those stored articles as its information.
    Every created recommendation's evidence must cite stored article urls
    and its ticker must appear in a cited article's ticker list — drafts
    violating either are DROPPED (reported in ``skipped``), so a fabricated
    citation can never reach the user.

    503 ``LLM_NOT_CONFIGURED`` / ``MARKET_DATA_NOT_CONFIGURED`` when either
    provider is unset — with no news source or no analyst there is nothing
    honest to generate. 503 with code ``NEWS_NOT_AVAILABLE`` when the data
    plan does not include the news endpoint (§16: reported, never faked).
    """
    require_llm_provider()
    require_market_data_provider()
    async with _refresh_lock():
        return await _refresh_locked(session)


async def _refresh_locked(session: AsyncSession) -> dict:
    """The refresh flow proper — caller holds the refresh lock."""
    settings = get_settings()
    provider_name = settings.llm_provider
    try:
        provider = get_recommendation_provider(provider_name)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ---- Phase 8 step 1: ingest real news (dedup by provider article id) --
    md_provider = get_market_data_provider(settings.market_data_provider)
    try:
        fetched = md_provider.get_news(limit=NEWS_FETCH_LIMIT)
    except CapabilityNotAvailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "NEWS_NOT_AVAILABLE",
                "message": (
                    f"the market data plan does not include news: {exc}. "
                    "Recommendations require real articles to ground on — "
                    "there is no synthetic fallback."
                ),
            },
        ) from exc
    except MarketDataError as exc:
        raise HTTPException(
            status_code=502, detail=f"news fetch failed: {exc}"
        ) from exc

    ingested = 0
    if fetched:
        existing_rows = await session.execute(
            select(NewsArticleRow.source_id).where(
                NewsArticleRow.source_id.in_([a.source_id for a in fetched])
            )
        )
        existing_ids = set(existing_rows.scalars().all())
        for article in fetched:
            if article.source_id in existing_ids:
                continue
            session.add(
                NewsArticleRow(
                    source_id=article.source_id,
                    title=article.title,
                    publisher=article.publisher,
                    published_at=article.published_at,
                    url=article.url,
                    tickers=list(article.tickers),
                    description=article.description,
                )
            )
            ingested += 1
        if ingested:
            await audit.record(
                session,
                actor_type=ActorType.SYSTEM,
                action=AuditAction.NEWS_INGESTED,
                entity_type="news_articles",
                entity_id=settings.market_data_provider,
                details={
                    "fetched": len(fetched),
                    "new": ingested,
                    "provider": settings.market_data_provider,
                },
            )
            try:
                await session.flush()
            except IntegrityError:
                # Cross-process backstop: another writer landed the same
                # article ids between our existence check and this flush.
                # Their rows ARE the articles — roll back our duplicates and
                # ground on what is stored.
                await session.rollback()
                ingested = 0

    watch_rows = await session.execute(select(WatchlistItem.ticker))
    watchlist_tickers = set(watch_rows.scalars().all())

    # --- auto-expire stale PENDING rows FIRST (user decision 2026-08-20):
    # they stop blocking re-proposal and leave the pending view; history
    # stays queryable under status=EXPIRED, transition audited below.
    expiry_cutoff = datetime.now(timezone.utc) - timedelta(days=EXPIRE_AFTER_DAYS)
    stale_rows = (
        await session.execute(
            select(Recommendation).where(
                Recommendation.status == STATUS_PENDING,
                Recommendation.ts < expiry_cutoff,
            )
        )
    ).scalars().all()
    expired_tickers = []
    for row in stale_rows:
        row.status = STATUS_EXPIRED
        expired_tickers.append({"id": row.id, "ticker": row.ticker, "ts": row.ts.isoformat()})
    if expired_tickers:
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.RECOMMENDATION_DISMISSED,
            entity_type="recommendations",
            entity_id="EXPIRED",
            details={
                "reason": f"auto-expired: PENDING older than {EXPIRE_AFTER_DAYS} days",
                "expired": expired_tickers,
            },
        )

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

    # ---- Phase 8 step 2: the newest STORED articles are the ONLY input ----
    article_rows = (
        (
            await session.execute(
                select(NewsArticleRow)
                .order_by(NewsArticleRow.published_at.desc())
                .limit(NEWS_ENRICH_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    news_report = {"fetched": len(fetched), "new": ingested, "grounding": len(article_rows)}
    if not article_rows:
        # Nothing to ground on -> nothing to claim. Honest no-op.
        await session.commit()
        return {"created": [], "skipped": skipped, "news": news_report}

    grounding = [
        GroundingArticle(
            url=row.url,
            title=row.title,
            publisher=row.publisher,
            published_at=row.published_at.isoformat(),
            tickers=tuple(row.tickers or ()),
            description=row.description,
        )
        for row in article_rows
    ]
    tickers_by_url = {g.url: set(g.tickers) for g in grounding}

    as_of = utcnow()
    try:
        drafts = provider.enrich(
            articles=grounding,
            exclude_tickers=exclude,
            as_of=as_of,
            limit=DEFAULT_REFRESH_LIMIT,
        )
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
        # GROUNDING VALIDATION (Phase 8): every evidence item must cite one
        # of the stored articles handed to the model, and the ticker must
        # appear in a cited article's own ticker list. A draft that fails
        # either check is fiction wearing a citation — dropped, reported.
        cited_urls = [e.get("source") for e in draft.evidence]
        if not cited_urls or any(u not in tickers_by_url for u in cited_urls):
            skipped.append(
                {"ticker": draft.ticker, "reason": "evidence not grounded in stored news"}
            )
            continue
        if not any(draft.ticker in tickers_by_url[u] for u in cited_urls):
            skipped.append(
                {"ticker": draft.ticker, "reason": "ticker absent from cited articles"}
            )
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
            # §38/§41: record WHICH provider/model produced this
            # interpretation, at generation time.
            llm_model=f"{provider_name}/{settings.llm_model}".rstrip("/"),
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
    return {"created": created, "skipped": skipped, "news": news_report}


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
