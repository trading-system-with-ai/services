"""Web-research gateway seam — BOTH SIDES (Catalyst research upgrade; plan
§5-§6). :func:`run_event_research` plans, searches and persists (the write
side, LOOP 8); :func:`web_research_section` reads what was stored (the read
side, LOOP 6). The pairing mirrors event_news, where
``ensure_event_news_window`` writes and ``build_event_news`` reads.

THE WRITE SIDE IS THE ONLY PAID PATH. It takes ``now``, never an ``as_of``:
ingestion has no as-of (audit §7.2 rule 1) — it records what the provider
served and stamps the run, and the read side decides what was knowable when.
Every bound it enforces (queries per event, results per query, unique
documents, accepted evidence) is a NAMED CONSTANT from the pure layer, so
"what one button press can cost" is answerable by reading them.

READS NEVER FETCH (audit §7.2 rule 1): everything here comes from the
stored ``event_search_runs``/``search_evidence`` rows an explicit USER
backfill wrote. An event with no stored run answers NEVER_RUN — an honest
runtime state, distinct from a provider being unconfigured (the providers
endpoint owns that story) and from a run that found nothing.

POINT-IN-TIME: the run selected is the latest whose own ``as_of`` is at or
before the requested instant, and every accepted row is RE-GATED in Python
(``published_at <= as_of``; undated rows admissible only via
``retrieved_at <= as_of``) — the SQL bound is an optimization, never the
contract, the same defense-in-depth the news read seam applies (§96).

MODEL-FACING TEXT IS SAFE TEXT ONLY: the bundle section carries
``safe_title`` (sanitize_for_llm output) and NEVER raw titles, snippets or
URLs — a URL in evidence text is an exfiltration invitation (§81), and the
system prompt forbids repeating one. The raw fields stay in the stored rows
for the research API/UI (display provenance), not in the prompt.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.telemetry import REGISTRY
from libs.trading_core.events.evidence import TIER_DATA
from libs.trading_core.models import ActorType, AuditAction

from . import audit
from .db import EventRow, EventSearchRunRow, SearchEvidenceRow

logger = logging.getLogger(__name__)

#: Audit/metric subject. One entity type for the research subsystem so the
#: audit log filters to "everything this event's research did".
ENTITY_TYPE = "event_research"

#: Phase 14 observability. Labelled by PROVIDER only — never by query text,
#: which can carry a company name a user typed and has no place in a metric.
SEARCH_REQUESTS = REGISTRY.counter(
    "search_requests_total",
    "External web-search queries executed by the research orchestrator.",
    ("provider",),
)
SEARCH_RESULTS_ACCEPTED = REGISTRY.counter(
    "search_results_accepted_total",
    "Search results admitted into the evidence bundle after gating.",
    ("provider",),
)
SEARCH_PROVIDER_ERRORS = REGISTRY.counter(
    "search_provider_errors_total",
    "Search queries that failed or were refused by the provider.",
    ("provider",),
)

#: The honest empty states (§44 rule 18) — distinct, never conflated.
REASON_NEVER_RUN = "NEVER_RUN"
REASON_NO_EVIDENCE_ACCEPTED = "NO_EVIDENCE_ACCEPTED"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


async def latest_search_run(
    session: AsyncSession, event_id: int, *, as_of: datetime
) -> EventSearchRunRow | None:
    """The newest stored run whose own as_of is at/before the requested
    instant — a historical replay sees the run that existed then, never a
    later refresh's evidence (§96)."""
    result = await session.execute(
        select(EventSearchRunRow)
        .where(
            EventSearchRunRow.event_id == event_id,
            EventSearchRunRow.as_of <= as_of,
            # A FAILED run stored nothing worth serving and must not shadow
            # the last good run's evidence; OK and PARTIAL runs both carry
            # real accepted rows.
            EventSearchRunRow.status != "FAILED",
        )
        .order_by(EventSearchRunRow.as_of.desc(), EventSearchRunRow.id.desc())
        .limit(1)
    )
    return result.scalars().first()


async def web_research_section(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> dict[str, Any]:
    """The §46 ``web_research`` bundle section — plan §6's exposure list:
    research_window, search_plan, counts, source/topic mix, the bounded
    ranked evidence set, suppressed_suspicious, skipped. Store-only."""
    moment = _as_utc(as_of)
    run = await latest_search_run(session, event_row.id, as_of=moment)
    if run is None:
        return {
            "available": False,
            "reason": REASON_NEVER_RUN,
            "tier": TIER_DATA,
        }

    rows = (
        (
            await session.execute(
                select(SearchEvidenceRow)
                .where(SearchEvidenceRow.run_id == run.id)
                .order_by(SearchEvidenceRow.id.asc())
            )
        )
        .scalars()
        .all()
    )

    # Defense-in-depth as-of re-gate over the STORED accepted rows: a stored
    # run's own gate already enforced this at write time, but the read
    # contract re-applies it so no storage or replay path can leak a
    # later-published document into an earlier instant.
    accepted: list[SearchEvidenceRow] = []
    excluded_by_as_of = 0
    excluded_suspicious = 0
    for row in rows:
        if not row.accepted:
            continue
        if row.suspicious_instruction:
            # Defense-in-depth (§81): the write gate already rejects flagged
            # rows, but a storage path that ever leaked one through must not
            # reach the model — withheld AND counted, never silently.
            excluded_suspicious += 1
            continue
        published = _as_utc(row.published_at) if row.published_at else None
        retrieved = _as_utc(row.retrieved_at)
        admissible = (
            published <= moment if published is not None else retrieved <= moment
        )
        if admissible:
            accepted.append(row)
        else:
            excluded_by_as_of += 1

    # Ranked as stored acceptance ranked them: relevance first (the pure
    # layer's ordering), id as the deterministic tie-break.
    accepted.sort(key=lambda r: (-(r.relevance or 0.0), r.id))

    important_evidence = [
        {
            # SAFE text only — no raw title/snippet, no URL (see module doc).
            "evidence_key": row.evidence_key,
            "safe_title": row.safe_title,
            "publisher": row.publisher,
            "domain": row.domain,
            "published_at": _iso(row.published_at),
            "source_tier": row.source_tier,
            "topic": row.topic,
            "relevance": row.relevance,
            "result_type": row.result_type,
        }
        for row in accepted
    ]
    section: dict[str, Any] = {
        "available": bool(accepted),
        "reason": None if accepted else REASON_NO_EVIDENCE_ACCEPTED,
        "tier": TIER_DATA,
        "provider": run.provider,
        "research_window": {
            "start": _iso(run.window_start),
            "end": _iso(run.window_end),
            "basis": run.window_basis,
            "previous_event_id": run.previous_event_id,
            "fallback_reason": run.fallback_reason,
        },
        "search_plan": run.plan,
        "queries_executed": run.queries_executed,
        "results_considered": run.results_considered,
        "results_accepted": len(accepted),
        "excluded_by_as_of": excluded_by_as_of,
        "excluded_suspicious_at_read": excluded_suspicious,
        "suppressed_suspicious": run.suppressed_suspicious,
        "skipped": run.skipped,
        "run_status": run.status,
        "source_mix": dict(Counter(r.source_tier or "UNKNOWN" for r in accepted)),
        "topic_mix": dict(Counter(r.topic or "unclassified" for r in accepted)),
        "important_evidence": important_evidence,
        # The fetch clock (volatile in the digest): when this evidence was
        # retrieved, so freshness is visible without being cache-poison.
        "retrieved_at": _iso(
            max((r.retrieved_at for r in accepted), default=run.created_at)
        ),
    }
    return section


# ---------------------------------------------------------------------------
# The WRITE side — the EventResearchOrchestrator (plan §5, Phases 1/12/13)
#
# ONE BUTTON PRESS, BOUNDED COST. The orchestrator decides WHAT EVIDENCE TO
# GATHER and nothing else: it never decides whether to trade, and no value it
# writes is read by strategy, risk or execution (proved structurally by
# tests/test_research_safety_adversarial.py). Its shape is deliberately
# linear and deterministic — window, plan, search, evaluate, persist — rather
# than an agent loop, because every step's bound must be readable in one pass.
# ---------------------------------------------------------------------------

#: Per-event throttle for the paid search. A research window moves slowly (a
#: new comparable event is weeks away), so a second press minutes later would
#: spend the quota to re-read the same web. Keyed per EVENT, unlike the news
#: throttle's per-ticker key: two events on one ticker have DIFFERENT research
#: windows and different plans, so one's refresh must not mute the other's.
RESEARCH_ATTEMPT_SECONDS = 60 * 60

#: Module-level attempt clock, matching the news seam's ``_fetch_attempts``.
#: In-process on purpose: it throttles a human pressing a button, and the DB
#: run rows are the durable record of what was actually spent.
_research_attempts: dict[int, datetime] = {}

RUN_STATUS_OK = "OK"
RUN_STATUS_PARTIAL = "PARTIAL"
RUN_STATUS_FAILED = "FAILED"

#: Reasons a backfill declined to spend, each distinct (§44 rule 18).
REASON_NOT_CONFIGURED = "NOT_CONFIGURED"
REASON_THROTTLED = "RECENTLY_REFRESHED"
REASON_NO_PLAN = "NO_QUERIES_PLANNED"


def reset_research_throttle() -> None:
    """Clear the in-process attempt clock (tests; operator recovery)."""
    _research_attempts.clear()


async def _past_comparable_rows(
    session: AsyncSession, event_row: EventRow, now: datetime
) -> list[EventRow]:
    """Earlier rows of the SAME TYPE that could be this event's precedent.

    Type-general on purpose: the price seam's equivalent helper is
    earnings-only, and reusing it would hand every macro event an empty pool
    — which ``research_window`` would honestly but needlessly report as a
    type-default fallback, when a real previous CPI print is sitting in the
    table. The pure resolver still owns the CHOICE; this only supplies the
    candidates, filtered to rows that already happened.
    """
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type == event_row.event_type,
                    EventRow.scheduled_at < event_row.scheduled_at,
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    moment = _as_utc(now)
    return [
        row
        for row in rows
        if row.id != event_row.id and _as_utc(row.scheduled_at) <= moment
    ]


async def run_event_research(
    session: AsyncSession,
    event_row: EventRow,
    *,
    provider_name: str,
    now: datetime,
) -> dict[str, Any]:
    """USER action: plan and execute this event's bounded web research.

    THE ONLY PATH THAT SPENDS SEARCH QUOTA. Returns an honest report in every
    degraded case (200, never 5xx — a button press must say why nothing
    arrived): unconfigured provider, throttled, no queries planned, or a
    provider that failed every query.

    PARTIAL IS A REAL OUTCOME. One query failing while four answer stores a
    run with ``status=PARTIAL`` and the failures named in ``skipped`` — never
    a silent OK, and never a discarded run whose four good answers are lost.
    """
    from libs.trading_core.events.web_research import (
        MAX_ACCEPTED_EVIDENCE,
        MAX_QUERIES_PER_EVENT,
        MAX_RESULTS_PER_QUERY,
        MAX_UNIQUE_DOCUMENTS,
        build_search_plan,
        evaluate_results,
        research_window,
    )
    from libs.market_data import ProviderNotConfigured
    from libs.web_search import get_provider
    from libs.web_search.provider import WebSearchError

    from .event_calendar import row_to_event

    moment = _as_utc(now)
    base: dict[str, Any] = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "as_of": moment.isoformat(),
    }

    if not provider_name:
        return {
            **base,
            "fetched": False,
            "reason": REASON_NOT_CONFIGURED,
            "queries_executed": 0,
            "results_accepted": 0,
        }

    last_attempt = _research_attempts.get(event_row.id)
    if (
        last_attempt is not None
        and (moment - last_attempt).total_seconds() < RESEARCH_ATTEMPT_SECONDS
    ):
        return {
            **base,
            "fetched": False,
            "reason": REASON_THROTTLED,
            "queries_executed": 0,
            "results_accepted": 0,
        }

    # --- window + plan: deterministic, and the LLM touches neither ---------
    event = row_to_event(event_row)
    past_rows = await _past_comparable_rows(session, event_row, moment)
    from libs.trading_core.events import previous_comparable

    previous, _reason = previous_comparable(
        event, [row_to_event(row) for row in past_rows]
    )
    window = research_window(event, previous, moment)
    plan = build_search_plan(event, window, max_queries=MAX_QUERIES_PER_EVENT)
    if not plan.queries:
        return {
            **base,
            "fetched": False,
            "reason": REASON_NO_PLAN,
            "queries_executed": 0,
            "results_accepted": 0,
        }

    # CONSTRUCT THE PROVIDER BEFORE ARMING THE THROTTLE. A misconfigured or
    # unknown provider name is a CONFIGURATION fault, not a spend: it costs
    # nothing, so it must neither 500 a button press nor burn the hour-long
    # throttle that would stop the operator retrying after fixing the setting.
    try:
        provider = get_provider(provider_name)
    except (ProviderNotConfigured, ValueError) as exc:
        return {
            **base,
            "fetched": False,
            "reason": REASON_NOT_CONFIGURED,
            "detail": str(exc),
            "queries_executed": 0,
            "results_accepted": 0,
        }

    # The attempt is recorded BEFORE the network work: a press that fails
    # mid-flight must still count against the throttle, or a failing provider
    # becomes a retry loop that bills on every click.
    _research_attempts[event_row.id] = moment

    raw_results: list[Any] = []
    skipped: list[dict[str, Any]] = []
    executed = 0
    attempted = 0
    for planned in plan.queries:
        search = (
            provider.search_news
            if planned.result_type == "news"
            else provider.search_web
        )
        attempted += 1
        try:
            found = search(
                planned.query,
                start_time=window.start,
                end_time=window.end,
                limit=MAX_RESULTS_PER_QUERY,
            )
        except WebSearchError as exc:
            # A named provider refusal (rate limit, auth, capability). The
            # OTHER queries still run — one failure is not a failed run.
            skipped.append(
                {"purpose": planned.purpose, "reason": str(exc)}
            )
            continue
        except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
            logger.exception(
                "event_research_query_failed",
                extra={"extra_fields": {"event_id": event_row.id}},
            )
            skipped.append(
                {"purpose": planned.purpose, "reason": f"{type(exc).__name__}"}
            )
            continue
        executed += 1
        for item in found:
            raw_results.append(_tagged(item, planned))

    if executed == 0:
        status = RUN_STATUS_FAILED
    elif skipped:
        status = RUN_STATUS_PARTIAL
    else:
        status = RUN_STATUS_OK

    outcome = evaluate_results(
        raw_results,
        event=event,
        plan=plan,
        as_of=moment,
        max_unique_documents=MAX_UNIQUE_DOCUMENTS,
        max_accepted=MAX_ACCEPTED_EVIDENCE,
    )

    run = EventSearchRunRow(
        event_id=event_row.id,
        as_of=moment,
        window_start=window.start,
        window_end=window.end,
        window_basis=window.basis,
        previous_event_id=window.previous_event_id,
        fallback_reason=window.fallback_reason,
        provider=provider_name,
        plan=_plan_to_json(plan),
        queries_executed=executed,
        results_considered=outcome.results_considered,
        results_accepted=outcome.results_accepted,
        suppressed_suspicious=outcome.suppressed_suspicious,
        skipped=skipped,
        status=status,
        error=None,
    )
    session.add(run)
    await session.flush()  # assign run.id for the evidence rows

    # ONE ROW PER DOCUMENT PER RUN, AND IT MUST BE THE ACCEPTED ONE.
    #
    # The pure layer reports every candidate it CONSIDERED, which is the right
    # transparency record but collides with UNIQUE(run_id, canonical_url):
    # several candidates can share one canonical_url (a tracking-tagged twin,
    # a stale-dated copy, an injection-shaped duplicate). Collapsing on first
    # occurrence would be WRONG — the pure layer emits time-rejected and
    # suspicious copies BEFORE the clean original, deliberately, so that a
    # bad twin cannot register itself as the dedup winner. Keeping the first
    # would hand the row to exactly the copy that layer set out to refuse,
    # store a run claiming evidence it did not keep, and (worst) let an
    # injection-shaped duplicate be the surviving record of a good document.
    #
    # So: the ACCEPTED candidate wins its URL; among equals, higher relevance,
    # then the earlier position (the pure layer's own ranking).
    best: dict[str, Any] = {}
    for candidate in outcome.candidates:
        incumbent = best.get(candidate.canonical_url)
        if incumbent is None or _better_candidate(candidate, incumbent):
            best[candidate.canonical_url] = candidate

    stored_accepted = sum(1 for c in best.values() if c.accepted)
    if stored_accepted != outcome.results_accepted:
        # The collapse must never lose an accepted document. Loud rather than
        # silent: a run that reports evidence it did not store is a lie the
        # audit row and the Evidence tab would both repeat.
        logger.error(
            "event_research_collapse_lost_evidence",
            extra={
                "extra_fields": {
                    "event_id": event_row.id,
                    "reported": outcome.results_accepted,
                    "stored": stored_accepted,
                }
            },
        )
        run.results_accepted = stored_accepted

    for candidate in best.values():
        session.add(
            SearchEvidenceRow(
                run_id=run.id,
                event_id=event_row.id,
                evidence_key=candidate.evidence_key,
                query=candidate.query,
                purpose=candidate.purpose,
                title=candidate.title,
                safe_title=candidate.safe_title,
                url=candidate.url,
                canonical_url=candidate.canonical_url,
                publisher=candidate.publisher,
                domain=candidate.domain,
                published_at=candidate.published_at,
                retrieved_at=candidate.retrieved_at,
                snippet=candidate.snippet,
                safe_snippet=candidate.safe_snippet,
                suspicious_instruction=candidate.suspicious_instruction,
                source_tier=candidate.source_tier,
                topic=candidate.topic,
                relevance=candidate.relevance,
                rank=candidate.rank,
                result_type=candidate.result_type,
                provider=candidate.provider,
                accepted=candidate.accepted,
                reject_reason=candidate.reject_reason,
            )
        )

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.EVENT_SEARCH_RUN,
        entity_type=ENTITY_TYPE,
        entity_id=str(event_row.id),
        details={
            "event_key": event_row.event_key,
            "provider": provider_name,
            "status": status,
            # Cost transparency (Phase 12): what this press actually bought.
            "queries_executed": executed,
            "results_considered": outcome.results_considered,
            "results_accepted": outcome.results_accepted,
            "suppressed_suspicious": outcome.suppressed_suspicious,
            "window_basis": window.basis,
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
        },
    )
    await session.commit()

    # Requests ISSUED, not merely those that answered: a provider erroring on
    # every call still consumed the operator's rate budget, and a metric that
    # hid that would make a failing integration look free.
    SEARCH_REQUESTS.inc(attempted, provider=provider_name)
    SEARCH_RESULTS_ACCEPTED.inc(outcome.results_accepted, provider=provider_name)
    if skipped:
        SEARCH_PROVIDER_ERRORS.inc(len(skipped), provider=provider_name)

    return {
        **base,
        "fetched": True,
        "provider": provider_name,
        "status": status,
        "queries_planned": len(plan.queries),
        "queries_executed": executed,
        "results_considered": outcome.results_considered,
        "results_accepted": outcome.results_accepted,
        "suppressed_suspicious": outcome.suppressed_suspicious,
        "skipped": skipped,
        "research_window": {
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "basis": window.basis,
            "previous_event_id": window.previous_event_id,
            "fallback_reason": window.fallback_reason,
        },
    }


def _better_candidate(candidate: Any, incumbent: Any) -> bool:
    """Whether `candidate` should replace `incumbent` for their shared URL.

    Acceptance first (an admitted document always outranks a refused twin),
    then relevance. Position is the implicit final tie-break: equal
    candidates leave the incumbent in place, which is the pure layer's own
    ordering.
    """
    if candidate.accepted != incumbent.accepted:
        return bool(candidate.accepted)
    return (candidate.relevance or 0.0) > (incumbent.relevance or 0.0)


class _TaggedResult:
    """A provider result carrying the planned query that produced it.

    The provider protocol returns ``SearchResult`` with the query STRING but
    not the plan's ``purpose`` (the provider knows nothing of research
    profiles), and the evidence rows record both. Wrapping rather than
    mutating keeps the provider's dataclass frozen and honest.
    """

    __slots__ = ("_result", "purpose")

    def __init__(self, result: Any, purpose: str) -> None:
        self._result = result
        self.purpose = purpose

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


def _tagged(result: Any, planned: Any) -> _TaggedResult:
    return _TaggedResult(result, planned.purpose)


def _plan_to_json(plan: Any) -> dict[str, Any]:
    """The plan stored verbatim — deterministic code's output, auditable."""
    return {
        "profile_key": plan.profile_key,
        "queries": [
            {
                "purpose": q.purpose,
                "query": q.query,
                "priority": q.priority,
                "result_type": q.result_type,
            }
            for q in plan.queries
        ],
    }
