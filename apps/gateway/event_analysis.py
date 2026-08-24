"""Event analysis packages — the gateway seam (Phase F, U3; event spec §16,
§46-§52, §69-§71, §99; audit §7.2, §9.3, §11.6).

WHAT THIS MODULE OWNS, AND WHAT IT DELIBERATELY DOES NOT. It owns the
lifecycle of one stored analysis: assemble the evidence bundle, decide whether
a stored answer already covers it, call the provider, validate what came back,
persist the row and write the audit trail. It owns NO judgement. The bundle is
composed by :mod:`apps.gateway.event_evidence`; the digest, the fact index and
the ``numbers_quoted`` validation belong to the pure layer
(:mod:`libs.trading_core.events.evidence`, :mod:`libs.llm.event_analysis`);
the narrative belongs to the provider. This module is the only place the four
meet, exactly as ``event_news.py`` is for the news pipeline.

THE BUNDLE IS PERSISTED WITH THE ANALYSIS (§47). The model may not compute a
number — every figure it writes must be QUOTED from the bundle, and the
validator checks each quoted path against the bundle's fact index. That check
is only meaningful against the document the model actually saw, so the exact
bundle is stored on the row. Re-deriving it at read time would rebuild it from
TODAY's filings, prices and articles: a different document, silently
validating the wrong evidence.

THE CACHE KEY IS THE EVIDENCE, NOT THE CLOCK (§72). ``(event_id,
bundle_digest, prompt_version, model)`` is UNIQUE in the database, so
re-pressing "Analyse" on unchanged evidence returns the stored row instead of
spending another provider call, and two concurrent handlers can only collide
— ADR-007 has no distributed lock, and the unique index is the whole
mechanism. ``force=True`` is the explicit override, and it does not delete the
old row: the previous answer stays readable as event memory (§69).

FAILURE IS A STATUS, NOT AN EXCEPTION (§44 rule 18; audit §9.3). A provider
that 403s, times out or refuses produces a stored row with ``status:
"FAILED"``, the honest error string and a NULL analysis — never a placeholder
narrative, and never a 5xx to the client, who asked a question the platform
CAN partly answer: the evidence bundle is right there. A model that answered
but quoted a number the bundle does not contain produces ``status:
"INVALID"``, and the text IS still stored with its violations list, because
hiding the misquote destroys the evidence that it happened (§99).

PRIOR ANALYSES ARE OPINIONS, NOT EVIDENCE (§70). :func:`prior_analyses_for_ticker`
feeds the bundle an ``LLM_PRIOR`` tier of past summaries so the model can see
what it said last quarter, and the tier name is what stops that from being
laundered into fact: the system prompt says so, the fact index never indexes
it, and no number may be quoted from it.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import Settings, get_settings
from libs.trading_core.models import ActorType, AuditAction

from . import audit
from .db import EventAnalysisRow, EventRow
from .event_calendar import ENTITY_TYPE

logger = logging.getLogger(__name__)

#: The analysis kinds (§46 preview vs §71 retrospective). They are different
#: documents about the same event and both are retained — §69 event memory
#: reads the series, so a retrospective must never overwrite the preview it
#: is grading.
KIND_PRE_EVENT = "PRE_EVENT"
KIND_POST_EVENT = "POST_EVENT"

#: The honest outcome vocabulary. Deliberately four values and not a boolean:
#: "the model answered but misquoted a number" and "the provider never
#: answered" are different facts and the UI badges them differently.
STATUS_OK = "OK"
STATUS_INVALID = "INVALID"
STATUS_FAILED = "FAILED"
STATUS_BUNDLE_ONLY = "BUNDLE_ONLY"

#: What an OK row becomes when a FORCED re-run produces a newer good answer
#: for the SAME evidence. It is not a failure and not a correction — the text
#: is untouched and stays readable as §69 event memory — it is a statement
#: that this is no longer the answer the cache serves. The status exists
#: because the cache index holds at most ONE good answer per (event, evidence,
#: prompt, model): without it, "re-ask the model on the same evidence" would
#: have to either collide or DELETE the previous answer, and deleting is what
#: makes a regression between two model versions undiagnosable.
STATUS_SUPERSEDED = "SUPERSEDED"

#: How many prior analyses travel into a bundle as the §69 event memory. Three
#: is one year of quarterly prints — enough for "it has said this before" to
#: be visible, short enough that the LLM_PRIOR block cannot crowd out the
#: DATA/QUANT evidence it is supposed to be subordinate to.
PRIOR_ANALYSES_LIMIT = 3

#: The prompt version this seam falls back to when ``libs.llm.event_analysis``
#: does not export one. It is part of the UNIQUE cache key, so it must never
#: be blank: a NULL version would let two runs under different instructions
#: collide on one cache entry and serve the older answer under the newer
#: rules. The value is the contract's (§48 "event-analysis-v1").
DEFAULT_PROMPT_VERSION = "event-analysis-v1"


def _as_utc(value: datetime) -> datetime:
    """A datetime as tz-aware UTC. SQLite hands back naive datetimes even for
    a ``DateTime(timezone=True)`` column, so every instant read back off a row
    passes through here before it is compared or rendered."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


# ---------------------------------------------------------------------------
# Event memory (§69, §70)
# ---------------------------------------------------------------------------


def _analysis_summary(row: EventAnalysisRow, *, event_key: str | None = None) -> dict:
    """One prior analysis, reduced to what a later run may SEE of it (§70).

    Summaries only — the regime, the confidence and the executive paragraph.
    Never the scenarios, never the numbers. A past run's figures are that
    run's quotations of a DIFFERENT bundle, and re-exposing them here would
    let this run quote a number whose provenance is another document, which is
    precisely what the fact-index check exists to prevent.
    """
    analysis = row.analysis or {}
    return {
        "id": row.id,
        "event_id": row.event_id,
        "event_key": event_key,
        "as_of": _iso(row.as_of),
        "kind": row.kind,
        "status": row.status,
        "expectations_gap_regime": analysis.get("expectations_gap_regime"),
        "confidence": analysis.get("confidence"),
        "executive_summary": analysis.get("executive_summary"),
        "created_at": _iso(row.created_at),
    }


async def list_analyses(session: AsyncSession, event_id: int) -> list[dict]:
    """Every stored analysis for one event, newest first (§69).

    Summaries, not packages: the history list is a navigation surface, and
    shipping N full bundles to render a list of dates would be megabytes of
    JSON to draw a few rows. The full package is one ``GET .../analysis`` away.
    """
    rows = (
        (
            await session.execute(
                select(EventAnalysisRow)
                .where(EventAnalysisRow.event_id == event_id)
                .order_by(EventAnalysisRow.created_at.desc(), EventAnalysisRow.id.desc())
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            **_analysis_summary(row),
            "provider": row.provider,
            "model": row.model,
            "prompt_version": row.prompt_version,
            "bundle_digest": row.bundle_digest,
            "violations": list(row.violations or []),
            "latency_ms": row.latency_ms,
            "usage": row.usage,
            "error": row.error,
        }
        for row in rows
    ]


async def prior_analyses_for_ticker(
    session: AsyncSession,
    ticker: str | None,
    *,
    before_as_of: datetime,
    limit: int = PRIOR_ANALYSES_LIMIT,
    exclude_event_id: int | None = None,
) -> list[dict]:
    """The last ``limit`` OK analyses this platform wrote about ``ticker``
    BEFORE ``before_as_of`` — the §69 event memory, as summaries only.

    ``before_as_of`` is a real as-of gate and not a convenience: an analysis
    written for a LATER instant knows things this run must not, and feeding it
    back in would be a look-ahead leak laundered through the model's own prose
    (§96). The gate is on ``as_of``, the instant the prior bundle was
    assembled as of — never on ``created_at``, which is merely when the row
    was written and can be today for a historical re-run.

    Only ``OK`` rows travel: a FAILED row has nothing to remember, and an
    INVALID one contains at least one number the platform has already proven
    wrong. Its violations stay visible on its own package for transparency
    (§99); repeating it into a fresh prompt would be propagating it.
    """
    if not ticker or not ticker.strip():
        return []
    cutoff = _as_utc(before_as_of)
    stmt = (
        select(EventAnalysisRow, EventRow.event_key)
        .join(EventRow, EventRow.id == EventAnalysisRow.event_id)
        .where(
            EventRow.ticker == ticker.strip().upper(),
            EventAnalysisRow.status == STATUS_OK,
            EventAnalysisRow.as_of < cutoff,
        )
        .order_by(EventAnalysisRow.as_of.desc(), EventAnalysisRow.id.desc())
        .limit(max(0, int(limit)) + (1 if exclude_event_id is not None else 0))
    )
    out: list[dict] = []
    for row, event_key in (await session.execute(stmt)).all():
        if exclude_event_id is not None and row.event_id == exclude_event_id:
            continue
        out.append(_analysis_summary(row, event_key=event_key))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


async def build_bundle(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    settings: Settings | None = None,
    include_prior: bool = True,
) -> tuple[dict, str]:
    """``(bundle_json, digest)`` for one event as of one instant.

    U1 owns the composition and the digest; this wrapper exists only to append
    the §69 ``prior_analyses`` block (which is gateway state — U1 is pure and
    cannot read the database) BEFORE the digest is taken, so that the digest
    covers everything the model will see. A digest that ignored the prior
    block would let two materially different prompts share one cache entry.

    The imports are function-local on purpose: they are the U1 seam, and
    keeping them lazy means this module (and therefore the router) still
    imports on an install where the evidence layer is being reworked.
    """
    from libs.trading_core.events.evidence import bundle_digest

    from .event_evidence import build_evidence_bundle

    settings = settings or get_settings()
    bundle = await build_evidence_bundle(
        session, event_row, as_of=as_of, settings=settings
    )
    if include_prior:
        bundle["prior_analyses"] = {
            "tier": "LLM_PRIOR",
            "note": (
                "past analyses this platform wrote about this issuer. They are "
                "OPINIONS, not evidence (§70): no number may be quoted from "
                "this section and nothing here is a fact about the world."
            ),
            "items": await prior_analyses_for_ticker(
                session,
                event_row.ticker,
                before_as_of=as_of,
                exclude_event_id=event_row.id,
            ),
        }
    return bundle, bundle_digest(bundle)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _event_status_badge(event_row: EventRow) -> dict:
    """The §7/§11 date-provenance badge that travels with every package.

    An ESTIMATED event MAY be analysed — a derived earnings date is still the
    best information there is — but the payload must say so, every time, so
    the UI can never render a cadence guess as a confirmed fact.
    """
    status = event_row.status
    return {
        "status": status,
        "is_estimated": status == "ESTIMATED",
        "source": event_row.source,
        "source_name": event_row.source_name,
        "note": (
            "this event's date is DERIVED from filing cadence, not confirmed "
            "by the issuer — the analysis below is about a date that may move"
            if status == "ESTIMATED"
            else None
        ),
    }


def _attempt_summary(row: EventAnalysisRow) -> dict:
    """A failed/invalid attempt reduced to what a reader needs to see it.

    No bundle and no narrative: this rides ALONGSIDE a good analysis, and the
    reader is being told "the newest run did not land", not being offered the
    failure's contents to read instead.
    """
    return {
        "id": row.id,
        "status": row.status,
        "error": row.error,
        "created_at": _iso(row.created_at),
        "as_of": _iso(row.as_of),
        "provider": row.provider,
        "model": row.model,
    }


def _good_summary(row: EventAnalysisRow) -> dict:
    """The pointer a failed POST hands back so the UI can fall back."""
    return {
        "id": row.id,
        "created_at": _iso(row.created_at),
        "as_of": _iso(row.as_of),
        "status": row.status,
    }


def serialize_analysis(
    row: EventAnalysisRow,
    event_row: EventRow,
    *,
    cached: bool = False,
    last_attempt: EventAnalysisRow | None = None,
    last_good: EventAnalysisRow | None = None,
) -> dict:
    """One stored package as the API payload (§49 tier separation).

    ``bundle`` carries the DATA and QUANT tiers, ``analysis`` the LLM tier,
    and they are separate KEYS rather than one merged document precisely so a
    client cannot accidentally render a model sentence with the authority of a
    filed number. ``violations`` is always present — ``[]`` means "checked,
    nothing wrong", which is a claim worth making explicitly.

    ``last_attempt`` and ``last_good`` are the two halves of ONE honesty rule:
    a reader must never be shown a stale analysis as if it were current, and
    must never lose a good analysis because a later run failed. When ``row``
    is the last GOOD package but a NEWER run failed, ``last_attempt`` carries
    that failure's status, error and instant, so the UI can show the analysis
    under a "the newest attempt failed" notice rather than silently serving
    yesterday's answer as today's. When ``row`` is a FAILED package and an OK
    one exists, ``last_good`` points at it so the UI can offer the fallback.
    Both are omitted (not null) when they do not apply — an absent key is one
    less thing for a client to branch on.
    """
    payload = {
        "id": row.id,
        "event_id": row.event_id,
        "event_key": event_row.event_key,
        "ticker": event_row.ticker,
        "as_of": _iso(row.as_of),
        "kind": row.kind,
        "status": row.status,
        "cached": cached,
        "bundle": row.bundle,
        "bundle_digest": row.bundle_digest,
        "analysis": row.analysis,
        "provider": row.provider,
        "model": row.model,
        "prompt_version": row.prompt_version,
        "usage": row.usage,
        "latency_ms": row.latency_ms,
        "violations": list(row.violations or []),
        "error": row.error,
        "created_at": _iso(row.created_at),
        "event_status_badge": _event_status_badge(event_row),
        "tiers": {
            "bundle": "DATA/QUANT — measured or filed facts and arithmetic over them",
            "analysis": "LLM — synthesis; every number in it is quoted from bundle",
            "prior_analyses": "LLM_PRIOR — past opinions, not evidence (§70)",
        },
    }
    if last_attempt is not None and last_attempt.id != row.id:
        payload["last_attempt"] = _attempt_summary(last_attempt)
    if last_good is not None and last_good.id != row.id:
        payload["last_good"] = _good_summary(last_good)
    return payload


# ---------------------------------------------------------------------------
# The lifecycle
# ---------------------------------------------------------------------------


async def _cached_row(
    session: AsyncSession,
    *,
    event_id: int,
    digest: str,
    prompt_version: str | None,
    provider: str,
) -> EventAnalysisRow | None:
    """The stored OK row for this exact evidence, if there is one.

    THE MODEL IS NOT MATCHED ON THE CONFIGURED NAME, and that is the subtle
    part. The row records the model that ACTUALLY answered, which a provider
    is free to report more precisely than the setting names it (a
    version-pinned id behind an alias, for instance). Probing on
    ``settings.llm_model`` would then never match the row it just wrote, the
    cache would miss on every second press, and the write would collide with
    the UNIQUE index instead — a spent provider call and a 500 for a request
    whose answer was already on disk. So the probe is on the three things that
    identify the QUESTION — event, evidence digest, prompt version — plus
    ``provider``, which is what actually changes the answer's character;
    ``model`` stays in the database UNIQUE key, where its job is to let a
    genuine model change insert a second row rather than collide.

    ``status == OK`` is required on top: a FAILED attempt on the same evidence
    must NOT satisfy a later request (the user pressing the button again is
    asking to retry), and an INVALID one is a known-misquoting answer that a
    retry may legitimately hope to improve on.
    """
    stmt = (
        select(EventAnalysisRow)
        .where(
            EventAnalysisRow.event_id == event_id,
            EventAnalysisRow.bundle_digest == digest,
            EventAnalysisRow.prompt_version == prompt_version,
            EventAnalysisRow.provider == provider,
            EventAnalysisRow.status == STATUS_OK,
        )
        .order_by(EventAnalysisRow.created_at.desc(), EventAnalysisRow.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def latest_analysis(
    session: AsyncSession, event_id: int
) -> EventAnalysisRow | None:
    """The most recently written package for one event, or None.

    Ordered by ``created_at`` (then id), not by ``as_of``: "the latest thing
    the platform said" is a statement about when it said it. A historical
    re-run for an earlier instant is still the newest opinion on record, and
    the payload carries both instants so the reader can see which is which.
    """
    stmt = (
        select(EventAnalysisRow)
        .where(EventAnalysisRow.event_id == event_id)
        .order_by(EventAnalysisRow.created_at.desc(), EventAnalysisRow.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def latest_ok_analysis(
    session: AsyncSession, event_id: int
) -> EventAnalysisRow | None:
    """The most recently written ``OK`` package for one event, or None.

    THE LAST GOOD ANSWER IS NOT THE LAST ANSWER. A provider timeout writes a
    FAILED row, and a FAILED row is newer than the good analysis it did not
    replace — so a read route that simply takes the newest row hides a
    perfectly valid piece of research behind the error of the run that came
    after it. That is not honesty about the failure, it is data loss with an
    error message on top: the platform still HAS the analysis, and the reader
    is told it does not.

    SUPERSEDED rows are excluded. A superseded row was demoted because a
    forced re-run produced a better answer on the SAME evidence; resurrecting
    it as "the last good one" would undo that decision. If the newer answer
    later fails, the row that failed is the one to report, not the one the
    operator already replaced.
    """
    stmt = (
        select(EventAnalysisRow)
        .where(
            EventAnalysisRow.event_id == event_id,
            EventAnalysisRow.status == STATUS_OK,
        )
        .order_by(EventAnalysisRow.created_at.desc(), EventAnalysisRow.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


def _provider_name(settings: Settings) -> str:
    return (settings.llm_provider or "").strip()


async def get_or_create_analysis(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    settings: Settings | None = None,
    force: bool = False,
    kind: str = KIND_PRE_EVENT,
) -> dict:
    """Assemble the evidence, reuse or produce the synthesis, persist, audit.

    NEVER RAISES FOR A PROVIDER FAILURE. A 403, a timeout, a refusal or a
    malformed response all end as a stored row with ``status: "FAILED"``, the
    honest error text and a NULL analysis, returned at HTTP 200 — because the
    caller asked a question the platform can still partly answer: the evidence
    bundle is assembled and is right there in the payload. Only a bug in this
    module's own bookkeeping is allowed to propagate.

    THE CACHE IS CHECKED AGAINST THE EVIDENCE, NOT THE CLOCK. If an ``OK`` row
    exists for this event with the same bundle digest, prompt version and
    model, it is returned with ``cached: true`` and no provider call is made,
    unless ``force`` is set. ``force`` INSERTS; it never deletes or updates the
    previous row, so the older answer stays readable as event memory (§69) and
    a regression in a new model version is diagnosable rather than overwritten.

    The provider call runs in :func:`asyncio.to_thread` because every adapter
    in ``libs/llm`` is synchronous httpx (the same shape the recommendations
    path uses); awaiting it inline would block the event loop for the whole
    model latency.
    """
    settings = settings or get_settings()
    bundle, digest = await build_bundle(
        session, event_row, as_of=as_of, settings=settings
    )

    # Imported here, not at module import: U2's validator and result type are
    # the LLM seam, and the read-only routes in this module must keep working
    # on an install whose libs.llm is mid-upgrade or missing httpx.
    from libs.llm import ProviderError, get_recommendation_provider
    from libs.llm import event_analysis as llm_event_analysis
    from libs.trading_core.events.evidence import fact_index

    validate_analysis = llm_event_analysis.validate_analysis
    # U2 owns the constant; the default here is the contract's value, so a
    # cache key exists even if the module spells the name differently. It is
    # read, never assumed: the version is part of the UNIQUE cache key, and a
    # silently wrong one would serve last version's answer under new rules.
    prompt_version = getattr(
        llm_event_analysis, "PROMPT_VERSION", DEFAULT_PROMPT_VERSION
    )

    provider_name = _provider_name(settings)
    model = settings.llm_model

    if not force:
        cached = await _cached_row(
            session,
            event_id=event_row.id,
            digest=digest,
            prompt_version=prompt_version,
            provider=provider_name,
        )
        if cached is not None:
            return serialize_analysis(cached, event_row, cached=True)

    analysis: dict | None = None
    usage: dict | None = None
    latency_ms: int | None = None
    violations: list[str] = []
    error: str | None = None
    result_model = model
    result_provider = provider_name
    result_prompt_version = prompt_version

    try:
        provider = get_recommendation_provider(provider_name)
        result = await asyncio.to_thread(
            provider.analyze_event, bundle, as_of=_as_utc(as_of)
        )
    except ProviderError as exc:
        # Includes LLMProviderNotConfigured (a subclass): reachable when this
        # seam is called outside the router's require_llm_provider guard.
        status = STATUS_FAILED
        error = str(exc) or exc.__class__.__name__
        logger.warning(
            "event analysis provider failed for event %s: %s", event_row.id, error
        )
    except Exception as exc:  # noqa: BLE001 - an adapter bug is still a FAILED row
        status = STATUS_FAILED
        error = f"{exc.__class__.__name__}: {exc}"
        logger.exception("event analysis raised for event %s", event_row.id)
    else:
        analysis = dict(result.analysis or {})
        usage = result.usage
        latency_ms = result.latency_ms
        result_model = result.model or model
        result_provider = result.provider or provider_name
        result_prompt_version = result.prompt_version or prompt_version
        # The provider may already have flagged violations of its own (a
        # refusal-adjacent response, a schema field it could not fill). They
        # are kept and the platform's own check is added to them: two
        # independent validators, one list.
        violations = list(result.violations or [])
        analysis, checked = validate_analysis(analysis, fact_index(bundle))
        violations.extend(checked)
        status = STATUS_INVALID if violations else STATUS_OK

    row = EventAnalysisRow(
        event_id=event_row.id,
        as_of=_as_utc(as_of),
        kind=kind,
        bundle=bundle,
        bundle_digest=digest,
        analysis=analysis,
        provider=result_provider,
        model=result_model if status != STATUS_FAILED else model,
        prompt_version=result_prompt_version,
        usage=usage,
        latency_ms=latency_ms,
        violations=violations,
        status=status,
        error=error,
    )

    superseded: EventAnalysisRow | None = None
    if status == STATUS_OK:
        # A forced re-run on unchanged evidence lands on the same cache key as
        # the answer it is replacing, and the partial UNIQUE index holds at
        # most one OK row there. Demote the old one rather than delete it: its
        # text, its bundle and its violations stay on disk and stay listed by
        # ``GET .../analyses``, so a regression introduced by a new model
        # version is still readable beside the answer that replaced it. This
        # runs only for a GOOD new answer — a forced re-run that FAILS or
        # comes back INVALID must NOT demote the good answer already on file.
        superseded = await _cached_row(
            session,
            event_id=event_row.id,
            digest=digest,
            prompt_version=result_prompt_version,
            provider=result_provider,
        )
        if superseded is not None and superseded.model == row.model:
            superseded.status = STATUS_SUPERSEDED
        else:
            superseded = None

    session.add(row)

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.EVENT_ANALYSIS_GENERATED,
        entity_type=ENTITY_TYPE,
        entity_id=str(event_row.id),
        details={
            "event_key": event_row.event_key,
            "ticker": event_row.ticker,
            "as_of": _iso(as_of),
            "kind": kind,
            "status": status,
            "provider": result_provider,
            "model": row.model,
            "prompt_version": result_prompt_version,
            "digest": digest,
            "violations_count": len(violations),
            "usage": usage,
            "latency_ms": latency_ms,
            "forced": bool(force),
            "superseded_id": superseded.id if superseded is not None else None,
            "error": error,
        },
    )
    await session.commit()
    await session.refresh(row)

    # A failed or misquoting run must not cost the reader an analysis the
    # platform already has. The pointer travels on the FAILURE payload so the
    # UI can offer "show the last good analysis" instead of a bare error —
    # the good row itself is one GET away and is deliberately NOT inlined
    # here, so nobody can mistake it for the answer to THIS request.
    last_good: EventAnalysisRow | None = None
    if status != STATUS_OK:
        last_good = await latest_ok_analysis(session, event_row.id)
    return serialize_analysis(
        row, event_row, cached=False, last_good=last_good
    )


__all__ = [
    "KIND_POST_EVENT",
    "KIND_PRE_EVENT",
    "PRIOR_ANALYSES_LIMIT",
    "STATUS_BUNDLE_ONLY",
    "STATUS_FAILED",
    "STATUS_INVALID",
    "STATUS_OK",
    "STATUS_SUPERSEDED",
    "build_bundle",
    "get_or_create_analysis",
    "latest_analysis",
    "latest_ok_analysis",
    "list_analyses",
    "prior_analyses_for_ticker",
    "serialize_analysis",
]
