"""Prediction-market gateway seam — the READ side (Catalyst research
upgrade; plan §5-§6; LOOP 6). The write side (discovery/snapshot/history
backfill) lands in LOOP 8; both halves share this module, the
event_research/event_news pairing.

READS NEVER FETCH: everything here comes from the stored match/snapshot/
history rows an explicit USER backfill wrote. The distinct honest states
(plan Phase 6 — these are DIFFERENT answers, never conflated):

- ``NEVER_RUN``            — no match decision stored for this event yet;
- ``NO_RELEVANT_PREDICTION_MARKET`` — matching ran and honestly accepted
                              nothing (the valid, common outcome);
- provider failure is a WRITE-time story: a failed backfill stores nothing,
  so the read side keeps reporting the last good state or NEVER_RUN, and
  the backfill response/audit carries the failure.

LANGUAGE RULE (plan §3): every price this section exposes is
``market_implied_probability`` / contract pricing — a statement about what
the market CHARGES, never about the outcome's actual likelihood, and the
data-quality block carries the depth facts (spread, liquidity known or
not) instead of folding them into one confidence number (Phase 23).

POINT-IN-TIME: the match batch selected is the latest decided at/before
``as_of``; snapshots and history points are filtered to observations
at/before ``as_of`` — a replay sees the pricing that was observable then.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.telemetry import REGISTRY
from libs.trading_core.events.evidence import TIER_DATA
from libs.common.config import get_settings
from libs.llm import get_recommendation_provider
from libs.llm.market_selection import MarketEventOption
from libs.trading_core.events.news_intel import sanitize_for_llm
from libs.trading_core.events.prediction_intel import history_features, notable_moves
from libs.trading_core.models import ActorType, AuditAction

from . import audit
from .db import (
    EventIngestStateRow,
    EventPredictionMarketRow,
    EventRow,
    PredictionMarketPricePointRow,
    PredictionMarketRow,
    PredictionMarketSnapshotRow,
)

logger = logging.getLogger(__name__)

#: Audit/metric subject for the prediction-market subsystem.
ENTITY_TYPE = "event_prediction_markets"

#: Phase 14 observability. Labelled by provider only — a market question is
#: third-party text and has no business in a metric label.
MARKET_REQUESTS = REGISTRY.counter(
    "prediction_market_requests_total",
    "Prediction-market discovery queries issued by the research orchestrator.",
    ("provider",),
)
MARKETS_MATCHED = REGISTRY.counter(
    "prediction_markets_matched_total",
    "Prediction markets accepted as relevant to an event.",
    ("provider",),
)
MARKET_PROVIDER_ERRORS = REGISTRY.counter(
    "prediction_market_provider_errors_total",
    "Prediction-market requests that failed or were refused.",
    ("provider",),
)

REASON_NEVER_RUN = "NEVER_RUN"
REASON_NO_RELEVANT_MARKET = "NO_RELEVANT_PREDICTION_MARKET"
#: Accepted matches exist but none could be rendered (orphaned market rows,
#: suspicious wording) — a degradation, NOT the honest "nothing relevant".
REASON_METADATA_UNAVAILABLE = "MARKET_METADATA_UNAVAILABLE"

#: History points loaded per market for the feature computation — bounds the
#: read without truncating any realistic inter-event series (hourly points
#: for a quarter is ~2200).
MAX_HISTORY_POINTS = 2500


class _Point:
    """PricePoint-shaped adapter over the stored row (the pure layer types
    structurally and never imports the ORM)."""

    __slots__ = ("ts", "price")

    def __init__(self, ts: datetime, price: float) -> None:
        self.ts = ts
        self.price = price


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


def _primary_outcome_name(market: PredictionMarketRow) -> str | None:
    outcomes = market.outcomes if isinstance(market.outcomes, list) else []
    for outcome in outcomes:
        if isinstance(outcome, dict) and isinstance(outcome.get("name"), str):
            return outcome["name"]
    return None


async def latest_match_batch(
    session: AsyncSession, event_id: int, *, as_of: datetime
) -> list[EventPredictionMarketRow]:
    """Every decision row from the NEWEST match run decided at/before
    ``as_of`` — accepted and rejected alike (the audit record is the whole
    batch), or [] when no run has been stored."""
    latest_as_of = (
        await session.execute(
            select(EventPredictionMarketRow.as_of)
            .where(
                EventPredictionMarketRow.event_id == event_id,
                EventPredictionMarketRow.as_of <= as_of,
            )
            .order_by(EventPredictionMarketRow.as_of.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest_as_of is None:
        return []
    result = await session.execute(
        select(EventPredictionMarketRow)
        .where(
            EventPredictionMarketRow.event_id == event_id,
            EventPredictionMarketRow.as_of == latest_as_of,
        )
        .order_by(EventPredictionMarketRow.id.asc())
    )
    return list(result.scalars().all())


async def _market_entry(
    session: AsyncSession,
    match: EventPredictionMarketRow,
    *,
    as_of: datetime,
    previous_event_at: datetime | None = None,
    window_start: datetime | None = None,
) -> dict[str, Any] | None:
    market = await session.get(PredictionMarketRow, match.market_id)
    if market is None:
        return None  # a cascade removed the market; the match is an orphan

    # PROVIDER TEXT IS UNTRUSTED (§81): a market question is a third party's
    # words, and this section is model-facing. Only the sanitized forms ride
    # in the bundle; an injection-shaped market is withheld entirely (the
    # caller counts it) — it may cost itself, never alter behavior.
    safe_question = sanitize_for_llm(market.question)
    safe_resolution = sanitize_for_llm(market.resolution_criteria)
    if safe_question.suspicious_instruction or safe_resolution.suspicious_instruction:
        return None

    snapshot = (
        await session.execute(
            select(PredictionMarketSnapshotRow)
            .where(
                PredictionMarketSnapshotRow.market_id == market.id,
                PredictionMarketSnapshotRow.observed_at <= as_of,
            )
            .order_by(PredictionMarketSnapshotRow.observed_at.desc())
            .limit(1)
        )
    ).scalars().first()
    # Python re-gate (the SQL bound is an optimization, never the contract —
    # the news/web-research discipline): a snapshot observed after as_of
    # must not price an earlier instant.
    if snapshot is not None and _as_utc(snapshot.observed_at) > as_of:
        snapshot = None

    primary = _primary_outcome_name(market)
    implied = None
    if snapshot is not None and primary is not None:
        prices = snapshot.outcome_prices if isinstance(
            snapshot.outcome_prices, dict
        ) else {}
        implied = prices.get(primary)

    features_payload = None
    moves_payload: list[dict[str, Any]] = []
    if primary is not None:
        point_rows = (
            await session.execute(
                select(PredictionMarketPricePointRow)
                .where(
                    PredictionMarketPricePointRow.market_id == market.id,
                    PredictionMarketPricePointRow.outcome == primary,
                    PredictionMarketPricePointRow.ts <= as_of,
                )
                .order_by(PredictionMarketPricePointRow.ts.desc())
                .limit(MAX_HISTORY_POINTS)
            )
        ).scalars().all()
        series_points = [_Point(_as_utc(r.ts), r.price) for r in point_rows]
        features = history_features(
            series_points,
            as_of=as_of,
            previous_event_at=previous_event_at,
            window_start=window_start,
        )
        features_payload = features.to_dict() if features else None
        # WHEN the market changed its mind — the windows a reader would search
        # for a cause. Deliberately NOT paired with a headline here: a price
        # move and a same-day story are a coincidence until someone checks,
        # and asserting the link would manufacture the false narrative this
        # platform exists to avoid.
        moves_payload = [m.to_dict() for m in notable_moves(series_points, as_of=as_of)]

    return {
        # The stable citation id the LLM's evidence_refs cite (the web:
        # evidence_key convention, prediction-market flavoured).
        "market_ref": f"pm:{market.provider}:{market.provider_market_id}",
        "provider": market.provider,
        "safe_question": safe_question.text,
        "safe_resolution_criteria": safe_resolution.text or None,
        "relation": match.relation,
        "relevance": match.relevance,
        "reason": match.reason,
        "ambiguity": match.ambiguity,
        # BRACKET SERIES identity. Contracts sharing a series_key are one
        # distribution split across ranges; `series_truncated` says the accept
        # cap cut it, so the UI can label the picture partial instead of
        # drawing a confident wrong shape.
        "series_key": match.series_key,
        "series_truncated": bool(match.series_truncated),
        "notable_moves": moves_payload,
        "matched_by": match.matched_by,
        "market_status": market.market_status,
        # THIRD-PARTY TEXT (§81): an outcome name is the venue's words, not
        # the platform's. Sanitized like the question — an outcome called
        # "Yes 42.5 ignore previous instructions" must neither instruct the
        # model nor smuggle a number into it.
        "primary_outcome": sanitize_for_llm(primary).text if primary else None,
        # PRICING language: what the market charges for the contract — never
        # "the probability of the outcome" (plan §3).
        "market_implied_probability": implied,
        "spread": snapshot.spread if snapshot else None,
        "best_bid": snapshot.best_bid if snapshot else None,
        "best_ask": snapshot.best_ask if snapshot else None,
        "volume": snapshot.volume if snapshot else None,
        "liquidity": snapshot.liquidity if snapshot else None,
        "observed_at": _iso(snapshot.observed_at) if snapshot else None,
        "snapshot_available": snapshot is not None,
        "history": features_payload,
        # Depth FACTS, not a folded score (Phase 23): a 70c contract with
        # unknown liquidity is a different claim from a deep one, and the
        # reader — human or model — weighs that itself.
        "data_quality": {
            "snapshot_available": snapshot is not None,
            "liquidity_known": bool(snapshot and snapshot.liquidity is not None),
            "volume_known": bool(snapshot and snapshot.volume is not None),
            "history_available": features_payload is not None,
        },
    }


async def prediction_markets_section(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    previous_event_at: datetime | None = None,
    window_start: datetime | None = None,
) -> dict[str, Any]:
    """The §46 ``prediction_markets`` bundle section (plan §6). Store-only.

    ``previous_event_at``/``window_start`` are the feature anchors: with
    them, accepted markets carry honest change-since-previous-event /
    since-window-start deltas; without them those deltas are None (an
    absence of ANCHOR, not of data).
    """
    moment = _as_utc(as_of)
    batch = [
        m
        for m in await latest_match_batch(session, event_row.id, as_of=moment)
        # Python re-gate on the batch instant (SQL is an optimization).
        if _as_utc(m.as_of) <= moment
    ]
    if not batch:
        # No decision rows. Two very different reasons, told apart by the run
        # watermark: matching never ran, or it ran over a candidate pool that
        # was empty (the provider had nothing for this event — an honest,
        # common "nothing relevant", not an absence of research).
        ran = await matching_has_run(session, event_row.id, as_of=moment)
        section = {
            "available": False,
            "reason": REASON_NO_RELEVANT_MARKET if ran else REASON_NEVER_RUN,
            "tier": TIER_DATA,
        }
        if ran:
            # Only a run that HAPPENED has a count to report. NEVER_RUN keeps
            # the bare shape — a zero there would assert the platform looked
            # at nothing, when in truth it never looked.
            section["candidates_considered"] = 0
        return section

    accepted = [m for m in batch if m.accepted]
    if not accepted:
        return {
            "available": False,
            "reason": REASON_NO_RELEVANT_MARKET,
            "tier": TIER_DATA,
            "candidates_considered": len(batch),
            "matched_at": _iso(batch[0].as_of),
        }

    entries = []
    for match in accepted:
        entry = await _market_entry(
            session,
            match,
            as_of=moment,
            previous_event_at=previous_event_at,
            window_start=window_start,
        )
        if entry is not None:
            entries.append(entry)

    # Accepted matches that could not be rendered (orphaned market rows,
    # injection-withheld wording) are a DEGRADATION, never the honest
    # "nothing relevant" answer — the two states must stay distinct.
    return {
        "available": bool(entries),
        "reason": None if entries else REASON_METADATA_UNAVAILABLE,
        "tier": TIER_DATA,
        "candidates_considered": len(batch),
        "markets_unrenderable": len(accepted) - len(entries),
        "matched_at": _iso(accepted[0].as_of),
        "matched_markets": entries,
        "market_series": _series_blocks(entries),
    }


#: Mutually exclusive, collectively exhaustive brackets price to ~1.00 in
#: total — a little OVER, by the bid/ask spread, which is why the bar is below
#: 1.0 rather than at it. A set summing far under this is missing brackets.
#: 0.80 is deliberately generous: the failure this catches (four brackets
#: summing to 0.21) is nowhere near the boundary, and a tight bar would flag
#: complete-but-wide series as broken.
SERIES_COMPLETE_MIN_SUM = 0.80


def _series_blocks(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the rendered contracts into BRACKET SERIES.

    A venue publishes a distribution as one contract per range, and the
    brackets are only meaningful read together: four cheap brackets of a
    seven-bracket GDP series showed every outcome priced near zero while the
    market's real central estimate sat in the three that were missing.

    The completeness test is the series' OWN ARITHMETIC — exclusive exhaustive
    brackets sum to about 1.00 — so it catches a gap whatever caused it,
    including brackets discovery never surfaced. That matters: keying only off
    this platform's accept cap would miss exactly the case that produced the
    original wrong picture.

    Standalone contracts (no range wording) are not a series and are omitted:
    a single yes/no contract has no siblings to be incomplete against.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        key = entry.get("series_key")
        if key:
            grouped.setdefault(str(key), []).append(entry)

    blocks: list[dict[str, Any]] = []
    for key, members in grouped.items():
        prices = [m.get("market_implied_probability") for m in members]
        known = [p for p in prices if isinstance(p, (int, float))]
        # Judge only when EVERY bracket carries a price: a partial price set
        # cannot distinguish "a bracket is missing" from "a bracket is
        # unpriced", and guessing between them is how a complete series gets
        # labelled broken.
        total = sum(known) if len(known) == len(members) and known else None
        blocks.append(
            {
                "series_key": key,
                "n_brackets": len(members),
                "market_refs": [m.get("market_ref") for m in members],
                # The sum IS the evidence, so it travels rather than just the
                # verdict — a reader can check the judgement.
                "price_sum": round(total, 4) if total is not None else None,
                "complete": (
                    None if total is None else bool(total >= SERIES_COMPLETE_MIN_SUM)
                ),
                "flagged_truncated": any(
                    bool(m.get("series_truncated")) for m in members
                ),
            }
        )
    blocks.sort(key=lambda b: (-b["n_brackets"], b["series_key"]))
    return blocks


# ---------------------------------------------------------------------------
# The WRITE side — prediction-market discovery and observation (plan §5)
#
# READ ONLY, STRUCTURALLY. Everything below fetches PUBLIC market data and
# writes rows. There is no wallet, no signing key, no order and no position:
# the provider protocol has no method that could place one, and the schema
# has no column that could hold one.
# ---------------------------------------------------------------------------

#: Per-event throttle. Prediction-market pricing moves faster than web
#: research, so this is shorter than the search throttle — but it is still a
#: throttle: a UI poll must never reach this path at all (only the explicit
#: POST does), and a human clicking twice should not double the requests.
MARKET_ATTEMPT_SECONDS = 15 * 60

_market_attempts: dict[int, datetime] = {}

#: Stored ``relation`` for a candidate the matcher never classified.
#:
#: The pure layer returns ``None`` there and is right to: "no relation-defining
#: wording for this event type" is an honest absence, not a category. But the
#: column is NOT NULL by design (a MATCH is a claim, and every stored claim
#: names its kind), and the rejected rows exist for transparency. So the
#: absence gets its own NAMED value rather than a null the schema forbids or a
#: relation the matcher never concluded — the one thing that would be wrong
#: here is defaulting to CONTEXT, which would read as a judgement nobody made.
RELATION_NOT_CLASSIFIED = "NOT_CLASSIFIED"

#: Reasons a backfill declined to spend or could not complete (each distinct).
REASON_NOT_CONFIGURED = "NOT_CONFIGURED"
REASON_THROTTLED = "RECENTLY_REFRESHED"
REASON_PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
#: Some discovery queries failed and the rest matched nothing. NOT the same
#: claim as NO_RELEVANT_PREDICTION_MARKET: the platform did not see the whole
#: candidate space, so "there is no relevant market" is not a conclusion it
#: has earned.
REASON_PARTIAL_DISCOVERY = "PARTIAL_DISCOVERY"

#: How much price history one accepted market may pull per refresh.
#:
#: THE FULL LIFE OF THE CONTRACT, at daily resolution.
#:
#: The venue offers two shapes of the same endpoint and they differ in what
#: they can reach. The bounded `startTs`/`endTs` form is SPAN-CAPPED at ~14
#: days (measured live 2026-08-22: hourly succeeds at 14d, 400s at 16d, and
#: daily fidelity does not widen it). The `interval` form is not capped, and
#: at daily fidelity returns a market's whole history — 265 points back to
#: first trade on the GDP contracts, versus 14 days.
#:
#: Asking for a long window therefore no longer fails; the provider switches
#: shapes (see polymarket.MAX_BOUNDED_HISTORY_DAYS). That is what makes
#: `change_since_previous_event` answerable for a quarterly catalyst, and it
#: is what lets a reader see WHEN a contract repriced rather than only that
#: it sits where it sits today.
#:
#: The earlier 120 here failed for a different reason worth remembering: it
#: used the bounded form, so every history call 400'd while snapshots still
#: stored and the run still reported success — a silent degradation whose
#: only symptom was permanently absent deltas.
HISTORY_LOOKBACK_DAYS = 400


def reset_market_throttle() -> None:
    """Clear the in-process attempt clock (tests; operator recovery)."""
    _market_attempts.clear()


def _group_candidates(candidates: list[Any]) -> dict[str, list[Any]]:
    """Candidates grouped by the VENUE'S own event id.

    A candidate with no group id is its own group of one: it is a standalone
    contract, and treating it as such keeps ungrouped venues working
    unchanged.
    """
    groups: dict[str, list[Any]] = {}
    for info in candidates:
        gid = getattr(info, "provider_event_id", None)
        key = f"venue:{gid}" if gid else f"solo:{info.provider}:{info.market_id}"
        groups.setdefault(key, []).append(info)
    return groups


def _llm_select_events(
    event_row: EventRow,
    candidates: list[Any],
    *,
    settings: Any,
    as_of: datetime,
) -> tuple[list[Any] | None, dict[str, Any] | None]:
    """Ask a model WHICH venue events to read, returning the narrowed pool.

    Returns ``(None, note)`` whenever the pool should be left alone — no
    provider, one group (nothing to choose between), a transport failure, or
    a reply that selected nothing usable. The caller then runs the pure
    matcher over the untouched pool, which is a complete answer on its own:
    this step is an enhancement to ranking, never a dependency of it.

    THE STRUCTURAL GUARANTEE: the returned pool is always a SUBSET of the
    pool passed in, assembled by re-resolving opaque refs minted here. A
    model cannot add a market, only remove one — so a hallucinated id yields
    a smaller pool, never a fabricated contract.
    """
    groups = _group_candidates(candidates)
    if len(groups) <= 1:
        # Nothing to choose between — spending a model call to confirm the
        # only option would be pure cost.
        return None, None

    name = (getattr(settings, "llm_provider", "") or "").strip()
    if not name:
        return None, {"used": False, "reason": "LLM_NOT_CONFIGURED"}
    try:
        provider = get_recommendation_provider(name)
    except Exception:  # noqa: BLE001 — unconfigured LLM is a normal state
        return None, {"used": False, "reason": "LLM_NOT_CONFIGURED"}
    select = getattr(provider, "select_prediction_market_events", None)
    if select is None:
        return None, {"used": False, "reason": "PROVIDER_LACKS_SELECTION"}

    # Refs are OPAQUE and minted here, so the model never handles a venue id
    # and a plausible-looking fabricated one cannot resolve to anything.
    refs = {f"e{i}": key for i, key in enumerate(sorted(groups))}
    options = []
    for ref, key in refs.items():
        members = groups[key]
        # Venue text is UNTRUSTED (§81) — sanitized before it is ever
        # rendered into a prompt.
        questions = tuple(sanitize_for_llm(m.question).text for m in members[:4])
        ends = [m.end_date for m in members if m.end_date is not None]
        options.append(
            MarketEventOption(
                ref=ref,
                title=(questions[0] if questions else "")[:120],
                n_markets=len(members),
                end_date=_iso(min(ends)) if ends else None,
                sample_questions=questions,
            )
        )

    try:
        result = select(
            event_type=event_row.event_type,
            event_title=event_row.title or "",
            scheduled_at=_iso(event_row.scheduled_at) or "",
            options=options,
            as_of=as_of,
        )
    except Exception as exc:  # noqa: BLE001 — degradation, never a 5xx
        logger.warning(
            "prediction_market_selection_failed",
            extra={
                "extra_fields": {
                    "event_id": event_row.id,
                    "error": type(exc).__name__,
                }
            },
        )
        return None, {"used": False, "reason": "SELECTION_FAILED"}

    chosen = [s.ref for s in result.selections if s.ref in refs]
    if not chosen:
        # An empty selection is a VALID answer ("none of these fit"), but the
        # platform does not act on it by discarding everything: the pure
        # matcher still gets the full pool and applies its own floor. The
        # model narrows; it does not veto.
        return None, {
            "used": True,
            "selected": 0,
            "note": result.note,
            "version": result.version,
        }

    narrowed: list[Any] = []
    for ref in chosen:
        # WHOLE GROUPS ONLY. There is no code path here that could take part
        # of one — selection is at event granularity precisely because a
        # distribution read partially is worse than one not read at all.
        narrowed.extend(groups[refs[ref]])
    return narrowed, {
        "used": True,
        "selected": len(chosen),
        "groups_offered": len(groups),
        "note": result.note,
        "version": result.version,
        "reasons": [
            {"relation": s.relation, "reason": s.reason} for s in result.selections
        ],
    }


async def refresh_event_prediction_markets(
    session: AsyncSession,
    event_row: EventRow,
    *,
    provider_name: str,
    now: datetime,
) -> dict[str, Any]:
    """USER action: discover, match and observe this event's markets.

    Deterministic code owns every decision that matters: it builds the
    candidate pool (bounded discovery queries), applies the relevance
    threshold and the accept cap, and no accepted market can exist that the
    provider did not return. Snapshots and history are fetched ONLY for
    accepted markets — the pool is for judging, not for hoarding.

    NO RELEVANT MARKET IS A SUCCESS. A run that considers twenty candidates
    and accepts none stores the decisions and reports honestly; it is not a
    failure and must never be dressed as one.
    """
    from libs.market_data import ProviderNotConfigured
    from libs.prediction_markets import get_provider
    from libs.prediction_markets.provider import PredictionMarketError
    from libs.trading_core.events.prediction_intel import (
        MAX_ACCEPTED_MARKETS,
        MAX_CANDIDATE_MARKETS,
        MAX_MARKET_QUERIES,
        MAX_MARKETS_PER_QUERY,
        market_discovery_queries,
        match_markets,
    )

    from .event_calendar import row_to_event

    moment = _as_utc(now)
    base: dict[str, Any] = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "as_of": moment.isoformat(),
    }

    if not provider_name:
        return {**base, "fetched": False, "reason": REASON_NOT_CONFIGURED}

    last_attempt = _market_attempts.get(event_row.id)
    if (
        last_attempt is not None
        and (moment - last_attempt).total_seconds() < MARKET_ATTEMPT_SECONDS
    ):
        return {**base, "fetched": False, "reason": REASON_THROTTLED}

    # CONSTRUCT BEFORE ARMING THE THROTTLE (see the research seam): an
    # unknown or unconfigured provider name costs nothing and must not 500,
    # nor lock the operator out for the throttle window after they fix it.
    try:
        provider = get_provider(provider_name)
    except (ProviderNotConfigured, ValueError) as exc:
        return {
            **base,
            "fetched": False,
            "reason": REASON_NOT_CONFIGURED,
            "detail": str(exc),
        }
    _market_attempts[event_row.id] = moment

    event = row_to_event(event_row)
    queries = market_discovery_queries(event)[:MAX_MARKET_QUERIES]

    candidates: list[Any] = []
    seen_ids: set[str] = set()
    skipped: list[dict[str, Any]] = []
    for query in queries:
        try:
            # PER-QUERY limit, not the per-EVENT pool cap: passing the pool
            # cap here let one press fetch up to MAX_MARKET_QUERIES x that
            # many markets. The pool cap is enforced separately, below, by
            # match_markets' own max_candidates.
            found = provider.search_markets(
                query, limit=MAX_MARKETS_PER_QUERY, active_only=True
            )
        except PredictionMarketError as exc:
            skipped.append({"query": query, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
            logger.exception(
                "prediction_market_discovery_failed",
                extra={"extra_fields": {"event_id": event_row.id}},
            )
            skipped.append({"query": query, "reason": type(exc).__name__})
            continue
        for info in found:
            key = f"{info.provider}:{info.market_id}"
            if key in seen_ids:
                continue
            seen_ids.add(key)
            candidates.append(info)

    if not candidates and skipped:
        # Every discovery query failed: the provider is unavailable, which is
        # a DIFFERENT answer from "nothing relevant exists" and must not be
        # stored as a match batch that would read as the latter.
        MARKET_PROVIDER_ERRORS.inc(len(skipped), provider=provider_name)
        return {
            **base,
            "fetched": False,
            "reason": REASON_PROVIDER_UNAVAILABLE,
            "skipped": skipped,
        }

    # LLM EVENT SELECTION (optional narrowing, never admission).
    #
    # Discovery returns several venue events — a Q3 distribution, a full-year
    # one, a Eurozone one, a recession contract. Deciding which of those a
    # given release is best read against is a semantic judgement, and the
    # deterministic matcher answered it by scoring contracts INDIVIDUALLY,
    # which is how brackets of two different distributions ended up
    # interleaved in one panel.
    #
    # The model may only NARROW this pool, and only at whole-event
    # granularity. Whatever it picks goes through the same match_markets gate
    # as always — every guard (horizon, foreign jurisdiction, other issuer,
    # ACTIVE status, relevance floor) is re-applied by deterministic code. If
    # the model is unconfigured, errors, or returns nothing usable, the pool
    # is unchanged and the pure matcher decides alone.
    selection_note: dict[str, Any] | None = None
    selected, selection_note = _llm_select_events(
        event_row, candidates, settings=get_settings(), as_of=moment
    )
    if selected:
        candidates = selected

    outcome = match_markets(
        event,
        candidates,
        as_of=moment,
        max_accepted=MAX_ACCEPTED_MARKETS,
        max_candidates=MAX_CANDIDATE_MARKETS,
    )
    by_key = {f"{c.provider}:{c.market_id}": c for c in candidates}

    # A COMPLETED RUN ALWAYS LEAVES A MARK, even one that matched nothing.
    # The read side infers its state from stored rows, so a run over an EMPTY
    # candidate pool would otherwise store nothing and be indistinguishable
    # from "nobody ever researched this event" — the two states this module's
    # docstring says must never be conflated. The watermark is what lets the
    # read side answer NO_RELEVANT_PREDICTION_MARKET honestly.
    await _mark_run(session, event_row.id, now=moment, provider=provider_name)

    # RE-DECIDING AT THE SAME INSTANT OVERWRITES, IT DOES NOT DUPLICATE.
    # ``UNIQUE(event_id, market_id, as_of)`` exists so a match can be
    # re-decided under a LATER as_of without rewriting history — but pressing
    # refresh twice within the same instant is the same decision, not a
    # second one, and inserting it blindly raises IntegrityError and 500s a
    # button press. Backfill idempotence is a database property here
    # (ADR-007), so the existing row for this exact key is UPDATED in place.
    existing_matches = {
        row.market_id: row
        for row in (
            await session.execute(
                select(EventPredictionMarketRow).where(
                    EventPredictionMarketRow.event_id == event_row.id,
                    EventPredictionMarketRow.as_of == moment,
                )
            )
        )
        .scalars()
        .all()
    }

    accepted_rows: list[tuple[Any, int]] = []
    seen_market_ids: set[int] = set()
    for decision in outcome.decisions:
        info = by_key.get(f"{decision.provider}:{decision.market_id}")
        if info is None:
            continue
        market_id = await _upsert_market(session, info, now=moment)
        if market_id in seen_market_ids:
            # Two candidates resolved to ONE stored market (the same contract
            # returned by two discovery queries). One market is one decision.
            continue
        seen_market_ids.add(market_id)
        relation = decision.relation or RELATION_NOT_CLASSIFIED
        row = existing_matches.get(market_id)
        if row is not None:
            row.relation = relation
            row.relevance = decision.relevance
            row.reason = decision.reason
            row.ambiguity = decision.ambiguity
            row.matched_by = decision.matched_by
            row.accepted = decision.accepted
            row.reject_reason = decision.reject_reason
        else:
            session.add(
                EventPredictionMarketRow(
                    event_id=event_row.id,
                    market_id=market_id,
                    as_of=moment,
                    relation=relation,
                    relevance=decision.relevance,
                    reason=decision.reason,
                    ambiguity=decision.ambiguity,
                    matched_by=decision.matched_by,
                    accepted=decision.accepted,
                    reject_reason=decision.reject_reason,
                    series_key=(
                        # Prefer the VENUE'S own grouping over wording: it is
                        # the authoritative statement that two contracts are
                        # brackets of one distribution, and it distinguishes
                        # two same-worded series (Q3 2026 vs full-year 2026)
                        # that a wording key could only separate by luck.
                        f"venue:{decision.provider_event_id}"
                        if decision.provider_event_id
                        else decision.series_key
                    ),
                    series_truncated=decision.series_truncated,
                )
            )
        if decision.accepted:
            accepted_rows.append((info, market_id))

    # --- observe ONLY the accepted markets ---------------------------------
    observed = 0
    history_points = 0
    for info, market_id in accepted_rows:
        observed += await _store_snapshot(
            session, provider, info, market_id, now=moment, skipped=skipped
        )
        history_points += await _store_history(
            session, provider, info, market_id, now=moment, skipped=skipped
        )

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.PREDICTION_MARKET_FETCHED,
        entity_type=ENTITY_TYPE,
        entity_id=str(event_row.id),
        details={
            "event_key": event_row.event_key,
            "provider": provider_name,
            "queries": len(queries),
            "candidates": len(candidates),
            "accepted": len(accepted_rows),
            "snapshots": observed,
            "history_points": history_points,
            "skipped": len(skipped),
        },
    )
    await session.commit()

    MARKET_REQUESTS.inc(len(queries), provider=provider_name)
    MARKETS_MATCHED.inc(len(accepted_rows), provider=provider_name)
    if skipped:
        MARKET_PROVIDER_ERRORS.inc(len(skipped), provider=provider_name)

    return {
        **base,
        "fetched": True,
        "provider": provider_name,
        "queries": len(queries),
        "candidates_considered": len(candidates),
        "markets_accepted": len(accepted_rows),
        # "Nothing relevant" is a claim about the MARKETS, and it is only
        # true if the platform actually saw them. When some discovery queries
        # failed, the honest answer is that the search was incomplete —
        # PARTIAL, not a confident negative the operator would read as "this
        # event has no market".
        "reason": (
            None
            if accepted_rows
            else (REASON_PARTIAL_DISCOVERY if skipped else REASON_NO_RELEVANT_MARKET)
        ),
        "snapshots_stored": observed,
        "history_points_stored": history_points,
        "skipped": skipped,
    }


def _run_state_key(event_id: int) -> str:
    """The ``event_ingest_state`` key recording that matching RAN."""
    return f"prediction_markets:event:{event_id}"


async def _mark_run(
    session: AsyncSession, event_id: int, *, now: datetime, provider: str
) -> None:
    """Record that a match run completed for this event.

    Uses the existing per-adapter watermark table rather than a new one: this
    is precisely what ``event_ingest_state`` is for — "this fetcher ran, and
    here is when it last succeeded" — and a run that accepted nothing needs
    exactly that and nothing more.
    """
    key = _run_state_key(event_id)
    row = await session.get(EventIngestStateRow, key)
    if row is None:
        row = EventIngestStateRow(key=key)
        session.add(row)
    row.last_fetched_at = now
    row.last_ok_at = now
    row.last_error = None
    row.meta = {"provider": provider}


async def matching_has_run(
    session: AsyncSession, event_id: int, *, as_of: datetime
) -> bool:
    """Whether a match run completed for this event at/before ``as_of``.

    Point-in-time like everything else on the read path: a run that happened
    AFTER the requested instant did not exist then, so a replay must still
    answer NEVER_RUN.
    """
    row = await session.get(EventIngestStateRow, _run_state_key(event_id))
    if row is None or row.last_ok_at is None:
        return False
    return _as_utc(row.last_ok_at) <= _as_utc(as_of)


async def _upsert_market(session: AsyncSession, info: Any, *, now: datetime) -> int:
    """Insert or refresh one market's metadata; return its surrogate id.

    Keyed on ``(provider, provider_market_id)`` — the table's UNIQUE pair —
    so re-running discovery updates the wording and status rather than
    duplicating the contract.
    """
    existing = (
        await session.execute(
            select(PredictionMarketRow).where(
                PredictionMarketRow.provider == info.provider,
                PredictionMarketRow.provider_market_id == info.market_id,
            )
        )
    ).scalars().first()

    outcomes = [
        {"name": o.name, "price": o.price} for o in (info.outcomes or [])
    ]
    if existing is not None:
        existing.question = info.question
        existing.url = info.url
        existing.outcomes = outcomes
        existing.resolution_criteria = info.resolution_criteria
        existing.end_date = info.end_date
        existing.market_status = info.status
        existing.last_seen_at = now
        existing.raw = info.raw or {}
        await session.flush()
        return existing.id

    # A concurrent writer can land this exact (provider, provider_market_id)
    # between the SELECT above and this INSERT — the pair is UNIQUE, so the
    # loser gets an IntegrityError. A SAVEPOINT keeps that recoverable: the
    # nested block rolls back alone, leaving the outer transaction (and every
    # row already staged in this run) intact, and we adopt the winner's row.
    row = PredictionMarketRow(
        provider=info.provider,
        provider_market_id=info.market_id,
        provider_event_id=info.provider_event_id,
        question=info.question,
        url=info.url,
        outcomes=outcomes,
        resolution_criteria=info.resolution_criteria,
        end_date=info.end_date,
        market_status=info.status,
        first_seen_at=now,
        last_seen_at=now,
        raw=info.raw or {},
    )
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
        return row.id
    except IntegrityError:
        winner = (
            await session.execute(
                select(PredictionMarketRow).where(
                    PredictionMarketRow.provider == info.provider,
                    PredictionMarketRow.provider_market_id == info.market_id,
                )
            )
        ).scalars().first()
        if winner is None:  # pragma: no cover - the constraint says otherwise
            raise
        winner.last_seen_at = now
        await session.flush()
        return winner.id


async def _store_snapshot(
    session: AsyncSession,
    provider: Any,
    info: Any,
    market_id: int,
    *,
    now: datetime,
    skipped: list[dict[str, Any]],
) -> int:
    """One market's current pricing. Partial failure degrades, never raises."""
    from libs.prediction_markets.provider import PredictionMarketError

    try:
        snapshot = provider.get_market_snapshot(info.market_id)
    except PredictionMarketError as exc:
        # A NAMED refusal (rate limit, unknown market, venue error): its
        # message is the provider's own and is safe to report.
        skipped.append(
            {"market_id": info.market_id, "stage": "snapshot", "reason": str(exc)}
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
        # An UNEXPECTED failure is a bug here, not a venue story: log the
        # stack for the operator and report the TYPE only, matching the
        # discovery loop rather than echoing an arbitrary internal message.
        logger.exception(
            "prediction_market_snapshot_failed",
            extra={"extra_fields": {"market_id": info.market_id}},
        )
        skipped.append(
            {
                "market_id": info.market_id,
                "stage": "snapshot",
                "reason": type(exc).__name__,
            }
        )
        return 0

    observed_at = _as_utc(getattr(snapshot, "observed_at", None) or now)
    existing = (
        await session.execute(
            select(PredictionMarketSnapshotRow).where(
                PredictionMarketSnapshotRow.market_id == market_id,
                PredictionMarketSnapshotRow.observed_at == observed_at,
            )
        )
    ).scalars().first()
    if existing is not None:
        # Same instant already recorded (the UNIQUE key). Re-observing is not
        # an error and must not raise on the shared transaction.
        return 0

    session.add(
        PredictionMarketSnapshotRow(
            market_id=market_id,
            observed_at=observed_at,
            outcome_prices=dict(snapshot.outcome_prices or {}),
            best_bid=snapshot.best_bid,
            best_ask=snapshot.best_ask,
            midpoint=snapshot.midpoint,
            spread=snapshot.spread,
            last_trade_price=snapshot.last_trade_price,
            volume=snapshot.volume,
            liquidity=snapshot.liquidity,
            open_interest=snapshot.open_interest,
            provider=snapshot.provider,
        )
    )
    return 1


async def _store_history(
    session: AsyncSession,
    provider: Any,
    info: Any,
    market_id: int,
    *,
    now: datetime,
    skipped: list[dict[str, Any]],
) -> int:
    """Bounded price history for the market's PRIMARY outcome.

    One outcome, not all of them: the features the bundle computes
    (change_1d/7d, range, trend) are about the contract the section quotes,
    and pulling every leg of a multi-outcome market multiplies the cost for
    numbers nothing reads.
    """
    from libs.prediction_markets.provider import PredictionMarketError

    outcomes = info.outcomes or []
    primary = next((o.name for o in outcomes if o.name), None)
    if primary is None:
        return 0

    start = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    try:
        points = provider.get_price_history(
            info.market_id, outcome=primary, start=start, end=now
        )
    except PredictionMarketError as exc:
        skipped.append(
            {"market_id": info.market_id, "stage": "history", "reason": str(exc)}
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
        logger.exception(
            "prediction_market_history_failed",
            extra={"extra_fields": {"market_id": info.market_id}},
        )
        skipped.append(
            {
                "market_id": info.market_id,
                "stage": "history",
                "reason": type(exc).__name__,
            }
        )
        return 0

    existing_ts = set(
        (
            await session.execute(
                select(PredictionMarketPricePointRow.ts).where(
                    PredictionMarketPricePointRow.market_id == market_id,
                    PredictionMarketPricePointRow.outcome == primary,
                )
            )
        )
        .scalars()
        .all()
    )
    existing_ts = {_as_utc(t) for t in existing_ts}

    stored = 0
    for point in points:
        ts = _as_utc(point.ts)
        if ts in existing_ts:
            continue  # composite PK (market_id, outcome, ts) — insert once
        existing_ts.add(ts)
        session.add(
            PredictionMarketPricePointRow(
                market_id=market_id,
                outcome=primary,
                ts=ts,
                price=point.price,
                provider=info.provider,
            )
        )
        stored += 1
    return stored
