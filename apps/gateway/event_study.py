"""§86 predictiveness study — the gateway seam (event spec §85, §86, §87, §92,
§101, §102; audit §7.2, §7.4; Phase L unit U2).

Every number this module returns is computed by
``libs/trading_core/events/event_study.py``, which is pure stdlib and cannot
import ``apps/``. This module is the only place the two halves meet: it reads
STORED rows, converts them into the pure layer's input shape, and renders the
report. It computes no correlation and no feature of its own.

STORED ROWS ONLY — THIS SEAM HOLDS NO PROVIDER (audit §7.2 rule 1). There is
no market-data import in this file and no lazy backfill, so a study over sixty
events cannot become sixty provider calls. Three consequences follow and all
three are deliberate:

1. **Features come out of the STORED BUNDLE, not out of a fresh assembly.**
   ``event_analyses.bundle`` is the evidence as it stood at the instant the
   analysis was run, with every as-of gate already applied. Reassembling it
   today would rebuild each feature from TODAY's bars, filings and articles —
   a look-ahead leak in the one module whose entire purpose is to check
   whether the pre-event features were worth anything, and the leak would
   inflate exactly the correlations the report exists to measure honestly.
   That is why the study is over events that were ANALYSED, and why an event
   with no stored analysis is counted as un-studiable rather than being
   assembled on the spot.
2. **Outcomes come from stored daily bars**, read directly rather than through
   ``ensure_daily_bars``: the outcome is a fact about the past and needs no
   freshness, while a lazy backfill here would silently fetch history for
   every ticker in the registry.
3. **A ticker with no stored bars produces ``None`` outcomes**, and those rows
   still appear in the sample with their features, so ``outcome_coverage``
   reports the gap instead of hiding it.

THE OUTCOME IS THE §17 REACTION, MEASURED BY THE SAME FUNCTION THE REPLAY TAB
USES. ``event_reaction`` resolves the pre-event and reaction sessions from the
event's own session label (an AMC print reacts the NEXT morning, a BMO print
the same day), so the 1D outcome here is the identical number the Catalyst
page renders for that print. A second implementation that "just took the next
close" would disagree with the UI on every after-market release, and the
disagreement would show up as a correlation nobody could reproduce.

EARNINGS ONLY, BY DEFAULT. The sample is single-name events with a ticker: a
CPI print has no issuer whose reaction could be measured, and pooling one
market-wide event's SPY move with forty single-name gaps would produce a
column whose rows are not the same kind of observation. ``event_type`` narrows
it further.

THE ENDPOINT MAKES NO CLAIM. ``feature_report`` carries the §92 caveats and
the ``NOT_MEANINGFUL`` flag; this seam adds the sample's own provenance
(which events, over which window, from how many analyses) and nothing that
reads as a finding. §87 is upstream of all of it: nothing here is wired to a
signal, a plan or an order, and there is no code path from this payload into
one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_core.events.event_study import (
    EVENT_STUDY_MODEL_VERSION,
    NOT_MEANINGFUL,
    FeatureRow,
    collect_feature_rows,
    feature_report,
)
from libs.trading_core.events.reaction import event_reaction
from libs.trading_core.models.enums import EventStatus

from .db import EventAnalysisRow, EventOptionMetricRow, EventRow, StockBarDaily
from .event_price import _session_of, event_date_et, to_daily_bars

#: How many per-event rows the payload carries. The REPORT is computed over
#: the whole sample; only the row listing is capped, so raising or lowering
#: this can never change a rho. 200 rows is roughly a megabyte of JSON at the
#: shapes above and is far more than any reader scrolls.
ROW_LIMIT = 200

#: Hard ceiling on the events pulled into one study. A watchlist install has
#: tens of analysed events; this exists so a future bulk backfill cannot turn
#: one GET into a full-table scan plus a bar load per ticker.
MAX_EVENTS = 1000

#: The horizons measured, in trading days after the reaction session.
HORIZONS: tuple[int, ...] = (1, 5)

#: Only OBSERVED dates make a usable observation (§15, §86): an ESTIMATED
#: date is a derivation from filing cadence, and a reaction measured around a
#: day the company may never have reported on is a number about nothing.
STUDY_STATUSES: frozenset[str] = frozenset(
    {EventStatus.CONFIRMED.value, EventStatus.REVISED.value}
)

#: Analysis statuses whose bundle is trustworthy evidence. FAILED rows carry a
#: bundle too, and it is a perfectly good bundle — the failure was the
#: provider's — so it is INCLUDED: the study measures features against
#: outcomes and never reads the model's text. Only BUNDLE_ONLY, OK, INVALID,
#: SUPERSEDED and FAILED exist, and all five carry NOT NULL evidence, so this
#: is a list of everything rather than a filter. It is written out anyway so
#: that a status added later has to be considered rather than silently
#: inherited.
ANALYSIS_STATUSES: frozenset[str] = frozenset(
    {"OK", "INVALID", "FAILED", "BUNDLE_ONLY", "SUPERSEDED"}
)

#: The basis whose stored straddle is a point-in-time claim about THIS print.
#: ``LIVE_CHAIN_SNAPSHOT`` rows are excluded on purpose: a live snapshot is
#: written when somebody opened the page, which for a past event may be long
#: after the print, and §85 says the platform cannot reconstruct a live chain
#: point-in-time. Correlating a possibly-post-event snapshot with the reaction
#: it was taken after would be the single largest look-ahead this report could
#: contain — and it would look like the strongest result in the table.
METRIC_BASIS = "HISTORICAL_DAILY_CLOSE_APPROXIMATION"


def _as_utc(value: datetime | None) -> datetime | None:
    """Stored instants are UTC; SQLite hands them back naive."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Stored reads
# ---------------------------------------------------------------------------


async def _study_events(
    session: AsyncSession,
    *,
    event_type: str | None,
    as_of: datetime,
) -> list[EventRow]:
    """Single-name, observed-date events at or before ``as_of``, oldest first.

    ``as_of`` bounds the sample the same way every other Catalyst read is
    bounded: an event scheduled after the instant being asked about has no
    outcome yet by definition, and including it would only pad ``n_events``
    with rows that can never contribute a pairing.
    """
    stmt = (
        select(EventRow)
        .where(
            EventRow.ticker.is_not(None),
            EventRow.scheduled_at <= as_of,
            EventRow.status.in_(sorted(STUDY_STATUSES)),
        )
        .order_by(EventRow.scheduled_at)
        .limit(MAX_EVENTS)
    )
    if event_type:
        stmt = stmt.where(EventRow.event_type == event_type)
    rows = (await session.execute(stmt)).scalars().all()
    return [row for row in rows if (row.ticker or "").strip()]


async def _bundles_for(
    session: AsyncSession, event_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """``{event_id: bundle}`` — the EARLIEST stored bundle per event.

    Earliest, not latest, and the choice is the whole point of the §86 gate.
    An event can carry several analyses at several ``as_of`` instants, and a
    re-run performed AFTER the print would have assembled its bundle from bars
    that already contain the reaction — its ``price_runup_pct`` would be
    measured through the event it is supposed to predict. Ordering by ``as_of``
    and keeping the first gives the earliest viewpoint anybody actually took,
    which is the only one that is unambiguously pre-event evidence.

    One query for the whole sample; per-event lookups would be one round trip
    per row on a page that already loads bars per ticker.
    """
    if not event_ids:
        return {}
    rows = (
        (
            await session.execute(
                select(EventAnalysisRow)
                .where(
                    EventAnalysisRow.event_id.in_(event_ids),
                    EventAnalysisRow.status.in_(sorted(ANALYSIS_STATUSES)),
                )
                .order_by(EventAnalysisRow.as_of, EventAnalysisRow.id)
            )
        )
        .scalars()
        .all()
    )
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.event_id in out:
            continue
        bundle = row.bundle
        if isinstance(bundle, dict) and bundle:
            out[row.event_id] = bundle
    return out


async def _metrics_for(
    session: AsyncSession, event_ids: list[int]
) -> dict[int, EventOptionMetricRow]:
    """``{event_id: metrics}`` on the HISTORICAL basis only (see
    :data:`METRIC_BASIS` for why LIVE rows are excluded)."""
    if not event_ids:
        return {}
    rows = (
        (
            await session.execute(
                select(EventOptionMetricRow).where(
                    EventOptionMetricRow.event_id.in_(event_ids),
                    EventOptionMetricRow.basis == METRIC_BASIS,
                )
            )
        )
        .scalars()
        .all()
    )
    return {row.event_id: row for row in rows}


async def _bars_for(session: AsyncSession, tickers: list[str]) -> dict[str, list]:
    """``{ticker: [DailyBar]}`` from STORED bars — never fetched.

    One query across every ticker in the sample rather than one per ticker:
    a study over eight symbols would otherwise be eight round trips before a
    single correlation is computed.
    """
    if not tickers:
        return {}
    rows = (
        (
            await session.execute(
                select(StockBarDaily)
                .where(StockBarDaily.ticker.in_(tickers))
                .order_by(StockBarDaily.ticker, StockBarDaily.ts)
            )
        )
        .scalars()
        .all()
    )
    grouped: dict[str, list[StockBarDaily]] = {}
    for row in rows:
        grouped.setdefault(row.ticker, []).append(row)
    return {ticker: to_daily_bars(items) for ticker, items in grouped.items()}


# ---------------------------------------------------------------------------
# Outcomes (§17) — the same measurement the replay tab renders
# ---------------------------------------------------------------------------


def outcomes_for(bars: list, event_row: EventRow) -> tuple[float | None, float | None]:
    """``(signed_1d, signed_5d)`` for one event from stored bars.

    Delegated to :func:`~libs.trading_core.events.reaction.event_reaction`, so
    the session rule (AMC reacts next morning, BMO the same day) is the one the
    rest of the platform already applies. ``None`` where the reaction has not
    completed — the 5th session after a print three days ago has not closed,
    and a partial window measured as if it were complete is the easiest way to
    manufacture a correlation.

    Bars are NOT as-of filtered here, and that is correct rather than an
    oversight: the OUTCOME is deliberately measured with hindsight — it is the
    realised reaction, the thing the features are being scored against. The
    look-ahead discipline belongs on the FEATURE side, and it is enforced there
    by reading features only out of a bundle stamped before the print.
    """
    if not bars:
        return None, None
    result = event_reaction(
        bars, event_date_et(event_row), _session_of(event_row), horizons=HORIZONS
    )
    if not result.bars_available:
        return None, None
    return result.returns.get(1), result.returns.get(5)


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


async def collect_rows(
    session: AsyncSession,
    *,
    as_of: datetime,
    event_type: str | None = None,
) -> tuple[list[FeatureRow], dict[str, Any]]:
    """``(rows, provenance)`` — the study sample assembled from stored rows.

    ``provenance`` is the sample's own audit trail: how many registry rows
    were in scope, how many of them carried a stored bundle (the ones that can
    contribute a feature), how many carried stored option metrics, and the
    date span. It is what lets a reader tell "this feature has 8% coverage"
    from "this install has barely been backfilled", which are the same number
    with very different meanings.
    """
    moment = _as_utc(as_of) or datetime.now(timezone.utc)
    events = await _study_events(session, event_type=event_type, as_of=moment)
    event_ids = [row.id for row in events]
    bundles = await _bundles_for(session, event_ids)
    metrics = await _metrics_for(session, event_ids)
    tickers = sorted({(row.ticker or "").strip() for row in events if row.ticker})
    bars_by_ticker = await _bars_for(session, [t for t in tickers if t])

    studied = [row for row in events if row.id in bundles]
    items: list[dict[str, Any]] = []
    for row in studied:
        ticker = (row.ticker or "").strip()
        outcome_1d, outcome_5d = outcomes_for(bars_by_ticker.get(ticker, []), row)
        metric = metrics.get(row.id)
        items.append(
            {
                "event_id": row.id,
                "event_key": row.event_key,
                "ticker": ticker or None,
                "event_date": event_date_et(row),
                "bundle": bundles.get(row.id),
                "option_metrics": (
                    {
                        "implied_move_pct": metric.implied_move_pct,
                        "iv_before": metric.iv_before,
                        "basis": metric.basis,
                    }
                    if metric is not None
                    else None
                ),
                "outcome_1d": outcome_1d,
                "outcome_5d": outcome_5d,
            }
        )

    dates = [item["event_date"] for item in items if item["event_date"]]
    provenance = {
        "events_in_scope": len(events),
        "events_with_stored_bundle": len(studied),
        "events_with_option_metrics": sum(
            1 for row in studied if row.id in metrics
        ),
        "tickers": len([t for t in tickers if t]),
        "tickers_with_stored_bars": sum(
            1 for t in tickers if bars_by_ticker.get(t)
        ),
        "first_event_date": min(dates).isoformat() if dates else None,
        "last_event_date": max(dates).isoformat() if dates else None,
        "event_type": event_type,
        "as_of": moment.isoformat().replace("+00:00", "Z"),
        "option_metric_basis": METRIC_BASIS,
        "bundle_selection": (
            "the EARLIEST stored analysis per event — a later re-run may have "
            "assembled its evidence after the print"
        ),
        "max_events": MAX_EVENTS,
    }
    return collect_feature_rows(items), provenance


async def build_study_payload(
    session: AsyncSession,
    *,
    as_of: datetime,
    event_type: str | None = None,
    min_n: int | None = None,
) -> dict[str, Any]:
    """The whole ``GET /api/events/study`` response.

    ``min_n`` does NOT filter the sample and cannot change a rho: it raises the
    bar at which a cell is flagged, so a reader who considers twelve events too
    few can ask for thirty and watch the table go dark. Lowering it below the
    library's own :data:`~libs.trading_core.events.event_study.MIN_MEANINGFUL_N`
    is refused by the endpoint, because the one thing this parameter must never
    be able to do is make a four-event correlation look quotable.

    ``insufficient_data`` is ``True`` when nothing in the store can be
    measured. It is not an error and not a 404 — an install that has analysed
    no events yet has a perfectly well-defined §86 answer ("nothing has been
    measured"), and the caveats and the feature list are the useful half of
    the response even then.
    """
    rows, provenance = await collect_rows(
        session, as_of=as_of, event_type=event_type
    )
    report = feature_report(rows)

    if min_n is not None and min_n > report["min_meaningful_n"]:
        report["min_meaningful_n"] = int(min_n)
        report["min_n_override"] = int(min_n)
        for stats in report["features"].values():
            for kind in ("signed", "absolute"):
                for cell in stats[kind].values():
                    cell["not_meaningful"] = cell["n"] < int(min_n)
                    cell["flag"] = NOT_MEANINGFUL if cell["not_meaningful"] else None
            # Re-point the primary aliases at the (now mutated) underlying
            # cells: they are convenience views of the same measurement, and a
            # raised bar that showed through only the nested one would let a
            # UI reading the alias quote a cell the report has just flagged.
            kind = stats["primary"]
            stats["rho_1d"] = stats[kind]["rho_1d"]
            stats["rho_5d"] = stats[kind]["rho_5d"]

    return {
        "model_version": EVENT_STUDY_MODEL_VERSION,
        "as_of": provenance["as_of"],
        "insufficient_data": report["outcome_coverage"]["outcome_1d"] == 0,
        "report": report,
        "provenance": provenance,
        "rows": [row.to_dict() for row in rows[:ROW_LIMIT]],
        "rows_total": len(rows),
        "rows_limit": ROW_LIMIT,
        "provenance_tier": "QUANT",
        "not_a_signal": (
            "RESEARCH ONLY (§87). Nothing in this payload is wired to a "
            "signal, a trade plan or an order, and no code path reads it into "
            "one. It is a measurement of this installation's own stored "
            "history, offered so the §86 question can be argued about with "
            "numbers instead of intuitions."
        ),
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "HORIZONS",
    "MAX_EVENTS",
    "METRIC_BASIS",
    "ROW_LIMIT",
    "STUDY_STATUSES",
    "build_study_payload",
    "collect_rows",
    "outcomes_for",
]
