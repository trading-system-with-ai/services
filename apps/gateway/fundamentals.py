"""Fundamentals context — the gateway seam (Phase E2, U3; event spec §16, §28,
§29, §30, §33, §35, §85, §96; audit §7, §11.3).

THE SPLIT THIS MODULE EXISTS TO KEEP, exactly as ``event_price.py`` keeps it
for prices. Every number in the payload is computed by
``libs/trading_core/events/fundamentals.py``, which is pure stdlib and may not
import ``apps/`` or ``libs.market_data`` (audit §7.4). This module is the only
place the two halves meet: it fetches filings through the market-data
provider, MIRRORS them into ``fundamental_statements``, reads them back, hands
the ORM rows to the library and renders the frozen results as JSON. It
computes no ratio itself — the one arithmetic expression here is
``price × diluted shares``, and it lives here only because ``price`` comes
from stored bars, which the pure layer may not read.

INGESTION AND ANALYSIS ARE SEPARATE CALL GRAPHS (audit §7.2 rule 1).
:func:`ensure_fundamentals` holds the provider handle and writes rows; it
takes no ``as_of``. :func:`build_fundamentals_context` takes ``as_of``, reads
STORED rows only, and never touches a provider except by asking
``ensure_fundamentals`` to top the mirror up first. That is why a historical
question is answerable at all: the filings the platform has are on disk with
their acceptance instants, and the gate is applied to those rows.

THE AS-OF GATE IS ON ``acceptance_datetime``, NOWHERE ELSE (§85, §96; audit
§7.1). This module does not implement the gate — ``select_statements_as_of``
does, once — but it must not defeat it, so the SQL here deliberately does not
filter statements by date at all: every stored row for the ticker is loaded
and the pure layer decides what was public. A WHERE clause on ``end_date``
here would be the exact look-ahead the audit's sentinel test plants, and it
would be invisible to the library's own tests. The bar used to price the
multiples goes through ``as_of_bar_filter`` (the §14 16:00-ET rule) for the
same reason.

HONEST ABSENCE, NEVER A ZERO (§44 rule 18, §85, §33). Each failure keeps its
own shape rather than collapsing into an empty payload:

- no market-data provider, or a plan without fundamentals (403) -> the
  ``statements`` block is ``{"available": false, "reason": ...}``, stored rows
  (possibly none) still serve, and the endpoint answers 200. The event's
  registry facts are real and are not hidden because a vendor is unpaid.
- no filing was public at ``as_of`` -> ``available: false`` with the library's
  own reason, every metric ``null``, and the excluded-row counts spelled out.
- a metric whose input the provider does not report (capex, cash, D&A) ->
  ``null`` plus the structural reason, collected into ``unavailable`` so the
  UI prints "Unavailable — capex not reported by provider" without knowing
  which formula produced it.
- CONSENSUS is unavailable at ANY instant, not merely historically (audit
  §7.3): Massive's Benzinga estimates endpoints 403 across the board. §33/§98
  require that to be stated, so the block is always present and always
  ``available: false`` — an omitted block would read as "not applicable" and
  a computed EPS surprise would be a fabrication.

PROVENANCE IS LABELLED AT BLOCK LEVEL (§91): the statement values are DATA (a
filer's own XBRL, cited back to ``source_filing_url``), everything derived
from them is QUANT (this platform's arithmetic). Nothing here is
LLM-generated. ``not_backtestable`` carries the §85/audit §7.3 labels for what
a fundamentals view cannot supply point-in-time — consensus, and the restated
periods the vendor overwrites — because an empty list would claim the whole
tab is reconstructable at any historical instant, which is false.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.market_data import (
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
    get_provider,
)
from libs.trading_core.events import previous_comparable
from libs.trading_core.events.fundamentals import (
    FUNDAMENTALS_MODEL_VERSION,
    METRIC_ORDER,
    FundamentalSnapshot,
    MetricChange,
    build_snapshot,
    expectations_gap_inputs,
    snapshot_change,
    valuation_context,
)
from libs.trading_core.events.reaction import as_of_bar_filter
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import EventType

from . import audit
from .db import EventRow, FundamentalStatementRow
from .event_calendar import row_to_event
from .event_price import (
    COMPARABLE_STATUSES,
    _as_utc,
    to_daily_bars,
)
from .routers.analysis import ensure_daily_bars

logger = logging.getLogger(__name__)

#: The audit ``entity_type`` for a fundamentals backfill — the table it wrote,
#: matching the ``stock_bars_daily`` precedent in ``ensure_daily_bars``.
ENTITY_TYPE = "fundamental_statements"

#: Quarterly periods fetched per refresh. Twelve is three fiscal years, which
#: is what the §29 trend column (eight quarters) and the year-over-year
#: comparison (five quarters) need with room for a filer that skips a period.
QUARTERLY_LIMIT = 12

#: TTM periods fetched per refresh. The snapshot uses the newest one; the
#: extra three give the §30 own-history multiples something to be a history OF
#: once earlier snapshots are priced at their own dates.
TTM_LIMIT = 4

#: How stale the newest STORED acceptance instant may get before a refresh is
#: due. Twenty hours, not twenty-four: a filer that accepts at 16:05 ET on
#: consecutive days must not be missed by a window that happens to align with
#: the previous day's fetch. Companies file quarterly, so this is generous —
#: it exists to catch the filing, not to poll.
REFRESH_AFTER_HOURS = 20

#: Minimum spacing between provider ATTEMPTS per ticker, in seconds. A ticker
#: whose newest filing is three months old is permanently "stale" by the rule
#: above, so without this every request would re-ask the vendor for statements
#: that will not exist for another six weeks. Same shape and the same reason
#: as ``analysis.REFRESH_ATTEMPT_SECONDS`` for bars.
REFRESH_ATTEMPT_SECONDS = 6 * 60 * 60

#: Per-ticker last provider attempt (success or failure), process-local like
#: ``analysis._refresh_attempts``. A restart re-attempts, which is correct: a
#: cold process has no evidence the vendor is still failing.
_refresh_attempts: dict[str, datetime] = {}

#: §33/§98 + audit §7.3 — consensus is unavailable at ANY instant, not merely
#: un-backtestable. Stated once here so every payload says it identically.
CONSENSUS_REASON = (
    "CONSENSUS DATA UNAVAILABLE — Massive Benzinga earnings/estimates not in "
    "subscription (403)"
)

#: §85 / audit §7.3 — what this view deliberately does NOT claim to
#: reconstruct point-in-time, carried in the payload so the UI labels it.
NOT_BACKTESTABLE: tuple[str, ...] = (
    "consensus_eps",
    "consensus_revenue",
    "estimate_revisions",
    "guidance",
    "analyst_ratings",
    "restated_prior_periods",
)


# ---------------------------------------------------------------------------
# ORM -> pure value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StoredStatement:
    """One stored row re-stamped as aware UTC, for the pure layer.

    WITHOUT THIS EVERY STORED FILING IS INVISIBLE, and silently so. SQLite
    hands ``TIMESTAMPTZ`` columns back NAIVE (the same convention
    ``event_price._as_utc`` exists for), and ``coerce_statement`` REFUSES a
    naive acceptance instant rather than assuming UTC — correctly, because an
    unknown zone cannot be proven to precede ``as_of`` and guessing would move
    the §85 boundary by hours. The two rules are each right and together they
    would drop every row the database returns, with no error and no reason
    string: the payload would just say "no statement was public at as_of"
    forever. Re-stamping the instant here, at the ORM boundary, is what keeps
    the refusal in the pure layer meaningful — after this point a ``None``
    acceptance really does mean the PROVIDER omitted it.

    It satisfies ``StatementLike`` structurally, so the pure module still
    imports nothing from ``apps/``.
    """

    fiscal_year: int | None
    fiscal_period: str
    end_date: date | None
    acceptance_datetime: datetime | None
    values: Mapping[str, float]
    timeframe: str | None = None
    filing_date: date | None = None


def to_statements(rows: Sequence[FundamentalStatementRow]) -> list[_StoredStatement]:
    """Stored rows as the pure library's duck-typed input, order preserved.

    Only the acceptance instant is transformed; nothing else is touched, and
    in particular ``values`` is passed through as stored — a field the filer
    did not report stays ABSENT rather than becoming a zero on the way out
    (§44 rule 18).
    """
    return [
        _StoredStatement(
            fiscal_year=row.fiscal_year,
            fiscal_period=row.fiscal_period,
            end_date=row.end_date,
            acceptance_datetime=(
                _as_utc(row.acceptance_datetime)
                if row.acceptance_datetime is not None
                else None
            ),
            values=dict(row.values or {}),
            timeframe=row.timeframe,
            filing_date=row.filing_date,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Ingestion — the only half of this module that holds a provider handle
# ---------------------------------------------------------------------------


def _period_key(row: FundamentalStatementRow) -> tuple:
    """The natural key of a filed period — the UNIQUE constraint, in Python.

    Spelled once so the in-memory upsert dedupe and the database constraint
    can never disagree about what "the same period" means.
    """
    return (
        row.ticker,
        row.timeframe,
        row.fiscal_year,
        row.fiscal_period,
        row.end_date,
    )


def _statement_key(statement) -> tuple:
    """:func:`_period_key` for a provider :class:`FinancialStatement`."""
    return (
        (statement.ticker or "").strip().upper(),
        (statement.timeframe or "").strip().lower(),
        statement.fiscal_year,
        (statement.fiscal_period or "").strip().upper(),
        statement.end_date,
    )


async def _stored_rows(
    session: AsyncSession, ticker: str
) -> list[FundamentalStatementRow]:
    """Every stored statement for ``ticker``, newest acceptance first.

    DELIBERATELY UNFILTERED BY DATE. The as-of gate belongs to the pure layer
    (``select_statements_as_of``); narrowing here would place a second,
    untested copy of the §85 rule in SQL, and the two would drift. Rows whose
    acceptance instant is NULL sort last rather than being dropped: they are
    real rows, they are why the payload can say "3 statements excluded, no
    acceptance_datetime", and the library refuses them for every as-of anyway.
    """
    rows = (
        (
            await session.execute(
                select(FundamentalStatementRow)
                .where(FundamentalStatementRow.ticker == ticker)
                .order_by(FundamentalStatementRow.end_date.desc())
            )
        )
        .scalars()
        .all()
    )
    return sorted(
        rows,
        key=lambda row: (
            _as_utc(row.acceptance_datetime)
            if row.acceptance_datetime is not None
            else datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )


def _refresh_is_due(
    stored: list[FundamentalStatementRow], now: datetime
) -> bool:
    """Whether the mirror should be topped up from the provider.

    Two cases, both about the MIRROR rather than the market: nothing stored at
    all (first call for this ticker), or the newest stored acceptance instant
    is older than :data:`REFRESH_AFTER_HOURS`. The second is true almost
    always — filings are quarterly — which is exactly why the attempt throttle
    below exists and why this predicate alone must never reach the vendor.
    """
    if not stored:
        return True
    newest = max(
        (
            _as_utc(row.acceptance_datetime)
            for row in stored
            if row.acceptance_datetime is not None
        ),
        default=None,
    )
    if newest is None:
        return True
    return (now - newest) > timedelta(hours=REFRESH_AFTER_HOURS)


def _to_row(statement, *, ticker: str, now: datetime) -> FundamentalStatementRow:
    """One provider statement as an unsaved ORM row.

    ``values`` is stored verbatim as the provider flattened it — floats only,
    unreported fields ABSENT. Nothing is defaulted to 0.0 on the way in: a
    field missing here must stay missing so the pure layer can say "not
    reported" instead of "reported zero" (§44 rule 18).
    """
    accepted = statement.acceptance_datetime
    if accepted is not None and accepted.tzinfo is not None:
        accepted = accepted.astimezone(timezone.utc)
    return FundamentalStatementRow(
        ticker=ticker,
        cik=statement.cik,
        timeframe=(statement.timeframe or "").strip().lower(),
        fiscal_year=statement.fiscal_year,
        fiscal_period=(statement.fiscal_period or "").strip().upper(),
        start_date=statement.start_date,
        end_date=statement.end_date,
        filing_date=statement.filing_date,
        acceptance_datetime=accepted,
        source_filing_url=statement.source_filing_url,
        values=dict(statement.values),
        raw_fields_count=statement.raw_fields_count,
        fetched_at=now,
    )


def _apply_update(row: FundamentalStatementRow, statement, *, now: datetime) -> bool:
    """Overwrite ``row`` from ``statement`` when the filing actually changed.

    RESTATEMENTS OVERWRITE (audit §7.3). The vendor serves only its current
    XBRL view, so a restated Q3 arrives under the same natural key with a
    later acceptance instant and new values; keeping the superseded original
    would claim a point-in-time archive this platform does not have. Returns
    whether anything moved, so an idempotent re-fetch writes no audit row and
    leaves ``fetched_at`` alone — a freshness stamp that ticks when nothing
    changed is a freshness stamp nobody can read.
    """
    accepted = statement.acceptance_datetime
    if accepted is not None and accepted.tzinfo is not None:
        accepted = accepted.astimezone(timezone.utc)
    incoming = {
        "cik": statement.cik,
        "start_date": statement.start_date,
        "filing_date": statement.filing_date,
        "acceptance_datetime": accepted,
        "source_filing_url": statement.source_filing_url,
        "values": dict(statement.values),
        "raw_fields_count": statement.raw_fields_count,
    }
    changed = False
    for name, value in incoming.items():
        current = getattr(row, name)
        if name == "acceptance_datetime":
            current = _as_utc(current) if current is not None else None
        if current != value:
            setattr(row, name, value)
            changed = True
    if changed:
        row.fetched_at = now
    return changed


def fundamentals_provider_name(settings) -> str:
    """Which market-data provider serves FINANCIAL STATEMENTS.

    Massive is the only provider with ``/vX/reference/financials`` (audit
    §3), and the platform's market-data provider is Alpaca (DEVLOG
    2026-08-13) — so statements must come from Massive whenever its key is
    configured, regardless of ``market_data_provider``. Prices for valuation
    keep coming from the market-data provider (see ``price_provider_name``
    on :func:`build_fundamentals_context`). Falls back to the market-data
    provider, whose ``get_financials`` then honestly raises
    ``CapabilityNotAvailable`` (Alpaca) — stored rows are served, never a
    fabricated statement.
    """
    if (getattr(settings, "massive_api_key", "") or "").strip():
        return "massive"
    return getattr(settings, "market_data_provider", "") or ""


async def ensure_fundamentals(
    session: AsyncSession,
    ticker: str,
    provider_name: str,
    *,
    now: datetime,
) -> list[FundamentalStatementRow]:
    """Stored statements for ``ticker``, newest acceptance first, topped up.

    The fundamentals counterpart of ``ensure_daily_bars`` and deliberately the
    ONLY path that writes ``fundamental_statements``, so the DATA_BACKFILL
    audit trail and the provenance story stay single-sourced (rule 12,
    ADR-003). On the first call for a symbol, or when the newest stored
    acceptance instant is older than :data:`REFRESH_AFTER_HOURS` AND the
    per-ticker attempt throttle has expired, it fetches ``quarterly`` and
    ``ttm`` periods, upserts them on the natural key and records ONE SYSTEM
    ``DATA_BACKFILL`` event in the same transaction.

    ``now`` is REQUIRED and is the only clock this function reads — the
    throttle and ``fetched_at`` both come from it, so a test can drive the
    refresh cadence without patching time. It is NOT an ``as_of``: ingestion
    has no as-of (audit §7.2 rule 1), it writes what the vendor currently
    serves and lets the analysis half decide what was public when.

    EVERY PROVIDER FAILURE SERVES STORED ROWS AND RAISES NOTHING. An
    unconfigured provider, a plan without fundamentals (403 ->
    :class:`CapabilityNotAvailable`), a transport error and an unexpected
    exception all return whatever is mirrored — possibly ``[]``. A read
    endpoint must not 5xx because a vendor is unpaid; the caller renders the
    absence with its reason, which is the §16 capability answer.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return []

    stored = await _stored_rows(session, symbol)
    if not _refresh_is_due(stored, now):
        return stored

    last_attempt = _refresh_attempts.get(symbol)
    if (
        last_attempt is not None
        and (now - last_attempt).total_seconds() < REFRESH_ATTEMPT_SECONDS
    ):
        # A quarterly filer is "stale" by the rule above for ~89 days out of
        # 90. Without this the endpoint would re-ask the vendor on every
        # request for a filing that does not exist yet.
        return stored
    _refresh_attempts[symbol] = now

    try:
        provider = get_provider(provider_name)
        fetched = list(
            provider.get_financials(symbol, timeframe="quarterly", limit=QUARTERLY_LIMIT)
        ) + list(provider.get_financials(symbol, timeframe="ttm", limit=TTM_LIMIT))
    except (ProviderNotConfigured, CapabilityNotAvailable, MarketDataError) as exc:
        # Named, expected refusals: no provider, no subscription, vendor
        # error. Stored rows still serve; the caller states the reason.
        logger.info(
            "fundamentals_fetch_unavailable",
            extra={"extra_fields": {"ticker": symbol, "reason": str(exc)}},
        )
        return stored
    except Exception:
        logger.exception(
            "fundamentals_fetch_failed", extra={"extra_fields": {"ticker": symbol}}
        )
        return stored

    existing = {_period_key(row): row for row in stored}
    inserted: list[FundamentalStatementRow] = []
    updated = 0
    seen: set[tuple] = set()
    for statement in fetched:
        key = _statement_key(statement)
        if key in seen:
            # The provider served the same period twice in one response (an
            # amended filing alongside the original). The later one in the
            # list already won; re-applying it would flip-flop the row.
            continue
        seen.add(key)
        row = existing.get(key)
        if row is None:
            row = _to_row(statement, ticker=symbol, now=now)
            session.add(row)
            existing[key] = row
            inserted.append(row)
        elif _apply_update(row, statement, now=now):
            updated += 1

    if not inserted and not updated:
        # Nothing moved: no audit row (an audit trail of no-ops is noise) and
        # no commit. The attempt is still throttled above, so this is cheap.
        return stored

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.DATA_BACKFILL,
        entity_type=ENTITY_TYPE,
        entity_id=symbol,
        details={
            "kind": "fundamentals",
            "ticker": symbol,
            "rows": len(inserted) + updated,
            "inserted": len(inserted),
            "updated": updated,
            "provider": provider_name,
        },
    )
    await session.commit()
    return await _stored_rows(session, symbol)


# ---------------------------------------------------------------------------
# Price for the valuation leg — stored bars only, gated at as_of
# ---------------------------------------------------------------------------


async def _price_at(
    session: AsyncSession, ticker: str, provider_name: str, as_of: datetime
) -> tuple[float | None, date | None, str | None]:
    """``(last_close, its_date, reason_if_unavailable)`` knowable at ``as_of``.

    The multiples need a price the platform could actually have seen, so the
    bars go through ``as_of_bar_filter`` — the §14 rule that a same-day bar
    does not exist until 16:00 ET — before the last close is taken. Pricing a
    P/E off today's close in an answer dated last October is the same
    look-ahead as reading an unfiled quarter, just wearing a price's clothes.

    Every market-data failure becomes a reason string and the multiples become
    ``None`` with it; nothing here raises, because a missing price must not
    remove the statement metrics that do not need one.
    """
    try:
        rows = await ensure_daily_bars(session, ticker, provider_name)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            reason = str(detail.get("message") or detail.get("code") or detail)
        else:
            reason = str(detail)
        return None, None, reason
    except (ProviderNotConfigured, MarketDataError) as exc:
        return None, None, str(exc)
    if not rows:
        return None, None, f"no stored bars for {ticker}"
    bars = as_of_bar_filter(to_daily_bars(rows), as_of)
    if not bars:
        return None, None, (
            f"no bar for {ticker} was knowable at {as_of.isoformat()}"
        )
    last = bars[-1]
    return last.close, last.date, None


def _market_cap(
    price: float | None, shares_diluted: float | None
) -> tuple[float | None, str | None]:
    """``price × diluted shares``, or ``None`` with the reason it could not be.

    THE ONE ARITHMETIC EXPRESSION IN THIS MODULE, and it is here only because
    ``price`` comes from stored bars, which the pure layer may not read. Both
    legs must be positive: a zero or negative share count is malformed input,
    and multiplying it out would print a market cap of 0.0 — a fabricated
    number that looks computed, which is precisely what §44 rule 18 forbids.
    Diluted shares are the honest denominator (basic ignores the options and
    convertibles a real capitalisation includes).
    """
    if price is None or price <= 0.0:
        return None, "market cap needs a price knowable at as_of"
    shares = shares_diluted
    if shares is None or shares <= 0.0:
        return None, (
            "market cap needs diluted average shares, which the newest "
            "statement does not report"
        )
    return price * shares, None


# ---------------------------------------------------------------------------
# Rendering — frozen results to JSON
# ---------------------------------------------------------------------------


def _date_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _instant_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _as_utc(value).isoformat()


def _ref_to_dict(ref) -> dict | None:
    """One :class:`StatementRef` as JSON — WHICH filing produced the numbers.

    ``acceptance_datetime`` travels with every ref on purpose (§85): a reader
    cannot check an as-of claim against a fiscal label alone, and "FY2025 Q3"
    means nothing about when it became knowable.
    """
    if ref is None:
        return None
    return {
        "label": ref.label,
        "fiscal_year": ref.fiscal_year,
        "fiscal_period": ref.fiscal_period,
        "end_date": _date_iso(ref.end_date),
        "filing_date": _date_iso(ref.filing_date),
        "acceptance_datetime": _instant_iso(ref.acceptance_datetime),
        "timeframe": ref.timeframe,
    }


def snapshot_to_dict(snapshot: FundamentalSnapshot) -> dict:
    """One :class:`FundamentalSnapshot` as JSON (§28).

    ``metrics`` is rendered in :data:`METRIC_ORDER` rather than as the raw
    mapping so the UI's row order is the library's, not a JSON accident, and
    every metric name is always present — a metric that vanished from the
    object would be indistinguishable from one the client forgot to render.
    ``reasons`` and ``notes`` ride alongside because a ``null`` without its
    reason is the exact shape §44 rule 18 forbids.
    """
    return {
        "ticker": snapshot.ticker,
        "as_of": _instant_iso(snapshot.as_of),
        "available": snapshot.available,
        "quarterly": _ref_to_dict(snapshot.quarterly),
        "ttm": _ref_to_dict(snapshot.ttm),
        "metrics": {name: snapshot.metrics.get(name) for name in METRIC_ORDER},
        "reasons": dict(snapshot.reasons),
        "notes": dict(snapshot.notes),
        "quarters_available": snapshot.quarters_available,
        "price": snapshot.price,
        "market_cap": snapshot.market_cap,
        "model_version": snapshot.model_version,
    }


def change_to_dict(change: MetricChange) -> dict:
    """One :class:`MetricChange` as the §29 comparison row.

    ``arrow`` is rendered here from the library's ``direction`` so every
    client draws the same glyph, and ``delta_bps`` is present-but-null for the
    dollar metrics rather than absent — "basis points do not apply to a
    revenue in dollars" is a fact about the metric, and a key that appears for
    some rows and not others makes the UI guess.
    """
    return {
        "metric": change.metric,
        "previous": change.previous,
        "current": change.current,
        "delta": change.delta,
        "delta_bps": change.delta_bps,
        "pct_change": change.pct_change,
        "direction": change.direction,
        "arrow": change.arrow,
        "trend": change.trend,
        "trend_points": change.trend_points,
        "reason": change.reason,
        "note": change.note,
    }


def _valuation_to_dict(context: dict) -> dict:
    """The §30 valuation block with its ``as_of`` instant JSON-safe.

    The library returns a real ``datetime`` there (it is a value object, not a
    payload); every other key is already JSON-native, so this is a targeted
    conversion rather than a re-shaping — a second copy of the block's
    structure here would be one more thing to keep in sync with §30.
    """
    rendered = dict(context)
    rendered["as_of"] = _instant_iso(context.get("as_of"))
    return rendered


def _unavailable_entries(prefix: str, reasons) -> list[dict]:
    """Flatten a reasons mapping into ``[{"field", "reason"}]`` rows.

    One flat list across the payload is what lets the UI render "Unavailable —
    <reason>" generically, without knowing which formula owns which key
    (identical to ``event_price._unavailable_entries``).
    """
    return [
        {"field": f"{prefix}.{key}", "reason": value}
        for key, value in sorted(reasons.items())
    ]


# ---------------------------------------------------------------------------
# Registry reads — the previous comparable event
# ---------------------------------------------------------------------------


async def _past_comparable_rows(
    session: AsyncSession, event_row: EventRow, as_of: datetime
) -> list[EventRow]:
    """Past comparable EARNINGS rows for this ticker, oldest first.

    The same two point-in-time filters ``event_price`` applies, for the same
    reasons: strictly before this event's own instant (a later print is not a
    precedent) and at or before ``as_of`` (a print the platform could not have
    seen at ``as_of`` is not history yet, however firmly the registry knows
    about it today). Status is narrowed to CONFIRMED/REVISED per §15 — an
    ESTIMATED past date is a derivation, and taking a fundamentals snapshot at
    a day nobody reported on would anchor the whole comparison to a fiction.
    """
    ticker = event_row.ticker
    if not ticker:
        return []
    scheduled = _as_utc(event_row.scheduled_at)
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type == EventType.EARNINGS.value,
                    EventRow.ticker == ticker,
                    EventRow.scheduled_at < event_row.scheduled_at,
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    keep: list[EventRow] = []
    for row in rows:
        if row.id == event_row.id:
            continue
        if row.status not in COMPARABLE_STATUSES:
            continue
        when = _as_utc(row.scheduled_at)
        if when >= scheduled or when > as_of:
            continue
        keep.append(row)
    return keep


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


async def build_fundamentals_context(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider_name: str,
    price_provider_name: str | None = None,
) -> dict:
    """The whole fundamentals block for one event, as of one instant (§16,
    §28, §29, §30, §33, §35).

    Order of operations is the contract: mirror the filings -> load ALL stored
    rows -> price the ticker at ``as_of`` -> let the pure layer gate the rows
    on ``acceptance_datetime`` and compute. Nothing is computed before the
    gate, so no metric can come from a filing the caller could not have seen.
    ``as_of`` is REQUIRED (audit §7.2 rule 2 — a seam that defaults it to
    ``now()`` cannot answer a historical question).

    THE PREVIOUS SNAPSHOT IS TAKEN AT THE PREVIOUS EVENT'S OWN INSTANT, not
    from an earlier row of today's data. "What changed since the last print"
    means the comparison must be against what was KNOWN then — the same
    statement list re-gated at the earlier instant — otherwise a restatement
    filed last week would silently rewrite the "previous" column and the
    delta would describe a change that never happened.

    Non-ticker events (macro, Fed) return ``{"available": false, "reason":
    "no_ticker"}``: a CPI release has no balance sheet, and substituting an
    index proxy would invent an issuer the event does not have.
    """
    ticker = (event_row.ticker or "").strip().upper()
    payload_base = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": event_row.ticker,
        "as_of": _as_utc(as_of).isoformat(),
    }
    if not ticker:
        return {**payload_base, "available": False, "reason": "no_ticker"}

    moment = _as_utc(as_of)
    now = datetime.now(timezone.utc)
    # Statements and prices may come from different providers (statements:
    # Massive; prices: the market-data provider, e.g. Alpaca).
    price_provider = price_provider_name or provider_name

    # --- ingestion (writes rows; no as_of) --------------------------------
    rows = await ensure_fundamentals(session, ticker, provider_name, now=now)

    unavailable: list[dict] = []
    if not rows:
        unavailable.append(
            {
                "field": "statements",
                "reason": (
                    f"no financial statements stored for {ticker} — the "
                    "configured provider serves none or is unavailable"
                ),
            }
        )

    # --- price for the valuation leg (stored bars, gated at as_of) --------
    price, price_date, price_reason = await _price_at(
        session, ticker, price_provider, moment
    )
    if price_reason is not None:
        unavailable.append({"field": "price", "reason": price_reason})

    # --- current snapshot -------------------------------------------------
    # Converted ONCE and reused for every snapshot below: the acceptance
    # instants must be aware-UTC before the pure layer sees them (see
    # :class:`_StoredStatement`), and re-converting per snapshot would be
    # three chances for one of them to be forgotten.
    statements = to_statements(rows)

    # Built once WITHOUT a market cap to learn the diluted share count the
    # as-of-visible statement actually reports; the derived cap then feeds a
    # second build. Deriving it from an unfiltered row would price the company
    # on a share count from a filing that was not public yet.
    probe = build_snapshot(statements, as_of=moment, ticker=ticker, price=price)
    market_cap, cap_reason = _market_cap(price, probe.metrics.get("shares_diluted"))
    current = build_snapshot(
        statements, as_of=moment, ticker=ticker, price=price, market_cap=market_cap
    )
    if cap_reason is not None:
        unavailable.append({"field": "market_cap", "reason": cap_reason})
    unavailable.extend(_unavailable_entries("current", current.reasons))

    # --- previous comparable event & its snapshot -------------------------
    past_rows = await _past_comparable_rows(session, event_row, moment)
    previous_event, comparison_reason = previous_comparable(
        row_to_event(event_row), [row_to_event(row) for row in past_rows]
    )
    anchor_row = None
    if previous_event is not None and previous_event.event_id is not None:
        anchor_row = next(
            (row for row in past_rows if row.id == previous_event.event_id), None
        )

    previous_snapshot: FundamentalSnapshot | None = None
    previous_block: dict | None = None
    if anchor_row is not None:
        anchor_at = _as_utc(anchor_row.scheduled_at)
        anchor_price, anchor_price_date, _ = await _price_at(
            session, ticker, price_provider, anchor_at
        )
        anchor_probe = build_snapshot(
            statements, as_of=anchor_at, ticker=ticker, price=anchor_price
        )
        anchor_cap, _ = _market_cap(
            anchor_price, anchor_probe.metrics.get("shares_diluted")
        )
        previous_snapshot = build_snapshot(
            statements,
            as_of=anchor_at,
            ticker=ticker,
            price=anchor_price,
            market_cap=anchor_cap,
        )
        previous_block = {
            "event_id": anchor_row.id,
            "event_key": anchor_row.event_key,
            "scheduled_at": anchor_at.isoformat(),
            "comparison_reason": comparison_reason,
            "price_date": _date_iso(anchor_price_date),
            "snapshot": snapshot_to_dict(previous_snapshot),
        }
    else:
        unavailable.append(
            {
                "field": "previous_snapshot",
                "reason": (
                    "no previous comparable earnings event was knowable at "
                    f"{moment.isoformat()}"
                ),
            }
        )

    # --- §30 own-history: every past event, priced at ITS OWN date --------
    # Each historical snapshot is gated at that event's instant AND valued
    # with the close that prevailed then, which is what makes "the current P/E
    # against its own history" a real comparison rather than today's price
    # divided by a series of old earnings.
    history: list[FundamentalSnapshot] = []
    for row in past_rows:
        row_at = _as_utc(row.scheduled_at)
        row_price, _, _ = await _price_at(session, ticker, price_provider, row_at)
        row_probe = build_snapshot(
            statements, as_of=row_at, ticker=ticker, price=row_price
        )
        row_cap, _ = _market_cap(row_price, row_probe.metrics.get("shares_diluted"))
        history.append(
            build_snapshot(
                statements,
                as_of=row_at,
                ticker=ticker,
                price=row_price,
                market_cap=row_cap,
            )
        )

    changes = snapshot_change(previous_snapshot, current, history=history)
    valuation = _valuation_to_dict(valuation_context(current, history))
    momentum = expectations_gap_inputs(changes)

    # --- freshness --------------------------------------------------------
    # §96: freshness must describe what was KNOWABLE at ``as_of``, so it is
    # built from the newest row whose acceptance_datetime clears the same
    # gate the metrics go through — NOT from rows[0], which is the newest
    # STORED row and may have been accepted after ``as_of`` (a filing the
    # caller could not have seen; pinned by tests/test_event_lookahead.py).
    # ``statements_stored`` deliberately stays the store-wide count: it
    # describes the platform's holdings, not the as-of view, and carries no
    # future filing's identity.
    visible_rows = [
        row
        for row in rows
        if row.acceptance_datetime is not None
        and _as_utc(row.acceptance_datetime) <= moment
    ]
    newest = visible_rows[0] if visible_rows else None
    freshness = {
        "latest_filing_date": _date_iso(newest.filing_date) if newest else None,
        "acceptance_datetime": (
            _instant_iso(newest.acceptance_datetime) if newest else None
        ),
        "fetched_at": _instant_iso(newest.fetched_at) if newest else None,
        "period_end": _date_iso(newest.end_date) if newest else None,
        "statements_stored": len(rows),
        "provider": provider_name,
        "source_filing_url": newest.source_filing_url if newest else None,
        "price_date": _date_iso(price_date),
        "price_source": price_provider,
    }

    return {
        **payload_base,
        "available": current.available,
        "provenance": {"statements": "DATA", "metrics": "QUANT"},
        "statements": (
            {"available": True, "count": len(rows)}
            if rows
            else {
                "available": False,
                "count": 0,
                "reason": (
                    f"no financial statements stored for {ticker} — the "
                    "configured provider serves none or is unavailable"
                ),
            }
        ),
        "current": snapshot_to_dict(current),
        "previous_event": previous_block,
        "changes": [change_to_dict(change) for change in changes],
        "valuation": valuation,
        "fundamental_momentum": momentum,
        "consensus": {"available": False, "reason": CONSENSUS_REASON},
        "freshness": freshness,
        "metric_order": list(METRIC_ORDER),
        "model_version": FUNDAMENTALS_MODEL_VERSION,
        "not_backtestable": list(NOT_BACKTESTABLE),
        "unavailable": unavailable,
    }
