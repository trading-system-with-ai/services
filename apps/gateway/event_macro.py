"""Macro intelligence — the gateway seam (Phase G, unit U3; event spec §8,
§38-§41, §39 multi-asset, §46 macro_context; audit §6 macro rows, §11.9).

THE SPLIT THIS MODULE EXISTS TO KEEP. Every number in the macro payload is
computed by ``libs/trading_core/events/macro.py``, which is pure stdlib and may
not import ``apps/`` or ``libs.market_data`` (audit §7.4). This module is the
only place the two halves meet: it reads stored observations, stored yield
curves and stored bars, converts ORM rows to the pure library's value types,
hands them to :func:`~libs.trading_core.events.macro.build_macro_packet` and
:func:`~libs.trading_core.events.macro.multi_asset_reaction`, and renders the
frozen results as JSON. It computes nothing itself — no MoM, no basis point
and no return is derived here.

READ AND WRITE ARE TWO DIFFERENT FUNCTIONS AND THAT IS THE CONTRACT (§27;
audit §7.2 rule 1). :func:`build_macro_payload` is DB-ONLY: it holds no
provider handle, cannot reach the network, and answers with honest absence
when nothing has been fetched. :func:`backfill_macro` is the USER action that
spends the requests. The separation matters more here than anywhere else in
the platform because BLS's unregistered API allows roughly TWENTY-FIVE
requests per day: a read endpoint that lazily topped up eight series would
exhaust the day's budget on a single page load and then serve errors to
everyone, including the backfill that could have fixed it.

WHAT THE AS-OF GATE IS ANCHORED TO, and why it is not the fetch time. BLS's
data API returns no timestamps whatsoever — a July CPI observation arrives as
``{"year": "2026", "period": "M07", "value": "324.1"}`` and nothing more. The
instant the number became PUBLIC is a separate fact that lives on the agency's
release schedule, which this platform already stores as CONFIRMED event rows
(``CPI:2026-08-12`` at 08:30 ET). So the backfill JOINS the two: an
observation's ``release_at`` comes from the stored schedule event covering its
reference period (basis SCHEDULED), and only when no such row exists does it
fall back to period-end + 45 days (basis ESTIMATED). Both the instant AND the
basis are stored, because "released at 08:30 on the 12th" and "probably out by
mid-September" are different claims and the second must never be quoted as
the first.

THE JOIN KEY IS THE REFERENCE PERIOD, NEVER THE RELEASE DATE. A schedule event
carries ``events.release_period`` ("2026-07") precisely so a CPI release that
slips a day still matches its own observations. Keying on the release date is
how a one-day slip silently orphans a month of prints.

HONEST ABSENCE, NEVER A ZERO (§44 rule 18). Four different failures each get
their own shape: no stored observations at all (``coverage.actuals.available =
false`` with the "run the backfill" reason), a series the platform has no
adapter for (GDP/PCE need a BEA key, RETAIL_SALES needs a Census adapter — the
pure layer ships an EMPTY catalogue for them and says so), an asset with no
stored bars (listed in ``unavailable`` with its reason, never a 0.0% return),
and a tenor Treasury did not quote (absent from the curve, never 0 bp).

CONSENSUS IS UNAVAILABLE AND SAYS SO, IN EVERY BRANCH (§33, §98). This
platform subscribes to no macro consensus source. There is no code path in
this module or the one below it that computes a surprise; the packet carries
:data:`~libs.trading_core.events.macro.CONSENSUS_UNAVAILABLE_REASON` verbatim,
and the payload repeats it as a top-level ``disclaimer`` so a reader who never
opens ``coverage`` still cannot mistake a missing expectation for a met one.

PROVENANCE IS LABELLED AT BLOCK LEVEL (§49, §91): published statistics and
Treasury curves are DATA (an agency's own numbers), everything derived from
them — the MoM transforms, the trend direction, the cross-asset returns, the
basis-point changes — is QUANT (this platform's arithmetic). Nothing in this
payload is LLM-generated.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from libs.market_data import (
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
    get_provider,
)
from libs.trading_core.events import EASTERN, previous_comparable
from libs.trading_core.events.evidence import TIER_DATA, TIER_QUANT, json_safe
from libs.trading_core.events.macro import (
    ASSET_ROLES,
    CONSENSUS_UNAVAILABLE_REASON,
    DEFAULT_MACRO_CONTEXT_HORIZON_DAYS,
    DEFAULT_MACRO_HORIZONS,
    ESTIMATED_LAG_DAYS,
    MACRO_CONTEXT_EVENT_TYPES,
    MACRO_MODEL_VERSION,
    RELEASE_BASIS_ESTIMATED,
    RELEASE_BASIS_SCHEDULED,
    SURPRISE_UNAVAILABLE_REASON,
    TRACKED_TENORS,
    AssetReaction,
    MacroObservation,
    MacroPacket,
    MultiAssetReaction,
    ScheduleRow,
    YieldChange,
    YieldCurveRow,
    derive_prints,
    macro_context_for,
    multi_asset_reaction,
    period_end_date,
    period_sort_key,
    related_evidence_window,
    series_for,
)
from libs.trading_core.events.reaction import DailyBar, as_of_bar_filter
from libs.trading_core.events.taxonomy import MACRO_EVENT_TYPES
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import EventSession, EventStatus, EventType

from . import audit
from .db import EventRow, MacroObservationRow, StockBarDaily, TreasuryYieldRow
from .event_calendar import row_to_event
from .event_price import _as_utc, _past_comparable_rows, event_date_et
from .routers.market import MACRO_REFERENCE_SYMBOLS

logger = logging.getLogger(__name__)

__all__ = [
    "BACKFILL_BAR_LEAD_DAYS",
    "BACKFILL_SERIES_YEARS",
    "MACRO_DISCLAIMER",
    "backfill_macro",
    "build_macro_payload",
    "ensure_daily_bars",
    "is_macro_event",
    "load_observations",
    "load_schedule_rows",
    "load_yield_curve",
    "macro_context_section",
]

#: How many years of history the backfill asks BLS for. THREE is not a
#: preference, it is the ceiling of the unregistered v1 API — it serves the
#: latest three years and silently truncates a wider ask, so requesting ten
#: would produce three years of data wearing a ten-year label.
BACKFILL_SERIES_YEARS = 3

#: Bars are fetched from ten calendar days BEFORE the previous release so the
#: pre-event close is inside the window even when the release lands on a
#: Tuesday after a long weekend. The reaction needs one bar strictly before
#: the event day, and a window that starts ON the release date has none.
BACKFILL_BAR_LEAD_DAYS = 10

#: The §33 disclaimer, carried at top level in every macro payload. The
#: machine-readable status lives in ``coverage.consensus``; this is the
#: sentence a human reads, and it is present even when everything else in the
#: payload succeeded — a complete-looking macro block with no consensus line
#: is exactly the shape that invites a reader to supply one from memory.
MACRO_DISCLAIMER = (
    "CONSENSUS DATA UNAVAILABLE — this platform subscribes to no macro "
    "forecast source, so no surprise-versus-expectations number exists for "
    "any release here. Every figure shown is a published actual or this "
    "platform's arithmetic over published actuals (§33)."
)

#: The provider label written into ``macro_observations.provider`` /
#: ``treasury_yields.provider`` when the real clients served the row. A stored
#: statistic whose source is unknown cannot be audited.
_TREASURY_PROVIDER = "treasury"


# ---------------------------------------------------------------------------
# Event-type helpers
# ---------------------------------------------------------------------------


def _event_type_of(event_row: EventRow) -> EventType | None:
    """The row's typed event type, or ``None`` for a value the enum dropped.

    ``None`` rather than a raise: a registry row whose type string no longer
    parses is a data problem, and a read endpoint answering "this is not a
    macro event" is a better failure than a 500 out of the enum.
    """
    try:
        return EventType(event_row.event_type)
    except ValueError:
        return None


def is_macro_event(event_row: EventRow) -> bool:
    """Whether this row is one of the §8 macro release types.

    Uses the taxonomy's own :data:`MACRO_EVENT_TYPES` rather than a local
    list, so a type added to the taxonomy gets macro treatment everywhere at
    once instead of in whichever module was remembered.
    """
    etype = _event_type_of(event_row)
    return etype is not None and etype in MACRO_EVENT_TYPES


def _release_period(event_row: EventRow) -> str | None:
    """The REFERENCE period a schedule event covers ("2026-07").

    ``events.release_period`` is a first-class column (migration 021), and it
    is the canonical join key between a schedule row and the observations that
    release publishes. Returns ``None`` when the ingesting provider did not
    supply one (an older row, or a type whose adapter does not model periods):
    the caller then has no join and SAYS so, rather than deriving a period
    from the release date — which would be off by one for every release in
    existence, because July's CPI comes out in August.
    """
    value = getattr(event_row, "release_period", None)
    text = str(value).strip() if value is not None else ""
    return text or None


def _session_of(event_row: EventRow) -> EventSession:
    """The row's session, defaulting to UNKNOWN (same rule as event_price)."""
    try:
        return EventSession(event_row.session)
    except ValueError:
        return EventSession.UNKNOWN


# ---------------------------------------------------------------------------
# ORM -> pure value
# ---------------------------------------------------------------------------


def _to_observation(row: MacroObservationRow) -> MacroObservation:
    """One stored observation as the pure library's value type."""
    return MacroObservation(
        series_id=row.series_id,
        period=row.period,
        value=None if row.value is None else float(row.value),
    )


def _to_schedule_row(row: MacroObservationRow) -> ScheduleRow | None:
    """The release instant stored ALONGSIDE an observation, as a schedule row.

    The pure layer takes the schedule as ``period -> instant``, and the
    backfill already resolved that join once (schedule event -> observation)
    and stored the answer on the row. Re-deriving it at read time would be a
    second join that can disagree with the first, and the stored ``basis`` —
    which the re-derivation could not recover — is what tells a reader whether
    the instant was published or estimated.
    """
    if row.release_at is None:
        return None
    return ScheduleRow(
        period=row.period,
        release_at_utc=_as_utc(row.release_at),
        basis=row.release_basis or RELEASE_BASIS_ESTIMATED,
    )


def _to_yield_row(row: TreasuryYieldRow) -> YieldCurveRow:
    """One stored curve as the pure library's value type.

    A tenor stored as ``None`` is DROPPED rather than passed through: the
    absence already means "Treasury did not quote it", and carrying a null
    into the tenor map would make ``"2 Yr" in tenors`` true for a day with no
    2Y yield.
    """
    tenors = row.tenors if isinstance(row.tenors, dict) else {}
    clean: dict[str, float] = {}
    for key, value in tenors.items():
        if value is None:
            continue
        try:
            clean[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return YieldCurveRow(curve_date=row.curve_date, tenors=clean)


def _to_daily_bars(rows: Iterable[StockBarDaily]) -> list[DailyBar]:
    """Stored bars as the pure reaction library's value type, oldest first.

    The sort is re-asserted (as ``event_price.to_daily_bars`` does) because
    ``event_reaction`` REFUSES unsorted input and a silent ValueError out of a
    read endpoint is a worse failure than a redundant sort.
    """
    return sorted(
        (
            DailyBar(
                date=row.ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume or 0.0),
            )
            for row in rows
        ),
        key=lambda bar: bar.date,
    )


# ---------------------------------------------------------------------------
# Stored reads — none of these can reach the network
# ---------------------------------------------------------------------------


async def load_observations(
    session: AsyncSession, series_ids: Sequence[str]
) -> dict[str, list[MacroObservationRow]]:
    """Every stored observation for ``series_ids``, grouped and period-sorted.

    Sorted with :func:`period_sort_key` rather than by the string, because
    ``"2026-Q1"`` and ``"2026-02"`` compare wrongly as text and a GDP series
    would come back with its quarters interleaved into the wrong months.
    """
    if not series_ids:
        return {}
    rows = (
        (
            await session.execute(
                select(MacroObservationRow).where(
                    MacroObservationRow.series_id.in_(list(series_ids))
                )
            )
        )
        .scalars()
        .all()
    )
    grouped: dict[str, list[MacroObservationRow]] = {sid: [] for sid in series_ids}
    for row in rows:
        grouped.setdefault(row.series_id, []).append(row)
    for items in grouped.values():
        items.sort(key=lambda r: period_sort_key(r.period))
    return grouped


async def load_yield_curve(
    session: AsyncSession, start: date, end: date
) -> list[YieldCurveRow]:
    """Stored Treasury curves in ``[start, end]``, ascending by date."""
    rows = (
        (
            await session.execute(
                select(TreasuryYieldRow)
                .where(
                    TreasuryYieldRow.curve_date >= start,
                    TreasuryYieldRow.curve_date <= end,
                )
                .order_by(TreasuryYieldRow.curve_date)
            )
        )
        .scalars()
        .all()
    )
    return [_to_yield_row(row) for row in rows]


async def load_schedule_rows(
    session: AsyncSession, event_type: EventType, *, as_of: datetime | None = None
) -> list[ScheduleRow]:
    """The agency's release schedule for one event type, from the REGISTRY.

    The stored CONFIRMED calendar rows ARE the schedule — this platform
    already ingests the BLS/BEA schedule pages into ``events``, so a second
    copy of the same dates would be a second thing to keep fresh. Rows with no
    ``release_period`` in their ``raw`` payload are skipped: without the
    reference period there is no join key, and a schedule row that cannot be
    matched to a period is worse than absent because it would attach an
    instant to the wrong month.

    ``as_of`` is deliberately NOT applied here. A release SCHEDULE is
    published a year ahead — the fact that CPI comes out on 2026-09-11 was
    knowable in 2025 — so gating the schedule at ``as_of`` would hide dates
    the platform genuinely knew. The as-of gate belongs on the OBSERVATIONS,
    where the pure layer applies it once (``visible_prints``).
    """
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(EventRow.event_type == event_type.value)
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    schedule: list[ScheduleRow] = []
    seen: set[str] = set()
    for row in rows:
        period = _release_period(row)
        if period is None or period in seen:
            continue
        seen.add(period)
        schedule.append(
            ScheduleRow(
                period=period,
                release_at_utc=_as_utc(row.scheduled_at),
                basis=RELEASE_BASIS_SCHEDULED,
            )
        )
    schedule.sort(key=lambda r: r.release_at_utc)
    return schedule


async def _load_bars_for(
    session: AsyncSession,
    symbols: Sequence[str],
    as_of: datetime,
    *,
    event_id: int | None = None,
) -> tuple[dict[str, list[DailyBar]], dict[str, str]]:
    """``({symbol: gated_bars}, {symbol: reason})`` for the §39 asset set.

    STORED BARS ONLY — this is a read path and holds no provider handle, so a
    symbol nobody has backfilled is a named absence rather than eight vendor
    calls on a page load. The as-of gate is applied to the BARS, once, before
    any reaction is measured (§14, §85): "what did we know at that instant" is
    answered by a shorter bar list, never by trimming an answer that already
    saw the future.
    """
    if not symbols:
        return {}, {}
    rows = (
        (
            await session.execute(
                select(StockBarDaily)
                .where(StockBarDaily.ticker.in_(list(symbols)))
                .order_by(StockBarDaily.ticker, StockBarDaily.ts)
            )
        )
        .scalars()
        .all()
    )
    grouped: dict[str, list[StockBarDaily]] = {}
    for row in rows:
        grouped.setdefault(row.ticker, []).append(row)

    bars: dict[str, list[DailyBar]] = {}
    reasons: dict[str, str] = {}
    for symbol in symbols:
        stored = grouped.get(symbol) or []
        if not stored:
            reasons[symbol] = (
                f"no stored daily bars for {symbol} — run "
                f"POST {_backfill_route(event_id)}"
            )
            continue
        gated = as_of_bar_filter(_to_daily_bars(stored), as_of)
        if not gated:
            reasons[symbol] = f"no {symbol} bars on or before as_of"
            continue
        bars[symbol] = gated
    return bars, reasons


# ---------------------------------------------------------------------------
# Rendering — frozen results to JSON
# ---------------------------------------------------------------------------


def _date_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _backfill_route(event_id: int | None) -> str:
    """The CONCRETE backfill URL for this event, for a reason string.

    A reason that names ``/api/events/{id}/macro/backfill`` makes the reader
    do the substitution, and in a payload full of real ids that reads like a
    templating bug. The route with the real id is copy-pasteable, which is the
    whole value of putting a remedy in a reason at all.
    """
    return f"/api/events/{event_id}/macro/backfill" if event_id is not None else (
        "/api/events/{event_id}/macro/backfill"
    )


def _asset_to_dict(result: AssetReaction, horizons: Sequence[int]) -> dict[str, Any]:
    """One :class:`AssetReaction` as JSON.

    ``returns`` are FRACTIONS (0.012 = +1.2%), inherited unchanged from
    ``reaction.event_reaction`` so a macro reaction and an earnings reaction
    are the same number in the same unit. ``unit`` is stated in the payload
    rather than assumed by the client, because the MACRO PRINTS in the same
    response are already percentages (a 0.2 MoM IS 0.2%) and the two must
    never share a formatter.
    """
    return {
        "symbol": result.symbol,
        "role": result.role,
        "is_proxy": result.is_proxy,
        "basis": result.basis,
        "pre_event_close": result.pre_event_close,
        "pre_event_date": _date_iso(result.pre_event_date),
        "react_date": _date_iso(result.react_date),
        "returns": {f"{k}D": result.returns.get(k) for k in horizons},
        "returns_unit": "fraction",
        "reasons": dict(result.reasons),
    }


def _yield_to_dict(result: YieldChange) -> dict[str, Any]:
    """One :class:`YieldChange` as JSON — the change is BASIS POINTS."""
    return {
        "tenor": result.tenor,
        "before": result.before,
        "before_date": _date_iso(result.before_date),
        "after": result.after,
        "after_date": _date_iso(result.after_date),
        "change_bp": result.change_bp,
        "change_unit": "basis_points",
        "level_unit": "percent",
        "reason": result.reason,
    }


def reaction_to_dict(result: MultiAssetReaction) -> dict[str, Any]:
    """The whole §39 cross-asset table as JSON.

    ``assets`` and ``yields`` are separate objects and never merged: a 7bp
    move in the 2Y and a 0.4% move in SPY are not the same kind of number, and
    one table with a "change" column would invite exactly that comparison.
    ``unavailable`` names every symbol that produced nothing AND why — an
    absent symbol with no reason reads as "it did not move".
    """
    horizons = result.horizons or DEFAULT_MACRO_HORIZONS
    return {
        "tier": TIER_QUANT,
        "available": bool(result.assets or result.yields),
        "event_at_utc": result.event_at_utc.isoformat()
        if result.event_at_utc is not None
        else None,
        "event_date_et": _date_iso(result.event_date_et),
        "session": result.session.value if result.session is not None else None,
        "horizons": [f"{k}D" for k in horizons],
        "assets": {
            symbol: _asset_to_dict(reaction, horizons)
            for symbol, reaction in sorted(result.assets.items())
        },
        "yields": {
            tenor: _yield_to_dict(change)
            for tenor, change in sorted(result.yields.items())
        },
        "unavailable": [
            {"symbol": symbol, "reason": reason}
            for symbol, reason in sorted(result.unavailable.items())
        ],
        "asset_roles": dict(sorted(ASSET_ROLES.items())),
        "proxy_note": (
            "Every symbol here is an ETF standing in for an exposure, never "
            "the exposure itself — TLT is a fund holding long Treasuries, not "
            "'long rates'. Rows flagged is_proxy say so (§39)."
        ),
        "model_version": result.model_version,
    }


def packet_to_dict(packet: MacroPacket) -> dict[str, Any]:
    """One :class:`MacroPacket` as JSON (§38).

    The published levels are DATA and everything derived from them is QUANT,
    so the tier is stamped per block rather than once for the packet: the
    ``actual`` values inside ``previous_release`` are the agency's own index
    levels and transforms of them, and calling the whole packet DATA would
    label this platform's MoM arithmetic as a government publication.
    """
    return {
        "tier": TIER_QUANT,
        "event_type": packet.event_type.value,
        "as_of": packet.as_of.isoformat(),
        "previous_release": {"tier": TIER_DATA, **json_safe(packet.previous_release)},
        "current_release": {"tier": TIER_DATA, **json_safe(packet.current_release)},
        "recent_trend": json_safe(packet.recent_trend),
        "coverage": json_safe(packet.coverage),
        "consensus_status": packet.consensus_status,
        "surprise_status": SURPRISE_UNAVAILABLE_REASON,
        "model_version": packet.model_version,
    }


# ---------------------------------------------------------------------------
# The read seam — DB ONLY
# ---------------------------------------------------------------------------


async def build_macro_payload(
    session: AsyncSession, event_row: EventRow, *, as_of: datetime
) -> dict[str, Any]:
    """The whole macro block for one event, as of one instant (§38-§41).

    Order of operations is the contract: load the catalogue's observations ->
    turn them into prints with their STORED release instants -> hand
    everything to the pure packet builder, which applies the as-of gate once
    -> measure the previous release's cross-asset reaction over as-of-gated
    bars -> collect the §40 related-evidence window. Nothing is measured
    before the gate.

    THIS FUNCTION NEVER FETCHES. It takes no provider name and imports no
    provider; a series nobody backfilled is ``coverage.actuals.available =
    false`` naming the POST that would fix it. That is not a limitation, it is
    the property that keeps a 25-request-per-day API usable: the read is free
    and repeatable, the write is a deliberate user action.

    A NON-MACRO event answers ``{"available": false, "reason": "not a macro
    event"}`` rather than 404 — the row exists, it simply has no macro packet,
    and 404 would say the event does not exist.
    """
    moment = _as_utc(as_of)
    etype = _event_type_of(event_row)
    base: dict[str, Any] = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_type": event_row.event_type,
        "as_of": moment.isoformat(),
        "model_version": MACRO_MODEL_VERSION,
    }
    if etype is None or etype not in MACRO_EVENT_TYPES:
        return {
            **base,
            "available": False,
            "reason": (
                f"not a macro event ({event_row.event_type}) — the §38 packet "
                "describes a published statistical release"
            ),
            "disclaimer": MACRO_DISCLAIMER,
        }

    specs = series_for(etype)
    series_ids = [spec.series_id for spec in specs]
    stored = await load_observations(session, series_ids)

    # --- prints, each carrying the instant the backfill resolved ----------
    prints_by_series: dict[str, list] = {}
    stored_schedule: dict[str, dict[str, ScheduleRow]] = {}
    for spec in specs:
        rows = stored.get(spec.series_id) or []
        schedule_for_series: dict[str, datetime] = {}
        basis_for_period: dict[str, str] = {}
        for row in rows:
            sched = _to_schedule_row(row)
            if sched is not None:
                schedule_for_series[sched.period] = sched.release_at_utc
                basis_for_period[sched.period] = sched.basis
        stored_schedule[spec.series_id] = {
            period: ScheduleRow(
                period=period,
                release_at_utc=instant,
                basis=basis_for_period.get(period, RELEASE_BASIS_ESTIMATED),
            )
            for period, instant in schedule_for_series.items()
        }
        prints_by_series[spec.series_id] = derive_prints(
            [_to_observation(row) for row in rows],
            spec,
            schedule=schedule_for_series or None,
        )

    # The packet's own schedule is the REGISTRY's, which covers periods no
    # observation has arrived for yet — that is what lets ``current_release``
    # name the upcoming print even when the series is empty.
    schedule_rows = await load_schedule_rows(session, etype)

    packet = None
    packet_error: str | None = None
    try:
        packet = _build_packet(
            etype,
            moment,
            schedule_rows,
            prints_by_series,
            event_row,
        )
    except Exception as exc:  # noqa: BLE001 — one bad row must not sink the read
        logger.warning("macro packet for event %s failed: %s", event_row.id, exc)
        packet_error = f"{type(exc).__name__}: {exc}"

    packet_json = packet_to_dict(packet) if packet is not None else None

    # --- §39 the previous release's cross-asset reaction ------------------
    previous_release = (packet.previous_release if packet is not None else {}) or {}
    previous_at = previous_release.get("release_at")
    previous_at = _as_utc(previous_at) if isinstance(previous_at, datetime) else None

    # FALL BACK TO THE EVENT REGISTRY when no OBSERVATION carries the instant.
    # The packet's `release_at` comes from the macro observation feed — the
    # actual GDP/CPI *values* — which is a separate feed from the calendar and
    # is often unpopulated. But measuring how SPY, the Dow and volatility
    # reacted to the last print needs only the DATE of that print, and the
    # registry knows it: `previous_comparable` names the prior release of the
    # same series. Refusing to measure a reaction for want of a data feed the
    # measurement does not use left every macro event with an empty
    # cross-asset panel while the bars sat in the database.
    #
    # This reads the SAME resolver the evidence bundle and price seam use, so
    # the release named here is the release named there.
    previous_source = "macro_observation" if previous_at is not None else None
    if previous_at is None:
        prior_rows = await _past_comparable_rows(session, event_row, moment)
        prior, _prior_reason = previous_comparable(
            row_to_event(event_row), [row_to_event(r) for r in prior_rows]
        )
        if prior is not None:
            previous_at = _as_utc(prior.scheduled_at)
            previous_source = "event_registry"

    reaction_json: dict[str, Any]
    if previous_at is None:
        reaction_json = {
            "tier": TIER_QUANT,
            "available": False,
            "reason": (
                "no previous release instant is known at as_of, so there is "
                "no window to measure a reaction over"
            ),
            "assets": {},
            "yields": {},
            "unavailable": [],
        }
    else:
        bars, bar_reasons = await _load_bars_for(
            session, MACRO_REFERENCE_SYMBOLS, moment, event_id=event_row.id
        )
        curve_start = previous_at.astimezone(EASTERN).date() - timedelta(
            days=BACKFILL_BAR_LEAD_DAYS
        )
        curve_end = moment.astimezone(EASTERN).date()
        curves = await load_yield_curve(session, curve_start, curve_end)
        result = multi_asset_reaction(
            bars,
            curves,
            event_at_utc=previous_at,
            session=_session_of(event_row),
            horizons=DEFAULT_MACRO_HORIZONS,
        )
        reaction_json = reaction_to_dict(result)
        # The library reports "no stored daily bars" for a missing symbol; the
        # loader knows the more useful sentence (which button fills it), so
        # its reason wins where it has one.
        merged = {
            entry["symbol"]: bar_reasons.get(entry["symbol"], entry["reason"])
            for entry in reaction_json["unavailable"]
        }
        reaction_json["unavailable"] = [
            {"symbol": symbol, "reason": reason}
            for symbol, reason in sorted(merged.items())
        ]
        if not curves:
            reaction_json["yields_reason"] = (
                "no stored Treasury curves in the window — run "
                f"POST {_backfill_route(event_row.id)}"
            )
        # WHERE THE ANCHOR INSTANT CAME FROM. "The release was at 08:30 on the
        # 25th" (an observation's own stamp) and "the calendar says the prior
        # release of this series was the 25th" are different claims, and the
        # reader is entitled to know which one the window was measured from.
        reaction_json["previous_release_at"] = previous_at.isoformat()
        reaction_json["previous_release_source"] = previous_source

    # --- §40 what else happened between the last release and now ----------
    related = await _related_events(session, event_row, previous_at, moment)

    n_observations = sum(len(rows) for rows in stored.values())
    coverage: dict[str, Any] = {
        "actuals": {
            "available": n_observations > 0,
            "reason": None
            if n_observations > 0
            else (
                "no macro observations stored for this event type — run "
                f"POST {_backfill_route(event_row.id)}"
                if series_ids
                else "no data source configured for this event type"
            ),
            "n_observations": n_observations,
            "n_series": len(series_ids),
        },
        "schedule": {
            "available": bool(schedule_rows),
            "reason": None
            if schedule_rows
            else (
                "no stored release-schedule rows carry a reference period, so "
                f"instants fall back to period end + {ESTIMATED_LAG_DAYS}d"
            ),
            "n_rows": len(schedule_rows),
        },
        "packet": {
            "available": packet_json is not None,
            "reason": packet_error,
        },
        "reaction": {
            "available": bool(reaction_json.get("available")),
            "reason": reaction_json.get("reason"),
        },
        "related_evidence": {
            "available": bool(related.get("available")),
            "reason": related.get("reason"),
        },
        "consensus": {
            "available": False,
            "reason": CONSENSUS_UNAVAILABLE_REASON,
        },
    }

    return {
        **base,
        "available": True,
        "provenance": {
            "observations": TIER_DATA,
            "yield_curve": TIER_DATA,
            "transforms": TIER_QUANT,
            "reaction": TIER_QUANT,
        },
        "packet": packet_json,
        "previous_release_reaction": reaction_json,
        "related_evidence": json_safe(related),
        "reference_symbols": list(MACRO_REFERENCE_SYMBOLS),
        "tracked_tenors": list(TRACKED_TENORS),
        "coverage": coverage,
        "disclaimer": MACRO_DISCLAIMER,
    }


def _build_packet(
    etype: EventType,
    moment: datetime,
    schedule_rows: Sequence[ScheduleRow],
    prints_by_series: Mapping[str, Sequence],
    event_row: EventRow,
) -> MacroPacket:
    """The pure packet, with THIS event as the current release.

    ``current_period``/``current_release_at`` are taken from the event ROW
    rather than left to the schedule scan, because the row IS the release
    being analysed: a packet for "CPI:2026-09-11" must name September's print
    as current even when ``as_of`` is after it and the schedule's "next
    upcoming" row is October's.
    """
    from libs.trading_core.events.macro import build_macro_packet

    return build_macro_packet(
        etype,
        as_of=moment,
        schedule=list(schedule_rows),
        prints_by_series=dict(prints_by_series),
        current_period=_release_period(event_row),
        current_release_at=_as_utc(event_row.scheduled_at),
    )


async def _related_events(
    session: AsyncSession,
    event_row: EventRow,
    previous_at: datetime | None,
    moment: datetime,
) -> dict[str, Any]:
    """The §40 window: other macro/Fed events between the last release and now.

    The candidate pool is every MARKET-WIDE registry row in the window — the
    pure layer decides which of them belong and the LLM layer decides which
    MATTER. No keyword list is applied anywhere in this path, which is §40's
    explicit requirement: the themes that matter for a CPI print shift with
    context, and a fixed list of them would be a rigid answer to a question
    that changes.
    """
    if previous_at is None:
        return related_evidence_window(None, moment, (), exclude_event_ids=(event_row.id,))
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.scheduled_at >= previous_at.replace(tzinfo=None)
                    if previous_at.tzinfo is not None
                    else EventRow.scheduled_at >= previous_at,
                    EventRow.scheduled_at <= moment.replace(tzinfo=None)
                    if moment.tzinfo is not None
                    else EventRow.scheduled_at <= moment,
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    payload = [
        {
            "event_id": row.id,
            "event_type": row.event_type,
            "title": row.title,
            "scheduled_at": _as_utc(row.scheduled_at),
            "importance": row.importance,
        }
        for row in rows
    ]
    return related_evidence_window(
        previous_at, moment, payload, exclude_event_ids=(event_row.id,)
    )


# ---------------------------------------------------------------------------
# §46 — the macro_context section every bundle carries
# ---------------------------------------------------------------------------


async def macro_context_section(
    session: AsyncSession, event_row: EventRow, as_of: datetime
) -> dict[str, Any]:
    """The evidence bundle's ``macro_context`` block, for ANY event (§46).

    Two shapes, one key, because the question "what is the macro backdrop"
    has two different answers depending on what the event is:

    - **an EARNINGS (or any non-macro) event** gets the §46 forward look: the
      macro releases landing between ``as_of`` and the horizon, with days-to
      and importance. A company reporting the day before CPI is a different
      trade from the same company reporting the day after, and the model
      should not have to infer that from a raw calendar.
    - **a MACRO event** gets its own packet plus the previous release's
      cross-asset reaction — the backdrop for a CPI print IS the last CPI
      print and what the market did with it.

    STORE-ONLY in both branches, and every failure is caught into an honest
    ``available: false`` with a reason: this block rides inside a bundle whose
    whole design is that one dead section never costs the other six.
    """
    moment = _as_utc(as_of)
    try:
        if is_macro_event(event_row):
            payload = await build_macro_payload(session, event_row, as_of=moment)
            packet = payload.get("packet") or {}
            return {
                "tier": TIER_QUANT,
                "available": bool(payload.get("available")),
                "reason": payload.get("reason"),
                "kind": "macro_event_packet",
                "event_type": event_row.event_type,
                "packet": packet,
                "previous_release_reaction": payload.get(
                    "previous_release_reaction"
                ),
                "related_evidence": payload.get("related_evidence"),
                "coverage": payload.get("coverage"),
                "consensus_status": CONSENSUS_UNAVAILABLE_REASON,
                "disclaimer": MACRO_DISCLAIMER,
                "model_version": MACRO_MODEL_VERSION,
            }
        return await upcoming_macro_context(session, as_of=moment)
    except Exception as exc:  # noqa: BLE001 — never sink the bundle
        logger.warning(
            "macro_context for event %s failed: %s", event_row.id, exc
        )
        return {
            "tier": TIER_QUANT,
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "kind": "unavailable",
            "consensus_status": CONSENSUS_UNAVAILABLE_REASON,
            "disclaimer": MACRO_DISCLAIMER,
            "model_version": MACRO_MODEL_VERSION,
        }


async def upcoming_macro_context(
    session: AsyncSession,
    *,
    as_of: datetime,
    horizon_days: int = DEFAULT_MACRO_CONTEXT_HORIZON_DAYS,
) -> dict[str, Any]:
    """"What macro lands in the next ``horizon_days``" — the earnings branch.

    Only CONFIRMED/REVISED rows enter the pool. An ESTIMATED macro date is a
    derivation, and "CPI is two days after this print" is a materially
    different statement when the CPI date is a guess — §7 forbids presenting
    a derived date as a fact, and this is a block the model writes prose from.
    """
    moment = _as_utc(as_of)
    horizon_end = moment + timedelta(days=int(horizon_days))
    wanted = sorted(t.value for t in MACRO_CONTEXT_EVENT_TYPES)
    rows = (
        (
            await session.execute(
                select(EventRow)
                .where(
                    EventRow.event_type.in_(wanted),
                    EventRow.status.in_(
                        [EventStatus.CONFIRMED.value, EventStatus.REVISED.value]
                    ),
                )
                .order_by(EventRow.scheduled_at)
            )
        )
        .scalars()
        .all()
    )
    payload = [
        {
            "event_id": row.id,
            "event_type": row.event_type,
            "title": row.title,
            "scheduled_at": _as_utc(row.scheduled_at),
            "importance": row.importance,
            "status": row.status,
        }
        for row in rows
    ]
    context = macro_context_for(
        payload, as_of=moment, horizon_days=horizon_days
    )
    return {
        "tier": TIER_DATA,
        "kind": "upcoming_macro_releases",
        **json_safe(context),
        "consensus_status": CONSENSUS_UNAVAILABLE_REASON,
        "disclaimer": MACRO_DISCLAIMER,
        "status_note": (
            "Only CONFIRMED/REVISED release dates are listed — an ESTIMATED "
            "macro date is this platform's derivation and must not be quoted "
            "as a scheduled release (§7)."
        ),
    }


# ---------------------------------------------------------------------------
# The write seam — the ONLY thing here that touches the network
# ---------------------------------------------------------------------------


async def _upsert_observations(
    session: AsyncSession,
    rows: Sequence[tuple[str, str, float | None, datetime | None, str | None]],
    *,
    provider: str,
) -> int:
    """Upsert ``(series_id, period, value, release_at, basis)`` rows.

    ONE FACT PER (series, period): a re-fetch OVERWRITES, because an agency
    revising July's CPI is restating the same fact, not publishing a second
    one. Uses the Postgres ``ON CONFLICT DO UPDATE`` when available and falls
    back to read-modify-write on SQLite (the test harness), so the same
    idempotence holds in both — a backfill pressed twice must not double a
    series.
    """
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        payload = [
            {
                "series_id": series_id,
                "period": period,
                "value": value,
                "release_at": release_at,
                "release_basis": basis,
                "provider": provider,
                "fetched_at": now,
            }
            for series_id, period, value, release_at, basis in rows
        ]
        stmt = pg_insert(MacroObservationRow).values(payload)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[
                    MacroObservationRow.series_id,
                    MacroObservationRow.period,
                ],
                set_={
                    "value": stmt.excluded.value,
                    "release_at": stmt.excluded.release_at,
                    "release_basis": stmt.excluded.release_basis,
                    "provider": stmt.excluded.provider,
                    "fetched_at": stmt.excluded.fetched_at,
                },
            )
        )
        return len(payload)

    written = 0
    for series_id, period, value, release_at, basis in rows:
        existing = await session.get(MacroObservationRow, (series_id, period))
        if existing is None:
            session.add(
                MacroObservationRow(
                    series_id=series_id,
                    period=period,
                    value=value,
                    release_at=release_at,
                    release_basis=basis,
                    provider=provider,
                    fetched_at=now,
                )
            )
        else:
            existing.value = value
            existing.release_at = release_at
            existing.release_basis = basis
            existing.provider = provider
            existing.fetched_at = now
        written += 1
    return written


async def _upsert_yield_curves(
    session: AsyncSession, curves: Sequence[Any], *, provider: str
) -> int:
    """Upsert Treasury curves keyed on ``curve_date``.

    A tenor the CSV left blank is ABSENT from the stored object, never 0.0:
    zero percent of yield is a claim nobody made, and a 0.0 stored for a
    missing 2Y would show as a ~420bp move across the release.
    """
    if not curves:
        return 0
    now = datetime.now(timezone.utc)
    written = 0
    for curve in curves:
        tenors = {
            str(key): float(value)
            for key, value in dict(getattr(curve, "tenors", {}) or {}).items()
            if value is not None
        }
        existing = await session.get(TreasuryYieldRow, curve.date)
        if existing is None:
            session.add(
                TreasuryYieldRow(
                    curve_date=curve.date,
                    tenors=tenors,
                    provider=provider,
                    fetched_at=now,
                )
            )
        else:
            existing.tenors = tenors
            existing.provider = provider
            existing.fetched_at = now
        written += 1
    return written


async def ensure_daily_bars(
    session: AsyncSession,
    symbol: str,
    *,
    provider_name: str,
    days: int,
) -> tuple[int, str | None]:
    """``(bars_stored, reason)`` — fetch and store daily bars for one symbol.

    A SMALL, EXPLICIT backfill rather than a call into
    ``routers/analysis.ensure_daily_bars``. That function is the lazy top-up
    path for WATCHLIST symbols: it raises the shared 503 for an unconfigured
    provider, raises 502 when a vendor serves nothing, and refreshes on its
    own throttle — all correct for a request that is about one ticker, and all
    wrong for a fan-out over eight macro proxies where a single symbol the
    vendor does not carry (UUP on a thin plan) must not fail the other seven.
    Here every symbol is attempted independently and each outcome is a
    ``reason`` string, exactly as the calendar ingest treats providers.

    Bars already stored are NEVER rewritten — only dates the platform does not
    have are inserted, so a re-press adds the tail and cannot duplicate a
    session or restate a close.
    """
    try:
        provider = get_provider(provider_name)
    except (ProviderNotConfigured, MarketDataError) as exc:
        return 0, str(exc)
    try:
        fetched = provider.get_daily_bars(symbol, days)
    except CapabilityNotAvailable as exc:
        return 0, f"{provider_name} cannot serve {symbol}: {exc}"
    except MarketDataError as exc:
        return 0, f"{provider_name} error for {symbol}: {exc}"
    except Exception as exc:  # noqa: BLE001 — one symbol must not sink the run
        return 0, f"{type(exc).__name__}: {exc}"
    if not fetched:
        return 0, f"{provider_name} returned no bars for {symbol}"

    existing = set(
        (
            await session.execute(
                select(StockBarDaily.ts).where(StockBarDaily.ticker == symbol)
            )
        )
        .scalars()
        .all()
    )
    today_et = datetime.now(timezone.utc).astimezone(EASTERN).date()
    stored = 0
    for bar in fetched:
        bar_date = getattr(bar, "date", None) or getattr(bar, "ts", None)
        if bar_date is None or bar_date in existing:
            continue
        # COMPLETE DAYS ONLY — today's daily bar is still forming, and stored
        # bars are never rewritten, so a provisional close would be permanent.
        if bar_date >= today_et:
            continue
        session.add(
            StockBarDaily(
                ticker=symbol,
                ts=bar_date,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(getattr(bar, "volume", 0.0) or 0.0),
            )
        )
        existing.add(bar_date)
        stored += 1
    return stored, None


def _schedule_lookup(schedule: Sequence[ScheduleRow]) -> dict[str, ScheduleRow]:
    return {row.period: row for row in schedule}


def _resolve_release(
    period: str, lookup: Mapping[str, ScheduleRow]
) -> tuple[datetime | None, str | None]:
    """``(instant, basis)`` for one reference period at STORE time.

    The schedule (the platform's own CONFIRMED calendar rows for this event
    type) wins; otherwise period-end + 45 days, stamped ESTIMATED. Resolving
    this ONCE here — at write time — rather than on every read is what makes
    ``macro_observations.release_at`` a stored fact with a stated provenance
    instead of a derivation two code paths could disagree about.
    """
    hit = lookup.get(period)
    if hit is not None:
        return hit.release_at_utc, RELEASE_BASIS_SCHEDULED
    end = period_end_date(period)
    if end is None:
        return None, None
    return (
        datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
        + timedelta(days=ESTIMATED_LAG_DAYS),
        RELEASE_BASIS_ESTIMATED,
    )


async def backfill_macro(
    session: AsyncSession,
    event_row: EventRow,
    *,
    settings,
    as_of: datetime | None = None,
    macro_provider: Any | None = None,
    treasury: Any | None = None,
) -> dict[str, Any]:
    """USER action: fetch this event type's series, the curve and the bars.

    THREE FETCHES, EACH INDEPENDENT, EVERY OUTCOME REPORTED. The statistical
    series come from the BLS v1 client, the yield curve from Treasury's CSV,
    the reference bars from the configured equity provider — and a failure in
    any one of them is a named ``reason`` in the response rather than an
    exception, because a CPI packet with no yield curve is still most of the
    §38 answer and refusing it entirely would be the worse trade.

    PACED FOR A TWENTY-FIVE-REQUEST BUDGET. BLS's unregistered v1 API allows
    roughly 25 requests per DAY and serves only the latest three years, so
    this asks for one request per series in the event type's catalogue
    (four for CPI, three for the employment report) and no more. There is no
    retry loop: a 429 is stored as a reason and the operator presses the
    button again tomorrow, because a retry would spend the budget it is
    trying to recover.

    IDEMPOTENT AT THE DATABASE LEVEL. Observations upsert on (series_id,
    period) and curves on curve_date, so two concurrent presses can collide
    but cannot double-write; bars are inserted only for dates not already
    stored. Pressing twice costs provider calls and changes nothing else.

    A NON-MACRO event returns ``{"available": false}`` without spending a
    single request — there is no catalogue to fetch.
    """
    moment = _as_utc(as_of) if as_of is not None else datetime.now(timezone.utc)
    etype = _event_type_of(event_row)
    base: dict[str, Any] = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_type": event_row.event_type,
        "as_of": moment.isoformat(),
    }
    if etype is None or etype not in MACRO_EVENT_TYPES:
        return {
            **base,
            "available": False,
            "reason": f"not a macro event ({event_row.event_type})",
            "series": [],
            "counts": {"observations": 0, "yield_curves": 0, "bars": 0},
        }

    specs = series_for(etype)
    schedule = _schedule_lookup(await load_schedule_rows(session, etype))

    # --- 1. the statistical series ----------------------------------------
    series_results: list[dict[str, Any]] = []
    observations_written = 0
    if not specs:
        series_results.append(
            {
                "series_id": None,
                "status": "UNAVAILABLE",
                "reason": (
                    f"no data source configured for {etype.value} — the "
                    "release DATES are still tracked, the actuals are not"
                ),
                "stored": 0,
            }
        )
    else:
        provider = macro_provider
        if provider is None:
            from libs.event_calendar import macro_data_provider

            provider = macro_data_provider(settings)
        end_year = moment.astimezone(EASTERN).year
        start_year = end_year - (BACKFILL_SERIES_YEARS - 1)
        for spec in specs:
            try:
                observations = provider.get_series(
                    spec.series_id, start_year=start_year, end_year=end_year
                )
            except CapabilityNotAvailable as exc:
                series_results.append(
                    {
                        "series_id": spec.series_id,
                        "status": "UNAVAILABLE",
                        "reason": str(exc),
                        "stored": 0,
                    }
                )
                continue
            except MarketDataError as exc:
                series_results.append(
                    {
                        "series_id": spec.series_id,
                        "status": "ERROR",
                        "reason": str(exc),
                        "stored": 0,
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                series_results.append(
                    {
                        "series_id": spec.series_id,
                        "status": "ERROR",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "stored": 0,
                    }
                )
                continue

            rows: list[tuple[str, str, float | None, datetime | None, str | None]] = []
            for obs in observations:
                release_at, basis = _resolve_release(obs.period, schedule)
                rows.append(
                    (
                        spec.series_id,
                        obs.period,
                        None if obs.value is None else float(obs.value),
                        release_at,
                        basis,
                    )
                )
            written = await _upsert_observations(
                session, rows, provider=getattr(provider, "name", "bls")
            )
            observations_written += written
            series_results.append(
                {
                    "series_id": spec.series_id,
                    "status": "OK" if written else "NO_DATA",
                    "reason": None if written else "provider returned no observations",
                    "stored": written,
                }
            )

    # --- 2. the yield curve -----------------------------------------------
    curves_written = 0
    curve_reason: str | None = None
    years = sorted({moment.astimezone(EASTERN).year, moment.astimezone(EASTERN).year - 1})
    client = treasury
    if client is None:
        try:
            from libs.event_calendar import _government_user_agent
            from libs.event_calendar.treasury import TreasuryYields

            client = TreasuryYields(user_agent=_government_user_agent(settings))
        except Exception as exc:  # noqa: BLE001
            client = None
            curve_reason = f"{type(exc).__name__}: {exc}"
    if client is not None:
        for year in years:
            try:
                curves = client.get_yield_curve(year)
            except (CapabilityNotAvailable, MarketDataError) as exc:
                curve_reason = str(exc)
                continue
            except Exception as exc:  # noqa: BLE001
                curve_reason = f"{type(exc).__name__}: {exc}"
                continue
            curves_written += await _upsert_yield_curves(
                session, curves, provider=_TREASURY_PROVIDER
            )

    # --- 3. the §39 reference bars ----------------------------------------
    provider_name = getattr(settings, "market_data_provider", "") or ""
    bar_days = _bar_window_days(event_row, moment)
    bars_written = 0
    bar_results: list[dict[str, Any]] = []
    for symbol in MACRO_REFERENCE_SYMBOLS:
        stored, reason = await ensure_daily_bars(
            session, symbol, provider_name=provider_name, days=bar_days
        )
        bars_written += stored
        bar_results.append(
            {
                "symbol": symbol,
                "stored": stored,
                "status": "OK" if reason is None else "ERROR",
                "reason": reason,
            }
        )

    counts = {
        "observations": observations_written,
        "yield_curves": curves_written,
        "bars": bars_written,
    }
    await audit.record(
        session,
        actor_type=ActorType.USER,
        action=AuditAction.DATA_BACKFILL,
        entity_type="event",
        entity_id=str(event_row.id),
        details={
            "kind": "event_macro",
            "event_key": event_row.event_key,
            "event_type": event_row.event_type,
            "as_of": moment.isoformat(),
            "counts": counts,
            "series": [r["series_id"] for r in series_results],
            "bar_days": bar_days,
        },
    )
    await session.commit()

    return {
        **base,
        "available": True,
        "counts": counts,
        "series": series_results,
        "yield_curve": {
            "years": years,
            "stored": curves_written,
            "reason": curve_reason,
        },
        "bars": bar_results,
        "bar_window_days": bar_days,
        "disclaimer": MACRO_DISCLAIMER,
    }


def _bar_window_days(event_row: EventRow, moment: datetime) -> int:
    """How many calendar days of bars the §39 reaction needs.

    Measured from ten days BEFORE this event back-to-back with the gap to
    ``as_of``, floored at a quarter: the reaction needs a pre-event close and
    five sessions after, and a window that only just spans the event has no
    room for a holiday. Bounded above at two years so a badly-dated row cannot
    turn one button press into a decade of bar requests for eight symbols.
    """
    event_day = event_date_et(event_row)
    as_of_day = moment.astimezone(EASTERN).date()
    span = abs((as_of_day - event_day).days) + BACKFILL_BAR_LEAD_DAYS
    return max(90, min(span + 30, 730))
