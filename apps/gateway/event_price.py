"""Price context & previous-event reaction — the gateway seam (Phase E1, U2).

THE SPLIT THIS MODULE EXISTS TO KEEP. Every number in the payload is computed
by ``libs/trading_core/events/reaction.py``, which is pure stdlib and may not
import ``apps/`` or ``libs.market_data`` (audit §7.4). This module is the only
place the two halves meet: it reads stored bars through
``routers/analysis.py::ensure_daily_bars`` (the one lazy-backfill path in the
codebase, so provenance and the DATA_BACKFILL audit trail stay single-sourced),
converts ORM rows to the pure :class:`~libs.trading_core.events.reaction.DailyBar`
value, hands them to the library, and renders the frozen results as JSON. It
computes nothing itself — no arithmetic on a price lives here.

THE AS-OF GATE IS APPLIED ONCE, TO THE BARS (§14, §85, §96). Every bar list —
the ticker's and SPY's — goes through ``as_of_bar_filter`` immediately after
loading, before any reaction is measured, so "what did we know at 15:59 ET on
the day of the print" is answered by a shorter bar list rather than by trimming
an answer that already saw the future. The past-event pool is filtered the same
way: an earnings row scheduled after ``as_of`` is not a "previous event" at
``as_of``, however firmly the registry knows about it today.

HONEST ABSENCE, NEVER A ZERO (§44 rule 18, §85). Three different failures each
get their own shape rather than collapsing into an empty payload:

- no market-data provider / provider error -> the ``bars`` block is
  ``{"available": false, "reason": ...}`` and the endpoint still answers 200
  with the registry facts it does have. A 503 here would hide the event.
- an event older than the stored bar history -> that row carries
  ``bars_available: false`` and the library's own reason string ("bars
  unavailable before 2024-03-20"), which is precisely the honest answer the
  48 seeded CONFIRMED earnings rows need for their oldest one or two prints.
- a metric whose window is longer than the history (sma200 over 120 bars) ->
  ``null`` plus a reason, collected into the payload's ``unavailable`` list so
  the UI can render "Unavailable — needs 200 bars, have 120" without knowing
  which library function produced it.

PROVENANCE IS LABELLED AT BLOCK LEVEL (§91): bars are DATA (a provider's real
closes), everything derived from them is QUANT (this platform's arithmetic).
Nothing in this payload is LLM-generated, and the ``not_backtestable`` list
carries the §85/§7.3 labels for the fields a price-only view cannot supply
point-in-time — an empty list would read as a claim that everything here is
reconstructable at any historical instant, which is false for the option-side
context the same UI tab will later carry.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.market_data import MarketDataError, ProviderNotConfigured
from libs.trading_core.events import EASTERN, previous_comparable
from libs.trading_core.events.reaction import (
    AbnormalResult,
    DailyBar,
    HistoryStats,
    PriceContext,
    ReactionResult,
    abnormal_vs,
    as_of_bar_filter,
    event_reaction,
    history_stats,
    pre_event_price_context,
)
from libs.trading_core.models.enums import EventSession, EventStatus, EventType

from .db import EventRow, StockBarDaily
from .event_calendar import row_to_event
from .routers.analysis import ensure_daily_bars

#: The abnormal-return benchmark (ADR-005 reference symbol, exempt from the
#: watchlist gating that applies to ordinary tickers — the same exemption
#: ``market_regime_from_spy`` relies on).
BENCHMARK_SYMBOL = "SPY"

#: Horizons the payload reports for every past event (§17).
HORIZONS: tuple[int, ...] = (1, 3, 5, 10)

#: Horizons the §19/§64 history strip summarises. 1D is the headline reaction;
#: 5D is the "did it hold" follow-through.
HISTORY_HORIZONS: tuple[int, ...] = (1, 5)

#: Statuses that make a past earnings row a usable comparison (§15): the date
#: must be observed, not derived. An ESTIMATED past date would anchor a run-up
#: measurement to a day the company may never have reported on.
COMPARABLE_STATUSES: frozenset[str] = frozenset(
    {EventStatus.CONFIRMED.value, EventStatus.REVISED.value}
)

#: §85 / audit §7.3 — fields this price view deliberately does NOT claim to
#: reconstruct point-in-time. Carried in the payload so the UI labels them
#: rather than the user assuming the whole tab is backtestable.
NOT_BACKTESTABLE: tuple[str, ...] = (
    "historical_implied_move",
    "historical_atm_iv",
    "historical_option_greeks",
    "historical_open_interest",
)


# ---------------------------------------------------------------------------
# ORM -> pure value
# ---------------------------------------------------------------------------


def to_daily_bars(rows: list[StockBarDaily]) -> list[DailyBar]:
    """Convert stored bars to the pure library's value type, oldest first.

    ``ensure_daily_bars`` already orders by ``ts``; the sort is re-asserted
    because :func:`event_reaction` REFUSES unsorted input (it cannot tell a
    mis-ordered series from a mis-dated one) and a silent ValueError out of a
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


async def _load_bars(
    session: AsyncSession,
    ticker: str,
    provider_name: str,
    as_of: datetime,
) -> tuple[list[DailyBar], str | None]:
    """``(bars_knowable_at_as_of, reason_if_unavailable)`` for one symbol.

    Every failure mode of the market-data layer is caught and turned into a
    reason string: an unconfigured provider (``ensure_daily_bars`` re-raises it
    as the shared 503), a provider that errored or returned nothing (502), and
    the raw provider exceptions. The endpoint stays 200 in all of them — the
    event's registry facts are real rows and are not hidden because a quote
    vendor is down.
    """
    try:
        rows = await ensure_daily_bars(session, ticker, provider_name)
    except HTTPException as exc:
        # ensure_daily_bars raises the canonical 503 for a missing provider and
        # a 502 when the provider returned no bars. Both are "no bars, here is
        # why", not a reason to fail this read.
        detail = exc.detail
        if isinstance(detail, dict):
            reason = str(detail.get("message") or detail.get("code") or detail)
        else:
            reason = str(detail)
        return [], reason
    except (ProviderNotConfigured, MarketDataError) as exc:
        return [], str(exc)
    if not rows:
        return [], f"no stored bars for {ticker}"
    return as_of_bar_filter(to_daily_bars(rows), as_of), None


# ---------------------------------------------------------------------------
# Registry reads
# ---------------------------------------------------------------------------


async def _past_comparable_rows(
    session: AsyncSession, event_row: EventRow, as_of: datetime
) -> list[EventRow]:
    """Past comparable rows of the event's OWN type, oldest first.

    Two filters, both point-in-time: strictly before the event's own scheduled
    instant (a later print is not a precedent) and at or before ``as_of`` (a
    print the platform could not have seen at ``as_of`` is not history yet,
    even though the row exists now). Status is narrowed to CONFIRMED/REVISED
    per §15 — an ESTIMATED past date is a derivation, and measuring a market
    reaction around a day nobody reported on would be a fabricated number
    wearing a measurement's clothes.

    The type filter reads the event's own ``event_type`` rather than pinning
    EARNINGS: this seam only assembles the *candidate pool*, and
    ``previous_comparable`` is what decides comparability within it (same
    series, different release period, same speaker, same ticker). Pinning
    EARNINGS here meant a GDP release was handed an empty pool and answered
    "no comparable event" — true of the rows it was given, false of the world,
    for a series published quarterly since 1947. §15's never-cross-type rule
    is preserved: the pool is single-type by construction.

    The ticker filter stays, but only for ticker-scoped events. A macro
    release has no ticker, and requiring one returned ``[]`` before the query
    ever ran.
    """
    ticker = event_row.ticker
    scheduled = _as_utc(event_row.scheduled_at)
    predicates = [
        EventRow.event_type == event_row.event_type,
        EventRow.scheduled_at < event_row.scheduled_at,
    ]
    if ticker:
        predicates.append(EventRow.ticker == ticker)
    rows = (
        (
            await session.execute(
                select(EventRow).where(*predicates).order_by(EventRow.scheduled_at)
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


def _as_utc(value: datetime) -> datetime:
    """Stored instants are UTC; SQLite hands them back naive (same convention
    as ``event_calendar._utc`` / ``routers/events._iso``)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def event_date_et(row: EventRow) -> date:
    """The ET calendar date an event row lands on.

    The reaction windows are indexed by TRADING DAY, and the trading day is an
    Eastern-time fact: an AMC release at 20:15 ET is 00:15 UTC the next day,
    and reading its UTC date would push the whole reaction window one bar
    forward. ``event_timezone`` is the event's own zone (a macro release
    asserts "08:30 ET"), but the US equity session the bars measure is ET, so
    the conversion target is ET for every row.
    """
    return _as_utc(row.scheduled_at).astimezone(EASTERN).date()


def _session_of(row: EventRow) -> EventSession:
    """The row's session, defaulting to UNKNOWN for an unrecognised value.

    UNKNOWN is the conservative branch in :func:`first_reaction_index` (a
    two-day span, explicitly flagged), so a session string the enum has since
    dropped degrades into the widest honest window rather than raising out of
    a read endpoint.
    """
    try:
        return EventSession(row.session)
    except ValueError:
        return EventSession.UNKNOWN


# ---------------------------------------------------------------------------
# Rendering — frozen results to JSON
# ---------------------------------------------------------------------------


def _date_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _horizon_map(values, horizons: tuple[int, ...]) -> dict[str, float | None]:
    """``{1: 0.04}`` -> ``{"1D": 0.04}`` — JSON object keys must be strings,
    and "1D" is the label the UI prints, so the conversion happens once here
    rather than in three places in the client."""
    return {f"{k}D": values.get(k) for k in horizons}


def reaction_to_dict(result: ReactionResult) -> dict:
    """One :class:`ReactionResult` as JSON.

    ``basis`` and ``window`` travel with the numbers on purpose (§85): a 1D
    return measured over an UNKNOWN-session two-day span is a different
    measurement from one measured over a single session, and the UI cannot
    say so unless the payload does.
    """
    horizons = result.horizons or HORIZONS
    return {
        "event_date_et": _date_iso(result.event_date_et),
        "session": result.session.value if result.session is not None else None,
        "basis": result.basis,
        "bars_available": result.bars_available,
        "pre_event_close": result.pre_event_close,
        "pre_event_date": _date_iso(result.pre_event_date),
        "react_open": result.react_open,
        "react_close": result.react_close,
        "react_date": _date_iso(result.react_date),
        "gap_return": result.gap_return,
        "returns": _horizon_map(result.returns, horizons),
        "abs_returns": _horizon_map(result.abs_returns, horizons),
        "window_end_dates": {
            f"{k}D": _date_iso(result.window_end_dates.get(k)) for k in horizons
        },
        "max_favorable_excursion": result.max_favorable_excursion,
        "max_adverse_excursion": result.max_adverse_excursion,
        "reasons": dict(result.reasons),
    }


def abnormal_to_dict(result: AbnormalResult, horizons: tuple[int, ...]) -> dict:
    """One :class:`AbnormalResult` as JSON (stock minus SPY, same windows)."""
    return {
        "benchmark": BENCHMARK_SYMBOL,
        "benchmark_available": result.benchmark_available,
        "basis": result.basis,
        "abnormal": _horizon_map(result.abnormal, horizons),
        "abnormal_gap": result.abnormal_gap,
        "benchmark_returns": _horizon_map(result.benchmark_returns, horizons),
        "benchmark_gap_return": result.benchmark_gap_return,
        "reasons": dict(result.reasons),
    }


def history_stats_to_dict(stats: HistoryStats) -> dict:
    """One :class:`HistoryStats` window as JSON.

    ``n`` (window asked for) and ``n_available`` (events that actually
    resolved) are both carried, and ``positive_count`` travels beside
    ``positive_frequency`` so the UI prints "5/8" rather than a bare 0.625
    that reads as a probability — §19/§64 forbid presenting a historical count
    as a forecast.
    """
    return {
        "horizon": f"{stats.horizon}D",
        "n": stats.n,
        "n_available": stats.n_available,
        "median_abs": stats.median_abs,
        "mean_abs": stats.mean_abs,
        "p75_abs": stats.p75_abs,
        "p90_abs": stats.p90_abs,
        "max_abs": stats.max_abs,
        "positive_count": stats.positive_count,
        "positive_frequency": stats.positive_frequency,
        "reasons": dict(stats.reasons),
    }


#: Metrics whose null-ness is entirely explained by another field's null-ness.
#: ``sma200_distance_pct`` cannot exist without ``sma200``, and the library
#: records the reason once, on the parent. Repeating the parent's reason here
#: is what lets the UI print "Unavailable — needs 200 bars, have 60" against
#: the DISTANCE tile the user is actually looking at, instead of leaving that
#: tile blank while the explanation sits on a field the tile does not show.
DERIVED_FROM: dict[str, str] = {
    "sma20_distance_pct": "sma20",
    "sma50_distance_pct": "sma50",
    "sma200_distance_pct": "sma200",
    "atr_pct": "atr14",
    "run_up_pct": "since_anchor_return",
}


def _with_derived_reasons(context: PriceContext, rendered: dict) -> dict:
    """Reasons for the fields whose absence is inherited, not independent.

    Never OVERWRITES a reason the library gave (``atr_pct`` has its own
    ``last_close_not_positive`` case): this only fills a gap, and only when
    the derived field is actually null and its parent carries an explanation.
    """
    reasons = dict(rendered["reasons"])
    for derived, parent in DERIVED_FROM.items():
        if rendered.get(derived) is None and derived not in reasons:
            parent_reason = context.reasons.get(parent)
            if parent_reason is not None:
                reasons[derived] = f"{parent} unavailable: {parent_reason}"
    rendered["reasons"] = reasons
    return rendered


def price_context_to_dict(context: PriceContext) -> dict:
    """One :class:`PriceContext` as JSON (§31, §32).

    ``anchor_basis`` is load-bearing: ``"previous_event"`` means the run-up is
    measured since the last print (the §32 framing), while
    ``"default_63_bars"`` means there was no previous print and the window is
    a plain 3-month lookback. Rendering the second as "since the last
    earnings" would be a fabricated claim about the measurement itself.
    """
    rendered = {
        "as_of_date_et": _date_iso(context.as_of_date_et),
        "anchor_date_et": _date_iso(context.anchor_date_et),
        "anchor_close": context.anchor_close,
        "anchor_basis": context.anchor_basis,
        "last_close": context.last_close,
        "bars_through": _date_iso(context.bars_through),
        "n_bars": context.n_bars,
        "since_anchor_return": context.since_anchor_return,
        "run_up_pct": context.run_up_pct,
        "benchmark": BENCHMARK_SYMBOL,
        "benchmark_return": context.benchmark_return,
        "relative_return": context.relative_return,
        "max_drawdown": context.max_drawdown,
        "realized_vol_20d": context.realized_vol_20d,
        "realized_vol_since_anchor": context.realized_vol_since_anchor,
        "volume_trend": context.volume_trend,
        "sma20": context.sma20,
        "sma50": context.sma50,
        "sma200": context.sma200,
        "sma20_distance_pct": context.sma20_distance_pct,
        "sma50_distance_pct": context.sma50_distance_pct,
        "sma200_distance_pct": context.sma200_distance_pct,
        "atr14": context.atr14,
        "atr_pct": context.atr_pct,
        "high_52w": context.high_52w,
        "low_52w": context.low_52w,
        "distance_from_52w_high_pct": context.distance_from_52w_high_pct,
        "distance_from_52w_low_pct": context.distance_from_52w_low_pct,
        "reasons": dict(context.reasons),
    }
    return _with_derived_reasons(context, rendered)


def _unavailable_entries(prefix: str, reasons) -> list[dict]:
    """Flatten a result's ``reasons`` into ``[{"field", "reason"}]`` rows.

    One flat list across the whole payload is what lets the UI render
    "Unavailable — <reason>" generically: it never has to know which library
    function owns which key, and a reason added upstream shows up without a
    client change.
    """
    return [
        {"field": f"{prefix}.{key}", "reason": value}
        for key, value in sorted(reasons.items())
    ]


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


async def build_price_context(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider_name: str,
) -> dict:
    """The whole price block for one event, as of one instant (§17, §31, §32).

    Order of operations is the contract: load bars -> gate them at ``as_of``
    -> select the past events knowable at ``as_of`` -> measure. Nothing is
    measured before the gate, so no reaction can be computed from a bar the
    caller could not have seen. ``as_of`` is REQUIRED (audit §7.2 rule 2 — a
    seam that defaults it to ``now()`` cannot answer a historical question).

    Non-ticker events (macro, Fed) return ``{"available": false, "reason":
    "no_ticker"}``: they move an index, not a single name, and Phase G is
    where the §39 macro proxies get their own multi-asset treatment. Guessing
    SPY as "the" underlying here would invent an exposure the event does not
    have.

    They still carry an ``anchor_event``. The anchor is registry identity —
    which prior release this one is compared against — not a price
    measurement, and it is what the evidence bundle reads to name the previous
    comparable event. Returning early without it made every macro release
    report "no comparable event" for want of a stock price it was never going
    to have. ``available`` stays false: no bars means no reactions, and the
    anchor block carries no measurement.
    """
    ticker = (event_row.ticker or "").strip().upper()
    payload_base = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": event_row.ticker,
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
    }
    if not ticker:
        rows = await _past_comparable_rows(session, event_row, as_of)
        prior, prior_reason = previous_comparable(
            row_to_event(event_row), [row_to_event(row) for row in rows]
        )
        return {
            **payload_base,
            "available": False,
            "reason": "no_ticker",
            "anchor_event": (
                {
                    "event_id": prior.event_id,
                    "event_key": prior.event_key,
                    "scheduled_at_utc": _as_utc(prior.scheduled_at).isoformat(),
                    "comparison_reason": prior_reason,
                }
                if prior is not None
                else None
            ),
        }

    ticker_bars, bars_reason = await _load_bars(session, ticker, provider_name, as_of)
    bench_bars, bench_reason = await _load_bars(
        session, BENCHMARK_SYMBOL, provider_name, as_of
    )

    unavailable: list[dict] = []
    if bars_reason is not None:
        unavailable.append({"field": "bars", "reason": bars_reason})
    if bench_reason is not None:
        unavailable.append(
            {"field": f"benchmark_bars.{BENCHMARK_SYMBOL}", "reason": bench_reason}
        )

    # --- previous comparable events ---------------------------------------
    past_rows = await _past_comparable_rows(session, event_row, as_of)
    previous, comparison_reason = previous_comparable(
        row_to_event(event_row), [row_to_event(row) for row in past_rows]
    )
    anchor_row = None
    if previous is not None and previous.event_id is not None:
        anchor_row = next(
            (row for row in past_rows if row.id == previous.event_id), None
        )
    anchor_date: date | None = None

    previous_events: list[dict] = []
    reactions: list[ReactionResult] = []
    for row in past_rows:
        row_date = event_date_et(row)
        row_session = _session_of(row)
        reaction = event_reaction(
            ticker_bars, row_date, row_session, horizons=HORIZONS
        )
        abnormal = abnormal_vs(reaction, bench_bars, row_date, row_session)
        reactions.append(reaction)
        entry = {
            "event_id": row.id,
            "event_key": row.event_key,
            "date_et": row_date.isoformat(),
            "session": row_session.value,
            "status": row.status,
            "is_previous_comparable": (
                previous is not None and previous.event_id == row.id
            ),
            "bars_available": reaction.bars_available,
            "reaction": reaction_to_dict(reaction),
            "abnormal_vs_spy": abnormal_to_dict(abnormal, HORIZONS),
        }
        if anchor_row is not None and row.id == anchor_row.id:
            # THE RUN-UP IS MEASURED FROM THE PREVIOUS PRINT'S PRE-EVENT
            # CLOSE, not from its calendar date. For an AMC print those are
            # the same bar, but for a BMO print the pre-event close is the
            # PREVIOUS day's — anchoring on the event date would measure the
            # run-up from a close that already contains that print's own
            # reaction, understating (or inverting) the move since it. The
            # reaction result already resolved that bar session-correctly, so
            # its ``pre_event_date`` is the honest anchor; when the reaction
            # could not be located there is no anchor bar to name and the
            # library falls back to its default window with a stated basis.
            anchor_date = reaction.pre_event_date
        if not reaction.bars_available:
            # The library's own string — "bars unavailable before 2024-03-20"
            # or "no bar after the event yet; bars end ..." — passed through
            # verbatim so the UI can print the real boundary date rather than
            # a generic "no data".
            entry["reason"] = reaction.reasons.get("bars", "reaction_unavailable")
        previous_events.append(entry)

    # --- §19/§64 history strip --------------------------------------------
    stats_block: dict[str, dict] = {}
    for horizon in HISTORY_HORIZONS:
        windows = history_stats(reactions, horizon=horizon)
        stats_block[f"{horizon}D"] = {
            label: history_stats_to_dict(stats) for label, stats in windows.items()
        }

    # --- §31/§32 pre-event positioning ------------------------------------
    as_of_et_date = as_of.astimezone(EASTERN).date()
    context = pre_event_price_context(
        ticker_bars,
        anchor_date_et=anchor_date,
        as_of_date_et=as_of_et_date,
        bench_bars=bench_bars or None,
    )
    pre_event = price_context_to_dict(context)
    unavailable.extend(_unavailable_entries("pre_event", pre_event["reasons"]))

    bars_available = bool(ticker_bars)
    return {
        **payload_base,
        "available": True,
        "provenance": {"bars": "DATA", "metrics": "QUANT"},
        "bars": (
            {"available": True}
            if bars_available
            else {"available": False, "reason": bars_reason or "no_bars_available"}
        ),
        "data_freshness": {
            "bars_through": _date_iso(context.bars_through),
            "bars_source": provider_name,
            "n_bars": context.n_bars,
            "benchmark": BENCHMARK_SYMBOL,
            "benchmark_bars_through": _date_iso(
                bench_bars[-1].date if bench_bars else None
            ),
            "benchmark_n_bars": len(bench_bars),
        },
        "pre_event": pre_event,
        "anchor_event": (
            {
                "event_id": previous.event_id,
                "event_key": previous.event_key,
                "date_et": _date_iso(anchor_date),
                "comparison_reason": comparison_reason,
            }
            if previous is not None
            else None
        ),
        "previous_events": previous_events,
        "history_stats": stats_block,
        "horizons": [f"{k}D" for k in HORIZONS],
        "not_backtestable": list(NOT_BACKTESTABLE),
        "unavailable": unavailable,
    }
