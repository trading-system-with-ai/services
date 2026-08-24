"""Event replay — minute bars, the §20 bundle and the §60 history (Phase C, U3).

THE SEAM, EXACTLY LIKE ``event_price.py``. Every number in these payloads is
computed by ``libs/trading_core/events/replay.py`` (and ``reaction.py``
underneath it), which are pure stdlib and may not import ``apps/`` or
``libs.market_data`` (audit §7.4). This module is the only place the two
halves meet for the intraday view: it fetches and stores minute bars, reads
them back, converts ORM rows to the pure :class:`MinuteBar`, hands them to the
library and renders the frozen results. It computes nothing itself — no
arithmetic on a price lives here.

WRITES ARE USER-TRIGGERED, READS NEVER FETCH. This is the one place the Phase
E1 lazy-backfill pattern is deliberately NOT copied. ``ensure_daily_bars``
fetches on a GET because a daily history is a few hundred rows for a symbol
the platform already tracks. An event minute window is ~1,000 bars for ONE
event, and the §60 history table shows up to twelve events — a GET that
backfilled them all would issue twelve paginated provider fetches and write
twelve thousand rows because somebody opened a page. So:

- ``GET  /replay`` and ``GET  /history`` read STORED minute bars only. An
  event with none says so, with a reason, and the daily reaction still
  answers. Absence here is the normal state, not an error.
- ``POST /replay/backfill`` (one event) and ``POST /history/backfill``
  (bounded, ``last<=12``) are USER actions that fetch and store.

THE AS-OF GATE IS APPLIED TO THE BARS, ONCE, ON BOTH SIDES (§14, §85, §96).
Daily bars go through ``reaction.as_of_bar_filter`` inside
``event_price._load_bars``; minute bars are gated by their own instant —
``ts <= as_of`` — in the SQL that loads them, before any reaction is measured.
The minute rule is simpler than the daily one on purpose: a daily bar is only
knowable after its 16:00 ET close, but a minute bar stamped 14:31 UTC IS the
fact of that minute and is knowable the moment it ends. Gating the query
rather than the answer is what makes "what did we know at 09:45" a shorter bar
list instead of a trimmed conclusion.

A FUTURE EVENT IS AN ANSWER, NOT AN ERROR. ``scheduled_at > as_of`` returns
``{"available": false, "reason": ...}`` WITH the event reference: the Catalyst
page opens on upcoming earnings, and the replay tab must say "this has not
happened yet" rather than 404 or invent an empty reaction.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.market_data import (
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
    get_provider,
)
from libs.trading_core.events import EASTERN
from libs.trading_core.events.reaction import (
    DailyBar,
    ReactionResult,
    abnormal_vs,
    event_reaction,
)
from libs.trading_core.events.replay import (
    EXTENDED_CLOSE_ET,
    EXTENDED_OPEN_ET,
    MinuteBar,
    build_event_replay,
    history_table,
    intraday_reaction,
)
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.models.enums import EventSession

from . import audit
from .db import EventRow, MarketCalendarRow, StockBar1mRow
from .event_price import (
    BENCHMARK_SYMBOL,
    HORIZONS,
    _as_utc,
    _load_bars,
    _past_comparable_rows,
    _session_of,
    abnormal_to_dict,
    event_date_et,
    reaction_to_dict,
)

logger = logging.getLogger(__name__)

#: The audit ``entity_type`` for a minute-bar backfill — the table it wrote,
#: matching the ``stock_bars_daily`` / ``fundamental_statements`` precedent.
ENTITY_TYPE = "stock_bars_1m"

#: Minimum spacing between provider ATTEMPTS for one (ticker, event) window,
#: in seconds. Process-local like ``analysis._refresh_attempts`` and
#: ``fundamentals._refresh_attempts``; a restart re-attempts, which is correct
#: — a cold process has no evidence the vendor is still failing. Six hours,
#: not thirty minutes: an event window is a CLOSED interval in the past, so a
#: fetch that came back empty will keep coming back empty (a halted symbol, a
#: date before the symbol listed, a plan without minute data). Re-asking
#: sooner cannot change the answer.
BACKFILL_ATTEMPT_SECONDS = 6 * 60 * 60

#: Per-``(ticker, event_key)`` last provider attempt, success or failure.
_backfill_attempts: dict[tuple[str, str], datetime] = {}

#: How many past events ``POST /history/backfill`` fetches by default, and the
#: hard ceiling. The default matches the §60 table's smallest toggle (LAST 4)
#: so the common button press is the cheap one; the ceiling matches its
#: largest (LAST 12) so no request can ask for an unbounded number of
#: paginated provider fetches.
DEFAULT_HISTORY_BACKFILL = 4
MAX_HISTORY_BACKFILL = 12

#: How many trading days forward the next-session search scans before giving
#: up. Five covers a long weekend plus a stacked holiday; beyond that the
#: honest answer is "no next session found", not a wider guess.
NEXT_SESSION_SEARCH_DAYS = 8

#: Sessions whose reaction lands on the NEXT trading day — the release happens
#: when the market cannot respond, so the window must span two ET dates.
#: UNKNOWN is here because ``intraday_reaction`` treats it as AFTER_MARKET
#: (with a low-confidence basis), and a window that stopped at 20:00 on the
#: release day would deny that assumed rule the bars it needs to be tested.
SPANS_NEXT_SESSION: frozenset[EventSession] = frozenset(
    {EventSession.AFTER_MARKET, EventSession.UNKNOWN}
)

#: §85 / audit §7.3 — what the replay view deliberately does NOT claim to
#: reconstruct point-in-time. Minute-level option data is the big one: the
#: §60 implied-move column stays unavailable until Phase I.
NOT_BACKTESTABLE: tuple[str, ...] = (
    "historical_implied_move",
    "historical_atm_iv",
    "intraday_option_quotes",
    "intraday_order_flow",
)

#: Reason a GET gives when an event has no stored minute bars. Stated once so
#: the replay endpoint and the history table say the SAME thing, and so the UI
#: can match on it to decide whether to offer the "Load minute bars" button.
NO_STORED_BARS_REASON = (
    "no minute bars stored for this event window — use POST "
    "/api/events/{id}/replay/backfill to fetch them"
)


# ---------------------------------------------------------------------------
# ORM -> pure value
# ---------------------------------------------------------------------------


def to_minute_bars(rows: list[StockBar1mRow]) -> list[MinuteBar]:
    """Convert stored minute rows to the pure library's value, oldest first.

    RE-STAMPING THE INSTANT IS LOAD-BEARING, not defensive tidying. SQLite
    hands a ``DateTime(timezone=True)`` column back NAIVE (the same caveat
    ``event_price._as_utc`` and ``fundamentals._StoredStatement`` exist for),
    and :class:`MinuteBar` REFUSES a naive timestamp rather than assuming one
    — correctly, since guessing the zone of a minute bar moves an after-hours
    print into the regular session. Without this conversion every stored bar
    would raise on construction under the test harness and be silently absent
    under Postgres-with-a-driver-that-drops-tzinfo.

    The sort is re-asserted because ``intraday_reaction`` REFUSES unsorted
    input (it cannot tell a mis-ordered series from a mis-stamped one), and a
    ValueError out of a read endpoint is a worse failure than a redundant
    sort over a list the query already ordered.
    """
    return sorted(
        (
            MinuteBar(
                ts_utc=_as_utc(row.ts),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume or 0),
            )
            for row in rows
        ),
        key=lambda bar: bar.ts_utc,
    )


# ---------------------------------------------------------------------------
# The event window — which instants a replay needs
# ---------------------------------------------------------------------------


def _et_instant(day: date, hm: tuple[int, int]) -> datetime:
    """An ET wall clock on ``day`` as its UTC instant.

    Built in Eastern and converted, never by adding a fixed UTC offset: 04:00
    ET is 08:00Z in summer and 09:00Z in winter, and an offset-arithmetic
    window would silently lose the first hour of pre-market for half the year.
    """
    return datetime.combine(day, time(*hm), tzinfo=EASTERN).astimezone(timezone.utc)


async def _next_session_date(
    session: AsyncSession,
    after: date,
    *,
    sessions: Sequence[date] | None = None,
) -> tuple[date, str]:
    """``(next trading date after ``after``, how it was determined)``.

    Prefers the STORED ``market_calendar`` grid — the table that exists
    precisely because holidays are not weekday arithmetic (a Thursday-Friday
    Thanksgiving break, a Monday holiday, a funeral closure). When no session
    row is stored for the range the fallback is the next WEEKDAY, and the
    basis string says so: a window built on the fallback may miss the true
    next session by a day over a holiday, and a caller that cannot tell which
    rule produced it cannot report that risk.

    The fallback is not a fabrication — it is a NAMED approximation of the
    window to FETCH, never of a price. A window one day short simply yields no
    next-session bars and the reaction reports its own honest absence.
    """
    horizon = after + timedelta(days=NEXT_SESSION_SEARCH_DAYS)
    if sessions is None:
        rows = (
            (
                await session.execute(
                    select(MarketCalendarRow.session_date)
                    .where(
                        MarketCalendarRow.session_date > after,
                        MarketCalendarRow.session_date <= horizon,
                    )
                    .order_by(MarketCalendarRow.session_date)
                )
            )
            .scalars()
            .all()
        )
    else:
        # A pre-loaded grid, passed by the history path so twelve events cost
        # ONE calendar query instead of twelve. Filtered to the same bounds
        # the query above uses, so both branches answer identically.
        rows = sorted(d for d in sessions if after < d <= horizon)
    if rows:
        return rows[0], "market_calendar"

    day = after + timedelta(days=1)
    for _ in range(NEXT_SESSION_SEARCH_DAYS):
        if day.weekday() < 5:
            return day, "next_weekday_fallback"
        day += timedelta(days=1)
    return after + timedelta(days=1), "next_weekday_fallback"  # pragma: no cover


async def event_window(
    session: AsyncSession,
    event_row: EventRow,
    *,
    sessions: Sequence[date] | None = None,
) -> tuple[datetime, datetime, str]:
    """``(start_utc, end_utc, basis)`` — the minutes one replay needs.

    The window always OPENS at 04:00 ET on the release day: a BMO release at
    07:00 ET moves the pre-market tape, and a window opening at 09:30 would
    measure the reaction from a price that had already absorbed it.

    Where it CLOSES is the session's question. An AMC (or UNKNOWN) release has
    its reaction on the NEXT trading day, so the window runs to 20:00 ET
    THERE — two ET dates, because §17's "+30m after the open" is a fact about
    tomorrow morning. A BMO or DURING_MARKET release is fully answered by its
    own day, so the window ends at 20:00 ET on it; fetching a second day would
    be a thousand bars nobody reads.
    """
    day = event_date_et(event_row)
    start = _et_instant(day, EXTENDED_OPEN_ET)
    session_kind = _session_of(event_row)
    if session_kind not in SPANS_NEXT_SESSION:
        return start, _et_instant(day, EXTENDED_CLOSE_ET), "release_day_only"
    next_day, how = await _next_session_date(session, day, sessions=sessions)
    return start, _et_instant(next_day, EXTENDED_CLOSE_ET), f"spans_next_session:{how}"


async def _session_dates(
    session: AsyncSession, rows: Sequence[EventRow]
) -> list[date]:
    """Stored session dates covering every row's next-session search.

    One range query spanning all the events, so the §60 table's per-row window
    resolution reads a list instead of re-querying. Returns ``[]`` when no
    grid is stored, which the caller's fallback handles by name.
    """
    if not rows:
        return []
    days = [event_date_et(row) for row in rows]
    return list(
        (
            await session.execute(
                select(MarketCalendarRow.session_date)
                .where(
                    MarketCalendarRow.session_date > min(days),
                    MarketCalendarRow.session_date
                    <= max(days) + timedelta(days=NEXT_SESSION_SEARCH_DAYS),
                )
                .order_by(MarketCalendarRow.session_date)
            )
        )
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# Stored minute bars
# ---------------------------------------------------------------------------


async def _stored_window_bars(
    session: AsyncSession,
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    as_of: datetime | None = None,
) -> list[StockBar1mRow]:
    """Stored minute rows in ``[start, end]``, ascending, gated at ``as_of``.

    THE AS-OF GATE IS IN THE QUERY, not applied to the result (§14, §96): the
    library must never see a bar the caller could not have seen, and filtering
    afterwards leaves a window in which some future code path reads the
    unfiltered list. ``ts <= as_of`` is the whole rule for a minute — unlike a
    daily bar, which is not knowable until its session closes, a completed
    minute IS the fact of that minute.
    """
    stmt = (
        select(StockBar1mRow)
        .where(
            StockBar1mRow.ticker == ticker,
            StockBar1mRow.ts >= start,
            StockBar1mRow.ts <= end,
        )
        .order_by(StockBar1mRow.ts)
    )
    if as_of is not None:
        stmt = stmt.where(StockBar1mRow.ts <= as_of)
    return list((await session.execute(stmt)).scalars().all())


async def _stored_window_count(
    session: AsyncSession, ticker: str, start: datetime, end: datetime
) -> int:
    """How many minute rows are already stored for this window.

    Counted rather than loaded: the only question the backfill asks is "is
    this window already here", and materialising a thousand ORM objects to
    answer a yes/no is the kind of cost that turns a page load into a
    timeout.
    """
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(StockBar1mRow)
                .where(
                    StockBar1mRow.ticker == ticker,
                    StockBar1mRow.ts >= start,
                    StockBar1mRow.ts <= end,
                )
            )
        ).scalar_one()
    )


# ---------------------------------------------------------------------------
# The backfill (USER-triggered writes)
# ---------------------------------------------------------------------------


async def ensure_event_window_bars(
    session: AsyncSession,
    ticker: str,
    event_row: EventRow,
    provider_name: str,
    *,
    now: datetime,
) -> dict:
    """Fetch and store the minute window for ONE event. The honest report.

    The minute-bar counterpart of ``ensure_daily_bars`` and
    ``ensure_fundamentals``, and deliberately the ONLY path that writes
    ``stock_bars_1m`` — so the DATA_BACKFILL audit trail and the provenance
    story stay single-sourced (rule 12, ADR-003).

    ALREADY-STORED WINDOWS ARE NOT REFETCHED. An event window is a CLOSED
    interval in the past: once the minutes are stored they cannot change, so a
    second press of the button is a no-op with ``fetched: false`` rather than
    a second thousand-row write. That is a stronger rule than the daily
    refresh's "append what is newer", and it is the right one here precisely
    because the window has an end.

    ``now`` is REQUIRED and is the only clock this function reads — the
    throttle uses it — so a test drives the cadence without patching time. It
    is NOT an ``as_of``: ingestion has no as-of (audit §7.2 rule 1); it writes
    what the vendor serves and lets the read half decide what was knowable
    when.

    EVERY PROVIDER FAILURE IS A NAMED SKIP, NEVER AN EXCEPTION. An
    unconfigured provider, a plan without minute data (403 ->
    :class:`CapabilityNotAvailable`), a transport error and an unexpected
    exception all return ``{"fetched": false, "reason": ...}``. The caller is
    a USER-pressed button; it must report why nothing arrived, not 5xx.

    Upsert is on the PK ``(ticker, ts)``: a bar whose instant is already
    stored is left alone rather than rewritten, so a window that partially
    overlaps a previous fetch cannot duplicate a minute (§44 rule 18 — stored
    real data is never silently replaced).
    """
    symbol = (ticker or "").strip().upper()
    base: dict = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": symbol or None,
    }
    if not symbol:
        return {**base, "fetched": False, "bars": 0, "reason": "no_ticker"}

    start, end, window_basis = await event_window(session, event_row)
    base |= {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "window_basis": window_basis,
    }

    stored = await _stored_window_count(session, symbol, start, end)
    if stored:
        return {
            **base,
            "fetched": False,
            "bars": 0,
            "stored_bars": stored,
            "reason": "window already stored",
        }

    key = (symbol, event_row.event_key or str(event_row.id))
    last_attempt = _backfill_attempts.get(key)
    if (
        last_attempt is not None
        and (now - last_attempt).total_seconds() < BACKFILL_ATTEMPT_SECONDS
    ):
        # A window that came back empty will keep coming back empty — the
        # interval is closed and in the past. Re-asking is cost with no
        # possible new answer.
        return {
            **base,
            "fetched": False,
            "bars": 0,
            "stored_bars": 0,
            "reason": "backfill recently attempted for this event window",
        }
    _backfill_attempts[key] = now

    try:
        provider = get_provider(provider_name)
        fetched = list(provider.get_intraday_bars(symbol, start, end))
    except (ProviderNotConfigured, CapabilityNotAvailable, MarketDataError) as exc:
        # Named, expected refusals: no provider, no subscription, vendor
        # error. Nothing is stored; the caller states the reason.
        logger.info(
            "intraday_backfill_unavailable",
            extra={"extra_fields": {"ticker": symbol, "reason": str(exc)}},
        )
        return {**base, "fetched": False, "bars": 0, "stored_bars": 0, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 — a button press must not 5xx
        logger.exception(
            "intraday_backfill_failed", extra={"extra_fields": {"ticker": symbol}}
        )
        return {**base, "fetched": False, "bars": 0, "stored_bars": 0, "reason": str(exc)}

    if not fetched:
        # An honest empty window (a holiday, a halted symbol, a range before
        # the symbol listed). No audit row: nothing was written.
        return {
            **base,
            "fetched": True,
            "bars": 0,
            "stored_bars": 0,
            "reason": f"provider {provider_name!r} returned no minute bars for this window",
        }

    seen: set[datetime] = set()
    rows: list[StockBar1mRow] = []
    for bar in fetched:
        ts = _as_utc(bar.ts)
        if ts < start or ts > end or ts in seen:
            # Outside the window this event asked for, or a duplicate instant
            # the provider served twice. Neither belongs in this window's
            # store, and a duplicate would violate the PK.
            continue
        seen.add(ts)
        rows.append(
            StockBar1mRow(
                ticker=symbol,
                ts=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
            )
        )
    if not rows:
        return {
            **base,
            "fetched": True,
            "bars": 0,
            "stored_bars": 0,
            "reason": "provider returned no bars inside the requested window",
        }

    session.add_all(rows)
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.DATA_BACKFILL,
        entity_type=ENTITY_TYPE,
        entity_id=symbol,
        details={
            "kind": "intraday_event_window",
            "ticker": symbol,
            "event_key": event_row.event_key,
            "event_id": event_row.id,
            "bars": len(rows),
            "provider": provider_name,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "window_basis": window_basis,
            "first": rows[0].ts.isoformat(),
            "last": rows[-1].ts.isoformat(),
        },
    )
    await session.commit()
    return {
        **base,
        "fetched": True,
        "bars": len(rows),
        "stored_bars": len(rows),
        "provider": provider_name,
        "first_ts_utc": rows[0].ts.isoformat(),
        "last_ts_utc": rows[-1].ts.isoformat(),
    }


async def backfill_history_windows(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider_name: str,
    last: int = DEFAULT_HISTORY_BACKFILL,
    now: datetime,
) -> dict:
    """Backfill the minute windows of the last ``last`` comparable events.

    BOUNDED BY CONSTRUCTION. ``last`` is clamped to
    ``[1, MAX_HISTORY_BACKFILL]`` — twelve is the §60 table's largest toggle,
    so no request can ask for more windows than the UI can display, however
    the query string is written. Each window is one paginated provider fetch
    of roughly a thousand bars; an unbounded ``last`` would be a
    self-inflicted denial of service wearing a button's clothes.

    The events are the SAME point-in-time pool the history table renders
    (``_past_comparable_rows``: CONFIRMED/REVISED, strictly earlier, at or
    before ``as_of``), NEWEST first, so pressing "backfill last 4" fills the
    four rows a reader is actually looking at.
    """
    requested = int(last)
    bounded = max(1, min(MAX_HISTORY_BACKFILL, requested))
    rows = await _past_comparable_rows(session, event_row, as_of)
    targets = list(reversed(rows))[:bounded]

    results: list[dict] = []
    for row in targets:
        results.append(
            await ensure_event_window_bars(
                session, row.ticker or "", row, provider_name, now=now
            )
        )
    return {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": event_row.ticker,
        "requested": requested,
        "last": bounded,
        "max_last": MAX_HISTORY_BACKFILL,
        "events_available": len(rows),
        "events_attempted": len(targets),
        "bars": sum(int(r.get("bars") or 0) for r in results),
        "results": results,
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Reading — the §20 replay and the §60 history
# ---------------------------------------------------------------------------


def _event_ref(event_row: EventRow) -> dict:
    """The registry facts every replay payload carries, available or not.

    An unavailable replay STILL carries this block: "NVDA reports after the
    close on 2026-08-27 and has not happened yet" is a useful answer, and a
    bare ``{"available": false}`` is not.
    """
    return {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_type": event_row.event_type,
        "ticker": event_row.ticker,
        "date_et": event_date_et(event_row).isoformat(),
        "session": _session_of(event_row).value,
        "status": event_row.status,
        "source_url": event_row.source_url,
        "scheduled_at_utc": _as_utc(event_row.scheduled_at).isoformat(),
    }


async def _intraday_for(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    pre_event_close: float | None,
    sessions: Sequence[date] | None = None,
) -> tuple[object | None, str | None, dict]:
    """``(IntradayReaction|None, reason_if_none, freshness)`` from STORED bars.

    Reads only; never fetches (see the module docstring). ``pre_event_close``
    comes from the DAILY series because only it knows the official close —
    reconstructing it from the last 15:59 minute bar would quietly substitute
    the last trade for the settled close, which is a different number on any
    day with a closing auction imbalance.
    """
    ticker = (event_row.ticker or "").strip().upper()
    if not ticker:
        return None, "no_ticker", {}
    start, end, window_basis = await event_window(session, event_row, sessions=sessions)
    rows = await _stored_window_bars(session, ticker, start, end, as_of=as_of)
    freshness = {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "window_basis": window_basis,
        "minute_bars_stored": len(rows),
        "minute_bars_through": _as_utc(rows[-1].ts).isoformat() if rows else None,
    }
    if not rows:
        return None, NO_STORED_BARS_REASON, freshness
    bars = to_minute_bars(rows)
    result = intraday_reaction(
        bars,
        event_ts_utc=_as_utc(event_row.scheduled_at),
        session=_session_of(event_row),
        pre_event_close=pre_event_close,
    )
    return result, None, freshness


def pre_event_close_for(
    bars: list[DailyBar], event_row: EventRow
) -> tuple[float | None, str | None]:
    """``(the settled close the intraday move is measured against, reason)``.

    RESOLVED INDEPENDENTLY OF THE DAILY REACTION, and that is the whole point
    of this function. ``event_reaction`` needs BOTH a pre-event bar and a
    REACTION bar, so at 09:44 ET on the morning after an AMC print — the exact
    instant the intraday view exists to answer — it returns
    ``bars_available: false`` and a ``pre_event_close`` of ``None``. Taking
    the anchor from there would make every minute-bar window null during the
    only window anyone is watching, and for a reason that is not true: the
    pre-event close settled at 16:00 the PREVIOUS afternoon and is perfectly
    knowable.

    The session rule is ``first_reaction_index``'s own, applied to the
    pre-event leg alone (§17): for an AMC/UNKNOWN release the anchor is the
    last bar dated ON OR BEFORE the release day — that day's close IS the
    last price before the release. For a BMO or DURING_MARKET release it is
    the last bar STRICTLY BEFORE it, because the release day's own close
    already contains the reaction.

    The bars are the caller's, already as-of gated, so this cannot reach past
    ``as_of`` however the anchor is chosen.
    """
    if not bars:
        return None, "no daily bars available to anchor the intraday move"
    day = event_date_et(event_row)
    kind = _session_of(event_row)
    if kind in (EventSession.AFTER_MARKET, EventSession.UNKNOWN):
        candidates = [bar for bar in bars if bar.date <= day]
    else:
        candidates = [bar for bar in bars if bar.date < day]
    if not candidates:
        return None, (
            f"no daily bar before the release; bars start {bars[0].date.isoformat()}"
        )
    close = float(candidates[-1].close)
    if close <= 0.0:
        return None, "pre_event_close_not_positive"
    return close, None


def _daily_pieces(
    ticker_bars: list[DailyBar],
    bench_bars: list[DailyBar],
    event_row: EventRow,
) -> tuple[ReactionResult, object]:
    """The E1 daily reaction and its SPY overlay for one event row.

    Uses ``event_price``'s own ``event_reaction``/``abnormal_vs`` call with
    the SAME ``HORIZONS`` the price tab uses, so the replay's "subsequent
    reaction" and the price tab's previous-event row can never disagree about
    the 1D number for the same print.
    """
    day = event_date_et(event_row)
    kind = _session_of(event_row)
    reaction = event_reaction(ticker_bars, day, kind, horizons=HORIZONS)
    return reaction, abnormal_vs(reaction, bench_bars, day, kind)


async def build_event_replay_payload(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider_name: str,
) -> dict:
    """The §20 replay bundle for one event, as of one instant.

    Order of operations is the contract, exactly as in
    ``event_price.build_price_context``: resolve the event -> load and gate
    the bars -> measure. Nothing is measured before the gate, so no reaction
    can be computed from a bar the caller could not have seen at ``as_of``.

    A FUTURE EVENT IS A 200 WITH A REASON. ``scheduled_at > as_of`` means the
    release has not happened at the instant asked about — there is no
    reaction to replay and inventing an empty one would read as "the market
    did nothing". The event reference travels with the refusal so the UI can
    still render the card.

    A NON-TICKER EVENT (macro, Fed) is likewise ``available: false`` with
    ``no_ticker``: a CPI print moves an index, not a single name, and Phase G
    is where the §39 macro proxies get their multi-asset treatment.
    """
    ref = _event_ref(event_row)
    as_of_utc = as_of.astimezone(timezone.utc)
    base = {"event": ref, "as_of": as_of_utc.isoformat()}

    scheduled = _as_utc(event_row.scheduled_at)
    if scheduled > as_of_utc:
        return {
            **base,
            "available": False,
            "reason": (
                f"event has not occurred as of as_of "
                f"({scheduled.isoformat()} > {as_of_utc.isoformat()})"
            ),
        }

    ticker = (event_row.ticker or "").strip().upper()
    if not ticker:
        return {**base, "available": False, "reason": "no_ticker"}

    ticker_bars, bars_reason = await _load_bars(session, ticker, provider_name, as_of_utc)
    bench_bars, bench_reason = await _load_bars(
        session, BENCHMARK_SYMBOL, provider_name, as_of_utc
    )

    reasons: dict[str, str] = {}
    if bars_reason is not None:
        reasons["daily_bars"] = bars_reason
    if bench_reason is not None:
        reasons[f"benchmark_bars.{BENCHMARK_SYMBOL}"] = bench_reason

    reaction, abnormal = _daily_pieces(ticker_bars, bench_bars, event_row)

    pre_close, pre_close_reason = pre_event_close_for(ticker_bars, event_row)
    if pre_close_reason is not None:
        reasons["pre_event_close"] = pre_close_reason
    intraday, intraday_reason, freshness = await _intraday_for(
        session, event_row, as_of=as_of_utc, pre_event_close=pre_close
    )
    if intraday_reason is not None:
        reasons["immediate_reaction"] = intraday_reason

    replay = build_event_replay(
        event_id=event_row.id,
        event_key=event_row.event_key,
        event_type=event_row.event_type,
        ticker=event_row.ticker,
        date_et=event_date_et(event_row),
        session=_session_of(event_row),
        status=event_row.status,
        source_url=event_row.source_url,
        release_ts_utc=scheduled,
        source_name=event_row.source_name,
        price_context_ref={
            "endpoint": f"/api/events/{event_row.id}/price-context",
            "as_of": as_of_utc.isoformat(),
        },
        fundamentals_ref={
            "endpoint": f"/api/events/{event_row.id}/fundamentals",
            "as_of": as_of_utc.isoformat(),
        },
        intraday=intraday,
        intraday_reason=intraday_reason,
        daily_dict=reaction_to_dict(reaction),
        abnormal_dict=abnormal_to_dict(abnormal, HORIZONS),
        data_freshness={
            **freshness,
            "daily_bars_through": (
                ticker_bars[-1].date.isoformat() if ticker_bars else None
            ),
            "daily_bars": len(ticker_bars),
            "benchmark": BENCHMARK_SYMBOL,
            "benchmark_bars": len(bench_bars),
            "bars_source": provider_name,
        },
        reasons=reasons,
    )
    rendered = replay.to_dict()
    # ONE "event" SHAPE ACROSS ALL THREE BRANCHES. ``EventReplay.to_dict``
    # emits its own ``event`` block, which is the §20 subset and lacks
    # ``scheduled_at_utc``; spreading it after ``base`` would silently drop
    # that field HERE while the unavailable branch above and the history
    # payload below both keep it. A UI reading ``event.scheduled_at_utc``
    # would then work on an upcoming event and break on a past one — so the
    # richer ref wins, deliberately and in one place.
    rendered["event"] = ref
    return {
        **base,
        "available": True,
        **rendered,
        "not_backtestable": list(NOT_BACKTESTABLE),
    }


async def build_event_history(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider_name: str,
    last: int | None = None,
) -> dict:
    """The §60 "LAST N EARNINGS" table for this event's ticker, as of ``as_of``.

    Same point-in-time pool as the price tab (``_past_comparable_rows``), same
    daily arithmetic, and ``intraday_30m`` filled ONLY from minute bars
    already stored — this function issues no provider call at all. On a fresh
    install every intraday cell is an honest absence with the backfill's
    reason, which is the correct and useful state: the table's daily columns
    are complete, and the reader chooses which windows are worth fetching.

    ``last`` trims to the newest N rows AFTER the reactions are measured, so
    the §19/§64 ``summary`` in the returned table describes exactly the rows
    shown rather than a wider history the reader cannot see.
    """
    ref = _event_ref(event_row)
    as_of_utc = as_of.astimezone(timezone.utc)
    base = {"event": ref, "as_of": as_of_utc.isoformat()}

    ticker = (event_row.ticker or "").strip().upper()
    if not ticker:
        return {**base, "available": False, "reason": "no_ticker"}

    ticker_bars, bars_reason = await _load_bars(session, ticker, provider_name, as_of_utc)
    bench_bars, bench_reason = await _load_bars(
        session, BENCHMARK_SYMBOL, provider_name, as_of_utc
    )

    rows = await _past_comparable_rows(session, event_row, as_of_utc)
    if last is not None:
        bounded = max(1, min(MAX_HISTORY_BACKFILL, int(last)))
        rows = rows[-bounded:]

    # ONE calendar query for the whole table. Each row's window needs the next
    # trading day after its release, and asking per row turned a twelve-row
    # page into twelve identical range queries over the same small grid.
    sessions = await _session_dates(session, rows)

    entries: list[dict] = []
    for row in rows:
        reaction, abnormal = _daily_pieces(ticker_bars, bench_bars, row)
        pre_close, _pre_reason = pre_event_close_for(ticker_bars, row)
        intraday, _reason, _freshness = await _intraday_for(
            session, row, as_of=as_of_utc, pre_event_close=pre_close, sessions=sessions
        )
        entries.append(
            {
                "event_id": row.id,
                "event_key": row.event_key,
                "date_et": event_date_et(row),
                "session": _session_of(row).value,
                "status": row.status,
                "reaction": reaction,
                "abnormal": abnormal,
                "intraday": intraday,
            }
        )

    table = history_table(entries)
    unavailable: list[dict] = []
    if bars_reason is not None:
        unavailable.append({"field": "bars", "reason": bars_reason})
    if bench_reason is not None:
        unavailable.append(
            {"field": f"benchmark_bars.{BENCHMARK_SYMBOL}", "reason": bench_reason}
        )
    return {
        **base,
        "available": True,
        **table,
        "ticker": event_row.ticker,
        "benchmark": BENCHMARK_SYMBOL,
        "data_freshness": {
            "daily_bars_through": (
                ticker_bars[-1].date.isoformat() if ticker_bars else None
            ),
            "daily_bars": len(ticker_bars),
            "bars_source": provider_name,
            "events_available": len(entries),
        },
        "unavailable": unavailable,
        "max_last": MAX_HISTORY_BACKFILL,
    }


__all__ = [
    "BACKFILL_ATTEMPT_SECONDS",
    "DEFAULT_HISTORY_BACKFILL",
    "ENTITY_TYPE",
    "MAX_HISTORY_BACKFILL",
    "NOT_BACKTESTABLE",
    "NO_STORED_BARS_REASON",
    "backfill_history_windows",
    "build_event_history",
    "build_event_replay_payload",
    "ensure_event_window_bars",
    "event_window",
    "pre_event_close_for",
    "to_minute_bars",
]
