"""Options / implied-move context — the gateway seam (Phase I, U3).

THE SPLIT THIS MODULE EXISTS TO KEEP, exactly like ``event_price.py`` and
``event_replay.py``. Every number in these payloads is computed by
``libs/trading_core/events/implied_move.py``, which is pure stdlib and may not
import ``apps/`` or ``libs.market_data`` (audit §7.4). This module is the only
place the two halves meet for the option view: it selects contracts through the
provider, stores their daily bars, reads them back, hands the closes to the
library and renders the frozen results. It computes nothing itself — no
arithmetic on a premium lives here.

TWO BASES, NEVER BLENDED (§37). An implied move read off a LIVE chain is a real
bid/ask midpoint at a known instant; one reconstructed from daily option CLOSES
is a settlement print standing in for a mark nobody observed. They are different
claims with different confidence, so every payload and every stored row carries
its ``basis`` — ``LIVE_CHAIN_SNAPSHOT`` or
``HISTORICAL_DAILY_CLOSE_APPROXIMATION`` — and the storage key is
``UNIQUE(event_id, basis)`` so one can never overwrite the other. The live path
is NOT stored as history: a snapshot taken while the print is still upcoming is
not a reconstruction of what the market charged, and filing it as one would make
today's guess indistinguishable from tomorrow's measurement.

WRITES ARE USER-TRIGGERED, READS NEVER FETCH BARS. Same rule ``event_replay``
draws, and for the same reason scaled up: one event's straddle costs a contract
listing plus two paginated bar fetches, and the history table shows up to eight
prior events — a GET that backfilled them all would issue seventeen provider
calls because somebody opened a page. So:

- ``GET  /options`` reads STORED option bars and STORED metrics. An event with
  none says so, with a reason.
- ``POST /options/backfill`` (one event) and ``POST /options/history/backfill``
  (bounded, ``last<=12``) are USER actions that fetch and store.

A FAILED FETCH IS ALSO A FACT, AND IT IS STORED. The history walk used to
report a per-event reason in its response and write nothing, so a run the
vendor rate-limited left a history table several rows short with no record of
why — indistinguishable from a history nobody had backfilled. Every degraded
outcome now upserts a ``status = NO_DATA`` row whose ``notes.reason`` names the
cause (every price column still NULL — this records an absence, it never
prices one). That makes the button REPEATABLE: NO_DATA rows are always
retried, OK/PARTIAL rows are skipped unless ``force=true``, so a second press
after a rate-limited run costs provider calls only for the events still
missing. The walk also PACES itself between events
(:data:`HISTORY_BACKFILL_PACING_SECONDS`) rather than firing ~32 requests in a
burst, which is what earned the 429s in the first place.

The ONE exception is the LIVE chain for an UPCOMING event, which the GET does
call. A chain snapshot is a single request for a symbol the platform already
tracks, it cannot be stored as history (see above), and an upcoming print with
no implied move is precisely the case the tab exists to answer. That call is
isolated: any provider failure becomes a ``NO_DATA`` block with a reason, never
a 5xx and never a number.

NEVER FABRICATE A PREMIUM (house rule, §44 rule 18). If the put leg has no bar
on the pre-event session there is NO straddle — status ``NO_DATA`` with the
reason — because a missing leg treated as zero would halve the implied move and
the error would look like a cheap option rather than like missing data. The same
rule governs IV: it is solved from a real close or it is ``None``.

POINT-IN-TIME IS A PROPERTY OF THE INPUTS, NOT OF A FILTER APPLIED LATER
(§14, §96). The PRE-event straddle is built from the last option bar dated
STRICTLY BEFORE the reaction — resolved by the same
:func:`~libs.trading_core.events.reaction.first_reaction_index` rule the equity
reaction uses, so the option and the stock never disagree about which session
was "before the print". Contracts are listed with ``as_of`` = that pre-event
date, so a strike listed in REACTION to the event cannot enter the straddle.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.market_data import (
    Bar,
    CapabilityNotAvailable,
    MarketDataError,
    OptionContractRef,
    ProviderNotConfigured,
    get_provider,
)
from libs.trading_core.events.implied_move import (
    BASIS_HISTORICAL,
    BASIS_LIVE,
    DISCLAIMER,
    STATUS_NO_DATA,
    STATUS_OK,
    STATUS_PARTIAL,
    ImpliedMoveSummary,
    build_summary,
    historical_move_stats,
    implied_vs_realized,
    nearest_strike,
    select_event_expiry,
    straddle_implied_move,
)
from libs.trading_core.events.reaction import first_reaction_index
from libs.trading_core.models import ActorType, AuditAction
from . import audit
from .db import EventOptionMetricRow, EventRow, OptionDailyBarRow, StockBarDaily
from .event_price import (
    _as_utc,
    _past_comparable_rows,
    _session_of,
    event_date_et,
    to_daily_bars,
)

logger = logging.getLogger(__name__)

#: The audit ``entity_type`` for an option-bar backfill — the table it wrote,
#: matching the ``stock_bars_1m`` / ``fundamental_statements`` precedent.
ENTITY_TYPE = "option_daily_bars"

#: How many EXPIRY CANDIDATES the contract probe tries before giving up. The
#: §18 rule wants the first listed expiry that still spans the event, and the
#: only way to learn which dates are listed is to ask per date — so the probe
#: walks the next few weekly Fridays. Three covers a standard weekly ladder
#: plus one holiday-shifted week; a wider walk would be three more paginated
#: reference calls per event for an expiry six weeks out that no earnings
#: straddle would use anyway.
EXPIRY_PROBE_WEEKS = 3

#: How many days of option bars the backfill fetches around the event. The
#: window must cover the last session BEFORE the print (the pre-event mark)
#: and the first session AFTER it (the post-event mark, which is what makes
#: the IV crush measurable), with slack for a long weekend on either side.
#: Deliberately small: this is two paginated fetches per event, and a wider
#: window buys nothing the straddle reads.
BARS_BEFORE_DAYS = 5
BARS_AFTER_DAYS = 3

#: How many previous events ``build_event_options_payload`` renders, and how
#: many ``POST /options/history/backfill`` fetches by default and at most.
#: Eight matches the §60 history table's middle toggle; the ceiling of twelve
#: matches its largest, so no request can ask for an unbounded number of
#: contract probes.
DEFAULT_HISTORY = 8
DEFAULT_HISTORY_BACKFILL = 4
MAX_HISTORY_BACKFILL = 12

#: Seconds to wait BETWEEN events in the history backfill (never before the
#: first — a single event pays no toll). Each event costs one contract probe
#: plus two paginated bar fetches, so eight events fire ~32 requests; sent
#: back-to-back that burst earned Massive's HTTP 429 and the tail of the run
#: came back empty. The adapter's bounded backoff (see
#: ``libs/market_data/massive.py``) is the recovery; this is the politeness
#: that keeps the recovery from being needed. A constant, not a literal, so a
#: deployment on a wider plan can shorten it in ONE place (§6.2).
HISTORY_BACKFILL_PACING_SECONDS = 1.5

#: Reason a GET gives when an event has no stored option metrics. Stated once
#: so the payload and the history rows say the SAME thing, and so the UI can
#: match on it to decide whether to offer the backfill button.
NO_STORED_METRICS_REASON = (
    "no option metrics stored for this event — use POST "
    "/api/events/{id}/options/backfill to fetch them"
)

#: The ``notes`` key under which a NO_DATA row records WHY the backfill came
#: back empty. One key, so the UI and a later re-run read the same field
#: rather than each guessing at the shape of the note.
REASON_NOTE_KEY = "reason"

#: §85 / audit §7.3 — what the OPTION view deliberately does not claim to
#: reconstruct point-in-time. The historical basis is built from daily
#: SETTLEMENT closes: the intraday mark at the moment of the release, the
#: bid/ask spread that mark sat inside, and the vendor's own greeks are all
#: unavailable for a past date on any plan this platform has.
NOT_BACKTESTABLE: tuple[str, ...] = (
    "intraday_option_marks",
    "historical_option_bid_ask",
    "historical_provider_greeks",
    "historical_open_interest",
)


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def option_history_provider_name(settings) -> str:
    """Which provider serves HISTORICAL option bars and dated contracts.

    Massive is the only vendor with ``/v3/reference/options/contracts?as_of=``
    and dated ``/v2/aggs`` option bars (probed live, contract header), and the
    platform's market-data provider is Alpaca (DEVLOG 2026-08-13) — so the §36
    reconstruction must come from Massive whenever its key is configured,
    regardless of ``market_data_provider``. This is exactly the
    :func:`apps.gateway.fundamentals.fundamentals_provider_name` pattern and
    delegates to it rather than restating the rule, so the two capabilities
    can never drift apart about which vendor is configured.

    The LIVE chain is a different question and keeps using
    ``settings.market_data_provider``: a current snapshot with greeks is what
    Alpaca DOES sell, and routing it to Massive would swap a real bid/ask for
    a reconstruction the live path is defined not to use.
    """
    from .fundamentals import fundamentals_provider_name

    return fundamentals_provider_name(settings)


# ---------------------------------------------------------------------------
# Stored option bars
# ---------------------------------------------------------------------------


async def _stored_bars(
    session: AsyncSession, tickers: list[str]
) -> dict[str, dict[date, OptionDailyBarRow]]:
    """``{option_ticker: {bar_date: row}}`` for the given contracts.

    One query for every leg the caller needs, keyed by date so the straddle
    can ask "was there a bar on THIS session" without scanning. An absent key
    is the honest "that contract did not trade that day" — the caller turns it
    into a named reason, never into a zero.
    """
    if not tickers:
        return {}
    rows = (
        (
            await session.execute(
                select(OptionDailyBarRow).where(
                    OptionDailyBarRow.option_ticker.in_(tickers)
                )
            )
        )
        .scalars()
        .all()
    )
    out: dict[str, dict[date, OptionDailyBarRow]] = {t: {} for t in tickers}
    for row in rows:
        out.setdefault(row.option_ticker, {})[row.bar_date] = row
    return out


async def _store_bars(
    session: AsyncSession,
    option_ticker: str,
    bars: list[Bar],
    *,
    provider_name: str,
) -> int:
    """Upsert one contract's daily bars. Returns how many rows were WRITTEN.

    The key is ``(option_ticker, bar_date)`` — the table's own composite
    primary key — so a refetch overwrites a session and can never duplicate
    one. Existing rows are UPDATED rather than skipped: a provider that
    revises a settlement close after the fact is correcting a fact, and
    keeping the stale copy would silently freeze the first fetch's answer.

    Bars whose close is not a positive finite number are DROPPED, not stored:
    a zero settlement is not a premium, and one stored zero would produce a
    straddle half its true size that nothing downstream could distinguish
    from a cheap option.
    """
    written = 0
    for bar in bars:
        close = float(bar.close)
        if not (close > 0.0) or close != close:  # NaN-safe positivity
            continue
        existing = await session.get(OptionDailyBarRow, (option_ticker, bar.ts))
        if existing is None:
            session.add(
                OptionDailyBarRow(
                    option_ticker=option_ticker,
                    bar_date=bar.ts,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=close,
                    volume=int(bar.volume) if bar.volume is not None else None,
                    provider=provider_name[:16],
                )
            )
        else:
            existing.open = float(bar.open)
            existing.high = float(bar.high)
            existing.low = float(bar.low)
            existing.close = close
            existing.volume = int(bar.volume) if bar.volume is not None else None
            existing.provider = provider_name[:16]
        written += 1
    return written


# ---------------------------------------------------------------------------
# The event's two anchor sessions
# ---------------------------------------------------------------------------


async def _equity_bars(session: AsyncSession, ticker: str) -> list:
    """Stored daily equity bars for ``ticker``, oldest first — NEVER fetched.

    Deliberately NOT ``ensure_daily_bars``: that path lazily backfills on a
    read, and the option seam must not turn "show me the implied move" into
    an equity backfill for a symbol the user has not asked to track. If no
    bars are stored the caller reports that absence by name; the price-context
    tab is where equity bars get fetched.
    """
    rows = (
        (
            await session.execute(
                select(StockBarDaily)
                .where(StockBarDaily.ticker == ticker)
                .order_by(StockBarDaily.ts)
            )
        )
        .scalars()
        .all()
    )
    return to_daily_bars(list(rows))


def anchor_sessions(bars: list, event_row: EventRow) -> tuple[dict, str | None]:
    """``({pre_date, pre_close, post_date, post_close, actual_move_pct}, reason)``.

    THE OPTION ANCHOR IS THE EQUITY ANCHOR, resolved by
    :func:`~libs.trading_core.events.reaction.first_reaction_index` — the same
    function ``event_reaction`` uses. That shared rule is the point: the
    pre-event option mark and the pre-event stock close must come from ONE
    session, or the straddle's ``pct`` would divide a premium from Tuesday by
    a spot from Wednesday and report a move that no chain ever priced.

    Per §17, "before" means different sessions for different releases: an AMC
    print's pre session is the release day itself (its 16:00 close is the last
    price before the news), a BMO print's is the day before. Getting that
    backwards puts the reaction INSIDE the pre-event straddle, which reads as
    an implied move that already knew the answer.

    ``actual_move_pct`` is the realized move over that same pair
    (``post_close / pre_close - 1``), so the §66 ratio compares two numbers
    measured across one interval. ``None`` with a reason when either leg is
    missing — an event too recent to have a reaction bar, or older than the
    stored history.
    """
    if not bars:
        return {}, "no stored daily bars for the underlying"
    day = event_date_et(event_row)
    kind = _session_of(event_row)
    located = first_reaction_index(bars, day, kind)
    if located is None:
        return {}, (
            f"no reaction bar for {day.isoformat()} — the event is more recent "
            f"than the stored bars (they end {bars[-1].date.isoformat()})"
        )
    pre_idx, react_idx, basis = located
    if react_idx >= len(bars):
        return {}, (
            f"no reaction bar after {bars[pre_idx].date.isoformat()} — the "
            "event has not had a trading session since"
        )
    pre = bars[pre_idx]
    post = bars[react_idx]
    pre_close = float(pre.close)
    if not pre_close > 0.0:
        return {}, "pre_event_close_not_positive"
    return (
        {
            "pre_date": pre.date,
            "pre_close": pre_close,
            "post_date": post.date,
            "post_close": float(post.close),
            "actual_move_pct": float(post.close) / pre_close - 1.0,
            "basis": basis,
        },
        None,
    )


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------


def _expiry_candidates(event_day: date, *, weeks: int = EXPIRY_PROBE_WEEKS) -> list[date]:
    """The next ``weeks`` weekly-Friday expiry dates on or after ``event_day``.

    A PROBE LIST, not an assertion that these dates are listed. The reference
    endpoint answers "which contracts expire on THIS date", so learning which
    expiries exist means asking per date, and US weekly options expire on
    Fridays. The caller stops at the first date that returns contracts, so a
    Friday that is a market holiday simply yields nothing and the next one is
    tried — no holiday calendar is consulted, because the provider's own empty
    answer IS the calendar.

    The event day itself is included when it is a Friday: whether a same-day
    expiry may price the event is §18's question, and
    :func:`select_event_expiry` decides it from the session. Excluding it here
    would pre-empt that rule for BEFORE_MARKET releases, whose same-day
    contract genuinely does price the print.
    """
    days_to_friday = (4 - event_day.weekday()) % 7
    first = event_day + timedelta(days=days_to_friday)
    return [first + timedelta(days=7 * i) for i in range(max(1, weeks))]


def _straddle_legs(
    refs: list[OptionContractRef], strike: float
) -> tuple[OptionContractRef | None, OptionContractRef | None]:
    """The call and put at ``strike`` from one expiry's contract list.

    Matched on the strike VALUE with a tolerance, not on identity: a provider
    that serves ``210.0`` in one row and ``210.00000000000003`` in another
    would otherwise drop a leg and turn a complete straddle into NO_DATA.
    ``None`` for a leg the expiry does not list — the caller names which one
    is missing rather than substituting the other.
    """
    call = next(
        (r for r in refs if r.right == "C" and abs(r.strike - strike) < 1e-6), None
    )
    put = next(
        (r for r in refs if r.right == "P" and abs(r.strike - strike) < 1e-6), None
    )
    return call, put


def _select_contracts(
    provider,
    ticker: str,
    *,
    event_row: EventRow,
    as_of_date: date,
    spot: float,
) -> tuple[dict, str | None]:
    """``({expiry, strike, call, put}, reason)`` — the ATM straddle to price.

    Three library decisions, none of them made here: which expiry spans the
    event (:func:`select_event_expiry`, applying §18's same-day rule from the
    session), which strike is at the money (:func:`nearest_strike`) and which
    two contracts sit on it (:func:`_straddle_legs`).

    ``as_of`` IS THE PRE-EVENT DATE, and that is the point-in-time guarantee
    (§96): the provider answers with the contracts that existed BEFORE the
    print, so a strike listed in reaction to the news — which today's universe
    would happily return — cannot enter the straddle. Every probe uses the
    same ``as_of``.

    Raises nothing: a provider without dated contracts, a symbol it does not
    know and an expiry with no listings all return ``(…, reason)``. The
    ``CapabilityNotAvailable`` case is left to the CALLER to catch, because
    "this vendor cannot do it at all" and "this expiry was empty" deserve
    different reasons.
    """
    for candidate in _expiry_candidates(event_date_et(event_row)):
        refs = provider.list_option_contracts(
            ticker, expiration_date=candidate, as_of=as_of_date
        )
        if not refs:
            continue
        chosen = select_event_expiry(
            event_date_et(event_row), [candidate], session=_session_of(event_row).value
        )
        if chosen is None:
            # §18: this expiry cannot price the event (a same-day contract for
            # an AMC print). Keep probing — the NEXT Friday can.
            continue
        strike = nearest_strike(spot, [r.strike for r in refs])
        if strike is None:
            continue
        call, put = _straddle_legs(refs, strike)
        if call is None or put is None:
            missing = "call" if call is None else "put"
            return {}, (
                f"expiry {candidate.isoformat()} lists no {missing} at strike "
                f"{strike} — half a straddle is not an implied move"
            )
        return (
            {"expiry": candidate, "strike": strike, "call": call, "put": put},
            None,
        )
    return {}, (
        f"no listed option contracts for {ticker} in the "
        f"{EXPIRY_PROBE_WEEKS} weekly expiries after "
        f"{event_date_et(event_row).isoformat()} as of {as_of_date.isoformat()}"
    )


# ---------------------------------------------------------------------------
# Persistence of the computed metrics
# ---------------------------------------------------------------------------


def _summary_row_values(summary: ImpliedMoveSummary) -> dict:
    """The ORM column values for one summary. The ONLY flattening point.

    ``pre``/``post`` are :class:`ImpliedMove` objects in the library and
    columns in the table, so exactly one function performs that translation —
    otherwise the upsert and the read would each have their own idea of which
    ``pct`` is the stored one.
    """
    pre = summary.pre
    post = summary.post
    return {
        "expiry": summary.expiry,
        "strike": summary.strike,
        "spot": pre.spot if pre is not None else None,
        "pre_call_close": (pre.inputs.get("call_px") if pre is not None else None),
        "pre_put_close": (pre.inputs.get("put_px") if pre is not None else None),
        "post_call_close": (post.inputs.get("call_px") if post is not None else None),
        "post_put_close": (post.inputs.get("put_px") if post is not None else None),
        "implied_move_pct": pre.pct if pre is not None else None,
        "implied_move_points": pre.points if pre is not None else None,
        "iv_before": summary.iv_before,
        "iv_after": summary.iv_after,
        "iv_crush_pct": summary.iv_crush_pct,
        "actual_move_pct": summary.actual_move_pct,
        "implied_realized_ratio": summary.ratio,
        "classification": summary.classification,
        "status": summary.status,
        "notes": dict(summary.notes),
    }


async def _upsert_metrics(
    session: AsyncSession,
    event_id: int,
    summary: ImpliedMoveSummary,
    *,
    as_of: datetime,
    call_ticker: str | None,
    put_ticker: str | None,
) -> EventOptionMetricRow:
    """Store (or refresh) the metrics for one ``(event, basis)`` pair.

    ``UNIQUE(event_id, basis)`` makes this idempotent at the DATABASE level,
    which is what ADR-007 relies on given there is no leader election: two
    concurrent presses of the backfill button collide on the constraint rather
    than writing two rival rows. A re-run UPDATES because a later backfill
    sees strictly more data — a post-event bar that had not settled the first
    time — and keeping the older, thinner answer would freeze the event at its
    first look.
    """
    existing = (
        await session.execute(
            select(EventOptionMetricRow).where(
                EventOptionMetricRow.event_id == event_id,
                EventOptionMetricRow.basis == summary.basis,
            )
        )
    ).scalar_one_or_none()
    values = _summary_row_values(summary)
    if existing is None:
        row = EventOptionMetricRow(
            event_id=event_id,
            as_of=as_of,
            basis=summary.basis,
            call_ticker=call_ticker,
            put_ticker=put_ticker,
            **values,
        )
        session.add(row)
        return row
    existing.as_of = as_of
    existing.call_ticker = call_ticker
    existing.put_ticker = put_ticker
    for key, value in values.items():
        setattr(existing, key, value)
    return existing


async def _record_no_data(
    session: AsyncSession,
    event_row: EventRow,
    reason: str,
    *,
    as_of: datetime,
) -> None:
    """Persist the HONEST GAP: a NO_DATA metrics row naming why nothing came.

    Before this existed, an event whose provider call was rate-limited or
    whose contracts the vendor would not serve wrote NOTHING — the run
    reported a reason in its response body and the database kept no trace, so
    a history table five rows short looked identical to a history table the
    user had never backfilled. That is the failure mode §44 rule 18 exists to
    prevent turned inside out: not a fabricated number, but a silent absence.

    The row carries ``status = NO_DATA``, every price column NULL, and
    ``notes = {"reason": ...}``. It is deliberately a REAL row so that:

    - the coverage block can say "attempted, came back empty" rather than
      "never attempted";
    - a later re-run finds it and RETRIES it (see ``_should_attempt``), which
      is exactly the recovery a transient 429 needs;
    - nothing downstream reads it as a price — ``build_event_options_payload``
      already suppresses a NO_DATA row's numbers, and there are none to
      suppress.

    Upserted through :func:`_upsert_metrics` like every other outcome, so
    ``UNIQUE(event_id, basis)`` stays the single idempotency mechanism.
    """
    await _upsert_metrics(
        session,
        event_row.id,
        ImpliedMoveSummary(
            basis=BASIS_HISTORICAL,
            status=STATUS_NO_DATA,
            event_date=event_date_et(event_row),
            session=_session_of(event_row).value,
            notes={REASON_NOTE_KEY: reason},
        ),
        as_of=as_of,
        call_ticker=None,
        put_ticker=None,
    )
    await session.commit()


async def _existing_metric(
    session: AsyncSession, event_id: int
) -> EventOptionMetricRow | None:
    """The stored HISTORICAL row for one event, or None."""
    return (
        await session.execute(
            select(EventOptionMetricRow).where(
                EventOptionMetricRow.event_id == event_id,
                EventOptionMetricRow.basis == BASIS_HISTORICAL,
            )
        )
    ).scalar_one_or_none()


# ---------------------------------------------------------------------------
# The backfill (USER-triggered writes)
# ---------------------------------------------------------------------------


async def backfill_event_options(
    session: AsyncSession,
    event_row: EventRow,
    *,
    provider: object | None = None,
    provider_name: str = "",
    now: datetime | None = None,
    force: bool = False,
    skip_if_ok: bool = False,
) -> dict:
    """Fetch and store ONE event's option straddle and metrics. Honest report.

    The option counterpart of ``ensure_event_window_bars`` and
    ``ensure_event_news_window``, and deliberately the ONLY path that writes
    ``option_daily_bars`` or ``event_option_metrics`` — so the DATA_BACKFILL
    audit trail and the provenance story stay single-sourced (rule 12,
    ADR-003).

    The sequence, each step failing with its own reason rather than a shared
    "unavailable":

    1. the event's ticker (a macro release has none — a CPI print has no
       issuer whose options these would be);
    2. the pre/post anchor sessions from STORED equity bars
       (:func:`anchor_sessions`), which also yields the realized move;
    3. the ATM straddle, listed AS OF the pre-event date so a strike created
       in reaction to the print cannot enter it (:func:`_select_contracts`);
    4. daily bars for both legs over a window spanning both anchors, STORED;
    5. the closes on the two anchor sessions handed to
       :func:`~libs.trading_core.events.implied_move.build_summary`, whose
       verdict is upserted under ``HISTORICAL_DAILY_CLOSE_APPROXIMATION``.

    A MISSING LEG IS NEVER ZERO. If either contract has no bar on an anchor
    session the library returns ``NO_DATA`` with the reason and that is what is
    stored — a straddle with one leg is not an implied move, and half a number
    is worse than none because it looks like a cheap option.

    200 IN EVERY DEGRADED CASE. An unconfigured provider, Alpaca's honest
    ``CapabilityNotAvailable``, a symbol with no listed options and an event
    the equity history does not reach all report ``fetched: false`` with a
    reason. A button press must say why nothing arrived, not 5xx.

    EVERY DEGRADED CASE IS ALSO WRITTEN DOWN. Each of those outcomes now
    upserts a ``status = NO_DATA`` metrics row whose ``notes.reason`` is the
    same string the response carries (:func:`_record_no_data`). A run that
    quietly stored nothing was indistinguishable from a run that never
    happened, which is how a rate-limited history backfill lost six of eight
    events without saying so. The one exception is a run that never had a
    subject: an event with no ticker gets no row, because a macro print has no
    straddle to be missing.

    ``skip_if_ok`` (used by the history walk) returns the STORED answer
    untouched when the event already has an OK/PARTIAL row, so a re-run costs
    provider calls only for the events that actually failed. ``force``
    overrides that and re-fetches regardless. A NO_DATA row is ALWAYS retried
    — it is a record of a gap, not of an answer.
    """
    stamp = now or datetime.now(timezone.utc)
    ticker = (event_row.ticker or "").strip().upper()
    base: dict = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": ticker or None,
        "basis": BASIS_HISTORICAL,
    }
    empty = {**base, "fetched": False, "stored_bars": 0, "status": STATUS_NO_DATA}

    async def _fail(reason: str) -> dict:
        """Report the reason AND write it down. Same string in both places."""
        await _record_no_data(session, event_row, reason, as_of=stamp)
        return {**empty, "reason": reason}

    if not ticker:
        # No row: a macro print has no issuer whose straddle could be missing,
        # so a NO_DATA option row would assert a gap that does not exist.
        return {**empty, "reason": "no_ticker"}

    if skip_if_ok and not force:
        existing = await _existing_metric(session, event_row.id)
        if existing is not None and existing.status != STATUS_NO_DATA:
            return {
                **base,
                "fetched": False,
                "stored_bars": 0,
                "status": existing.status,
                "skipped": True,
                "reason": "already stored — pass force=true to re-fetch",
                "expiry": existing.expiry.isoformat() if existing.expiry else None,
                "strike": existing.strike,
                "call_ticker": existing.call_ticker,
                "put_ticker": existing.put_ticker,
                "implied_move_pct": existing.implied_move_pct,
            }

    if provider is None:
        try:
            provider = get_provider(provider_name)
        except (ProviderNotConfigured, ValueError) as exc:
            return await _fail(str(exc) or "no provider configured")

    bars = await _equity_bars(session, ticker)
    anchors, reason = anchor_sessions(bars, event_row)
    if reason is not None:
        return await _fail(reason)

    try:
        picked, reason = _select_contracts(
            provider,
            ticker,
            event_row=event_row,
            as_of_date=anchors["pre_date"],
            spot=anchors["pre_close"],
        )
    except CapabilityNotAvailable as exc:
        # Alpaca sells option SNAPSHOTS but no dated contract reference. That
        # is a capability statement about the vendor, not an absence of
        # contracts, and it must not read as "this symbol has no options".
        return await _fail(f"capability_not_available: {exc}")
    except MarketDataError as exc:
        # Includes the adapter's exhausted-rate-limit error, which is the
        # single most common way this event comes back empty in a burst.
        return await _fail(f"provider_error: {exc}")
    if reason is not None:
        return await _fail(reason)

    call_ref: OptionContractRef = picked["call"]
    put_ref: OptionContractRef = picked["put"]
    start = anchors["pre_date"] - timedelta(days=BARS_BEFORE_DAYS)
    end = anchors["post_date"] + timedelta(days=BARS_AFTER_DAYS)

    stored = 0
    for ref in (call_ref, put_ref):
        try:
            leg_bars = provider.get_option_history_bars(ref.ticker, start, end)
        except CapabilityNotAvailable as exc:
            return await _fail(f"capability_not_available: {exc}")
        except MarketDataError as exc:
            return await _fail(f"provider_error: {exc}")
        stored += await _store_bars(
            session,
            ref.ticker,
            leg_bars,
            # ``provider_name`` is the AUTHORITATIVE label: the provider
            # objects carry no ``name`` attribute, and a stored premium whose
            # source is unknown cannot be audited. An injected provider with
            # no name declared is recorded as "unknown" rather than being
            # attributed to whatever the settings happen to say.
            provider_name=provider_name or "unknown",
        )

    # THE CLOSES ARE READ BACK FROM THE STORE, not from the list just fetched.
    # A second backfill of the same event can legitimately receive `[]` for a
    # leg — the vendor 404s a contract that has since delisted — and pricing
    # the straddle off the fetch alone would then downgrade a complete stored
    # answer to NO_DATA. The store is the accumulated truth; the fetch only
    # tops it up.
    await session.flush()
    stored_legs = await _stored_bars(session, [call_ref.ticker, put_ref.ticker])

    def _close(ref: OptionContractRef, day: date) -> float | None:
        bar = stored_legs.get(ref.ticker, {}).get(day)
        return float(bar.close) if bar is not None else None

    expiry: date = picked["expiry"]
    dte_days = float((expiry - anchors["pre_date"]).days)
    summary = build_summary(
        _close(call_ref, anchors["pre_date"]),
        _close(put_ref, anchors["pre_date"]),
        anchors["pre_close"],
        _close(call_ref, anchors["post_date"]),
        _close(put_ref, anchors["post_date"]),
        anchors["post_close"],
        strike=picked["strike"],
        expiry=expiry,
        event_date=event_date_et(event_row),
        session=_session_of(event_row).value,
        actual_move_pct=anchors["actual_move_pct"],
        dte_days=dte_days,
    )

    await _upsert_metrics(
        session,
        event_row.id,
        summary,
        as_of=stamp,
        call_ticker=call_ref.ticker,
        put_ticker=put_ref.ticker,
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.DATA_BACKFILL,
        entity_type=ENTITY_TYPE,
        entity_id=ticker,
        details={
            "kind": "event_option_straddle",
            "ticker": ticker,
            "event_key": event_row.event_key,
            "event_id": event_row.id,
            "basis": BASIS_HISTORICAL,
            "expiry": expiry.isoformat(),
            "strike": picked["strike"],
            "call_ticker": call_ref.ticker,
            "put_ticker": put_ref.ticker,
            "bars": stored,
            "status": summary.status,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
        },
    )
    await session.commit()

    result = {
        **base,
        "fetched": True,
        "stored_bars": stored,
        "status": summary.status,
        "expiry": expiry.isoformat(),
        "strike": picked["strike"],
        "call_ticker": call_ref.ticker,
        "put_ticker": put_ref.ticker,
        "implied_move_pct": summary.pre.pct if summary.pre is not None else None,
    }
    if summary.status == STATUS_NO_DATA:
        # The provider answered but the straddle did not price — most often
        # because a leg has no bar on an anchor session (an untraded strike,
        # or a fetch that came back empty). The row is already stored with the
        # library's own note map; the response says the same thing.
        result["reason"] = (
            summary.notes.get("pre")
            or summary.notes.get(REASON_NOTE_KEY)
            or "no implied move could be computed"
        )
    return result


async def backfill_options_history(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider: object | None = None,
    provider_name: str = "",
    last: int = DEFAULT_HISTORY_BACKFILL,
    now: datetime | None = None,
    force: bool = False,
) -> dict:
    """Backfill the straddles of the last N PREVIOUS comparable events.

    BOUNDED TWICE, like the minute-bar history backfill: the router's ``le=``
    rejects an out-of-range ``last`` at the boundary, and this clamps again —
    the ceiling is a property of the operation (twelve contract probes plus
    twenty-four paginated bar fetches is already a lot), not of one caller's
    query string, so it holds for internal callers too.

    Each event is attempted INDEPENDENTLY and every outcome appears in
    ``events`` and in the ``counts`` tally: one print whose contracts the
    vendor will not serve does not cost the other three theirs, which is the
    same per-item isolation the calendar ingest applies across providers (§8).
    The provider is instantiated ONCE and passed down, so N events cost one
    client rather than N.

    PACED, because the burst is what broke it. Eight events back-to-back is
    about thirty-two requests with no gap, and Massive answered the tail of
    that burst with HTTP 429 until the run gave up — six of eight events came
    back with no bars and, before this change, no record either. The walk now
    sleeps :data:`HISTORY_BACKFILL_PACING_SECONDS` BETWEEN events (never
    before the first: a one-event run should not pay a toll for a burst it is
    not creating), which together with the adapter's bounded 429 backoff turns
    a silent partial run into a complete one.

    EVERY OUTCOME IS WRITTEN DOWN. A failed event upserts a NO_DATA row naming
    the reason, so the stored history shows the honest gap and a LATER RE-RUN
    RETRIES IT: NO_DATA rows are always re-attempted, OK/PARTIAL rows are
    skipped unless ``force`` is set. That makes the button safely repeatable —
    press it again after a rate-limited run and only the missing events cost
    provider calls.
    """
    stamp = now or datetime.now(timezone.utc)
    bounded = max(1, min(int(last), MAX_HISTORY_BACKFILL))
    base = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": (event_row.ticker or None),
        "as_of": as_of.isoformat(),
        "requested": bounded,
        "force": bool(force),
    }
    empty_counts = {"ok": 0, "no_data": 0, "failed": 0, "skipped": 0}
    if provider is None:
        try:
            provider = get_provider(provider_name)
        except (ProviderNotConfigured, ValueError) as exc:
            return {
                **base,
                "events": [],
                "event_count": 0,
                "counts": dict(empty_counts),
                "stored_bars": 0,
                "status": STATUS_NO_DATA,
                "reason": str(exc) or "no provider configured",
                "results": [],
            }

    rows = await _past_comparable_rows(session, event_row, as_of)
    selected = rows[-bounded:] if bounded < len(rows) else rows
    results: list[dict] = []
    outcomes: list[dict] = []
    counts = dict(empty_counts)
    total = 0
    for index, row in enumerate(reversed(selected)):  # newest first, as the UI lists
        if index and HISTORY_BACKFILL_PACING_SECONDS > 0:
            # BETWEEN events only — the first one starts immediately.
            await asyncio.sleep(HISTORY_BACKFILL_PACING_SECONDS)
        outcome = await backfill_event_options(
            session,
            row,
            provider=provider,
            provider_name=provider_name,
            now=stamp,
            force=force,
            skip_if_ok=True,
        )
        stored_bars = int(outcome.get("stored_bars") or 0)
        total += stored_bars
        results.append(outcome)
        status = outcome.get("status") or STATUS_NO_DATA
        if outcome.get("skipped"):
            counts["skipped"] += 1
        elif status == STATUS_NO_DATA:
            # "failed" is the subset the PROVIDER refused (or never answered);
            # "no_data" is a provider that answered with nothing usable. Both
            # are honest gaps; distinguishing them tells the operator whether
            # to press again or to stop pressing.
            reason = str(outcome.get("reason") or "")
            if reason.startswith(("provider_error:", "capability_not_available:")):
                counts["failed"] += 1
            else:
                counts["no_data"] += 1
        else:
            counts["ok"] += 1
        outcomes.append(
            {
                "event_id": outcome.get("event_id"),
                "event_key": outcome.get("event_key"),
                "status": status,
                "reason": outcome.get("reason"),
                "stored_bars": stored_bars,
                "skipped": bool(outcome.get("skipped")),
            }
        )
    return {
        **base,
        # ``events`` is the PER-EVENT LIST (event_id, event_key, status,
        # reason, stored_bars) — a caller that only got a count could not tell
        # which prints are still missing. ``event_count`` keeps the old
        # cardinality available under its own name.
        "events": outcomes,
        "event_count": len(results),
        "counts": counts,
        "stored_bars": total,
        "status": STATUS_OK if results else STATUS_NO_DATA,
        "reason": None if results else "no previous comparable events for this one",
        # The full per-event bodies, unchanged, for callers that want the
        # strike/expiry/implied move of each.
        "results": results,
    }


# ---------------------------------------------------------------------------
# The LIVE chain (upcoming events)
# ---------------------------------------------------------------------------


def live_implied_move(
    event_row: EventRow,
    *,
    provider: object | None = None,
    provider_name: str = "",
    spot: float | None = None,
    as_of_date: date | None = None,
) -> ImpliedMoveSummary:
    """The implied move priced by the CURRENT chain, for an upcoming event.

    THE ONLY PROVIDER CALL A GET MAKES, and the exception is deliberate: a
    chain snapshot is one request for a symbol the platform already tracks, it
    cannot be stored as history (a snapshot taken before the print is not a
    reconstruction of what the market charged), and an upcoming print with no
    implied move is precisely the question this tab exists to answer.

    Builds its :class:`ImpliedMoveSummary` DIRECTLY with :data:`BASIS_LIVE`
    rather than through ``build_summary``, which always stamps the historical
    basis because it reconstructs from closes. Only the PRE side exists here —
    there is no post-event mark for an event that has not happened — so the
    status is :data:`STATUS_PARTIAL` on success, never OK: the crush, the
    realized move and the ratio are all genuinely unknown, and OK would claim
    a completeness the payload does not have.

    ``mid`` is the mark, matching the selector's own convention: a live
    straddle is what you would pay to cross the spread, and ``last`` can be
    hours stale on an illiquid contract.

    NEVER RAISES. An unconfigured provider, a chain call that fails, an empty
    chain, no expiry spanning the event and a missing leg each produce a
    ``NO_DATA`` summary whose ``notes`` name the cause. A quote vendor being
    down must not take the Catalyst page with it.
    """
    def _no_data(reason_key: str, reason: str) -> ImpliedMoveSummary:
        return ImpliedMoveSummary(
            basis=BASIS_LIVE,
            status=STATUS_NO_DATA,
            event_date=event_date_et(event_row),
            session=_session_of(event_row).value,
            notes={reason_key: reason},
        )

    ticker = (event_row.ticker or "").strip().upper()
    if not ticker:
        return _no_data("ticker", "no_ticker")
    reference = float(spot) if spot is not None else None
    if reference is None or not reference > 0.0:
        return _no_data(
            "spot",
            "no positive spot for the underlying — a straddle percentage "
            "needs one",
        )

    if provider is None:
        try:
            provider = get_provider(provider_name)
        except (ProviderNotConfigured, ValueError) as exc:
            return _no_data("provider", str(exc) or "no provider configured")

    day = as_of_date or datetime.now(timezone.utc).date()
    try:
        chain = provider.get_option_chain(ticker, reference, day)
    except CapabilityNotAvailable as exc:
        return _no_data("chain", f"capability_not_available: {exc}")
    except (MarketDataError, ProviderNotConfigured, ValueError) as exc:
        return _no_data("chain", f"provider_error: {exc}")
    if not chain:
        return _no_data("chain", f"the provider returned an empty chain for {ticker}")

    event_day = event_date_et(event_row)
    expiry = select_event_expiry(
        event_day,
        {quote.expiry for quote in chain},
        session=_session_of(event_row).value,
    )
    if expiry is None:
        return _no_data(
            "expiry",
            f"no listed expiry spans {event_day.isoformat()} — every contract "
            "in the chain expires before the event",
        )
    at_expiry = [quote for quote in chain if quote.expiry == expiry]
    strike = nearest_strike(reference, [quote.strike for quote in at_expiry])
    if strike is None:
        return _no_data(
            "strike", f"expiry {expiry.isoformat()} lists no usable strike"
        )

    call = next(
        (q for q in at_expiry if q.right == "C" and abs(q.strike - strike) < 1e-6), None
    )
    put = next(
        (q for q in at_expiry if q.right == "P" and abs(q.strike - strike) < 1e-6), None
    )
    if call is None or put is None:
        missing = "call" if call is None else "put"
        return _no_data(
            "legs",
            f"expiry {expiry.isoformat()} lists no {missing} at strike {strike} "
            "— half a straddle is not an implied move",
        )

    move = straddle_implied_move(call.mid, put.mid, reference)
    notes: dict[str, str] = {}
    if move.reason is not None:
        notes["pre"] = move.reason
    # The vendor's own ATM implied volatility, averaged across the two legs
    # when both carry one. NEVER zero-filled: a chain row without greeks is a
    # real quote the selector still accepts (§9), and inventing an IV for it
    # would put a fabricated number beside a real premium.
    ivs = [q.iv for q in (call, put) if q.iv is not None and q.iv > 0.0]
    iv_before = sum(ivs) / len(ivs) if ivs else None
    if iv_before is None:
        notes["iv_before"] = "the provider reported no implied volatility for either leg"
    notes["post"] = (
        "the event has not happened yet — there is no post-event mark, so the "
        "IV crush and the realized comparison are genuinely unknown"
    )
    dte = float((expiry - day).days)
    return ImpliedMoveSummary(
        basis=BASIS_LIVE,
        status=STATUS_NO_DATA if move.pct is None else STATUS_PARTIAL,
        pre=move,
        post=None,
        iv_before=iv_before,
        strike=strike,
        expiry=expiry,
        event_date=event_day,
        session=_session_of(event_row).value,
        dte_days=dte if dte > 0.0 else None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rendering — stored rows and frozen results to JSON
# ---------------------------------------------------------------------------


def metric_row_to_dict(row: EventOptionMetricRow, event_row: EventRow | None) -> dict:
    """One stored metrics row as the payload's ``current``/``history`` shape.

    The field names are the API contract the UI reads, and they match
    :meth:`ImpliedMoveSummary.to_dict` where the two overlap so a live block
    and a stored block are interchangeable to the renderer. ``notes`` is a
    LIST here rather than the library's mapping: the UI shows the first one as
    the "why", and a list has an order a dict does not.

    Nothing is defaulted. A NULL column renders as ``null``, and the status
    beside it says whether that is an absence or a retraction.
    """
    return {
        "basis": row.basis,
        "status": row.status,
        "as_of": _as_utc(row.as_of).isoformat() if row.as_of else None,
        "expiry": row.expiry.isoformat() if row.expiry else None,
        "strike": row.strike,
        "spot": row.spot,
        "implied_move_pct": row.implied_move_pct,
        "implied_move_points": row.implied_move_points,
        "iv_before": row.iv_before,
        "iv_after": row.iv_after,
        "iv_crush_pct": row.iv_crush_pct,
        "actual_move_pct": row.actual_move_pct,
        "implied_realized_ratio": row.implied_realized_ratio,
        "classification": row.classification,
        "call_ticker": row.call_ticker,
        "put_ticker": row.put_ticker,
        "notes": [str(v) for v in (row.notes or {}).values()],
        "event_id": event_row.id if event_row is not None else row.event_id,
        "event_key": event_row.event_key if event_row is not None else None,
        "event_date": (
            event_date_et(event_row).isoformat() if event_row is not None else None
        ),
    }


def summary_to_block(summary: ImpliedMoveSummary, event_row: EventRow) -> dict:
    """A live (unstored) summary in the SAME shape as a stored row.

    Deliberately identical keys to :func:`metric_row_to_dict`: the UI renders
    one component for ``current`` whether it came from a chain snapshot or
    from the store, and a shape that differed by basis would push the
    branching into the view where the §37 labelling is hardest to audit.
    """
    pre = summary.pre
    return {
        "basis": summary.basis,
        "status": summary.status,
        "as_of": None,
        "expiry": summary.expiry.isoformat() if summary.expiry else None,
        "strike": summary.strike,
        "spot": pre.spot if pre is not None else None,
        "implied_move_pct": pre.pct if pre is not None else None,
        "implied_move_points": pre.points if pre is not None else None,
        "iv_before": summary.iv_before,
        "iv_after": summary.iv_after,
        "iv_crush_pct": summary.iv_crush_pct,
        "actual_move_pct": summary.actual_move_pct,
        "implied_realized_ratio": summary.ratio,
        "classification": summary.classification,
        "call_ticker": None,
        "put_ticker": None,
        "notes": [str(v) for v in summary.notes.values()],
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "event_date": (
            summary.event_date.isoformat()
            if summary.event_date
            else event_date_et(event_row).isoformat()
        ),
    }


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


async def _stored_metrics_for(
    session: AsyncSession, event_ids: list[int], basis: str
) -> dict[int, EventOptionMetricRow]:
    """``{event_id: row}`` for one basis — ONE query for the whole history.

    The history table shows up to eight prior events; a per-row lookup would
    be eight queries for a page load, which is the shape of cost the
    read/write split exists to avoid.
    """
    if not event_ids:
        return {}
    rows = (
        (
            await session.execute(
                select(EventOptionMetricRow).where(
                    EventOptionMetricRow.event_id.in_(event_ids),
                    EventOptionMetricRow.basis == basis,
                )
            )
        )
        .scalars()
        .all()
    )
    return {row.event_id: row for row in rows}


async def build_event_options_payload(
    session: AsyncSession,
    event_row: EventRow,
    *,
    as_of: datetime,
    provider: object | None = None,
    provider_name: str = "",
    history_last: int = DEFAULT_HISTORY,
) -> dict:
    """The §18/§36/§37/§66 option context for one event, as of an instant.

    ``current`` is the event's OWN implied move and its basis depends on where
    the event sits relative to ``as_of``:

    - UPCOMING (``scheduled_at > as_of``) — the LIVE chain, priced now, basis
      ``LIVE_CHAIN_SNAPSHOT``, never stored. A stored historical row is
      preferred over nothing when the chain refuses, because a reconstruction
      labelled as one is more useful than silence — but it is never presented
      as the live number.
    - PAST — the STORED reconstruction, basis
      ``HISTORICAL_DAILY_CLOSE_APPROXIMATION``. This read does NOT fetch: an
      event with no stored row answers with ``coverage.reason`` naming the
      backfill.

    ``history`` walks the previous comparable EARNINGS prints for this ticker
    (the same ``_past_comparable_rows`` pool the price tab uses, gated at
    ``as_of``) and renders each one's stored reconstruction, newest first. Only
    events that HAVE a stored row appear — an empty history is honest, an
    invented one is not.

    ``stats`` is :func:`historical_move_stats` over those prints' REALIZED
    moves and, separately, over what was IMPLIED for them: "the market has
    charged 6% and the stock has moved 8%" is the §66 comparison, and it needs
    both distributions measured over the same set of events.

    ``comparison`` puts this event's implied move beside those historical
    statistics — the single most useful line on the tab, and the one §37's
    disclaimer is attached to: it is a comparison of PRICES to past MOVES, not
    a forecast.

    ``disclaimer`` carries the §37 wording verbatim into the payload, so a
    consumer that renders the number cannot render it without the caveat.
    """
    ticker = (event_row.ticker or "").strip().upper()
    scheduled = _as_utc(event_row.scheduled_at)
    upcoming = scheduled > as_of
    payload: dict = {
        "event_id": event_row.id,
        "event_key": event_row.event_key,
        "ticker": ticker or None,
        "as_of": as_of.isoformat(),
        "scheduled_at_utc": scheduled.isoformat(),
        "is_upcoming": upcoming,
        "disclaimer": DISCLAIMER,
        "not_backtestable": list(NOT_BACKTESTABLE),
    }

    stored = await _stored_metrics_for(session, [event_row.id], BASIS_HISTORICAL)
    stored_row = stored.get(event_row.id)

    current: dict | None = None
    coverage_reason: str | None = None
    if upcoming:
        spot = await _latest_close(session, ticker, as_of) if ticker else None
        summary = live_implied_move(
            event_row,
            provider=provider,
            provider_name=provider_name,
            spot=spot,
            as_of_date=as_of.date(),
        )
        if summary.status != STATUS_NO_DATA:
            current = summary_to_block(summary, event_row)
        elif stored_row is not None:
            # A reconstruction LABELLED as one beats silence — but it is never
            # dressed up as the live number: the block keeps its historical
            # basis and the reason says the chain is what failed.
            current = metric_row_to_dict(stored_row, event_row)
            coverage_reason = next(iter(summary.notes.values()), None)
        else:
            current = summary_to_block(summary, event_row)
            coverage_reason = next(iter(summary.notes.values()), None)
    elif stored_row is not None:
        current = metric_row_to_dict(stored_row, event_row)
    elif not ticker:
        # A CPI release or an FOMC decision has no issuer whose options these
        # would be. Naming the backfill here would offer a remedy that cannot
        # work — that button answers ``no_ticker`` too — and a reason that
        # sends the user somewhere useless is worse than one that says what is
        # actually missing.
        coverage_reason = "no_ticker"
    else:
        coverage_reason = NO_STORED_METRICS_REASON

    past_rows = await _past_comparable_rows(session, event_row, as_of)
    bounded = max(0, min(int(history_last), MAX_HISTORY_BACKFILL))
    selected = past_rows[-bounded:] if bounded else []
    metrics = await _stored_metrics_for(
        session, [row.id for row in selected], BASIS_HISTORICAL
    )
    history: list[dict] = []
    attempted_no_data = 0
    for row in reversed(selected):  # newest first
        metric = metrics.get(row.id)
        if metric is None:
            continue
        if metric.status == STATUS_NO_DATA:
            # An ATTEMPTED-BUT-EMPTY row stays OUT of the history list for the
            # same reason an unfetched event does: a row of nulls enters the
            # §66 chart as a zero-height bar and reads as "the market priced
            # nothing", which is a claim about the option market rather than
            # about this platform's backfill state. The row is not discarded
            # though — it is COUNTED, so ``coverage`` can distinguish "we
            # looked and came back empty" from "nobody has looked", which is
            # exactly the distinction storing it was for.
            attempted_no_data += 1
            continue
        history.append(metric_row_to_dict(metric, row))

    actual_moves = [
        r["actual_move_pct"] for r in history if r["actual_move_pct"] is not None
    ]
    implied_moves = [
        r["implied_move_pct"] for r in history if r["implied_move_pct"] is not None
    ]
    stats = {
        "actual": historical_move_stats(actual_moves),
        "implied": historical_move_stats(implied_moves),
    }

    implied_now = current.get("implied_move_pct") if current else None
    if current is not None and current.get("status") == STATUS_NO_DATA:
        # The status is the server retracting its own computation; carrying the
        # number into the comparison would reintroduce exactly what NO_DATA
        # withdrew.
        implied_now = None
    ratio, classification = implied_vs_realized(
        implied_now, stats["actual"]["median_abs"]
    )
    payload["comparison"] = {
        "implied_pct": implied_now,
        "hist_median_abs": stats["actual"]["median_abs"],
        "hist_p90_abs": stats["actual"]["p90_abs"],
        "hist_max_abs": stats["actual"]["max_abs"],
        "vs_median_ratio": ratio,
        "vs_median_classification": classification,
    }

    payload["current"] = current
    payload["history"] = history
    payload["stats"] = stats
    payload["coverage"] = {
        "stored_metrics": stored_row is not None,
        "history_events": len(past_rows),
        "history_with_metrics": len(history),
        # Prior prints the backfill DID attempt and that came back empty —
        # rate-limited, capability-refused, or with an untraded leg. Without
        # this the UI cannot tell a gap it should offer to retry from one it
        # already knows is unfillable.
        "history_attempted_no_data": attempted_no_data,
        "reason": coverage_reason,
    }
    return payload


async def _latest_close(
    session: AsyncSession, ticker: str, as_of: datetime
) -> float | None:
    """The newest stored equity close knowable at ``as_of`` — the live spot.

    STORED, never fetched, for the same reason :func:`_equity_bars` is: this
    runs on a GET. A daily close is a coarse spot for a live chain — the
    §36 percentage would be a fraction of last night's price rather than of
    the current print — but it is a REAL price, and the alternative is either
    an extra quote call on every page open or a fabricated one.

    Gated by the ET-16:00 rule the rest of the event layer uses via
    ``as_of_bar_filter``: a bar is only knowable after its session closes.
    """
    from libs.trading_core.events.reaction import as_of_bar_filter

    bars = await _equity_bars(session, ticker)
    knowable = as_of_bar_filter(bars, as_of)
    if not knowable:
        return None
    close = float(knowable[-1].close)
    return close if close > 0.0 else None
