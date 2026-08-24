"""Event taxonomy helpers — natural keys, session classification, lifecycle
and time handling (event spec §5, §6, §10, §11, §67).

Pure stdlib, deterministic, **no I/O** and — by the audit's §7.4 static
guard — no import of :mod:`libs.market_data` or :mod:`libs.event_calendar`.
This module owns the two things every other event module needs before it can
say anything: the deterministic natural key that makes ingestion idempotent,
and the timezone arithmetic that turns a UTC instant into "before market /
after market" on the correct exchange day.

Time contract (spec §10): timestamps are stored and compared in **UTC**;
the event's own timezone string travels alongside so the UI can render the
local wall clock. US market events default to ``America/New_York``. Any
naive datetime handed to this module is a programming error, not something
to silently assume UTC for — the functions raise ``ValueError``.

Note on the timezone constant: ``EASTERN`` here is the pure-library
definition. ``libs/market_data/{massive,alpaca}.py`` and
``apps/gateway/{risk_snapshot,routers/analysis}.py`` each carry their own
copy for the same zone; the pure event layer may not import any of them
(gateway → libs is the only legal direction, and the §7.4 guard forbids the
market_data edge), so this is the events-layer canonical name rather than a
gratuitous fifth definition.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from libs.trading_core.models.enums import EventLifecycle, EventSession, EventType

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

#: Default display/event timezone for US-listed and US-macro events (§10).
DEFAULT_EVENT_TIMEZONE = "America/New_York"

#: Regular US equity session, used when no ``market_calendar`` row exists for
#: the event's ET date (§10: never date-only logic, but a documented default
#: beats an UNKNOWN that hides a real BMO/AMC distinction).
DEFAULT_MARKET_OPEN = time(9, 30)
DEFAULT_MARKET_CLOSE = time(16, 0)

#: Event types whose natural key is keyed on the macro release period rather
#: than a calendar date — a CPI release that slips by a day is still the same
#: "CPI for 2026-07" event (§5, §15).
MACRO_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.CPI,
        EventType.PPI,
        EventType.PCE,
        EventType.GDP,
        EventType.EMPLOYMENT_REPORT,
        EventType.JOLTS,
        EventType.RETAIL_SALES,
        EventType.ISM,
        EventType.CONSUMER_SENTIMENT,
    }
)

#: Fed/FOMC types keyed on the ET calendar date of the event itself.
FOMC_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.FOMC_MEETING,
        EventType.FOMC_DECISION,
        EventType.FOMC_PRESS_CONFERENCE,
        EventType.FOMC_MINUTES,
    }
)

#: Types that are market-wide by construction — no ticker makes them
#: relevant, they move everything (§12 MARKET_WIDE tier).
MARKET_WIDE_EVENT_TYPES: frozenset[EventType] = (
    MACRO_EVENT_TYPES
    | FOMC_EVENT_TYPES
    | frozenset({EventType.FED_SPEECH, EventType.FED_BOARD_EVENT, EventType.MARKET_HOLIDAY})
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(value: str | None, *, max_length: int | None = None) -> str:
    """Lowercase ``[a-z0-9-]`` slug used inside natural keys.

    Deterministic and lossy on purpose: two titles that differ only in
    punctuation collapse to the same key, which is what keeps a re-worded
    RSS headline from creating a duplicate FED_SPEECH card.
    """
    if not value:
        return ""
    out = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    if max_length is not None:
        out = out[:max_length].strip("-")
    return out


def require_utc(value: datetime, *, name: str = "datetime") -> datetime:
    """Return ``value`` normalised to UTC; raise on a naive datetime.

    Naive datetimes are refused rather than assumed (spec §10): guessing the
    zone of an event timestamp is exactly how a BMO release becomes an AMC
    release.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); got naive {value!r}")
    return value.astimezone(UTC)


def to_local(value: datetime, tz: str | ZoneInfo = DEFAULT_EVENT_TIMEZONE) -> datetime:
    """Convert a tz-aware instant into the event's own timezone (§10)."""
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    return require_utc(value, name="value").astimezone(zone)


def eastern_date(value: datetime) -> date:
    """The ``America/New_York`` calendar date of a UTC instant.

    Natural keys use the ET date, not the UTC date: an AMC earnings release
    at 20:15 ET is 00:15 UTC the *next* day, and keying it on the UTC date
    would split one quarter's release across two cards.
    """
    return to_local(value, EASTERN).date()


def classify_session(
    scheduled_at_utc: datetime,
    market_open_utc: datetime | None = None,
    market_close_utc: datetime | None = None,
) -> EventSession:
    """Classify an instant as BEFORE / DURING / AFTER market (spec §6).

    ``market_open_utc``/``market_close_utc`` come from that ET day's
    ``market_calendar`` row when one exists — that is what makes a 13:00 ET
    release on a half-day (13:00 close) an AFTER_MARKET event rather than a
    DURING_MARKET one. With no calendar row the regular 09:30-16:00 ET
    session is assumed.

    Weekends and instants on a day whose calendar row is absent are still
    classified by clock time; UNKNOWN is reserved for the case where the
    caller has no usable timestamp at all (``classify_session`` is then not
    called — the event simply keeps ``EventSession.UNKNOWN``).
    """
    moment = require_utc(scheduled_at_utc, name="scheduled_at_utc")
    if (market_open_utc is None) != (market_close_utc is None):
        raise ValueError("market_open_utc and market_close_utc must be given together")
    if market_open_utc is None or market_close_utc is None:
        local = moment.astimezone(EASTERN)
        open_local = datetime.combine(local.date(), DEFAULT_MARKET_OPEN, tzinfo=EASTERN)
        close_local = datetime.combine(local.date(), DEFAULT_MARKET_CLOSE, tzinfo=EASTERN)
        open_utc = open_local.astimezone(UTC)
        close_utc = close_local.astimezone(UTC)
    else:
        open_utc = require_utc(market_open_utc, name="market_open_utc")
        close_utc = require_utc(market_close_utc, name="market_close_utc")
        if close_utc <= open_utc:
            raise ValueError("market_close_utc must be after market_open_utc")
    if moment < open_utc:
        return EventSession.BEFORE_MARKET
    if moment >= close_utc:
        return EventSession.AFTER_MARKET
    return EventSession.DURING_MARKET


def session_anchor_time(session: EventSession) -> time:
    """The ET wall-clock time an ESTIMATED event of this session is pinned to.

    Used by the cadence estimator (§7): an estimate must still be a real
    instant, and pinning BMO to 07:00 ET / AMC to 16:05 ET keeps the
    estimated card in the right half of the trading day without pretending
    to know the minute.
    """
    if session is EventSession.BEFORE_MARKET:
        return time(7, 0)
    if session is EventSession.AFTER_MARKET:
        return time(16, 5)
    if session is EventSession.DURING_MARKET:
        return time(12, 0)
    return time(12, 0)


# ---------------------------------------------------------------------------
# Natural keys (§5, §6) — deterministic, source-independent dedup identity.
# ---------------------------------------------------------------------------


def event_key(
    event_type: EventType,
    *,
    scheduled_at: datetime | None = None,
    ticker: str | None = None,
    release_period: str | None = None,
    speaker: str | None = None,
    title: str | None = None,
    exchange: str | None = None,
    subtype: str | None = None,
) -> str:
    """Build the deterministic natural key for an event.

    The key is what makes ingestion idempotent across providers and restarts:
    two sources describing the same release must produce the same string, and
    it must be computable from the candidate alone (no DB lookup).

    Shapes:

    - ``EARNINGS``            -> ``EARNINGS:{TICKER}:{YYYY-MM-DD}`` (ET date)
    - macro                   -> ``{TYPE}:{release_period}``, e.g. ``CPI:2026-07``
    - FOMC_*                  -> ``{TYPE}:{YYYY-MM-DD}`` (ET date)
    - ``FED_SPEECH``          -> ``FED_SPEECH:{YYYY-MM-DD}:{slug(speaker)}:{slug(title)[:40]}``
    - ``MARKET_HOLIDAY``      -> ``MARKET_HOLIDAY:{EXCHANGE}:{YYYY-MM-DD}``
    - ``CORPORATE_EVENT``     -> ``CORPORATE_EVENT:{TICKER}:{subtype}:{YYYY-MM-DD}``

    Note the ET date for EARNINGS: for an ESTIMATED event that is the
    *estimated* ET date, so an estimate that later resolves to the same day
    merges instead of duplicating (the ±21-day drift case is handled by
    :func:`libs.trading_core.events.models.same_event`).
    """
    def _et_date() -> str:
        if scheduled_at is None:
            raise ValueError(f"{event_type} event_key requires scheduled_at")
        return eastern_date(scheduled_at).isoformat()

    if event_type is EventType.EARNINGS:
        if not ticker:
            raise ValueError("EARNINGS event_key requires a ticker")
        return f"EARNINGS:{ticker.strip().upper()}:{_et_date()}"

    if event_type in MACRO_EVENT_TYPES:
        if not release_period:
            raise ValueError(f"{event_type} event_key requires release_period")
        return f"{event_type.value}:{release_period.strip()}"

    if event_type in FOMC_EVENT_TYPES:
        return f"{event_type.value}:{_et_date()}"

    if event_type is EventType.FED_SPEECH:
        return (
            f"FED_SPEECH:{_et_date()}:{slug(speaker)}:{slug(title, max_length=40)}"
        )

    if event_type is EventType.FED_BOARD_EVENT:
        return f"FED_BOARD_EVENT:{_et_date()}:{slug(title, max_length=40)}"

    if event_type is EventType.MARKET_HOLIDAY:
        exch = (exchange or "US").strip().upper()
        return f"MARKET_HOLIDAY:{exch}:{_et_date()}"

    if event_type is EventType.CORPORATE_EVENT:
        if not ticker:
            raise ValueError("CORPORATE_EVENT event_key requires a ticker")
        sub = slug(subtype) or "event"
        return f"CORPORATE_EVENT:{ticker.strip().upper()}:{sub}:{_et_date()}"

    raise ValueError(f"no event_key rule for {event_type}")


# ---------------------------------------------------------------------------
# Lifecycle (§67)
# ---------------------------------------------------------------------------

#: An event enters PRE_EVENT this many days before it happens — the same
#: horizon the T-minus alert uses (§11 "roughly one week before").
PRE_EVENT_DAYS = 7
#: LIVE window around the scheduled instant.
LIVE_LEAD = timedelta(minutes=5)
LIVE_TAIL = timedelta(minutes=60)
#: Trading days after the event during which it is still POST_EVENT.
POST_EVENT_TRADING_DAYS = 5


def _add_trading_days(start: datetime, days: int) -> datetime:
    """``start`` advanced by ``days`` weekdays, in ET, returned as UTC.

    Weekday arithmetic, not a holiday calendar: this module is pure and has
    no ``market_calendar`` access. The boundary it draws is a display
    lifecycle, not a trading decision, so a holiday shifting POST_EVENT ->
    ARCHIVED by one day is acceptable and documented rather than faked.
    """
    local = require_utc(start, name="start").astimezone(EASTERN)
    remaining = days
    while remaining > 0:
        local += timedelta(days=1)
        if local.weekday() < 5:
            remaining -= 1
    return local.astimezone(UTC)


def lifecycle(scheduled_at: datetime, now: datetime) -> EventLifecycle:
    """Where an event sits relative to ``now`` (spec §67).

    - ``SCHEDULED``  : more than 7 days out
    - ``PRE_EVENT``  : within 7 days, before the LIVE window
    - ``LIVE``       : ``[t - 5min, t + 60min]`` (inclusive both ends)
    - ``POST_EVENT`` : up to 5 trading days after the LIVE window
    - ``ARCHIVED``   : beyond that
    """
    moment = require_utc(scheduled_at, name="scheduled_at")
    current = require_utc(now, name="now")
    if current < moment - LIVE_LEAD:
        return (
            EventLifecycle.SCHEDULED
            if moment - current > timedelta(days=PRE_EVENT_DAYS)
            else EventLifecycle.PRE_EVENT
        )
    if current <= moment + LIVE_TAIL:
        return EventLifecycle.LIVE
    if current <= _add_trading_days(moment, POST_EVENT_TRADING_DAYS):
        return EventLifecycle.POST_EVENT
    return EventLifecycle.ARCHIVED
