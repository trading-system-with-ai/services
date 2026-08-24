"""Event replay — intraday reaction, the §20 replay bundle, the §60 history
table and §15 previous-event linkage (event spec §15, §17, §19, §20, §60,
§85, §96; audit §7, §11.4; Phase C unit U2).

Pure stdlib, deterministic, **no I/O**. Like every other module under
``libs/trading_core/events/`` this one may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4 static guard); the
gateway seam (``apps/gateway/event_replay.py``) hands it plain
:class:`MinuteBar` values and gets frozen results back.

Four ideas carry the module:

1. **The anchor is chosen by the session, and it is never guessed** (§17).
   An AMC print is measured against that day's regular close and reacts at
   the NEXT session's open; a BMO print is measured against the previous
   close and reacts at the SAME day's open; a DURING_MARKET print has no
   open to gap into, so it is measured from the last minute bar at or before
   the release. UNKNOWN is treated like AMC because that is the conservative
   read of an unlabelled release, but it is flagged
   ``unknown_session_assumed_after_market`` with ``confidence="low"`` so the
   payload never lets a caller mistake an assumption for a measurement.

2. **The reaction windows come off real bars, never off the clock** (§17
   "do not calculate unavailable precision"). ``+5m`` is the close of the
   LAST bar whose timestamp is ``<= open_ts + 5 minutes``; if the tape is
   sparse (an illiquid name, an extended-hours stretch) that bar may be
   older than the nominal mark, so :class:`IntradayWindow` carries the bar
   timestamp it actually used and the lag in seconds. A window with no bar
   at all is ``None`` with a reason — never the open price standing in for a
   move of zero.

3. **Absence is a value** (house rule, §85). Every numeric field is
   ``float | None``; every ``None`` has a companion string in ``reasons``.
   A non-positive or non-finite reference price makes the return ``None``
   with ``"reference_price_not_positive"``, not a division blow-up, and
   nothing here can emit a NaN or an ``inf``.

4. **ET is the only calendar that matters.** 09:30/16:00/04:00/20:00 are ET
   wall-clock boundaries, so every comparison converts through
   :data:`taxonomy.EASTERN`. A UTC-offset implementation is right for half
   the year and silently wrong across a DST transition — the tests pin both
   sides of both transitions.

:func:`build_event_replay` is a pure composer: it takes the pieces the
gateway assembled (the event reference, the point-in-time refs for what was
knowable BEFORE the release, the release facts, this module's intraday
result and Phase E1's daily :class:`~libs.trading_core.events.reaction.ReactionResult`)
and lays them out in the §20 order — information before, release, immediate
reaction, subsequent reaction — with the provenance labels the UI switches
on. It computes nothing it was not handed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from libs.trading_core.events.models import previous_comparable
from libs.trading_core.events.reaction import (
    AbnormalResult,
    HistoryStats,
    ReactionResult,
    history_stats,
)
from libs.trading_core.events.taxonomy import EASTERN, UTC
from libs.trading_core.models.enums import EventSession, EventStatus, EventType

__all__ = [
    "BASIS_INTRADAY_AFTER_MARKET",
    "BASIS_INTRADAY_BEFORE_MARKET",
    "BASIS_INTRADAY_DURING_MARKET",
    "BASIS_INTRADAY_UNKNOWN",
    "CONSENSUS_UNAVAILABLE_REASON",
    "DEFAULT_WINDOWS_MIN",
    "EXTENDED_CLOSE_ET",
    "EXTENDED_OPEN_ET",
    "IMPLIED_MOVE_UNAVAILABLE_REASON",
    "INTRADAY_MODEL_VERSION",
    "REGULAR_CLOSE_ET",
    "REGULAR_OPEN_ET",
    "REPLAY_MODEL_VERSION",
    "EventReplay",
    "IntradayReaction",
    "IntradayWindow",
    "MinuteBar",
    "build_event_replay",
    "history_table",
    "intraday_reaction",
    "link_previous_events",
]

#: Regular US equity session in ET. Same numbers as
#: ``reaction.REGULAR_CLOSE_ET`` / ``taxonomy.DEFAULT_MARKET_OPEN``, expressed
#: here as ``(hour, minute)`` pairs because they are function DEFAULTS a
#: caller may override for a half-day session.
REGULAR_OPEN_ET = (9, 30)
REGULAR_CLOSE_ET = (16, 0)

#: Extended-hours bounds in ET: pre-market opens 04:00, after-hours ends
#: 20:00. The AMC after-hours window closes at 20:00 on the release day —
#: anything printed after that is the next session's business.
EXTENDED_OPEN_ET = (4, 0)
EXTENDED_CLOSE_ET = (20, 0)

#: Post-open marks in minutes (§17 "5m / 30m / 1h post-event reactions").
DEFAULT_WINDOWS_MIN: tuple[int, ...] = (5, 30, 60)

#: First-hour scan length, in minutes, for ``max_move_first_hour``.
FIRST_HOUR_MINUTES = 60

#: Opening-volume window, in minutes, for ``volume_first_30m``.
VOLUME_WINDOW_MINUTES = 30

#: Basis labels — fixed strings that travel into the API payload and the UI
#: tooltip (§85: the window is always shown, never implied). Deliberately
#: distinct from ``reaction.BASIS_*`` (which label DAILY windows) so a payload
#: carrying both cannot confuse them.
BASIS_INTRADAY_AFTER_MARKET = "after_market_next_open_anchor"
BASIS_INTRADAY_BEFORE_MARKET = "before_market_same_day_open_anchor"
BASIS_INTRADAY_DURING_MARKET = "during_market_release_anchor"
BASIS_INTRADAY_UNKNOWN = "unknown_session_assumed_after_market"

#: §33/§98 — consensus is unavailable at ANY instant (Benzinga 403), so the
#: §60 surprise columns are an explicit absence rather than a null cell. The
#: gateway states the same fact in its own longer form; the SHORT form the
#: history table prints is pinned here so the pure layer needs no import.
CONSENSUS_UNAVAILABLE_REASON = "CONSENSUS DATA UNAVAILABLE"

#: §36/§60 — historical implied move needs option quotes the platform does
#: not have yet; Phase I reconstructs an approximation from option daily bars.
IMPLIED_MOVE_UNAVAILABLE_REASON = (
    "options intelligence not yet available (Phase I)"
)

#: Model versions, mirroring ``IMPORTANCE_MODEL_VERSION`` /
#: ``FUNDAMENTALS_MODEL_VERSION``: a stored replay says which arithmetic
#: produced it.
INTRADAY_MODEL_VERSION = "c1-intraday-v1"
REPLAY_MODEL_VERSION = "c1-replay-v1"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinuteBar:
    """One intraday OHLCV bar, stamped at its bar-OPEN instant in UTC.

    The pure-library mirror of the ``stock_bars_1m`` row and of
    ``libs.market_data.provider.IntradayBar`` (which this layer may not
    import). ``ts_utc`` must be timezone-aware — a naive timestamp is refused
    at construction rather than assumed, because guessing the zone of a
    minute bar moves an after-hours print into the regular session.
    ``volume`` is ``float`` so a provider reporting fractional/notional
    volume is not silently truncated.
    """

    ts_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        ts = self.ts_utc
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            raise ValueError(
                f"ts_utc must be timezone-aware (UTC); got naive {ts!r}"
            )
        object.__setattr__(self, "ts_utc", ts.astimezone(UTC))

    @property
    def ts_et(self) -> datetime:
        """This bar's ET wall clock — the only calendar the rules use."""
        return self.ts_utc.astimezone(EASTERN)


@dataclass(frozen=True)
class IntradayWindow:
    """One post-anchor mark (``+5m``/``+30m``/``+60m``) and how it was found.

    ``bar_ts_utc`` is the timestamp of the bar actually used, which on a
    sparse tape can be EARLIER than ``target_ts_utc``; ``lag_seconds`` is the
    gap between them. Both travel with the number because "+30m" measured off
    a bar 11 minutes stale is a weaker claim than one measured off the 30th
    minute, and the UI cannot say so unless the payload does.
    """

    minutes: int
    target_ts_utc: datetime | None = None
    bar_ts_utc: datetime | None = None
    price: float | None = None
    move: float | None = None
    lag_seconds: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class IntradayReaction:
    """The minute-bar view of one event's immediate reaction (§17, §20).

    ``basis`` names the anchor rule that was applied and ``confidence`` is
    ``"low"`` only for an UNKNOWN session, where the anchor is an assumption
    rather than a measurement. ``available`` is ``False`` when no bar could
    anchor anything at all — then every numeric field is ``None`` and
    ``reasons`` says why, once.

    Field groups:

    - pre-release: ``pre_event_close`` (supplied by the caller — the daily
      close for AMC/UNKNOWN, the prior close for BMO), ``after_hours_move``
      /``after_hours_last_ts``/``after_hours_bars`` for an AMC release,
      ``premarket_move``/``premarket_last_ts``/``premarket_bars`` for a BMO
      release.
    - the anchor: ``reference_price`` and ``reference_ts`` — the price every
      window is measured against, whichever session rule produced it.
    - the reaction: ``open_price``/``open_ts`` and ``gap_at_open``, then
      ``windows[k]``.
    - context: ``max_move_first_hour``, ``volume_first_30m`` and
      ``avg_volume_first_30m_prior_5_days``.
    """

    session: EventSession | None = None
    basis: str | None = None
    confidence: str = "high"
    available: bool = False
    event_ts_utc: datetime | None = None
    event_date_et: date | None = None
    pre_event_close: float | None = None
    after_hours_move: float | None = None
    after_hours_last_ts: datetime | None = None
    after_hours_bars: int = 0
    premarket_move: float | None = None
    premarket_last_ts: datetime | None = None
    premarket_bars: int = 0
    reference_price: float | None = None
    reference_ts: datetime | None = None
    open_price: float | None = None
    open_ts: datetime | None = None
    gap_at_open: float | None = None
    windows: Mapping[int, IntradayWindow] = field(default_factory=dict)
    max_move_first_hour: float | None = None
    volume_first_30m: float | None = None
    avg_volume_first_30m_prior_5_days: float | None = None
    volume_ratio_first_30m: float | None = None
    session_date_et: date | None = None
    bars_used: int = 0
    reasons: Mapping[str, str] = field(default_factory=dict)
    model_version: str = INTRADAY_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", dict(self.windows))
        object.__setattr__(self, "reasons", dict(self.reasons))

    def move(self, minutes: int) -> float | None:
        """The ``+minutes`` move, or ``None`` when that window has no bar."""
        window = self.windows.get(minutes)
        return window.move if window is not None else None

    @property
    def horizons(self) -> tuple[int, ...]:
        """Window marks this result was computed for, ascending."""
        return tuple(sorted(self.windows))


@dataclass(frozen=True)
class EventReplay:
    """The §20 replay bundle for one event, in the §20 order.

    Four blocks, and nothing between them is inferred:

    ``information_before`` — REFERENCES the caller supplies to what was
    knowable before the release (a fundamentals snapshot ref, a price-context
    ref, a news-window ref). Each is optional and each absent one is a reason,
    because "no news block" and "no news" are different claims (§85).

    ``release`` — the observed facts of the release itself: the UTC instant,
    the ET wall clock, the session and the source that asserted them.

    ``immediate_reaction`` — this module's :class:`IntradayReaction`, QUANT
    arithmetic over DATA minute bars.

    ``subsequent_reaction`` — Phase E1's daily gap/1D/3D/5D/10D plus the
    abnormal-vs-SPY overlay, unchanged.
    """

    event_id: int | None = None
    event_key: str | None = None
    event_type: EventType | None = None
    ticker: str | None = None
    date_et: date | None = None
    session: EventSession | None = None
    status: EventStatus | str | None = None
    source_url: str | None = None
    information_before: Mapping[str, object] = field(default_factory=dict)
    release: Mapping[str, object] = field(default_factory=dict)
    immediate_reaction: Mapping[str, object] = field(default_factory=dict)
    subsequent_reaction: Mapping[str, object] = field(default_factory=dict)
    data_freshness: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)
    model_version: str = REPLAY_MODEL_VERSION

    def __post_init__(self) -> None:
        for name in (
            "information_before",
            "release",
            "immediate_reaction",
            "subsequent_reaction",
            "data_freshness",
            "provenance",
            "reasons",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))

    def to_dict(self) -> dict:
        """JSON-ready mapping, in the §20 block order.

        Enums render as their ``value`` and dates as ISO strings so the
        gateway can hand this straight to FastAPI without a second walk.
        """
        return {
            "event": {
                "event_id": self.event_id,
                "event_key": self.event_key,
                "event_type": _enum_value(self.event_type),
                "ticker": self.ticker,
                "date_et": _date_iso(self.date_et),
                "session": _enum_value(self.session),
                "status": _enum_value(self.status),
                "source_url": self.source_url,
            },
            "information_before": dict(self.information_before),
            "release": dict(self.release),
            "immediate_reaction": dict(self.immediate_reaction),
            "subsequent_reaction": dict(self.subsequent_reaction),
            "data_freshness": dict(self.data_freshness),
            "provenance": dict(self.provenance),
            "reasons": dict(self.reasons),
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Small helpers — every one of them refuses to fabricate
# ---------------------------------------------------------------------------


def _finite(value: float | None) -> float | None:
    """``None`` unless the value is a finite float (no NaN, no ±inf)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _pct_change(later: float | None, earlier: float | None) -> float | None:
    """``later / earlier - 1``, or ``None`` when the base is unusable.

    Same rule as :func:`reaction._pct_change`: a zero or negative base is
    malformed price data, not a divide-by-zero to paper over with ``inf``.
    """
    later = _finite(later)
    earlier = _finite(earlier)
    if later is None or earlier is None or earlier <= 0.0:
        return None
    return _finite(later / earlier - 1.0)


def _date_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _ts_iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _enum_value(value: object) -> object:
    """``EventSession.AFTER_MARKET`` -> ``"AFTER_MARKET"``; passthrough else."""
    return getattr(value, "value", value)


def _require_aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{name} must be timezone-aware (UTC); got naive {value!r}"
        )
    return value.astimezone(UTC)


def _et_boundary(day: date, hm: tuple[int, int]) -> datetime:
    """The UTC instant of an ET wall-clock time on an ET calendar date.

    Built in ET and converted, never by adding a fixed offset to UTC: 09:30
    ET is 13:30Z in summer and 14:30Z in winter, and a DST-blind boundary
    puts an hour of the regular session into "pre-market" twice a year.
    """
    hour, minute = hm
    return datetime.combine(day, time(hour, minute), tzinfo=EASTERN).astimezone(UTC)


def _check_sorted(bars: Sequence[MinuteBar]) -> None:
    """Bars must be strictly increasing by timestamp.

    Refused rather than sorted, matching ``reaction._check_sorted``: this
    layer cannot tell a mis-ordered series from a mis-stamped one, and a
    duplicate minute silently doubles the volume window.
    """
    for i in range(1, len(bars)):
        if not bars[i].ts_utc > bars[i - 1].ts_utc:
            raise ValueError(
                "bars must be strictly increasing by ts_utc, got "
                f"{bars[i - 1].ts_utc.isoformat()} followed by "
                f"{bars[i].ts_utc.isoformat()} at index {i}"
            )


def _in_span(
    bars: Sequence[MinuteBar],
    start: datetime,
    end: datetime,
    *,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
) -> list[MinuteBar]:
    """Bars inside ``[start, end]`` with per-edge inclusivity."""
    out: list[MinuteBar] = []
    for bar in bars:
        ts = bar.ts_utc
        if start_inclusive:
            if ts < start:
                continue
        elif ts <= start:
            continue
        if end_inclusive:
            if ts > end:
                continue
        elif ts >= end:
            continue
        out.append(bar)
    return out


def _last_at_or_before(
    bars: Sequence[MinuteBar], moment: datetime
) -> MinuteBar | None:
    found: MinuteBar | None = None
    for bar in bars:
        if bar.ts_utc <= moment:
            found = bar
        else:
            break
    return found


# ---------------------------------------------------------------------------
# §17 — the intraday reaction
# ---------------------------------------------------------------------------


def _unavailable_intraday(
    *,
    session: EventSession,
    basis: str,
    confidence: str,
    event_ts_utc: datetime,
    event_date_et: date,
    pre_event_close: float | None,
    windows_min: Sequence[int],
    reason: str,
    extra: Mapping[str, object] | None = None,
) -> IntradayReaction:
    """An all-``None`` result whose single reason explains the whole absence."""
    payload: dict[str, object] = {
        "session": session,
        "basis": basis,
        "confidence": confidence,
        "available": False,
        "event_ts_utc": event_ts_utc,
        "event_date_et": event_date_et,
        "pre_event_close": pre_event_close,
        "windows": {
            k: IntradayWindow(minutes=k, reason=reason) for k in windows_min
        },
        "reasons": {"bars": reason},
    }
    if extra:
        extra_reasons = dict(extra.get("reasons") or {})  # type: ignore[arg-type]
        payload.update(extra)
        # The caller's partial reasons come FIRST and the top-level ``bars``
        # reason wins: a partially-resolved result (after-hours measured, the
        # next session missing) must still say in one place why the whole
        # thing is unavailable, and losing that key to a merge would leave the
        # gateway printing "unavailable" with no explanation.
        payload["reasons"] = {**extra_reasons, "bars": reason}
    return IntradayReaction(**payload)  # type: ignore[arg-type]


def _open_bar_on(
    bars: Sequence[MinuteBar],
    day: date,
    *,
    regular_open_et: tuple[int, int],
    regular_close_et: tuple[int, int],
) -> MinuteBar | None:
    """The first regular-session bar on ``day`` (``>= 09:30``, ``< 16:00`` ET)."""
    start = _et_boundary(day, regular_open_et)
    end = _et_boundary(day, regular_close_et)
    for bar in bars:
        if start <= bar.ts_utc < end:
            return bar
    return None


def _next_session_date(
    bars: Sequence[MinuteBar],
    after: date,
    *,
    regular_open_et: tuple[int, int],
    regular_close_et: tuple[int, int],
) -> date | None:
    """The earliest ET date after ``after`` that HAS a regular-session bar.

    Derived from the bars themselves rather than from a calendar: the tape
    is the authority on which days traded, so weekends and holidays need no
    table and a half-day session needs no special case.
    """
    for bar in bars:
        day = bar.ts_et.date()
        if day <= after:
            continue
        start = _et_boundary(day, regular_open_et)
        end = _et_boundary(day, regular_close_et)
        if start <= bar.ts_utc < end:
            return day
    return None


def _window_marks(
    bars: Sequence[MinuteBar],
    *,
    anchor_ts: datetime,
    reference_price: float | None,
    windows_min: Sequence[int],
    session_end: datetime | None,
    reference_reason: str | None,
) -> dict[int, IntradayWindow]:
    """``{k: IntradayWindow}`` — the close of the LAST bar at/ before +k min.

    Never interpolates and never reaches past ``session_end``: a "+60m" mark
    filled from tomorrow's tape is not a first-hour reaction.
    """
    out: dict[int, IntradayWindow] = {}
    for k in windows_min:
        target = anchor_ts + timedelta(minutes=int(k))
        if session_end is not None and target > session_end:
            capped = session_end
        else:
            capped = target
        bar = _last_at_or_before(bars, capped)
        if bar is None or bar.ts_utc < anchor_ts:
            out[int(k)] = IntradayWindow(
                minutes=int(k),
                target_ts_utc=target,
                reason="no_bar_in_window",
            )
            continue
        price = _finite(bar.close)
        if price is None:
            out[int(k)] = IntradayWindow(
                minutes=int(k),
                target_ts_utc=target,
                bar_ts_utc=bar.ts_utc,
                reason="close_unavailable",
            )
            continue
        if reference_reason is not None:
            out[int(k)] = IntradayWindow(
                minutes=int(k),
                target_ts_utc=target,
                bar_ts_utc=bar.ts_utc,
                price=price,
                lag_seconds=int((target - bar.ts_utc).total_seconds()),
                reason=reference_reason,
            )
            continue
        move = _pct_change(price, reference_price)
        out[int(k)] = IntradayWindow(
            minutes=int(k),
            target_ts_utc=target,
            bar_ts_utc=bar.ts_utc,
            price=price,
            move=move,
            lag_seconds=int((target - bar.ts_utc).total_seconds()),
            reason=None if move is not None else "reference_price_not_positive",
        )
    return out


def _first_hour_max_move(
    bars: Sequence[MinuteBar], *, anchor_ts: datetime, open_price: float | None
) -> tuple[float | None, str | None]:
    """``max |close/open - 1|`` over the first :data:`FIRST_HOUR_MINUTES`.

    Measured against the OPEN of the reaction window (the contract's
    "close/open − 1"), which is what makes it a range statistic of the first
    hour rather than a second reading of the event move.
    """
    base = _finite(open_price)
    if base is None or base <= 0.0:
        return None, "open_price_not_positive"
    end = anchor_ts + timedelta(minutes=FIRST_HOUR_MINUTES)
    span = _in_span(bars, anchor_ts, end)
    moves = [
        abs(value)
        for value in (_pct_change(bar.close, base) for bar in span)
        if value is not None
    ]
    if not moves:
        return None, "no_bars_in_first_hour"
    return _finite(max(moves)), None


def _volume_first_30m(
    bars: Sequence[MinuteBar], *, anchor_ts: datetime
) -> tuple[float | None, str | None]:
    span = _in_span(bars, anchor_ts, anchor_ts + timedelta(minutes=VOLUME_WINDOW_MINUTES))
    values = [v for v in (_finite(bar.volume) for bar in span) if v is not None]
    if not values:
        return None, "no_bars_in_volume_window"
    return _finite(math.fsum(values)), None


def _avg_prior_open_volume(
    prior_bars: Sequence[MinuteBar] | None,
    *,
    session_date: date,
    regular_open_et: tuple[int, int],
    days: int = 5,
) -> tuple[float | None, str | None]:
    """Mean first-30-minute volume over the ``days`` sessions before ``session_date``.

    Only sessions that actually have opening bars count, and the mean names
    how many it had: an "average of 5 days" computed off 2 is a different
    claim, so fewer than two usable sessions refuses to average at all.
    """
    if not prior_bars:
        return None, "prior_session_bars_not_supplied"
    per_day: dict[date, float] = {}
    for bar in prior_bars:
        day = bar.ts_et.date()
        if day >= session_date:
            continue
        start = _et_boundary(day, regular_open_et)
        if not start <= bar.ts_utc < start + timedelta(minutes=VOLUME_WINDOW_MINUTES):
            continue
        volume = _finite(bar.volume)
        if volume is None:
            continue
        per_day[day] = per_day.get(day, 0.0) + volume
    if not per_day:
        return None, "no_prior_session_opening_bars"
    recent = [per_day[day] for day in sorted(per_day)[-int(days):]]
    if len(recent) < 2:
        return (
            None,
            f"insufficient_prior_sessions: {len(recent)} of {days}, need >= 2",
        )
    return _finite(math.fsum(recent) / len(recent)), None


def intraday_reaction(
    bars: Sequence[MinuteBar],
    *,
    event_ts_utc: datetime,
    session: EventSession,
    pre_event_close: float | None,
    regular_open_et: tuple[int, int] = REGULAR_OPEN_ET,
    regular_close_et: tuple[int, int] = REGULAR_CLOSE_ET,
    windows_min: Sequence[int] = DEFAULT_WINDOWS_MIN,
    prior_session_bars: Sequence[MinuteBar] | None = None,
) -> IntradayReaction:
    """The minute-bar reaction to one release (§17, §20).

    ``bars`` must be sorted strictly ascending by ``ts_utc`` and must ALREADY
    have been as-of gated by the caller (§14/§96 — the gateway filters
    ``ts <= as_of``; this layer never sees a bar it should not have).
    ``pre_event_close`` is the settled daily close the move is measured
    against: that day's close for an AMC/UNKNOWN release, the PREVIOUS day's
    close for a BMO one. It is the caller's to supply because only the daily
    series knows it — reconstructing it from the last 15:59 minute bar would
    quietly substitute the last trade for the official close.

    Session rules (the anchor is the whole design):

    ``AFTER_MARKET``
        ``after_hours_move`` = the close of the last bar in
        ``(event_ts, 20:00 ET on the release day]`` versus
        ``pre_event_close``. The reaction anchor is the first regular-session
        bar (``>= 09:30 ET``) of the next trading day PRESENT IN THE BARS, so
        ``gap_at_open = open / pre_event_close - 1`` and ``+5m/+30m/+60m``
        are measured from ``pre_event_close`` to the close of the last bar at
        or before ``open_ts + k``.

    ``BEFORE_MARKET``
        ``premarket_move`` = the last bar in ``[04:00, 09:30) ET`` on the
        release day versus ``pre_event_close``; then the same open/+k
        windows, on the release day itself.

    ``DURING_MARKET``
        There is no open to gap into. The anchor is the last bar at or before
        ``event_ts`` (``reference_price``/``reference_ts``) and the windows
        run from ``event_ts`` — basis
        ``during_market_release_anchor``, ``gap_at_open`` ``None`` with the
        reason ``"during_market_release_has_no_opening_gap"``.

    ``UNKNOWN``
        Treated exactly like ``AFTER_MARKET``, but basis
        ``unknown_session_assumed_after_market`` and ``confidence="low"``:
        the anchor is an assumption, and the payload says so.

    Every ``None`` carries a reason; no field is ever NaN or ``inf``.
    """
    event_ts_utc = _require_aware(event_ts_utc, name="event_ts_utc")
    windows_min = tuple(sorted({int(k) for k in windows_min}))
    for k in windows_min:
        if k < 1:
            raise ValueError(f"windows_min values must be >= 1, got {k}")
    _check_sorted(bars)

    event_et = event_ts_utc.astimezone(EASTERN)
    event_date_et = event_et.date()
    pre_close = _finite(pre_event_close)

    if session is EventSession.DURING_MARKET:
        basis, confidence = BASIS_INTRADAY_DURING_MARKET, "high"
    elif session is EventSession.BEFORE_MARKET:
        basis, confidence = BASIS_INTRADAY_BEFORE_MARKET, "high"
    elif session is EventSession.AFTER_MARKET:
        basis, confidence = BASIS_INTRADAY_AFTER_MARKET, "high"
    else:
        basis, confidence = BASIS_INTRADAY_UNKNOWN, "low"

    if not bars:
        return _unavailable_intraday(
            session=session,
            basis=basis,
            confidence=confidence,
            event_ts_utc=event_ts_utc,
            event_date_et=event_date_et,
            pre_event_close=pre_close,
            windows_min=windows_min,
            reason="no_minute_bars_available",
        )

    reasons: dict[str, str] = {}
    if pre_close is None:
        reasons["pre_event_close"] = "pre_event_close_not_supplied"
    elif pre_close <= 0.0:
        reasons["pre_event_close"] = "pre_event_close_not_positive"
        pre_close = None

    # --- DURING_MARKET: the anchor is the release itself -------------------
    if session is EventSession.DURING_MARKET:
        anchor_bar = _last_at_or_before(bars, event_ts_utc)
        if anchor_bar is None:
            return _unavailable_intraday(
                session=session,
                basis=basis,
                confidence=confidence,
                event_ts_utc=event_ts_utc,
                event_date_et=event_date_et,
                pre_event_close=pre_close,
                windows_min=windows_min,
                reason="no_bar_at_or_before_event",
            )
        reference_price = _finite(anchor_bar.close)
        reference_reason: str | None = None
        if reference_price is None or reference_price <= 0.0:
            reference_reason = "reference_price_not_positive"
            reasons["reference_price"] = reference_reason
            reference_price = None
        session_end = _et_boundary(event_date_et, regular_close_et)
        windows = _window_marks(
            bars,
            anchor_ts=event_ts_utc,
            reference_price=reference_price,
            windows_min=windows_min,
            session_end=session_end,
            reference_reason=reference_reason,
        )
        for k, window in windows.items():
            if window.move is None and window.reason:
                reasons[f"window_{k}m"] = window.reason
        max_move, max_reason = _first_hour_max_move(
            bars, anchor_ts=event_ts_utc, open_price=reference_price
        )
        if max_reason:
            reasons["max_move_first_hour"] = max_reason
        volume_30m, volume_reason = _volume_first_30m(bars, anchor_ts=event_ts_utc)
        if volume_reason:
            reasons["volume_first_30m"] = volume_reason
        avg_volume, avg_reason = _avg_prior_open_volume(
            prior_session_bars,
            session_date=event_date_et,
            regular_open_et=regular_open_et,
        )
        if avg_reason:
            reasons["avg_volume_first_30m_prior_5_days"] = avg_reason
        ratio = None
        if volume_30m is not None and avg_volume is not None and avg_volume > 0.0:
            ratio = _finite(volume_30m / avg_volume)
        else:
            reasons["volume_ratio_first_30m"] = (
                volume_reason or avg_reason or "volume_ratio_inputs_unavailable"
            )
        reasons["gap_at_open"] = "during_market_release_has_no_opening_gap"
        reasons["after_hours_move"] = "not applicable for a DURING_MARKET release"
        reasons["premarket_move"] = "not applicable for a DURING_MARKET release"
        return IntradayReaction(
            session=session,
            basis=basis,
            confidence=confidence,
            available=True,
            event_ts_utc=event_ts_utc,
            event_date_et=event_date_et,
            pre_event_close=pre_close,
            reference_price=reference_price,
            reference_ts=anchor_bar.ts_utc,
            open_price=None,
            open_ts=None,
            gap_at_open=None,
            windows=windows,
            max_move_first_hour=max_move,
            volume_first_30m=volume_30m,
            avg_volume_first_30m_prior_5_days=avg_volume,
            volume_ratio_first_30m=ratio,
            session_date_et=event_date_et,
            bars_used=len(bars),
            reasons=reasons,
        )

    # --- BMO / AMC / UNKNOWN: the anchor is a session open -----------------
    after_hours_move: float | None = None
    after_hours_last_ts: datetime | None = None
    after_hours_bars = 0
    premarket_move: float | None = None
    premarket_last_ts: datetime | None = None
    premarket_bars = 0

    if session is EventSession.BEFORE_MARKET:
        window_start = _et_boundary(event_date_et, EXTENDED_OPEN_ET)
        window_end = _et_boundary(event_date_et, regular_open_et)
        pre_bars = _in_span(
            bars, window_start, window_end, end_inclusive=False
        )
        premarket_bars = len(pre_bars)
        if not pre_bars:
            reasons["premarket_move"] = "no pre-market bars"
        else:
            premarket_last_ts = pre_bars[-1].ts_utc
            premarket_move = _pct_change(pre_bars[-1].close, pre_close)
            if premarket_move is None:
                reasons["premarket_move"] = (
                    "pre_event_close_not_positive"
                    if pre_close is None
                    else "close_unavailable"
                )
        reasons["after_hours_move"] = "not applicable for a BEFORE_MARKET release"
        session_date = event_date_et
    else:
        window_end = _et_boundary(event_date_et, EXTENDED_CLOSE_ET)
        post_bars = _in_span(
            bars, event_ts_utc, window_end, start_inclusive=False
        )
        after_hours_bars = len(post_bars)
        if not post_bars:
            reasons["after_hours_move"] = "no after-hours bars"
        else:
            after_hours_last_ts = post_bars[-1].ts_utc
            after_hours_move = _pct_change(post_bars[-1].close, pre_close)
            if after_hours_move is None:
                reasons["after_hours_move"] = (
                    "pre_event_close_not_positive"
                    if pre_close is None
                    else "close_unavailable"
                )
        reasons["premarket_move"] = (
            "not applicable for an AFTER_MARKET release"
            if session is EventSession.AFTER_MARKET
            else "not applicable for an assumed AFTER_MARKET release"
        )
        session_date = _next_session_date(
            bars,
            event_date_et,
            regular_open_et=regular_open_et,
            regular_close_et=regular_close_et,
        )

    partial = {
        "after_hours_move": after_hours_move,
        "after_hours_last_ts": after_hours_last_ts,
        "after_hours_bars": after_hours_bars,
        "premarket_move": premarket_move,
        "premarket_last_ts": premarket_last_ts,
        "premarket_bars": premarket_bars,
        "bars_used": len(bars),
    }

    if session_date is None:
        reasons["open"] = "no next-session regular bars"
        return _unavailable_intraday(
            session=session,
            basis=basis,
            confidence=confidence,
            event_ts_utc=event_ts_utc,
            event_date_et=event_date_et,
            pre_event_close=pre_close,
            windows_min=windows_min,
            reason="no next-session regular bars",
            extra={**partial, "reasons": reasons},
        )

    open_bar = _open_bar_on(
        bars,
        session_date,
        regular_open_et=regular_open_et,
        regular_close_et=regular_close_et,
    )
    if open_bar is None:
        reason = "no regular-session bar on the reaction day"
        reasons["open"] = reason
        return _unavailable_intraday(
            session=session,
            basis=basis,
            confidence=confidence,
            event_ts_utc=event_ts_utc,
            event_date_et=event_date_et,
            pre_event_close=pre_close,
            windows_min=windows_min,
            reason=reason,
            extra={
                **partial,
                "session_date_et": session_date,
                "reasons": reasons,
            },
        )

    open_price = _finite(open_bar.open)
    if open_price is None:
        reasons["open_price"] = "open_unavailable"
    gap_at_open = _pct_change(open_price, pre_close)
    if gap_at_open is None:
        reasons["gap_at_open"] = (
            "pre_event_close_not_positive"
            if pre_close is None
            else "open_unavailable"
        )

    # The windows are measured against ``pre_event_close``; when that base is
    # missing the window still names the BAR it found (the price is real) but
    # carries the base's own reason rather than inventing a move of zero.
    reference_reason = None if pre_close is not None else reasons.get(
        "pre_event_close", "pre_event_close_not_supplied"
    )
    session_end = _et_boundary(session_date, regular_close_et)
    windows = _window_marks(
        bars,
        anchor_ts=open_bar.ts_utc,
        reference_price=pre_close,
        windows_min=windows_min,
        session_end=session_end,
        reference_reason=reference_reason,
    )
    for k, window in windows.items():
        if window.move is None and window.reason:
            reasons[f"window_{k}m"] = window.reason

    max_move, max_reason = _first_hour_max_move(
        bars, anchor_ts=open_bar.ts_utc, open_price=open_price
    )
    if max_reason:
        reasons["max_move_first_hour"] = max_reason
    volume_30m, volume_reason = _volume_first_30m(bars, anchor_ts=open_bar.ts_utc)
    if volume_reason:
        reasons["volume_first_30m"] = volume_reason
    avg_volume, avg_reason = _avg_prior_open_volume(
        prior_session_bars,
        session_date=session_date,
        regular_open_et=regular_open_et,
    )
    if avg_reason:
        reasons["avg_volume_first_30m_prior_5_days"] = avg_reason
    ratio = None
    if volume_30m is not None and avg_volume is not None and avg_volume > 0.0:
        ratio = _finite(volume_30m / avg_volume)
    else:
        reasons["volume_ratio_first_30m"] = (
            volume_reason or avg_reason or "volume_ratio_inputs_unavailable"
        )

    # The open-anchored sessions measure against the DAILY pre-event close,
    # which has no minute timestamp to report. Saying so beats leaving
    # ``reference_ts`` an unexplained null next to a populated
    # ``reference_price``.
    reasons["reference_ts"] = (
        "reference is the daily pre-event close, which has no minute timestamp"
    )

    return IntradayReaction(
        session=session,
        basis=basis,
        confidence=confidence,
        available=True,
        event_ts_utc=event_ts_utc,
        event_date_et=event_date_et,
        pre_event_close=pre_close,
        after_hours_move=after_hours_move,
        after_hours_last_ts=after_hours_last_ts,
        after_hours_bars=after_hours_bars,
        premarket_move=premarket_move,
        premarket_last_ts=premarket_last_ts,
        premarket_bars=premarket_bars,
        reference_price=pre_close,
        reference_ts=None,
        open_price=open_price,
        open_ts=open_bar.ts_utc,
        gap_at_open=gap_at_open,
        windows=windows,
        max_move_first_hour=max_move,
        volume_first_30m=volume_30m,
        avg_volume_first_30m_prior_5_days=avg_volume,
        volume_ratio_first_30m=ratio,
        session_date_et=session_date,
        bars_used=len(bars),
        reasons=reasons,
    )


def intraday_reaction_to_dict(result: IntradayReaction) -> dict:
    """One :class:`IntradayReaction` as JSON.

    ``basis`` and ``confidence`` travel with the numbers on purpose (§85):
    an UNKNOWN-session reaction is an assumed measurement and the UI cannot
    label it unless the payload says so.
    """
    return {
        "available": result.available,
        "session": _enum_value(result.session),
        "basis": result.basis,
        "confidence": result.confidence,
        "event_ts_utc": _ts_iso(result.event_ts_utc),
        "event_date_et": _date_iso(result.event_date_et),
        "session_date_et": _date_iso(result.session_date_et),
        "pre_event_close": result.pre_event_close,
        "after_hours_move": result.after_hours_move,
        "after_hours_last_ts": _ts_iso(result.after_hours_last_ts),
        "after_hours_bars": result.after_hours_bars,
        "premarket_move": result.premarket_move,
        "premarket_last_ts": _ts_iso(result.premarket_last_ts),
        "premarket_bars": result.premarket_bars,
        "reference_price": result.reference_price,
        "reference_ts": _ts_iso(result.reference_ts),
        "open_price": result.open_price,
        "open_ts": _ts_iso(result.open_ts),
        "gap_at_open": result.gap_at_open,
        "windows": {
            f"{k}m": {
                "minutes": window.minutes,
                "target_ts_utc": _ts_iso(window.target_ts_utc),
                "bar_ts_utc": _ts_iso(window.bar_ts_utc),
                "price": window.price,
                "move": window.move,
                "lag_seconds": window.lag_seconds,
                "reason": window.reason,
            }
            for k, window in sorted(result.windows.items())
        },
        "max_move_first_hour": result.max_move_first_hour,
        "volume_first_30m": result.volume_first_30m,
        "avg_volume_first_30m_prior_5_days": (
            result.avg_volume_first_30m_prior_5_days
        ),
        "volume_ratio_first_30m": result.volume_ratio_first_30m,
        "bars_used": result.bars_used,
        "reasons": dict(result.reasons),
        "model_version": result.model_version,
    }


# ---------------------------------------------------------------------------
# §20 — the replay bundle
# ---------------------------------------------------------------------------


def build_event_replay(
    *,
    event_id: int | None = None,
    event_key: str | None = None,
    event_type: EventType | str | None = None,
    ticker: str | None = None,
    date_et: date | None = None,
    session: EventSession | None = None,
    status: EventStatus | str | None = None,
    source_url: str | None = None,
    release_ts_utc: datetime | None = None,
    source_name: str | None = None,
    fundamentals_ref: Mapping[str, object] | None = None,
    price_context_ref: Mapping[str, object] | None = None,
    news_window_ref: Mapping[str, object] | None = None,
    intraday: IntradayReaction | None = None,
    intraday_reason: str | None = None,
    daily: ReactionResult | None = None,
    daily_dict: Mapping[str, object] | None = None,
    abnormal: AbnormalResult | None = None,
    abnormal_dict: Mapping[str, object] | None = None,
    data_freshness: Mapping[str, object] | None = None,
    reasons: Mapping[str, str] | None = None,
) -> EventReplay:
    """Compose the §20 replay from pieces the caller already resolved.

    PURE COMPOSITION: nothing here fetches, and nothing here computes a
    number it was not handed. The gateway resolves the event row, the
    point-in-time refs, the minute bars and the daily reaction; this function
    only decides the SHAPE, which is the §20 order — information available
    before the event, the release, the immediate reaction, the subsequent
    reaction — and attaches the §91 provenance labels.

    Each block of ``information_before`` is a REFERENCE, not a copy: a
    fundamentals snapshot ref, a price-context ref and a news-window ref, each
    optional. An absent one becomes ``{"available": false, "reason": ...}``
    rather than a missing key, because a UI that cannot distinguish "no news
    block was built" from "there was no news" will print the wrong sentence.

    ``daily`` may be passed as the frozen :class:`ReactionResult` or, when the
    gateway has already rendered it with its own ``reaction_to_dict``, as
    ``daily_dict`` — the rendered form wins, so the pure layer never
    re-implements the gateway's JSON shape.
    """
    all_reasons: dict[str, str] = dict(reasons or {})

    information_before: dict[str, object] = {}
    for name, ref, missing_reason in (
        (
            "fundamentals",
            fundamentals_ref,
            "fundamentals snapshot not supplied for this replay",
        ),
        (
            "price_context",
            price_context_ref,
            "pre-event price context not supplied for this replay",
        ),
        (
            "news_window",
            news_window_ref,
            "news window not yet available (Phase D)",
        ),
    ):
        if ref is None:
            information_before[name] = {
                "available": False,
                "reason": missing_reason,
            }
            all_reasons[f"information_before.{name}"] = missing_reason
        else:
            entry = dict(ref)
            entry.setdefault("available", True)
            information_before[name] = entry

    release: dict[str, object] = {
        "timestamp_utc": _ts_iso(release_ts_utc),
        "timestamp_et": (
            release_ts_utc.astimezone(EASTERN).isoformat()
            if release_ts_utc is not None
            else None
        ),
        "session": _enum_value(session),
        "source_name": source_name,
        "source_url": source_url,
    }
    if release_ts_utc is None:
        all_reasons["release.timestamp"] = "release timestamp not supplied"

    if intraday is not None:
        immediate: dict[str, object] = intraday_reaction_to_dict(intraday)
    else:
        reason = intraday_reason or "minute bars not available for this event"
        immediate = {"available": False, "reason": reason}
        all_reasons["immediate_reaction"] = reason
    immediate["provenance"] = "QUANT"

    subsequent: dict[str, object] = {}
    if daily_dict is not None:
        subsequent["reaction"] = dict(daily_dict)
        subsequent["available"] = bool(daily_dict.get("bars_available", True))
    elif daily is not None:
        subsequent["reaction"] = {
            "event_date_et": _date_iso(daily.event_date_et),
            "session": _enum_value(daily.session),
            "basis": daily.basis,
            "bars_available": daily.bars_available,
            "pre_event_close": daily.pre_event_close,
            "pre_event_date": _date_iso(daily.pre_event_date),
            "react_open": daily.react_open,
            "react_close": daily.react_close,
            "react_date": _date_iso(daily.react_date),
            "gap_return": daily.gap_return,
            "returns": {f"{k}D": v for k, v in sorted(daily.returns.items())},
            "abs_returns": {
                f"{k}D": v for k, v in sorted(daily.abs_returns.items())
            },
            "max_favorable_excursion": daily.max_favorable_excursion,
            "max_adverse_excursion": daily.max_adverse_excursion,
            "reasons": dict(daily.reasons),
        }
        subsequent["available"] = daily.bars_available
    else:
        reason = "daily bars not available for this event"
        subsequent = {"available": False, "reason": reason}
        all_reasons["subsequent_reaction"] = reason

    if abnormal_dict is not None:
        subsequent["abnormal"] = dict(abnormal_dict)
    elif abnormal is not None:
        subsequent["abnormal"] = {
            "benchmark_available": abnormal.benchmark_available,
            "basis": abnormal.basis,
            "abnormal": {f"{k}D": v for k, v in sorted(abnormal.abnormal.items())},
            "abnormal_gap": abnormal.abnormal_gap,
            "benchmark_returns": {
                f"{k}D": v for k, v in sorted(abnormal.benchmark_returns.items())
            },
            "benchmark_gap_return": abnormal.benchmark_gap_return,
            "reasons": dict(abnormal.reasons),
        }
    else:
        subsequent["abnormal"] = {
            "available": False,
            "reason": "benchmark comparison not supplied for this replay",
        }
    subsequent["provenance"] = "QUANT"

    return EventReplay(
        event_id=event_id,
        event_key=event_key,
        event_type=event_type if isinstance(event_type, EventType) else event_type,
        ticker=ticker,
        date_et=date_et,
        session=session,
        status=status,
        source_url=source_url,
        information_before=information_before,
        release=release,
        immediate_reaction=immediate,
        subsequent_reaction=subsequent,
        data_freshness=dict(data_freshness or {}),
        provenance={
            "release": "DATA",
            "minute_bars": "DATA",
            "daily_bars": "DATA",
            "metrics": "QUANT",
        },
        reasons=all_reasons,
    )


# ---------------------------------------------------------------------------
# §60 — the multi-event history table
# ---------------------------------------------------------------------------


def _unavailable_cell(reason: str) -> dict:
    """The one shape every un-computable §60 cell takes."""
    return {"available": False, "reason": reason}


def history_table(previous_events: Sequence[Mapping[str, object]]) -> dict:
    """The §60 "LAST 4 / 8 / 12 EARNINGS" table, oldest-event-first.

    Each input mapping is one past event as the gateway resolved it:
    ``date_et``/``session``/``status`` plus a ``reaction``
    (:class:`ReactionResult`), an optional ``abnormal``
    (:class:`AbnormalResult`) and an optional ``intraday``
    (:class:`IntradayReaction`). Rows come back in the §60 column order with
    every missing number an explicit absence:

    - ``intraday_30m`` — the ``+30m`` move when minute bars were stored for
      that event, otherwise ``{"available": false, "reason": ...}``. Minute
      bars are backfilled ONE EVENT AT A TIME on user action, so most rows
      of a freshly loaded table legitimately have none.
    - ``eps_surprise`` / ``rev_surprise`` — always
      :data:`CONSENSUS_UNAVAILABLE_REASON` (§33/§98: no consensus vendor at
      any tier, so this is unavailable at every instant, not merely
      un-backtestable).
    - ``implied_move`` — always :data:`IMPLIED_MOVE_UNAVAILABLE_REASON`
      (§36/§60, Phase I).
    - ``actual_move_abs`` — ``|ret_1d|``, the half of the §60
      implied-vs-actual pair that IS computable today.

    ``summary`` carries the §19/§64 distribution over the same events via
    :func:`reaction.history_stats` — the same function the price tab uses, so
    the two views can never disagree about the median.
    """
    rows: list[dict] = []
    reactions: list[ReactionResult] = []

    for entry in previous_events:
        reaction = entry.get("reaction")
        abnormal = entry.get("abnormal")
        intraday = entry.get("intraday")

        if isinstance(reaction, ReactionResult):
            reactions.append(reaction)
            gap = reaction.gap_return
            ret_1d = reaction.returns.get(1)
            ret_5d = reaction.returns.get(5)
            bars_available = reaction.bars_available
            reaction_reasons = dict(reaction.reasons)
        else:
            gap = ret_1d = ret_5d = None
            bars_available = False
            reaction_reasons = {"bars": "reaction not supplied for this event"}

        abnormal_1d = (
            abnormal.abnormal.get(1)
            if isinstance(abnormal, AbnormalResult)
            else None
        )

        if isinstance(intraday, IntradayReaction) and intraday.available:
            window = intraday.windows.get(30)
            if window is not None and window.move is not None:
                intraday_30m: dict = {
                    "available": True,
                    "move": window.move,
                    "basis": intraday.basis,
                    "confidence": intraday.confidence,
                    "bar_ts_utc": _ts_iso(window.bar_ts_utc),
                }
            else:
                intraday_30m = _unavailable_cell(
                    (window.reason if window is not None else None)
                    or "no +30m minute bar for this event"
                )
        elif isinstance(intraday, IntradayReaction):
            intraday_30m = _unavailable_cell(
                intraday.reasons.get("bars", "minute bars unavailable")
            )
        else:
            intraday_30m = _unavailable_cell("no minute bars stored for this event")

        row_reasons: dict[str, str] = {}
        if not bars_available:
            row_reasons["reaction"] = reaction_reasons.get(
                "bars", "reaction_unavailable"
            )
        for label, value in (("gap", gap), ("ret_1d", ret_1d), ("ret_5d", ret_5d)):
            if value is None and label not in row_reasons:
                key = {"gap": "gap_return", "ret_1d": "return_1D", "ret_5d": "return_5D"}[label]
                row_reasons[label] = reaction_reasons.get(
                    key, row_reasons.get("reaction", "not_measured")
                )
        if abnormal_1d is None:
            if isinstance(abnormal, AbnormalResult):
                row_reasons["abnormal_1d"] = abnormal.reasons.get(
                    "abnormal_1D",
                    abnormal.reasons.get("benchmark", "benchmark_unavailable"),
                )
            else:
                row_reasons["abnormal_1d"] = "benchmark comparison not supplied"

        rows.append(
            {
                "event_id": entry.get("event_id"),
                "event_key": entry.get("event_key"),
                "date_et": _date_iso(_as_date(entry.get("date_et"))),
                "session": _enum_value(entry.get("session")),
                "status": _enum_value(entry.get("status")),
                "gap": gap,
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "abnormal_1d": abnormal_1d,
                "intraday_30m": intraday_30m,
                "eps_surprise": _unavailable_cell(CONSENSUS_UNAVAILABLE_REASON),
                "rev_surprise": _unavailable_cell(CONSENSUS_UNAVAILABLE_REASON),
                "implied_move": _unavailable_cell(IMPLIED_MOVE_UNAVAILABLE_REASON),
                "actual_move_abs": abs(ret_1d) if ret_1d is not None else None,
                "bars_available": bars_available,
                "reasons": row_reasons,
            }
        )

    rows.sort(key=lambda row: (row["date_et"] is None, row["date_et"] or ""))

    summary: dict[str, dict] = {}
    for horizon in (1, 5):
        windows = history_stats(reactions, horizon=horizon)
        summary[f"{horizon}D"] = {
            label: _history_stats_to_dict(stats)
            for label, stats in windows.items()
        }

    return {
        "rows": rows,
        "n_rows": len(rows),
        "summary": summary,
        "columns": list(HISTORY_COLUMNS),
        "provenance": {"bars": "DATA", "metrics": "QUANT"},
        "not_backtestable": ["eps_surprise", "rev_surprise", "implied_move"],
        "model_version": REPLAY_MODEL_VERSION,
    }


#: §60 column order, carried in the payload so the UI renders the table the
#: spec describes without hardcoding an order that can drift from this module.
HISTORY_COLUMNS: tuple[str, ...] = (
    "date_et",
    "session",
    "status",
    "eps_surprise",
    "rev_surprise",
    "implied_move",
    "actual_move_abs",
    "gap",
    "intraday_30m",
    "ret_1d",
    "ret_5d",
    "abnormal_1d",
)


def _history_stats_to_dict(stats: HistoryStats) -> dict:
    """One :class:`HistoryStats` window as JSON — same keys as the price tab."""
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


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


# ---------------------------------------------------------------------------
# §15 — previous-event linkage
# ---------------------------------------------------------------------------

#: Reason recorded when an event genuinely has no comparable predecessor —
#: the first print in the stored history, or a type that has none by
#: construction (MARKET_HOLIDAY). An empty string here would be read by the
#: gateway as "not computed yet" and re-run forever.
NO_PREVIOUS_REASON = "no previous comparable event in the stored history"

#: Reason recorded for an event that is excluded from linkage entirely.
CANCELED_REASON = "canceled events are not linked to a previous comparable"


def link_previous_events(
    events: Sequence,
) -> list[tuple[str, str | None, str | None]]:
    """``[(event_key, previous_key|None, reason)]`` for a batch of events (§15).

    A thin, deterministic batch wrapper over
    :func:`models.previous_comparable` — the matching RULES stay in one place
    and this function only decides which events to ask about and in which
    order. Three properties the gateway depends on:

    1. **The pool is the batch.** Every event is matched against every other
       event handed in, so a caller passing one ticker's earnings history
       gets that ticker's chain; the type check inside
       ``previous_comparable`` means a batch mixing EARNINGS and
       FOMC_DECISION rows still never links across types.
    2. **CANCELED is excluded on both sides.** A withdrawn event is neither a
       predecessor (``previous_comparable`` already drops it from the pool)
       nor a subject (it gets ``CANCELED_REASON`` and no link) — a canceled
       print has no reaction to compare against.
    3. **An ESTIMATED row may point at a CONFIRMED predecessor**, which is
       the normal case for an upcoming earnings card: the date ahead is a
       guess, but the print behind it is a fact, and that is exactly the
       comparison §15 wants. The reverse never happens for EARNINGS —
       ``previous_comparable`` restricts the pool to CONFIRMED/REVISED.

    Results are ordered by ``(scheduled_at, event_key)`` so a persistence
    loop writes in a stable order, and every ``None`` predecessor carries a
    reason rather than an empty cell.
    """
    ordered = sorted(events, key=lambda e: (e.scheduled_at, e.event_key))
    out: list[tuple[str, str | None, str | None]] = []
    for event in ordered:
        if event.status is EventStatus.CANCELED:
            out.append((event.event_key, None, CANCELED_REASON))
            continue
        previous, reason = previous_comparable(event, ordered)
        if previous is None:
            out.append((event.event_key, None, NO_PREVIOUS_REASON))
            continue
        out.append((event.event_key, previous.event_key, reason))
    return out
