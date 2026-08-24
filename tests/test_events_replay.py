"""Event replay — intraday reaction, §20 bundle, §60 history, §15 linkage
(event spec §15, §17, §19, §20, §60, §85, §96; Phase C unit U2).

Every number here is hand-checkable: the fixtures price bars at round values
so a move like ``110/100 - 1`` reads off the page, and no test asserts a
value the module computed for itself. Five contracts are pinned:

1. **The anchor is chosen by the session** (§17) — AMC anchors on the NEXT
   session's open, BMO on the SAME day's open, DURING_MARKET on the last bar
   at or before the release, UNKNOWN on the AMC rule but flagged as an
   assumption. Choosing the wrong anchor silently reports the wrong move,
   so each rule is asserted on the bar timestamp it selected, not only on
   the number.
2. **ET is the calendar** — 09:30/16:00/04:00/20:00 are wall-clock ET
   boundaries. Tested on both sides of both DST transitions, because a
   fixed-UTC-offset implementation is right for half the year.
3. **Absence is a value** (§85) — a missing window, a missing next session,
   a non-positive reference price and an unsupplied ``pre_event_close`` each
   yield ``None`` plus a reason, never a zero, a NaN or an ``inf``.
4. **§60 honesty** — EPS/revenue surprise and implied move are ALWAYS
   explicit unavailability with their fixed reason strings; ``intraday_30m``
   is unavailable unless minute bars were actually stored.
5. **§15 linkage never crosses types** and CANCELED is excluded on both
   sides.
"""
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from libs.trading_core.events import (
    DailyBar,
    Event,
    EventReplay,
    IntradayReaction,
    IntradayWindow,
    MinuteBar,
    abnormal_vs,
    build_event_replay,
    event_reaction,
    history_table,
    intraday_reaction,
    intraday_reaction_to_dict,
    link_previous_events,
)
from libs.trading_core.events.replay import (
    BASIS_INTRADAY_AFTER_MARKET,
    BASIS_INTRADAY_BEFORE_MARKET,
    BASIS_INTRADAY_DURING_MARKET,
    BASIS_INTRADAY_UNKNOWN,
    CANCELED_REASON,
    CONSENSUS_UNAVAILABLE_REASON,
    HISTORY_COLUMNS,
    IMPLIED_MOVE_UNAVAILABLE_REASON,
    NO_PREVIOUS_REASON,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

AMC = EventSession.AFTER_MARKET
BMO = EventSession.BEFORE_MARKET
DURING = EventSession.DURING_MARKET
UNKNOWN = EventSession.UNKNOWN


# ---------------------------------------------------------------------------
# Fixtures — minute bars built in ET so the rules are read in ET
# ---------------------------------------------------------------------------


def et(d: date, hour: int, minute: int) -> datetime:
    """An ET wall-clock instant, as aware UTC (DST handled by zoneinfo)."""
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET).astimezone(UTC)


def mb(d: date, hour: int, minute: int, close: float, *, open_: float | None = None,
       volume: float = 1000.0) -> MinuteBar:
    """One minute bar whose OHLC hangs off its close, stamped in ET."""
    o = close if open_ is None else open_
    return MinuteBar(
        ts_utc=et(d, hour, minute),
        open=o,
        high=max(o, close),
        low=min(o, close),
        close=close,
        volume=volume,
    )


#: Release day (Thu) and reaction day (Fri) of a clean, non-DST week.
D1 = date(2026, 1, 8)
D2 = date(2026, 1, 9)


def amc_bars() -> list[MinuteBar]:
    """AMC fixture: pre-close 100 (daily), 3 after-hours bars, next session.

    After-hours prints 110 (a +10% reaction); the next session opens at 105
    (+5% gap) and walks 106 / 107 / 108 at +5m / +30m / +60m so each window
    is a different, hand-checkable number.
    """
    return [
        mb(D1, 15, 59, 100.0),
        mb(D1, 16, 5, 108.0),
        mb(D1, 16, 30, 109.0),
        mb(D1, 19, 30, 110.0),
        mb(D2, 9, 30, 105.0, open_=105.0),
        mb(D2, 9, 35, 106.0),
        mb(D2, 10, 0, 107.0),
        mb(D2, 10, 30, 108.0),
        mb(D2, 11, 0, 120.0),
    ]


def bmo_bars() -> list[MinuteBar]:
    """BMO fixture: pre-market prints 110, the same day opens 105 and walks."""
    return [
        mb(D1, 4, 30, 108.0),
        mb(D1, 8, 0, 109.0),
        mb(D1, 9, 29, 110.0),
        mb(D1, 9, 30, 105.0, open_=105.0),
        mb(D1, 9, 35, 106.0),
        mb(D1, 10, 0, 107.0),
        mb(D1, 10, 30, 108.0),
    ]


def during_bars() -> list[MinuteBar]:
    """DURING fixture: release at 12:00 with the anchor bar at 100."""
    return [
        mb(D1, 11, 30, 99.0),
        mb(D1, 12, 0, 100.0),
        mb(D1, 12, 5, 102.0),
        mb(D1, 12, 30, 105.0),
        mb(D1, 13, 0, 110.0),
    ]


AMC_TS = et(D1, 16, 2)
BMO_TS = et(D1, 7, 0)
DURING_TS = et(D1, 12, 0)


# ---------------------------------------------------------------------------
# 1. MinuteBar — the value object refuses to guess a timezone
# ---------------------------------------------------------------------------


# The no-I/O and no-numerics invariants for THIS module are enforced
# package-wide by tests/test_pure_layer_boundary.py, which walks every
# module under libs/trading_core/ — a per-file copy here protected this
# one file and left sixty-six others to the habit of copying a test.


def test_minute_bar_refuses_a_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        MinuteBar(
            ts_utc=datetime(2026, 1, 8, 16, 5),
            open=1.0, high=1.0, low=1.0, close=1.0,
        )


def test_minute_bar_normalises_an_et_timestamp_to_utc():
    bar = MinuteBar(
        ts_utc=datetime(2026, 1, 8, 16, 5, tzinfo=ET),
        open=1.0, high=1.0, low=1.0, close=1.0,
    )
    assert bar.ts_utc.tzinfo is not None
    assert bar.ts_utc.utcoffset() == timedelta(0)
    # 16:05 EST is 21:05Z.
    assert (bar.ts_utc.hour, bar.ts_utc.minute) == (21, 5)


def test_minute_bar_ts_et_round_trips_to_the_wall_clock():
    bar = mb(D1, 16, 5, 100.0)
    assert (bar.ts_et.hour, bar.ts_et.minute) == (16, 5)
    assert bar.ts_et.date() == D1


def test_minute_bar_is_frozen():
    bar = mb(D1, 16, 5, 100.0)
    with pytest.raises(Exception):
        bar.close = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


def test_intraday_reaction_refuses_a_naive_event_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        intraday_reaction(
            amc_bars(),
            event_ts_utc=datetime(2026, 1, 8, 21, 2),
            session=AMC,
            pre_event_close=100.0,
        )


def test_intraday_reaction_refuses_unsorted_bars():
    bars = amc_bars()
    bars[2], bars[3] = bars[3], bars[2]
    with pytest.raises(ValueError, match="strictly increasing"):
        intraday_reaction(
            bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
        )


def test_intraday_reaction_refuses_duplicate_timestamps():
    bars = [mb(D1, 16, 5, 108.0), mb(D1, 16, 5, 109.0)]
    with pytest.raises(ValueError, match="strictly increasing"):
        intraday_reaction(
            bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
        )


def test_intraday_reaction_refuses_a_non_positive_window():
    with pytest.raises(ValueError, match=">= 1"):
        intraday_reaction(
            amc_bars(),
            event_ts_utc=AMC_TS,
            session=AMC,
            pre_event_close=100.0,
            windows_min=(0, 30),
        )


def test_intraday_reaction_with_no_bars_is_unavailable_with_a_reason():
    result = intraday_reaction(
        [], event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.available is False
    assert result.reasons["bars"] == "no_minute_bars_available"
    assert all(w.move is None for w in result.windows.values())
    assert result.basis == BASIS_INTRADAY_AFTER_MARKET


# ---------------------------------------------------------------------------
# 3. AFTER_MARKET — the after-hours window and the next-session anchor
# ---------------------------------------------------------------------------


def test_amc_after_hours_move_uses_the_last_bar_before_2000_et():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    # last after-hours bar closes 110 vs the 100 pre-event close.
    assert result.after_hours_move == pytest.approx(0.10)
    assert result.after_hours_last_ts == et(D1, 19, 30)
    assert result.after_hours_bars == 3


def test_amc_after_hours_window_excludes_the_release_bar_itself():
    """The span is ``(event_ts, 20:00]`` — OPEN at the left edge."""
    bars = [mb(D1, 16, 2, 999.0)] + amc_bars()
    bars.sort(key=lambda b: b.ts_utc)
    result = intraday_reaction(
        bars, event_ts_utc=et(D1, 16, 2), session=AMC, pre_event_close=100.0
    )
    assert result.after_hours_bars == 3  # the 16:02 bar is not counted
    assert result.after_hours_move == pytest.approx(0.10)


def test_amc_after_hours_window_stops_at_2000_et():
    bars = amc_bars() + [mb(D1, 20, 30, 200.0)]
    bars.sort(key=lambda b: b.ts_utc)
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.after_hours_last_ts == et(D1, 19, 30)
    assert result.after_hours_move == pytest.approx(0.10)


def test_amc_with_no_after_hours_bars_says_no_after_hours_bars():
    bars = [b for b in amc_bars() if b.ts_et.date() != D1 or b.ts_et.hour < 16]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.after_hours_move is None
    assert result.reasons["after_hours_move"] == "no after-hours bars"
    assert result.after_hours_bars == 0
    # The next-session anchor is unaffected — the two are independent.
    assert result.gap_at_open == pytest.approx(0.05)


def test_amc_gap_at_open_is_next_session_open_over_pre_close():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.open_ts == et(D2, 9, 30)
    assert result.open_price == pytest.approx(105.0)
    assert result.gap_at_open == pytest.approx(0.05)  # 105/100 - 1
    assert result.session_date_et == D2
    assert result.basis == BASIS_INTRADAY_AFTER_MARKET
    assert result.confidence == "high"


def test_amc_windows_measure_against_the_pre_event_close():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.move(5) == pytest.approx(0.06)   # 106/100 - 1
    assert result.move(30) == pytest.approx(0.07)  # 107/100 - 1
    assert result.move(60) == pytest.approx(0.08)  # 108/100 - 1


def test_amc_window_uses_the_last_bar_at_or_before_the_mark_and_reports_lag():
    """A sparse tape gives an older bar — the lag is reported, not hidden."""
    bars = [
        mb(D1, 15, 59, 100.0),
        mb(D2, 9, 30, 105.0, open_=105.0),
        mb(D2, 9, 40, 106.0),   # newest bar at or before +30m
        mb(D2, 10, 30, 108.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0,
        windows_min=(30,),
    )
    window = result.windows[30]
    assert window.bar_ts_utc == et(D2, 9, 40)
    assert window.target_ts_utc == et(D2, 10, 0)
    assert window.lag_seconds == 20 * 60
    assert window.move == pytest.approx(0.06)


def test_amc_window_with_no_bar_at_all_is_none_with_a_reason():
    bars = [mb(D1, 15, 59, 100.0), mb(D2, 9, 30, 105.0, open_=105.0)]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0,
        windows_min=(5,),
    )
    # The open bar itself IS at or before +5m, so it fills the window at the
    # open price — the reaction is real, it just has not moved off the open.
    assert result.windows[5].bar_ts_utc == et(D2, 9, 30)
    assert result.windows[5].move == pytest.approx(0.05)


def test_amc_windows_never_reach_past_the_regular_close():
    """A +60m window may not be filled from an after-hours bar."""
    bars = [
        mb(D1, 15, 59, 100.0),
        mb(D2, 15, 45, 105.0, open_=105.0),  # a late (half-day-ish) open
        mb(D2, 18, 0, 200.0),                # after-hours print
    ]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0,
        windows_min=(60,),
    )
    assert result.windows[60].bar_ts_utc == et(D2, 15, 45)
    assert result.windows[60].move == pytest.approx(0.05)


def test_amc_next_session_skips_a_day_with_only_extended_hours_bars():
    """A day whose only prints are after-hours is not a session open."""
    gap_day = date(2026, 1, 9)
    real_day = date(2026, 1, 12)
    bars = [
        mb(D1, 15, 59, 100.0),
        mb(D1, 17, 0, 110.0),
        mb(gap_day, 18, 0, 111.0),          # extended-hours only
        mb(real_day, 9, 30, 105.0, open_=105.0),
        mb(real_day, 10, 30, 107.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.session_date_et == real_day
    assert result.open_ts == et(real_day, 9, 30)


def test_amc_over_a_weekend_reacts_on_monday():
    friday = date(2026, 1, 9)
    monday = date(2026, 1, 12)
    bars = [
        mb(friday, 15, 59, 100.0),
        mb(friday, 17, 0, 110.0),
        mb(monday, 9, 30, 105.0, open_=105.0),
        mb(monday, 10, 30, 107.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=et(friday, 16, 5), session=AMC, pre_event_close=100.0
    )
    assert result.session_date_et == monday
    assert result.gap_at_open == pytest.approx(0.05)


def test_amc_with_no_next_session_bars_is_unavailable_with_a_named_reason():
    bars = [mb(D1, 15, 59, 100.0), mb(D1, 17, 0, 110.0)]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.available is False
    assert result.reasons["bars"] == "no next-session regular bars"
    # The after-hours half still survives — a partial answer beats a blank.
    assert result.after_hours_move == pytest.approx(0.10)
    assert result.after_hours_bars == 1
    assert result.gap_at_open is None


# ---------------------------------------------------------------------------
# 4. BEFORE_MARKET — the pre-market window and the same-day anchor
# ---------------------------------------------------------------------------


def test_bmo_premarket_move_uses_the_last_bar_before_0930_et():
    result = intraday_reaction(
        bmo_bars(), event_ts_utc=BMO_TS, session=BMO, pre_event_close=100.0
    )
    assert result.premarket_move == pytest.approx(0.10)  # 110/100 - 1
    assert result.premarket_last_ts == et(D1, 9, 29)
    assert result.premarket_bars == 3
    assert result.basis == BASIS_INTRADAY_BEFORE_MARKET


def test_bmo_premarket_window_starts_at_0400_et():
    bars = [mb(D1, 3, 30, 200.0)] + bmo_bars()
    bars.sort(key=lambda b: b.ts_utc)
    result = intraday_reaction(
        bars, event_ts_utc=et(D1, 3, 0), session=BMO, pre_event_close=100.0
    )
    assert result.premarket_bars == 3  # the 03:30 bar is outside 04:00-09:30
    assert result.premarket_move == pytest.approx(0.10)


def test_bmo_premarket_window_excludes_the_0930_open_bar():
    result = intraday_reaction(
        bmo_bars(), event_ts_utc=BMO_TS, session=BMO, pre_event_close=100.0
    )
    assert result.premarket_last_ts == et(D1, 9, 29)
    assert result.open_ts == et(D1, 9, 30)


def test_bmo_reacts_on_the_release_day_not_the_next():
    result = intraday_reaction(
        bmo_bars(), event_ts_utc=BMO_TS, session=BMO, pre_event_close=100.0
    )
    assert result.session_date_et == D1
    assert result.open_price == pytest.approx(105.0)
    assert result.gap_at_open == pytest.approx(0.05)
    assert result.move(5) == pytest.approx(0.06)
    assert result.move(30) == pytest.approx(0.07)
    assert result.move(60) == pytest.approx(0.08)


def test_bmo_with_no_premarket_bars_says_no_premarket_bars():
    bars = [b for b in bmo_bars() if b.ts_et.hour >= 9 and b.ts_et.minute >= 30
            or b.ts_et.hour >= 10]
    result = intraday_reaction(
        bars, event_ts_utc=BMO_TS, session=BMO, pre_event_close=100.0
    )
    assert result.premarket_move is None
    assert result.reasons["premarket_move"] == "no pre-market bars"
    assert result.gap_at_open == pytest.approx(0.05)


def test_open_anchored_sessions_explain_the_absent_reference_timestamp():
    """``reference_price`` is a DAILY close, so it has no minute stamp."""
    for bars, ts, session in (
        (amc_bars(), AMC_TS, AMC),
        (bmo_bars(), BMO_TS, BMO),
        (amc_bars(), AMC_TS, UNKNOWN),
    ):
        result = intraday_reaction(
            bars, event_ts_utc=ts, session=session, pre_event_close=100.0
        )
        assert result.reference_price == pytest.approx(100.0)
        assert result.reference_ts is None
        assert "no minute timestamp" in result.reasons["reference_ts"]


def test_during_market_reference_timestamp_is_a_real_bar():
    """The DURING anchor IS a minute bar, so it has a stamp and no reason."""
    result = intraday_reaction(
        during_bars(), event_ts_utc=DURING_TS, session=DURING,
        pre_event_close=90.0,
    )
    assert result.reference_ts == et(D1, 12, 0)
    assert "reference_ts" not in result.reasons


def test_bmo_marks_after_hours_as_not_applicable():
    result = intraday_reaction(
        bmo_bars(), event_ts_utc=BMO_TS, session=BMO, pre_event_close=100.0
    )
    assert result.after_hours_move is None
    assert "not applicable" in result.reasons["after_hours_move"]


def test_bmo_with_no_regular_bars_on_the_release_day_is_unavailable():
    bars = [mb(D1, 4, 30, 108.0), mb(D1, 9, 29, 110.0)]
    result = intraday_reaction(
        bars, event_ts_utc=BMO_TS, session=BMO, pre_event_close=100.0
    )
    assert result.available is False
    assert result.reasons["bars"] == "no regular-session bar on the reaction day"
    assert result.premarket_move == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 5. DURING_MARKET — the release-anchored rule
# ---------------------------------------------------------------------------


def test_during_market_anchors_on_the_last_bar_at_or_before_the_release():
    result = intraday_reaction(
        during_bars(), event_ts_utc=DURING_TS, session=DURING,
        pre_event_close=90.0,
    )
    assert result.basis == BASIS_INTRADAY_DURING_MARKET
    assert result.reference_ts == et(D1, 12, 0)
    assert result.reference_price == pytest.approx(100.0)


def test_during_market_windows_measure_against_the_reference_price():
    result = intraday_reaction(
        during_bars(), event_ts_utc=DURING_TS, session=DURING,
        pre_event_close=90.0,
    )
    assert result.move(5) == pytest.approx(0.02)   # 102/100 - 1
    assert result.move(30) == pytest.approx(0.05)  # 105/100 - 1
    assert result.move(60) == pytest.approx(0.10)  # 110/100 - 1


def test_during_market_has_no_opening_gap():
    result = intraday_reaction(
        during_bars(), event_ts_utc=DURING_TS, session=DURING,
        pre_event_close=90.0,
    )
    assert result.gap_at_open is None
    assert result.open_price is None
    assert (
        result.reasons["gap_at_open"]
        == "during_market_release_has_no_opening_gap"
    )


def test_during_market_with_no_bar_before_the_release_is_unavailable():
    bars = [mb(D1, 13, 0, 110.0)]
    result = intraday_reaction(
        bars, event_ts_utc=DURING_TS, session=DURING, pre_event_close=90.0
    )
    assert result.available is False
    assert result.reasons["bars"] == "no_bar_at_or_before_event"


def test_during_market_marks_both_extended_windows_not_applicable():
    result = intraday_reaction(
        during_bars(), event_ts_utc=DURING_TS, session=DURING,
        pre_event_close=90.0,
    )
    assert result.after_hours_move is None
    assert result.premarket_move is None
    assert "not applicable" in result.reasons["after_hours_move"]
    assert "not applicable" in result.reasons["premarket_move"]


# ---------------------------------------------------------------------------
# 6. UNKNOWN — the AMC rule, flagged as an assumption
# ---------------------------------------------------------------------------


def test_unknown_session_uses_the_after_market_anchor():
    unknown = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=UNKNOWN, pre_event_close=100.0
    )
    known = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert unknown.open_ts == known.open_ts
    assert unknown.gap_at_open == known.gap_at_open
    assert unknown.after_hours_move == known.after_hours_move
    assert {k: w.move for k, w in unknown.windows.items()} == {
        k: w.move for k, w in known.windows.items()
    }


def test_unknown_session_is_flagged_and_low_confidence():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=UNKNOWN, pre_event_close=100.0
    )
    assert result.basis == BASIS_INTRADAY_UNKNOWN
    assert result.confidence == "low"
    assert result.session is UNKNOWN


def test_known_sessions_are_high_confidence():
    for session, ts, bars in (
        (AMC, AMC_TS, amc_bars()),
        (BMO, BMO_TS, bmo_bars()),
        (DURING, DURING_TS, during_bars()),
    ):
        result = intraday_reaction(
            bars, event_ts_utc=ts, session=session, pre_event_close=95.0
        )
        assert result.confidence == "high", session


# ---------------------------------------------------------------------------
# 7. DST — the boundaries are ET wall clock, not a UTC offset
# ---------------------------------------------------------------------------


def test_open_boundary_holds_in_est_winter():
    """09:30 EST is 14:30Z — a 13:30Z bar is pre-market, not the open."""
    day = date(2026, 1, 8)
    bars = [
        MinuteBar(ts_utc=datetime(2026, 1, 8, 13, 30, tzinfo=UTC),
                  open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
        MinuteBar(ts_utc=datetime(2026, 1, 8, 14, 30, tzinfo=UTC),
                  open=105.0, high=105.0, low=105.0, close=105.0, volume=1.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=et(day, 7, 0), session=BMO, pre_event_close=100.0
    )
    assert result.open_ts == datetime(2026, 1, 8, 14, 30, tzinfo=UTC)
    assert result.premarket_bars == 1
    assert result.premarket_move == pytest.approx(0.10)


def test_open_boundary_holds_in_edt_summer():
    """09:30 EDT is 13:30Z — the SAME UTC instant is now the open."""
    day = date(2026, 7, 8)
    bars = [
        MinuteBar(ts_utc=datetime(2026, 7, 8, 12, 30, tzinfo=UTC),
                  open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
        MinuteBar(ts_utc=datetime(2026, 7, 8, 13, 30, tzinfo=UTC),
                  open=105.0, high=105.0, low=105.0, close=105.0, volume=1.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=et(day, 7, 0), session=BMO, pre_event_close=100.0
    )
    assert result.open_ts == datetime(2026, 7, 8, 13, 30, tzinfo=UTC)
    assert result.premarket_bars == 1


def test_after_hours_2000_boundary_holds_across_the_spring_transition():
    """DST starts Sun 2026-03-08; Mon 2026-03-09 is EDT, 20:00 ET = 00:00Z."""
    release = date(2026, 3, 9)
    react = date(2026, 3, 10)
    bars = [
        mb(release, 19, 59, 110.0),
        # 20:30 ET on 03-09 is 00:30Z on 03-10 — outside the window.
        mb(release, 20, 30, 200.0),
        mb(react, 9, 30, 105.0, open_=105.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=et(release, 16, 5), session=AMC, pre_event_close=100.0
    )
    assert result.after_hours_last_ts == et(release, 19, 59)
    assert result.after_hours_move == pytest.approx(0.10)
    assert result.session_date_et == react


def test_after_hours_2000_boundary_holds_across_the_autumn_transition():
    """DST ends Sun 2026-11-01; Mon 2026-11-02 is EST, 20:00 ET = 01:00Z."""
    release = date(2026, 11, 2)
    react = date(2026, 11, 3)
    bars = [
        mb(release, 19, 59, 110.0),
        mb(release, 20, 30, 200.0),
        mb(react, 9, 30, 105.0, open_=105.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=et(release, 16, 5), session=AMC, pre_event_close=100.0
    )
    assert result.after_hours_last_ts == et(release, 19, 59)
    assert result.after_hours_move == pytest.approx(0.10)


def test_et_date_of_an_after_midnight_utc_release_is_the_release_day():
    """21:05Z on Jan 8 is 16:05 ET the SAME day; 02:05Z Jan 9 is 21:05 ET Jan 8."""
    bars = amc_bars()
    result = intraday_reaction(
        bars,
        event_ts_utc=datetime(2026, 1, 9, 2, 5, tzinfo=UTC),
        session=AMC,
        pre_event_close=100.0,
    )
    assert result.event_date_et == D1
    assert result.session_date_et == D2


# ---------------------------------------------------------------------------
# 8. Absence is a value — no zeros, no NaN, no inf
# ---------------------------------------------------------------------------


def test_missing_pre_event_close_makes_every_move_none_with_a_reason():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=None
    )
    assert result.reasons["pre_event_close"] == "pre_event_close_not_supplied"
    assert result.after_hours_move is None
    assert result.gap_at_open is None
    assert all(w.move is None for w in result.windows.values())
    for k in result.windows:
        assert result.reasons[f"window_{k}m"] == "pre_event_close_not_supplied"


def test_non_positive_pre_event_close_labels_the_windows_distinctly():
    """"Not supplied" and "not positive" are different failures (§85)."""
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=0.0
    )
    for k in result.windows:
        assert result.reasons[f"window_{k}m"] == "pre_event_close_not_positive"
        # The BAR was still found — only the base is missing.
        assert result.windows[k].price is not None


def test_zero_pre_event_close_is_a_reason_not_a_division():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=0.0
    )
    assert result.reasons["pre_event_close"] == "pre_event_close_not_positive"
    assert result.gap_at_open is None
    assert result.pre_event_close is None


def test_negative_pre_event_close_is_a_reason():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=-5.0
    )
    assert result.reasons["pre_event_close"] == "pre_event_close_not_positive"
    assert result.after_hours_move is None


def test_nan_pre_event_close_never_propagates():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC,
        pre_event_close=float("nan"),
    )
    assert result.pre_event_close is None
    for value in (result.after_hours_move, result.gap_at_open):
        assert value is None


def test_during_market_zero_reference_price_is_a_reason():
    bars = [
        mb(D1, 12, 0, 0.0),
        mb(D1, 12, 30, 105.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=DURING_TS, session=DURING, pre_event_close=90.0
    )
    assert result.reference_price is None
    assert result.reasons["reference_price"] == "reference_price_not_positive"
    assert result.move(30) is None


def test_no_numeric_field_is_ever_nan_or_inf():
    """A sweep over the fixtures: every float that IS present is finite."""
    cases = [
        (amc_bars(), AMC_TS, AMC, 100.0),
        (amc_bars(), AMC_TS, UNKNOWN, 0.0),
        (bmo_bars(), BMO_TS, BMO, None),
        (during_bars(), DURING_TS, DURING, float("inf")),
        ([], AMC_TS, AMC, 100.0),
    ]
    for bars, ts, session, pre in cases:
        result = intraday_reaction(
            bars, event_ts_utc=ts, session=session, pre_event_close=pre
        )
        floats = [
            result.pre_event_close, result.after_hours_move,
            result.premarket_move, result.reference_price, result.open_price,
            result.gap_at_open, result.max_move_first_hour,
            result.volume_first_30m, result.avg_volume_first_30m_prior_5_days,
            result.volume_ratio_first_30m,
        ] + [w.move for w in result.windows.values()] + [
            w.price for w in result.windows.values()
        ]
        for value in floats:
            assert value is None or math.isfinite(value), (session, value)


def test_every_none_window_carries_a_reason():
    for bars, ts, session, pre in (
        (amc_bars(), AMC_TS, AMC, None),
        ([mb(D1, 15, 59, 100.0), mb(D2, 9, 30, 105.0, open_=105.0)],
         AMC_TS, AMC, 100.0),
        (during_bars(), DURING_TS, DURING, 90.0),
    ):
        result = intraday_reaction(
            bars, event_ts_utc=ts, session=session, pre_event_close=pre
        )
        for window in result.windows.values():
            if window.move is None:
                assert window.reason, window


# ---------------------------------------------------------------------------
# 9. First-hour max move and the volume windows
# ---------------------------------------------------------------------------


def test_max_move_first_hour_is_measured_against_the_open():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    # First hour from 09:30: closes 105/106/107/108 vs open 105 -> max 108/105-1.
    assert result.max_move_first_hour == pytest.approx(108.0 / 105.0 - 1.0)


def test_max_move_first_hour_ignores_bars_past_sixty_minutes():
    """The 11:00 bar at 120 is 90 minutes out and must not leak in."""
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.max_move_first_hour < 120.0 / 105.0 - 1.0


def test_max_move_first_hour_takes_the_absolute_value():
    bars = [
        mb(D1, 15, 59, 100.0),
        mb(D2, 9, 30, 100.0, open_=100.0),
        mb(D2, 9, 45, 80.0),   # -20%
        mb(D2, 10, 0, 105.0),  # +5%
    ]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.max_move_first_hour == pytest.approx(0.20)


def test_volume_first_30m_sums_the_opening_half_hour_only():
    bars = [
        mb(D1, 15, 59, 100.0, volume=999.0),
        mb(D2, 9, 30, 105.0, open_=105.0, volume=10.0),
        mb(D2, 9, 45, 106.0, volume=20.0),
        mb(D2, 10, 0, 107.0, volume=30.0),   # exactly +30m — included
        mb(D2, 10, 5, 108.0, volume=999.0),  # outside
    ]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.volume_first_30m == pytest.approx(60.0)


def test_avg_prior_volume_is_none_with_a_reason_when_not_supplied():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert result.avg_volume_first_30m_prior_5_days is None
    assert (
        result.reasons["avg_volume_first_30m_prior_5_days"]
        == "prior_session_bars_not_supplied"
    )
    assert result.volume_ratio_first_30m is None
    assert result.reasons["volume_ratio_first_30m"]


def test_avg_prior_volume_averages_the_prior_sessions_opening_thirty_minutes():
    prior_days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    prior = []
    for i, day in enumerate(prior_days):
        prior.append(mb(day, 9, 30, 100.0, volume=100.0 * (i + 1)))
        prior.append(mb(day, 9, 45, 100.0, volume=100.0 * (i + 1)))
        prior.append(mb(day, 11, 0, 100.0, volume=9999.0))  # outside the window
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0,
        prior_session_bars=prior,
    )
    # per-day opening volumes 200 / 400 / 600 -> mean 400.
    assert result.avg_volume_first_30m_prior_5_days == pytest.approx(400.0)


def test_avg_prior_volume_refuses_a_one_session_average():
    prior = [mb(date(2026, 1, 7), 9, 30, 100.0, volume=100.0)]
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0,
        prior_session_bars=prior,
    )
    assert result.avg_volume_first_30m_prior_5_days is None
    assert "insufficient_prior_sessions" in (
        result.reasons["avg_volume_first_30m_prior_5_days"]
    )


def test_volume_ratio_is_the_quotient_when_both_legs_exist():
    prior = []
    for day in (date(2026, 1, 5), date(2026, 1, 6)):
        prior.append(mb(day, 9, 30, 100.0, volume=500.0))
    bars = [
        mb(D1, 15, 59, 100.0),
        mb(D2, 9, 30, 105.0, open_=105.0, volume=1000.0),
    ]
    result = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0,
        prior_session_bars=prior,
    )
    assert result.avg_volume_first_30m_prior_5_days == pytest.approx(500.0)
    assert result.volume_first_30m == pytest.approx(1000.0)
    assert result.volume_ratio_first_30m == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 10. Determinism, purity and rendering
# ---------------------------------------------------------------------------


def test_intraday_reaction_is_deterministic_and_does_not_mutate_its_input():
    bars = amc_bars()
    snapshot = [(b.ts_utc, b.close, b.volume) for b in bars]
    first = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    second = intraday_reaction(
        bars, event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    assert [(b.ts_utc, b.close, b.volume) for b in bars] == snapshot
    assert first.gap_at_open == second.gap_at_open
    assert {k: w.move for k, w in first.windows.items()} == {
        k: w.move for k, w in second.windows.items()
    }


def test_intraday_result_is_frozen_and_copies_its_mappings():
    windows = {5: IntradayWindow(minutes=5)}
    result = IntradayReaction(windows=windows, reasons={"a": "b"})
    windows[5] = IntradayWindow(minutes=999)
    assert result.windows[5].minutes == 5
    with pytest.raises(Exception):
        result.basis = "x"  # type: ignore[misc]


def test_intraday_reaction_to_dict_renders_iso_strings_and_labelled_windows():
    result = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    payload = intraday_reaction_to_dict(result)
    assert payload["session"] == "AFTER_MARKET"
    assert payload["basis"] == BASIS_INTRADAY_AFTER_MARKET
    assert payload["confidence"] == "high"
    assert set(payload["windows"]) == {"5m", "30m", "60m"}
    assert payload["windows"]["30m"]["move"] == pytest.approx(0.07)
    assert payload["open_ts"].endswith("+00:00")
    assert payload["event_date_et"] == D1.isoformat()




def _daily_series() -> list[DailyBar]:
    return [
        DailyBar(date=date(2026, 1, 6), open=98.0, high=99.0, low=97.0, close=98.0),
        DailyBar(date=date(2026, 1, 7), open=99.0, high=100.0, low=98.0, close=99.0),
        DailyBar(date=D1, open=100.0, high=101.0, low=99.0, close=100.0),
        DailyBar(date=D2, open=105.0, high=112.0, low=104.0, close=110.0),
        DailyBar(date=date(2026, 1, 12), open=110.0, high=113.0, low=109.0, close=112.0),
    ]


def _replay(**overrides):
    intraday = intraday_reaction(
        amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
    )
    daily = event_reaction(_daily_series(), D1, AMC, horizons=(1, 3))
    kwargs = dict(
        event_id=7,
        event_key="EARNINGS:AAPL:2026-01-08",
        event_type=EventType.EARNINGS,
        ticker="AAPL",
        date_et=D1,
        session=AMC,
        status=EventStatus.CONFIRMED,
        source_url="https://example.invalid/ir",
        release_ts_utc=AMC_TS,
        source_name="company_ir",
        intraday=intraday,
        daily=daily,
    )
    kwargs.update(overrides)
    return build_event_replay(**kwargs)


def test_build_event_replay_carries_the_event_reference():
    replay = _replay()
    assert isinstance(replay, EventReplay)
    payload = replay.to_dict()
    assert payload["event"] == {
        "event_id": 7,
        "event_key": "EARNINGS:AAPL:2026-01-08",
        "event_type": "EARNINGS",
        "ticker": "AAPL",
        "date_et": "2026-01-08",
        "session": "AFTER_MARKET",
        "status": "CONFIRMED",
        "source_url": "https://example.invalid/ir",
    }


def test_build_event_replay_has_the_four_spec_blocks_in_order():
    payload = _replay().to_dict()
    keys = list(payload)
    for name in (
        "information_before", "release", "immediate_reaction",
        "subsequent_reaction",
    ):
        assert name in keys
    assert keys.index("information_before") < keys.index("release")
    assert keys.index("release") < keys.index("immediate_reaction")
    assert keys.index("immediate_reaction") < keys.index("subsequent_reaction")


def test_build_event_replay_release_block_carries_both_clocks():
    release = _replay().to_dict()["release"]
    assert release["timestamp_utc"] == AMC_TS.isoformat()
    assert release["timestamp_et"].startswith("2026-01-08T16:02")
    assert release["session"] == "AFTER_MARKET"
    assert release["source_name"] == "company_ir"


def test_build_event_replay_absent_information_before_refs_are_explicit():
    before = _replay().to_dict()["information_before"]
    for name in ("fundamentals", "price_context", "news_window"):
        assert before[name]["available"] is False
        assert before[name]["reason"]
    assert "Phase D" in before["news_window"]["reason"]


def test_build_event_replay_supplied_information_before_refs_pass_through():
    replay = _replay(
        fundamentals_ref={"as_of": "2026-01-08T21:00:00+00:00", "quarters": 4},
        price_context_ref={"anchor_date_et": "2025-10-30"},
        news_window_ref={"articles": 3},
    )
    before = replay.to_dict()["information_before"]
    assert before["fundamentals"]["available"] is True
    assert before["fundamentals"]["quarters"] == 4
    assert before["price_context"]["anchor_date_et"] == "2025-10-30"
    assert before["news_window"]["articles"] == 3


def test_build_event_replay_immediate_block_is_the_intraday_result_plus_quant():
    immediate = _replay().to_dict()["immediate_reaction"]
    assert immediate["provenance"] == "QUANT"
    assert immediate["basis"] == BASIS_INTRADAY_AFTER_MARKET
    assert immediate["gap_at_open"] == pytest.approx(0.05)
    assert immediate["windows"]["30m"]["move"] == pytest.approx(0.07)


def test_build_event_replay_without_minute_bars_states_the_reason():
    replay = _replay(intraday=None, intraday_reason="no minute bars stored yet")
    immediate = replay.to_dict()["immediate_reaction"]
    assert immediate["available"] is False
    assert immediate["reason"] == "no minute bars stored yet"
    assert replay.reasons["immediate_reaction"] == "no minute bars stored yet"


def test_build_event_replay_subsequent_block_carries_the_daily_reaction():
    subsequent = _replay().to_dict()["subsequent_reaction"]
    reaction = subsequent["reaction"]
    assert reaction["bars_available"] is True
    assert reaction["pre_event_close"] == pytest.approx(100.0)
    assert reaction["gap_return"] == pytest.approx(0.05)   # 105/100 - 1
    assert reaction["returns"]["1D"] == pytest.approx(0.10)  # 110/100 - 1
    assert subsequent["provenance"] == "QUANT"


def test_build_event_replay_accepts_a_pre_rendered_daily_dict():
    replay = _replay(daily=None, daily_dict={"bars_available": True, "x": 1})
    subsequent = replay.to_dict()["subsequent_reaction"]
    assert subsequent["reaction"] == {"bars_available": True, "x": 1}
    assert subsequent["available"] is True


def test_build_event_replay_without_daily_bars_states_the_reason():
    replay = _replay(daily=None)
    subsequent = replay.to_dict()["subsequent_reaction"]
    assert subsequent["available"] is False
    assert replay.reasons["subsequent_reaction"]


def test_build_event_replay_includes_the_abnormal_overlay():
    daily = event_reaction(_daily_series(), D1, AMC, horizons=(1, 3))
    bench = [
        DailyBar(date=b.date, open=100.0, high=101.0, low=99.0, close=100.0)
        for b in _daily_series()
    ]
    abnormal = abnormal_vs(daily, bench, D1, AMC)
    subsequent = _replay(abnormal=abnormal).to_dict()["subsequent_reaction"]
    assert subsequent["abnormal"]["benchmark_available"] is True
    # A flat benchmark leaves the abnormal return equal to the raw one.
    assert subsequent["abnormal"]["abnormal"]["1D"] == pytest.approx(0.10)


def test_build_event_replay_without_a_benchmark_is_an_explicit_absence():
    subsequent = _replay().to_dict()["subsequent_reaction"]
    assert subsequent["abnormal"]["available"] is False
    assert subsequent["abnormal"]["reason"]


def test_build_event_replay_labels_provenance_data_vs_quant():
    provenance = _replay().to_dict()["provenance"]
    assert provenance["minute_bars"] == "DATA"
    assert provenance["daily_bars"] == "DATA"
    assert provenance["metrics"] == "QUANT"


def test_build_event_replay_is_frozen_and_copies_its_mappings():
    freshness = {"bars_through": "2026-01-12"}
    replay = _replay(data_freshness=freshness)
    freshness["bars_through"] = "1999-01-01"
    assert replay.data_freshness["bars_through"] == "2026-01-12"
    with pytest.raises(Exception):
        replay.ticker = "MSFT"  # type: ignore[misc]


def test_build_event_replay_computes_nothing_it_was_not_handed():
    """A replay with no pieces at all still returns the four blocks."""
    replay = build_event_replay(event_id=1, event_key="k")
    payload = replay.to_dict()
    assert payload["immediate_reaction"]["available"] is False
    assert payload["subsequent_reaction"]["available"] is False
    assert payload["release"]["timestamp_utc"] is None
    assert replay.reasons["release.timestamp"]


# ---------------------------------------------------------------------------
# 12. §60 — history_table
# ---------------------------------------------------------------------------


def _history_entry(day: date, *, close_after: float, with_intraday: bool = False,
                   status: str = "CONFIRMED") -> dict:
    """One past event with a hand-checkable 1D/5D reaction off a 100 base."""
    bars = [
        DailyBar(date=day - timedelta(days=2), open=100.0, high=100.0, low=100.0, close=100.0),
        DailyBar(date=day, open=100.0, high=100.0, low=100.0, close=100.0),
        DailyBar(date=day + timedelta(days=1), open=close_after, high=close_after,
                 low=close_after, close=close_after),
        DailyBar(date=day + timedelta(days=2), open=close_after, high=close_after,
                 low=close_after, close=close_after),
        DailyBar(date=day + timedelta(days=3), open=close_after, high=close_after,
                 low=close_after, close=close_after),
        DailyBar(date=day + timedelta(days=4), open=close_after, high=close_after,
                 low=close_after, close=close_after),
        DailyBar(date=day + timedelta(days=5), open=close_after, high=close_after,
                 low=close_after, close=close_after),
    ]
    entry: dict = {
        "event_id": int(day.strftime("%Y%m%d")),
        "event_key": f"EARNINGS:AAPL:{day.isoformat()}",
        "date_et": day,
        "session": AMC,
        "status": status,
        "reaction": event_reaction(bars, day, AMC, horizons=(1, 3, 5, 10)),
    }
    if with_intraday:
        entry["intraday"] = intraday_reaction(
            amc_bars(), event_ts_utc=AMC_TS, session=AMC, pre_event_close=100.0
        )
    return entry


def test_history_table_row_shape_matches_the_spec_columns():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    row = table["rows"][0]
    for column in HISTORY_COLUMNS:
        assert column in row, column
    assert table["columns"] == list(HISTORY_COLUMNS)


def test_history_table_computes_gap_and_returns_from_the_reaction():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    row = table["rows"][0]
    assert row["gap"] == pytest.approx(0.10)     # AMC next open 110 vs 100
    assert row["ret_1d"] == pytest.approx(0.10)
    assert row["ret_5d"] == pytest.approx(0.10)
    assert row["bars_available"] is True


def test_history_table_actual_move_abs_is_the_absolute_one_day_return():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=90.0)])
    row = table["rows"][0]
    assert row["ret_1d"] == pytest.approx(-0.10)
    assert row["actual_move_abs"] == pytest.approx(0.10)


def test_history_table_surprise_columns_are_always_unavailable():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    row = table["rows"][0]
    for column in ("eps_surprise", "rev_surprise"):
        assert row[column] == {
            "available": False,
            "reason": CONSENSUS_UNAVAILABLE_REASON,
        }


def test_history_table_implied_move_is_always_unavailable_with_phase_i_reason():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    assert table["rows"][0]["implied_move"] == {
        "available": False,
        "reason": IMPLIED_MOVE_UNAVAILABLE_REASON,
    }
    assert "Phase I" in IMPLIED_MOVE_UNAVAILABLE_REASON


def test_history_table_intraday_30m_is_unavailable_without_minute_bars():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    cell = table["rows"][0]["intraday_30m"]
    assert cell["available"] is False
    assert cell["reason"] == "no minute bars stored for this event"


def test_history_table_intraday_30m_present_when_minute_bars_were_stored():
    table = history_table(
        [_history_entry(date(2025, 10, 30), close_after=110.0, with_intraday=True)]
    )
    cell = table["rows"][0]["intraday_30m"]
    assert cell["available"] is True
    assert cell["move"] == pytest.approx(0.07)
    assert cell["basis"] == BASIS_INTRADAY_AFTER_MARKET
    assert cell["confidence"] == "high"


def test_history_table_rows_are_sorted_oldest_first():
    entries = [
        _history_entry(date(2026, 1, 29), close_after=103.0),
        _history_entry(date(2025, 7, 31), close_after=101.0),
        _history_entry(date(2025, 10, 30), close_after=102.0),
    ]
    table = history_table(entries)
    assert [r["date_et"] for r in table["rows"]] == [
        "2025-07-31", "2025-10-30", "2026-01-29",
    ]


def test_history_table_summary_reuses_history_stats():
    entries = [
        _history_entry(date(2025, 7, 31), close_after=110.0),   # +10%
        _history_entry(date(2025, 10, 30), close_after=90.0),   # -10%
        _history_entry(date(2026, 1, 29), close_after=120.0),   # +20%
    ]
    table = history_table(entries)
    last4 = table["summary"]["1D"]["last4"]
    assert last4["n_available"] == 3
    assert last4["median_abs"] == pytest.approx(0.10)
    assert last4["max_abs"] == pytest.approx(0.20)
    assert last4["positive_count"] == 2
    assert set(table["summary"]["1D"]) == {"last4", "last8", "last12"}


def test_history_table_summary_refuses_a_one_event_median():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    last4 = table["summary"]["1D"]["last4"]
    assert last4["median_abs"] is None
    assert "insufficient_sample" in last4["reasons"]["sample"]


def test_history_table_unmeasurable_event_carries_reasons_not_zeros():
    entry = {
        "event_id": 1,
        "event_key": "EARNINGS:AAPL:2026-06-01",
        "date_et": date(2026, 6, 1),
        "session": AMC,
        "status": "CONFIRMED",
        "reaction": event_reaction([], date(2026, 6, 1), AMC, horizons=(1, 5)),
    }
    row = history_table([entry])["rows"][0]
    assert row["gap"] is None
    assert row["ret_1d"] is None
    assert row["actual_move_abs"] is None
    assert row["bars_available"] is False
    assert row["reasons"]["reaction"] == "no_bars_available"
    assert row["reasons"]["ret_1d"]


def test_history_table_missing_reaction_object_is_an_explicit_reason():
    entry = {"event_id": 1, "event_key": "k", "date_et": date(2026, 6, 1),
             "session": AMC, "status": "CONFIRMED"}
    row = history_table([entry])["rows"][0]
    assert row["bars_available"] is False
    assert "not supplied" in row["reasons"]["reaction"]


def test_history_table_abnormal_column_is_none_with_a_reason_when_absent():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    row = table["rows"][0]
    assert row["abnormal_1d"] is None
    assert row["reasons"]["abnormal_1d"]


def test_history_table_abnormal_column_is_filled_when_supplied():
    entry = _history_entry(date(2025, 10, 30), close_after=110.0)
    reaction = entry["reaction"]
    bench = [
        DailyBar(date=d, open=100.0, high=100.0, low=100.0, close=100.0)
        for d in (
            date(2025, 10, 28), date(2025, 10, 30), date(2025, 10, 31),
            date(2025, 11, 1), date(2025, 11, 2), date(2025, 11, 3),
            date(2025, 11, 4),
        )
    ]
    entry["abnormal"] = abnormal_vs(reaction, bench, date(2025, 10, 30), AMC)
    row = history_table([entry])["rows"][0]
    assert row["abnormal_1d"] == pytest.approx(0.10)


def test_history_table_is_empty_but_well_formed_with_no_events():
    table = history_table([])
    assert table["rows"] == []
    assert table["n_rows"] == 0
    assert table["summary"]["1D"]["last4"]["n_available"] == 0
    assert table["not_backtestable"] == [
        "eps_surprise", "rev_surprise", "implied_move",
    ]


def test_history_table_labels_provenance():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    assert table["provenance"] == {"bars": "DATA", "metrics": "QUANT"}


def test_history_table_renders_enums_and_dates_as_json_scalars():
    table = history_table([_history_entry(date(2025, 10, 30), close_after=110.0)])
    row = table["rows"][0]
    assert row["session"] == "AFTER_MARKET"
    assert isinstance(row["date_et"], str)
    assert row["status"] == "CONFIRMED"


# ---------------------------------------------------------------------------
# 13. §15 — link_previous_events
# ---------------------------------------------------------------------------


def _event(
    key: str,
    *,
    when: datetime,
    ticker: str | None = "AAPL",
    event_type: EventType = EventType.EARNINGS,
    status: EventStatus = EventStatus.CONFIRMED,
    event_id: int | None = None,
) -> Event:
    return Event(
        event_key=key,
        event_type=event_type,
        title=key,
        scheduled_at=when,
        status=status,
        source=EventSourceKind.COMPANY_IR_SEC,
        source_name="ir",
        ticker=ticker,
        event_id=event_id,
        session=AMC,
    )


def _q(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 21, 0, tzinfo=UTC)


def test_link_previous_events_chains_one_tickers_earnings():
    events = [
        _event("E1", when=_q(2025, 4, 30), event_id=1),
        _event("E2", when=_q(2025, 7, 31), event_id=2),
        _event("E3", when=_q(2025, 10, 30), event_id=3),
    ]
    links = link_previous_events(events)
    assert [(k, p) for k, p, _ in links] == [
        ("E1", None), ("E2", "E1"), ("E3", "E2"),
    ]


def test_link_previous_events_first_event_has_a_named_reason_not_a_blank():
    links = link_previous_events([_event("E1", when=_q(2025, 4, 30), event_id=1)])
    assert links == [("E1", None, NO_PREVIOUS_REASON)]


def test_link_previous_events_carries_the_matcher_reason():
    events = [
        _event("E1", when=_q(2025, 4, 30), event_id=1),
        _event("E2", when=_q(2025, 7, 31), event_id=2),
    ]
    links = dict((k, r) for k, _p, r in link_previous_events(events))
    assert links["E2"] == "prior quarterly earnings"


def test_link_previous_events_never_crosses_tickers():
    events = [
        _event("AAPL-1", when=_q(2025, 7, 31), ticker="AAPL", event_id=1),
        _event("MSFT-1", when=_q(2025, 10, 1), ticker="MSFT", event_id=2),
        _event("MSFT-2", when=_q(2026, 1, 5), ticker="MSFT", event_id=3),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["MSFT-1"] is None       # AAPL's print is not comparable
    assert links["MSFT-2"] == "MSFT-1"


def test_link_previous_events_never_crosses_types():
    events = [
        _event("EARN", when=_q(2025, 7, 31), event_id=1),
        _event(
            "FOMC-1", when=_q(2025, 9, 17), ticker=None,
            event_type=EventType.FOMC_DECISION, event_id=2,
        ),
        _event(
            "FOMC-2", when=_q(2025, 11, 5), ticker=None,
            event_type=EventType.FOMC_DECISION, event_id=3,
        ),
        _event("EARN-2", when=_q(2025, 12, 1), event_id=4),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["FOMC-1"] is None
    assert links["FOMC-2"] == "FOMC-1"
    assert links["EARN-2"] == "EARN"     # not FOMC-2, despite being nearer


def test_link_previous_events_links_fomc_minutes_only_to_minutes():
    events = [
        _event("DEC-1", when=_q(2025, 9, 17), ticker=None,
               event_type=EventType.FOMC_DECISION, event_id=1),
        _event("MIN-1", when=_q(2025, 10, 8), ticker=None,
               event_type=EventType.FOMC_MINUTES, event_id=2),
        _event("MIN-2", when=_q(2025, 11, 26), ticker=None,
               event_type=EventType.FOMC_MINUTES, event_id=3),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["MIN-2"] == "MIN-1"
    assert links["MIN-1"] is None


def test_link_previous_events_estimated_points_at_the_latest_confirmed():
    events = [
        _event("E1", when=_q(2025, 7, 31), event_id=1),
        _event("E2", when=_q(2025, 10, 30), event_id=2),
        _event("E3", when=_q(2026, 1, 29), event_id=3,
               status=EventStatus.ESTIMATED),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["E3"] == "E2"


def test_link_previous_events_skips_an_estimated_predecessor_for_earnings():
    """A guessed past date is not an anchor — §15 wants CONFIRMED/REVISED."""
    events = [
        _event("E1", when=_q(2025, 7, 31), event_id=1),
        _event("E2", when=_q(2025, 10, 30), event_id=2,
               status=EventStatus.ESTIMATED),
        _event("E3", when=_q(2026, 1, 29), event_id=3),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["E3"] == "E1"


def test_link_previous_events_accepts_a_revised_predecessor():
    events = [
        _event("E1", when=_q(2025, 10, 30), event_id=1,
               status=EventStatus.REVISED),
        _event("E2", when=_q(2026, 1, 29), event_id=2),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["E2"] == "E1"


def test_link_previous_events_excludes_canceled_as_a_subject():
    events = [
        _event("E1", when=_q(2025, 10, 30), event_id=1),
        _event("E2", when=_q(2026, 1, 29), event_id=2,
               status=EventStatus.CANCELED),
    ]
    links = dict((k, (p, r)) for k, p, r in link_previous_events(events))
    assert links["E2"] == (None, CANCELED_REASON)


def test_link_previous_events_excludes_canceled_as_a_predecessor():
    events = [
        _event("E1", when=_q(2025, 7, 31), event_id=1),
        _event("E2", when=_q(2025, 10, 30), event_id=2,
               status=EventStatus.CANCELED),
        _event("E3", when=_q(2026, 1, 29), event_id=3),
    ]
    links = {k: p for k, p, _ in link_previous_events(events)}
    assert links["E3"] == "E1"


def test_link_previous_events_is_ordered_by_schedule_and_deterministic():
    events = [
        _event("E3", when=_q(2026, 1, 29), event_id=3),
        _event("E1", when=_q(2025, 4, 30), event_id=1),
        _event("E2", when=_q(2025, 7, 31), event_id=2),
    ]
    first = link_previous_events(events)
    second = link_previous_events(list(reversed(events)))
    assert [k for k, _p, _r in first] == ["E1", "E2", "E3"]
    assert first == second


def test_link_previous_events_never_links_an_event_to_itself():
    events = [
        _event("E1", when=_q(2025, 7, 31), event_id=1),
        _event("E2", when=_q(2025, 10, 30), event_id=2),
    ]
    for key, previous, _reason in link_previous_events(events):
        assert previous != key


def test_link_previous_events_handles_an_empty_batch():
    assert link_previous_events([]) == []


def test_link_previous_events_market_holiday_has_no_predecessor():
    events = [
        _event("H1", when=_q(2025, 12, 25), ticker=None,
               event_type=EventType.MARKET_HOLIDAY, event_id=1),
        _event("H2", when=_q(2026, 1, 1), ticker=None,
               event_type=EventType.MARKET_HOLIDAY, event_id=2),
    ]
    links = {k: (p, r) for k, p, r in link_previous_events(events)}
    assert links["H2"] == (None, NO_PREVIOUS_REASON)


def test_link_previous_events_does_not_mutate_the_input_sequence():
    events = [
        _event("E2", when=_q(2025, 10, 30), event_id=2),
        _event("E1", when=_q(2025, 7, 31), event_id=1),
    ]
    order = [e.event_key for e in events]
    link_previous_events(events)
    assert [e.event_key for e in events] == order
