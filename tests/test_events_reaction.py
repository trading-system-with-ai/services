"""Price context & previous-event reaction — pure arithmetic (event spec §14,
§17, §19, §31, §32, §64, §96; Phase E1 unit U1).

Every number here is hand-checkable: the fixtures are integer-ish closes so a
1-day return like ``101/100 - 1`` reads off the page, and no test asserts a
value the module computed for itself. Four contracts are pinned:

1. **The as-of gate** (§14/§96) — a daily bar dated *d* is invisible until
   16:00 ET on *d*. Tested at 15:59 vs 16:00 on both sides of both DST
   transitions, because a UTC-only implementation is right for half the year.
2. **The reaction window** (§17) — AMC/BMO/DURING/UNKNOWN pick different
   pre/react bars, a Friday AMC print reacts on Monday, and a holiday date
   with no bar falls through to the next bar rather than vanishing.
3. **Absence is a value** — a horizon past the end of the data, an event
   before the first bar, a missing benchmark day, a zero pre-event close:
   each yields ``None`` plus a reason string, and never a zero, a NaN or an
   ``inf``.
4. **§19/§64 honesty** — statistics carry their sample size, a window with
   fewer than two usable events refuses to produce a median, and the
   percentiles are nearest-rank (an observed move, not an interpolation).
"""
import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from libs.trading_core.events import (
    DailyBar,
    abnormal_vs,
    as_of_bar_filter,
    event_reaction,
    first_reaction_index,
    history_stats,
    percentile_nearest_rank,
    pre_event_price_context,
)
from libs.trading_core.events.reaction import (
    BASIS_AFTER_MARKET,
    BASIS_BEFORE_MARKET,
    BASIS_DURING_MARKET,
    BASIS_UNKNOWN,
    ReactionResult,
)
from libs.trading_core.models.enums import EventSession

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

AMC = EventSession.AFTER_MARKET
BMO = EventSession.BEFORE_MARKET
DURING = EventSession.DURING_MARKET
UNKNOWN = EventSession.UNKNOWN


def _bar(d: date, close: float, *, open_: float | None = None, volume: float = 1_000_000.0) -> DailyBar:
    """A bar whose OHLC hangs off one close so returns stay hand-checkable."""
    o = close if open_ is None else open_
    return DailyBar(
        date=d,
        open=o,
        high=max(o, close) + 1.0,
        low=min(o, close) - 1.0,
        close=close,
        volume=volume,
    )


def _series(pairs) -> list[DailyBar]:
    return [_bar(d, c) for d, c in pairs]


#: Mon 2026-01-05 .. Fri 2026-01-16 (two clean trading weeks), close = 100 + i.
WEEKS = _series(
    [
        (date(2026, 1, 5), 100.0),
        (date(2026, 1, 6), 101.0),
        (date(2026, 1, 7), 102.0),
        (date(2026, 1, 8), 103.0),
        (date(2026, 1, 9), 104.0),
        (date(2026, 1, 12), 105.0),
        (date(2026, 1, 13), 106.0),
        (date(2026, 1, 14), 107.0),
        (date(2026, 1, 15), 108.0),
        (date(2026, 1, 16), 109.0),
    ]
)


# ---------------------------------------------------------------------------
# §14/§96 — as_of_bar_filter, the look-ahead gate
# ---------------------------------------------------------------------------


def test_as_of_1559_et_excludes_the_same_day_bar():
    kept = as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 15, 59, tzinfo=ET))
    assert [b.date for b in kept] == [
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
    ]


def test_as_of_1600_et_includes_the_same_day_bar():
    kept = as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 16, 0, tzinfo=ET))
    assert kept[-1].date == date(2026, 1, 9)
    assert len(kept) == 5


def test_as_of_boundary_is_exactly_1600_not_1601():
    at_1600 = as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 16, 0, tzinfo=ET))
    at_1601 = as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 16, 1, tzinfo=ET))
    assert [b.date for b in at_1600] == [b.date for b in at_1601]


def test_as_of_accepts_utc_input_and_converts_to_et():
    # 20:59 UTC on 2026-01-09 is 15:59 ET (EST, UTC-5) -> same-day bar excluded.
    kept = as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 20, 59, tzinfo=UTC))
    assert kept[-1].date == date(2026, 1, 8)
    # 21:00 UTC is 16:00 ET -> included.
    kept = as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 21, 0, tzinfo=UTC))
    assert kept[-1].date == date(2026, 1, 9)


def test_as_of_dst_summer_boundary_uses_et_wall_clock_not_utc():
    """In EDT (UTC-4) the close is 20:00 UTC — a UTC-only gate is an hour off."""
    bars = _series([(date(2026, 7, 9), 50.0), (date(2026, 7, 10), 51.0)])
    at_1959_utc = as_of_bar_filter(bars, datetime(2026, 7, 10, 19, 59, tzinfo=UTC))
    at_2000_utc = as_of_bar_filter(bars, datetime(2026, 7, 10, 20, 0, tzinfo=UTC))
    assert [b.date for b in at_1959_utc] == [date(2026, 7, 9)]
    assert [b.date for b in at_2000_utc] == [date(2026, 7, 9), date(2026, 7, 10)]


def test_as_of_across_the_spring_forward_day():
    """2026-03-08 is the DST switch; 2026-03-09 15:59 ET still excludes 03-09."""
    bars = _series([(date(2026, 3, 6), 10.0), (date(2026, 3, 9), 11.0)])
    assert [b.date for b in as_of_bar_filter(bars, datetime(2026, 3, 9, 15, 59, tzinfo=ET))] == [
        date(2026, 3, 6)
    ]
    assert len(as_of_bar_filter(bars, datetime(2026, 3, 9, 16, 0, tzinfo=ET))) == 2


def test_as_of_across_the_fall_back_day():
    """2026-11-01 falls back to EST; 11-02 16:00 ET is 21:00 UTC again."""
    bars = _series([(date(2026, 10, 30), 10.0), (date(2026, 11, 2), 11.0)])
    assert len(as_of_bar_filter(bars, datetime(2026, 11, 2, 20, 59, tzinfo=UTC))) == 1
    assert len(as_of_bar_filter(bars, datetime(2026, 11, 2, 21, 0, tzinfo=UTC))) == 2


def test_as_of_refuses_a_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        as_of_bar_filter(WEEKS, datetime(2026, 1, 9, 16, 0))


def test_as_of_does_not_mutate_or_reorder_the_input():
    original = list(WEEKS)
    kept = as_of_bar_filter(WEEKS, datetime(2026, 1, 20, 16, 0, tzinfo=ET))
    assert WEEKS == original
    assert [b.date for b in kept] == [b.date for b in WEEKS]


# ---------------------------------------------------------------------------
# §17 — first_reaction_index windows
# ---------------------------------------------------------------------------


def test_after_market_reacts_on_the_next_bar():
    pre, react, basis = first_reaction_index(WEEKS, date(2026, 1, 6), AMC)
    assert (WEEKS[pre].date, WEEKS[react].date) == (date(2026, 1, 6), date(2026, 1, 7))
    assert basis == BASIS_AFTER_MARKET


def test_after_market_on_a_friday_reacts_on_monday():
    pre, react, basis = first_reaction_index(WEEKS, date(2026, 1, 9), AMC)
    assert (WEEKS[pre].date, WEEKS[react].date) == (date(2026, 1, 9), date(2026, 1, 12))
    assert basis == BASIS_AFTER_MARKET


def test_after_market_on_a_non_trading_day_uses_the_prior_close():
    """A Saturday AMC timestamp: pre is Friday's close, react is Monday."""
    pre, react, _ = first_reaction_index(WEEKS, date(2026, 1, 10), AMC)
    assert (WEEKS[pre].date, WEEKS[react].date) == (date(2026, 1, 9), date(2026, 1, 12))


def test_before_market_reacts_on_the_same_day():
    pre, react, basis = first_reaction_index(WEEKS, date(2026, 1, 7), BMO)
    assert (WEEKS[pre].date, WEEKS[react].date) == (date(2026, 1, 6), date(2026, 1, 7))
    assert basis == BASIS_BEFORE_MARKET


def test_before_market_on_a_holiday_falls_through_to_the_next_bar():
    """2026-01-19 (MLK) has no bar: pre is Friday 01-16, react is Tue 01-20."""
    bars = WEEKS + [_bar(date(2026, 1, 20), 110.0)]
    pre, react, basis = first_reaction_index(bars, date(2026, 1, 19), BMO)
    assert (bars[pre].date, bars[react].date) == (date(2026, 1, 16), date(2026, 1, 20))
    assert basis == BASIS_BEFORE_MARKET


def test_during_market_measures_the_same_day_against_the_prior_close():
    pre, react, basis = first_reaction_index(WEEKS, date(2026, 1, 8), DURING)
    assert (WEEKS[pre].date, WEEKS[react].date) == (date(2026, 1, 7), date(2026, 1, 8))
    assert basis == BASIS_DURING_MARKET


def test_unknown_session_spans_two_days_conservatively():
    pre, react, basis = first_reaction_index(WEEKS, date(2026, 1, 8), UNKNOWN)
    assert (WEEKS[pre].date, WEEKS[react].date) == (date(2026, 1, 7), date(2026, 1, 9))
    assert basis == BASIS_UNKNOWN


def test_unknown_and_during_differ_on_the_same_event():
    _, during_react, _ = first_reaction_index(WEEKS, date(2026, 1, 8), DURING)
    _, unknown_react, _ = first_reaction_index(WEEKS, date(2026, 1, 8), UNKNOWN)
    assert WEEKS[during_react].date < WEEKS[unknown_react].date


def test_first_reaction_index_none_when_no_bar_follows_the_event():
    assert first_reaction_index(WEEKS, date(2026, 1, 16), AMC) is None


def test_first_reaction_index_none_when_the_event_predates_the_history():
    assert first_reaction_index(WEEKS, date(2025, 11, 3), BMO) is None


def test_first_reaction_index_none_on_empty_bars():
    assert first_reaction_index([], date(2026, 1, 6), AMC) is None


# ---------------------------------------------------------------------------
# §17 — event_reaction arithmetic
# ---------------------------------------------------------------------------


def test_event_reaction_amc_returns_are_hand_checkable():
    r = event_reaction(WEEKS, date(2026, 1, 6), AMC)
    assert r.bars_available is True
    assert r.pre_event_close == 101.0
    assert r.react_date == date(2026, 1, 7)
    # 1D = 102/101 - 1, 3D = close of 01-09 = 104/101 - 1, 5D = 106/101 - 1.
    assert r.returns[1] == pytest.approx(102.0 / 101.0 - 1.0)
    assert r.returns[3] == pytest.approx(104.0 / 101.0 - 1.0)
    assert r.returns[5] == pytest.approx(106.0 / 101.0 - 1.0)
    assert r.abs_returns[1] == pytest.approx(abs(r.returns[1]))


def test_event_reaction_gap_uses_the_react_open_not_its_close():
    bars = [
        _bar(date(2026, 2, 2), 100.0),
        _bar(date(2026, 2, 3), 95.0, open_=90.0),  # gapped down, closed up off the low
        _bar(date(2026, 2, 4), 96.0),
    ]
    r = event_reaction(bars, date(2026, 2, 2), AMC, horizons=(1,))
    assert r.gap_return == pytest.approx(90.0 / 100.0 - 1.0)
    assert r.returns[1] == pytest.approx(95.0 / 100.0 - 1.0)
    assert r.gap_return != r.returns[1]


def test_event_reaction_bmo_1d_is_the_event_days_own_close():
    r = event_reaction(WEEKS, date(2026, 1, 7), BMO, horizons=(1,))
    assert (r.pre_event_date, r.react_date) == (date(2026, 1, 6), date(2026, 1, 7))
    assert r.returns[1] == pytest.approx(102.0 / 101.0 - 1.0)


def test_event_reaction_horizon_beyond_data_is_none_with_a_reason():
    r = event_reaction(WEEKS, date(2026, 1, 14), AMC, horizons=(1, 3, 5, 10))
    assert r.returns[1] is not None
    assert r.returns[5] is None
    assert r.reasons["return_5D"] == "insufficient_bars_after_event"
    assert r.reasons["return_10D"] == "insufficient_bars_after_event"


def test_event_reaction_before_first_bar_says_bars_unavailable_before():
    r = event_reaction(WEEKS, date(2023, 11, 2), AMC)
    assert r.bars_available is False
    assert r.reasons["bars"] == "bars unavailable before 2026-01-05"
    assert all(v is None for v in r.returns.values())
    assert r.pre_event_close is None


def test_event_reaction_too_recent_says_no_bar_after_the_event():
    r = event_reaction(WEEKS, date(2026, 1, 16), AMC)
    assert r.bars_available is False
    assert "no bar after the event yet" in r.reasons["bars"]
    assert "2026-01-16" in r.reasons["bars"]


def test_event_reaction_on_empty_bars_is_unavailable_not_an_exception():
    r = event_reaction([], date(2026, 1, 6), AMC)
    assert r.bars_available is False
    assert r.reasons["bars"] == "no_bars_available"


def test_event_reaction_zero_pre_close_yields_none_not_inf():
    bars = [
        _bar(date(2026, 1, 5), 0.0),
        _bar(date(2026, 1, 6), 10.0),
        _bar(date(2026, 1, 7), 11.0),
    ]
    r = event_reaction(bars, date(2026, 1, 5), AMC, horizons=(1,))
    assert r.pre_event_close is None
    assert r.returns[1] is None
    assert r.gap_return is None
    assert r.reasons["pre_event_close"] == "pre_event_close_not_positive"


def test_event_reaction_never_produces_nan_or_inf():
    r = event_reaction(WEEKS, date(2026, 1, 6), AMC)
    numbers = [
        r.gap_return,
        r.max_favorable_excursion,
        r.max_adverse_excursion,
        *r.returns.values(),
        *r.abs_returns.values(),
    ]
    assert all(v is None or math.isfinite(v) for v in numbers)


def test_event_reaction_excursions_bracket_the_path():
    bars = [
        _bar(date(2026, 3, 2), 100.0),
        _bar(date(2026, 3, 3), 110.0),  # +10%
        _bar(date(2026, 3, 4), 90.0),  # -10%
        _bar(date(2026, 3, 5), 105.0),
    ]
    r = event_reaction(bars, date(2026, 3, 2), AMC, horizons=(1, 3))
    assert r.max_favorable_excursion == pytest.approx(0.10)
    assert r.max_adverse_excursion == pytest.approx(-0.10)


def test_event_reaction_records_window_end_dates_only_for_measured_horizons():
    # React bar is 01-15; 1D ends there, 3D would end on the 3rd bar (01-19,
    # which does not exist -> only 01-16 remains, so 3D is unmeasured too).
    r = event_reaction(WEEKS, date(2026, 1, 14), AMC, horizons=(1, 2, 3))
    assert r.window_end_dates == {1: date(2026, 1, 15), 2: date(2026, 1, 16)}
    assert r.returns[3] is None and 3 not in r.window_end_dates


def test_event_reaction_rejects_a_zero_horizon():
    with pytest.raises(ValueError, match="horizons"):
        event_reaction(WEEKS, date(2026, 1, 6), AMC, horizons=(0, 1))


def test_event_reaction_rejects_unsorted_bars():
    scrambled = [WEEKS[1], WEEKS[0], WEEKS[2]]
    with pytest.raises(ValueError, match="strictly increasing"):
        event_reaction(scrambled, date(2026, 1, 6), AMC)


# ---------------------------------------------------------------------------
# §17 — abnormal_vs (benchmark aligned by DATE)
# ---------------------------------------------------------------------------


def test_abnormal_subtracts_the_benchmark_move_on_the_same_window():
    stock = event_reaction(WEEKS, date(2026, 1, 6), AMC, horizons=(1,))
    bench = _series([(d.date, 200.0) for d in WEEKS])  # flat benchmark
    ab = abnormal_vs(stock, bench, date(2026, 1, 6), AMC)
    assert ab.benchmark_available is True
    assert ab.benchmark_returns[1] == pytest.approx(0.0)
    assert ab.abnormal[1] == pytest.approx(stock.returns[1])


def test_abnormal_is_zero_when_the_benchmark_moves_identically():
    stock = event_reaction(WEEKS, date(2026, 1, 6), AMC, horizons=(1, 3))
    bench = _series([(b.date, b.close * 2.0) for b in WEEKS])
    ab = abnormal_vs(stock, bench, date(2026, 1, 6), AMC)
    assert ab.abnormal[1] == pytest.approx(0.0, abs=1e-12)
    assert ab.abnormal[3] == pytest.approx(0.0, abs=1e-12)


def test_abnormal_aligns_by_date_when_the_benchmark_misses_a_day():
    """The benchmark has no 01-07 bar; its window must still END on the stock's
    window date, not slide one index forward."""
    stock = event_reaction(WEEKS, date(2026, 1, 6), AMC, horizons=(1,))
    bench = _series([(b.date, 200.0 + i) for i, b in enumerate(WEEKS) if b.date != date(2026, 1, 7)])
    ab = abnormal_vs(stock, bench, date(2026, 1, 6), AMC)
    # Falls back to the last bench bar on or before 01-07, i.e. 01-06 -> 0%.
    assert ab.benchmark_returns[1] == pytest.approx(0.0)
    assert ab.reasons["benchmark_gap"].startswith("benchmark has no bar on 2026-01-07")


def test_abnormal_without_benchmark_bars_is_unavailable_with_a_reason():
    stock = event_reaction(WEEKS, date(2026, 1, 6), AMC, horizons=(1,))
    ab = abnormal_vs(stock, [], date(2026, 1, 6), AMC)
    assert ab.benchmark_available is False
    assert ab.abnormal[1] is None
    assert ab.reasons["benchmark"] == "no_benchmark_bars_available"


def test_abnormal_when_the_benchmark_history_starts_after_the_event():
    stock = event_reaction(WEEKS, date(2026, 1, 6), AMC, horizons=(1,))
    bench = _series([(date(2026, 6, 1), 300.0), (date(2026, 6, 2), 301.0)])
    ab = abnormal_vs(stock, bench, date(2026, 1, 6), AMC)
    assert ab.benchmark_available is False
    assert "benchmark bars unavailable before 2026-06-01" in ab.reasons["benchmark"]


def test_abnormal_when_the_stock_reaction_itself_is_unavailable():
    stock = event_reaction(WEEKS, date(2023, 11, 2), AMC, horizons=(1,))
    ab = abnormal_vs(stock, WEEKS, date(2023, 11, 2), AMC)
    assert ab.benchmark_available is False
    assert ab.reasons["benchmark"] == "stock_reaction_unavailable"


def test_abnormal_horizon_the_stock_could_not_measure_is_none():
    stock = event_reaction(WEEKS, date(2026, 1, 14), AMC, horizons=(1, 10))
    bench = _series([(b.date, 200.0) for b in WEEKS])
    ab = abnormal_vs(stock, bench, date(2026, 1, 14), AMC)
    assert ab.abnormal[10] is None
    assert ab.reasons["abnormal_10D"] == "stock_window_unavailable"


def test_abnormal_gap_subtracts_the_benchmark_gap():
    bars = [
        _bar(date(2026, 4, 6), 100.0),
        _bar(date(2026, 4, 7), 104.0, open_=105.0),
        _bar(date(2026, 4, 8), 106.0),
    ]
    bench = [
        _bar(date(2026, 4, 6), 400.0),
        _bar(date(2026, 4, 7), 404.0, open_=408.0),
        _bar(date(2026, 4, 8), 406.0),
    ]
    stock = event_reaction(bars, date(2026, 4, 6), AMC, horizons=(1,))
    ab = abnormal_vs(stock, bench, date(2026, 4, 6), AMC)
    assert ab.benchmark_gap_return == pytest.approx(408.0 / 400.0 - 1.0)
    assert ab.abnormal_gap == pytest.approx(
        (105.0 / 100.0 - 1.0) - (408.0 / 400.0 - 1.0)
    )


# ---------------------------------------------------------------------------
# §19/§64 — history_stats and the percentile definition
# ---------------------------------------------------------------------------


def _reaction(day: date, ret: float) -> ReactionResult:
    """A synthetic result standing in for one past event's measured 1D move."""
    return ReactionResult(
        event_date_et=day,
        session=AMC,
        basis=BASIS_AFTER_MARKET,
        bars_available=True,
        pre_event_close=100.0,
        returns={1: ret},
        abs_returns={1: abs(ret)},
    )


FOUR = [
    _reaction(date(2025, 2, 1), 0.02),
    _reaction(date(2025, 5, 1), -0.04),
    _reaction(date(2025, 8, 1), 0.06),
    _reaction(date(2025, 11, 1), -0.08),
]


def test_history_stats_median_and_max_over_four_events():
    stats = history_stats(FOUR, horizon=1, last_n=(4,))["last4"]
    assert stats.n == 4
    assert stats.n_available == 4
    # |moves| = 2,4,6,8% -> nearest-rank p50 is the 2nd value = 4%.
    assert stats.median_abs == pytest.approx(0.04)
    assert stats.max_abs == pytest.approx(0.08)
    assert stats.mean_abs == pytest.approx(0.05)


def test_history_stats_positive_frequency_is_a_count_not_a_probability():
    stats = history_stats(FOUR, horizon=1, last_n=(4,))["last4"]
    assert stats.positive_count == 2
    assert stats.positive_frequency == pytest.approx(0.5)


def test_history_stats_takes_the_LAST_n_by_event_date():
    older = [_reaction(date(2024, 1, 1), 0.50), _reaction(date(2024, 4, 1), 0.60)]
    stats = history_stats(older + FOUR, horizon=1, last_n=(4,))["last4"]
    assert stats.n_available == 4
    assert stats.max_abs == pytest.approx(0.08)  # the 50%/60% outliers dropped


def test_history_stats_orders_unsorted_input_by_date():
    shuffled = [FOUR[2], FOUR[0], FOUR[3], FOUR[1]]
    assert (
        history_stats(shuffled, horizon=1, last_n=(4,))["last4"].max_abs
        == history_stats(FOUR, horizon=1, last_n=(4,))["last4"].max_abs
    )


def test_history_stats_requested_window_larger_than_the_sample():
    stats = history_stats(FOUR, horizon=1, last_n=(12,))["last12"]
    assert stats.n == 12  # what was asked for
    assert stats.n_available == 4  # what actually existed


def test_history_stats_refuses_a_median_of_one_event():
    stats = history_stats(FOUR[:1], horizon=1, last_n=(4,))["last4"]
    assert stats.n_available == 1
    assert stats.median_abs is None
    assert stats.p90_abs is None
    assert "insufficient_sample" in stats.reasons["sample"]


def test_history_stats_empty_input_is_all_none_with_a_reason():
    stats = history_stats([], horizon=1, last_n=(4, 8, 12))
    assert set(stats) == {"last4", "last8", "last12"}
    assert all(s.n_available == 0 and s.median_abs is None for s in stats.values())


def test_history_stats_skips_events_with_no_usable_return():
    unavailable = ReactionResult(
        event_date_et=date(2024, 12, 1), bars_available=False, returns={1: None}
    )
    stats = history_stats([*FOUR, unavailable], horizon=1, last_n=(8,))["last8"]
    assert stats.n_available == 4


def test_history_stats_horizon_5_uses_the_5d_column():
    reactions = [
        ReactionResult(
            event_date_et=date(2025, m, 1),
            bars_available=True,
            returns={1: 0.01, 5: 0.10 * i},
        )
        for i, m in enumerate((2, 5, 8), start=1)
    ]
    stats = history_stats(reactions, horizon=5, last_n=(4,))["last4"]
    assert stats.horizon == 5
    assert stats.max_abs == pytest.approx(0.30)


def test_history_stats_rejects_a_zero_horizon():
    with pytest.raises(ValueError, match="horizon"):
        history_stats(FOUR, horizon=0)


def test_percentile_nearest_rank_returns_an_observed_value():
    sample = [0.01, 0.02, 0.03, 0.04, 0.05]
    # ceil(0.9*5) = 5 -> the 5th value; no interpolation between 0.04 and 0.05.
    assert percentile_nearest_rank(sample, 90.0) == 0.05
    assert percentile_nearest_rank(sample, 75.0) == 0.04
    assert percentile_nearest_rank(sample, 50.0) == 0.03


def test_percentile_nearest_rank_edges_and_validation():
    assert percentile_nearest_rank([], 50.0) is None
    assert percentile_nearest_rank([0.07], 90.0) == 0.07
    assert percentile_nearest_rank([3.0, 1.0, 2.0], 1.0) == 1.0  # rank clamps to 1
    with pytest.raises(ValueError):
        percentile_nearest_rank([1.0], 0.0)
    with pytest.raises(ValueError):
        percentile_nearest_rank([1.0], 101.0)


# ---------------------------------------------------------------------------
# §31/§32 — pre_event_price_context
# ---------------------------------------------------------------------------


def _ramp(n: int, start: date = date(2025, 1, 1), first_close: float = 100.0) -> list[DailyBar]:
    """``n`` consecutive-calendar-day bars rising by 1 each — enough history to
    exercise the 20/50/200 windows without a market calendar."""
    from datetime import timedelta

    return [
        _bar(start + timedelta(days=i), first_close + i, volume=1_000_000.0)
        for i in range(n)
    ]


def test_pre_context_run_up_is_measured_from_the_anchor_close():
    bars = _ramp(30)
    ctx = pre_event_price_context(
        bars, anchor_date_et=bars[9].date, as_of_date_et=bars[-1].date
    )
    assert ctx.anchor_basis == "previous_event"
    assert ctx.anchor_close == 109.0
    assert ctx.since_anchor_return == pytest.approx(129.0 / 109.0 - 1.0)
    assert ctx.run_up_pct == ctx.since_anchor_return  # §32 label, same number


def test_pre_context_without_an_anchor_falls_back_to_the_default_window():
    bars = _ramp(120)
    ctx = pre_event_price_context(bars, anchor_date_et=None, as_of_date_et=bars[-1].date)
    assert ctx.anchor_basis == "default_63_bars"
    assert ctx.anchor_date_et == bars[-64].date


def test_pre_context_sma200_is_none_with_a_reason_under_200_bars():
    bars = _ramp(120)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    assert ctx.sma20 is not None
    assert ctx.sma50 is not None
    assert ctx.sma200 is None
    assert ctx.sma200_distance_pct is None
    assert ctx.reasons["sma200"] == "needs 200 bars, have 120"


def test_pre_context_sma_matches_indicators_sma():
    from libs.trading_core.features.indicators import sma

    bars = _ramp(60)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    closes = [b.close for b in bars]
    assert ctx.sma20 == pytest.approx(sma(closes, 20)[-1])
    assert ctx.sma20_distance_pct == pytest.approx(closes[-1] / sma(closes, 20)[-1] - 1.0)


def test_pre_context_atr_matches_indicators_atr():
    from libs.trading_core.features.indicators import atr

    bars = _ramp(40)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    expected = atr([b.high for b in bars], [b.low for b in bars], [b.close for b in bars], 14)[-1]
    assert ctx.atr14 == pytest.approx(expected)
    assert ctx.atr_pct == pytest.approx(expected / bars[-1].close)


def test_pre_context_realized_vol_matches_indicators_realized_vol():
    from libs.trading_core.features.indicators import realized_vol

    bars = _ramp(40)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    assert ctx.realized_vol_20d == pytest.approx(
        realized_vol([b.close for b in bars], 20)[-1]
    )


def test_pre_context_max_drawdown_is_negative_and_window_scoped():
    bars = _series(
        [
            (date(2026, 1, 5), 100.0),
            (date(2026, 1, 6), 120.0),
            (date(2026, 1, 7), 90.0),  # -25% from the 120 peak
            (date(2026, 1, 8), 110.0),
        ]
    )
    ctx = pre_event_price_context(
        bars, anchor_date_et=date(2026, 1, 5), as_of_date_et=date(2026, 1, 8)
    )
    assert ctx.max_drawdown == pytest.approx(90.0 / 120.0 - 1.0)


def test_pre_context_relative_return_is_the_difference_vs_the_benchmark():
    bars = _ramp(30)
    bench = [_bar(b.date, 200.0 + i * 0.5) for i, b in enumerate(bars)]
    ctx = pre_event_price_context(
        bars,
        anchor_date_et=bars[0].date,
        as_of_date_et=bars[-1].date,
        bench_bars=bench,
    )
    expected_bench = (200.0 + 29 * 0.5) / 200.0 - 1.0
    assert ctx.benchmark_return == pytest.approx(expected_bench)
    assert ctx.relative_return == pytest.approx(ctx.since_anchor_return - expected_bench)


def test_pre_context_without_a_benchmark_says_so():
    ctx = pre_event_price_context(_ramp(30), as_of_date_et=_ramp(30)[-1].date)
    assert ctx.benchmark_return is None
    assert ctx.relative_return is None
    assert ctx.reasons["relative_return"] == "no_benchmark_bars_provided"


def test_pre_context_respects_the_as_of_date_and_ignores_later_bars():
    bars = _ramp(30)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[9].date)
    assert ctx.bars_through == bars[9].date
    assert ctx.n_bars == 10
    assert ctx.last_close == 109.0


def test_pre_context_with_no_usable_bars_is_all_none_with_a_reason():
    bars = _ramp(5)
    ctx = pre_event_price_context(bars, as_of_date_et=date(2024, 1, 1))
    assert ctx.n_bars == 0
    assert ctx.last_close is None
    assert ctx.reasons["bars"] == "no_bars_available"


def test_pre_context_anchor_before_the_first_bar_is_flagged():
    bars = _ramp(30)
    ctx = pre_event_price_context(
        bars, anchor_date_et=date(2020, 1, 1), as_of_date_et=bars[-1].date
    )
    assert ctx.anchor_basis == "anchor_before_first_bar"
    assert ctx.since_anchor_return is None
    assert "bars unavailable before" in ctx.reasons["anchor"]


def test_pre_context_volume_trend_needs_80_bars():
    short = pre_event_price_context(_ramp(40), as_of_date_et=_ramp(40)[-1].date)
    assert short.volume_trend is None
    assert short.reasons["volume_trend"] == "needs 80 bars, have 40"


def test_pre_context_volume_trend_compares_last20_to_prior60():
    from datetime import timedelta

    bars = [
        _bar(date(2025, 1, 1) + timedelta(days=i), 100.0, volume=(2_000_000.0 if i >= 80 else 1_000_000.0))
        for i in range(100)
    ]
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    assert ctx.volume_trend == pytest.approx(1.0)  # 2M / 1M - 1


def test_pre_context_52w_distances_are_signed_from_the_last_close():
    bars = _ramp(30)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    assert ctx.distance_from_52w_high_pct is not None
    assert ctx.distance_from_52w_high_pct <= 0.0
    assert ctx.distance_from_52w_low_pct >= 0.0
    assert ctx.reasons["52w_window"] == "partial 52w window: 30 bars"


def test_pre_context_never_produces_nan_or_inf():
    bars = _ramp(120)
    ctx = pre_event_price_context(
        bars, anchor_date_et=bars[50].date, as_of_date_et=bars[-1].date, bench_bars=bars
    )
    from dataclasses import fields

    checked = 0
    for f in fields(ctx):
        value = getattr(ctx, f.name)
        if isinstance(value, float):
            assert math.isfinite(value), f.name
            checked += 1
    assert checked >= 15, f"only {checked} float fields populated"


def test_pre_context_rejects_unsorted_bars():
    bars = _ramp(5)
    with pytest.raises(ValueError, match="strictly increasing"):
        pre_event_price_context([bars[1], bars[0], bars[2]], as_of_date_et=bars[-1].date)


def test_every_none_field_on_a_degraded_context_carries_a_reason():
    """The house rule end-to-end: a 10-bar history cannot fill most windows,
    and every window it cannot fill names itself in ``reasons``."""
    bars = _ramp(10)
    ctx = pre_event_price_context(bars, as_of_date_et=bars[-1].date)
    for field_name in ("sma20", "sma50", "sma200", "atr14", "realized_vol_20d", "volume_trend"):
        assert getattr(ctx, field_name) is None
        assert field_name in ctx.reasons


def test_pre_context_realized_vol_since_anchor_needs_three_bars():
    bars = _ramp(30)
    two_bar = pre_event_price_context(
        bars, anchor_date_et=bars[-2].date, as_of_date_et=bars[-1].date
    )
    assert two_bar.realized_vol_since_anchor is None
    assert two_bar.reasons["realized_vol_since_anchor"] == (
        "needs 3 bars since the anchor, have 2"
    )
    three_bar = pre_event_price_context(
        bars, anchor_date_et=bars[-3].date, as_of_date_et=bars[-1].date
    )
    assert three_bar.realized_vol_since_anchor is not None


def test_unknown_session_reaction_is_measured_over_the_wider_span():
    """End-to-end counterpart to the index-level UNKNOWN test: the conservative
    span must produce a LARGER measured move than the same-day window on a
    steadily rising tape, and label its basis."""
    during = event_reaction(WEEKS, date(2026, 1, 8), DURING, horizons=(1,))
    unknown = event_reaction(WEEKS, date(2026, 1, 8), UNKNOWN, horizons=(1,))
    assert during.basis == BASIS_DURING_MARKET
    assert unknown.basis == BASIS_UNKNOWN
    # Same pre-event close (01-07), but UNKNOWN's 1D lands on 01-09 not 01-08.
    assert during.pre_event_close == unknown.pre_event_close == 102.0
    assert during.returns[1] == pytest.approx(103.0 / 102.0 - 1.0)
    assert unknown.returns[1] == pytest.approx(104.0 / 102.0 - 1.0)
    assert unknown.returns[1] > during.returns[1]
