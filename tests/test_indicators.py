"""Tests for the feature engine indicators (development plan Phase 2, §6).

Reference values are computed by hand (arithmetic shown in comments) and the
structural contract — same-length output, None/False warmup, determinism,
short-input handling, no look-ahead (plan §20.3) — is checked for every
function.
"""
import math

import pytest

from libs.trading_core.features import (
    atr,
    ema,
    macd,
    pivot_highs,
    pivot_lows,
    realized_vol,
    rsi,
    sma,
    true_range,
)

approx = pytest.approx


# ---------------------------------------------------------------------------
# Reference-value tests (hand-computed)
# ---------------------------------------------------------------------------


def test_sma_reference_values():
    # values [1,2,3,4,5], period 3:
    #   idx2 = (1+2+3)/3 = 2, idx3 = (2+3+4)/3 = 3, idx4 = (3+4+5)/3 = 4
    assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 3) == [None, None, approx(2.0), approx(3.0), approx(4.0)]


def test_ema_reference_values():
    # values [1,2,3,4,5], period 3, alpha = 2/(3+1) = 0.5:
    #   seed (SMA of first 3) at idx2 = (1+2+3)/3 = 2
    #   idx3 = 0.5*4 + 0.5*2 = 3
    #   idx4 = 0.5*5 + 0.5*3 = 4
    assert ema([1.0, 2.0, 3.0, 4.0, 5.0], 3) == [None, None, approx(2.0), approx(3.0), approx(4.0)]


def test_ema_is_seeded_with_sma():
    values = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0]
    period = 4
    result = ema(values, period)
    # The first defined EMA must equal the SMA of the first `period` values:
    # (10+12+11+13)/4 = 46/4 = 11.5
    assert result[period - 1] == approx(11.5)
    assert result[period - 1] == approx(sma(values, period)[period - 1])


def test_rsi_wilder_smoothing_reference_values():
    # values [44, 45, 46, 45, 47, 46], period 3.
    # Price changes: +1, +1, -1, +2, -1.
    #
    # Seed (simple means of the first 3 changes), first RSI at idx3:
    #   avg_gain = (1+1+0)/3 = 2/3,  avg_loss = (0+0+1)/3 = 1/3
    #   RS = 2, RSI = 100 - 100/(1+2) = 66.6667
    #
    # idx4 (change +2), Wilder: avg = (prev*(period-1) + current)/period
    #   avg_gain = (2/3*2 + 2)/3 = 10/9,  avg_loss = (1/3*2 + 0)/3 = 2/9
    #   RS = 5, RSI = 100 - 100/6 = 83.3333
    #
    # idx5 (change -1):
    #   avg_gain = (10/9*2 + 0)/3 = 20/27, avg_loss = (2/9*2 + 1)/3 = 13/27
    #   RS = 20/13, RSI = 100 - 100/(33/13) = 100 - 1300/33 = 60.6061
    result = rsi([44.0, 45.0, 46.0, 45.0, 47.0, 46.0], period=3)
    assert result[:3] == [None, None, None]
    assert result[3] == approx(100.0 - 100.0 / 3.0)
    assert result[4] == approx(100.0 - 100.0 / 6.0)
    assert result[5] == approx(100.0 - 1300.0 / 33.0)


def test_rsi_all_gains_is_100():
    # Monotonic rise: avg_loss = 0 at every defined bar -> RSI = 100.
    result = rsi([1.0, 2.0, 3.0, 4.0, 5.0], period=3)
    assert result[3] == approx(100.0)
    assert result[4] == approx(100.0)


def test_true_range_reference_values():
    # bars:      0            1             2
    # highs   [10, 12, 11], lows [8, 9, 9], closes [9, 11, 10]
    # TR[0] = None (no previous close, never fabricated)
    # TR[1] = max(12-9, |12-9|, |9-9|)   = max(3, 3, 0) = 3
    # TR[2] = max(11-9, |11-11|, |9-11|) = max(2, 0, 2) = 2
    result = true_range([10.0, 12.0, 11.0], [8.0, 9.0, 9.0], [9.0, 11.0, 10.0])
    assert result == [None, approx(3.0), approx(2.0)]


def test_atr_wilder_smoothing_reference_values():
    highs = [10.0, 12.0, 11.0, 13.0]
    lows = [8.0, 9.0, 9.0, 10.0]
    closes = [9.0, 11.0, 10.0, 12.0]
    # TR = [None, 3, 2, max(13-10, |13-10|, |10-10|) = 3]
    # period=2 seed at idx2: mean(TR[1], TR[2]) = (3+2)/2 = 2.5
    # idx3 Wilder: (2.5*(2-1) + 3)/2 = 5.5/2 = 2.75
    result = atr(highs, lows, closes, period=2)
    assert result == [None, None, approx(2.5), approx(2.75)]


def test_macd_reference_values():
    # values 1..10, fast=3, slow=5, signal=3.
    # EMA(3): seed idx2 = (1+2+3)/3 = 2, alpha = 0.5:
    #   idx3 = 0.5*4+0.5*2 = 3; idx4 = 4; ... -> ema3[t] = t for t >= 2
    # EMA(5): seed idx4 = (1+2+3+4+5)/5 = 3, alpha = 1/3:
    #   idx5 = 6/3 + 2*3/3 = 4; idx6 = 7/3 + 8/3 = 5; idx7 = 6; idx8 = 7;
    #   idx9 = 8  -> ema5[t] = t - 1 for t >= 5 (and 3 at idx4)
    # MACD line: idx4 = 4-3 = 1; idx5.. = t - (t-1) = 1  -> constant 1.0
    # Signal (EMA(3) of the MACD line, seeded with its SMA):
    #   first defined at idx 4 + (3-1) = 6: (1+1+1)/3 = 1; stays 1.
    # Histogram: 1 - 1 = 0 from idx6 on.
    values = [float(v) for v in range(1, 11)]
    result = macd(values, fast=3, slow=5, signal=3)

    assert result["macd"][:4] == [None] * 4
    for t in range(4, 10):
        assert result["macd"][t] == approx(1.0)

    assert result["signal"][:6] == [None] * 6
    for t in range(6, 10):
        assert result["signal"][t] == approx(1.0)
        assert result["histogram"][t] == approx(0.0)
    assert result["histogram"][:6] == [None] * 6


def test_macd_matches_independent_emas():
    # The MACD line must equal EMA(fast) - EMA(slow) computed independently.
    values = [100.0, 101.5, 99.8, 102.3, 103.1, 101.9, 104.2, 105.0, 103.6, 106.1, 107.4, 106.8]
    fast, slow = 3, 5
    result = macd(values, fast=fast, slow=slow, signal=3)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    for t in range(len(values)):
        if ema_slow[t] is None:
            assert result["macd"][t] is None
        else:
            assert result["macd"][t] == approx(ema_fast[t] - ema_slow[t])


def test_realized_vol_reference_values():
    # closes [100, 110, 100, 110], period=2, annualization=252.
    # Log returns: r1 = ln(1.1), r2 = ln(100/110) = -ln(1.1), r3 = ln(1.1)
    # idx2 window [r1, r2]: mean = 0
    #   sample var = (ln(1.1)^2 + ln(1.1)^2)/(2-1) = 2*ln(1.1)^2
    #   stdev = ln(1.1)*sqrt(2); annualized = ln(1.1)*sqrt(2)*sqrt(252)
    # idx3 window [r2, r3]: same magnitudes -> same value.
    r = math.log(1.1)
    expected = r * math.sqrt(2.0) * math.sqrt(252.0)
    result = realized_vol([100.0, 110.0, 100.0, 110.0], period=2, annualization=252)
    assert result == [None, None, approx(expected), approx(expected)]


def test_realized_vol_annualization_is_parameterized():
    closes = [100.0, 110.0, 100.0, 110.0]
    daily = realized_vol(closes, period=2, annualization=252)
    hourly = realized_vol(closes, period=2, annualization=252 * 7)
    # Same stdev, different annualization factor: ratio must be sqrt(7).
    assert hourly[2] == approx(daily[2] * math.sqrt(7.0))


def test_pivot_highs_reference_values():
    #        idx:  0  1  2  3  4  5  6  7
    highs = [1.0, 3.0, 2.0, 5.0, 4.0, 3.0, 6.0, 2.0]
    # window=2: candidates are idx 2..5.
    #   idx3: 5 > {3, 2} on the left and 5 > {4, 3} on the right -> pivot.
    #   idx2 (2 > 3 fails), idx4 (4 > 5 fails), idx5 (3 > 4 fails) -> not pivots.
    #   idx6 has the global max (6) but sits in the final `window` bars,
    #   so it can never be confirmed -> False (no look-ahead, plan §20.3).
    assert pivot_highs(highs, window=2) == [False, False, False, True, False, False, False, False]


def test_pivot_lows_reference_values():
    #       idx:  0  1  2  3  4  5  6  7
    lows = [5.0, 2.0, 4.0, 1.0, 3.0, 4.0, 0.0, 5.0]
    # window=2: idx3: 1 < {2, 4} left and 1 < {3, 4} right -> pivot.
    # idx6 is the global min (0) but is inside the final `window` bars -> False.
    assert pivot_lows(lows, window=2) == [False, False, False, True, False, False, False, False]


def test_pivots_require_strict_inequality():
    # Plateau: equal highs must NOT be pivots (strictly greater required, §6.3).
    assert pivot_highs([1.0, 2.0, 2.0, 2.0, 1.0], window=1) == [False] * 5
    assert pivot_lows([2.0, 1.0, 1.0, 1.0, 2.0], window=1) == [False] * 5


# ---------------------------------------------------------------------------
# Structural tests for every function
# ---------------------------------------------------------------------------

CLOSES = [100.0, 101.0, 99.5, 102.0, 103.5, 102.5, 104.0, 106.0, 105.0, 107.5,
          108.0, 106.5, 109.0, 110.5, 109.5, 111.0, 112.5, 111.5, 113.0, 114.5]
HIGHS = [c + 1.0 for c in CLOSES]
LOWS = [c - 1.0 for c in CLOSES]

# (name, callable of no args, warmup length, warmup padding value)
CASES = [
    ("sma", lambda: sma(CLOSES, 5), 4, None),
    ("ema", lambda: ema(CLOSES, 5), 4, None),
    ("rsi", lambda: rsi(CLOSES, period=5), 5, None),
    ("true_range", lambda: true_range(HIGHS, LOWS, CLOSES), 1, None),
    ("atr", lambda: atr(HIGHS, LOWS, CLOSES, period=5), 5, None),
    ("realized_vol", lambda: realized_vol(CLOSES, period=5, annualization=252), 5, None),
    ("pivot_highs", lambda: pivot_highs(HIGHS, window=3), 3, False),
    ("pivot_lows", lambda: pivot_lows(LOWS, window=3), 3, False),
]


@pytest.mark.parametrize("name,compute,warmup,pad", CASES, ids=[c[0] for c in CASES])
def test_output_length_and_warmup(name, compute, warmup, pad):
    result = compute()
    assert len(result) == len(CLOSES)
    assert result[:warmup] == [pad] * warmup
    if pad is None:
        # Post-warmup region is fully defined.
        assert all(v is not None for v in result[warmup:])


@pytest.mark.parametrize("name,compute,warmup,pad", CASES, ids=[c[0] for c in CASES])
def test_deterministic(name, compute, warmup, pad):
    assert compute() == compute()


def test_macd_length_warmup_and_determinism():
    result = macd(CLOSES, fast=3, slow=5, signal=3)
    assert set(result) == {"macd", "signal", "histogram"}
    for series in result.values():
        assert len(series) == len(CLOSES)
    # MACD line warmup ends at slow-1 = idx4; signal/histogram at slow+signal-2 = idx6.
    assert result["macd"][:4] == [None] * 4
    assert all(v is not None for v in result["macd"][4:])
    assert result["signal"][:6] == [None] * 6
    assert all(v is not None for v in result["signal"][6:])
    assert result["histogram"][:6] == [None] * 6
    assert all(v is not None for v in result["histogram"][6:])
    assert macd(CLOSES, fast=3, slow=5, signal=3) == result


def test_short_input_all_none():
    short = CLOSES[:3]
    short_h, short_l = HIGHS[:3], LOWS[:3]
    assert sma(short, 5) == [None] * 3
    assert ema(short, 5) == [None] * 3
    assert rsi(short, period=5) == [None] * 3
    assert atr(short_h, short_l, short, period=5) == [None] * 3
    assert realized_vol(short, period=5) == [None] * 3
    # MACD: input shorter than the slow period -> every series is all None.
    result = macd(short, fast=3, slow=5, signal=3)
    for series in result.values():
        assert series == [None] * 3
    # Pivots: input shorter than the confirmation span -> all False.
    assert pivot_highs(short_h, window=5) == [False] * 3
    assert pivot_lows(short_l, window=5) == [False] * 3
    # Degenerate but legal: empty input.
    assert sma([], 5) == []
    assert true_range([], [], []) == []
    assert pivot_highs([], window=5) == []


def test_pivots_never_flag_final_window_bars():
    """No look-ahead (plan §20.3): the last `window` bars can never be
    confirmed pivots, even when they are the extreme of the whole series."""
    for window in (1, 2, 3, 5):
        n = 30
        highs = [float(i) for i in range(n)]  # strictly rising: max is last bar
        lows = [float(n - i) for i in range(n)]  # strictly falling: min is last bar
        assert pivot_highs(highs, window=window)[-window:] == [False] * window
        assert pivot_lows(lows, window=window)[-window:] == [False] * window


def test_pivot_confirmation_is_stable_as_bars_arrive():
    """Backtest/live parity (plan §21): once confirmed on the full series, a
    pivot flag must be identical to the flag computed on any longer series."""
    highs = [1.0, 3.0, 2.0, 5.0, 4.0, 3.0, 6.0, 2.0, 1.0, 0.5]
    window = 2
    full = pivot_highs(highs, window=window)
    for cut in range(len(highs) + 1):
        partial = pivot_highs(highs[:cut], window=window)
        # Every bar already confirmable in the partial series agrees with the
        # full-series result.
        confirmable = max(0, cut - window)
        assert partial[:confirmable] == full[:confirmable]
