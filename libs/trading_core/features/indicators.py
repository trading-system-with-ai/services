"""Technical indicators for the feature engine (development plan Phase 2, §6).

Pure, deterministic, dependency-free functions usable identically in backtest
and live (plan §21):

- Inputs are plain Python lists (or sequences) of floats.
- Every function returns a list of the SAME LENGTH as its input, left-padded
  with ``None`` (``False`` for pivot flags) during the warmup region where the
  indicator is not yet defined. No value is ever fabricated for the warmup
  bars and no future bar is ever read (plan §20.3: no look-ahead).
- Every period / window / annualization factor is a parameter with a sensible
  default; nothing is hardcoded inside the logic.
"""
from __future__ import annotations

import math
from typing import Sequence


def _validate_period(period: int, name: str = "period", minimum: int = 1) -> None:
    """Reject non-sensical periods early with a clear error."""
    if not isinstance(period, int) or period < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {period!r}")


def _validate_ohlc(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> None:
    """OHLC series must be aligned bar-for-bar."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError(
            "highs, lows and closes must have equal length, got "
            f"{len(highs)}/{len(lows)}/{len(closes)}"
        )


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average (plan §6).

    Warmup: indices ``0 .. period-2`` are ``None``; the first value appears at
    index ``period-1`` (the first bar with a full window of history).
    """
    _validate_period(period)
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    window_sum = math.fsum(values[:period])
    out[period - 1] = window_sum / period
    for t in range(period, n):
        window_sum += values[t] - values[t - period]
        out[t] = window_sum / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average (plan §6), seeded with the SMA of the first
    ``period`` values so backtest and live warm up identically (plan §21).

    Smoothing factor ``alpha = 2 / (period + 1)``. Warmup: indices
    ``0 .. period-2`` are ``None``; the seed appears at index ``period-1``.
    """
    _validate_period(period)
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = math.fsum(values[:period]) / period  # SMA seed
    out[period - 1] = prev
    for t in range(period, n):
        prev = alpha * values[t] + (1.0 - alpha) * prev
        out[t] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index with Wilder smoothing (plan §6).

    The first RSI appears at index ``period`` (it needs ``period`` price
    changes, i.e. ``period + 1`` prices). The seed averages are the simple
    means of the first ``period`` gains/losses; afterwards Wilder smoothing:
    ``avg = (prev_avg * (period - 1) + current) / period``.

    Conventions: if the average loss is zero the RSI is 100.0; if both average
    gain and average loss are zero (flat series) the RSI is the neutral 50.0.
    """
    _validate_period(period)
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out

    def _rsi_value(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0:
            return 50.0 if avg_gain == 0.0 else 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    deltas = [values[t] - values[t - 1] for t in range(1, period + 1)]
    avg_gain = math.fsum(d for d in deltas if d > 0.0) / period
    avg_loss = math.fsum(-d for d in deltas if d < 0.0) / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for t in range(period + 1, n):
        delta = values[t] - values[t - 1]
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[t] = _rsi_value(avg_gain, avg_loss)
    return out


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float | None]:
    """True range per bar (plan §6):
    ``max(high - low, |high - prev_close|, |low - prev_close|)``.

    Index 0 is ``None`` — the first bar has no previous close and we never
    fabricate warmup data (plan §20.3).
    """
    _validate_ohlc(highs, lows, closes)
    n = len(highs)
    out: list[float | None] = [None] * n
    for t in range(1, n):
        prev_close = closes[t - 1]
        out[t] = max(
            highs[t] - lows[t],
            abs(highs[t] - prev_close),
            abs(lows[t] - prev_close),
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Average True Range with Wilder smoothing (plan §6).

    Seeded with the simple mean of the first ``period`` true ranges (bars
    ``1 .. period``, since TR is undefined on bar 0), so the first ATR appears
    at index ``period``; afterwards Wilder smoothing:
    ``atr = (prev_atr * (period - 1) + tr) / period``.
    """
    _validate_period(period)
    tr = true_range(highs, lows, closes)
    n = len(tr)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out
    prev = math.fsum(tr[1 : period + 1]) / period  # type: ignore[arg-type]
    out[period] = prev
    for t in range(period + 1, n):
        prev = (prev * (period - 1) + tr[t]) / period  # type: ignore[operator]
        out[t] = prev
    return out


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[float | None]]:
    """MACD (plan §6): ``macd = EMA(fast) - EMA(slow)``, ``signal`` is the
    EMA of the MACD line (seeded with its SMA, same convention as :func:`ema`),
    ``histogram = macd - signal``.

    Returns ``{"macd": [...], "signal": [...], "histogram": [...]}``, each the
    same length as the input. Warmup: the MACD line starts at index
    ``slow - 1``; the signal line and histogram start ``signal - 1`` bars
    later, at index ``slow + signal - 2``.
    """
    _validate_period(fast, "fast")
    _validate_period(slow, "slow")
    _validate_period(signal, "signal")
    if fast >= slow:
        raise ValueError(f"fast period must be < slow period, got {fast} >= {slow}")

    n = len(values)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: list[float | None] = [
        f - s if f is not None and s is not None else None
        for f, s in zip(ema_fast, ema_slow)
    ]

    signal_line: list[float | None] = [None] * n
    if n >= slow:
        start = slow - 1  # first index where the MACD line is defined
        compact = macd_line[start:]  # contiguous, no Nones
        signal_line = [None] * start + ema(compact, signal)  # type: ignore[arg-type]

    histogram: list[float | None] = [
        m - s if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def realized_vol(
    closes: Sequence[float],
    period: int = 20,
    annualization: int = 252,
) -> list[float | None]:
    """Realized (historical) volatility (plan §6): sample standard deviation
    (ddof=1) of the last ``period`` close-to-close log returns, annualized by
    ``sqrt(annualization)``.

    ``annualization`` is the number of return periods per year (252 trading
    days for daily bars) — a parameter, never a hardcoded truth. The first
    value appears at index ``period`` (a bar's value uses only returns ending
    at that bar — no look-ahead, plan §20.3).
    """
    _validate_period(period, minimum=2)  # sample stdev needs >= 2 returns
    _validate_period(annualization, "annualization")
    n = len(closes)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out
    returns = [math.log(closes[t] / closes[t - 1]) for t in range(1, n)]
    for t in range(period, n):
        # returns[i] is the return of price bar i+1; the window of the last
        # `period` returns ending at price bar t is returns[t-period : t].
        window = returns[t - period : t]
        mean = math.fsum(window) / period
        variance = math.fsum((r - mean) ** 2 for r in window) / (period - 1)
        out[t] = math.sqrt(variance) * math.sqrt(annualization)
    return out


def pivot_highs(highs: Sequence[float], window: int = 5) -> list[bool]:
    """Confirmed pivot highs (plan §6.3).

    Bar ``t`` is a pivot high iff ``highs[t]`` is STRICTLY greater than the
    highs of the ``window`` bars on EACH side. A pivot needs ``window`` future
    bars to be confirmed, so the final ``window`` bars are always ``False`` —
    we never pretend a not-yet-confirmable pivot exists (no look-ahead,
    plan §20.3).
    """
    _validate_period(window, "window")
    n = len(highs)
    out = [False] * n
    for t in range(window, n - window):
        h = highs[t]
        if all(
            h > highs[t - i] and h > highs[t + i] for i in range(1, window + 1)
        ):
            out[t] = True
    return out


def pivot_lows(lows: Sequence[float], window: int = 5) -> list[bool]:
    """Confirmed pivot lows (plan §6.3). Mirror of :func:`pivot_highs`:
    bar ``t`` is a pivot low iff ``lows[t]`` is strictly lower than the lows
    of the ``window`` bars on each side; the final ``window`` bars are always
    ``False`` (no look-ahead, plan §20.3).
    """
    _validate_period(window, "window")
    n = len(lows)
    out = [False] * n
    for t in range(window, n - window):
        low = lows[t]
        if all(
            low < lows[t - i] and low < lows[t + i] for i in range(1, window + 1)
        ):
            out[t] = True
    return out
