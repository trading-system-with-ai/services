"""Feature engine (development plan Phase 2, §6).

Pure, deterministic, dependency-free indicator functions shared verbatim by
backtest and live paths (plan §21).
"""
from .indicators import (  # noqa: F401
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

__all__ = [
    "atr",
    "ema",
    "macd",
    "pivot_highs",
    "pivot_lows",
    "realized_vol",
    "rsi",
    "sma",
    "true_range",
]
