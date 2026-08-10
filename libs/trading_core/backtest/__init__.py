"""Backtest Engine V1 (development plan Phase 3, §20).

Pure, deterministic replay of the shared signal engines
(:mod:`libs.trading_core.signals` — the exact code the live path runs,
plan §21) over daily OHLCV bars. V1 is LONG STOCK ONLY (plan §5): no
shorting ever, and Long Call / Long Put backtesting is deferred until real
option-chain data exists — option prices are never fabricated.
"""
from .engine import (  # noqa: F401
    BacktestParams,
    BacktestResult,
    SegmentMetrics,
    Trade,
    run_backtest,
)

__all__ = [
    "BacktestParams",
    "BacktestResult",
    "SegmentMetrics",
    "Trade",
    "run_backtest",
]
