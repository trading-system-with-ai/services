"""Backtest engines (development plan Phase 3, §20).

Pure, deterministic replay of the shared signal engines
(:mod:`libs.trading_core.signals` — the exact code the live path runs,
plan §21) over daily OHLCV bars. Two legs:

- ``run_backtest`` — LONG_STOCK; ``run_short_stock_backtest`` — the Phase 3
  margin-backed bear mirror (2026-08-17 unlock).
- ``run_call_backtest`` — LONG_CALL over REAL historical contract bars
  (2026-08-17: Alpaca serves full contract-life daily bars from ~Feb 2024,
  so the fabrication ban holds — option prices are never invented; see
  backtest/options.py and data-source-architecture.md).
"""
from .auto import (  # noqa: F401
    AutoDecision,
    run_auto_backtest,
)
from .engine import (  # noqa: F401
    BacktestParams,
    BacktestResult,
    SegmentMetrics,
    Trade,
    run_backtest,
    run_short_stock_backtest,
)
from .portfolio import (  # noqa: F401
    PortfolioBacktestResult,
    SymbolBars,
    run_portfolio_backtest,
)
from .options import (  # noqa: F401
    ContractProvider,
    OptionLegBars,
    OptionTrade,
    SpreadLegBars,
    SpreadProvider,
    run_call_backtest,
    run_covered_call_backtest,
    run_csp_backtest,
    run_spread_backtest,
)

__all__ = [
    "AutoDecision",
    "BacktestParams",
    "BacktestResult",
    "ContractProvider",
    "OptionLegBars",
    "PortfolioBacktestResult",
    "SymbolBars",
    "OptionTrade",
    "SegmentMetrics",
    "Trade",
    "SpreadLegBars",
    "SpreadProvider",
    "run_auto_backtest",
    "run_backtest",
    "run_portfolio_backtest",
    "run_short_stock_backtest",
    "run_call_backtest",
    "run_covered_call_backtest",
    "run_csp_backtest",
    "run_spread_backtest",
]
