"""Exit Engine v0 (development plan §11).

Pure, deterministic exit decisions for open LONG_STOCK and long-option
paper positions — no DB, no FastAPI. Signal logic is REUSED verbatim from
:mod:`libs.trading_core.signals` (plan §21 — never reimplemented here), and
the rule priority mirrors the backtest engine
(:mod:`libs.trading_core.backtest`) so paper positions are managed by the
same rules the backtest validated. Option positions share the SAME
underlying-driven rule internals (plan §21) with PREMIUM_HARD_STOP (§11.3)
and DTE_EXIT (§11.7) in front, replacing the underlying HARD_STOP. An exit
decision can only ever map to a SELL_TO_CLOSE — Sell-to-Open does not exist
anywhere in this system (plan §5).
"""
from .engine import (  # noqa: F401
    ExitDecision,
    ExitParams,
    OptionState,
    PositionState,
    ShortPremiumState,
    evaluate_exit,
    evaluate_option_exit,
    evaluate_short_premium_exit,
)

__all__ = [
    "ExitDecision",
    "ExitParams",
    "OptionState",
    "PositionState",
    "ShortPremiumState",
    "evaluate_exit",
    "evaluate_option_exit",
    "evaluate_short_premium_exit",
]
