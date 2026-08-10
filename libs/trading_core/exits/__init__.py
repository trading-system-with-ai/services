"""Exit Engine v0 (development plan §11).

Pure, deterministic exit decisions for open LONG_STOCK paper positions — no
DB, no FastAPI. Signal logic is REUSED verbatim from
:mod:`libs.trading_core.signals` (plan §21 — never reimplemented here), and
the rule priority mirrors the backtest engine
(:mod:`libs.trading_core.backtest`) so paper positions are managed by the
same rules the backtest validated. An exit decision can only ever map to a
SELL_TO_CLOSE — Sell-to-Open does not exist anywhere in this system
(plan §5).
"""
from .engine import (  # noqa: F401
    ExitDecision,
    ExitParams,
    PositionState,
    evaluate_exit,
)

__all__ = [
    "ExitDecision",
    "ExitParams",
    "PositionState",
    "evaluate_exit",
]
