"""Signal engines (development plan Phase 2, §6).

Pure, deterministic library code — no DB, no FastAPI — shared verbatim by the
backtest and live paths (plan §21):

- :mod:`.regime` — Market Regime Engine v0 (plan §6.1).
- :mod:`.directional` — Directional Signal Engine v0 (plan §6.2).

All indicator math comes exclusively from :mod:`libs.trading_core.features`;
every threshold is a parameter, never a hardcoded truth (plan §6.2).
"""
from .directional import (  # noqa: F401
    DirectionalParams,
    DirectionalResult,
    SignalComponent,
    score_direction,
)
from .regime import RegimeParams, RegimeResult, classify_regime  # noqa: F401

__all__ = [
    "DirectionalParams",
    "DirectionalResult",
    "RegimeParams",
    "RegimeResult",
    "SignalComponent",
    "classify_regime",
    "score_direction",
]
