"""Strategy-level decisions (development plan §8).

Pure, deterministic library code — no DB, no FastAPI. Today this package
holds Instrument Selection v1: the §8 (direction, strength, vol regime) ->
instrument matrix, degraded to the §5 account constraints (no short stock,
no naked short options, spreads only when permitted AND implemented).
"""
from .instrument import (  # noqa: F401
    AccountPermissions,
    InstrumentDecision,
    select_instrument,
)

__all__ = [
    "AccountPermissions",
    "InstrumentDecision",
    "select_instrument",
]
