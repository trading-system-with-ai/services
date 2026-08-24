"""Strategy-level decisions (development plan §8).

Pure, deterministic library code — no DB, no FastAPI. Today this package
holds Instrument Selection v1: the §8 (direction, strength, vol regime) ->
instrument matrix, degraded to the §5 account constraints (no short stock,
no naked short options, spreads only when permitted AND implemented).
"""
from .income import (  # noqa: F401
    IncomeParams,
    IncomeSelection,
    select_cash_secured_put,
    select_covered_call,
)
from .instrument import (  # noqa: F401
    FORBIDDEN_PERMISSION_FIELDS,
    AccountPermissions,
    InstrumentDecision,
    select_instrument,
)

__all__ = [
    "IncomeParams",
    "IncomeSelection",
    "select_cash_secured_put",
    "select_covered_call",
    "FORBIDDEN_PERMISSION_FIELDS",
    "AccountPermissions",
    "InstrumentDecision",
    "select_instrument",
]
