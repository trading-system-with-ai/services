"""Contract Selector v0 (development plan §9).

Pure, deterministic library code — no DB, no FastAPI, no market data
fetching. Given an option chain snapshot and a directional bias, it filters
(§9.1) and ranks (§9.2 v0 heuristic; Phase 10 upgrades ranking to EV-based)
long calls/puts for a long-only account (plan §5). Every contract comes back
with an eligibility verdict so the UI can render All / Eligible / Recommended
views (plan §34). Chain reads are read-only — no audit events here.
"""
from .selector import (  # noqa: F401
    ContractQuote,
    ScoredContract,
    SelectorParams,
    select_contracts,
)
from .spreads import (  # noqa: F401
    SpreadCandidate,
    SpreadParams,
    SpreadSelection,
    select_vertical_spread,
)

__all__ = [
    "ContractQuote",
    "ScoredContract",
    "SelectorParams",
    "SpreadCandidate",
    "SpreadParams",
    "SpreadSelection",
    "select_contracts",
    "select_vertical_spread",
]
