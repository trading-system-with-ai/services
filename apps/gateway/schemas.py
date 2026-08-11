"""API request/response schemas for the gateway."""
import re
from datetime import datetime

from pydantic import BaseModel, field_validator

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class TickerRequest(BaseModel):
    ticker: str
    note: str = ""

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.match(v):
            raise ValueError(f"invalid ticker: {v!r}")
        return v


class WatchlistItemOut(BaseModel):
    ticker: str
    added_by: str
    note: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TradingPoolItemOut(BaseModel):
    ticker: str
    trading_enabled: bool
    allowed_strategies: list[str]
    promoted_by: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TradingPoolAddRequest(TickerRequest):
    # V1 account constraints: long-only defined-risk instruments
    allowed_strategies: list[str] = ["LONG_STOCK", "LONG_CALL", "LONG_PUT"]
    # §4.3: a promotion whose readiness checks fail may still proceed when the
    # user explicitly acknowledges the risk characteristics — the override is
    # recorded permanently in the TRADING_POOL_ADD audit details.
    acknowledge_risks: bool = False


class PromotionCheck(BaseModel):
    """One §4.3 promotion readiness check result (honest detail, §44 rule 18)."""

    name: str
    passed: bool
    detail: str


class TradingPoolPromotedOut(TradingPoolItemOut):
    """POST /api/trading-pool response: the created row plus the §4.3
    promotion-check results and whether the user overrode failures."""

    promotion_checks: list[PromotionCheck]
    risks_acknowledged: bool


class TradingToggleRequest(BaseModel):
    enabled: bool


class AuditEventOut(BaseModel):
    id: int
    ts: datetime
    actor_type: str
    actor_id: str
    action: str
    entity_type: str
    entity_id: str
    details: dict
    correlation_id: str

    model_config = {"from_attributes": True}
