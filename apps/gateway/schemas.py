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
