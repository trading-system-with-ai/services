"""Shared request-time guards for the gateway (§44 rule 18: honest absence).

THE RULE THIS MODULE ENFORCES: Massive is the only data source. Everything the
platform reports is either Massive raw data or computed from it. When no market
data provider is configured there is NO fallback — the affected endpoints
answer HTTP 503 and the UI shows nothing. Synthetic prices, bars, chains or
recommendations must never reach a user (the stub providers exist for tests and
local development, reachable only by explicitly opting in).

ONE helper per missing dependency, used by EVERY affected route, so no route
can forget the check and the error body can never drift between endpoints:

- :func:`require_market_data_provider` -> 503 ``MARKET_DATA_NOT_CONFIGURED``
- :func:`require_llm_provider`         -> 503 ``LLM_NOT_CONFIGURED``

Deliberately NOT applied to endpoints whose content is real DB state:
``GET /api/positions`` (positions are real rows the user actually holds) and
``GET /api/portfolio/risk`` (NAV / cash / positions come from the database)
stay 200 and report their market-derived fields as honest NULLS instead — see
:func:`market_data_status`. Hiding a real position because a quote is missing
would be its own kind of dishonesty.
"""
from fastapi import HTTPException

from libs.common.config import get_settings
from libs.llm import LLMProviderNotConfigured
from libs.market_data import ProviderNotConfigured

# Machine-readable error codes carried in the 503 detail block.
MARKET_DATA_NOT_CONFIGURED = "MARKET_DATA_NOT_CONFIGURED"
LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"


def market_data_configured() -> bool:
    """True when a market data provider name is configured (non-blank)."""
    name = get_settings().market_data_provider
    return bool(name and name.strip())


def llm_configured() -> bool:
    """True when an LLM provider name is configured (non-blank)."""
    name = get_settings().llm_provider
    return bool(name and name.strip())


def market_data_unavailable(exc: ProviderNotConfigured) -> HTTPException:
    """The canonical 503 for a missing market data provider."""
    return HTTPException(
        status_code=503,
        detail={"code": MARKET_DATA_NOT_CONFIGURED, "message": str(exc)},
    )


def require_market_data_provider() -> None:
    """Raise 503 ``MARKET_DATA_NOT_CONFIGURED`` when no provider is configured.

    Call this FIRST in any route that would otherwise report prices, bars,
    quotes, option chains or anything computed from them. The 503 body is
    ``{"detail": {"code": ..., "message": ...}}`` where the message is the
    provider layer's own text, so the operator is told exactly which setting is
    missing.
    """
    if not market_data_configured():
        raise market_data_unavailable(ProviderNotConfigured())


def require_llm_provider() -> None:
    """Raise 503 ``LLM_NOT_CONFIGURED`` when no LLM provider is configured."""
    if not llm_configured():
        raise HTTPException(
            status_code=503,
            detail={"code": LLM_NOT_CONFIGURED, "message": str(LLMProviderNotConfigured())},
        )


def market_data_status() -> dict:
    """The ``market_data`` block for responses that degrade instead of 503-ing.

    ``{"configured": bool, "message": str | None}`` — the message names the
    missing configuration when unset, and is null when a provider is
    configured. Lets a client render "no market data" honestly next to the real
    DB-derived numbers it IS showing (NAV, cash, position quantities).
    """
    if market_data_configured():
        return {"configured": True, "message": None}
    return {"configured": False, "message": str(ProviderNotConfigured())}
