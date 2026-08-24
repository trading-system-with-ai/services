"""Read-only configuration API (plan §28 settings view, §44 rule 2).

"Every rule must be configuration-driven" is only trustworthy when the
configuration is VISIBLE: ``GET /api/config`` renders the REAL engine
parameter objects — the same dataclass defaults the risk / exit / selector /
signal / allocation / backtest engines actually run with — plus provider
selection, paper-account economics and the live kill-switch state.

HARD RULE (no secret material): every group is built with
``dataclasses.asdict`` of the real parameter objects or from individually
named non-secret Settings fields. The Settings object is NEVER dict-dumped,
so ``llm_api_key`` / ``massive_api_key`` and friends cannot leak here, and
no field named key/secret/token/password exists anywhere in the response
(enforced by test_config_api's recursive walk).
"""
import asyncio
import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.allocation import VolTargetParams
from libs.trading_core.backtest import BacktestParams
from libs.trading_core.contracts import SelectorParams, SpreadParams
from libs.trading_core.exits import ExitParams
from libs.trading_core.risk import RiskLimits
from libs.trading_core.signals import (
    DirectionalParams,
    EdgeClassificationParams,
    RegimeParams,
)
from libs.trading_core.tradeability import TradeabilityParams

from .. import audit, runtime_config
from ..db import (
    Order,
    Position,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
    utcnow,
)
from ..deps import (
    account_permissions_from_settings,
    broker_configured,
    broker_unavailable_reason,
    llm_configured,
    llm_unavailable_reason,
    market_data_configured,
    market_data_unavailable_reason,
    prediction_markets_configured,
    prediction_markets_unavailable_reason,
    resolve_broker,
    simulated_broker_mode,
    web_search_configured,
    web_search_unavailable_reason,
)

CURRENT_USER = "local-user"
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


def _risk_limits_payload() -> dict:
    """RiskLimits with JSON-friendly mappings (plan §12, §13).

    ``cash_floors`` is re-keyed by regime NAME (the enum value string) and
    ``correlation_buckets`` becomes {bucket name: [tickers]} — the same
    facts the engine runs on, in a shape any client can read.
    """
    limits = RiskLimits()
    payload = asdict(limits)
    payload["cash_floors"] = {
        regime.value: floor for regime, floor in limits.cash_floors.items()
    }
    payload["correlation_buckets"] = {
        name: list(members) for name, members in limits.correlation_buckets.items()
    }
    return payload


@router.get("")
async def get_config(session: AsyncSession = Depends(get_session)) -> dict:
    """The complete effective configuration, read-only (plan §44 rule 2).

    Every engine-parameter group is ``asdict`` of the REAL dataclass the
    engines default to — not a copy that could drift. ``kill_switch``
    reflects the persisted system_state row (plan §18) at request time.
    """
    settings = get_settings()
    state = await get_or_create_system_state(session)
    return {
        "environment": settings.environment,
        "providers": {
            # Provider NAMES are reported verbatim — "" when unset, never a
            # cosmetic default — alongside explicit booleans so a client can
            # tell "no data source" from "some data source" without parsing
            # names it does not know (§44 rule 18).
            "market_data": settings.market_data_provider,
            "market_data_configured": market_data_configured(),
            "llm": settings.llm_provider,
            "llm_configured": llm_configured(),
            "llm_model": settings.llm_model,
            # The execution venue, same shape and same honesty as the two
            # above: "" when unset (approve/close 503 and the exit sweep
            # skips), "simulated" for internal fills (development only), or a
            # real broker name. No credential material — the Alpaca key id and
            # secret are never read here.
            "broker": settings.broker_provider,
            "broker_configured": broker_configured(),
        },
        # The EFFECTIVE permissions — the same factory the §10 gate chain
        # sizes under (guide §8), so display and enforcement cannot drift.
        # All ten fields render, including the six display-and-refuse ones
        # (short_stock, naked_short_call, naked_short_put, covered_call,
        # cash_secured_put, margin): they are always false — the platform
        # has no code path for them, Settings hard-rejects their ALLOW_*
        # flags at startup (guide §33), and Alpaca Paper capability does not
        # override platform permissions (§2). Shown so the restriction is
        # visible, never so it can be lifted.
        "account_permissions": asdict(account_permissions_from_settings()),
        "risk_limits": _risk_limits_payload(),
        "exit_params": asdict(ExitParams()),
        "selector_params": asdict(SelectorParams()),
        # §9-S vertical spread parameters (execution-chains roadmap Phase 1;
        # research layer — spread EXECUTION is not built yet).
        "spread_params": asdict(SpreadParams()),
        "vol_target_params": asdict(VolTargetParams()),
        "regime_params": asdict(RegimeParams()),
        "directional_params": asdict(DirectionalParams()),
        # Edge band thresholds (upgrade §7) — research parameters, versioned;
        # the analysis endpoint's edge_legend derives from these same values.
        "edge_classification_params": asdict(EdgeClassificationParams()),
        # Layer-2 environment rules (upgrade §9) — research parameters,
        # versioned; direction-agnostic, never an execution authorization.
        "tradeability_params": asdict(TradeabilityParams()),
        "backtest_defaults": asdict(BacktestParams()),
        "paper_trading": {
            "initial_cash": settings.paper_initial_cash,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_share": settings.paper_commission_per_share,
            "commission_per_contract": settings.paper_commission_per_contract,
        },
        "kill_switch": {
            "trading_enabled": state.trading_enabled,
            "reason": state.reason,
        },
    }


# ---------------------------------------------------------------------------
# Provider connections (UI-managed, runtime_config layer).
# ---------------------------------------------------------------------------


class ProviderConfigRequest(BaseModel):
    """PUT /api/config/providers body — only present fields are changed.

    Empty string explicitly DISCONNECTS (unsets) a field. Secrets are write-
    only: they can be set here but no API ever returns them.
    """

    market_data_provider: str | None = None
    massive_api_key: str | None = None
    llm_provider: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_output_language: str | None = None
    # Instrument permissions ("true"/"false" — §5 real flags only, §33
    # forbidden capabilities are not settable anywhere).
    allow_long_stock: str | None = None
    allow_long_call: str | None = None
    allow_long_put: str | None = None
    allow_defined_risk_spreads: str | None = None
    allow_covered_call: str | None = None
    allow_cash_secured_put: str | None = None
    allow_short_stock: str | None = None
    allow_margin: str | None = None
    broker_provider: str | None = None
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    # External research (Catalyst research upgrade): RESEARCH ONLY providers.
    # No wallet / trading credentials exist for prediction markets — the
    # subsystem has no trading surface to configure.
    web_search_provider: str | None = None
    brave_api_key: str | None = None
    prediction_markets_provider: str | None = None


def _providers_status() -> dict:
    """Live per-provider state: name, configured (RESOLVES), reason if not.

    Secrets appear ONLY as presence booleans — the recursive no-secret walk
    in test_config_api covers this payload too.
    """
    settings = get_settings()
    return {
        "market_data": {
            "provider": settings.market_data_provider,
            "configured": market_data_configured(),
            "reason": market_data_unavailable_reason(),
        },
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            # Language NEW narrative generations are produced in; stored rows
            # keep the language they were generated in.
            "output_language": settings.llm_output_language or "en",
            "configured": llm_configured(),
            "reason": llm_unavailable_reason(),
        },
        "broker": {
            "provider": settings.broker_provider,
            "configured": broker_configured(),
            "reason": broker_unavailable_reason(),
        },
        # External research (Catalyst research upgrade): RESEARCH ONLY.
        # Unconfigured degrades only the research sections of event pages —
        # the reason strings say so, and no route 503s over either.
        "web_search": {
            "provider": settings.web_search_provider,
            "configured": web_search_configured(),
            "reason": web_search_unavailable_reason(),
        },
        "prediction_markets": {
            "provider": settings.prediction_markets_provider,
            "configured": prediction_markets_configured(),
            "reason": prediction_markets_unavailable_reason(),
        },
        "secrets_set": {
            "massive_api_key": bool(settings.massive_api_key),
            "llm_api_key": bool(settings.llm_api_key),
            "alpaca_api_key_id": bool(settings.alpaca_api_key_id),
            "alpaca_api_secret_key": bool(settings.alpaca_api_secret_key),
            "brave_api_key": bool(settings.brave_api_key),
        },
    }


@router.get("/providers")
async def get_providers() -> dict:
    """Current provider connections — names, live resolve-status, secret
    PRESENCE (never values). The Settings UI's Connections panel reads this.
    """
    return _providers_status()


@router.put("/providers")
async def put_providers(
    req: ProviderConfigRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Change provider connections from the UI (runtime_config layer).

    Only fields present in the body change; empty string disconnects. The
    stored values override .env immediately (environment + Settings cache
    are rebuilt) — every surface in the app sees the new providers on its
    next request, with no restart.

    NO CASH IS COPIED ANYWHERE: the platform stores no local copy of the
    broker account — cash is read live from the broker wherever displayed
    or sized from (user rule: the only account API is Alpaca).

    Audit: one CONFIG_CHANGED event listing the changed KEYS — never values.
    """
    changes = {
        key: value
        for key, value in req.model_dump().items()
        if value is not None
    }
    if not changes:
        return _providers_status()

    for field, allowed in runtime_config.ALLOWED_PROVIDERS.items():
        if field in changes and changes[field] not in allowed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{field} must be one of {sorted(allowed)!r}; "
                    f"got {changes[field]!r}"
                ),
            )

    changed_keys = await runtime_config.set_values(session, changes)
    if not changed_keys:
        return _providers_status()

    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.CONFIG_CHANGED,
        entity_type="runtime_config",
        entity_id="providers",
        details={
            "changed_keys": changed_keys,
            "note": "values are never audited; secrets never leave the server",
        },
    )

    await session.commit()
    return _providers_status()
