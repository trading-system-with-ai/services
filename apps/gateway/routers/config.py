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
from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.trading_core.allocation import VolTargetParams
from libs.trading_core.backtest import BacktestParams
from libs.trading_core.contracts import SelectorParams
from libs.trading_core.exits import ExitParams
from libs.trading_core.risk import RiskLimits
from libs.trading_core.signals import DirectionalParams, RegimeParams
from libs.trading_core.strategies import AccountPermissions

from ..db import get_or_create_system_state, get_session

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
            "market_data": settings.market_data_provider,
            "llm": settings.llm_provider,
            "llm_model": settings.llm_model,
        },
        "account_permissions": asdict(AccountPermissions()),
        "risk_limits": _risk_limits_payload(),
        "exit_params": asdict(ExitParams()),
        "selector_params": asdict(SelectorParams()),
        "vol_target_params": asdict(VolTargetParams()),
        "regime_params": asdict(RegimeParams()),
        "directional_params": asdict(DirectionalParams()),
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
