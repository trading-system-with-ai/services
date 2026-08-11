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
- :func:`require_broker`               -> 503 ``BROKER_NOT_CONFIGURED``

THE SAME RULE, APPLIED TO EXECUTION. ``BROKER_PROVIDER`` unset means the
platform places NOTHING: approve and close answer 503 and the mechanical exit
sweep skips. It must never silently fall back to the internal fill simulator —
a simulated fill reported as a broker fill is invented data in the one place
that claims to have touched a real market. The simulator is still reachable,
but ONLY by explicitly setting ``BROKER_PROVIDER=simulated`` (see
:data:`SIMULATED_BROKER`), exactly like the opt-in stub providers elsewhere.

Deliberately NOT applied to endpoints whose content is real DB state:
``GET /api/positions`` (positions are real rows the user actually holds) and
``GET /api/portfolio/risk`` (NAV / cash / positions come from the database)
stay 200 and report their market-derived fields as honest NULLS instead — see
:func:`market_data_status`. Hiding a real position because a quote is missing
would be its own kind of dishonesty.
"""
from fastapi import HTTPException

from libs.broker import BrokerNotConfigured, BrokerProvider, get_broker
from libs.common.config import get_settings
from libs.llm import LLMProviderNotConfigured, get_recommendation_provider
from libs.market_data import ProviderNotConfigured, get_provider

# Machine-readable error codes carried in the 503 detail block.
MARKET_DATA_NOT_CONFIGURED = "MARKET_DATA_NOT_CONFIGURED"
LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"
BROKER_NOT_CONFIGURED = "BROKER_NOT_CONFIGURED"

# The ONE gateway-level execution mode that is not a broker: the internal §11
# paper fill model. It lives HERE and not in the libs.broker registry on
# purpose — see that package's docstring. Setting BROKER_PROVIDER=simulated is
# an explicit, documented development / backtest-comparison opt-in; nothing
# defaults to it.
SIMULATED_BROKER = "simulated"


def market_data_unavailable_reason() -> str | None:
    """Why market data is unusable right now, or None when it works.

    Checks that the configured name actually RESOLVES to a provider, not just
    that it is non-blank. A name for a provider that does not exist yet (e.g.
    ``massive`` before its adapter ships) is an unusable configuration, and the
    honest answer is the same 503 the unset case gets — not a 500 from a
    ValueError escaping a route, and not a claim that market data is available.
    """
    name = get_settings().market_data_provider
    if not name or not name.strip():
        return str(ProviderNotConfigured())
    try:
        get_provider(name)
    except ProviderNotConfigured as exc:
        return str(exc)
    except ValueError as exc:
        # Unknown/unimplemented name: report the operator's own setting back.
        return f"{exc} (MARKET_DATA_PROVIDER={name!r})"
    except Exception as exc:  # construction failed (bad credentials, etc.)
        return f"market data provider {name!r} could not be initialised: {exc}"
    return None


def market_data_configured() -> bool:
    """True when a market data provider is configured AND usable."""
    return market_data_unavailable_reason() is None


def llm_unavailable_reason() -> str | None:
    """Why LLM recommendations are unusable right now, or None when they work."""
    name = get_settings().llm_provider
    if not name or not name.strip():
        return str(LLMProviderNotConfigured())
    try:
        get_recommendation_provider(name)
    except LLMProviderNotConfigured as exc:
        return str(exc)
    except ValueError as exc:
        return f"{exc} (LLM_PROVIDER={name!r})"
    except Exception as exc:
        return f"LLM provider {name!r} could not be initialised: {exc}"
    return None


def llm_configured() -> bool:
    """True when an LLM provider is configured AND usable."""
    return llm_unavailable_reason() is None


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
    reason = market_data_unavailable_reason()
    if reason is not None:
        raise HTTPException(
            status_code=503,
            detail={"code": MARKET_DATA_NOT_CONFIGURED, "message": reason},
        )


def require_llm_provider() -> None:
    """Raise 503 ``LLM_NOT_CONFIGURED`` when no usable LLM provider exists."""
    reason = llm_unavailable_reason()
    if reason is not None:
        raise HTTPException(
            status_code=503,
            detail={"code": LLM_NOT_CONFIGURED, "message": reason},
        )


# ---------------------------------------------------------------------------
# Broker (plan §11 — real execution). Same three-part shape as the two guards
# above: a "why is it unusable" reason, a boolean, and a raiser.
# ---------------------------------------------------------------------------


def broker_mode() -> str:
    """The configured execution mode, normalised: "", "simulated" or a broker.

    Whitespace-only is the unconfigured state (""), matching the registry.
    """
    return (get_settings().broker_provider or "").strip()


def simulated_broker_mode() -> bool:
    """True only when execution is EXPLICITLY set to the internal simulator.

    Never true by default. This is the single place the gateway decides that
    an internal paper fill is acceptable, and it takes an operator writing
    ``BROKER_PROVIDER=simulated`` to make it so.
    """
    return broker_mode() == SIMULATED_BROKER


def broker_unavailable_reason() -> str | None:
    """Why real execution is unusable right now, or None when it works.

    Like :func:`market_data_unavailable_reason`, this checks that the
    configured name actually RESOLVES: a typo, or credentials the adapter
    refuses to run keyless with, is an unusable configuration and gets the same
    honest 503 as the unset case rather than a 500 from a route.

    ``simulated`` deliberately resolves to None (usable): it is not a broker,
    but it IS a configured, explicitly-opted-into execution mode, and the
    approve/close guards must let it through to the internal fill path.
    """
    name = broker_mode()
    if not name:
        return str(BrokerNotConfigured())
    if name == SIMULATED_BROKER:
        return None
    try:
        get_broker(name)
    except BrokerNotConfigured as exc:
        return str(exc)
    except ValueError as exc:
        return f"{exc} (BROKER_PROVIDER={name!r})"
    except Exception as exc:  # construction failed (missing credentials, ...)
        return f"broker {name!r} could not be initialised: {exc}"
    return None


def broker_configured() -> bool:
    """True when an execution mode is configured AND usable (incl. simulated)."""
    return broker_unavailable_reason() is None


def require_broker() -> None:
    """Raise 503 ``BROKER_NOT_CONFIGURED`` when no usable execution mode exists.

    Call this FIRST in any route that would place, cancel or fill an order.
    There is NO fallback to the internal simulator: an install that never
    configured a broker must place nothing at all (§44 rule 18).
    """
    reason = broker_unavailable_reason()
    if reason is not None:
        raise HTTPException(
            status_code=503,
            detail={"code": BROKER_NOT_CONFIGURED, "message": reason},
        )


def resolve_broker() -> BrokerProvider:
    """The configured broker adapter, or 503 ``BROKER_NOT_CONFIGURED``.

    Only valid for REAL broker modes: calling this in ``simulated`` mode is a
    programming error (there is no broker to return — that is the whole point
    of keeping the simulator out of the registry), so it raises ``RuntimeError``
    rather than inventing one.
    """
    require_broker()
    if simulated_broker_mode():
        raise RuntimeError(
            "resolve_broker() called in simulated mode: the internal fill "
            "simulator is deliberately NOT a broker (see libs/broker)"
        )
    return get_broker(broker_mode())


def broker_status_block() -> dict:
    """The ``broker`` block for responses that degrade instead of 503-ing.

    ``{"configured": bool, "provider": str, "message": str | None}`` — the
    provider NAME is reported verbatim ("" when unset, never a cosmetic
    default) and the message names the missing configuration.
    """
    reason = broker_unavailable_reason()
    return {
        "configured": reason is None,
        "provider": broker_mode(),
        "message": reason,
    }


def market_data_status() -> dict:
    """The ``market_data`` block for responses that degrade instead of 503-ing.

    ``{"configured": bool, "message": str | None}`` — the message names the
    missing configuration when unset, and is null when a provider is
    configured. Lets a client render "no market data" honestly next to the real
    DB-derived numbers it IS showing (NAV, cash, position quantities).
    """
    reason = market_data_unavailable_reason()
    if reason is None:
        return {"configured": True, "message": None}
    return {"configured": False, "message": reason}
