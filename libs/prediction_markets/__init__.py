"""Prediction-market provider registry (Catalyst research upgrade; plan §3).

A SEPARATE registry cloning the market-data/calendar/LLM contract exactly —
``get_provider(name)``, :class:`ProviderNotConfigured` on an empty name,
``ValueError`` on an unknown one, **no default and no cross-provider
fallback**:

==============  ===========================================================
``polymarket``  Polymarket's PUBLIC read APIs (Gamma discovery/metadata +
                CLOB pricing) — keyless, but still opt-in
``stub``        SYNTHETIC markets — tests/development only, opt-in only
==============  ===========================================================

READ ONLY, RESEARCH ONLY. No trading surface exists in this package (no
wallet, no signing, no order placement — plan §3) and nothing here may
import risk, strategies, signals, broker or order code — both enforced by
tests/test_research_safety_adversarial.py.

Keyless does NOT mean default-on: unlike the calendar registry's
``KEYLESS_PROVIDERS`` (free primary sources the platform always consults),
prediction markets stay unset until the operator explicitly selects a
provider — an outbound market-data dependency is a conscious choice, and an
unconfigured install reports the capability honestly absent (plan §8).
"""
from typing import Callable

from .provider import (  # noqa: F401 — the package's public surface
    CAPABILITY_KEYS,
    MARKET_STATUS_ACTIVE,
    MARKET_STATUS_CLOSED,
    MARKET_STATUS_RESOLVED,
    MARKET_STATUS_UNKNOWN,
    MARKET_STATUSES,
    NO_CAPABILITIES,
    PREDICTION_MARKETS_NOT_CONFIGURED_MESSAGE,
    CapabilityNotAvailable,
    MarketDataError,
    MarketOutcome,
    MarketSnapshot,
    PredictionMarketError,
    PredictionMarketInfo,
    PredictionMarketProvider,
    PricePoint,
    ProviderNotConfigured,
    blank_capabilities,
)


def _make_polymarket() -> PredictionMarketProvider:
    # Imported lazily so importing libs.prediction_markets never requires
    # httpx (mirrors libs/market_data/__init__.py). Keyless, but never
    # anonymous: the shared operator contact string (sec_user_agent) is the
    # User-Agent, exactly as the government calendar sources reuse it.
    from libs.common.config import get_settings

    from .polymarket import PolymarketProvider

    settings = get_settings()
    user_agent = (getattr(settings, "sec_user_agent", "") or "").strip() or (
        "trading-system-with-ai/0.1 (catalyst research; set SEC_USER_AGENT)"
    )
    return PolymarketProvider(user_agent=user_agent)


def _make_stub() -> PredictionMarketProvider:
    from .stub import StubPredictionMarketProvider

    return StubPredictionMarketProvider()


_PROVIDERS: dict[str, Callable[[], PredictionMarketProvider]] = {
    # The real provider: Polymarket's public Gamma + CLOB read APIs.
    "polymarket": _make_polymarket,
    # Opt-in only (development + tests): SYNTHETIC, non-real markets.
    "stub": _make_stub,
}


def get_provider(name: str) -> PredictionMarketProvider:
    """Instantiate the provider registered under `name`.

    Raises :class:`ProviderNotConfigured` when `name` is empty or whitespace —
    the unconfigured state — and ``ValueError`` for an unknown non-empty name
    (an operator typo, not an absent configuration).
    """
    if not name or not name.strip():
        raise ProviderNotConfigured(PREDICTION_MARKETS_NOT_CONFIGURED_MESSAGE)
    try:
        factory = _PROVIDERS[name.strip()]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(
            f"unknown prediction markets provider {name!r}; known: {known}"
        ) from None
    return factory()
