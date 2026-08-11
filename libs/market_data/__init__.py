"""Market data provider abstraction (development plan §22.1).

Providers are selected by name (Settings.market_data_provider). MASSIVE is the
only supported source of REAL market data; the stub is an explicitly opt-in
development/test provider that produces synthetic, NON-MARKET numbers.

THERE IS NO DEFAULT PROVIDER (§44 rule 18). ``Settings.market_data_provider``
defaults to ``""`` and :func:`get_provider` raises
:class:`ProviderNotConfigured` for an empty/whitespace name, so an
unconfigured install serves NOTHING instead of silently serving invented
prices, bars or option chains. An unknown (non-empty) name is still a
``ValueError`` — that is an operator typo, not an absent configuration.
"""
from typing import Callable

from .provider import (  # noqa: F401
    MARKET_DATA_NOT_CONFIGURED_MESSAGE,
    MarketDataProvider,
    ProviderNotConfigured,
    Quote,
)
from .stub import StubProvider  # noqa: F401

_PROVIDERS: dict[str, Callable[[], MarketDataProvider]] = {
    # Opt-in only (development + tests): synthetic, non-market data.
    "stub": StubProvider,
    # "massive": MassiveProvider,  # arrives with the MASSIVE integration
}


def get_provider(name: str) -> MarketDataProvider:
    """Instantiate the provider registered under `name`.

    Raises :class:`ProviderNotConfigured` when `name` is empty or whitespace —
    the unconfigured state — and ``ValueError`` for an unknown non-empty name.
    """
    if not name or not name.strip():
        raise ProviderNotConfigured()
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(f"unknown market data provider {name!r}; known: {known}") from None
    return factory()
