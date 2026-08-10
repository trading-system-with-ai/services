"""Market data provider abstraction (development plan §22.1).

Providers are selected by name (Settings.market_data_provider), so the stub
can be swapped for the real MASSIVE-backed provider by configuration alone.
"""
from typing import Callable

from .provider import MarketDataProvider, Quote  # noqa: F401
from .stub import StubProvider  # noqa: F401

_PROVIDERS: dict[str, Callable[[], MarketDataProvider]] = {
    "stub": StubProvider,
    # "massive": MassiveProvider,  # arrives with the MASSIVE integration
}


def get_provider(name: str) -> MarketDataProvider:
    """Instantiate the provider registered under `name` (e.g. "stub")."""
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(f"unknown market data provider {name!r}; known: {known}") from None
    return factory()
