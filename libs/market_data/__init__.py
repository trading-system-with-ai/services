"""Market data provider abstraction (development plan §22.1).

Providers are selected by name (Settings.market_data_provider). Per the
data-source architecture (prompts/data_source.md): ALPACA is the
authoritative market-data source (stocks, options, news); MASSIVE remains
registered for installs still keyed to it, and is otherwise reserved for
company fundamentals; the stub is an explicitly opt-in development/test
provider that produces synthetic, NON-MARKET numbers.

THERE IS NO DEFAULT PROVIDER (§44 rule 18). ``Settings.market_data_provider``
defaults to ``""`` and :func:`get_provider` raises
:class:`ProviderNotConfigured` for an empty/whitespace name, so an
unconfigured install serves NOTHING instead of silently serving invented
prices, bars or option chains. An unknown (non-empty) name is still a
``ValueError`` — that is an operator typo, not an absent configuration.

NO CROSS-PROVIDER FALLBACK (data_source.md §33): if the configured provider
fails, the platform degrades honestly — it never silently substitutes the
other vendor's numbers, which would corrupt price provenance.
"""
from typing import Callable

from .provider import (  # noqa: F401
    MARKET_DATA_NOT_CONFIGURED_MESSAGE,
    Bar,
    CapabilityNotAvailable,
    HistoricalOptionProvider,
    MarketDataError,
    MarketDataProvider,
    OptionContractRef,
    ProviderNotConfigured,
    Quote,
)
from .stub import StubProvider  # noqa: F401


def _make_massive() -> MarketDataProvider:
    # Imported lazily so importing libs.market_data never requires httpx or an
    # API key (mirrors libs/llm/__init__.py). Construction raises
    # MarketDataError when settings.massive_api_key is empty, guaranteeing the
    # Massive provider is NEVER used keyless — and there is no synthetic
    # fallback to degrade to (§44 rule 18).
    from libs.common.config import get_settings

    from .massive import MassiveProvider

    settings = get_settings()
    return MassiveProvider(api_key=settings.massive_api_key)


def _make_alpaca() -> MarketDataProvider:
    # Same lazy-import discipline. Alpaca market data authenticates with the
    # SAME account credentials the broker uses (verified live against
    # data.alpaca.markets); construction refuses blank keys.
    from libs.common.config import get_settings

    from .alpaca import AlpacaMarketDataProvider

    settings = get_settings()
    return AlpacaMarketDataProvider(
        api_key_id=settings.alpaca_api_key_id,
        api_secret_key=settings.alpaca_api_secret_key,
    )


_PROVIDERS: dict[str, Callable[[], MarketDataProvider]] = {
    # Opt-in only (development + tests): synthetic, non-market data.
    "stub": StubProvider,
    # The AUTHORITATIVE market-data source (data_source.md §1).
    "alpaca": _make_alpaca,
    # Legacy/alternative real source; per data_source.md Massive is reserved
    # for fundamentals — kept registered so existing installs keep working.
    "massive": _make_massive,
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
