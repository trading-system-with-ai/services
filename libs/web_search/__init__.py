"""Web search provider registry (Catalyst research upgrade; plan §1-§2).

A SEPARATE registry from :mod:`libs.market_data` / :mod:`libs.event_calendar`
/ :mod:`libs.llm`, cloning their contract exactly — ``get_provider(name)``,
:class:`ProviderNotConfigured` on an empty name, ``ValueError`` on an unknown
one, **no default and no cross-provider fallback**:

============  =============================================================
``brave``     the Brave Search API (keyed: ``BRAVE_API_KEY``) — the real
              provider
``stub``      SYNTHETIC results — tests/development only, opt-in only
============  =============================================================

RESEARCH ONLY: web search feeds the event Evidence Bundle and nothing else.
No module here may import risk, strategies, signals, broker or order code —
enforced by tests/test_research_safety_adversarial.py.

Selection is single-provider via ``Settings.web_search_provider`` (the
market-data pattern, not the calendar's multi-provider set): one search
vendor answers per install, and an unconfigured install reports the
capability honestly absent rather than searching anywhere.
"""
from typing import Callable

from .provider import (  # noqa: F401 — the package's public surface
    CAPABILITY_KEYS,
    NO_CAPABILITIES,
    RESULT_TYPE_NEWS,
    RESULT_TYPE_WEB,
    RESULT_TYPES,
    WEB_SEARCH_NOT_CONFIGURED_MESSAGE,
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
    SearchResult,
    WebSearchError,
    WebSearchProvider,
    blank_capabilities,
)


def _make_brave() -> WebSearchProvider:
    # Imported lazily so importing libs.web_search never requires httpx or a
    # key (mirrors libs/market_data/__init__.py). Construction raises when
    # BRAVE_API_KEY is blank — the adapter never fires keyless.
    from libs.common.config import get_settings

    from .brave import BraveSearchProvider

    settings = get_settings()
    return BraveSearchProvider(api_key=settings.brave_api_key)


def _make_stub() -> WebSearchProvider:
    from .stub import StubWebSearchProvider

    return StubWebSearchProvider()


_PROVIDERS: dict[str, Callable[[], WebSearchProvider]] = {
    # The real provider: Brave's official Search API, keyed (BRAVE_API_KEY).
    "brave": _make_brave,
    # Opt-in only (development + tests): SYNTHETIC, non-real results.
    "stub": _make_stub,
}


def get_provider(name: str) -> WebSearchProvider:
    """Instantiate the provider registered under `name`.

    Raises :class:`ProviderNotConfigured` when `name` is empty or whitespace —
    the unconfigured state — and ``ValueError`` for an unknown non-empty name
    (an operator typo, not an absent configuration).
    """
    if not name or not name.strip():
        raise ProviderNotConfigured(WEB_SEARCH_NOT_CONFIGURED_MESSAGE)
    try:
        factory = _PROVIDERS[name.strip()]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(
            f"unknown web search provider {name!r}; known: {known}"
        ) from None
    return factory()
