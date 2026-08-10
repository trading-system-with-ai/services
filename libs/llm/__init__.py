"""LLM recommendation provider abstraction (development plan §4.1).

Providers are selected by name (Settings.llm_provider), mirroring the
libs.market_data pattern, so the deterministic stub can be swapped for the
real Anthropic-backed provider by configuration alone. The default provider
is "stub" (works keyless — see libs/common/config.py).

SAFETY (plan §4.1, §44 rule 5, §46): everything served through this package
is an information feature. Recommendations carry zero execution authority.
"""
from typing import Callable

from .provider import (  # noqa: F401
    ProviderError,
    RecommendationDraft,
    RecommendationProvider,
)
from .stub import StubRecommendationProvider  # noqa: F401


def _make_stub() -> RecommendationProvider:
    return StubRecommendationProvider()


def _make_anthropic() -> RecommendationProvider:
    # Imported lazily so that importing libs.llm never requires httpx or an
    # API key. Construction raises ProviderError when settings.llm_api_key is
    # empty, guaranteeing the Anthropic provider is NEVER used keyless.
    from libs.common.config import get_settings

    from .anthropic import AnthropicRecommendationProvider

    settings = get_settings()
    return AnthropicRecommendationProvider(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


_PROVIDERS: dict[str, Callable[[], RecommendationProvider]] = {
    "stub": _make_stub,
    "anthropic": _make_anthropic,
}


def get_recommendation_provider(name: str) -> RecommendationProvider:
    """Instantiate the provider registered under `name` ("stub" | "anthropic")."""
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(f"unknown LLM provider {name!r}; known: {known}") from None
    return factory()
