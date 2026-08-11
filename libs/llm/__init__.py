"""LLM recommendation provider abstraction (development plan §4.1).

Providers are selected by name (Settings.llm_provider), mirroring the
libs.market_data pattern, so the deterministic stub can be swapped for the
real Anthropic-backed provider by configuration alone.

THERE IS NO DEFAULT PROVIDER (§44 rule 18). ``Settings.llm_provider`` defaults
to ``""`` and :func:`get_recommendation_provider` raises
:class:`LLMProviderNotConfigured` for an empty/whitespace name, so an
unconfigured install produces NO recommendations rather than
template-generated text that reads like real analysis. The stub stays in the
registry as an explicitly opt-in development/test provider only. An unknown
(non-empty) name is still a ``ValueError`` — an operator typo, not an absent
configuration.

SAFETY (plan §4.1, §44 rule 5, §46): everything served through this package
is an information feature. Recommendations carry zero execution authority.
"""
from typing import Callable

from .provider import (  # noqa: F401
    LLM_NOT_CONFIGURED_MESSAGE,
    LLMProviderNotConfigured,
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


def _make_openai() -> RecommendationProvider:
    # Imported lazily for the same reason as the Anthropic factory above.
    # Construction raises ProviderError when settings.llm_api_key is empty, so
    # the OpenAI provider is NEVER used keyless.
    from libs.common.config import get_settings

    from .openai import OpenAIRecommendationProvider

    settings = get_settings()
    return OpenAIRecommendationProvider(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


_PROVIDERS: dict[str, Callable[[], RecommendationProvider]] = {
    # Opt-in only (development + tests): template-generated, NON-LLM drafts.
    "stub": _make_stub,
    "anthropic": _make_anthropic,
    "openai": _make_openai,
}


def get_recommendation_provider(name: str) -> RecommendationProvider:
    """Instantiate the provider registered under `name` ("stub" | "anthropic" | "openai").

    Raises :class:`LLMProviderNotConfigured` when `name` is empty or
    whitespace — the unconfigured state — and ``ValueError`` for an unknown
    non-empty name.
    """
    if not name or not name.strip():
        raise LLMProviderNotConfigured()
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(f"unknown LLM provider {name!r}; known: {known}") from None
    return factory()
