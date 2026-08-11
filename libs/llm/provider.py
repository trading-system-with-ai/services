"""LLM recommendation provider interface (development plan §4.1).

Consumers depend only on this Protocol. Concrete providers (the deterministic
stub today, the Anthropic-backed provider when an API key is configured) are
selected by configuration via :func:`libs.llm.get_recommendation_provider`,
never imported directly by callers — this keeps the provider swappable without
touching consumer code.

CENTRAL SAFETY RULE (plan §4.1, §44 rule 5, §46): the LLM proposes, the user
curates. A RecommendationDraft is an *information* artifact only — it carries
zero execution authority. Nothing in this package (or anything consuming it)
may mutate the Watchlist, the Trading Pool, or orders; only an explicit USER
API action can promote a recommendation into anything actionable.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


class ProviderError(Exception):
    """Raised when a recommendation provider fails (network, HTTP, config)."""


# The message every unconfigured-LLM path reports verbatim, so the API error,
# the logs and the tests all name the SAME missing configuration.
LLM_NOT_CONFIGURED_MESSAGE = (
    "LLM provider is not configured — set LLM_PROVIDER and the corresponding "
    "credentials"
)


class LLMProviderNotConfigured(ProviderError):
    """No LLM provider is configured (``LLM_PROVIDER`` unset).

    Mirrors :class:`libs.market_data.ProviderNotConfigured`: an unknown
    provider name stays a ``ValueError`` (an operator typo), while this is the
    absence of any configuration at all. Callers map it to HTTP 503
    ``LLM_NOT_CONFIGURED`` — an unconfigured install must never be served
    template-generated recommendations that read like real analysis.
    """

    def __init__(self, message: str = LLM_NOT_CONFIGURED_MESSAGE) -> None:
        super().__init__(message)


@dataclass
class RecommendationDraft:
    """One LLM-proposed candidate, matching the plan §4.1 recommendation schema.

    Score semantics (all validated in __post_init__, plan §4.1):
      - sentiment: directional read of the catalyst, -1.0 (bearish) .. +1.0 (bullish)
      - impact: expected materiality of the catalyst, 0.0 .. 1.0
      - novelty: how new the information is, 0.0 .. 1.0
      - source_reliability: trustworthiness of the cited sources, 0.0 .. 1.0

    ``evidence`` is a list of citation dicts, each with keys "source",
    "published_at" (ISO-8601), and "snippet". Plan §20.3 (news timestamp
    integrity) requires every evidence item to be published strictly before
    the as-of time the draft was generated for.
    """

    ticker: str
    company: str | None
    sentiment: float
    impact: float
    novelty: float
    source_reliability: float
    horizon: str
    catalyst_type: str
    reason_codes: list[str] = field(default_factory=list)
    summary: str = ""
    evidence: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string")
        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError(f"sentiment must be in [-1, 1], got {self.sentiment}")
        for name in ("impact", "novelty", "source_reliability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


class RecommendationProvider(Protocol):
    """Structural interface every recommendation provider must satisfy."""

    def generate(
        self,
        exclude_tickers: set[str],
        as_of: datetime,
        limit: int = 5,
    ) -> list[RecommendationDraft]:
        """Return up to `limit` candidate drafts, never for excluded tickers.

        `as_of` is the information cut-off: evidence must be published strictly
        before it (plan §20.3).
        """
        ...
