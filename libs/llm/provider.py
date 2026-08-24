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
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .event_analysis import EventAnalysisResult


class ProviderError(Exception):
    """Raised when a recommendation provider fails (network, HTTP, config)."""


# The message every unconfigured-LLM path reports verbatim, so the API error,
# the logs and the tests all name the SAME missing configuration.
LLM_NOT_CONFIGURED_MESSAGE = (
    "LLM provider is not configured — set LLM_PROVIDER and the corresponding "
    "credentials"
)

#: Output-language addendum appended to every provider system prompt
#: (Settings.llm_output_language). NARRATIVE fields only: machine-read fields
#: (horizon, catalyst_type, reason_codes, tickers, urls, timestamps) stay
#: English regardless — downstream filtering/analytics key on them, and a
#: mixed-language enum column would be silent data corruption. "en"/unknown →
#: no addendum (English is the prompts' native language).
OUTPUT_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "zh": (
        "\nOutput language: write the NARRATIVE fields — summary and every "
        "evidence snippet — in Simplified Chinese (简体中文), in the register "
        "of a professional sell-side research note. Keep machine-read fields "
        "in English exactly as specified: ticker, company (official English "
        "name), horizon, catalyst_type, reason_codes, every url and "
        "timestamp."
    ),
}


def language_instruction(output_language: str) -> str:
    """The prompt addendum for `output_language`; "" when none applies."""
    return OUTPUT_LANGUAGE_INSTRUCTIONS.get(output_language, "")


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


@dataclass(frozen=True)
class GroundingArticle:
    """One REAL stored news article handed to the LLM as grounding material.

    The ONLY information a provider's ``enrich`` may use. ``url`` doubles as
    the citation key: every evidence item a draft returns must carry one of
    these urls in its "source" field, or the router drops the draft.
    """

    url: str
    title: str
    publisher: str
    published_at: str  # ISO-8601
    tickers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
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

    def enrich(
        self,
        articles: list[GroundingArticle],
        exclude_tickers: set[str],
        as_of: datetime,
        limit: int = 5,
    ) -> list[RecommendationDraft]:
        """Drafts grounded EXCLUSIVELY in `articles` (Phase 8 enrichment).

        Every evidence item must cite one of the given articles by its url in
        "source"; a draft's ticker must appear in a cited article's ticker
        list. The router re-validates both and DROPS violations — a citation
        that is not in the stored news table never reaches the user.
        """
        ...

    def analyze_event(
        self,
        bundle_json: dict,
        *,
        as_of: datetime,
    ) -> "EventAnalysisResult":
        """Pre-event research note for ONE catalyst (event spec §46-§52).

        `bundle_json` is the deterministic Evidence Bundle
        (``libs.trading_core.events.evidence.bundle_to_json``): the sole
        information source. The provider must not fetch anything, and the
        model must not compute numbers (§47) — every number in the returned
        narrative has to be copied from the bundle and listed in
        ``numbers_quoted`` with its dotted fact path.

        Returns an :class:`~libs.llm.event_analysis.EventAnalysisResult`. The
        implementation returns it UNVALIDATED (``violations`` empty) except
        where it can cheaply tell; the caller runs
        ``event_analysis.validate_analysis`` against the bundle's fact index
        and persists the violations, so the enforcement lives in one place
        for every provider.

        Raises :class:`ProviderError` on transport/HTTP failure and on a model
        refusal — unlike ``generate``/``enrich`` there is no "fewer results"
        degradation available for a single-event note; the caller records the
        failure and still serves the bundle.
        """
        ...
