"""Web search provider interface (Catalyst research upgrade; plan
SEARCH_PREDICTION_MARKET_UPGRADE_PLAN.md §1-§2).

A SEPARATE provider registry from :mod:`libs.market_data` and
:mod:`libs.event_calendar`, cloning their contract exactly —
``get_provider(name)``, :class:`ProviderNotConfigured` on an empty name,
``ValueError`` on an unknown one, no default and no cross-provider fallback —
because web search answers yet another question (public-web evidence discovery
for one catalyst) and fails in yet other ways (API quota, per-query billing,
result-quality drift).

RESEARCH ONLY. Nothing in this package may reach instrument selection, risk
assessment, the gate chain or order submission — enforced structurally by
tests/test_research_safety_adversarial.py (import-graph AST), not just by
convention.

EVERY RESULT IS UNTRUSTED TEXT. A search snippet is a third party's words
fetched from the public web; a page can carry prompt-injection-shaped text on
purpose. Providers return the payload VERBATIM (provenance requires it) and
NOTHING model-facing may read it raw: the pure research layer
(libs/trading_core/events/web_research.py) runs every title/snippet through
``sanitize_for_llm`` and the evidence bundle exposes only the ``safe_*``
forms, exactly as the news pipeline does (§81).

NO FABRICATION (§44 rule 18): a provider that cannot answer returns FEWER
results (or raises :class:`WebSearchError`), never padded ones, and
``published_at`` is ``None`` whenever the provider did not state a
publication time — a missing timestamp is a different fact from any invented
one, and the as-of gate treats it conservatively (excluded from
point-in-time-sensitive views, with a reason).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from libs.market_data.provider import (  # noqa: F401 — re-exported on purpose:
    # ONE failure taxonomy platform-wide (audit §6). `except MarketDataError`
    # written for market data keeps working unchanged for web search.
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
)

#: The message every unconfigured-web-search path reports verbatim, so the API
#: error, the logs and the tests all name the SAME missing configuration.
#: Web search is a RESEARCH capability: unset means event pages still work and
#: only the web-research section reports itself honestly unavailable.
WEB_SEARCH_NOT_CONFIGURED_MESSAGE = (
    "web search provider is not configured — set WEB_SEARCH_PROVIDER (and its "
    "credentials, e.g. BRAVE_API_KEY); event research degrades, nothing else"
)

#: The FIXED capability key set every web-search provider reports, in the same
#: tri-state shape as the market-data/calendar registries (audit §6): ``True``
#: (probed and works) / ``False`` (proven absence — the plan does not include
#: it) / an error string (fault, availability unknown).
CAPABILITY_KEYS: tuple[str, ...] = (
    "web_search",   # general web index search
    "news_search",  # news-vertical search with publication metadata
)

#: Every capability False — the base a provider overrides for what it serves.
NO_CAPABILITIES: dict[str, bool | str] = {k: False for k in CAPABILITY_KEYS}

#: ``SearchResult.result_type`` vocabulary. Lives in code, not a DB CHECK
#: (migration-017 lesson: a CHECK is a second copy that drifts).
RESULT_TYPE_WEB = "web"
RESULT_TYPE_NEWS = "news"
RESULT_TYPES: tuple[str, ...] = (RESULT_TYPE_WEB, RESULT_TYPE_NEWS)


class WebSearchError(MarketDataError):
    """A web-search request could not be answered honestly.

    A subclass of :class:`libs.market_data.provider.MarketDataError` rather
    than a parallel hierarchy: the gateway seam catches ONE exception type for
    "a data source could not answer" (the discipline
    :class:`libs.event_calendar.CalendarProviderError` set).
    """


def blank_capabilities() -> dict[str, bool | str]:
    """A fresh all-False capability report (never share the module dict)."""
    return dict(NO_CAPABILITIES)


@dataclass(frozen=True)
class SearchResult:
    """One search hit, provider-verbatim (plan §2 normalized result shape).

    This is the ACQUISITION shape, not the evidence shape: URL normalization,
    deduplication, the as-of gate, source-tier classification and relevance
    all happen downstream in the pure research layer
    (libs/trading_core/events/web_research.py), which turns accepted results
    into evidence objects. Keeping the provider dumb keeps every judgement
    call in deterministic, tested code.

    ``published_at`` is ``None`` when the provider stated no publication time
    (§44 rule 18: never a fake timestamp). ``retrieved_at`` is when THIS
    platform fetched the result — always known, never None — and is the
    provenance clock, not the evidence clock.

    ``rank`` is the provider's own ordering (0-based). Search ranking is NOT
    evidence reliability (plan §2 source-quality rule); it is carried only as
    retrieval provenance.

    ``title``/``snippet`` are UNTRUSTED third-party text, stored verbatim for
    provenance. Nothing model-facing reads them raw — see the module
    docstring.
    """

    provider: str
    provider_result_id: str | None
    query: str
    title: str
    url: str
    snippet: str
    publisher: str | None
    published_at: datetime | None
    retrieved_at: datetime
    result_type: str
    rank: int


class WebSearchProvider(Protocol):
    """Structural interface every web-search provider must satisfy.

    Deliberately narrow (the market-data Protocol's lesson): the two verticals
    every consumer needs, plus the capability probe. Providers may expose more
    concretely; consumers type against this Protocol only.

    ``domains``/``exclude_domains`` are HINTS a provider may fold into its own
    query syntax (e.g. ``site:``); they are NOT the enforcement point. The
    pure research layer re-applies domain policy deterministically over the
    returned results, so a provider that ignores the hint can never widen
    what the platform admits as evidence.
    """

    name: str

    def capabilities(self) -> dict[str, bool | str]:
        """Tri-state report over :data:`CAPABILITY_KEYS` (audit §6).

        Never raises: a capability probe that raised would take the whole
        research refresh down with it.
        """
        ...

    def search_web(
        self,
        query: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
        country: str | None = None,
        language: str | None = None,
        domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """General web-index results for ``query``, best-effort in
        ``[start_time, end_time]``.

        The time bounds are HINTS to the provider's freshness filter; the
        research layer re-applies ``published_at <= as_of`` deterministically
        (a provider's freshness window is not this platform's look-ahead
        gate). Fewer results than ``limit`` is a normal, honest answer.
        """
        ...

    def search_news(
        self,
        query: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
        country: str | None = None,
        language: str | None = None,
        domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """News-vertical results for ``query`` — same contract as
        :meth:`search_web`, with publication metadata when the vertical
        carries it. Providers without a news vertical raise
        :class:`CapabilityNotAvailable`, never silently substitute web
        results labelled as news.
        """
        ...
