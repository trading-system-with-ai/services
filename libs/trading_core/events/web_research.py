"""Pure web-research layer: window, taxonomy, planning, normalization
(Catalyst research upgrade; plan §1-§2, Phases 1/2/5; LOOP 3).

PURE COMPUTATION — NO I/O. This module never imports the provider packages,
the gateway, or an HTTP client (enforced by the AST test in
tests/test_events_web_research.py, the news_intel/macro discipline): the
gateway seam fetches through libs.web_search and hands the results here,
exactly as event_news hands articles to news_intel.

WHAT IS DETERMINISTIC HERE IS THE WHOLE POINT (mission: the LLM must not
control date boundaries, network destinations, rate limits, source-quality
rules or evidence persistence). Every judgement this module makes — the
research window, which queries to run, which results to admit, what tier a
source is — is rule-based, versioned and testable. An LLM-assisted query
planner may LATER propose better query wording under a strict schema, but
this module's plan is the baseline AND the enforcement: counts, dates and
domains are clamped here regardless of who proposed the words.

POINT-IN-TIME HONESTY (§44 rule 18, plan Phase 10):

- the window's end is ALWAYS the request as_of; nothing searchable beyond it;
- BOTH window bounds are enforced HERE (the provider's freshness filter is
  a hint): ``published_at > as_of`` and dated pre-window results are each
  rejected with their named reason;
- a result with NO publication time is admissible only because its
  ``retrieved_at`` proves it existed when fetched — when even that cannot be
  placed at/before as_of (a historical replay reading a later run's rows),
  it is rejected as UNPLACEABLE_IN_TIME rather than assumed innocent;
- rejected candidates are kept WITH their reasons: "what the platform
  refused to admit and why" is the Evidence tab's transparency promise.

UNTRUSTED TEXT: titles/snippets pass through ``sanitize_for_llm`` (the §81
news discipline); injection-shaped text is flagged, EXCLUDED from anything
model-facing, counted in ``suppressed_suspicious``, and never alters this
module's behavior — the plan is built before any result text is read.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from libs.trading_core.events.news_intel import (
    DEFAULT_DEDUPE_JACCARD,
    jaccard,
    sanitize_for_llm,
    story_shingles,
    tokens,
)
from libs.trading_core.events.taxonomy import require_utc
from libs.trading_core.models.enums import EventType

#: Version stamp for this layer's rules (window fallbacks, tiers, scoring).
#: Bump when a rule change would alter which evidence gets admitted.
WEB_RESEARCH_MODEL_VERSION = "web-research-v1"

# ---------------------------------------------------------------------------
# Cost control (plan Phase 2 / 12) — the bounds are the contract, not advice.
# One research refresh can never issue more than MAX_QUERIES_PER_EVENT paid
# searches, consider more than MAX_UNIQUE_DOCUMENTS distinct documents, or
# admit more than MAX_ACCEPTED_EVIDENCE into the bundle.
# ---------------------------------------------------------------------------
MAX_QUERIES_PER_EVENT = 6
MAX_RESULTS_PER_QUERY = 10
MAX_UNIQUE_DOCUMENTS = 40
MAX_ACCEPTED_EVIDENCE = 20

#: Below this deterministic relevance, a result is off-topic noise and is
#: rejected rather than padded into the bundle (a bounded, RANKED set — the
#: model never receives filler).
MIN_RELEVANCE_ACCEPT = 0.15

# ---------------------------------------------------------------------------
# Research window (plan Phase 1)
# ---------------------------------------------------------------------------

WINDOW_BASIS_PREVIOUS_COMPARABLE = "PREVIOUS_COMPARABLE_EVENT"
WINDOW_BASIS_TYPE_DEFAULT = "TYPE_DEFAULT_LOOKBACK"

#: Documented per-type fallback lookbacks used ONLY when no valid previous
#: comparable event exists (never a silent nearest-neighbour guess). The
#: numbers state each type's natural cycle plus slack: earnings ~ a quarter
#: (13 weeks + filing drift), monthly macro ~ one release cycle + slack,
#: GDP ~ a quarterly cycle, FOMC ~ the 8-meetings-a-year spacing.
TYPE_DEFAULT_LOOKBACK_DAYS: Mapping[EventType, int] = {
    EventType.EARNINGS: 98,
    EventType.CPI: 45,
    EventType.PPI: 45,
    EventType.PCE: 45,
    EventType.GDP: 100,
    EventType.EMPLOYMENT_REPORT: 45,
    EventType.JOLTS: 45,
    EventType.RETAIL_SALES: 45,
    EventType.ISM: 45,
    EventType.CONSUMER_SENTIMENT: 45,
    EventType.FOMC_MEETING: 56,
    EventType.FOMC_DECISION: 56,
    EventType.FOMC_PRESS_CONFERENCE: 56,
    EventType.FOMC_MINUTES: 56,
    EventType.FED_SPEECH: 30,
    EventType.FED_BOARD_EVENT: 30,
    EventType.CORPORATE_EVENT: 45,
}
DEFAULT_LOOKBACK_DAYS = 30


@dataclass(frozen=True)
class ResearchWindow:
    """The point-in-time research window contract (plan Phase 1).

    ``end`` is ALWAYS the request as_of. ``basis`` states which rule set
    ``start``; when the fallback fired, ``fallback_reason`` says why in
    words — a fallback window can never masquerade as one anchored on a
    real previous event.
    """

    start: datetime
    end: datetime
    basis: str
    previous_event_id: int | None
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "basis": self.basis,
            "previous_event_id": self.previous_event_id,
            "fallback_reason": self.fallback_reason,
        }


def research_window(event, previous_event, as_of: datetime) -> ResearchWindow:
    """The window ``[previous comparable event -> as_of]``, or the documented
    per-type fallback when no valid previous event exists.

    The comparable-event resolver (``previous_comparable``) stays
    authoritative — this function only consumes its answer. A "previous"
    event that is not actually earlier than as_of falls back too (with the
    anomaly named), rather than producing an empty or inverted window.

    DELIBERATE divergence from the news pipeline's ``news_window`` (which
    starts at prev - 1 day for article-timing slop): the mission contract
    for search is "previous comparable event timestamp -> as_of" exactly,
    and search freshness is day-granular anyway, so the boundary day is
    effectively inclusive without a buffer.
    """
    end = require_utc(as_of, name="as_of")
    if previous_event is not None:
        prev_at = require_utc(previous_event.scheduled_at, name="previous_event.scheduled_at")
        if prev_at < end:
            return ResearchWindow(
                start=prev_at,
                end=end,
                basis=WINDOW_BASIS_PREVIOUS_COMPARABLE,
                previous_event_id=previous_event.event_id,
                fallback_reason=None,
            )
        fallback_reason = (
            "previous comparable event is not earlier than as_of "
            f"({prev_at.isoformat()} >= {end.isoformat()})"
        )
    else:
        fallback_reason = "no valid previous comparable event"
    days = TYPE_DEFAULT_LOOKBACK_DAYS.get(event.event_type, DEFAULT_LOOKBACK_DAYS)
    return ResearchWindow(
        start=end - timedelta(days=days),
        end=end,
        basis=WINDOW_BASIS_TYPE_DEFAULT,
        previous_event_id=None,
        fallback_reason=f"{fallback_reason}; using {days}d {event.event_type.value} lookback",
    )


# ---------------------------------------------------------------------------
# Event research taxonomy (plan Phase 1) — data, not branching code.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchConcept:
    """One research angle: the purpose label the evidence will carry as its
    topic, the query tail, and a priority deciding which concepts survive
    the query cap."""

    purpose: str
    terms: tuple[str, ...]
    priority: float
    result_type: str = "news"  # "news" (dated vertical) | "web" (official/reference)


@dataclass(frozen=True)
class ResearchProfile:
    """The deterministic research profile for one event type (plan Phase 1).

    ``subject_template`` builds the query head from event metadata; concepts
    supply the tails. NOT immutable truth: this table is a maintained
    parameter of the research layer, versioned by
    :data:`WEB_RESEARCH_MODEL_VERSION`.
    """

    profile_key: str
    subject_template: str  # .format(ticker=..., title=..., speaker=...)
    concepts: tuple[ResearchConcept, ...]
    preferred_domains: tuple[str, ...] = ()


_EARNINGS_PROFILE = ResearchProfile(
    profile_key="earnings-v1",
    subject_template="{ticker}",
    concepts=(
        ResearchConcept("guidance_and_results", ("earnings preview", "guidance"), 1.0),
        ResearchConcept("analyst_revisions", ("analyst estimates", "price target"), 0.9),
        ResearchConcept("demand_and_products", ("revenue drivers", "product demand"), 0.8),
        ResearchConcept("supply_and_costs", ("supply chain", "margins", "costs"), 0.7),
        ResearchConcept("regulation_and_legal", ("regulation", "lawsuit", "investigation"), 0.6),
        ResearchConcept("peer_reads", ("competitors earnings", "peer results"), 0.5),
        ResearchConcept("management_commentary", ("CEO comments", "conference"), 0.4),
    ),
    preferred_domains=("reuters.com", "bloomberg.com", "wsj.com"),
)

_INFLATION_PROFILE = ResearchProfile(
    profile_key="inflation-v1",
    subject_template="US {title}",
    concepts=(
        ResearchConcept("inflation_trajectory", ("inflation forecast", "expectations"), 1.0),
        ResearchConcept("shelter_and_services", ("shelter costs", "services inflation"), 0.9),
        ResearchConcept("energy_and_food", ("energy prices", "food prices"), 0.8),
        ResearchConcept("wages_and_labor", ("wage growth", "labor market"), 0.7),
        ResearchConcept("producer_prices", ("producer prices", "input costs"), 0.6),
        ResearchConcept("fed_commentary", ("Federal Reserve inflation comments",), 0.5),
        ResearchConcept("used_vehicles_goods", ("used car prices", "goods prices"), 0.4),
    ),
    preferred_domains=("bls.gov", "reuters.com", "bloomberg.com"),
)

_GDP_PROFILE = ResearchProfile(
    profile_key="gdp-v1",
    subject_template="US {title}",
    concepts=(
        ResearchConcept("growth_forecast", ("GDP forecast", "nowcast"), 1.0),
        ResearchConcept("consumer_strength", ("consumer spending", "retail sales"), 0.9),
        ResearchConcept("investment_and_inventories", ("business investment", "inventories"), 0.8),
        ResearchConcept("trade_balance", ("trade deficit", "imports exports"), 0.7),
        ResearchConcept("industrial_activity", ("industrial production", "manufacturing"), 0.6),
        ResearchConcept("labor_market", ("employment", "jobless claims"), 0.5),
        ResearchConcept("revisions", ("GDP revision",), 0.4),
    ),
    preferred_domains=("bea.gov", "reuters.com"),
)

_EMPLOYMENT_PROFILE = ResearchProfile(
    profile_key="employment-v1",
    subject_template="US {title}",
    concepts=(
        ResearchConcept("payrolls_forecast", ("payrolls forecast", "jobs report preview"), 1.0),
        ResearchConcept("claims_and_layoffs", ("jobless claims", "layoff announcements"), 0.9),
        ResearchConcept("openings_and_quits", ("job openings", "quits rate"), 0.8),
        ResearchConcept("wages", ("average hourly earnings", "wage growth"), 0.7),
        ResearchConcept("participation", ("labor force participation",), 0.6),
        ResearchConcept("private_reads", ("ADP employment", "hiring surveys"), 0.5),
    ),
    preferred_domains=("bls.gov", "reuters.com"),
)

_FOMC_PROFILE = ResearchProfile(
    profile_key="fomc-v1",
    subject_template="Federal Reserve FOMC",
    concepts=(
        ResearchConcept("rate_expectations", ("rate decision expectations", "rate cut"), 1.0),
        ResearchConcept("inflation_picture", ("inflation data", "price pressures"), 0.9),
        ResearchConcept("labor_picture", ("employment data", "labor market"), 0.8),
        ResearchConcept("fed_speeches", ("Fed officials speeches", "Fed commentary"), 0.7),
        ResearchConcept("financial_conditions", ("Treasury yields", "financial conditions"), 0.6),
        ResearchConcept("balance_sheet", ("balance sheet", "QT"), 0.5),
        ResearchConcept("previous_statement", ("FOMC statement minutes",), 0.4),
    ),
    preferred_domains=("federalreserve.gov", "reuters.com"),
)

_FED_SPEECH_PROFILE = ResearchProfile(
    profile_key="fed-speech-v1",
    subject_template="Federal Reserve {speaker}",
    concepts=(
        ResearchConcept("speaker_stance", ("speech", "policy stance"), 1.0),
        ResearchConcept("rate_expectations", ("rate expectations",), 0.8),
        ResearchConcept("recent_commentary", ("recent comments",), 0.6),
    ),
    preferred_domains=("federalreserve.gov",),
)

_GENERIC_PROFILE = ResearchProfile(
    profile_key="generic-v1",
    subject_template="{title}",
    concepts=(
        ResearchConcept("event_developments", ("latest developments",), 1.0),
        ResearchConcept("expectations", ("expectations", "forecast"), 0.8),
        ResearchConcept("official_information", ("official announcement",), 0.6, "web"),
    ),
)

#: Profile lookup by event type. Types not named fall to the generic
#: profile — an honest, weaker plan, never a crash and never a silent skip.
#: CORPORATE_EVENT is DELIBERATELY unlisted: dividends/splits/lockups are
#: too heterogeneous for one concept set, so the generic profile's
#: title-driven queries fit better than a wrong specific one.
RESEARCH_PROFILES: Mapping[EventType, ResearchProfile] = {
    EventType.EARNINGS: _EARNINGS_PROFILE,
    EventType.CPI: _INFLATION_PROFILE,
    EventType.PPI: _INFLATION_PROFILE,
    EventType.PCE: _INFLATION_PROFILE,
    EventType.GDP: _GDP_PROFILE,
    EventType.EMPLOYMENT_REPORT: _EMPLOYMENT_PROFILE,
    EventType.JOLTS: _EMPLOYMENT_PROFILE,
    EventType.RETAIL_SALES: _GDP_PROFILE,
    EventType.ISM: _GDP_PROFILE,
    EventType.CONSUMER_SENTIMENT: _GDP_PROFILE,
    EventType.FOMC_MEETING: _FOMC_PROFILE,
    EventType.FOMC_DECISION: _FOMC_PROFILE,
    EventType.FOMC_PRESS_CONFERENCE: _FOMC_PROFILE,
    EventType.FOMC_MINUTES: _FOMC_PROFILE,
    EventType.FED_SPEECH: _FED_SPEECH_PROFILE,
    EventType.FED_BOARD_EVENT: _FED_SPEECH_PROFILE,
}


def research_profile(event) -> ResearchProfile:
    """The profile for this event's type (generic fallback, never a KeyError)."""
    return RESEARCH_PROFILES.get(event.event_type, _GENERIC_PROFILE)


def event_subject(event, profile: ResearchProfile | None = None) -> str:
    """The deterministic query head, from event metadata only. Public: the
    prediction-market matcher anchors its aboutness on the SAME subject the
    search planner uses, so the two capabilities can never disagree about
    what an event is about."""
    profile = profile or research_profile(event)
    return " ".join(
        profile.subject_template.format(
            ticker=event.ticker or "",
            title=event.title or "",
            speaker=event.speaker or "",
        ).split()
    )


# Backwards-compatible internal alias (module-private call sites).
_subject = event_subject


# ---------------------------------------------------------------------------
# Search plan (plan Phase 2) — deterministic code's output, stored verbatim.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedQuery:
    purpose: str
    query: str
    priority: float
    result_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "query": self.query,
            "priority": self.priority,
            "result_type": self.result_type,
        }


@dataclass(frozen=True)
class SearchPlan:
    """The controlled search plan (plan Phase 2): what will be asked, why,
    and under which window. Query TEXT never carries dates — the window is
    enforced by the executor's bounds and this layer's as-of gate, so no
    wording (human or future-LLM-proposed) can widen the time range."""

    event_id: int | None
    as_of: datetime
    window: ResearchWindow
    profile_key: str
    queries: tuple[PlannedQuery, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": WEB_RESEARCH_MODEL_VERSION,
            "event_id": self.event_id,
            "as_of": self.as_of.isoformat(),
            "window": self.window.to_dict(),
            "profile_key": self.profile_key,
            "queries": [q.to_dict() for q in self.queries],
        }


def build_search_plan(
    event,
    window: ResearchWindow,
    *,
    max_queries: int = MAX_QUERIES_PER_EVENT,
) -> SearchPlan:
    """The deterministic baseline plan: highest-priority concepts first,
    clamped to ``max_queries``, duplicate query strings folded. Consumes
    event METADATA only — never result text, so no fetched content can
    steer what gets searched."""
    profile = research_profile(event)
    subject = _subject(event, profile)
    queries: list[PlannedQuery] = []
    seen: set[str] = set()
    for concept in sorted(profile.concepts, key=lambda c: (-c.priority, c.purpose)):
        if len(queries) >= max(0, max_queries):
            break
        text = " ".join(f"{subject} {' '.join(concept.terms)}".split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        queries.append(
            PlannedQuery(
                purpose=concept.purpose,
                query=text,
                priority=concept.priority,
                result_type=concept.result_type,
            )
        )
    return SearchPlan(
        event_id=event.event_id,
        as_of=window.end,
        window=window,
        profile_key=profile.profile_key,
        queries=tuple(queries),
    )


# ---------------------------------------------------------------------------
# URL normalization / identity
# ---------------------------------------------------------------------------

#: Tracking parameters stripped during canonicalization — they decorate the
#: same document into "different" URLs. Prefix match for utm_*.
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset(
    {"gclid", "fbclid", "igshid", "mc_cid", "mc_eid", "ref", "cmpid", "s_kwcid", "smid"}
)


def canonical_url(url: str | None) -> str | None:
    """The document's canonical identity, or None for an unusable URL.

    Lowercases scheme/host, drops ``www.``, default ports, fragments and
    tracking parameters, trims a trailing slash (except the root). Only
    http(s) URLs are usable — anything else (javascript:, data:, a bare
    word) is None, and the caller rejects the result as INVALID_URL rather
    than storing an unaddressable row.
    """
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        parts = urlsplit(url.strip())
        # .port defers validation to access time: an out-of-range or
        # non-numeric port raises HERE, not in urlsplit — and one hostile
        # URL must cost itself (INVALID_URL), never the whole evaluation.
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
        and not key.lower().startswith(_TRACKING_PARAM_PREFIXES)
    ]
    # Sorted so ?a=1&b=2 and ?b=2&a=1 are ONE identity (stable sort keeps
    # duplicate-key value order) — param order must not defeat dedup or
    # fork the stable citation id.
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, host, path, query, ""))


def domain_of(canonical: str) -> str:
    """The canonical URL's registrable-ish host (no www, keeps subdomains)."""
    return urlsplit(canonical).hostname or ""


def evidence_key(canonical: str) -> str:
    """The STABLE citation id for one canonical document (``web:<sha1-12>``).

    Derived from the canonical URL, not a row id, so the same document cites
    identically across runs and analyses — the id the LLM's evidence_refs
    cite and the validator resolves.
    """
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"web:{digest}"


# ---------------------------------------------------------------------------
# Source quality (plan Phase 2) — metadata, never a binary truth flag.
# ---------------------------------------------------------------------------

SOURCE_TIER_OFFICIAL = "OFFICIAL"
SOURCE_TIER_PRIMARY = "PRIMARY"
SOURCE_TIER_HIGH_QUALITY_NEWS = "HIGH_QUALITY_NEWS"
SOURCE_TIER_INDUSTRY = "INDUSTRY"
SOURCE_TIER_SECONDARY = "SECONDARY"
SOURCE_TIER_SOCIAL = "SOCIAL"
SOURCE_TIER_UNKNOWN = "UNKNOWN"

#: Display/sort order, most authoritative first. UNKNOWN is a CLASSIFICATION
#: ("we cannot place this source"), not an absence.
SOURCE_TIERS: tuple[str, ...] = (
    SOURCE_TIER_OFFICIAL,
    SOURCE_TIER_PRIMARY,
    SOURCE_TIER_HIGH_QUALITY_NEWS,
    SOURCE_TIER_INDUSTRY,
    SOURCE_TIER_SECONDARY,
    SOURCE_TIER_SOCIAL,
    SOURCE_TIER_UNKNOWN,
)

_TIER_RANK: Mapping[str, int] = {tier: i for i, tier in enumerate(SOURCE_TIERS)}

#: Maintained domain->tier table — a PARAMETER of the research layer, not
#: immutable truth (plan Phase 2). Matches the domain or any subdomain.
DOMAIN_TIERS: Mapping[str, str] = {
    # Official statistical / regulatory sources.
    "sec.gov": SOURCE_TIER_OFFICIAL,
    "bls.gov": SOURCE_TIER_OFFICIAL,
    "bea.gov": SOURCE_TIER_OFFICIAL,
    "federalreserve.gov": SOURCE_TIER_OFFICIAL,
    "treasury.gov": SOURCE_TIER_OFFICIAL,
    "census.gov": SOURCE_TIER_OFFICIAL,
    "whitehouse.gov": SOURCE_TIER_OFFICIAL,
    # Primary distribution: company newsrooms / wire services for releases.
    "prnewswire.com": SOURCE_TIER_PRIMARY,
    "businesswire.com": SOURCE_TIER_PRIMARY,
    "globenewswire.com": SOURCE_TIER_PRIMARY,
    # High-quality journalism.
    "reuters.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "bloomberg.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "wsj.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "ft.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "apnews.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "cnbc.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "nytimes.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    "economist.com": SOURCE_TIER_HIGH_QUALITY_NEWS,
    # Industry / ratings research.
    "spglobal.com": SOURCE_TIER_INDUSTRY,
    "moodys.com": SOURCE_TIER_INDUSTRY,
    "fitchratings.com": SOURCE_TIER_INDUSTRY,
    # Secondary financial media / aggregators.
    "seekingalpha.com": SOURCE_TIER_SECONDARY,
    "marketwatch.com": SOURCE_TIER_SECONDARY,
    "barrons.com": SOURCE_TIER_SECONDARY,
    "investing.com": SOURCE_TIER_SECONDARY,
    "benzinga.com": SOURCE_TIER_SECONDARY,
    "fool.com": SOURCE_TIER_SECONDARY,
    "finance.yahoo.com": SOURCE_TIER_SECONDARY,
    # Social / retail narrative (future SOCIAL layer lands here — Phase 17).
    "reddit.com": SOURCE_TIER_SOCIAL,
    "twitter.com": SOURCE_TIER_SOCIAL,
    "x.com": SOURCE_TIER_SOCIAL,
    "stocktwits.com": SOURCE_TIER_SOCIAL,
    "youtube.com": SOURCE_TIER_SOCIAL,
    "tiktok.com": SOURCE_TIER_SOCIAL,
    "facebook.com": SOURCE_TIER_SOCIAL,
}

#: Company IR subdomain shapes ("ir.nvidia.com", "investor.apple.com") —
#: primary-source distribution of the company's own statements.
_PRIMARY_SUBDOMAIN_PREFIXES = ("ir.", "investor.", "investors.")


def classify_source(domain: str) -> str:
    """The source tier for a domain — rules, then the maintained table.

    - any ``.gov`` host is OFFICIAL (a government agency's own words);
    - ``ir.``/``investor.`` subdomains are PRIMARY (company IR);
    - explicit table entries match the domain or any subdomain of it;
    - everything else is UNKNOWN — a classification, never a guess upward.
    """
    host = (domain or "").lower().strip(".")
    if not host:
        return SOURCE_TIER_UNKNOWN
    if host.endswith(".gov") or host == "gov":
        return SOURCE_TIER_OFFICIAL
    # The explicit table wins over the IR-prefix heuristic: "ir.reddit.com"
    # is still reddit (SOCIAL), not a company's investor-relations site —
    # a prefix must never upgrade a host whose parent the table already
    # places.
    parts = host.split(".")
    for start in range(len(parts) - 1):
        candidate = ".".join(parts[start:])
        tier = DOMAIN_TIERS.get(candidate)
        if tier is not None:
            return tier
    if host.startswith(_PRIMARY_SUBDOMAIN_PREFIXES):
        return SOURCE_TIER_PRIMARY
    return SOURCE_TIER_UNKNOWN


# ---------------------------------------------------------------------------
# Acceptance pipeline: normalize -> dedup -> as-of gate -> relevance ->
# tier -> topic -> evidence decision (plan Phase 2 pipeline).
# ---------------------------------------------------------------------------

REJECT_INVALID_URL = "INVALID_URL"
REJECT_DUPLICATE = "DUPLICATE"
REJECT_AFTER_AS_OF = "PUBLISHED_AFTER_AS_OF"
REJECT_BEFORE_WINDOW_START = "PUBLISHED_BEFORE_WINDOW_START"
REJECT_UNPLACEABLE_IN_TIME = "UNPLACEABLE_IN_TIME"
REJECT_LOW_RELEVANCE = "LOW_RELEVANCE"
REJECT_SUSPICIOUS_INSTRUCTION = "SUSPICIOUS_INSTRUCTION"
REJECT_OVER_DOCUMENT_CAP = "OVER_DOCUMENT_CAP"
REJECT_OVER_ACCEPT_CAP = "OVER_ACCEPT_CAP"


@dataclass(frozen=True)
class EvidenceCandidate:
    """One considered document with every decision this layer made about it.

    ``accepted=False`` rows carry their named ``reject_reason`` — they are
    part of the record, not discarded. ``safe_title``/``safe_snippet`` are
    the ONLY model-facing text forms; the raw fields exist for
    display/provenance and stay fenced away from prompts.
    """

    evidence_key: str
    canonical_url: str
    url: str
    title: str
    safe_title: str
    snippet: str
    safe_snippet: str
    publisher: str | None
    domain: str
    published_at: datetime | None
    retrieved_at: datetime
    source_tier: str
    topic: str
    relevance: float
    rank: int | None
    result_type: str
    provider: str
    query: str
    purpose: str
    suspicious_instruction: bool
    accepted: bool
    reject_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_key": self.evidence_key,
            "canonical_url": self.canonical_url,
            "url": self.url,
            "title": self.title,
            "safe_title": self.safe_title,
            "snippet": self.snippet,
            "safe_snippet": self.safe_snippet,
            "publisher": self.publisher,
            "domain": self.domain,
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "retrieved_at": self.retrieved_at.isoformat(),
            "source_tier": self.source_tier,
            "topic": self.topic,
            "relevance": self.relevance,
            "rank": self.rank,
            "result_type": self.result_type,
            "provider": self.provider,
            "query": self.query,
            "purpose": self.purpose,
            "suspicious_instruction": self.suspicious_instruction,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class ResearchOutcome:
    """The full, auditable result of evaluating one refresh's raw results."""

    candidates: tuple[EvidenceCandidate, ...]
    suppressed_suspicious: int
    results_considered: int
    results_accepted: int

    @property
    def accepted(self) -> tuple[EvidenceCandidate, ...]:
        return tuple(c for c in self.candidates if c.accepted)

    @property
    def rejected(self) -> tuple[EvidenceCandidate, ...]:
        return tuple(c for c in self.candidates if not c.accepted)

    def source_mix(self) -> dict[str, int]:
        return dict(Counter(c.source_tier for c in self.accepted))

    def topic_mix(self) -> dict[str, int]:
        return dict(Counter(c.topic for c in self.accepted))


def relevance_score(
    text: str,
    *,
    subject_tokens: frozenset[str],
    concept_tokens: frozenset[str],
) -> float:
    """Deterministic aboutness of ``text`` for this event (QUANT tier).

    Subject overlap (the ticker/series the event IS about) dominates;
    concept overlap (the research angle) refines. Source tier deliberately
    plays NO part — reliability is separate metadata, never folded into
    aboutness (plan Phase 2: "search ranking is not evidence reliability",
    and neither is source quality relevance).
    """
    text_tokens = frozenset(tokens(text))
    if not text_tokens:
        return 0.0
    subject_hit = (
        len(subject_tokens & text_tokens) / len(subject_tokens)
        if subject_tokens
        else 0.0
    )
    concept_hit = (
        len(concept_tokens & text_tokens) / len(concept_tokens)
        if concept_tokens
        else 0.0
    )
    return round(0.6 * subject_hit + 0.4 * concept_hit, 4)


def _as_aware_utc(value: datetime) -> datetime:
    """A naive instant is treated as UTC (the platform's provider-payload
    convention, e.g. Brave ``page_age``) — never compared naive-vs-aware,
    which would raise instead of yielding a named decision."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _passes_time_gates(
    published_at: datetime | None,
    retrieved_at: datetime,
    *,
    window_start: datetime,
    as_of: datetime,
) -> str | None:
    """None when admissible in ``[window_start, as_of]``; else the reason.

    BOTH bounds are enforced here, deterministically — the provider's
    freshness filter is only a hint (its adapter says so), so a provider
    that returns months-old pre-window articles cannot smuggle stale
    context into the "since the previous event" evidence set.

    A result with no publication time is admissible only through its
    retrieval clock: being FETCHED at/before as_of proves it existed then
    (it cannot prove recency, so undated rows skip the start bound and
    carry ``published_at=None`` visibly). When even retrieval cannot be
    placed at/before as_of (historical replay), the row is unplaceable in
    time and is excluded — conservative, with the exclusion named
    (§44 rule 18).
    """
    if published_at is not None:
        published = _as_aware_utc(published_at)
        if published > as_of:
            return REJECT_AFTER_AS_OF
        if published < window_start:
            return REJECT_BEFORE_WINDOW_START
        return None
    return (
        None if _as_aware_utc(retrieved_at) <= as_of
        else REJECT_UNPLACEABLE_IN_TIME
    )


def evaluate_results(
    raw_results: Sequence,
    *,
    event,
    plan: SearchPlan,
    as_of: datetime,
    max_unique_documents: int = MAX_UNIQUE_DOCUMENTS,
    max_accepted: int = MAX_ACCEPTED_EVIDENCE,
    min_relevance: float = MIN_RELEVANCE_ACCEPT,
    dedupe_threshold: float = DEFAULT_DEDUPE_JACCARD,
) -> ResearchOutcome:
    """Turn raw provider results into admitted-or-rejected evidence.

    ``raw_results`` are :class:`libs.web_search.provider.SearchResult`-shaped
    objects (structural: any object with the same attributes works — this
    module never imports the provider package). Order matters and is
    deterministic: results are processed in plan order (the caller executes
    queries in plan order), so identical inputs always yield identical
    decisions.
    """
    as_of = require_utc(as_of, name="as_of")
    profile = research_profile(event)
    subject_tokens = frozenset(tokens(_subject(event, profile)))
    # Fallback aboutness terms for a purpose the profile no longer names
    # (a stored plan replayed against a newer profile): the query's own words.
    purpose_terms: dict[str, frozenset[str]] = {
        q.purpose: frozenset(tokens(q.query)) for q in plan.queries
    }
    concept_terms: dict[str, frozenset[str]] = {
        c.purpose: frozenset(tokens(" ".join(c.terms)))
        for c in profile.concepts
    }

    candidates: list[EvidenceCandidate] = []
    seen_keys: dict[str, str] = {}  # evidence_key -> first canonical
    seen_title_shingles: list[frozenset[str]] = []
    unique_documents = 0

    def decide(result, *, accepted: bool, reject_reason: str | None,
               canonical: str | None, relevance: float,
               safe_title, safe_snippet, suspicious: bool, topic: str) -> None:
        can = canonical or (result.url if isinstance(result.url, str) else "")
        key = evidence_key(can) if canonical else f"web:invalid:{len(candidates)}"
        candidates.append(
            EvidenceCandidate(
                evidence_key=key,
                canonical_url=can,
                url=result.url if isinstance(result.url, str) else "",
                title=result.title,
                safe_title=safe_title.text if safe_title else "",
                snippet=result.snippet,
                safe_snippet=safe_snippet.text if safe_snippet else "",
                publisher=result.publisher,
                domain=domain_of(canonical) if canonical else "",
                published_at=result.published_at,
                retrieved_at=result.retrieved_at,
                source_tier=classify_source(domain_of(canonical)) if canonical else SOURCE_TIER_UNKNOWN,
                topic=topic,
                relevance=relevance,
                rank=result.rank,
                result_type=result.result_type,
                provider=result.provider,
                query=result.query,
                purpose=topic,
                suspicious_instruction=suspicious,
                accepted=accepted,
                reject_reason=reject_reason,
            )
        )

    query_purpose = {q.query: q.purpose for q in plan.queries}

    for result in raw_results:
        topic = query_purpose.get(result.query, "unplanned")
        safe_title = sanitize_for_llm(result.title)
        safe_snippet = sanitize_for_llm(result.snippet)
        suspicious = bool(
            safe_title.suspicious_instruction or safe_snippet.suspicious_instruction
        )

        canonical = canonical_url(result.url)
        if canonical is None:
            decide(result, accepted=False, reject_reason=REJECT_INVALID_URL,
                   canonical=None, relevance=0.0, safe_title=safe_title,
                   safe_snippet=safe_snippet, suspicious=suspicious, topic=topic)
            continue

        # BOTH time gates run BEFORE dedup registration: an out-of-window
        # document must never register dedup state, or an ADMISSIBLE
        # in-window near-duplicate (the same story republished inside the
        # window) would fold against a rejected twin and be lost entirely.
        time_reject = _passes_time_gates(
            result.published_at, result.retrieved_at,
            window_start=_as_aware_utc(plan.window.start), as_of=as_of,
        )
        if time_reject is not None:
            decide(result, accepted=False, reject_reason=time_reject,
                   canonical=canonical, relevance=0.0, safe_title=safe_title,
                   safe_snippet=safe_snippet, suspicious=suspicious, topic=topic)
            continue

        if suspicious:
            # The suspicious check ALSO precedes dedup registration, for the
            # adversarial version of the same reason: an injected copy of a
            # real headline must never become the dedup representative and
            # suppress the clean original as a "duplicate" — injection may
            # cost itself, never legitimate evidence (§81, plan Phase 2).
            decide(result, accepted=False,
                   reject_reason=REJECT_SUSPICIOUS_INSTRUCTION,
                   canonical=canonical, relevance=0.0, safe_title=safe_title,
                   safe_snippet=safe_snippet, suspicious=True, topic=topic)
            continue

        key = evidence_key(canonical)
        title_sh = story_shingles(result.title)
        is_near_dup = key in seen_keys or any(
            jaccard(title_sh, prior) >= dedupe_threshold
            for prior in seen_title_shingles
            if title_sh
        )
        if is_near_dup:
            decide(result, accepted=False, reject_reason=REJECT_DUPLICATE,
                   canonical=canonical, relevance=0.0, safe_title=safe_title,
                   safe_snippet=safe_snippet, suspicious=suspicious, topic=topic)
            continue
        seen_keys[key] = canonical
        if title_sh:
            seen_title_shingles.append(title_sh)

        unique_documents += 1
        if unique_documents > max_unique_documents:
            decide(result, accepted=False, reject_reason=REJECT_OVER_DOCUMENT_CAP,
                   canonical=canonical, relevance=0.0, safe_title=safe_title,
                   safe_snippet=safe_snippet, suspicious=suspicious, topic=topic)
            continue

        relevance = relevance_score(
            f"{result.title} {result.snippet}",
            subject_tokens=subject_tokens,
            concept_tokens=concept_terms.get(topic, purpose_terms.get(topic, frozenset())),
        )
        if relevance < min_relevance:
            decide(result, accepted=False, reject_reason=REJECT_LOW_RELEVANCE,
                   canonical=canonical, relevance=relevance, safe_title=safe_title,
                   safe_snippet=safe_snippet, suspicious=suspicious, topic=topic)
            continue

        decide(result, accepted=True, reject_reason=None, canonical=canonical,
               relevance=relevance, safe_title=safe_title,
               safe_snippet=safe_snippet, suspicious=False, topic=topic)

    # Enforce the accept cap on the RANKED accepted set: relevance first,
    # then source-tier authority, then provider rank — deterministic ties.
    accepted_indexes = [i for i, c in enumerate(candidates) if c.accepted]
    ranked = sorted(
        accepted_indexes,
        key=lambda i: (
            -(candidates[i].relevance or 0.0),
            SOURCE_TIERS.index(candidates[i].source_tier)
            if candidates[i].source_tier in SOURCE_TIERS
            else len(SOURCE_TIERS),
            candidates[i].rank if candidates[i].rank is not None else 10**6,
            i,
        ),
    )
    for overflow_index in ranked[max(0, max_accepted):]:
        candidates[overflow_index] = replace(
            candidates[overflow_index],
            accepted=False,
            reject_reason=REJECT_OVER_ACCEPT_CAP,
        )

    accepted_count = sum(1 for c in candidates if c.accepted)
    # Derived from the rows, not a branch counter: EVERY flagged row counts
    # (even one that also failed an earlier gate), so the diagnostic number
    # always equals what an auditor sees on the candidates themselves.
    suppressed = sum(1 for c in candidates if c.suspicious_instruction)
    return ResearchOutcome(
        candidates=tuple(candidates),
        suppressed_suspicious=suppressed,
        results_considered=len(candidates),
        results_accepted=accepted_count,
    )


def near_duplicate_of_structured_news(
    candidate_title: str,
    news_titles: Iterable[str],
    *,
    threshold: float = DEFAULT_DEDUPE_JACCARD,
) -> bool:
    """True when a web result retells a structured-news article (Phase 5:
    never show the same Reuters story twice because two retrieval systems
    found it). Compared on story shingles, the news pipeline's own key."""
    candidate = story_shingles(candidate_title)
    if not candidate:
        return False
    return any(
        jaccard(candidate, story_shingles(title)) >= threshold
        for title in news_titles
    )
