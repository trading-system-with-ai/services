"""News intelligence — normalisation, deduplication, story clustering,
materiality, novelty, source quality and the explainable evidence score
(event spec §21-§27, §59, §79, §81, §96; audit §5.1, §11.5; Phase D unit U2).

Pure stdlib, deterministic, **no I/O**. Like every other module under
``libs/trading_core/events/`` this one may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4 static guard):
the gateway seam (``apps/gateway/event_news.py``) hands it plain
:class:`RawArticle` values — built from provider dataclasses or from the
``news_articles`` mirror, this module cannot tell which — and gets a frozen
:class:`NewsIntelResult` back.

Six ideas carry the module:

1. **A count is a claim, so the counts are earned** (§26). The output is not
   "143 articles" but "143 raw / 87 unique / 19 clusters / 7 material / 4
   themes", and each of those five numbers is produced by a stage that can
   be inspected on its own. :func:`dedupe` explains every drop with a
   ``duplicate_of``; :func:`cluster_articles` explains every link with the
   rule that fired; :func:`score_materiality` explains every category with
   the ``matched_terms`` that chose it.

2. **Materiality is not sentiment** (§24, spelled out in the spec: "a
   negative article can be immaterial, a neutral regulatory filing can be
   highly material"). There is no sentiment field anywhere in this module —
   not in the dataclasses, not in the payload, not as a hidden term in the
   score. A lexicon maps text to one of the sixteen §24 categories and the
   category's published weight IS the materiality; nothing scores the
   article's mood.

3. **The score is a product of five named factors, and every one of them
   travels with it** (§25). :class:`EvidenceScore` carries ``relevance``,
   ``materiality``, ``novelty``, ``source_quality`` and ``decay`` beside the
   product, because §13's rule against "a mysterious LLM-generated score"
   applies here too — the UI's ⓘ tooltip renders the components verbatim and
   a reader can multiply them by hand. Two products travel, not one: ``score``
   includes decay and decides the ORDER, ``score_no_decay`` excludes it and
   decides what COUNTS as a §26 material development. Time is a reason to
   read something later, never a reason for it to stop having happened.

4. **The as-of gate is absolute** (§96). :func:`analyze_window` drops every
   article whose ``published_at`` is later than ``as_of`` before the first
   stage runs, so an article published after the decision instant cannot
   reach a cluster, a count, a theme or a score. Novelty is measured against
   EARLIER clusters only for the same reason: a story is not un-novel
   because of something published after it.

5. **Retrieved text is untrusted** (§81). Nothing here executes, follows or
   even preserves an instruction found inside an article.
   :func:`sanitize_for_llm` strips markup, control characters and URLs, caps
   the length, and — critically — *flags* imperative "ignore previous
   instructions"-shaped lines with ``suspicious_instruction=True`` rather
   than silently deleting them, so the seam can log the attempt. The flag is
   a report, never a branch: no code path in this platform treats article
   text as instructions in the first place.

6. **Ids are deterministic functions of content** (§72 discipline). A
   cluster id is ``"c:" + sha1(canonical source_id)[:12]`` and an evidence id
   is ``"news:" + source_id``, so the same window analysed twice — in a test,
   in a re-run, in a cache comparison — yields byte-identical ids, and a
   stored ``cluster_id`` from yesterday still names the same story today.

Provenance (§91): the article fields are DATA — publisher, headline, instant
and URL are the provider's verbatim. Every number this module produces —
similarity, materiality, novelty, decay, the product — is QUANT, computed
here from those fields alone. Nothing is inferred, and no LLM runs.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from libs.trading_core.events.taxonomy import UTC, require_utc

__all__ = [
    "CATEGORY_ORDER",
    "CATEGORY_WEIGHTS",
    "CLUSTER_MAX_SHARE",
    "CLUSTER_MIN_CAP",
    "DEFAULT_CLUSTER_ENTITY_WINDOW",
    "DEFAULT_CLUSTER_JACCARD",
    "DEFAULT_CLUSTER_TITLE_WINDOW",
    "DEFAULT_DEDUPE_JACCARD",
    "DEFAULT_SHINGLE_K",
    "DECAY_FLOOR",
    "HALF_LIFE_DAYS",
    "MATERIAL_SCORE_THRESHOLD",
    "MATERIALITY_LEXICON",
    "MAX_TAGS_AS_ENTITIES",
    "NEWS_MODEL_VERSION",
    "RELEVANCE_DESCRIPTION_ONLY",
    "RELEVANCE_TITLE_OR_TAGGED",
    "SANITIZE_MAX_CHARS",
    "SOURCE_QUALITY",
    "TEMPLATE_STOPWORDS",
    "UBIQUITOUS_ENTITY_SHARE",
    "UNKNOWN_SOURCE_QUALITY",
    "ArticleCluster",
    "EvidenceScore",
    "MaterialityResult",
    "NewsIntelResult",
    "NewsTheme",
    "RawArticle",
    "SanitizedText",
    "analyze_window",
    "cluster_articles",
    "dedupe",
    "jaccard",
    "materiality_of",
    "normalize",
    "novelty_of",
    "salient_entities",
    "sanitize_for_llm",
    "score_evidence",
    "score_materiality",
    "shingles",
    "source_quality",
    "story_shingles",
    "story_tokens",
    "themes_from_clusters",
    "ticker_relevance",
    "time_decay",
    "tokens",
]

#: Bumped whenever a stage, weight or threshold changes so a persisted
#: cluster/materiality column can be told apart from one written by a later
#: definition (same discipline as ``IMPORTANCE_MODEL_VERSION``).
NEWS_MODEL_VERSION = "news-intel-v1"

# ---------------------------------------------------------------------------
# Tunables — every one of them is a research parameter, spelled once
# ---------------------------------------------------------------------------

#: Shingle width for the title-similarity measures. Three words is short
#: enough that a ten-word headline still yields eight shingles and long
#: enough that "the" and "of" cannot carry a match on their own.
DEFAULT_SHINGLE_K = 3

#: §23 near-duplicate threshold on normalised-title 3-shingle Jaccard.
#: Syndicated copies of one story rewrite a word or two of the headline;
#: 0.8 tolerates that and still separates two genuinely different stories.
DEFAULT_DEDUPE_JACCARD = 0.8

#: §23 story-clustering threshold. Looser than dedupe on purpose: two
#: publishers covering the same development from different angles share
#: roughly half a headline, not four fifths of it.
DEFAULT_CLUSTER_JACCARD = 0.45

#: The entity rule's time bound — two articles naming the same two salient
#: entities are the same development only if they landed close together.
DEFAULT_CLUSTER_ENTITY_WINDOW = timedelta(hours=48)

#: The TITLE rule's time bound. A story is a development, and a development
#: happens on a date: two headlines that phrase the same sentence three weeks
#: apart are a recurring beat ("Apple raises dividend"), not one story. A week
#: is loose enough to hold a development that keeps producing coverage and
#: tight enough that a monthly template cannot chain a whole window together.
DEFAULT_CLUSTER_TITLE_WINDOW = timedelta(days=7)

#: Number of shared salient entities the entity rule requires.
CLUSTER_ENTITY_MIN_SHARED = 2

#: The smallest the ``max_share`` cap may ever compute to. A share of a small
#: window rounds to one or zero, which would forbid the two- and
#: three-article clusters that ARE the normal output of clustering, so the cap
#: is not allowed below this.
CLUSTER_MIN_CAP = 5

#: A hard ceiling on how much of a window one story may claim. No real
#: development accounts for half a ticker's month of coverage; a cluster that
#: large is a chaining failure, not a mega-story, so growth past this share of
#: the window's unique articles forces a NEW cluster instead. It is a
#: circuit-breaker, not a tuning knob: with leader clustering it should never
#: fire on healthy data, and when it does fire the shape of the output stays
#: readable instead of collapsing to one row.
CLUSTER_MAX_SHARE = 0.40

#: §22 time decay: a fortnight-old article counts half as much as today's.
HALF_LIFE_DAYS = 14.0

#: …but never zero. A very old article that is still the ONLY evidence for a
#: development should rank last, not vanish, so decay bottoms out here.
DECAY_FLOOR = 0.2

#: §25 cut for "material development" in the §26 counts. A product of five
#: factors reaches 0.25 when, say, an in-title ticker (1.0) meets an
#: EARNINGS headline (0.8) that is fully novel (1.0) from a mid-quality
#: publisher (0.5) eleven days old (0.6) — i.e. the floor of what a reader
#: would want surfaced. Documented and constant so the count is comparable
#: across events; it is NOT tuned per ticker.
MATERIAL_SCORE_THRESHOLD = 0.25

#: §22 relevance tiers. Ternary on purpose: the ticker is either the story's
#: subject (tagged by the provider or named in the headline), a body mention,
#: or absent — and an absent ticker is an excluded article, not a small
#: number.
RELEVANCE_TITLE_OR_TAGGED = 1.0
RELEVANCE_DESCRIPTION_ONLY = 0.7

#: §81 default cap for text handed to an LLM.
SANITIZE_MAX_CHARS = 600

# ---------------------------------------------------------------------------
# Source quality (§22) — an extensible substring table, not a hard-coded set
# ---------------------------------------------------------------------------

#: Publisher-name substrings (lower-cased) mapped to a quality weight. The
#: table is matched by substring so "Reuters" and "Thomson Reuters Business
#: News" both land on 1.0, and it is deliberately a plain mapping so a new
#: publisher is one line rather than a code change. Longer keys are tried
#: first, so "wall street journal" cannot be shadowed by a shorter key.
SOURCE_QUALITY: Mapping[str, float] = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "wsj": 1.0,
    "wall street journal": 1.0,
    "financial times": 1.0,
    "associated press": 1.0,
    "cnbc": 1.0,
    "company": 1.0,
    "investor relations": 1.0,
    "ir": 1.0,
    "sec": 1.0,
    "barron's": 0.8,
    "barrons": 0.8,
    "marketwatch": 0.8,
    "benzinga": 0.7,
    "the motley fool": 0.5,
    "motley fool": 0.5,
    "seeking alpha": 0.5,
    "investorplace": 0.5,
    "zacks": 0.5,
}

#: An unrecognised publisher is neither trusted nor punished — it sits at the
#: same 0.5 as the retail-commentary tier rather than at zero, because an
#: unknown name is missing information about the source, not evidence that
#: the source is bad.
UNKNOWN_SOURCE_QUALITY = 0.5

#: Keys ordered longest-first so substring matching is unambiguous.
_SOURCE_QUALITY_KEYS: tuple[str, ...] = tuple(
    sorted(SOURCE_QUALITY, key=len, reverse=True)
)

# ---------------------------------------------------------------------------
# Materiality (§24) — a lexicon, its weights, and nothing about sentiment
# ---------------------------------------------------------------------------

#: The §24 categories in the order they are reported. ``OTHER`` is last and
#: is a real answer, not a failure: most of a news window genuinely is
#: routine coverage.
CATEGORY_ORDER: tuple[str, ...] = (
    "EARNINGS",
    "GUIDANCE",
    "PRODUCT",
    "CUSTOMER",
    "CONTRACT",
    "REGULATION",
    "LEGAL",
    "MANAGEMENT",
    "M&A",
    "CAPITAL_ALLOCATION",
    "SUPPLY_CHAIN",
    "COMPETITION",
    "ANALYST_REVISION",
    "MACRO_EXPOSURE",
    "INDUSTRY",
    "OTHER",
)

#: Per-category materiality weight (§24/§25). These are research parameters
#: about how much a category MOVES a stock, not a ranking of how interesting
#: it is to read, and explicitly not a sentiment: "guidance cut" and
#: "guidance raised" are both GUIDANCE at 0.9.
CATEGORY_WEIGHTS: Mapping[str, float] = {
    "GUIDANCE": 0.9,
    "M&A": 0.9,
    "EARNINGS": 0.8,
    "REGULATION": 0.8,
    "LEGAL": 0.7,
    "CONTRACT": 0.7,
    "CUSTOMER": 0.6,
    "PRODUCT": 0.6,
    "MANAGEMENT": 0.6,
    "CAPITAL_ALLOCATION": 0.6,
    "SUPPLY_CHAIN": 0.6,
    "ANALYST_REVISION": 0.5,
    "COMPETITION": 0.5,
    "MACRO_EXPOSURE": 0.4,
    "INDUSTRY": 0.3,
    "OTHER": 0.1,
}

#: Term lexicon per category, matched against the normalised text on word
#: boundaries. Multi-word terms are matched as phrases. The matched terms
#: travel into the payload (:attr:`MaterialityResult.matched_terms`) — that
#: is the whole point: a category with no visible evidence for it is exactly
#: the "mysterious score" §13 forbids.
MATERIALITY_LEXICON: Mapping[str, tuple[str, ...]] = {
    "EARNINGS": (
        "earnings",
        "quarterly results",
        "q1 results",
        "q2 results",
        "q3 results",
        "q4 results",
        "eps",
        "earnings per share",
        "revenue beat",
        "revenue miss",
        "top line",
        "bottom line",
        "reports results",
        "fiscal quarter",
        "beat estimates",
        "missed estimates",
    ),
    "GUIDANCE": (
        "guidance",
        "outlook",
        "forecast",
        "raises full year",
        "cuts full year",
        "guides",
        "full-year view",
        "reaffirms",
        "preannounce",
        "pre-announce",
        "warns on",
        "profit warning",
        "raises outlook",
        "raises forecast",
        "raises guidance",
        "cuts outlook",
        "cuts forecast",
        "cuts guidance",
        "lowers outlook",
        "lowers forecast",
        "expects revenue",
        "full year outlook",
        "guidance range",
    ),
    "PRODUCT": (
        "launch",
        "launches",
        "unveils",
        "new product",
        "product line",
        "release of",
        "next generation",
        "next-generation",
        "roadmap",
        "chip",
        "device",
        "platform launch",
        "ships",
        "unveil",
        "unveiled",
        "launched",
        "launching",
        "iphone",
        "ipad",
        "macbook",
        "mac",
        "apple watch",
        "airpods",
        "vision pro",
        "siri",
        "new model",
        "new models",
        "announces device",
        "announced device",
        "chips",
        "silicon",
        "hardware",
        "software update",
    ),
    "CUSTOMER": (
        "customer",
        "customers",
        "client win",
        "adoption",
        "orders from",
        "demand from",
        "signs up",
        "user growth",
        "subscriber",
    ),
    "CONTRACT": (
        "contract",
        "deal with",
        "agreement",
        "partnership",
        "awarded",
        "wins order",
        "order worth",
        "supply agreement",
        "multi-year deal",
        "signs contract",
    ),
    "REGULATION": (
        "regulator",
        "regulators",
        "regulation",
        "regulatory",
        "sec filing",
        "ftc",
        "doj",
        "antitrust",
        "approval",
        "fda",
        "export controls",
        "sanctions",
        "license",
        "compliance order",
        "probe",
        "investigation",
        "regulate",
        "regulates",
        "eu",
        "european commission",
        "dma",
        "digital markets act",
        "commission",
        "compliance",
        "watchdog",
        "ruling",
        "mandate",
        "senators",
        "lawmakers",
        "congress",
    ),
    "LEGAL": (
        "lawsuit",
        "sues",
        "sued",
        "litigation",
        "court",
        "settlement",
        "patent dispute",
        "class action",
        "verdict",
        "appeal",
        "damages",
        "injunction",
        "legal challenge",
        "judge",
        "antitrust",
        "doj",
        "department of justice",
        "sues over",
        "files suit",
        "legal battle",
        "settle",
        "settles",
        "jury",
        "plaintiff",
        "subpoena",
    ),
    "MANAGEMENT": (
        "ceo",
        "cfo",
        "coo",
        "chief executive",
        "chief financial",
        "resigns",
        "steps down",
        "appoints",
        "names new",
        "board of directors",
        "successor",
        "executive departure",
    ),
    "M&A": (
        "acquisition",
        "acquires",
        "to acquire",
        "merger",
        "merges",
        "takeover",
        "buyout",
        "stake in",
        "divest",
        "divestiture",
        "spin-off",
        "spinoff",
        "bid for",
        "acquire",
        "acquired",
        "acquiring",
        "acquisitions",
        "merge",
        "merged",
        "deal to buy",
        "agrees to buy",
        "buys stake",
    ),
    "CAPITAL_ALLOCATION": (
        "buyback",
        "share repurchase",
        "repurchase program",
        "dividend",
        "special dividend",
        "capital return",
        "debt offering",
        "convertible notes",
        "secondary offering",
        "capex plan",
        "repurchase",
        "repurchases",
        "buybacks",
        "capex",
        "capital allocation",
        "raises dividend",
        "dividend hike",
    ),
    "SUPPLY_CHAIN": (
        "supply chain",
        "supplier",
        "suppliers",
        "shortage",
        "capacity constraint",
        "inventory",
        "logistics",
        "foundry",
        "production halt",
        "factory",
        "lead times",
        "supply crunch",
        "tariff",
        "tariffs",
        "foundries",
        "memory chips",
        "memory chip",
        "dram",
        "nand",
        "tsmc",
        "component",
        "components",
        "assembler",
        "contract manufacturer",
        "sourcing",
        "procurement",
        "supply constraints",
    ),
    "COMPETITION": (
        "competitor",
        "competition",
        "rival",
        "rivals",
        "market share",
        "loses share",
        "gains share",
        "challenger",
        "price war",
    ),
    "ANALYST_REVISION": (
        "upgrade",
        "upgrades",
        "downgrade",
        "downgrades",
        "price target",
        "initiates coverage",
        "analyst",
        "rating",
        "overweight",
        "underweight",
        "buy rating",
        "sell rating",
        "estimates raised",
        "estimates cut",
        # A sell-side note that changes nothing is still a sell-side note:
        # "reiterates"/"maintains" are the most common shape on the wire and
        # were the reason "Needham Reiterates Hold on Apple" classified as
        # OTHER at 0.1 instead of ANALYST_REVISION at 0.5.
        "reiterates",
        "reiterated",
        "maintains",
        "maintained",
        "upgrade",
        "upgraded",
        "downgraded",
        "initiates",
        "initiated coverage",
        "outperform",
        "underperform",
        "hold rating",
        "neutral rating",
        "equal weight",
        "market perform",
        "raises price target",
        "lowers price target",
        "cuts price target",
        "reiterates buy",
        "reiterates hold",
        "reiterates sell",
        "analyst forecasts",
        "analysts",
        "sees upside",
    ),
    "MACRO_EXPOSURE": (
        "tariff",
        "tariffs",
        "inflation",
        "interest rates",
        "federal reserve",
        "fed",
        "recession",
        "currency",
        "dollar strength",
        "trade war",
        "gdp",
        "jobs report",
    ),
    "INDUSTRY": (
        "sector",
        "industry",
        "peers",
        "index",
        "etf",
        "market wrap",
        "stocks to watch",
        "sector rotation",
        "industry outlook",
    ),
}

#: Compiled once at import: {category: [(term, pattern), ...]}. Word
#: boundaries on both ends so "fed" does not fire inside "federated" and
#: "ir" does not fire inside "first".
_LEXICON_PATTERNS: Mapping[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    category: tuple(
        (term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)"))
        for term in terms
    )
    for category, terms in MATERIALITY_LEXICON.items()
}

# ---------------------------------------------------------------------------
# §81 prompt-injection detection — patterns, never behaviour
# ---------------------------------------------------------------------------

#: Imperative shapes that an article body has no business containing. A hit
#: sets :attr:`SanitizedText.suspicious_instruction`; it never changes what
#: the platform does, because retrieved text is never executed here in the
#: first place. The flag exists so the seam can log and the UI can warn.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
               r"(?:instruction|instructions|prompt|prompts|direction|directions)",
               re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|"
               r"earlier|system)\s*\w*", re.IGNORECASE),
    re.compile(r"forget\s+(?:everything|all)\b", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.IGNORECASE),
    re.compile(r"\bsystem\s*(?:prompt|message)\b", re.IGNORECASE),
    re.compile(r"</?\s*(?:system|assistant|user|instructions?)\s*>", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(?:the\s+)?(?:previous|prior|system)\b",
               re.IGNORECASE),
    re.compile(r"reveal\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)",
               re.IGNORECASE),
    re.compile(r"override\s+(?:your|the)\s+\w+", re.IGNORECASE),
)

_TAG_RE = re.compile(r"<[^>]*>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'&]*")
_ENTITY_TOKEN_RE = re.compile(r"[A-Z][A-Za-z0-9&.\-']*")

#: A handful of HTML entities worth decoding to their character so a
#: normalised title does not carry "amp" as a token.
_ENTITIES: Mapping[str, str] = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&#39;": "'",
    "&nbsp;": " ",
}

#: Bound on the strip/decode fixed-point loop in :func:`sanitize_for_llm`.
#: Each pass peels one layer of entity encoding, so a handful covers any
#: realistic nesting ("&amp;amp;lt;") while keeping the work constant-time on
#: hostile input — an unbounded loop over attacker-chosen text is its own
#: denial-of-service.
_MAX_ENTITY_DECODE_PASSES = 4

#: How many provider ticker tags an article may carry before its tag list
#: stops naming a subject and starts naming a basket. Three is the shape of a
#: real multi-party story — an acquirer, a target and an incumbent — while a
#: market wrap carries ten to forty-five.
MAX_TAGS_AS_ENTITIES = 3

#: Capitalised words too common to be a salient entity on their own. Without
#: this every headline shares "The" and "A" and the entity rule collapses.
_ENTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "A", "AN", "AND", "AS", "AT", "BUT", "BY", "FOR", "FROM", "HOW", "IN",
        "IS", "IT", "OF", "ON", "OR", "THE", "TO", "UP", "WHY", "WITH", "WHAT",
        "WHEN", "AFTER", "BEFORE", "THIS", "THAT", "THESE", "THOSE", "NEW",
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
        "SUNDAY", "TODAY", "STOCK", "STOCKS", "SHARES", "REPORT", "NEWS",
        "MARKET", "MARKETS", "HERE", "WHY'S", "COULD", "WILL", "SHOULD",
        # Headline furniture that is capitalised in title case and therefore
        # looked like a named entity. The live AAPL window linked unrelated
        # stories on {"SAYS", "HEADING"} and fused a quarter's coverage on
        # {"Q3", "EARNINGS"} — a fiscal-period label names WHEN, not WHO, and
        # "shared entities" is supposed to mean shared SUBJECTS.
        "SAYS", "SAID", "HEADING", "WATCH", "MOVING", "MOVERS", "UPDATE",
        "UPDATED", "PREVIEW", "RECAP", "TOP", "THINGS", "KNOW", "BEFORE",
        "BELL", "HOURS", "SESSION", "PREMARKET", "WEEK", "WEEKLY", "TONIGHT",
        "LATEST", "LIVE", "WRAP", "AHEAD", "AMID", "EARNINGS", "RESULTS",
        "REVENUE", "SALES", "GUIDANCE", "OUTLOOK", "ANALYST", "ANALYSTS",
        "PRICE", "TARGET", "RATING", "Q1", "Q2", "Q3", "Q4", "FY", "H1", "H2",
        "CEO", "CFO", "COO", "US", "U.S", "USA", "EPS", "AI", "ETF", "ETFS",
        "INTO", "OVER", "AMID", "ABOUT", "MORE", "THAN", "OTHER", "OTHERS",
        "BE", "BEEN", "WOULD", "INC", "CORP", "CO", "LTD", "PLC", "JUST",
        "SINCE", "FROM", "INVESTORS", "INVESTOR", "COMPANY", "BIG", "GET",
    }
)

#: Headline FURNITURE — words that describe the shape of a headline rather
#: than its subject, removed before a title is shingled for clustering.
#:
#: This is the fix for the failure the live AAPL window exposed: financial
#: wires publish templated headlines ("Apple, Microsoft And 3 Stocks To Watch
#: Heading Into Thursday", "Amazon, Apple and 3 Stocks to Watch Heading Into
#: Friday") whose shared 3-shingles — "stocks to watch", "to watch heading",
#: "watch heading into" — are pure template. Two unrelated stories that happen
#: to wear the same template measured 0.5+ Jaccard on that template alone, and
#: single-link transitivity then chained 268 of 278 articles into one "story".
#:
#: Dropping the furniture makes similarity measure what the headline SAYS. It
#: is deliberately a stopword list and not a weighting scheme: a reader can
#: check membership by eye, and a term that turns out to carry meaning for
#: some ticker is one line to remove. Note the words are removed for
#: SIMILARITY only — display titles, materiality matching and theme labels all
#: still see the original text, so "stocks to watch" can still classify an
#: article as INDUSTRY while contributing nothing to whether it is the same
#: story as another one.
TEMPLATE_STOPWORDS: frozenset[str] = frozenset(
    {
        # headline furniture proper
        "stocks", "stock", "shares", "share", "watch", "watching", "heading",
        "into", "today", "tonight", "tomorrow", "yesterday", "week", "weekly",
        "daily", "market", "markets", "report", "reports", "reported",
        "reporting", "says", "said", "say", "earlier", "update", "updates",
        "updated", "preview", "recap", "roundup", "top", "things", "thing",
        "know", "before", "after", "bell", "hours", "session", "sessions",
        "premarket", "pre", "close", "open", "opening", "closing", "movers",
        "moving", "gainers", "losers", "highlights", "wrap", "live", "latest",
        "here", "look", "looks", "amid", "ahead",
        # days of the week — a template's only moving part
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday",
        # generic function words: they cannot distinguish two stories and
        # they pad every shingle they appear in
        "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "at",
        "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those", "his", "her", "their",
        "they", "he", "she", "we", "us", "you", "will", "would", "could",
        "should", "may", "might", "can", "has", "have", "had", "do", "does",
        "did", "not", "no", "but", "or", "if", "than", "then", "so", "up",
        "down", "out", "over", "about", "more", "most", "new", "now", "one",
        "two", "three", "four", "five", "s", "t",
    }
)

#: Words dropped from theme labels — they describe the shape of a headline,
#: not the subject of one.
_THEME_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "as", "at", "but", "by", "for", "from", "how", "in",
        "is", "it", "its", "of", "on", "or", "the", "to", "up", "with", "what",
        "when", "why", "after", "before", "this", "that", "these", "those",
        "will", "would", "could", "should", "says", "said", "amid", "over",
        "into", "than", "then", "more", "most", "new", "now", "has", "have",
        "was", "were", "are", "be", "been", "his", "her", "their", "they",
        "not", "no", "you", "we", "us", "he", "she", "one", "two", "about",
        "inc", "corp", "co", "ltd", "plc", "s",
    }
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawArticle:
    """One news article as this module sees it — provider-agnostic.

    Built by the gateway seam from either a provider ``NewsArticle`` or a
    ``NewsArticleRow`` mirror; this module cannot tell which, and that is
    what keeps the audit §7.4 direction intact. Every field is DATA — the
    provider's own text and instants, verbatim.

    ``id`` is the local row id when the article came from the store and
    ``None`` when it came straight off the wire; ``source_id`` is the
    provider's article id and is the identity that matters — it is the
    dedupe key in ``news_articles`` (UNIQUE) and the stem of every id this
    module mints.
    """

    source_id: str
    title: str = ""
    description: str = ""
    publisher: str = ""
    published_at: datetime | None = None
    url: str = ""
    tickers: tuple[str, ...] = ()
    id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tickers", tuple(self.tickers or ()))
        if self.published_at is not None:
            object.__setattr__(
                self,
                "published_at",
                require_utc(self.published_at, name="published_at"),
            )

    def to_ref(self) -> dict:
        """The compact article reference embedded in clusters and evidence.

        Title and description are NOT sanitised here — they are the display
        strings, and the display path is HTML-escaped by the UI. The
        LLM-facing copies come from :func:`sanitize_for_llm` under the
        ``safe_title``/``safe_description`` keys, which is the §81 split.
        """
        safe_title = sanitize_for_llm(self.title)
        safe_description = sanitize_for_llm(self.description)
        return {
            "id": self.id,
            "source_id": self.source_id,
            "title": self.title,
            "publisher": self.publisher,
            "published_at": _iso(self.published_at),
            "url": self.url,
            "tickers": list(self.tickers),
            "safe_title": safe_title.text,
            "safe_description": safe_description.text,
            "suspicious_instruction": (
                safe_title.suspicious_instruction
                or safe_description.suspicious_instruction
            ),
        }


@dataclass(frozen=True)
class SanitizedText:
    """The §81 result of laundering one untrusted string.

    ``text`` is markup-free, control-character-free, URL-free and capped.
    ``suspicious_instruction`` says an imperative shaped like a prompt
    injection was seen; ``matched_patterns`` names the offending fragments so
    a human can read what was attempted. Neither field ever changes what the
    platform does — the text is evidence, and evidence is not obeyed.
    """

    text: str
    suspicious_instruction: bool = False
    matched_patterns: tuple[str, ...] = ()
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "suspicious_instruction": self.suspicious_instruction,
            "matched_patterns": list(self.matched_patterns),
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class MaterialityResult:
    """§24 classification of one article: a category, its weight, its proof.

    There is no sentiment here and there never will be — see the module
    docstring. ``matched_terms`` is the explainability contract: the terms
    that put the article in this category, in lexicon order.
    """

    category: str
    score: float
    matched_terms: tuple[str, ...] = ()
    category_hits: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_terms", tuple(self.matched_terms))
        object.__setattr__(
            self,
            "category_hits",
            {key: tuple(value) for key, value in dict(self.category_hits).items()},
        )

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "category_hits": {
                key: list(value) for key, value in self.category_hits.items()
            },
        }


@dataclass(frozen=True)
class EvidenceScore:
    """§25 evidence score — the product AND all five factors that made it.

    ``score = relevance × materiality × novelty × source_quality × decay``.
    The identity is checkable by hand from the payload, which is the point:
    §13 forbids an unexplainable score, and the same rule governs news.
    """

    relevance: float
    materiality: float
    novelty: float
    source_quality: float
    decay: float
    score: float
    score_no_decay: float = 0.0
    category: str = "OTHER"
    matched_terms: tuple[str, ...] = ()
    model_version: str = NEWS_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_terms", tuple(self.matched_terms))

    @property
    def material(self) -> bool:
        """Whether this is a §26 material development — decay NOT applied.

        The cut is taken on :attr:`score_no_decay`, and the distinction is the
        whole point. Decay is a RANKING weight: it answers "which of these
        should a reader look at first", and older evidence rightly sinks. It
        is not a classification: an antitrust probe filed three weeks ago is
        still a material development, and a definition that lets it stop
        being one because time passed makes the §26 "N material" count a
        function of when you happened to ask rather than of what happened.

        The live AAPL window made the difference concrete. Over 283 real
        articles the with-decay cut returned ``material: 1`` — a three-week
        window means most of its own developments are more than a half-life
        old, so the 0.25 threshold was being applied to scores that time had
        already halved. The same window with the decay factor removed from
        the cut returns eighteen, which is what a reader looking at the month
        would recognise as the month's material news.

        Ranking is unaffected: :func:`analyze_window` still orders by
        :attr:`score`, decay included, so recency still decides what sits at
        the top of the list — it just no longer decides what is on the list.
        """
        return self.score_no_decay >= MATERIAL_SCORE_THRESHOLD

    def components(self) -> dict:
        """The five factors alone — what the UI's ⓘ tooltip renders."""
        return {
            "relevance": self.relevance,
            "materiality": self.materiality,
            "novelty": self.novelty,
            "source_quality": self.source_quality,
            "decay": self.decay,
        }

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "score_no_decay": self.score_no_decay,
            "material": self.material,
            "components": self.components(),
            "category": self.category,
            "matched_terms": list(self.matched_terms),
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class ArticleCluster:
    """One story (§23): its canonical article, its members, its identity.

    ``cluster_id`` is a deterministic function of the canonical article's
    ``source_id``, so the same story keeps the same id across re-analysis and
    a stored ``news_articles.cluster_id`` still names it tomorrow.
    ``article_count`` counts the members INCLUDING near-duplicate copies that
    :func:`dedupe` folded away — a syndicated story is one development, and
    §23 is explicit that duplicated coverage must not inflate importance.
    """

    cluster_id: str
    canonical: RawArticle
    members: tuple[RawArticle, ...] = ()
    duplicates: Mapping[str, str] = field(default_factory=dict)
    link_reasons: tuple[str, ...] = ()
    score: "EvidenceScore | None" = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "duplicates", dict(self.duplicates))
        object.__setattr__(self, "link_reasons", tuple(self.link_reasons))

    def with_score(self, score: "EvidenceScore") -> "ArticleCluster":
        """A copy of this cluster carrying its canonical's §25 score.

        :func:`cluster_articles` runs before anything is scored — it only
        knows about titles, entities and instants — so the cluster it returns
        has ``score=None``. :func:`analyze_window` scores each canonical and
        re-attaches the result here, which is what lets a cluster serialise
        its own materiality instead of forcing every consumer to join
        ``clusters`` against ``evidence`` on ``cluster_id`` and invent a
        display for the rows where the join misses.
        """
        return ArticleCluster(
            cluster_id=self.cluster_id,
            canonical=self.canonical,
            members=self.members,
            duplicates=self.duplicates,
            link_reasons=self.link_reasons,
            score=score,
        )

    @property
    def materiality(self) -> str | None:
        """The canonical's §24 category, or ``None`` before scoring."""
        return None if self.score is None else self.score.category

    @property
    def materiality_score(self) -> float | None:
        """The §24 category weight, or ``None`` before scoring."""
        return None if self.score is None else self.score.materiality

    @property
    def material(self) -> bool:
        """Whether this cluster is a §26 material development."""
        return self.score is not None and self.score.material

    @property
    def article_count(self) -> int:
        """Unique articles in the story (duplicates already folded in)."""
        return len(self.members)

    @property
    def published_at(self) -> datetime | None:
        """The canonical article's instant — the story's timestamp."""
        return self.canonical.published_at

    def to_dict(self) -> dict:
        """JSON-ready mapping — every field the §59 UI renders for a story.

        The scoring block is flattened onto the cluster rather than left for
        the reader to look up in ``evidence``: the news tab draws a
        materiality badge and a score chip on every cluster row, and a payload
        that carries the numbers only for the ranked-evidence subset makes
        those badges read ``None`` for exactly the clusters a reader scrolled
        down to find. Before scoring the fields are ``None`` honestly — an
        unscored cluster is a real intermediate state, not a zero.
        """
        return {
            "cluster_id": self.cluster_id,
            "article_count": self.article_count,
            "canonical_article": self.canonical.to_ref(),
            "member_source_ids": [member.source_id for member in self.members],
            "duplicate_of": dict(self.duplicates),
            "link_reasons": list(self.link_reasons),
            "materiality": self.materiality,
            "materiality_score": self.materiality_score,
            "score": None if self.score is None else self.score.score,
            "score_no_decay": (
                None if self.score is None else self.score.score_no_decay
            ),
            "material": self.material,
            "matched_terms": (
                [] if self.score is None else list(self.score.matched_terms)
            ),
            "components": None if self.score is None else self.score.components(),
        }


@dataclass(frozen=True)
class NewsTheme:
    """§26/§59 theme — material clusters grouped by materiality category.

    The label is the category plus its two most salient terms, so the UI
    prints "REGULATION · export controls, china" rather than a bare enum, and
    ``n_developments`` is the §59 "4 material developments" count.
    """

    label: str
    category: str
    n_developments: int
    cluster_ids: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    top_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_ids", tuple(self.cluster_ids))
        object.__setattr__(self, "terms", tuple(self.terms))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "category": self.category,
            "n_developments": self.n_developments,
            "cluster_ids": list(self.cluster_ids),
            "terms": list(self.terms),
            "top_score": self.top_score,
        }


@dataclass(frozen=True)
class NewsIntelResult:
    """The §26 output of one window: counts, clusters, themes, ranked evidence.

    ``counts`` is the five-number headline the spec asks for in place of a
    bare article total. ``evidence`` is ranked by score, descending, and each
    entry carries its cluster, its five components and its article reference,
    so "View Evidence" (§27) resolves without a second query.
    """

    ticker: str
    as_of: datetime
    window_start: datetime | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    clusters: tuple[ArticleCluster, ...] = ()
    themes: tuple[NewsTheme, ...] = ()
    evidence: tuple[Mapping[str, object], ...] = ()
    excluded: Mapping[str, int] = field(default_factory=dict)
    untrusted_text_policy: Mapping[str, object] = field(default_factory=dict)
    model_version: str = NEWS_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", dict(self.counts))
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "themes", tuple(self.themes))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "excluded", dict(self.excluded))
        object.__setattr__(
            self, "untrusted_text_policy", dict(self.untrusted_text_policy)
        )

    def to_dict(self) -> dict:
        """JSON-ready mapping — the gateway hands this straight to FastAPI."""
        return {
            "ticker": self.ticker,
            "as_of": _iso(self.as_of),
            "window_start": _iso(self.window_start),
            "counts": dict(self.counts),
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "themes": [theme.to_dict() for theme in self.themes],
            "evidence": [dict(item) for item in self.evidence],
            "excluded": dict(self.excluded),
            "untrusted_text_policy": dict(self.untrusted_text_policy),
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Normalisation (§22) — one lower-cased, markup-free string for matching
# ---------------------------------------------------------------------------


def normalize(text: str | None) -> str:
    """Return ``text`` reduced to a matching form: lower-case, no markup.

    HTML tags and a handful of entities are removed, markdown links collapse
    to their anchor text, control characters go, whitespace collapses to
    single spaces and the result is lower-cased. The ORIGINAL string is never
    mutated — display always uses :attr:`RawArticle.title`; this is purely
    the key similarity is measured on.
    """
    if not text:
        return ""
    value = _MARKDOWN_LINK_RE.sub(r"\1", str(text))
    value = _TAG_RE.sub(" ", value)
    for entity, char in _ENTITIES.items():
        value = value.replace(entity, char)
    value = _ENTITY_RE.sub(" ", value)
    value = _CONTROL_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip().lower()


def tokens(text: str | None) -> tuple[str, ...]:
    """Word tokens of the normalised text, in order, duplicates kept.

    Order and duplicates matter because :func:`shingles` runs over this.
    """
    return tuple(_WORD_RE.findall(normalize(text)))


def shingles(
    source: Sequence[str] | str | None, k: int = DEFAULT_SHINGLE_K
) -> frozenset[str]:
    """The set of ``k``-word shingles of a token sequence (or of raw text).

    A text with fewer than ``k`` tokens yields a single shingle of the whole
    thing, so a three-word headline still compares against itself at 1.0
    rather than collapsing to the empty set (which would compare at 0.0
    against everything, itself included).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    words: tuple[str, ...]
    if source is None:
        words = ()
    elif isinstance(source, str):
        words = tokens(source)
    else:
        words = tuple(source)
    if not words:
        return frozenset()
    if len(words) <= k:
        return frozenset({" ".join(words)})
    return frozenset(
        " ".join(words[index : index + k]) for index in range(len(words) - k + 1)
    )


def story_tokens(text: str | None) -> tuple[str, ...]:
    """Tokens of ``text`` with headline template furniture removed (§23).

    The similarity view of a headline: what the story SAYS, with the words
    that describe the shape of a headline taken out. "Apple, Microsoft And 3
    Stocks To Watch Heading Into Thursday" reduces to ``("apple",
    "microsoft", "3")`` — which shares nothing with Friday's version except
    the tickers it happens to list, exactly as it should.

    Purely a matching key: :func:`tokens` (and the display title) are
    untouched, so materiality, themes and the UI still read the full headline.
    """
    return tuple(
        word for word in tokens(text)
        if word not in TEMPLATE_STOPWORDS
    )


def story_shingles(
    text: str | None, k: int = DEFAULT_SHINGLE_K
) -> frozenset[str]:
    """:func:`shingles` of :func:`story_tokens` — the clustering key.

    Template removal shortens headlines, so ``k`` is applied to the SURVIVING
    words: a six-word headline that loses three of them compares on the three
    that carried the story rather than on nothing at all (see :func:`shingles`
    for the short-text rule).
    """
    return shingles(story_tokens(text), k=k)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """|A ∩ B| / |A ∪ B|, with two empty sets scoring 0.0, not 1.0.

    Two articles with unparseable titles are not "identical"; they are
    unknown, and 0.0 keeps them out of each other's clusters.
    """
    first = frozenset(left)
    second = frozenset(right)
    if not first or not second:
        return 0.0
    union = len(first | second)
    if union == 0:
        return 0.0
    return len(first & second) / union


def salient_entities(
    article: "RawArticle", *, exclude: Iterable[str] = ()
) -> frozenset[str]:
    """Capitalised multi-character tokens and tickers naming the story's subjects.

    Used by the §23 entity clustering rule. Single letters, common
    capitalised connectives and headline furniture ("Stocks", "Report") are
    dropped, because an entity every headline shares links every headline.
    The article's provider ``tickers`` are salient too — a provider tag is
    the strongest available statement about who a story is about.

    ``exclude`` removes entities that carry no discriminating power *in this
    particular window*. :func:`analyze_window` passes the subject ticker and
    its company aliases: every article in a single-ticker window names AAPL
    and Apple, so counting those two as "shared entities" would satisfy the
    two-entity rule for every pair and fuse an antitrust probe into an
    earnings beat. What must be shared is what makes the stories the SAME
    story, not what made them both eligible for the window.
    """
    blocked = {str(item).strip().upper() for item in exclude if str(item).strip()}
    tags = tuple(
        ticker.strip().upper() for ticker in article.tickers if ticker.strip()
    )
    # A provider tag list is a statement about who a story is about — until it
    # gets long, at which point it is a statement that the story is about a
    # BASKET. Benzinga tags its market wraps with every symbol they mention,
    # so "Dow Records Worst Session Since April 2025" and "Apple, Microsoft
    # And 3 Stocks To Watch" share MSFT, NVDA, AMZN and META without sharing a
    # development. Beyond the cap the tags are dropped entirely rather than
    # trimmed: any subset of a basket is still a basket, and there is no
    # principled way to pick which few of forty-three tags were the subject.
    found: set[str] = set(tags) if len(tags) <= MAX_TAGS_AS_ENTITIES else set()
    for chunk in (article.title or "", article.description or ""):
        text = _TAG_RE.sub(" ", str(chunk))
        for match in _ENTITY_TOKEN_RE.findall(text):
            token = match.strip(".-'").upper()
            if len(token) < 2 or token in _ENTITY_STOPWORDS:
                continue
            found.add(token)
    return frozenset(found - blocked)


# ---------------------------------------------------------------------------
# §81 sanitisation — strip, cap, flag; never obey
# ---------------------------------------------------------------------------


def sanitize_for_llm(
    text: str | None, max_chars: int = SANITIZE_MAX_CHARS
) -> SanitizedText:
    """Launder one untrusted article string for inclusion in a prompt (§81).

    Removes markup, control characters and URLs (a URL in evidence text is
    an exfiltration invitation and adds nothing a reader needs — the real
    link travels as a structured field), collapses whitespace, caps the
    length at ``max_chars`` on a word boundary, and FLAGS — never silently
    swallows — text shaped like an instruction. The flag is diagnostic: this
    platform never treats retrieved text as instructions, so nothing
    branches on it.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1; got {max_chars}")
    raw = "" if text is None else str(text)
    matched: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        for hit in pattern.findall(raw):
            fragment = hit if isinstance(hit, str) else " ".join(part for part in hit if part)
            fragment = _WHITESPACE_RE.sub(" ", fragment).strip()
            if fragment and fragment not in matched:
                matched.append(fragment)
    value = _MARKDOWN_LINK_RE.sub(r"\1", raw)
    value = _TAG_RE.sub(" ", value)
    for entity, char in _ENTITIES.items():
        value = value.replace(entity, char)
    value = _ENTITY_RE.sub(" ", value)
    # RE-STRIP after decoding. Decoding runs second on purpose (an entity may
    # hide inside what looked like a tag), but it can also RECONSTITUTE markup:
    # "&lt;/untrusted_web_research&gt;" survives the first pass as plain text
    # and decodes into a live closing tag afterwards. Left there, third-party
    # text could close the prompt's own trust fence from the inside and speak
    # in the unfenced region (§81). Repeated to a fixed point so a nested
    # encoding ("&amp;lt;") cannot peel one layer per pass and escape.
    for _ in range(_MAX_ENTITY_DECODE_PASSES):
        stripped = _TAG_RE.sub(" ", value)
        if stripped == value:
            break
        value = stripped
        for entity, char in _ENTITIES.items():
            value = value.replace(entity, char)
        value = _ENTITY_RE.sub(" ", value)
    value = _CONTROL_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    truncated = False
    if len(value) > max_chars:
        clipped = value[:max_chars]
        cut = clipped.rfind(" ")
        if cut > max_chars // 2:
            clipped = clipped[:cut]
        value = clipped.rstrip()
        truncated = True
    return SanitizedText(
        text=value,
        suspicious_instruction=bool(matched),
        matched_patterns=tuple(matched),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# §23 deduplication
# ---------------------------------------------------------------------------


def dedupe(
    articles: Sequence["RawArticle"],
    *,
    threshold: float = DEFAULT_DEDUPE_JACCARD,
    k: int = DEFAULT_SHINGLE_K,
) -> tuple[tuple["RawArticle", ...], dict[str, str]]:
    """Fold near-duplicate copies of one story onto their earliest printing.

    Two articles are near-duplicates when their normalised titles are
    identical, or when the Jaccard of their title 3-shingles is at or above
    ``threshold`` (§23). The EARLIEST publication survives as canonical —
    the first outlet to print a story is the one that broke it, and keeping
    the earliest keeps the story's timestamp honest for time decay.

    Returns ``(unique_articles, duplicate_of)`` where ``duplicate_of`` maps a
    dropped article's ``source_id`` to the surviving canonical's.
    """
    ordered = _sorted_articles(articles)
    survivors: list[RawArticle] = []
    survivor_shingles: list[frozenset[str]] = []
    survivor_titles: list[str] = []
    duplicate_of: dict[str, str] = {}
    for article in ordered:
        title = normalize(article.title)
        grams = shingles(tokens(article.title), k=k)
        canonical: RawArticle | None = None
        for index, existing in enumerate(survivors):
            if title and title == survivor_titles[index]:
                canonical = existing
                break
            if grams and jaccard(grams, survivor_shingles[index]) >= threshold:
                canonical = existing
                break
        if canonical is None:
            survivors.append(article)
            survivor_shingles.append(grams)
            survivor_titles.append(title)
        else:
            duplicate_of[article.source_id] = canonical.source_id
    return tuple(survivors), duplicate_of


# ---------------------------------------------------------------------------
# §23 story clustering
# ---------------------------------------------------------------------------


def cluster_articles(
    articles: Sequence["RawArticle"],
    *,
    title_threshold: float = DEFAULT_CLUSTER_JACCARD,
    entity_window: timedelta = DEFAULT_CLUSTER_ENTITY_WINDOW,
    title_window: timedelta = DEFAULT_CLUSTER_TITLE_WINDOW,
    min_shared_entities: int = CLUSTER_ENTITY_MIN_SHARED,
    k: int = DEFAULT_SHINGLE_K,
    duplicate_of: Mapping[str, str] | None = None,
    exclude_entities: Iterable[str] = (),
    max_share: float = CLUSTER_MAX_SHARE,
) -> tuple["ArticleCluster", ...]:
    """LEADER clustering of a window's unique articles into stories (§23).

    Articles are walked OLDEST FIRST — the order the developments actually
    happened in — and each one either joins an existing story or opens a new
    one. It joins the story whose LEADER (the first article of that cluster,
    the one that broke the development) it matches best:

    * **title rule** — story-shingle Jaccard against the leader's title at or
      above ``title_threshold``, AND published within ``title_window`` of it;
    * **entity rule** — at least ``min_shared_entities`` salient entities in
      common with the leader, excluding the subject ticker and its aliases,
      AND published within ``entity_window`` of it.

    Ties go to the highest title similarity, then to the OLDEST leader, so the
    assignment does not depend on dictionary or input order.

    Why not single-link. Single-link says A joins the cluster if it touches
    ANY member, which makes membership transitive: A~B and B~C put A and C in
    one story even when A and C share nothing. On real wire copy that is
    catastrophic. The live AAPL window (283 articles, 2026-07-29..08-19) has
    dozens of templated headlines and a single subject entity every article
    names, and transitivity chained 268 of 278 unique articles into one
    "story" canonically titled "Apple, Microsoft And 3 Stocks To Watch Heading
    Into Thursday". Every individual link looked plausible; the chain of them
    did not. Leader clustering breaks the chain by construction — an article
    must resemble the story's OWN defining article, not merely something that
    resembled something that resembled it — so a mistake costs one misplaced
    article instead of the whole window.

    Three defences reinforce it, each aimed at a failure the live data showed:
    :func:`story_shingles` strips headline template furniture before
    measuring, so "Stocks To Watch Heading Into Thursday" contributes nothing;
    the title rule carries a time bound like the entity rule already did, so a
    recurring monthly phrasing is not one story; and ``max_share`` caps a
    cluster at that fraction of the window's unique articles, after which an
    article starts a new cluster no matter how well it matches. The cap is a
    circuit-breaker — with the first two defences in place it does not fire on
    the live window — and its job is to keep a pathological input readable
    rather than to shape a healthy one.

    The canonical article is the earliest of the highest-source-quality
    members — quality first, then time — so a Reuters print an hour after a
    blog aggregation is the one the cluster is named and dated by. The LEADER
    (matching target) and the CANONICAL (display identity) are deliberately
    different jobs: the leader is fixed at first sight so membership cannot
    drift as a cluster grows, while the canonical may be re-chosen when a
    better-sourced member arrives.
    """
    ordered = _sorted_articles(articles)
    count = len(ordered)
    if count == 0:
        return ()

    grams = [story_shingles(item.title, k=k) for item in ordered]
    blocked = tuple(exclude_entities)
    entities = [salient_entities(item, exclude=blocked) for item in ordered]

    # The cap is a share of the window, so it needs the window's size — which
    # is known here and only here. The floor of CLUSTER_MIN_CAP is essential
    # rather than defensive: 40% of a two-article window is a cap of ZERO, so
    # a naive share would refuse to cluster the very case clustering exists
    # for — two syndications of one story. The circuit-breaker is aimed at
    # windows large enough for chaining to be possible at all, and below that
    # size it must not fire.
    limit = count
    if 0.0 < max_share < 1.0:
        limit = max(CLUSTER_MIN_CAP, int(math.floor(max_share * count)))

    members: list[list[int]] = []          # cluster index -> member positions
    leaders: list[int] = []                # cluster index -> leader position
    reasons: list[list[str]] = []          # cluster index -> link explanations
    assigned: list[int] = []               # article position -> cluster index

    for position in range(count):
        best: tuple[float, int] | None = None   # (-similarity, leader position)
        best_cluster = -1
        best_reason = ""
        for cluster_index, leader in enumerate(leaders):
            if len(members[cluster_index]) >= limit:
                # Circuit-breaker: this story has already claimed as much of
                # the window as any real development plausibly could.
                continue
            similarity = jaccard(grams[position], grams[leader])
            gap = _gap(ordered[position], ordered[leader])
            reason = ""
            if (
                similarity >= title_threshold
                and gap is not None
                and gap <= title_window
            ):
                reason = (
                    f"title_jaccard={similarity:.2f}:"
                    f"{ordered[leader].source_id}~{ordered[position].source_id}"
                )
            else:
                shared = entities[position] & entities[leader]
                if (
                    len(shared) >= min_shared_entities
                    and gap is not None
                    and gap <= entity_window
                ):
                    reason = (
                        "shared_entities="
                        + "|".join(sorted(shared)[:4])
                        + f":{ordered[leader].source_id}~{ordered[position].source_id}"
                    )
            if not reason:
                continue
            candidate = (-similarity, leader)
            if best is None or candidate < best:
                best = candidate
                best_cluster = cluster_index
                best_reason = reason
        if best_cluster < 0:
            leaders.append(position)
            members.append([position])
            reasons.append([])
            assigned.append(len(leaders) - 1)
        else:
            members[best_cluster].append(position)
            reasons[best_cluster].append(best_reason)
            assigned.append(best_cluster)

    duplicates = dict(duplicate_of or {})
    clusters: list[ArticleCluster] = []
    for cluster_index, positions in enumerate(members):
        group = [ordered[index] for index in positions]
        canonical = _canonical_of(group)
        member_ids = {member.source_id for member in group}
        cluster_duplicates = {
            duplicate: parent_id
            for duplicate, parent_id in duplicates.items()
            if parent_id in member_ids
        }
        clusters.append(
            ArticleCluster(
                cluster_id=cluster_id_for(canonical.source_id),
                canonical=canonical,
                members=tuple(group),
                duplicates=cluster_duplicates,
                link_reasons=tuple(sorted(set(reasons[cluster_index]))),
            )
        )
    clusters.sort(key=lambda cluster: _sort_key(cluster.canonical))
    return tuple(clusters)


def cluster_id_for(source_id: str) -> str:
    """``"c:" + sha1(canonical source_id)[:12]`` — deterministic, stable.

    A content-derived id rather than a sequence number: the same story
    analysed in two processes, or re-analysed a day later, gets the same id,
    which is what makes the persisted ``news_articles.cluster_id`` column
    meaningful across runs.
    """
    digest = hashlib.sha1(str(source_id).encode("utf-8")).hexdigest()
    return "c:" + digest[:12]


# ---------------------------------------------------------------------------
# §22 relevance / §24 materiality / §22 novelty, source quality, decay
# ---------------------------------------------------------------------------


def ticker_relevance(article: "RawArticle", ticker: str) -> float:
    """§22 ternary relevance of one article to one ticker.

    1.0 when the provider tagged the article with the ticker or the headline
    names it; 0.7 when only the description mentions it; 0.0 otherwise — and
    a 0.0 article is EXCLUDED by :func:`analyze_window` rather than carried
    at a low weight, because a product of five factors with a zero in it is
    not "weak evidence", it is not evidence.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return 0.0
    if any(symbol == tag.strip().upper() for tag in article.tickers):
        return RELEVANCE_TITLE_OR_TAGGED
    pattern = re.compile(r"(?<!\w)" + re.escape(symbol.lower()) + r"(?!\w)")
    if pattern.search(normalize(article.title)):
        return RELEVANCE_TITLE_OR_TAGGED
    if pattern.search(normalize(article.description)):
        return RELEVANCE_DESCRIPTION_ONLY
    return 0.0


def score_materiality(text: str | None) -> "MaterialityResult":
    """Classify free text into a §24 category with the terms that chose it.

    Every category's lexicon is matched against the normalised text; the
    winner is the category with the most distinct hits, ties broken by the
    higher category weight and then by :data:`CATEGORY_ORDER` so the result
    is deterministic. Text that matches nothing is ``OTHER`` at 0.1 — a real
    classification, not an error.

    Materiality is not sentiment (§24): the lexicon contains no polarity
    words and no direction is inferred from a hit.
    """
    normalized = normalize(text)
    hits: dict[str, tuple[str, ...]] = {}
    if normalized:
        for category, patterns in _LEXICON_PATTERNS.items():
            matched = tuple(
                term for term, pattern in patterns if pattern.search(normalized)
            )
            if matched:
                hits[category] = matched
    if not hits:
        return MaterialityResult(
            category="OTHER", score=CATEGORY_WEIGHTS["OTHER"], matched_terms=()
        )
    winner = min(
        hits,
        key=lambda category: (
            -len(hits[category]),
            -CATEGORY_WEIGHTS[category],
            CATEGORY_ORDER.index(category),
        ),
    )
    return MaterialityResult(
        category=winner,
        score=CATEGORY_WEIGHTS[winner],
        matched_terms=hits[winner],
        category_hits=hits,
    )


def materiality_of(article: "RawArticle") -> "MaterialityResult":
    """§24 materiality of an article — title weighted ahead of description.

    The headline is what the publisher chose to say the story is about, so
    it classifies first; the description only decides when the headline
    matched nothing.
    """
    from_title = score_materiality(article.title)
    if from_title.category != "OTHER":
        return from_title
    from_body = score_materiality(article.description)
    if from_body.category != "OTHER":
        return from_body
    return from_title


def source_quality(publisher: str | None) -> float:
    """§22 publisher weight from the extensible substring table.

    Matched on lower-cased substrings, longest key first, so
    "The Wall Street Journal" cannot be captured by a shorter entry. An
    unrecognised publisher gets :data:`UNKNOWN_SOURCE_QUALITY` — missing
    information about a source is not evidence against it.
    """
    name = normalize(publisher)
    if not name:
        return UNKNOWN_SOURCE_QUALITY
    for key in _SOURCE_QUALITY_KEYS:
        if _substring_hit(name, key):
            return SOURCE_QUALITY[key]
    return UNKNOWN_SOURCE_QUALITY


def novelty_of(
    title: str | None,
    earlier_titles: Sequence[str],
    *,
    k: int = DEFAULT_SHINGLE_K,
) -> float:
    """§22 novelty: 1 − the closest match against EARLIER canonical titles.

    The first cluster in a window is 1.0 by construction — there is nothing
    before it to have said this already. "Earlier" is strictly by publication
    instant: a story is not made un-novel by coverage that came after it, and
    measuring against later titles would let the as-of boundary leak
    backwards.
    """
    grams = shingles(tokens(title), k=k)
    if not grams or not earlier_titles:
        return 1.0
    closest = max(
        (jaccard(grams, shingles(tokens(other), k=k)) for other in earlier_titles),
        default=0.0,
    )
    return _clamp(1.0 - closest)


def time_decay(
    published_at: datetime | None,
    as_of: datetime,
    *,
    half_life_days: float = HALF_LIFE_DAYS,
    floor: float = DECAY_FLOOR,
) -> float:
    """§22 recency weight ``exp(−ln2 · age_days / half_life)``, floored.

    A 14-day-old article is worth half a fresh one and a 28-day-old a
    quarter, until the floor catches it. An article dated AFTER ``as_of`` is
    not extrapolated to a bonus — it clamps to 1.0, though
    :func:`analyze_window` has already excluded it under §96. An unknown
    instant returns the floor rather than a guess.
    """
    reference = require_utc(as_of, name="as_of")
    if published_at is None:
        return floor
    stamp = require_utc(published_at, name="published_at")
    age_days = (reference - stamp).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return floor
    weight = math.exp(-math.log(2.0) * age_days / half_life_days)
    return max(floor, min(1.0, weight))


def score_evidence(
    *,
    relevance: float,
    materiality: "MaterialityResult" | float,
    novelty: float,
    source_quality_weight: float,
    decay: float,
) -> "EvidenceScore":
    """§25 product of the five factors, returned with all five (explainable).

    ``score`` is the full product and ``score_no_decay`` the same product
    without the recency weight, so a caller can rank by one and classify by
    the other without recomputing anything or knowing the formula.
    """
    if isinstance(materiality, MaterialityResult):
        weight = materiality.score
        category = materiality.category
        terms = materiality.matched_terms
    else:
        weight = float(materiality)
        category = "OTHER"
        terms = ()
    factors = (
        _clamp(relevance),
        _clamp(weight),
        _clamp(novelty),
        _clamp(source_quality_weight),
        _clamp(decay),
    )
    # The four content factors first, then decay on top. Keeping them apart
    # is what lets `material` ask a question about the DEVELOPMENT while
    # `score` answers a question about the reading order (see
    # :attr:`EvidenceScore.material`).
    without_decay = 1.0
    for factor in factors[:4]:
        without_decay *= factor
    product = without_decay * factors[4]
    return EvidenceScore(
        relevance=factors[0],
        materiality=factors[1],
        novelty=factors[2],
        source_quality=factors[3],
        decay=factors[4],
        score=product,
        score_no_decay=without_decay,
        category=category,
        matched_terms=terms,
    )


# ---------------------------------------------------------------------------
# §26/§59 themes
# ---------------------------------------------------------------------------


def themes_from_clusters(
    entries: Sequence[Mapping[str, object]], *, max_terms: int = 2
) -> tuple["NewsTheme", ...]:
    """Group material clusters by category into the §59 KEY THEMES list.

    Each entry is ``{"cluster": ArticleCluster, "score": EvidenceScore}``.
    Only material entries are themed — an immaterial development is not a
    theme, it is noise the §26 counts already reported. The label is the
    category plus the two most frequent salient terms across the group's
    canonical headlines, which is what turns "REGULATION" into
    "REGULATION · export controls, china".
    """
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for entry in entries:
        score = entry.get("score")
        if not isinstance(score, EvidenceScore) or not score.material:
            continue
        grouped.setdefault(score.category, []).append(entry)

    themes: list[NewsTheme] = []
    for category, members in grouped.items():
        counts: dict[str, int] = {}
        order: dict[str, int] = {}
        for position, entry in enumerate(members):
            cluster = entry["cluster"]
            assert isinstance(cluster, ArticleCluster)
            seen: set[str] = set()
            for word in tokens(cluster.canonical.title):
                if word in _THEME_STOPWORDS or len(word) < 3 or word.isdigit():
                    continue
                if word in seen:
                    continue
                seen.add(word)
                counts[word] = counts.get(word, 0) + 1
                order.setdefault(word, position)
        top = sorted(
            counts, key=lambda word: (-counts[word], order[word], word)
        )[:max_terms]
        top_score = max(
            float(entry["score"].score)  # type: ignore[union-attr]
            for entry in members
        )
        label = category if not top else f"{category} · {', '.join(top)}"
        themes.append(
            NewsTheme(
                label=label,
                category=category,
                n_developments=len(members),
                cluster_ids=tuple(
                    entry["cluster"].cluster_id  # type: ignore[union-attr]
                    for entry in members
                ),
                terms=tuple(top),
                top_score=top_score,
            )
        )
    themes.sort(
        key=lambda theme: (
            -theme.n_developments,
            -theme.top_score,
            CATEGORY_ORDER.index(theme.category),
        )
    )
    return tuple(themes)


# ---------------------------------------------------------------------------
# The pipeline (§22)
# ---------------------------------------------------------------------------


def analyze_window(
    articles: Sequence["RawArticle"],
    *,
    ticker: str,
    as_of: datetime,
    window_start: datetime | None = None,
    material_threshold: float = MATERIAL_SCORE_THRESHOLD,
) -> "NewsIntelResult":
    """Run the whole §22 pipeline over one window and return the §26 output.

    Stages, in order: **as-of gate** (§96 — anything published after
    ``as_of`` is dropped before anything else runs, so it cannot influence a
    count, a cluster, a novelty measurement or a score), window-start gate,
    relevance gate (§22 — a 0.0-relevance article is excluded), dedupe (§23),
    clustering (§23), then per-cluster materiality, novelty against earlier
    clusters, source quality, decay and the §25 product. Themes (§26/§59)
    group the material clusters.

    The counts are the §26 headline: ``raw`` is what came in (after the as-of
    and window gates — a count of articles that were not knowable is not a
    fact about this window), ``unique`` is post-dedupe, ``clusters`` is
    stories, ``material`` is clusters at or above the cut, ``themes`` is the
    §59 grouping.
    """
    reference = require_utc(as_of, name="as_of")
    start = (
        require_utc(window_start, name="window_start")
        if window_start is not None
        else None
    )

    excluded = {"after_as_of": 0, "before_window_start": 0, "not_relevant": 0,
                "no_published_at": 0}
    in_window: list[RawArticle] = []
    relevance_by_id: dict[str, float] = {}
    for article in articles:
        stamp = article.published_at
        if stamp is None:
            excluded["no_published_at"] += 1
            continue
        if stamp > reference:
            excluded["after_as_of"] += 1
            continue
        if start is not None and stamp < start:
            excluded["before_window_start"] += 1
            continue
        relevance = ticker_relevance(article, ticker)
        if relevance <= 0.0:
            excluded["not_relevant"] += 1
            continue
        relevance_by_id[article.source_id] = relevance
        in_window.append(article)

    unique, duplicate_of = dedupe(in_window)
    # The subject ticker is shared by construction — every article survived
    # the relevance gate BECAUSE it names the ticker — so it carries no
    # information about whether two of them are the same story (§23). Nor
    # does the company's NAME, which is the same statement in words: a
    # single-ticker AAPL window is wall-to-wall "Apple", and counting it as a
    # shared entity satisfies the two-entity rule for nearly every pair. The
    # ubiquity is measured from this window rather than read off a hard-coded
    # alias table, so it works for any ticker and needs no maintenance.
    ubiquitous = _ubiquitous_entities(unique)
    clusters = cluster_articles(
        unique,
        duplicate_of=duplicate_of,
        exclude_entities=(((ticker or "").strip().upper(),) + ubiquitous),
    )

    entries: list[dict[str, object]] = []
    earlier_titles: list[str] = []
    for cluster in clusters:
        canonical = cluster.canonical
        relevance = max(
            (
                relevance_by_id.get(member.source_id, 0.0)
                for member in cluster.members
            ),
            default=0.0,
        )
        materiality = materiality_of(canonical)
        novelty = novelty_of(canonical.title, earlier_titles)
        quality = source_quality(canonical.publisher)
        decay = time_decay(canonical.published_at, reference)
        score = score_evidence(
            relevance=relevance,
            materiality=materiality,
            novelty=novelty,
            source_quality_weight=quality,
            decay=decay,
        )
        entries.append({"cluster": cluster.with_score(score), "score": score})
        earlier_titles.append(canonical.title or "")

    # Re-bind the returned clusters to the SCORED copies, so ``result.clusters``
    # and ``result.evidence`` describe the same objects and cannot drift.
    clusters = tuple(
        entry["cluster"] for entry in entries  # type: ignore[misc]
    )

    # The cut is taken WITHOUT decay (see :attr:`EvidenceScore.material`):
    # "material" is a property of the development, and an old development is
    # still a development (§26). Ranking below still uses the full score.
    material_entries = [
        entry
        for entry in entries
        if float(entry["score"].score_no_decay) >= material_threshold  # type: ignore[union-attr]
    ]
    themes = themes_from_clusters(material_entries)

    ranked = sorted(
        entries,
        key=lambda entry: (
            -float(entry["score"].score),  # type: ignore[union-attr]
            _sort_key(entry["cluster"].canonical),  # type: ignore[union-attr]
        ),
    )
    evidence: list[dict] = []
    for entry in ranked:
        cluster = entry["cluster"]
        score = entry["score"]
        assert isinstance(cluster, ArticleCluster)
        assert isinstance(score, EvidenceScore)
        evidence.append(
            {
                "evidence_id": "news:" + cluster.canonical.source_id,
                "cluster_id": cluster.cluster_id,
                "score": score.score,
                "score_no_decay": score.score_no_decay,
                "material": score.score_no_decay >= material_threshold,
                "components": score.components(),
                # ``category`` IS the materiality category — the cluster block
                # spells it as ``materiality`` because that is the column name
                # it persists to, and repeating both spellings here would be
                # two keys that can disagree. One name per fact.
                "category": score.category,
                "matched_terms": list(score.matched_terms),
                "link_reasons": list(cluster.link_reasons),
                "article_count": cluster.article_count,
                "article": cluster.canonical.to_ref(),
                "model_version": NEWS_MODEL_VERSION,
            }
        )

    counts = {
        "raw": len(in_window),
        "unique": len(unique),
        "clusters": len(clusters),
        "material": len(material_entries),
        "themes": len(themes),
    }

    suspicious = sum(
        1 for item in evidence if item["article"]["suspicious_instruction"]
    )
    policy = {
        "sanitized": True,
        "max_chars": SANITIZE_MAX_CHARS,
        "rule": (
            "Article text is untrusted evidence (§81): markup, control "
            "characters and URLs are stripped, length is capped, and "
            "instruction-shaped lines are flagged, never followed."
        ),
        "suspicious_articles": suspicious,
    }

    return NewsIntelResult(
        ticker=(ticker or "").strip().upper(),
        as_of=reference,
        window_start=start,
        counts=counts,
        clusters=clusters,
        themes=themes,
        evidence=tuple(evidence),
        excluded=excluded,
        untrusted_text_policy=policy,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Finite value pinned into ``[low, high]``; NaN/inf become ``low``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def _iso(value: datetime | None) -> str | None:
    """UTC ISO-8601 string, or ``None`` — never an invented instant."""
    if value is None:
        return None
    return require_utc(value, name="value").isoformat()


_MAX_UTC = datetime.max.replace(tzinfo=UTC)


def _sort_key(article: "RawArticle") -> tuple[datetime, str]:
    """Total order: publication instant, then ``source_id`` as the tiebreak.

    Undated articles sort last rather than first — an unknown instant must
    not win a canonical-article contest against a known one.
    """
    return (article.published_at or _MAX_UTC, article.source_id)


def _sorted_articles(
    articles: Sequence["RawArticle"],
) -> tuple["RawArticle", ...]:
    """Oldest-first, deterministic even when instants tie."""
    return tuple(sorted(articles, key=_sort_key))


#: An entity naming this share or more of a window's articles is describing
#: the window, not a story inside it. A third is enough: a genuine shared
#: subject shows up in a handful of articles about one development, while
#: anything present in a third of a MONTH of coverage is the beat itself.
UBIQUITOUS_ENTITY_SHARE = 0.33


def _ubiquitous_entities(
    articles: Sequence["RawArticle"],
) -> tuple[str, ...]:
    """Entities so common in this window that sharing one says nothing (§23).

    The subject company's name and its aliases appear in most articles of a
    single-ticker window — they are WHY the articles are in the window.
    Treating them as shared subjects makes the entity rule fire on nearly
    every pair, which is how an antitrust probe ends up in the same "story" as
    an earnings beat. On the live AAPL window "APPLE" appeared in 144 of 283
    articles and "AAPL" in all 283; the pair {APPLE, AMAZON} alone was enough
    to satisfy the two-entity rule across a dozen unrelated market wraps.

    Measured, not listed. There is no alias table to maintain and no
    per-ticker tuning: whatever a window is saturated with loses its power to
    link, whether that is "APPLE", "AAPL", "COOK" or a name no table knows.
    The frequency is computed over the window's UNIQUE articles, so a
    heavily-syndicated story cannot vote its own entities into the block list.
    """
    total = len(articles)
    if total < 3:
        return ()
    counts: dict[str, int] = {}
    for item in articles:
        for entity in salient_entities(item):
            counts[entity] = counts.get(entity, 0) + 1
    cutoff = UBIQUITOUS_ENTITY_SHARE * total
    return tuple(sorted(name for name, seen in counts.items() if seen >= cutoff))


def _gap(left: "RawArticle", right: "RawArticle") -> timedelta | None:
    """Absolute publication gap, or ``None`` when either instant is unknown.

    ``None`` fails every temporal bound rather than passing it: an article
    with no instant cannot be shown to be close to anything, and guessing
    "close enough" is how undated wire copy joins whichever story it is
    compared against first.
    """
    if left.published_at is None or right.published_at is None:
        return None
    return abs(left.published_at - right.published_at)


def _canonical_of(members: Sequence["RawArticle"]) -> "RawArticle":
    """Earliest article among the highest-quality publishers in the group."""
    best_quality = max(source_quality(member.publisher) for member in members)
    candidates = [
        member
        for member in members
        if source_quality(member.publisher) == best_quality
    ]
    return min(candidates, key=_sort_key)


def _substring_hit(haystack: str, needle: str) -> bool:
    """Substring match that still respects word edges for short keys.

    "ir" and "sec" are legitimate source keys (investor relations, the SEC)
    but they hide inside "first" and "second", so keys of three characters or
    fewer are matched on word boundaries while longer names match anywhere —
    which is what lets "Thomson Reuters Business" reach ``reuters``.
    """
    if len(needle) <= 3:
        return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None
    return needle in haystack
