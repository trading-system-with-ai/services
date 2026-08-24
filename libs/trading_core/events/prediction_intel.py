"""Pure prediction-market intelligence: historical features AND the
event↔market matcher (Catalyst research upgrade; plan §3, Phases 3-4;
LOOPs 4-5).

PURE COMPUTATION — NO I/O (the news_intel/web_research discipline, enforced
by the AST test): the gateway seam fetches through libs.prediction_markets
and hands the dated points here.

A MARKET PRICE IS NOT A CLEAN PROBABILITY. Everything this module computes
is a statement about PRICING — "the contract's market-implied probability
moved from X to Y" — never about the underlying outcome. Field names and
docstrings say so, and the interpretation layer (bundle/LLM/UI) carries the
same language (plan §3 interpretation rule).

NO INVENTED INTERPOLATION (plan §3, explicit): every anchor comparison uses
the LAST OBSERVED price AT OR BEFORE the anchor instant. When no observation
exists at/before an anchor, that delta is ``None`` — a market that started
trading three days ago has no honest 7-day change, and a gap in the series
stays a gap.

POINT-IN-TIME: the feature builder filters points to ``ts <= as_of`` FIRST,
so a historical replay computes exactly what was knowable then — later
observations can never contaminate an earlier as-of view (§44 rule 18).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from libs.trading_core.events.news_intel import tokens
from libs.trading_core.events.web_research import event_subject, research_profile
from libs.trading_core.models.enums import EventType

#: Version stamp for this layer's feature definitions. Bump when an anchor
#: rule or the trend threshold changes — the numbers' meaning changes.
PREDICTION_INTEL_MODEL_VERSION = "prediction-intel-v1"

#: Trend vocabulary (defined calculation, never an LLM judgement — plan
#: Phase 23): the full-series move (last minus first observed price) beyond
#: +/- TREND_FLAT_THRESHOLD is RISING/FALLING; inside it, FLAT.
TREND_RISING = "RISING"
TREND_FALLING = "FALLING"
TREND_FLAT = "FLAT"
TREND_FLAT_THRESHOLD = 0.02

#: Named anchor offsets for the short-horizon deltas.
_CHANGE_ANCHORS: tuple[tuple[str, timedelta], ...] = (
    ("change_1h", timedelta(hours=1)),
    ("change_1d", timedelta(days=1)),
    ("change_7d", timedelta(days=7)),
)


def _as_aware_utc(value: datetime) -> datetime:
    """Naive instants are treated as UTC (the platform's provider-payload
    convention) — never compared naive-vs-aware."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MarketHistoryFeatures:
    """Deterministic pricing features for ONE outcome's observed series.

    ``current_price`` is the last observed contract price at/before as_of —
    the market-implied probability the bundle quotes. Every ``change_*`` is
    ``current_price`` minus the last observed price at/before that anchor,
    or ``None`` when the series has no observation there (no interpolation,
    ever). ``observation_count``/``history_start``/``history_end`` state how
    much data stands behind the numbers, so a two-point series can never
    masquerade as a deep one (plan §3: preserve them).
    """

    current_price: float
    change_1h: float | None
    change_1d: float | None
    change_7d: float | None
    change_since_previous_event: float | None
    change_since_window_start: float | None
    recent_high: float
    recent_low: float
    price_range: float
    trend: str
    observation_count: int
    history_start: datetime
    history_end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": PREDICTION_INTEL_MODEL_VERSION,
            "current_price": self.current_price,
            "change_1h": self.change_1h,
            "change_1d": self.change_1d,
            "change_7d": self.change_7d,
            "change_since_previous_event": self.change_since_previous_event,
            "change_since_window_start": self.change_since_window_start,
            "recent_high": self.recent_high,
            "recent_low": self.recent_low,
            "price_range": self.price_range,
            "trend": self.trend,
            "observation_count": self.observation_count,
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
        }


def _last_price_at_or_before(
    points: Sequence[tuple[datetime, float]], anchor: datetime
) -> float | None:
    """The LAST observed price at/before ``anchor`` — the no-interpolation
    anchor rule. None when the series starts after the anchor."""
    best: float | None = None
    for ts, price in points:
        if ts <= anchor:
            best = price
        else:
            break
    return best


def history_features(
    points: Sequence,
    *,
    as_of: datetime,
    previous_event_at: datetime | None = None,
    window_start: datetime | None = None,
) -> MarketHistoryFeatures | None:
    """Features over PricePoint-shaped observations (``.ts``/``.price``),
    or ``None`` when nothing is observable at/before ``as_of`` — an honest
    absence the bundle reports as such, never a zeroed shell.

    Points are as-of filtered FIRST, then sorted; duplicates on the same
    instant keep the last-listed value (a refetched point overwrites, the
    storage layer's idempotence rule).
    """
    as_of = _as_aware_utc(as_of)
    filtered = sorted(
        (
            (_as_aware_utc(p.ts), float(p.price))
            for p in points
            if _as_aware_utc(p.ts) <= as_of
        ),
        key=lambda pair: pair[0],
    )
    # Collapse same-instant duplicates keeping the LAST-LISTED value (the
    # storage layer's refetch-overwrites idempotence rule) BEFORE any
    # statistic — a superseded value must not linger in high/low/range/
    # count or tilt the trend.
    by_ts = dict(filtered)  # insertion order is sorted; later value wins
    observable = list(by_ts.items())
    if not observable:
        return None

    prices = [price for _, price in observable]
    current = prices[-1]

    def delta_at(anchor: datetime | None) -> float | None:
        if anchor is None:
            return None
        anchor = _as_aware_utc(anchor)
        if anchor > as_of:
            # A future anchor (e.g. previous_event_at after as_of) has no
            # honest baseline — None, never a fabricated 0.0 self-delta.
            return None
        base = _last_price_at_or_before(observable, anchor)
        if base is None:
            return None
        return round(current - base, 4)

    anchor_deltas = {
        name: delta_at(as_of - offset) for name, offset in _CHANGE_ANCHORS
    }
    # Rounded like every delta, so the threshold comparison is not at the
    # mercy of float representation (0.42-0.40 must mean 0.02 exactly, and
    # exactly-at-threshold is FLAT per the "beyond the threshold" rule).
    overall_move = round(current - prices[0], 4)
    if overall_move > TREND_FLAT_THRESHOLD:
        trend = TREND_RISING
    elif overall_move < -TREND_FLAT_THRESHOLD:
        trend = TREND_FALLING
    else:
        trend = TREND_FLAT

    return MarketHistoryFeatures(
        current_price=current,
        change_1h=anchor_deltas["change_1h"],
        change_1d=anchor_deltas["change_1d"],
        change_7d=anchor_deltas["change_7d"],
        change_since_previous_event=delta_at(previous_event_at),
        change_since_window_start=delta_at(window_start),
        recent_high=max(prices),
        recent_low=min(prices),
        price_range=round(max(prices) - min(prices), 4),
        trend=trend,
        observation_count=len(observable),
        history_start=observable[0][0],
        history_end=observable[-1][0],
    )


# ---------------------------------------------------------------------------
# Event <-> prediction-market matching (plan Phase 4; LOOP 5).
#
# ONE CATALYST IS NOT ONE MARKET. The matcher takes a provider-discovered
# candidate pool and labels each candidate DIRECT / DERIVED / CONTEXT — or
# rejects it with a named reason. Everything here is deterministic v1
# (matched_by = DETERMINISTIC_V1): the rule tables below are maintained
# parameters, and a future LLM-assisted classifier may only RE-LABEL
# candidates from the same pool under a strict schema — it can never add a
# market that provider data does not contain, and the thresholds/caps here
# stay authoritative regardless of who proposed the labels.
#
# NO RELEVANT MARKET IS A VALID, COMMON OUTCOME: an empty accepted set is
# reported as NO_RELEVANT_PREDICTION_MARKET, never padded with loosely
# related contracts.
# ---------------------------------------------------------------------------

RELATION_DIRECT = "DIRECT"
RELATION_DERIVED = "DERIVED"
RELATION_CONTEXT = "CONTEXT"
RELATIONS: tuple[str, ...] = (RELATION_DIRECT, RELATION_DERIVED, RELATION_CONTEXT)
_RELATION_RANK: Mapping[str, int] = {r: i for i, r in enumerate(RELATIONS)}

#: The matcher version stamped on every decision (mirrors the
#: event_prediction_markets.matched_by column vocabulary).
MATCHED_BY_DETERMINISTIC_V1 = "DETERMINISTIC_V1"

#: The honest empty outcome (plan Phase 4 special case).
NO_RELEVANT_PREDICTION_MARKET = "NO_RELEVANT_PREDICTION_MARKET"

#: Acceptance bounds — deterministic code's authority, never the LLM's.
MIN_MATCH_RELEVANCE = 0.5
#: Now a GROUP-AWARE bound: acceptance admits whole venue events (see
#: match_markets), so this caps roughly how many CONTRACTS fit, and a group
#: that would overrun is refused entire rather than sliced. One complete
#: distribution is always admitted even if it exceeds this alone.
#:
#: Raised from 5 (2026-08-23) because a venue's BRACKET SERIES is one
#: distribution split across one contract per range, and Polymarket routinely
#: publishes 6-10 of them for a single release. A cap of 5 silently kept the
#: cheapest four brackets of a seven-bracket GDP series and dropped the three
#: holding 80% of the mass, so the panel read "the market prices every outcome
#: near zero" when the market in fact had a clear central estimate. The cap
#: still exists to bound an unbounded pool; it is now above the largest series
#: a venue actually publishes.
MAX_ACCEPTED_MARKETS = 10
MAX_CANDIDATE_MARKETS = 25

#: How many markets ONE discovery query may return. Distinct from
#: ``MAX_CANDIDATE_MARKETS``, which caps the whole event's pool: passing the
#: pool cap to each of ``MAX_MARKET_QUERIES`` calls would let a single press
#: fetch four times the intended ceiling. Sized so the four queries together
#: can still fill the pool.
MAX_MARKETS_PER_QUERY = 10
MAX_MARKET_QUERIES = 4

#: A DIRECT contract should RESOLVE around the event: its end date within
#: this many days AFTER the event instant (a September-CPI contract ends in
#: September, not next year) and never meaningfully BEFORE it (a contract
#: that resolves before the release measures the PRIOR period). Outside
#: the two-sided window, direct-keyword contracts demote to DERIVED with
#: the ambiguity named.
DIRECT_HORIZON_DAYS = 45
#: Lower-bound grace: providers may close trading shortly before the
#: release instant, so an end date up to this long before the event still
#: counts as resolving "around" it.
DIRECT_HORIZON_GRACE_DAYS = 1

REJECT_MARKET_NOT_ACTIVE = "MARKET_NOT_ACTIVE"
REJECT_MARKET_LOW_RELEVANCE = "LOW_RELEVANCE"
REJECT_MARKET_DUPLICATE = "DUPLICATE"
REJECT_MARKET_OVER_CANDIDATE_CAP = "OVER_CANDIDATE_CAP"
REJECT_MARKET_OVER_ACCEPT_CAP = "OVER_ACCEPT_CAP"


@dataclass(frozen=True)
class MatchTerms:
    """The per-profile relation keyword buckets — maintained parameters of
    the matcher, versioned by :data:`PREDICTION_INTEL_MODEL_VERSION`.

    ``direct``: the profile FAMILY's measure phrases. The event-TYPE table
    (:data:`DIRECT_TERMS_BY_EVENT_TYPE`) decides which of these are this
    event's OWN measure; family hits that are not the event's own measure
    (a GDP contract seen from a retail-sales event) classify as DERIVED —
    a related measure in the same complex, never "the event's own measure".
    ``derived``: contracts this event materially moves without being them.
    ``context``: broader macro/corporate backdrop wording.
    """

    direct: tuple[str, ...]
    derived: tuple[str, ...]
    context: tuple[str, ...]


#: Keyed by research-profile key (the taxonomy web_research owns), so the
#: two capabilities share one notion of event identity. Profiles not named
#: fall to _GENERIC_MATCH_TERMS.
_RATE_MOVE_TERMS = (
    "fed rate", "rate cut", "cut rates", "rate hike", "raise rates", "fomc",
)

MATCH_TERMS_BY_PROFILE: Mapping[str, MatchTerms] = {
    "inflation-v1": MatchTerms(
        direct=("cpi", "inflation rate", "core inflation", "pce"),
        derived=_RATE_MOVE_TERMS,
        context=("recession", "stagflation"),
    ),
    "gdp-v1": MatchTerms(
        direct=("gdp",),
        derived=_RATE_MOVE_TERMS,
        context=("recession", "soft landing"),
    ),
    "employment-v1": MatchTerms(
        direct=("payrolls", "jobs report", "unemployment rate", "jolts"),
        derived=_RATE_MOVE_TERMS,
        context=("recession",),
    ),
    "fomc-v1": MatchTerms(
        direct=_RATE_MOVE_TERMS + ("fed funds",),
        derived=("inflation rate", "cpi"),
        context=("recession",),
    ),
    "fed-speech-v1": MatchTerms(
        direct=(),  # a speech has no resolving contract of its own
        derived=_RATE_MOVE_TERMS,
        context=("recession",),
    ),
    "earnings-v1": MatchTerms(
        direct=("earnings", "revenue", "eps"),
        derived=("stock price", "market cap", "all-time high"),
        context=("recession", "s&p 500", "nasdaq"),
    ),
}

_GENERIC_MATCH_TERMS = MatchTerms(direct=(), derived=(), context=("recession",))

#: Which phrases are THIS event type's OWN measure — finer-grained than the
#: profile families above, because DIRECT means "the contract resolves on
#: this print", and profile siblings (GDP vs retail sales, CPI vs PPI) are
#: related measures, not the same one. Types not named have no DIRECT
#: vocabulary at all (e.g. a speech has no resolving contract of its own).
DIRECT_TERMS_BY_EVENT_TYPE: Mapping[EventType, tuple[str, ...]] = {
    EventType.CPI: ("cpi", "inflation rate", "core inflation"),
    EventType.PPI: ("ppi", "producer prices"),
    EventType.PCE: ("pce", "core pce"),
    EventType.GDP: ("gdp",),
    EventType.RETAIL_SALES: ("retail sales",),
    EventType.ISM: ("ism",),
    EventType.CONSUMER_SENTIMENT: ("consumer sentiment", "consumer confidence"),
    EventType.EMPLOYMENT_REPORT: ("payrolls", "jobs report", "unemployment rate"),
    EventType.JOLTS: ("jolts", "job openings"),
    EventType.FOMC_MEETING: _RATE_MOVE_TERMS + ("fed funds",),
    EventType.FOMC_DECISION: _RATE_MOVE_TERMS + ("fed funds",),
    EventType.FOMC_PRESS_CONFERENCE: _RATE_MOVE_TERMS + ("fed funds",),
    EventType.FOMC_MINUTES: _RATE_MOVE_TERMS + ("fed funds",),
    EventType.EARNINGS: ("earnings", "revenue", "eps"),
}

#: Relevance formula weights: aboutness (subject/title token overlap) plus a
#: relation-bucket bonus. Calibration, by construction against
#: MIN_MATCH_RELEVANCE = 0.5:
#: - NO bucket tops out at 0.45 — keyword-free lookalikes can never be
#:   forced into the bundle by overlap alone;
#: - a clean DERIVED hit sits AT the threshold even with zero subject
#:   overlap, because a derived contract's wording is by nature about a
#:   different measure (the mission's own example: a September Fed-cut
#:   contract for an upcoming CPI release shares no words with "CPI");
#: - CONTEXT needs modest subject overlap on top of its bucket — a backdrop
#:   contract must still be ABOUT this event's world to be admitted.
_ABOUTNESS_WEIGHT = 0.45
_BUCKET_BONUS: Mapping[str, float] = {
    RELATION_DIRECT: 0.55,
    RELATION_DERIVED: 0.5,
    RELATION_CONTEXT: 0.4,
}

#: Monetary authorities and jurisdictions this platform does NOT cover.
#:
#: WHY THIS EXISTS. A DERIVED contract legitimately shares no subject terms
#: with its event — the mission's own example, "Will the Fed cut rates in
#: September?" for a US CPI release, has zero token overlap — so the DERIVED
#: bucket cannot demand subject grounding without rejecting the very case it
#: was built for. That leniency is correct for US policy contracts and wrong
#: for everyone else's: "Will the Bank of Argentina cut rates?" matches the
#: same rate-move vocabulary while measuring a different country's policy.
#:
#: The events this platform tracks are US releases and US Federal Reserve
#: decisions, so a contract naming a FOREIGN authority or jurisdiction is
#: about something else — however familiar its wording. Matching is on word
#: boundaries, so "US" is never a substring hit inside another word.
FOREIGN_JURISDICTION_TERMS: tuple[str, ...] = (
    "argentina", "australia", "bank of canada", "bank of england",
    "bank of japan", "boe", "boj", "brazil", "canada", "china", "ecb",
    "england", "euro area", "european central bank", "eurozone", "france",
    "germany", "india", "indonesia", "japan", "mexico", "nigeria",
    "people's bank", "pboc", "reserve bank", "russia", "south africa",
    "south korea", "swiss national bank", "switzerland", "turkey",
    "ukraine", "united kingdom",
    # ABBREVIATIONS COUNT. Found live 2026-08-22: "Will UK annual GDP growth
    # in 2026 be between 0% and 1%?" was admitted to a US GDP release,
    # because the list spelled out "united kingdom" but not "uk". A venue
    # writes whichever form fits the headline, so both must be listed.
    "uk", "eu", "prc", "bank of korea", "banxico", "rbi", "rba", "boc",
    "britain", "british", "chinese", "european", "german", "italy", "spain",
    "netherlands", "poland", "sweden", "norway", "israel", "taiwan",
    "hong kong", "singapore", "thailand", "vietnam", "philippines",
    "colombia", "chile", "peru", "egypt", "pakistan", "bangladesh",
)

#: A foreign-subject contract is refused with its own named reason rather
#: than silently scored down — "what the platform refused to admit and why"
#: is the Evidence tab's promise.
REJECT_MARKET_FOREIGN_SUBJECT = "FOREIGN_SUBJECT"

#: A contract about a DIFFERENT company than this ticker-scoped event.
REJECT_MARKET_OTHER_ISSUER = "OTHER_ISSUER"

#: A parenthesised ticker symbol as prediction venues write it in a question:
#: "Will Broadcom (AVGO) beat quarterly earnings?". 1-5 upper-case letters so
#: it matches real symbols without swallowing ordinary parenthetical prose.
_TICKER_MENTION_RE = re.compile(r"\(([A-Z]{1,5})\)")


def market_discovery_queries(event) -> tuple[str, ...]:
    """Deterministic provider-discovery queries for this event — the
    candidate POOL generator (plan Phase 4: deterministic code generates the
    pool). Bounded by :data:`MAX_MARKET_QUERIES`, deduped, built from event
    metadata and the shared research profile only."""
    profile = research_profile(event)
    subject = event_subject(event, profile)
    terms = MATCH_TERMS_BY_PROFILE.get(profile.profile_key, _GENERIC_MATCH_TERMS)
    raw = [subject, event.title or "", event.ticker or ""]
    raw.extend(terms.direct[:1])
    raw.extend(terms.derived[:1])
    queries: list[str] = []
    seen: set[str] = set()
    for query in raw:
        text = " ".join((query or "").split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        queries.append(text)
        if len(queries) >= MAX_MARKET_QUERIES:
            break
    return tuple(queries)


@dataclass(frozen=True)
class MarketMatchDecision:
    """One candidate market with everything the matcher decided about it.

    Rejected candidates are part of the record ("what the platform refused
    to admit and why" — the transparency promise), and ``matched_by`` names
    the matcher version so rule-based and any future LLM-assisted labels
    stay distinguishable."""

    provider: str
    market_id: str
    question: str
    relation: str | None
    relevance: float
    reason: str
    ambiguity: str | None
    matched_by: str
    accepted: bool
    reject_reason: str | None
    #: The VENUE'S OWN grouping id (a Polymarket "event", a Kalshi "series").
    #: This is the authoritative statement that two contracts are brackets of
    #: one distribution — stronger than the wording-derived series_key below,
    #: which exists for venues that do not group.
    provider_event_id: str | None = None
    #: Which BRACKET SERIES this contract belongs to, when it belongs to one.
    #: Contracts like "GDP 0.5-1.0%" and "GDP 1.0-1.5%" are siblings: one
    #: distribution split across ranges. Derived deterministically from the
    #: question wording — see :func:`series_key_for`.
    series_key: str | None = None
    #: True when the accept cap cut this contract's series in half. The
    #: survivors are still shown, but a partial distribution must be LABELLED
    #: as partial rather than drawn as if it were whole.
    series_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "market_id": self.market_id,
            "question": self.question,
            "relation": self.relation,
            "relevance": self.relevance,
            "reason": self.reason,
            "ambiguity": self.ambiguity,
            "matched_by": self.matched_by,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "provider_event_id": self.provider_event_id,
            "series_key": self.series_key,
            "series_truncated": self.series_truncated,
        }


@dataclass(frozen=True)
class MatchOutcome:
    """The full, auditable result of matching one candidate pool."""

    decisions: tuple[MarketMatchDecision, ...]

    @property
    def accepted(self) -> tuple[MarketMatchDecision, ...]:
        return tuple(d for d in self.decisions if d.accepted)

    @property
    def no_relevant_market(self) -> bool:
        return not self.accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": PREDICTION_INTEL_MODEL_VERSION,
            "decisions": [d.to_dict() for d in self.decisions],
            "accepted_count": len(self.accepted),
            "no_relevant_market": self.no_relevant_market,
        }


#: A RANGE CLAUSE — the wording a venue uses to slice one distribution into
#: one contract per bracket. Deliberately narrow: it requires an explicit
#: comparator ("less than", "between … and", "greater than", "or more") AND a
#: percent/bp unit on the number. A bare year ("in Q3 2026") or a plain count
#: must not match, or two unrelated contracts would collapse into one "series"
#: and be flagged as a truncated distribution they never belonged to.
_NUM = r"-?\d+(?:\.\d+)?\s*(?:%|pp|bps?)"
_BRACKET_CLAUSE_RE = re.compile(
    "(?:"
    + r"between\s+" + _NUM + r"\s*(?:and|to|[-\u2013])\s*" + _NUM
    + r"|(?:less\s+than|under|below|at\s+most|no\s+more\s+than)\s+" + _NUM
    + r"|(?:greater\s+than|more\s+than|above|over|at\s+least)\s+" + _NUM
    + "|" + _NUM + r"\s+or\s+(?:less|lower|more|greater|higher)"
    + ")",
    re.IGNORECASE,
)


def series_key_for(question: str) -> str | None:
    """The BRACKET SERIES a contract belongs to, or None if it is standalone.

    A venue publishes a distribution as one contract per range — "Will US GDP
    growth in Q3 2026 be less than 0.5%?", "…between 0.5% and 1.0%?", "…be
    greater than 3.0%?". Those are siblings of one estimate, and reading a
    subset of them is actively misleading in a way reading a subset of
    unrelated contracts is not: four cheap brackets of a seven-bracket series
    look like "the market expects nothing", when the mass sat in the three
    that were dropped.

    The key is the question with its range clause removed, so every bracket of
    one series collapses to the same stem while the surrounding wording — the
    subject AND the period, which is what separates "Q3 2026" from "2026" —
    is preserved. Returns None when no range clause is present: a plain
    yes/no contract has no siblings to be truncated away from.
    """
    text = (question or "").strip()
    if not text:
        return None
    if not _BRACKET_CLAUSE_RE.search(text):
        return None
    stem = " ".join(_BRACKET_CLAUSE_RE.sub(" ", text).lower().split())
    # A stem that collapsed to almost nothing cannot identify a series.
    return stem if len(stem) >= 12 else None


def _phrase_hits(text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Word-boundary phrase matches — 'eps' must not hit 'sleeps' and
    'gdp' must not hit a ticker fragment; substring matching would."""
    lowered = text.lower()
    return tuple(
        p for p in phrases
        if re.search(rf"\b{re.escape(p)}\b", lowered)
    )


def _classify(event, market, *, as_of: datetime) -> MarketMatchDecision:
    """One candidate's relation/relevance/reason — pure rules, no I/O."""
    profile = research_profile(event)
    subject_tokens = frozenset(
        tokens(f"{event_subject(event, profile)} {event.title or ''}")
    )
    market_text = " ".join(
        part for part in (
            market.question,
            market.resolution_criteria or "",
        ) if part
    )
    market_tokens = frozenset(tokens(market_text))
    overlap = (
        len(subject_tokens & market_tokens) / len(subject_tokens)
        if subject_tokens and market_tokens else 0.0
    )

    terms = MATCH_TERMS_BY_PROFILE.get(profile.profile_key, _GENERIC_MATCH_TERMS)
    type_direct = DIRECT_TERMS_BY_EVENT_TYPE.get(event.event_type, ())
    direct_hits = _phrase_hits(market_text, type_direct)
    # Profile-family measures that are NOT this event's own measure (a GDP
    # contract seen from a retail-sales event): related, never DIRECT.
    sibling_hits = _phrase_hits(
        market_text, tuple(t for t in terms.direct if t not in type_direct)
    )
    derived_hits = sibling_hits + _phrase_hits(market_text, terms.derived)
    context_hits = _phrase_hits(market_text, terms.context)

    relation: str | None = None
    reason = ""
    ambiguity: str | None = None
    if direct_hits:
        event_at = _as_aware_utc(event.scheduled_at)
        # TWO-SIDED horizon: a contract resolving meaningfully BEFORE the
        # event measures the PRIOR period (last month's still-active CPI
        # contract), and one resolving far after may span several — both
        # demote. DIRECT also needs SOME subject grounding: bare measure
        # wording with zero subject overlap could be a different
        # country's/entity's contract.
        horizon_ok = market.end_date is not None and (
            event_at - timedelta(days=DIRECT_HORIZON_GRACE_DAYS)
            <= _as_aware_utc(market.end_date)
            <= event_at + timedelta(days=DIRECT_HORIZON_DAYS)
        )
        # A TICKER-SCOPED EVENT DEMANDS ITS TICKER. Generic token overlap is
        # not grounding here: measured live 2026-08-22, an HPE earnings event
        # matched Broadcom/Dell/Snowflake "beat quarterly earnings" contracts
        # as DIRECT at 0.775, because each contract's RESOLUTION CRITERIA
        # happened to name the same release date ("September 2, 2026") and
        # that date supplied the overlap. Pricing for another company's
        # earnings would then have been quoted to the model as this event's.
        #
        # For an event with a ticker, the contract must name that ticker to
        # be DIRECT; otherwise it demotes and is judged on its own merits.
        subject_ok = overlap > 0
        event_ticker = (getattr(event, "ticker", None) or "").strip().lower()
        if event_ticker:
            subject_ok = event_ticker in market_tokens

        if horizon_ok and subject_ok:
            relation = RELATION_DIRECT
            reason = (
                f"contract wording matches the event's own measure "
                f"({', '.join(direct_hits)}) and resolves within "
                f"{DIRECT_HORIZON_DAYS}d of the event"
            )
        else:
            relation = RELATION_DERIVED
            reason = (
                f"contract wording matches the event's measure "
                f"({', '.join(direct_hits)}) but "
                + (
                    "its resolution horizon is not tied to this release"
                    if not horizon_ok
                    else (
                        f"its wording does not name {event_ticker.upper()}"
                        if event_ticker
                        else "its wording shares no subject terms with this event"
                    )
                )
            )
            ambiguity = (
                "resolution wording may span a different release period "
                "than this event"
                if not horizon_ok
                else "direct wording but the contract may measure a "
                "different subject or geography"
            )
    elif derived_hits:
        relation = RELATION_DERIVED
        if sibling_hits:
            reason = (
                f"related measure in the same complex "
                f"({', '.join(sibling_hits)}) — not this event's own measure"
            )
        else:
            reason = (
                "the event materially affects this contract "
                f"({', '.join(derived_hits)}) without being its measure"
            )
    elif context_hits:
        relation = RELATION_CONTEXT
        reason = (
            f"broader backdrop contract ({', '.join(context_hits)}) "
            "related to the event's macro context"
        )

    # FOREIGN SUBJECT OVERRIDES ANY BUCKET. A contract naming another
    # country's monetary authority speaks the same rate-move vocabulary as a
    # Fed contract and would otherwise ride the DERIVED bonus straight past
    # the threshold — the DIRECT branch's own ambiguity text already names
    # this risk ("may measure a different subject or geography") but only
    # demotes into DERIVED, which has no floor. The platform tracks US
    # releases and US Fed decisions; another jurisdiction's contract is about
    # something else, and is refused with its own named reason.
    # A TICKER-SCOPED EVENT AND ANOTHER ISSUER'S CONTRACT ARE DIFFERENT
    # SUBJECTS, not a derived relationship. Measured live 2026-08-22: an HPE
    # earnings event accepted Broadcom/Dell/Snowflake "beat quarterly
    # earnings" contracts because their resolution criteria named the same
    # release date. Once demoted out of DIRECT they landed in DERIVED, which
    # carries no subject floor — so the refusal has to be explicit.
    #
    # Only fires when the wording NAMES a measure of the event's own type
    # (earnings): a Fed-cut contract during an earnings week names no issuer
    # and is legitimately DERIVED, and is untouched here.
    #
    # The trigger is that the contract NAMES A DIFFERENT TICKER — a
    # parenthesised symbol like "(AVGO)" — not merely that it lacks ours.
    # A Fed-cut contract during an earnings week names no issuer at all and
    # remains legitimately DERIVED.
    event_ticker = (getattr(event, "ticker", None) or "").strip().lower()
    other_issuer = None
    if event_ticker and relation is not None:
        named = {
            sym.lower()
            for sym in _TICKER_MENTION_RE.findall(market.question or "")
        }
        named.discard(event_ticker)
        if named and event_ticker not in market_tokens:
            other_issuer = sorted(named)[0]
    if other_issuer is not None:
        return MarketMatchDecision(
            provider=market.provider,
            market_id=market.market_id,
            question=market.question,
            relation=relation,
            relevance=0.0,
            reason=(
                f"contract measures {other_issuer.upper()}, not "
                f"{event_ticker.upper()}"
            ),
            ambiguity=None,
            matched_by=MATCHED_BY_DETERMINISTIC_V1,
            accepted=False,
            reject_reason=REJECT_MARKET_OTHER_ISSUER,
        )

    foreign_hits = _phrase_hits(market_text, FOREIGN_JURISDICTION_TERMS)
    if foreign_hits and relation is not None:
        return MarketMatchDecision(
            provider=market.provider,
            market_id=market.market_id,
            question=market.question,
            relation=relation,
            relevance=0.0,
            reason=(
                f"contract names a foreign jurisdiction "
                f"({', '.join(foreign_hits)}) — not this event's subject"
            ),
            ambiguity=None,
            matched_by=MATCHED_BY_DETERMINISTIC_V1,
            accepted=False,
            reject_reason=REJECT_MARKET_FOREIGN_SUBJECT,
        )

    bonus = _BUCKET_BONUS.get(relation, 0.0) if relation else 0.0
    relevance = round(min(1.0, _ABOUTNESS_WEIGHT * overlap + bonus), 4)
    if not relation:
        reason = "no relation-defining wording for this event type"
    return MarketMatchDecision(
        provider=market.provider,
        market_id=market.market_id,
        question=market.question,
        relation=relation,
        relevance=relevance,
        reason=reason,
        ambiguity=ambiguity,
        matched_by=MATCHED_BY_DETERMINISTIC_V1,
        accepted=False,  # acceptance is match_markets' decision
        reject_reason=None,
        provider_event_id=getattr(market, "provider_event_id", None),
        series_key=series_key_for(market.question),
    )


def match_markets(
    event,
    candidates: Sequence,
    *,
    as_of: datetime,
    min_relevance: float = MIN_MATCH_RELEVANCE,
    max_accepted: int = MAX_ACCEPTED_MARKETS,
    max_candidates: int = MAX_CANDIDATE_MARKETS,
) -> MatchOutcome:
    """Label a provider-discovered candidate pool for one event.

    STRUCTURAL GUARANTEE (plan Phase 4): every decision refers to a market
    from ``candidates`` — this function cannot mint one, so nothing
    downstream (bundle, LLM, UI) can ever cite a market that provider data
    does not contain. Candidates are PredictionMarketInfo-shaped
    (structural typing; this module imports no provider package).

    Deterministic acceptance: relevance >= ``min_relevance``, ranked by
    (relevance desc, DIRECT before DERIVED before CONTEXT, market_id), the
    top ``max_accepted`` accepted. An empty accepted set is the valid
    NO_RELEVANT_PREDICTION_MARKET outcome, never padded.
    """
    as_of = _as_aware_utc(as_of)
    decisions: list[MarketMatchDecision] = []
    seen: set[tuple[str, str]] = set()
    considered = 0
    from dataclasses import replace as _replace

    for market in candidates:
        key = (market.provider, market.market_id)
        decision = _classify(event, market, as_of=as_of)
        if key in seen:
            decisions.append(_replace(
                decision, reject_reason=REJECT_MARKET_DUPLICATE
            ))
            continue
        seen.add(key)
        considered += 1
        if considered > max_candidates:
            decisions.append(_replace(
                decision, reject_reason=REJECT_MARKET_OVER_CANDIDATE_CAP
            ))
            continue
        if market.status != "ACTIVE":
            # A closed/resolved contract cannot price the UPCOMING event.
            decisions.append(_replace(
                decision, reject_reason=REJECT_MARKET_NOT_ACTIVE
            ))
            continue
        if decision.reject_reason is not None:
            # The classifier already refused this candidate for a SPECIFIC
            # reason (a foreign jurisdiction). Keep it: overwriting with the
            # generic LOW_RELEVANCE would erase the one fact the Evidence
            # tab needs to explain the refusal.
            decisions.append(decision)
            continue
        if decision.relation is None or decision.relevance < min_relevance:
            decisions.append(_replace(
                decision, reject_reason=REJECT_MARKET_LOW_RELEVANCE
            ))
            continue
        decisions.append(decision)  # acceptance-eligible; capped below

    eligible = [
        i for i, d in enumerate(decisions)
        if d.reject_reason is None
    ]
    ranked = sorted(
        eligible,
        key=lambda i: (
            -decisions[i].relevance,
            _RELATION_RANK.get(decisions[i].relation, len(RELATIONS)),
            decisions[i].market_id,
        ),
    )

    # THE VENUE EVENT IS THE UNIT OF ACCEPTANCE, NOT THE CONTRACT.
    #
    # A venue publishes one distribution as one contract per range, grouped
    # under its own event id. Those brackets are only meaningful together:
    # accepting the four highest-ranked of seven does not give a smaller
    # picture of the market, it gives a WRONG one, because the survivors are
    # whichever the ranking favoured and the probability mass may sit
    # entirely in the ones dropped.
    #
    # So the cap is applied to GROUPS. A group is admitted whole while it
    # fits and refused whole when it does not — and the refusal keeps its own
    # reason, so the Evidence tab can say "this distribution was left out"
    # rather than showing part of it. Ungrouped contracts (no series, no
    # venue event) are their own group of one and behave exactly as before.
    def _group_of(index: int) -> str:
        decision = decisions[index]
        return (
            decision.provider_event_id
            or decision.series_key
            or f"solo:{decision.market_id}"
        )

    order: list[str] = []
    members: dict[str, list[int]] = {}
    for index in ranked:
        key = _group_of(index)
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(index)

    accepted_count = 0
    for key in order:
        indexes = members[key]
        # An empty pool takes the first group whatever its size: refusing a
        # nine-bracket distribution for a cap of eight would leave the reader
        # with nothing at all, which is strictly worse than one complete
        # distribution that overruns.
        fits = accepted_count == 0 or accepted_count + len(indexes) <= max(
            0, max_accepted
        )
        for index in indexes:
            if fits:
                decisions[index] = _replace(decisions[index], accepted=True)
            else:
                decisions[index] = _replace(
                    decisions[index],
                    reject_reason=REJECT_MARKET_OVER_ACCEPT_CAP,
                )
        if fits:
            accepted_count += len(indexes)

    truncated = {
        d.series_key
        for d in decisions
        if d.reject_reason == REJECT_MARKET_OVER_ACCEPT_CAP and d.series_key
    }
    if truncated:
        for index, decision in enumerate(decisions):
            if decision.accepted and decision.series_key in truncated:
                decisions[index] = _replace(
                    decision, series_truncated=True
                )
    return MatchOutcome(decisions=tuple(decisions))


# ---------------------------------------------------------------------------
# PRICE MOVES WORTH ASKING ABOUT
#
# A contract sitting at 36c tells you where the market is. It does not tell
# you when it got there, and "when" is the question that leads to "why" — the
# day a bracket jumped 15 points is the day something happened, and that
# something is usually findable in the record.
#
# This layer does the FIRST half only: it locates the moves deterministically.
# It does not name a cause, because a price move and a headline on the same
# day are a COINCIDENCE UNTIL SOMEONE CHECKS, and a layer that quietly paired
# them would manufacture exactly the false narrative the platform exists to
# avoid. Naming the candidate window is useful; asserting the reason is not
# this module's to do.
# ---------------------------------------------------------------------------

#: A day's absolute price change, in probability points, at or above which a
#: move is worth surfacing. 10 points is deliberately high: the aim is the
#: handful of days a reader should actually investigate, not every wiggle.
#: Below this, a contract is drifting; at this size, the market changed its
#: mind about something.
MOVE_MIN_ABS_CHANGE = 0.10

#: How many moves to report, largest first. A "top moves" list that runs to
#: forty entries is a second chart, not a summary.
MOVE_MAX_REPORTED = 8


@dataclass(frozen=True)
class PriceMove:
    """One notable step between two consecutive observations.

    ``from_ts``/``to_ts`` bound the window a reader would search for a cause.
    They are the OBSERVATION instants, not a guess at when the news broke:
    with daily points the true moment lies somewhere inside, and pretending
    otherwise would put false precision on the one number a reader would use
    to go looking.
    """

    from_ts: datetime
    to_ts: datetime
    from_price: float
    to_price: float
    change: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_ts": self.from_ts.isoformat(),
            "to_ts": self.to_ts.isoformat(),
            "from_price": round(self.from_price, 4),
            "to_price": round(self.to_price, 4),
            "change": round(self.change, 4),
            # The direction a reader is looking for in the record: did the
            # market start believing this outcome, or stop?
            "direction": "UP" if self.change > 0 else "DOWN",
        }


def notable_moves(
    points: Sequence,
    *,
    as_of: datetime,
    min_abs_change: float = MOVE_MIN_ABS_CHANGE,
    limit: int = MOVE_MAX_REPORTED,
) -> tuple[PriceMove, ...]:
    """The largest step-changes in an observed series, largest first.

    Steps are between CONSECUTIVE observations, so a gap in the series
    produces one wide step rather than a fabricated daily path — the
    no-interpolation rule applied to differences. As-of gated first, like
    every other function here.
    """
    as_of = _as_aware_utc(as_of)
    observable = sorted(
        {
            _as_aware_utc(p.ts): float(p.price)
            for p in points
            if _as_aware_utc(p.ts) <= as_of
        }.items()
    )
    if len(observable) < 2:
        return ()

    moves: list[PriceMove] = []
    for (prev_ts, prev_price), (ts, price) in zip(observable, observable[1:]):
        change = round(price - prev_price, 4)
        if abs(change) >= min_abs_change:
            moves.append(
                PriceMove(
                    from_ts=prev_ts,
                    to_ts=ts,
                    from_price=prev_price,
                    to_price=price,
                    change=change,
                )
            )
    moves.sort(key=lambda m: (-abs(m.change), m.to_ts))
    return tuple(moves[: max(0, limit)])
