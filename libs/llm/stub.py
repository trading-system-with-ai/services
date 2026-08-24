"""Deterministic stub recommendation provider.

NOT LLM OUTPUT. NEVER A DEFAULT.

Every draft this module produces is TEMPLATE-GENERATED — assembled from
hand-written phrases and seeded scores, with no model call, no news, and no
analysis behind it. It exists so local development and the test suite can
exercise the recommendation pipeline deterministically, and it is reachable
ONLY by explicitly setting ``LLM_PROVIDER=stub``. It must NEVER be used as a
default or a fallback: when no LLM provider is configured the platform
produces NO recommendations (HTTP 503 ``LLM_NOT_CONFIGURED``) rather than text
that reads like real analysis but is not (§44 rule 18).

STUB ONLY (development plan §4.1): this provider serves hand-written,
plausible-sounding candidate drafts for local development and tests until the
real news ingestion + LLM pipeline lands (Phase 8 full). It performs no I/O.

Determinism: both the selection of tickers and every score are seeded by
``zlib.crc32(f"{as_of.date()}|{ticker}")`` — the same calendar day always
yields the exact same drafts, across calls and processes, which keeps tests
and local demos reproducible. Different days rotate the picks and scores.

News timestamp integrity (plan §20.3): every evidence entry carries a
``published_at`` strictly BEFORE ``as_of`` — a recommendation may only cite
information that existed at its as-of time. The stub enforces this by
construction: timestamps are anchored to midnight of ``as_of``'s calendar day
minus at least one hour, so they land strictly before any ``as_of`` on that
day (and the same day therefore yields byte-identical drafts).

SAFETY (plan §4.1, §44 rule 5, §46): drafts are information only — nothing
here touches the Watchlist, the Trading Pool, or orders.
"""
import random
import zlib
from datetime import datetime, time, timedelta

from .market_selection import MarketSelectionResult
from .event_analysis import (
    CONFIDENCE_LEVELS,
    PROMPT_VERSION,
    EventAnalysisResult,
)
from .provider import GroundingArticle, RecommendationDraft

# Minimum gap between an evidence item's published_at and as_of, guaranteeing
# the strict "published before as_of" ordering of plan §20.3. A parameter,
# not a truth.
DEFAULT_MIN_EVIDENCE_AGE_HOURS = 1
DEFAULT_MAX_EVIDENCE_AGE_HOURS = 36

# Fixed universe of liquid, optionable large-caps with hand-written catalyst
# templates. Tuple: (ticker, company, catalyst_type, direction, horizon,
# reason_codes, summary_template, snippet_template).
_UNIVERSE: list[tuple[str, str, str, int, str, list[str], str, str]] = [
    ("AAPL", "Apple Inc.", "product_launch", +1, "2-6 weeks",
     ["PRODUCT_CYCLE", "SUPPLY_CHAIN_CHECKS"],
     "Supply-chain checks point to a stronger-than-expected product refresh cycle for {t}.",
     "Component orders for {c} suggest unit builds above consensus for the coming quarter."),
    ("MSFT", "Microsoft Corporation", "earnings_surprise", +1, "1-2 weeks",
     ["CLOUD_ACCELERATION", "EARNINGS_MOMENTUM"],
     "Cloud consumption trends set up a potential earnings beat for {t}.",
     "Channel data shows {c} Azure bookings re-accelerating quarter over quarter."),
    ("NVDA", "NVIDIA Corporation", "guidance_raise", +1, "1-4 weeks",
     ["AI_CAPEX", "BACKLOG_GROWTH"],
     "Datacenter order backlog supports another guidance raise for {t}.",
     "Hyperscaler capex commentary implies sustained accelerator demand for {c}."),
    ("AMZN", "Amazon.com Inc.", "margin_inflection", +1, "1-3 months",
     ["MARGIN_EXPANSION", "COST_DISCIPLINE"],
     "Fulfillment cost leverage points to a retail margin inflection at {t}.",
     "Analysis of {c} logistics costs shows per-unit shipping expense declining."),
    ("GOOGL", "Alphabet Inc.", "regulatory", -1, "1-3 months",
     ["ANTITRUST_RISK", "HEADLINE_RISK"],
     "Pending antitrust remedies create headline risk for {t}.",
     "Filings indicate the remedies phase for {c} could target default-placement deals."),
    ("META", "Meta Platforms Inc.", "earnings_surprise", +1, "1-2 weeks",
     ["AD_PRICING", "ENGAGEMENT_TRENDS"],
     "Ad-pricing checks suggest upside to consensus revenue for {t}.",
     "Third-party trackers show {c} ad impressions and pricing both firming."),
    ("TSLA", "Tesla Inc.", "delivery_miss", -1, "1-2 weeks",
     ["DELIVERY_TRACKERS", "DEMAND_SOFTNESS"],
     "Registration trackers point to a potential delivery shortfall for {t}.",
     "Weekly registration data for {c} runs below the pace implied by guidance."),
    ("JPM", "JPMorgan Chase & Co.", "rate_environment", +1, "1-3 months",
     ["NET_INTEREST_INCOME", "CREDIT_QUALITY"],
     "A stable rate path supports net interest income durability at {t}.",
     "Deposit beta commentary for {c} implies NII guidance is conservative."),
    ("XOM", "Exxon Mobil Corporation", "commodity_move", +1, "2-8 weeks",
     ["CRUDE_STRENGTH", "REFINING_MARGINS"],
     "Tightening crude balances favor upstream earnings for {t}.",
     "Inventory draws and refining margin strength both benefit {c} this quarter."),
    ("UNH", "UnitedHealth Group Inc.", "regulatory", -1, "1-3 months",
     ["MEDICAL_COST_TREND", "POLICY_RISK"],
     "Elevated medical cost trend and policy scrutiny weigh on {t}.",
     "Utilization commentary across insurers implies cost trend above {c} plan."),
    ("AVGO", "Broadcom Inc.", "ai_orders", +1, "1-4 weeks",
     ["CUSTOM_SILICON", "ORDER_MOMENTUM"],
     "Custom AI accelerator orders are tracking ahead of plan for {t}.",
     "Supply-chain checks show {c} custom silicon programs expanding with new customers."),
    ("LLY", "Eli Lilly and Company", "clinical_readout", +1, "1-3 months",
     ["PIPELINE_CATALYST", "SCRIPT_TRENDS"],
     "Prescription trends and a pipeline readout set up a catalyst window for {t}.",
     "Weekly script data for {c} incretin franchise continues to beat consensus models."),
    ("COST", "Costco Wholesale Corporation", "comps_beat", +1, "2-6 weeks",
     ["TRAFFIC_GROWTH", "MEMBERSHIP_RENEWALS"],
     "Monthly comparable sales momentum supports upside at {t}.",
     "Traffic and renewal metrics for {c} remain near record levels."),
    ("AMD", "Advanced Micro Devices Inc.", "market_share", +1, "1-3 months",
     ["SHARE_GAINS", "DATACENTER_RAMP"],
     "Server share gains and a datacenter GPU ramp support estimates for {t}.",
     "OEM checks show {c} server CPU share continuing to climb."),
    ("NFLX", "Netflix Inc.", "subscriber_trends", +1, "1-2 weeks",
     ["SUB_ADDS", "AD_TIER_RAMP"],
     "Download and panel data point to a subscriber beat for {t}.",
     "Panel trackers show {c} ad-tier sign-ups accelerating into quarter end."),
]

_SOURCES = ["Reuters", "Bloomberg", "WSJ", "FT", "Company IR"]


def _seed(as_of: datetime, ticker: str) -> int:
    """Stable per-(day, ticker) seed — the determinism contract of this stub."""
    return zlib.crc32(f"{as_of.date().isoformat()}|{ticker}".encode("utf-8"))


#: How many bundle facts the template note copies. Small on purpose: the stub
#: is a pipeline fixture, not a report.
STUB_MAX_QUOTED_FACTS = 3


def _fact_index(bundle_json: dict) -> dict:
    """The bundle's flat {dotted path: value} fact map.

    Prefers the canonical implementation in
    ``libs.trading_core.events.evidence`` — quoting a path the REAL validator
    will not recognise would make the stub fail its own validation — and falls
    back to an equivalent local flattening when that module is unavailable
    (libs.llm must stay importable on its own).
    """
    if not isinstance(bundle_json, dict):
        return {}
    try:
        from libs.trading_core.events.evidence import fact_index as _canonical
    except Exception:  # pragma: no cover - fallback path
        return _flatten(bundle_json)
    try:
        index = _canonical(bundle_json)
    except Exception:  # pragma: no cover - defensive
        return _flatten(bundle_json)
    return index if isinstance(index, dict) else {}


def _flatten(value: object, prefix: str = "") -> dict:
    """Fallback flattening: dotted paths to scalar leaves, list index as a key."""
    out: dict = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            out.update(_flatten(child, f"{prefix}.{i}"))
    elif isinstance(value, bool) or value is None:
        return out
    elif isinstance(value, (int, float, str)):
        if prefix:
            out[prefix] = value
    return out


def _render(value: float | int) -> str:
    """A number as the stub prints it — the exact text it also quotes."""
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _evidence_refs(facts: dict, bundle_json: dict) -> list[str]:
    """Deterministic citations: known news evidence ids, else quoted paths.

    Never invents an id — an unknown ref is a validator violation, and the
    stub's whole purpose is to pass the same gate a real provider must.
    """
    news_ids = sorted(
        {
            value
            for key, value in facts.items()
            if isinstance(value, str)
            and (value.startswith("news:") or key.endswith("evidence_id"))
        }
    )[:STUB_MAX_QUOTED_FACTS]
    if news_ids:
        return news_ids
    return sorted(facts)[:STUB_MAX_QUOTED_FACTS]


class StubRecommendationProvider:
    """Deterministic, offline implementation of RecommendationProvider."""

    def __init__(
        self,
        min_evidence_age_hours: int = DEFAULT_MIN_EVIDENCE_AGE_HOURS,
        max_evidence_age_hours: int = DEFAULT_MAX_EVIDENCE_AGE_HOURS,
    ) -> None:
        self.min_evidence_age_hours = min_evidence_age_hours
        self.max_evidence_age_hours = max_evidence_age_hours

    def generate(
        self,
        exclude_tickers: set[str],
        as_of: datetime,
        limit: int = 5,
    ) -> list[RecommendationDraft]:
        """Return up to `limit` deterministic drafts for `as_of`'s calendar day.

        Excluded tickers are never returned (they are dropped before
        selection, so an exclusion changes which candidates fill the slots
        rather than shrinking the result below `limit` unnecessarily).
        """
        candidates = [row for row in _UNIVERSE if row[0] not in exclude_tickers]
        # Day-seeded rotation: rank by the per-(day, ticker) hash so the same
        # day always picks the same tickers, and different days rotate them.
        candidates.sort(key=lambda row: _seed(as_of, row[0]))
        return [self._draft(row, as_of) for row in candidates[:limit]]

    def _draft(
        self,
        row: tuple[str, str, str, int, str, list[str], str, str],
        as_of: datetime,
    ) -> RecommendationDraft:
        ticker, company, catalyst_type, direction, horizon, reason_codes, summary_t, snippet_t = row
        rng = random.Random(_seed(as_of, ticker))

        sentiment = round(direction * rng.uniform(0.30, 0.90), 4)
        impact = round(rng.uniform(0.30, 0.90), 4)
        novelty = round(rng.uniform(0.20, 0.90), 4)
        source_reliability = round(rng.uniform(0.50, 0.95), 4)

        # Plan §20.3 news timestamp integrity: evidence published STRICTLY
        # before as_of. Anchoring to midnight of as_of's day minus at least
        # min_evidence_age_hours guarantees strictness for ANY as_of on that
        # day (day_start - 1h < day_start <= as_of), and keeps the drafts
        # identical for the whole day (the determinism contract).
        day_start = datetime.combine(as_of.date(), time.min, tzinfo=as_of.tzinfo)
        first_age_h = rng.randint(self.min_evidence_age_hours, self.max_evidence_age_hours)
        second_age_h = first_age_h + rng.randint(1, 24)
        source_a, source_b = rng.sample(_SOURCES, 2)
        evidence = [
            {
                "source": source_a,
                "published_at": (day_start - timedelta(hours=first_age_h)).isoformat(),
                "snippet": snippet_t.format(c=company),
            },
            {
                "source": source_b,
                "published_at": (day_start - timedelta(hours=second_age_h)).isoformat(),
                "snippet": f"Earlier coverage of the same {catalyst_type} theme for {company}.",
            },
        ]

        return RecommendationDraft(
            ticker=ticker,
            company=company,
            sentiment=sentiment,
            impact=impact,
            novelty=novelty,
            source_reliability=source_reliability,
            horizon=horizon,
            catalyst_type=catalyst_type,
            reason_codes=list(reason_codes),
            summary=summary_t.format(t=ticker),
            evidence=evidence,
        )

    def enrich(
        self,
        articles: list["GroundingArticle"],
        exclude_tickers: set[str],
        as_of: datetime,
        limit: int = 5,
    ) -> list[RecommendationDraft]:
        """Deterministic grounded drafts: one per usable article, in order.

        Mirrors the REAL contract exactly — every draft's evidence cites the
        source article's url verbatim and the ticker comes from that
        article's own ticker list — so the router's grounding validation
        exercises the same code path it will run against OpenAI output.
        """
        drafts: list[RecommendationDraft] = []
        seen: set[str] = set()
        for article in articles:
            ticker = next(
                (t for t in article.tickers if t not in exclude_tickers and t not in seen),
                None,
            )
            if ticker is None:
                continue
            seen.add(ticker)
            seed = zlib.crc32(f"{ticker}:{article.url}".encode())
            rng = random.Random(seed)
            sentiment = round(rng.uniform(-0.4, 0.9), 2)
            drafts.append(
                RecommendationDraft(
                    ticker=ticker,
                    company=None,
                    sentiment=sentiment,
                    impact=round(rng.uniform(0.3, 0.9), 2),
                    novelty=round(rng.uniform(0.3, 0.9), 2),
                    source_reliability=0.5,  # stub newsroom: honestly middling
                    horizon="1-5d",
                    catalyst_type="news_catalyst",
                    reason_codes=["STUB_NEWS_GROUNDED"],
                    summary=f"[SYNTHETIC] {article.title}",
                    evidence=[
                        {
                            "source": article.url,
                            "published_at": article.published_at,
                            "snippet": article.title,
                        }
                    ],
                )
            )
            if len(drafts) >= limit:
                break
        return drafts

    def select_prediction_market_events(
        self,
        *,
        event_type: str,
        event_title: str,
        scheduled_at: str,
        options,
        as_of: datetime,
    ) -> MarketSelectionResult:
        """DETERMINISTIC, NOT A SELECTION. The stub makes no semantic
        judgement, so it selects NOTHING and says so — which routes the
        caller to the pure matcher over the untouched pool.

        Returning a plausible-looking pick here would be the worst option
        available: a test or a demo would appear to exercise model selection
        while exercising a hardcoded index, and the difference would only
        surface against a real venue.
        """
        return MarketSelectionResult(
            selections=(),
            note="stub provider makes no selection; deterministic matching decides",
        )

    def analyze_event(
        self,
        bundle_json: dict,
        *,
        as_of: datetime,
    ) -> EventAnalysisResult:
        """A deterministic, TEMPLATE-GENERATED pre-event note. NOT ANALYSIS.

        Exists so the persistence/API/UI path can be exercised end-to-end
        offline. It obeys the same §47 contract the real providers are held
        to — every number it prints is copied out of the bundle's own fact
        index and listed in ``numbers_quoted`` with its dotted path — so it
        passes :func:`~libs.llm.event_analysis.validate_analysis` and the
        validator is genuinely exercised in tests rather than bypassed.

        It quotes at most :data:`STUB_MAX_QUOTED_FACTS` numeric facts, chosen
        by sorted path order, so the same bundle always yields byte-identical
        output. Confidence is LOW and the regime is INSUFFICIENT_DATA whenever
        fundamentals are missing — the stub has no view, and must never read
        like one.
        """
        facts = _fact_index(bundle_json)
        numeric = [
            (path, value)
            for path, value in sorted(facts.items())
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ][:STUB_MAX_QUOTED_FACTS]
        quoted_phrase = (
            "; ".join(f"{path} = {_render(value)}" for path, value in numeric)
            or "no numeric facts are present in the evidence bundle"
        )

        event = bundle_json.get("event") if isinstance(bundle_json, dict) else None
        event_key = ""
        if isinstance(event, dict):
            event_key = str(event.get("event_key") or event.get("ticker") or "")
        subject = event_key or "this event"

        fundamentals = (
            bundle_json.get("fundamentals") if isinstance(bundle_json, dict) else None
        )
        have_fundamentals = bool(
            isinstance(fundamentals, dict) and fundamentals.get("available")
        )
        prediction_markets = (
            bundle_json.get("prediction_markets")
            if isinstance(bundle_json, dict)
            else None
        )
        have_prediction_markets = bool(
            isinstance(prediction_markets, dict)
            and prediction_markets.get("available")
        )
        regime = "INSUFFICIENT_DATA"

        refs = _evidence_refs(facts, bundle_json)

        summary = (
            f"[SYNTHETIC — no model call] Template pre-event note for {subject}. "
            f"Facts copied from the evidence bundle: {quoted_phrase}."
        )
        scenario = lambda name: {  # noqa: E731 - local template, not logic
            "conditions": (
                f"[SYNTHETIC] {name} conditions are not assessed by the stub "
                f"provider; configure a real LLM provider for analysis."
            ),
            "guidance_conditions": "[SYNTHETIC] guidance conditions not assessed.",
            "why_market_reacts": (
                "[SYNTHETIC] market-reaction mechanics not assessed by the stub."
            ),
            "evidence_refs": list(refs),
        }

        analysis = {
            "executive_summary": summary,
            "what_happened_last_time": (
                "[SYNTHETIC] The stub provider does not reconstruct the previous "
                "event; see the DATA and QUANT sections of the bundle."
            ),
            "what_changed_since": (
                "[SYNTHETIC] No change assessment is produced without a model."
            ),
            "fundamental_developments": (
                "[SYNTHETIC] Fundamentals are present in the bundle."
                if have_fundamentals
                else "[SYNTHETIC] Fundamentals are unavailable in this bundle."
            ),
            "price_and_positioning": (
                f"[SYNTHETIC] Positioning is not assessed. Bundle facts: "
                f"{quoted_phrase}."
            ),
            "market_expectations": (
                "[SYNTHETIC] Consensus data is unavailable in this subscription, "
                "so market expectations are not quantified."
            ),
            # v2: null unless the bundle actually carries matched markets —
            # the stub must exercise the honest-absence branch of the
            # contract, never narrate pricing that is not there.
            "prediction_market_expectations": (
                "[SYNTHETIC] Prediction-market pricing is present in the "
                "bundle but is not interpreted by the stub provider."
                if have_prediction_markets
                else None
            ),
            "key_positive_catalysts": ["[SYNTHETIC] none identified by the stub"],
            "key_negative_catalysts": ["[SYNTHETIC] none identified by the stub"],
            "what_matters_most": (
                "[SYNTHETIC] Configure a real LLM provider; the stub has no view."
            ),
            "scenarios": {
                "upside": scenario("UPSIDE"),
                "base": scenario("BASE"),
                "downside": scenario("DOWNSIDE"),
            },
            "surprise_threshold": {
                "narrative": (
                    "[SYNTHETIC] No surprise threshold is estimated without a model."
                ),
                "confidence": "NOT_MEANINGFUL",
            },
            "key_unknowns": [
                "[SYNTHETIC] everything — this note was template-generated, not analysed",
            ],
            # v2: honestly empty — the stub compares nothing and highlights
            # nothing; both fields' contracts say empty beats invented.
            "evidence_conflicts": [],
            "web_research_highlights": [],
            "invalidation": (
                "[SYNTHETIC] This note carries no thesis, so nothing invalidates it."
            ),
            "expectations_gap_regime": regime,
            "confidence": "LOW",
            "evidence_refs": list(refs),
            "numbers_quoted": [
                {"path": path, "value": value} for path, value in numeric
            ],
        }
        assert analysis["confidence"] in CONFIDENCE_LEVELS
        return EventAnalysisResult(
            analysis=analysis,
            model="stub",
            provider="stub",
            prompt_version=PROMPT_VERSION,
            usage=None,  # honest None: no tokens were spent, none are reported
            latency_ms=0,
            violations=[],
        )
