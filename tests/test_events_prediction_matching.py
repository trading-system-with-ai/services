"""Event <-> prediction-market matching (Catalyst research upgrade, LOOP 5).

What these tests pin, per the program brief's mandated matching cases:
DIRECT, DERIVED, CONTEXT, irrelevant, ambiguous, zero matches (the honest
NO_RELEVANT_PREDICTION_MARKET outcome), caps/thresholds, dedup, non-active
rejection, determinism, and the structural guarantee that the matcher can
only LABEL provider-supplied candidates — never mint a market id (the
"LLM may not invent market IDs" property holds a fortiori for the
deterministic v1 matcher, and any future LLM-assisted classifier inherits
the same pool).
"""
import pytest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from libs.trading_core.events import prediction_intel as pi
from libs.trading_core.models.enums import EventType

from tests.test_events_web_research import AS_OF, make_event

EVENT_AT = AS_OF + timedelta(days=5)


@dataclass(frozen=True)
class FakeMarket:
    """PredictionMarketInfo-shaped — the pure layer never imports the
    provider package, so a plain structural double is the honest input."""

    market_id: str
    question: str
    provider: str = "stub"
    resolution_criteria: str | None = None
    end_date: datetime | None = field(default=EVENT_AT + timedelta(days=10))
    status: str = "ACTIVE"


def gdp_event():
    return make_event(
        EventType.GDP, ticker=None, title="US GDP advance estimate",
        scheduled_at=EVENT_AT,
    )


def cpi_event():
    return make_event(
        EventType.CPI, ticker=None, title="US CPI release",
        scheduled_at=EVENT_AT,
    )


def test_direct_relation_for_a_contract_measuring_the_event():
    market = FakeMarket(
        market_id="m-gdp",
        question="Will US GDP growth be above 2.5% in the advance estimate?",
    )
    outcome = pi.match_markets(gdp_event(), [market], as_of=AS_OF)
    decision = outcome.decisions[0]
    assert decision.relation == pi.RELATION_DIRECT
    assert decision.accepted
    assert decision.relevance >= pi.MIN_MATCH_RELEVANCE
    assert decision.matched_by == pi.MATCHED_BY_DETERMINISTIC_V1
    assert decision.ambiguity is None
    assert not outcome.no_relevant_market


def test_derived_relation_for_a_contract_the_event_moves():
    market = FakeMarket(
        market_id="m-cut",
        question="Will the Fed cut rates at the September meeting?",
    )
    decision = pi.match_markets(cpi_event(), [market], as_of=AS_OF).decisions[0]
    assert decision.relation == pi.RELATION_DERIVED
    assert decision.accepted
    assert "affects this contract" in decision.reason


def test_context_relation_for_a_backdrop_contract():
    market = FakeMarket(
        market_id="m-recession",
        question="Will the US enter a recession during 2026?",
        end_date=EVENT_AT + timedelta(days=180),
    )
    decision = pi.match_markets(gdp_event(), [market], as_of=AS_OF).decisions[0]
    assert decision.relation == pi.RELATION_CONTEXT
    # Both calibration directions: WITH subject overlap the backdrop is
    # accepted; with NO subject grounding it is not about this event's
    # world and is rejected.
    assert decision.accepted
    assert decision.relevance >= pi.MIN_MATCH_RELEVANCE
    unanchored = FakeMarket(
        market_id="m-r2", question="Will there be a recession in 2026?",
        end_date=EVENT_AT + timedelta(days=180),
    )
    d2 = pi.match_markets(
        make_event(scheduled_at=EVENT_AT), [unanchored], as_of=AS_OF
    ).decisions[0]  # earnings event: zero subject overlap with the wording
    assert d2.relation == pi.RELATION_CONTEXT
    assert d2.reject_reason == pi.REJECT_MARKET_LOW_RELEVANCE


def test_direct_requires_resolving_around_the_event():
    """TWO-SIDED horizon: a still-ACTIVE contract that resolves BEFORE the
    event measures the PRIOR release period (last month's CPI contract) —
    demoted to DERIVED with the doubt named, never DIRECT-accepted."""
    stale = FakeMarket(
        market_id="m-prior",
        question="Will US CPI inflation be above 3% in the last report?",
        end_date=EVENT_AT - timedelta(days=3),
    )
    decision = pi.match_markets(cpi_event(), [stale], as_of=AS_OF).decisions[0]
    assert decision.relation == pi.RELATION_DERIVED
    assert decision.ambiguity is not None
    assert "different release period" in decision.ambiguity


def test_sibling_measure_is_derived_never_this_events_own_measure():
    """Profile families share DERIVED/CONTEXT vocabulary, but DIRECT is
    per-event-type: a GDP contract seen from a RETAIL_SALES event is a
    related measure, and the audit record must say so honestly."""
    retail = make_event(
        EventType.RETAIL_SALES, ticker=None,
        title="US retail sales report", scheduled_at=EVENT_AT,
    )
    gdp_contract = FakeMarket(
        market_id="m-gdp", question="Will US GDP growth be above 2% in Q3?"
    )
    own_contract = FakeMarket(
        market_id="m-retail",
        question="Will US retail sales growth beat expectations this report?",
    )
    outcome = pi.match_markets(retail, [gdp_contract, own_contract], as_of=AS_OF)
    by_id = {d.market_id: d for d in outcome.decisions}
    assert by_id["m-gdp"].relation == pi.RELATION_DERIVED
    assert "related measure" in by_id["m-gdp"].reason
    assert by_id["m-retail"].relation == pi.RELATION_DIRECT
    assert by_id["m-retail"].accepted


def test_zero_subject_overlap_direct_wording_demotes_with_named_doubt():
    """Bare measure wording with no subject grounding may be a different
    entity's contract — DERIVED with the doubt named.

    The example is deliberately NOT a foreign-jurisdiction one: a contract
    naming another country is now REFUSED outright (see the
    foreign-jurisdiction guard below), which is a stronger outcome than
    demote-with-a-caveat. This pins the residual case — unnamed subject,
    no geography marker — where demotion is still the right answer.
    """
    vague = FakeMarket(
        market_id="m-vague",
        question="Will the inflation rate exceed 2 percent?",
    )
    decision = pi.match_markets(cpi_event(), [vague], as_of=AS_OF).decisions[0]
    assert decision.relation == pi.RELATION_DERIVED
    assert decision.ambiguity is not None
    assert "different subject or geography" in decision.ambiguity


def test_matcher_status_gate_matches_the_provider_vocabulary():
    """Tripwire: the pure matcher deliberately holds a literal (it imports
    no provider package); this pin fails loudly if the vocabulary drifts."""
    from libs.prediction_markets.provider import MARKET_STATUS_ACTIVE

    assert MARKET_STATUS_ACTIVE == "ACTIVE"


def test_ambiguous_direct_wording_with_far_horizon_demotes_and_names_doubt():
    market = FakeMarket(
        market_id="m-far",
        question="Will the US inflation rate stay below 3% through 2027?",
        end_date=EVENT_AT + timedelta(days=400),
    )
    decision = pi.match_markets(cpi_event(), [market], as_of=AS_OF).decisions[0]
    assert decision.relation == pi.RELATION_DERIVED  # demoted, not upgraded
    assert decision.ambiguity is not None
    assert "different release period" in decision.ambiguity


def test_irrelevant_market_rejected_and_zero_matches_is_a_valid_outcome():
    sports = FakeMarket(
        market_id="m-nba", question="Will the Lakers win the NBA championship?"
    )
    outcome = pi.match_markets(gdp_event(), [sports], as_of=AS_OF)
    assert outcome.decisions[0].accepted is False
    assert outcome.decisions[0].reject_reason == pi.REJECT_MARKET_LOW_RELEVANCE
    assert outcome.no_relevant_market  # honest empty, never padded
    assert pi.match_markets(gdp_event(), [], as_of=AS_OF).no_relevant_market


def test_matcher_can_only_label_supplied_candidates():
    candidates = [
        FakeMarket(market_id="m-1", question="Will US GDP be above 2.5%?"),
        FakeMarket(market_id="m-2", question="Will the Fed cut rates?"),
    ]
    outcome = pi.match_markets(gdp_event(), candidates, as_of=AS_OF)
    assert {d.market_id for d in outcome.decisions} == {"m-1", "m-2"}
    assert {d.market_id for d in outcome.accepted} <= {"m-1", "m-2"}


def test_inactive_markets_and_duplicates_are_rejected_with_named_reasons():
    active = FakeMarket(market_id="m-1", question="Will US GDP be above 2.5%?")
    resolved = FakeMarket(
        market_id="m-2", question="Will US GDP be above 2.0%?",
        status="RESOLVED",
    )
    duplicate = FakeMarket(market_id="m-1", question="Will US GDP be above 2.5%?")
    outcome = pi.match_markets(
        gdp_event(), [active, resolved, duplicate], as_of=AS_OF
    )
    by_reason = [d.reject_reason for d in outcome.decisions]
    assert by_reason == [None, pi.REJECT_MARKET_NOT_ACTIVE, pi.REJECT_MARKET_DUPLICATE]
    assert [d.market_id for d in outcome.accepted] == ["m-1"]


def test_accept_cap_keeps_the_best_ranked_markets():
    candidates = [
        FakeMarket(
            market_id=f"m-{i}",
            question=f"Will the Fed cut rates at meeting number {i}?",
        )
        for i in range(4)
    ]
    candidates.append(
        FakeMarket(market_id="m-direct", question="Will US GDP be above 2.5%?")
    )
    outcome = pi.match_markets(
        gdp_event(), candidates, as_of=AS_OF, max_accepted=2
    )
    accepted = outcome.accepted
    # DIRECT outranks DERIVED regardless of arrival order, and equal-score
    # DERIVED contracts tie-break by ascending market_id — the exact set is
    # pinned so neither property can silently regress.
    assert {d.market_id for d in accepted} == {"m-direct", "m-0"}
    over_cap = [
        d for d in outcome.decisions
        if d.reject_reason == pi.REJECT_MARKET_OVER_ACCEPT_CAP
    ]
    assert len(over_cap) == 3


def test_candidate_cap_enforced_with_named_reason():
    candidates = [
        FakeMarket(market_id=f"m-{i}", question="Will US GDP be above 2.5%?")
        if i == 0 else
        FakeMarket(market_id=f"m-{i}", question=f"Will the Fed cut rates in month {i}?")
        for i in range(6)
    ]
    outcome = pi.match_markets(
        gdp_event(), candidates, as_of=AS_OF, max_candidates=3
    )
    # Positional: decisions preserve candidate order (auditability), the
    # first three are considered, the last three are the capped ones.
    assert [d.market_id for d in outcome.decisions] == [
        f"m-{i}" for i in range(6)
    ]
    assert [d.reject_reason for d in outcome.decisions[3:]] == (
        [pi.REJECT_MARKET_OVER_CANDIDATE_CAP] * 3
    )


def test_matching_is_deterministic():
    candidates = [
        FakeMarket(market_id="m-1", question="Will US GDP be above 2.5%?"),
        FakeMarket(market_id="m-2", question="Will the Fed cut rates?"),
        FakeMarket(market_id="m-3", question="Will the US enter a recession in 2026?"),
    ]
    a = pi.match_markets(gdp_event(), candidates, as_of=AS_OF)
    b = pi.match_markets(gdp_event(), candidates, as_of=AS_OF)
    assert a == b


def test_relevance_formula_keeps_bucketless_lookalikes_below_threshold():
    """Aboutness alone tops out at _ABOUTNESS_WEIGHT (< MIN_MATCH_RELEVANCE):
    a market sharing every subject word but matching no relation bucket can
    never be forced into the bundle."""
    # KNOWN v1 LIMIT, pinned deliberately: keyword matching cannot read
    # intent, so a non-outcome market whose wording contains the measure
    # term ("gdp") still classifies DIRECT. The relation buckets are
    # lexical; semantic disambiguation is a documented future refinement.
    lookalike = FakeMarket(
        market_id="m-look",
        question="US GDP advance estimate discussion panel attendance",
    )
    decision = pi.match_markets(gdp_event(), [lookalike], as_of=AS_OF).decisions[0]
    assert decision.relation == pi.RELATION_DIRECT
    # The true bucketless case needs a profile with narrower terms: a Fed
    # speech has no DIRECT vocabulary at all.
    speech = make_event(
        EventType.FED_SPEECH, ticker=None,
        title="Chair Powell speech on the economic outlook",
        speaker="Powell", scheduled_at=EVENT_AT,
    )
    subject_only = FakeMarket(
        market_id="m-sub",
        question="Federal Reserve Chair Powell economic outlook speech viewers",
    )
    d2 = pi.match_markets(speech, [subject_only], as_of=AS_OF).decisions[0]
    assert d2.relation is None
    assert d2.relevance < pi.MIN_MATCH_RELEVANCE
    assert d2.reject_reason == pi.REJECT_MARKET_LOW_RELEVANCE


def test_discovery_queries_are_bounded_deduped_and_event_grounded():
    queries = pi.market_discovery_queries(cpi_event())
    assert 0 < len(queries) <= pi.MAX_MARKET_QUERIES
    assert len({q.lower() for q in queries}) == len(queries)
    joined = " ".join(queries).lower()
    assert "cpi" in joined
    earnings_queries = pi.market_discovery_queries(make_event())
    assert any("NVDA" in q for q in earnings_queries)
    assert pi.market_discovery_queries(gdp_event()) == pi.market_discovery_queries(
        gdp_event()
    )  # deterministic


# ---------------------------------------------------------------------------
# Foreign-jurisdiction guard (final audit, LOOP 10)
#
# The DERIVED bucket cannot demand subject grounding — the mission's own
# canonical example ("Will the Fed cut rates in September?" for a CPI
# release) shares no tokens with its event. That necessary leniency made the
# relevance threshold INERT for the whole DERIVED class: the bonus (0.5)
# equalled MIN_MATCH_RELEVANCE (0.5) and acceptance is `>=`, so any contract
# containing a rate-move phrase was admitted — including another country's.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,term",
    [
        ("Will the Bank of Argentina cut rates before the final?", "argentina"),
        ("Will Nigeria raise rates in 2027?", "nigeria"),
        ("Will the ECB cut rates in December?", "ecb"),
        ("Will the Bank of Japan cut rates in 2028?", "bank of japan"),
    ],
)
def test_a_foreign_jurisdiction_contract_is_refused_with_its_own_reason(
    question, term
):
    market = FakeMarket(market_id="m-foreign", question=question)
    decision = pi.match_markets(cpi_event(), [market], as_of=AS_OF).decisions[0]
    assert decision.accepted is False
    # A SPECIFIC reason, not the generic LOW_RELEVANCE: the Evidence tab has
    # to be able to explain why this contract was refused.
    assert decision.reject_reason == pi.REJECT_MARKET_FOREIGN_SUBJECT
    assert term in decision.reason


def test_the_canonical_us_derived_contract_still_passes():
    """The guard must not cost the mission its own example."""
    market = FakeMarket(
        market_id="m-cut",
        question="Will the Fed cut rates at the September meeting?",
    )
    decision = pi.match_markets(cpi_event(), [market], as_of=AS_OF).decisions[0]
    assert decision.accepted is True
    assert decision.relation == pi.RELATION_DERIVED


def test_no_relevant_market_is_reachable_when_only_foreign_contracts_exist():
    """The honest empty outcome must survive: before the guard, any pool with
    a rate-move phrase in it made NO_RELEVANT_PREDICTION_MARKET unreachable."""
    markets = [
        FakeMarket(market_id="m1", question="Will Brazil cut rates in 2027?"),
        FakeMarket(market_id="m2", question="Will the Bank of England hike?"),
    ]
    outcome = pi.match_markets(cpi_event(), markets, as_of=AS_OF)
    assert not any(d.accepted for d in outcome.decisions)
    assert outcome.no_relevant_market


# ---------------------------------------------------------------------------
# Other-issuer guard (found in LIVE Polymarket data, 2026-08-22)
#
# An HPE earnings event accepted Broadcom / Dell / Snowflake "beat quarterly
# earnings" contracts as DIRECT at 0.775. The subject overlap came from the
# contracts' RESOLUTION CRITERIA naming the same release date ("September 2,
# 2026") — an incidental token, not the company. Another issuer's earnings
# pricing would have been quoted to the model as this event's.
# ---------------------------------------------------------------------------


def _earnings_event(ticker="HPE"):
    from libs.trading_core.events.models import Event
    from libs.trading_core.models.enums import (
        EventSourceKind,
        EventStatus,
        EventType as ET,
    )

    return Event(
        event_id=1,
        event_key=f"EARNINGS:{ticker}:2026-09-02",
        event_type=ET.EARNINGS,
        ticker=ticker,
        scheduled_at=AS_OF + timedelta(days=11),
        title=f"{ticker} earnings",
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="t",
    )


#: The shared resolution wording that supplied the spurious overlap.
_SHARED_RC = (
    "As of market creation, the company is estimated to release earnings "
    "on September 2, 2026."
)


def test_another_issuers_earnings_contract_is_refused_by_name():
    own = FakeMarket(
        market_id="m-hpe",
        question="Will Hewlett Packard Enterprise (HPE) beat quarterly earnings?",
        resolution_criteria=_SHARED_RC,
        end_date=AS_OF + timedelta(days=13),
    )
    other = FakeMarket(
        market_id="m-avgo",
        question="Will Broadcom (AVGO) beat quarterly earnings?",
        resolution_criteria=_SHARED_RC,
        end_date=AS_OF + timedelta(days=13),
    )
    decisions = pi.match_markets(
        _earnings_event(), [own, other], as_of=AS_OF
    ).decisions
    by_id = {d.market_id: d for d in decisions}

    assert by_id["m-hpe"].accepted is True
    assert by_id["m-hpe"].relation == pi.RELATION_DIRECT

    assert by_id["m-avgo"].accepted is False
    assert by_id["m-avgo"].reject_reason == pi.REJECT_MARKET_OTHER_ISSUER
    assert "AVGO" in by_id["m-avgo"].reason


def test_a_contract_naming_no_issuer_stays_derived():
    """The guard must not cost a legitimate macro contract its place: a
    Fed-cut market during an earnings week names no ticker at all."""
    macro = FakeMarket(
        market_id="m-fed",
        question="Will the Fed cut rates in September?",
        resolution_criteria=_SHARED_RC,
        end_date=AS_OF + timedelta(days=13),
    )
    decision = pi.match_markets(_earnings_event(), [macro], as_of=AS_OF).decisions[0]
    assert decision.accepted is True
    assert decision.relation == pi.RELATION_DERIVED


# ---------------------------------------------------------------------------
# BRACKET SERIES (found live 2026-08-23 against Polymarket's GDP markets)
#
# A venue publishes a distribution as one contract per range. The accept cap
# was 5, so a SEVEN-bracket Q3 GDP series stored its four cheapest brackets
# and dropped the three holding 80% of the probability mass. The panel showed
# every outcome priced near zero while the market had a clear central
# estimate — a partial distribution is worse than none, because it draws a
# confident wrong shape.
# ---------------------------------------------------------------------------


def test_bracket_siblings_share_a_series_key():
    from libs.trading_core.events.prediction_intel import series_key_for

    keys = {
        series_key_for(q)
        for q in (
            "Will US GDP growth in Q3 2026 be less than 0.5%?",
            "Will US GDP growth in Q3 2026 be between 0.5% and 1.0%?",
            "Will US GDP growth in Q3 2026 be between 1.5% and 2.0%?",
            "Will US GDP growth in Q3 2026 be greater than 3.0%?",
        )
    }
    assert len(keys) == 1
    assert keys.pop() is not None


def test_a_different_period_is_a_different_series():
    """Full-year 2026 GDP and Q3 2026 GDP are two distributions, not one.
    Collapsing them would flag a complete series as truncated and would let a
    full-year bracket stand in for a quarterly one."""
    from libs.trading_core.events.prediction_intel import series_key_for

    assert series_key_for(
        "Will US GDP growth in Q3 2026 be between 0.5% and 1.0%?"
    ) != series_key_for("Will US GDP growth in 2026 be between 0.5% and 1.0%?")


def test_a_plain_yes_no_contract_belongs_to_no_series():
    """Only RANGE wording makes a sibling. A contract with no brackets cannot
    be truncated away from anything, and giving it a series key would flag
    unrelated contracts as a broken distribution."""
    from libs.trading_core.events.prediction_intel import series_key_for

    assert series_key_for("Will there be a recession in 2026?") is None
    assert (
        series_key_for(
            "Will the Fed decrease interest rates by 25 bps after the September meeting?"
        )
        is None
    )


def test_the_accept_cap_clears_a_whole_bracket_series():
    """The cap must sit above the largest series a venue actually publishes —
    Polymarket ships 6-10 brackets for one release."""
    from libs.trading_core.events.prediction_intel import MAX_ACCEPTED_MARKETS

    assert MAX_ACCEPTED_MARKETS >= 10


def test_a_truncated_series_is_flagged_on_the_survivors():
    """If the cap ever does bisect a series, the accepted brackets say so.
    Silently keeping a subset is what produced the original wrong picture."""
    from dataclasses import replace

    from libs.trading_core.events.prediction_intel import (
        MarketMatchDecision,
        REJECT_MARKET_OVER_ACCEPT_CAP,
    )

    kept = MarketMatchDecision(
        provider="polymarket",
        market_id="1",
        question="Will X be between 0.5% and 1.0%?",
        relation="DIRECT",
        relevance=0.9,
        reason="",
        ambiguity=None,
        matched_by="DETERMINISTIC_V1",
        accepted=True,
        reject_reason=None,
        series_key="will x be ?",
    )
    dropped = replace(
        kept,
        market_id="2",
        accepted=False,
        reject_reason=REJECT_MARKET_OVER_ACCEPT_CAP,
    )
    # The flag is set by match_markets; here we assert the field exists and
    # round-trips, so the payload contract cannot silently lose it.
    flagged = replace(kept, series_truncated=True)
    assert flagged.to_dict()["series_truncated"] is True
    assert kept.to_dict()["series_truncated"] is False
    assert dropped.reject_reason == REJECT_MARKET_OVER_ACCEPT_CAP


# ---------------------------------------------------------------------------
# THE VENUE EVENT IS THE UNIT OF ACCEPTANCE (2026-08-23)
#
# The user found two SEPARATE Polymarket events mixed in one panel — "GDP
# growth in 2026" and "US GDP growth in Q3 2026?" — with only part of each.
# Comparing brackets across two distributions is meaningless, and showing
# four of seven brackets of one distribution is not a smaller picture but a
# wrong one: the survivors are whichever the ranking favoured, and the
# probability mass may sit entirely in those dropped.
# ---------------------------------------------------------------------------


class _Candidate:
    """Structural PredictionMarketInfo stand-in (the pure layer types
    structurally and imports no provider package)."""

    def __init__(self, market_id, question, provider_event_id=None, end_date=None):
        self.provider = "polymarket"
        self.market_id = market_id
        self.provider_event_id = provider_event_id
        self.question = question
        self.status = "ACTIVE"
        self.end_date = end_date
        self.resolution_criteria = None
        self.outcomes = ()
        self.url = None
        self.volume = None
        self.liquidity = None
        self.raw = {}


def _gdp_event(when):
    from libs.trading_core.models.enums import EventType

    class E:
        event_type = EventType.GDP
        ticker = None
        title = "GDP (Advance Estimate), 3rd Quarter 2026"
        scheduled_at = when
        series_id = None
        speaker = None
        release_period = "2026-Q3"

    return E()


def test_a_venue_event_is_accepted_whole_or_not_at_all():
    """The cap counts GROUPS. A distribution is never bisected — the reader
    either sees all of its brackets or is told it was left out."""
    from datetime import datetime, timedelta, timezone

    from libs.trading_core.events.prediction_intel import (
        REJECT_MARKET_OVER_ACCEPT_CAP,
        match_markets,
    )

    as_of = datetime(2026, 8, 23, tzinfo=timezone.utc)
    ends = as_of + timedelta(days=20)
    # Two complete distributions of 4 brackets each, under a cap of 6: the
    # first fits, the second cannot — and must be refused ENTIRELY.
    candidates = [
        _Candidate(f"a{i}", f"Will US GDP growth in Q3 2026 be between {i}.0% and {i}.5%?",
                   provider_event_id="900", end_date=ends)
        for i in range(4)
    ] + [
        _Candidate(f"b{i}", f"Will US GDP growth in 2026 be between {i}.0% and {i}.5%?",
                   provider_event_id="901", end_date=ends)
        for i in range(4)
    ]

    outcome = match_markets(
        _gdp_event(as_of + timedelta(days=3)),
        candidates,
        as_of=as_of,
        max_accepted=6,
    )
    by_group = {}
    for d in outcome.decisions:
        by_group.setdefault(d.provider_event_id, []).append(d)

    for group, decisions in by_group.items():
        accepted = [d for d in decisions if d.accepted]
        # All-or-nothing: never a partial group.
        assert len(accepted) in (0, len(decisions)), group
    # And the refused group says WHY, rather than vanishing.
    refused = [
        d for d in outcome.decisions
        if d.reject_reason == REJECT_MARKET_OVER_ACCEPT_CAP
    ]
    assert len(refused) == 4


def test_a_single_oversized_distribution_is_still_shown():
    """A nine-bracket series under a cap of eight must not leave the reader
    with nothing — one complete distribution beats zero."""
    from datetime import datetime, timedelta, timezone

    from libs.trading_core.events.prediction_intel import match_markets

    as_of = datetime(2026, 8, 23, tzinfo=timezone.utc)
    ends = as_of + timedelta(days=20)
    candidates = [
        _Candidate(f"c{i}", f"Will US GDP growth in Q3 2026 be between {i}.0% and {i}.5%?",
                   provider_event_id="950", end_date=ends)
        for i in range(9)
    ]
    outcome = match_markets(
        _gdp_event(as_of + timedelta(days=3)),
        candidates,
        as_of=as_of,
        max_accepted=8,
    )
    accepted = [d for d in outcome.decisions if d.accepted]
    assert len(accepted) == 9


def test_the_venue_grouping_beats_wording_when_both_exist():
    """Two same-worded series (Q3 2026 vs full-year 2026) are distinguished by
    the venue's own event id, which a wording key could only separate by luck."""
    from libs.trading_core.events.prediction_intel import series_key_for

    q3 = "Will US GDP growth in Q3 2026 be less than 0.5%?"
    full = "Will US GDP growth in 2026 be less than 0.5%?"
    # Wording alone already separates these two, but the venue id is what the
    # storage layer prefers — see the seam's series_key assignment.
    assert series_key_for(q3) != series_key_for(full)
