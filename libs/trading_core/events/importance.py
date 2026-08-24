"""Transparent event importance model (event spec §12, §13).

§13 is explicit: *"Do not create a mysterious LLM-generated importance
score. Quantitative components must be identifiable."* So this module is a
plain additive model over named integer components — every point in the
final score is attributable to a labelled input, and the UI's ⓘ tooltip
renders :attr:`ImportanceResult.components` verbatim (§90). No LLM, no
learned weights, no hidden normalisation.

Two components exist today:

- ``event_type`` — the systemic weight of the event class itself (an FOMC
  decision moves the whole tape; a JOLTS print rarely does).
- ``relevance`` — how close the event is to the user's own money (§12:
  POSITION > TRADING_POOL > WATCHLIST > MARKET_WIDE > OTHER).

Later phases add ``implied_move`` (§36), ``news_intensity`` (§25) and
``historical_reaction`` (§17) components. :func:`score_importance` therefore
accepts an ``extra_components`` mapping and folds it into the same dict
rather than requiring a signature change — the dict is the contract, and it
stays additive so the arithmetic remains checkable by eye.

Pure stdlib, no I/O, and no import of the provider or persistence layers
(audit §7.4).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from libs.trading_core.events.taxonomy import MARKET_WIDE_EVENT_TYPES
from libs.trading_core.models.enums import EventType

__all__ = [
    "BASE_IMPORTANCE",
    "CHAIR_SPEAKER_BONUS",
    "ImportanceResult",
    "RELEVANCE_TIERS",
    "RELEVANCE_WEIGHT",
    "IMPORTANCE_MODEL_VERSION",
    "default_relevance_tier",
    "relevance_rank",
    "score_importance",
]

#: Bump when a weight or component changes so persisted scores stay
#: interpretable (same discipline as ``TradeabilityParams.version``).
IMPORTANCE_MODEL_VERSION = "event-importance-v1"

#: Base weight per event type — the systemic-impact half of §13. These are
#: research parameters, not universal financial truths.
BASE_IMPORTANCE: Mapping[EventType, int] = {
    EventType.FOMC_DECISION: 90,
    EventType.CPI: 80,
    EventType.EMPLOYMENT_REPORT: 80,
    EventType.PCE: 60,
    EventType.EARNINGS: 60,
    EventType.GDP: 50,
    EventType.FOMC_MINUTES: 40,
    EventType.PPI: 40,
    EventType.FOMC_PRESS_CONFERENCE: 40,
    EventType.FOMC_MEETING: 30,
    EventType.ISM: 30,
    EventType.RETAIL_SALES: 30,
    EventType.JOLTS: 30,
    EventType.CONSUMER_SENTIMENT: 30,
    EventType.FED_SPEECH: 20,
    EventType.CORPORATE_EVENT: 20,
    EventType.FED_BOARD_EVENT: 15,
    EventType.MARKET_HOLIDAY: 5,
}

#: A speech by the Chair or Vice Chair carries policy weight a regional
#: president's does not (§9 "do not label every Fed event as an FOMC
#: meeting" — the converse also holds: do not flatten every speaker).
CHAIR_SPEAKER_BONUS = 20
_CHAIR_MARKERS = ("chair", "powell", "vice chair", "vice-chair")

#: §12 relevance ladder, highest first. The tier name travels into the API
#: payload and drives the UI grouping order.
RELEVANCE_TIERS: tuple[str, ...] = (
    "POSITION",
    "TRADING_POOL",
    "WATCHLIST",
    "MARKET_WIDE",
    "OTHER",
)

#: Points added per tier. MARKET_WIDE scores 0 *on top of* an already-high
#: macro base — an FOMC decision is important because of what it is, not
#: because the user happens to hold something.
RELEVANCE_WEIGHT: Mapping[str, int] = {
    "POSITION": 30,
    "TRADING_POOL": 20,
    "WATCHLIST": 10,
    "MARKET_WIDE": 0,
    "OTHER": 0,
}


@dataclass(frozen=True)
class ImportanceResult:
    """A score plus the arithmetic that produced it (§13, §90).

    ``sum(components.values())`` equals the score *before* clamping; ``score``
    is that sum clamped to ``[0, 100]``. Both are exposed so the UI can show
    "90 + 30 = 120 -> 100" honestly rather than presenting 100 as if the
    components added to it.
    """

    score: int
    components: Mapping[str, int]
    relevance_tier: str
    version: str = IMPORTANCE_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", dict(self.components))

    @property
    def raw_total(self) -> int:
        """Unclamped component sum — what the tooltip's arithmetic shows."""
        return sum(self.components.values())

    @property
    def was_clamped(self) -> bool:
        return self.raw_total != self.score


def relevance_rank(tier: str) -> int:
    """Sort key for the §12 ladder; lower sorts first (POSITION = 0).

    An unknown tier sorts last rather than raising — a relevance label the
    API does not recognise must not break the ordering of everything else.
    """
    try:
        return RELEVANCE_TIERS.index(tier)
    except ValueError:
        return len(RELEVANCE_TIERS)


def default_relevance_tier(event_type: EventType) -> str:
    """Tier for an event with no user linkage.

    Macro and Fed events are MARKET_WIDE by construction (they have no
    ticker to be relevant *to*); anything else with no position, pool or
    watchlist membership is OTHER.
    """
    return "MARKET_WIDE" if event_type in MARKET_WIDE_EVENT_TYPES else "OTHER"


def _is_senior_speaker(speaker: str | None) -> bool:
    if not speaker:
        return False
    lowered = speaker.lower()
    return any(marker in lowered for marker in _CHAIR_MARKERS)


def score_importance(
    event_type: EventType,
    *,
    relevance_tier: str | None = None,
    speaker: str | None = None,
    extra_components: Mapping[str, int] | None = None,
) -> ImportanceResult:
    """Score an event 0-100 with an identifiable component breakdown (§13).

    ``relevance_tier`` is supplied by the gateway, which alone knows whether
    the ticker is held / pooled / watched; omitting it falls back to
    :func:`default_relevance_tier`. ``extra_components`` is the extension
    seam for later phases (implied move, news intensity) — its entries are
    merged into the same additive dict.

    Raises ``ValueError`` on an unknown ``relevance_tier``: a typo'd tier
    would silently score 0 and quietly demote a position-linked event.
    """
    tier = relevance_tier if relevance_tier is not None else default_relevance_tier(event_type)
    if tier not in RELEVANCE_WEIGHT:
        raise ValueError(f"unknown relevance tier {tier!r}; expected one of {RELEVANCE_TIERS}")

    components: dict[str, int] = {
        "event_type": BASE_IMPORTANCE.get(event_type, 0),
        "relevance": RELEVANCE_WEIGHT[tier],
    }
    if event_type is EventType.FED_SPEECH and _is_senior_speaker(speaker):
        components["speaker_seniority"] = CHAIR_SPEAKER_BONUS
    for name, value in (extra_components or {}).items():
        components[name] = components.get(name, 0) + int(value)

    total = sum(components.values())
    return ImportanceResult(
        score=max(0, min(100, total)),
        components=components,
        relevance_tier=tier,
    )
