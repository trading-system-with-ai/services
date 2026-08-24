"""Prediction-market provider interface (Catalyst research upgrade; plan §3).

A SEPARATE provider registry cloning the market-data/calendar/LLM contract —
``get_provider(name)``, :class:`ProviderNotConfigured` on an empty name,
``ValueError`` on an unknown one, no default, no cross-provider fallback.
Provider-independent by design: a future ``KalshiProvider`` must fit these
models without touching Event Analysis, the Evidence Bundle, matching logic
or the UI (plan Phase 18); no Polymarket-specific field may leak into them.

READ ONLY — THE WHOLE SUBSYSTEM. There is no place-order, sign-order, wallet,
or credential surface here and none may ever be added: prediction markets are
an OBSERVED layer of market expectations, research input only. Enforced
structurally by tests/test_research_safety_adversarial.py (a word-ban AST
sweep over this package plus the import-graph check), not just by this
docstring.

A MARKET PRICE IS NOT A CLEAN PROBABILITY (plan §3 interpretation rule).
Every price this package carries is "what the market currently charges for
the contract" — model-facing and UI language must say *market-implied
probability* / *prediction-market pricing*, never *actual probability*. A 70c
contract with no depth is a different claim from a 70c contract in a deep
book, which is why the snapshot model carries spread/volume/liquidity beside
the price instead of reducing to one number.

NO FABRICATION (§44 rule 18): a field the provider did not state is ``None``,
never 0 — a market with no reported liquidity is not a market with zero
liquidity.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol

from libs.market_data.provider import (  # noqa: F401 — re-exported on purpose:
    # ONE failure taxonomy platform-wide (audit §6).
    CapabilityNotAvailable,
    MarketDataError,
    ProviderNotConfigured,
)

#: The message every unconfigured-prediction-market path reports verbatim.
#: Keyless does not mean default-on: even a free public API is an outbound
#: network dependency the operator must consciously enable (plan §8).
PREDICTION_MARKETS_NOT_CONFIGURED_MESSAGE = (
    "prediction markets provider is not configured — set "
    "PREDICTION_MARKETS_PROVIDER (public read-only, no trading credentials); "
    "event research degrades, nothing else"
)

#: The FIXED capability key set every prediction-market provider reports, in
#: the platform's tri-state shape (audit §6): ``True`` / ``False`` (proven
#: absence) / an error string (fault, availability unknown).
CAPABILITY_KEYS: tuple[str, ...] = (
    "market_search",    # free-text market/event discovery
    "market_metadata",  # question, outcomes, resolution criteria, status
    "market_snapshot",  # current pricing: bid/ask/mid/spread/last
    "price_history",    # dated price points per outcome
)

#: Every capability False — the base a provider overrides for what it serves.
NO_CAPABILITIES: dict[str, bool | str] = {k: False for k in CAPABILITY_KEYS}

#: ``PredictionMarketInfo.status`` vocabulary. Lives in code, not a DB CHECK
#: (migration-017 lesson). UNKNOWN is an honest verdict for a provider payload
#: that states no lifecycle at all — never guessed into ACTIVE.
MARKET_STATUS_ACTIVE = "ACTIVE"
MARKET_STATUS_CLOSED = "CLOSED"
MARKET_STATUS_RESOLVED = "RESOLVED"
MARKET_STATUS_UNKNOWN = "UNKNOWN"
MARKET_STATUSES: tuple[str, ...] = (
    MARKET_STATUS_ACTIVE,
    MARKET_STATUS_CLOSED,
    MARKET_STATUS_RESOLVED,
    MARKET_STATUS_UNKNOWN,
)


class PredictionMarketError(MarketDataError):
    """A prediction-market request could not be answered honestly.

    A subclass of :class:`libs.market_data.provider.MarketDataError` so the
    gateway seam catches ONE exception type for "a data source could not
    answer" — prediction-market failure is research degradation, never a
    system failure (plan Phase 11).
    """


def blank_capabilities() -> dict[str, bool | str]:
    """A fresh all-False capability report (never share the module dict)."""
    return dict(NO_CAPABILITIES)


@dataclass(frozen=True)
class MarketOutcome:
    """One outcome leg of a market, as the provider states it.

    ``price`` is the provider's current price for the leg (their headline
    number, typically last/mid) or ``None`` when the payload omitted it —
    never 0, which would read as "the market says impossible".
    """

    name: str
    price: float | None


@dataclass(frozen=True)
class PredictionMarketInfo:
    """One market's METADATA, provider-independent (discovery/metadata shape).

    ``market_id`` is the provider's own identifier and is only meaningful
    beside ``provider`` — the storage key is ``(provider, market_id)``, so a
    Kalshi ticker can never collide with a Polymarket condition id.

    ``provider_event_id`` is the PROVIDER'S grouping of related markets (a
    Polymarket "event", a Kalshi "series") — it is NOT this platform's
    ``events.id`` and nothing may join the two directly; the explicit
    event↔market matching layer (plan Phase 4) owns that association.

    ``resolution_criteria`` is carried because the contract's exact wording
    decides what the price means — "GDP above 2.5%" and "GDP above 2.5% in
    the advance estimate" are different claims. ``None`` = the provider did
    not state it, an absence the matching layer must weigh, not paper over.

    ``raw`` keeps the provider's own payload for provenance/debugging; it is
    never a computation input for the pure analysis layer.
    """

    provider: str
    market_id: str
    provider_event_id: str | None
    question: str
    url: str | None
    outcomes: tuple[MarketOutcome, ...]
    resolution_criteria: str | None
    end_date: datetime | None
    status: str
    volume: float | None
    liquidity: float | None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MarketSnapshot:
    """One market's PRICING at one observed instant (plan §3 normalized model).

    ``observed_at`` is when THIS platform saw these numbers — the snapshot's
    point-in-time identity, which the as-of gate filters on. It is never the
    provider's own quote timestamp (providers rarely state one) and never a
    substitute for it.

    Every liquidity-adjacent field (``volume``, ``liquidity``,
    ``open_interest``) is ``None`` when unreported (§44 rule 18). The
    interpretation layer treats a priced-but-thin market with less confidence
    than a deep one, so zeroing an absent liquidity here would silently
    upgrade the thinnest markets to "confidently priced at zero depth".

    ``outcome_prices`` maps outcome name -> price for every outcome the
    provider priced; an unpriced outcome is ABSENT from the mapping, never
    present as 0.
    """

    provider: str
    market_id: str
    observed_at: datetime
    outcome_prices: Mapping[str, float]
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    spread: float | None
    last_trade_price: float | None
    volume: float | None
    liquidity: float | None
    open_interest: float | None


@dataclass(frozen=True)
class PricePoint:
    """One dated price for ONE outcome of one market (history series).

    Points are provider truth verbatim — no interpolation, no resampling
    (plan §3: "no invented interpolation"). Gaps in the series are gaps in
    the data.
    """

    ts: datetime
    price: float


class PredictionMarketProvider(Protocol):
    """Structural interface every prediction-market provider must satisfy.

    READ-ONLY BY CONSTRUCTION: these four methods are the whole surface, and
    none of them mutates anything anywhere. There is deliberately no
    ``place_order``/``sign``/``wallet`` method to forget to guard.
    """

    name: str

    def capabilities(self) -> dict[str, bool | str]:
        """Tri-state report over :data:`CAPABILITY_KEYS` (audit §6).

        Never raises: a probe that raised would take the research refresh
        down with it.
        """
        ...

    def search_markets(
        self,
        query: str,
        *,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[PredictionMarketInfo]:
        """Markets matching ``query`` — the deterministic candidate pool for
        the matching layer (plan Phase 4). Fewer than ``limit`` results is a
        normal, honest answer; zero is the common and valid
        NO_RELEVANT_MARKET path, never padded."""
        ...

    def get_market(self, market_id: str) -> PredictionMarketInfo:
        """Metadata for one market. Raises :class:`PredictionMarketError`
        when the id is unknown — an unknown id is a fault to surface, never
        an empty shell to fabricate."""
        ...

    def get_market_snapshot(self, market_id: str) -> MarketSnapshot:
        """Current pricing for one market. ``observed_at`` is the instant
        THIS platform observed it (real adapters stamp the fetch instant;
        the deterministic stub stamps its anchor) — see
        :class:`MarketSnapshot`."""
        ...

    def get_price_history(
        self,
        market_id: str,
        *,
        outcome: str,
        start: datetime,
        end: datetime,
    ) -> list[PricePoint]:
        """Dated prices for one outcome in ``[start, end]``, ascending.

        An empty list means the provider has no history there — an honest
        absence the features layer reports as such (plan §3: preserve
        ``observation_count``/``history_start``/``history_end``)."""
        ...
