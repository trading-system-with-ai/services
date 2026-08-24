"""Domain enums shared by backtest and live code (single source of truth)."""
from enum import StrEnum


class ActorType(StrEnum):
    """Who performed an action. Central to the authorization model:
    only USER actors may mutate Watchlist / Trading Pool membership."""

    USER = "USER"
    SYSTEM = "SYSTEM"
    LLM = "LLM"


class AuditAction(StrEnum):
    WATCHLIST_ADD = "WATCHLIST_ADD"
    WATCHLIST_REMOVE = "WATCHLIST_REMOVE"
    TRADING_POOL_ADD = "TRADING_POOL_ADD"
    TRADING_POOL_REMOVE = "TRADING_POOL_REMOVE"
    TRADING_POOL_TOGGLE = "TRADING_POOL_TOGGLE"
    RECOMMENDATION_CREATED = "RECOMMENDATION_CREATED"
    RECOMMENDATION_DISMISSED = "RECOMMENDATION_DISMISSED"
    RECOMMENDATION_PROMOTED = "RECOMMENDATION_PROMOTED"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    RISK_DECISION = "RISK_DECISION"
    ORDER_REQUESTED = "ORDER_REQUESTED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    EXIT_GENERATED = "EXIT_GENERATED"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    TRADING_PAUSED = "TRADING_PAUSED"
    TRADING_RESUMED = "TRADING_RESUMED"
    DATA_BACKFILL = "DATA_BACKFILL"
    BACKTEST_STARTED = "BACKTEST_STARTED"
    BACKTEST_COMPLETED = "BACKTEST_COMPLETED"
    BACKTEST_FAILED = "BACKTEST_FAILED"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    NEWS_INGESTED = "NEWS_INGESTED"
    PLAN_GENERATED = "PLAN_GENERATED"
    PLAN_APPLIED = "PLAN_APPLIED"
    PLAN_SUPERSEDED = "PLAN_SUPERSEDED"
    PLAN_CANCELLED = "PLAN_CANCELLED"
    # Catalyst & Event Intelligence (prompts/event_analy_system.md, Phase B)
    CALENDAR_INGESTED = "CALENDAR_INGESTED"
    EVENT_DISCOVERED = "EVENT_DISCOVERED"
    EVENT_UPDATED = "EVENT_UPDATED"
    EVENT_APPROACHING = "EVENT_APPROACHING"
    EVENT_ANALYSIS_GENERATED = "EVENT_ANALYSIS_GENERATED"
    # Catalyst research upgrade (plan §5, Phase 13). Each is one USER-
    # initiated, externally-billed refresh — not a per-HTTP-GET trace, which
    # would drown the log in provenance nobody reads.
    EVENT_SEARCH_RUN = "EVENT_SEARCH_RUN"
    PREDICTION_MARKET_FETCHED = "PREDICTION_MARKET_FETCHED"


class PlanStatus(StrEnum):
    """Research trade plan lifecycle (upgrade 2026-08-12 §40).

    GENERATED -> (REVIEWED) -> ACTIVE via the user's explicit Apply (§19);
    a newly applied plan SUPERSEDES the previous ACTIVE plan for the same
    symbol. Applying never places an order (§19: research approval is not
    execution). DRAFT and EXPIRED are reserved for later phases."""

    DRAFT = "DRAFT"
    GENERATED = "GENERATED"
    REVIEWED = "REVIEWED"
    APPLIED = "APPLIED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class MarketRegime(StrEnum):
    STRONG_BULL = "STRONG_BULL"
    MILD_BULL = "MILD_BULL"
    NEUTRAL_RANGE = "NEUTRAL_RANGE"
    MILD_BEAR = "MILD_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    TRANSITION = "TRANSITION"  # defaults to NO TRADE


class DirectionalBias(StrEnum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class DirectionalEdgeClass(StrEnum):
    """Directional Edge classification bands (upgrade 2026-08-12 §7).

    Purely a labeling layer over ``directional_edge`` — it never gates a
    trade by itself (Tradeability and Risk do that). Band thresholds live in
    :class:`libs.trading_core.signals.EdgeClassificationParams` and are
    research parameters, not universal financial truths."""

    STRONG_BULL = "STRONG_BULL"
    MODERATE_BULL = "MODERATE_BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"
    WEAK_BEAR = "WEAK_BEAR"
    MODERATE_BEAR = "MODERATE_BEAR"
    STRONG_BEAR = "STRONG_BEAR"


class IVRegime(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TradeabilityState(StrEnum):
    """Layer-2 verdict: is this symbol's ENVIRONMENT tradeable right now
    (upgrade 2026-08-12 §9)? Independent of direction — STRONG_BULL with
    BLOCKED is a valid, explainable state (§10), never a contradiction."""

    TRADEABLE = "TRADEABLE"
    CONDITIONAL = "CONDITIONAL"
    BLOCKED = "BLOCKED"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"


class InstrumentType(StrEnum):
    """Account-permission-gated instruments (§5/§8).

    Long-only singles are live; the DEFINED-RISK SPREADS (execution-chains
    roadmap Phase 1) exist at the research layer — the §8 matrix emits them
    when ``AccountPermissions.defined_risk_spreads`` is on — while their
    execution/positions/backtest chains are under construction, so the live
    §10 chain refuses them with an honest 422 until the full chain lands.
    No short stock, no naked shorts (§5)."""

    LONG_STOCK = "LONG_STOCK"
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    # Phase 2 (roadmap): COLLATERALIZED short premium — the short call is
    # backed by 100 shares/contract, the short put by locked cash. These are
    # income overlays, not §8 directional entries; naked variants have no
    # instrument and never will (broker refusal + §4 charter).
    COVERED_CALL = "COVERED_CALL"
    CASH_SECURED_PUT = "CASH_SECURED_PUT"
    # Phase 3 (roadmap): margin-backed short stock — enabled ONLY in the §8
    # cells where premium is unbuyably expensive (bear + EXTREME/HIGH vol);
    # puts remain the preferred bear expression (defined risk first).
    SHORT_STOCK = "SHORT_STOCK"
    NO_TRADE = "NO_TRADE"


class OpportunityStatus(StrEnum):
    NO_SIGNAL = "NO_SIGNAL"
    WATCH = "WATCH"
    SETUP_FORMING = "SETUP_FORMING"
    ENTRY_READY = "ENTRY_READY"
    DATA_ISSUE = "DATA_ISSUE"
    BACKTEST_FAILED = "BACKTEST_FAILED"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    NEWS_INGESTED = "NEWS_INGESTED"


class RiskDecision(StrEnum):
    APPROVE = "APPROVE"
    APPROVE_WITH_RESIZE = "APPROVE_WITH_RESIZE"
    REJECT = "REJECT"
    PAUSE_STRATEGY = "PAUSE_STRATEGY"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


class EventType(StrEnum):
    """Typed catalyst/event taxonomy (event spec §5). Never a free-form
    string: the UI, importance model and previous-event matcher switch on
    these members. CORPORATE_EVENT covers dividends/splits/IPO-lockups and
    other scheduled non-earnings corporate items."""

    EARNINGS = "EARNINGS"
    CPI = "CPI"
    PPI = "PPI"
    PCE = "PCE"
    GDP = "GDP"
    EMPLOYMENT_REPORT = "EMPLOYMENT_REPORT"
    JOLTS = "JOLTS"
    RETAIL_SALES = "RETAIL_SALES"
    ISM = "ISM"
    CONSUMER_SENTIMENT = "CONSUMER_SENTIMENT"
    FOMC_MEETING = "FOMC_MEETING"
    FOMC_DECISION = "FOMC_DECISION"
    FOMC_PRESS_CONFERENCE = "FOMC_PRESS_CONFERENCE"
    FOMC_MINUTES = "FOMC_MINUTES"
    FED_SPEECH = "FED_SPEECH"
    FED_BOARD_EVENT = "FED_BOARD_EVENT"
    CORPORATE_EVENT = "CORPORATE_EVENT"
    MARKET_HOLIDAY = "MARKET_HOLIDAY"


class EventStatus(StrEnum):
    """Date-knowledge status of an event (spec §6/§7/§11). ESTIMATED dates
    are deterministic derivations (e.g. filing cadence) and are NEVER
    presented as confirmed facts nor alerted on; CONFIRMED comes from an
    authoritative source (company IR/SEC/agency/Fed/subscribed calendar or
    the user); REVISED = a confirmed date that moved; CANCELED = withdrawn."""

    ESTIMATED = "ESTIMATED"
    CONFIRMED = "CONFIRMED"
    REVISED = "REVISED"
    CANCELED = "CANCELED"


class EventSession(StrEnum):
    """When in the trading day a corporate event lands (spec §6)."""

    BEFORE_MARKET = "BEFORE_MARKET"
    DURING_MARKET = "DURING_MARKET"
    AFTER_MARKET = "AFTER_MARKET"
    UNKNOWN = "UNKNOWN"


class EventLifecycle(StrEnum):
    """Lifecycle relative to now (spec §67). Derived, never user-set."""

    SCHEDULED = "SCHEDULED"
    PRE_EVENT = "PRE_EVENT"
    LIVE = "LIVE"
    POST_EVENT = "POST_EVENT"
    ARCHIVED = "ARCHIVED"


class EventSourceKind(StrEnum):
    """Source-priority tiers for event facts (spec §78). Lower tier number
    = higher authority; a lower-authority source never overwrites a
    higher-authority one."""

    USER = "USER"                      # user-confirmed (IR page / SEC link)
    COMPANY_IR_SEC = "COMPANY_IR_SEC"  # SEC EDGAR filings, company IR
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"  # BLS / BEA / Census
    FEDERAL_RESERVE = "FEDERAL_RESERVE"
    STRUCTURED_PROVIDER = "STRUCTURED_PROVIDER"  # Massive/Alpaca/subscribed calendar
    DERIVED = "DERIVED"                # deterministic estimate (cadence)
    NEWS = "NEWS"
    LLM = "LLM"                        # never authoritative for dates

