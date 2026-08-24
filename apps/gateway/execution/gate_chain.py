"""THE GATE CHAIN — the platform's trading decision, as a library.

Everything an order must survive before a human is even offered the choice:
instrument selection, directional scoring, strength tier, volatility regime,
liquidity, sizing, the statistical and stress SHADOW evaluations, and the
PASS/FAIL chain itself.

WHY THIS IS NOT IN ``routers/orders.py`` ANY MORE. It was — 2,500 lines of it,
under four HTTP endpoints — and the shape leaked:

- ``order_sync.py`` and ``monitor.py`` are BACKGROUND LOOPS, and they had to
  import from a router to reach this logic. A task that never serves an HTTP
  request depending on the HTTP layer is the dependency arrow pointing the
  wrong way.
- Thirty-two test modules reached into a router to test business rules.
- Two sibling routers had to be imported from inside functions to break the
  resulting cycles.

None of that was a correctness bug — the tests were thorough and the chain was
right. It was an ORIENTATION bug: "where does the trading decision live" had
no answer a newcomer could guess, and any second entry point (a CLI, a
scheduled strategy) would have had to invert the dependency again.

The rule this file restores: **a router validates input, calls one function,
and turns the result into a status code.** The decision lives here.

Imported by ``routers/orders.py`` (which re-exports these names, so existing
call sites and tests keep working) and directly by ``order_sync.py``.
"""
import asyncio
import dataclasses
import weakref
from dataclasses import dataclass
from datetime import date, datetime, timezone
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fastapi import Depends, HTTPException
from pydantic import Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
# §65 decision-path counters (see the metric declarations below).
from libs.common.telemetry import REGISTRY
from libs.trading_core.contracts import (
    ContractQuote,
    SelectorParams,
    SpreadCandidate,
    SpreadParams,
    select_contracts,
    select_vertical_spread,
)
from libs.trading_core.features import atr
from libs.trading_core.greeks import PositionGreeksInput
from libs.trading_core.models import (
    ActorType,
    AuditAction,
    DirectionalBias,
    InstrumentType,
    MarketRegime,
    RiskDecision,
)
from libs.trading_core.risk import (
    RiskAssessment,
    RiskLimits,
    RiskRequest,
    assess,
)
from libs.trading_core.exits import ExitParams
# ATR_STOP_MULTIPLE (§12.1): single source shared with the backtest engine —
# live sizing and replayed sizing can never drift apart.
from libs.trading_core.risk.engine import ATR_STOP_MULTIPLE, strength_tier
# Underlying LIQUIDITY gate, REPORT mode (risk-engine audit §7.3 / B0): pure
# evaluation over data the chain already holds; never vetoes until promoted.
from libs.trading_core.risk.squeeze import (
    SqueezeProxyParams,
    assess_squeeze_proxy,
)
from libs.trading_core.risk.liquidity import (
    LiquidityLimits,
    evaluate_underlying_liquidity,
    liquidity_report_detail,
)
# Phase C pre-trade statistical layer (design contract §7.1/§7.2), SHADOW:
# the proposed-book comparison, the hypothetical quantity caps and the
# verdict they imply. NOTHING here is passed to `assess` (no `extra_caps`)
# — promotion out of SHADOW is an explicit, separate human step (§70).
from libs.trading_core.risk.pretrade import (
    CandidateSpec,
    SizingV2Params,
    StatisticalLimits,
    compare as pretrade_compare,
    shadow_verdict,
    sizing_v2_shadow,
    statistical_caps,
)
# §55: the staleness KIND the pre-trade caps are judged against. The
# statistical TTL (a trading day) is the right one: these caps are derived
# from daily-close VaR/ES, not from live greeks.
from libs.trading_core.risk.snapshot import STALENESS_KIND_STATISTICAL
# Phase D stress layer (design contract §8.3/§8.5), SHADOW: the candidate's
# legs are revalued under the SAME catalogue the snapshot just ran over the
# book, and the STRESS cap joins the SAME shadow verdict Phase C computes.
# Still nothing reaches `assess` (`extra_caps` stays empty — §70).
from libs.trading_core.options.reval import OptionLeg, StockLeg
from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.stress import StressLimits, run_stress, stress_caps
from libs.trading_core.signals import RegimeParams, classify_regime, score_direction
from libs.trading_core.strategies import select_instrument
from libs.broker import BrokerOrderLeg
from libs.broker.provider import BUY_TO_CLOSE, SELL_TO_OPEN
from libs.broker.alpaca import occ_option_symbol
from libs.trading_core.volatility import VolRegimeParams, classify_vol_regime

from .. import audit
from .. import event_risk  # Phase K §62-§67 event-risk seam, SHADOW ONLY
from .. import market_stream  # live NBBO cache (data_source.md §5) for LIQUIDITY
from ..broker_exec import (
    BrokerError,
    BrokerRejected,
    broker_order_details,
    new_client_order_id,
    submit_and_poll,
    submit_mleg_and_poll,
    submit_stock_cover_and_poll,
    submit_stock_short_and_poll,
)
from ..db import (
    Order,
    Position,
    StockBarDaily,
    TradingPoolItem,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
    utcnow,
)
from ..deps import (
    account_permissions_from_settings,
    broker_mode,
    require_broker,
    require_market_data_provider,
    resolve_broker,
    simulated_broker_mode,
)
from ..risk_inputs import build_portfolio_snapshot
from ..risk_snapshot import (
    DAYS_PER_YEAR,
    NEW_YORK,
    STRESS_DIVIDEND_YIELD,
    STRESS_RATE,
    TRIGGER_PRE_TRADE,
    build_risk_snapshot,
    correlation_state_api,
    correlation_state_for,
    record_atm_iv,
)
from ..schemas import TickerRequest
from ..routers.analysis import ensure_daily_bars
from ..routers.options import CHAIN_CACHE_TTL_SECONDS, build_option_chain, chain_iv_summary

# The portfolio-picture helpers live in routers/portfolio.py so the risk view
# and this order path build the identical book (plan §21). find_option_contract
# and is_option_position are also RE-EXPORTED through this module on purpose:
# routers/positions.py imports them from here.
from ..routers.portfolio import (
    _ewma_vol_targeting_side_by_side,
    find_option_contract,
    find_spread_short_leg,
    is_option_position,
    is_short_stock_position,
    is_spread_position,
    portfolio_greeks_read,
    stored_bars_by_ticker,
    stored_closes_by_ticker,
    vol_targeting_block,
)


# The only order sides that exist in this system (§5): a long-only account
# BUYS to open and SELLS to close. Sell-to-Open does not exist, ever.
BUY_TO_OPEN = "BUY_TO_OPEN"
SELL_TO_CLOSE = "SELL_TO_CLOSE"

# Position lifecycle states (positions.status column).
POSITION_OPEN = "OPEN"
POSITION_CLOSED = "CLOSED"

# Order lifecycle (§11, migration 010): the local row is created in this
# state and COMMITTED before the submit leaves the process, so a crash
# mid-submit leaves a resolvable row instead of an invisible broker order.
# The order-sync sweep (apps/gateway/order_sync.py) settles these against
# the broker by client_order_id.
PENDING_SUBMIT = "PENDING_SUBMIT"

# Single-user V1: fixed user identity until auth-service lands (matches
# routers/trading_control.py).
CURRENT_USER = "local-user"

# ---------------------------------------------------------------------------
# Execution serialization (§42 duplicate protection; V1 no-pyramiding).
#
# Every paper-execution mutation (approve, close, check-exits) runs its
# check-then-act sequence — idempotency lookup, open-position check, cash
# guard — and its fill inside ONE critical section. Without it, two rapid
# approves can BOTH pass the open-position check before either commits and
# double-fill (pyramiding, forbidden in V1), or a duplicate client_order_id
# can crash into the UNIQUE constraint with a 500 instead of replaying the
# existing order (§42). V1 runs a single process, so an in-process asyncio
# lock closes the race completely; the UNIQUE constraint on
# orders.client_order_id stays as the multi-process backstop.
#
# The lock is per event loop (test suites create one loop per test; asyncio
# primitives may not be shared across loops).
# ---------------------------------------------------------------------------
_EXECUTION_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)

BUY_TO_OPEN = "BUY_TO_OPEN"
SELL_TO_CLOSE = "SELL_TO_CLOSE"

# Position lifecycle states (positions.status column).
POSITION_OPEN = "OPEN"
POSITION_CLOSED = "CLOSED"

# Order lifecycle (§11, migration 010): the local row is created in this
# state and COMMITTED before the submit leaves the process, so a crash
# mid-submit leaves a resolvable row instead of an invisible broker order.
# The order-sync sweep (apps/gateway/order_sync.py) settles these against
# the broker by client_order_id.
PENDING_SUBMIT = "PENDING_SUBMIT"

# Single-user V1: fixed user identity until auth-service lands (matches
# routers/trading_control.py).
CURRENT_USER = "local-user"

# ---------------------------------------------------------------------------
# Execution serialization (§42 duplicate protection; V1 no-pyramiding).
#
# Every paper-execution mutation (approve, close, check-exits) runs its
# check-then-act sequence — idempotency lookup, open-position check, cash
# guard — and its fill inside ONE critical section. Without it, two rapid
# approves can BOTH pass the open-position check before either commits and
# double-fill (pyramiding, forbidden in V1), or a duplicate client_order_id
# can crash into the UNIQUE constraint with a 500 instead of replaying the
# existing order (§42). V1 runs a single process, so an in-process asyncio
# lock closes the race completely; the UNIQUE constraint on
# orders.client_order_id stays as the multi-process backstop.
#
# The lock is per event loop (test suites create one loop per test; asyncio
# primitives may not be shared across loops).
# ---------------------------------------------------------------------------
_EXECUTION_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def execution_lock() -> asyncio.Lock:
    """The current event loop's paper-execution lock (see block comment)."""
    loop = asyncio.get_running_loop()
    lock = _EXECUTION_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _EXECUTION_LOCKS[loop] = lock
    return lock

# --- V1 parameters (plan §6.2: parameters, never hardcoded truths) ----------
ATR_PERIOD = 14
# DATA_QUALITY staleness bound: last stored bar must be within this many
# calendar days of today (stub data always passes; the check must still exist).
MAX_BAR_AGE_DAYS = 5
# Standard US equity option contract multiplier (plan §12.1: cash and max
# loss are per-share premium x this).
OPTION_MULTIPLIER = 100
# Parameter seams for the option gates (plan §6.2): module-level so tests and
# future config wiring can substitute custom thresholds without touching the
# chain logic — §7 vol-regime thresholds and §9.1/§9.2 selector thresholds.
# The §5/§8 account permission flags the §8 matrix degrades under are NOT a
# module constant: they come from Settings via the ONE factory
# apps.gateway.deps.account_permissions_from_settings() (guide §8), read
# fresh on every chain run so configuration and enforcement cannot drift.
VOL_REGIME_PARAMS = VolRegimeParams()
SELECTOR_PARAMS = SelectorParams()
SPREAD_PARAMS = SpreadParams()
# §12.1 SHORT STOCK (roadmap Phase 3): a short's loss is unbounded and an
# overnight gap can blow through any stop, so the risk basis is the 2×ATR14
# stop distance INFLATED by this gap factor — sizing halves relative to the
# long-stock formula. Industry standard practice: stop-based risk with a
# gap multiplier; a parameter, never a hardcoded truth (§6.2).
# Squeeze-proxy research defaults (§6.2) — REPORT mode only.
SQUEEZE_PROXY_PARAMS = SqueezeProxyParams()

SHORT_STOCK_GAP_RISK_FACTOR = 2.0

# The §10 gate chain, in its exact order.
GATE_ORDER = (
    "TRADING_POOL_AUTHORIZATION",
    "DATA_QUALITY",
    "REGIME",
    "DIRECTIONAL_SIGNAL",
    "VOLATILITY",
    "INSTRUMENT",
    "SQUEEZE_RISK",
    "LIQUIDITY",
    "CONTRACT_SELECTION",
    "RISK_APPROVAL",
)

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"

# Exact V1 skip details (contract-fixed strings).
SKIP_EARLIER_FAIL = "not evaluated: earlier gate failed"
# Retired 2026-08-17 (audit §7.3 / B0): gate 7 now evaluates underlying
# liquidity in REPORT mode and no longer emits this text; kept only so any
# external reader of the historical string keeps resolving.
SKIP_NO_OPTION_DATA = "no option/quote data yet — arrives with the Massive integration"
SKIP_STOCK_ORDER = "stock order — no contract selection needed"

# Underlying LIQUIDITY gate thresholds (risk-engine audit §7.3 / §10 B0):
# RESEARCH DEFAULTS, UNVALIDATED — the gate runs in REPORT mode (PASS with
# the measured numbers + hypothetical verdict; `shadow.liquidity` in the
# RISK_DECISION audit) until promoted under the Q3 shadow window. Option-leg
# liquidity stays enforced by the §9.1 filters inside CONTRACT_SELECTION.
LIQUIDITY_LIMITS = LiquidityLimits()

# Phase C statistical limits (design contract §7.2): RESEARCH DEFAULTS,
# UNVALIDATED. The pre-trade statistical layer runs in SHADOW — its caps and
# verdicts are computed, logged under `shadow.statistical` and shown in the
# preview, and NOTHING is passed to `assess`. Promotion (mode=PRODUCTION plus
# `extra_caps`) is an explicit human step after the Q3 window (audit §11 Q3).
STATISTICAL_LIMITS = StatisticalLimits()

# Phase D stress limits (design contract §8.3): RESEARCH DEFAULT, UNVALIDATED
# (`max_stress_loss_pct_nav=0.10`). Spec §27 gives the stress test veto
# authority; that veto IS the promotion to PRODUCTION, which is a human step.
# Until then the STRESS cap is computed, merged into the SAME hypothetical
# verdict the statistical caps produce, logged, and passed to NOTHING.
STRESS_LIMITS = StressLimits()

# Sizing v2 parameters (spec §36/§37/§59; compliance §3 Tier A): RESEARCH
# DEFAULTS, UNVALIDATED. The three §37 modifiers the production budget still
# does not compose (ES, correlation, model health) and the §36 risk-linked
# cash floor, computed SHADOW beside the real numbers. `assess` receives the
# UNCHANGED `budget_multiplier` and the UNCHANGED `limits.cash_floors`;
# promotion would mean feeding `candidate_budget_pct` and the floor into
# those two, and that is an explicit human step after the Q3 window.
SIZING_V2_PARAMS = SizingV2Params()

# --- §65 decision-path telemetry (compliance §3 Tier A) --------------------
#
# The three DECISION-PATH instruments §65 names, incremented at the ONE place
# the RISK_DECISION audit event is written, so a counter can never disagree
# with the audit trail. They observe; they decide nothing.
#
# Deliberately UNLABELLED (§65 names them as plain counters): a per-ticker or
# per-reason-code label would fork one time series into hundreds and defeat
# the alertable rate these exist to provide. The reason codes are already in
# the audit row for anyone who needs the breakdown.
RISK_RESIZE_COUNT = REGISTRY.counter(
    "risk_resize_count",
    "Tier 0 risk decisions that RESIZED the requested quantity "
    "(decision APPROVE_WITH_RESIZE), risk spec §65.",
)
RISK_REJECT_COUNT = REGISTRY.counter(
    "risk_reject_count",
    "Gate-chain runs that produced no order — a Tier 0 risk-engine REJECT or "
    "a veto by an earlier gate (the chain never reached `assess`), "
    "risk spec §65.",
)
STRESS_LIMIT_BLOCKS = REGISTRY.counter(
    "stress_limit_blocks",
    "Runs where the SHADOW stress cap would have BOUND — its `cap_qty` is "
    "below the quantity that was actually approved/requested (risk spec §65, "
    "§27). SHADOW: the stress cap changed nothing; this counter is the "
    "evidence series for how often it WOULD have.",
)

# Regimes that veto a new LONG (bullish) entry (§6.1: TRANSITION defaults to
# NO TRADE; bear regimes give a long-only account no bullish edge, §5). A
# resolved BEAR direction may still trade a bear regime — via long puts only.
_BEAR_REGIMES = frozenset({MarketRegime.MILD_BEAR, MarketRegime.STRONG_BEAR})


class OrderPreviewRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)
    # Direction resolution seam, mirroring GET .../options (plan §9): an
    # explicit BULL/BEAR wins; AUTO (default) defers to the signal bias.
    direction: Literal["AUTO", "BULL", "BEAR"] = "AUTO"


@dataclass
class GateChainResult:
    """One full §10 gate-chain evaluation, shared by preview and approve.

    ``preview`` is the exact dict ``POST /api/orders/preview`` responds with;
    ``veto_gate`` names the first FAILing gate (``None`` when the chain fully
    passes — only then may an approval fill, §42). ``entry_price`` is the last
    stored UNDERLYING close, ``stop_distance`` the §12.1 stock stop
    (2 * ATR14), ``edge`` the directional edge at evaluation, and
    ``last_bar_date`` the last stored bar date (YYYY-MM-DD) — the entry-bar
    anchor for ``bars_held`` (plan §11).

    Option wiring (plan §8/§9): ``instrument`` is the §8 matrix verdict
    ("LONG_STOCK" | "LONG_CALL" | "LONG_PUT" | "NO_TRADE"; ``None`` when the
    chain vetoed before the matrix ran — honest null), ``vol_regime`` the §7
    classification value (``None`` without chain data), and ``contract`` the
    §9 top-ranked :class:`ContractQuote` an option approval fills against
    (``None`` for stock / no trade). For option chains the risk engine was
    fed ``mid * 100`` as both entry and stop (§12.1), so
    ``assessment.approved_quantity`` counts CONTRACTS.

    The chain has already recorded its SYSTEM RISK_DECISION audit event on
    the session but NOT committed — the caller owns the transaction, so the
    audit lands atomically with whatever state change follows (rule 12).
    """

    preview: dict
    assessment: RiskAssessment | None
    entry_price: float | None
    stop_distance: float | None
    edge: float | None
    last_bar_date: str | None
    veto_gate: str | None
    instrument: str | None = None
    vol_regime: str | None = None
    contract: ContractQuote | None = None
    #: §9-S verdict for spread instruments (roadmap Phase 1): the two-leg
    #: candidate an approval fills against; None otherwise.
    spread: "SpreadCandidate | None" = None

    @property
    def failed(self) -> bool:
        """True when any gate FAILed — no order may be produced (§42)."""
        return self.veto_gate is not None


async def _broker_cash_for_sizing() -> tuple[float | None, str | None]:
    """The REAL broker's reported CASH for §14 sizing, or why it is unknown.

    Returns ``(cash, None)`` when a real broker is configured and answered,
    ``(None, None)`` in simulated/unset modes (no clamp applies — §14 is
    about honoring a real cash account, and those modes have no broker
    account to honor), and ``(None, reason)`` when a real broker is
    configured but its account could not be fetched. In that last case the
    caller must FAIL CLOSED (guide §28): an unreachable broker means
    deployable cash cannot be verified, and sizing from local cash alone
    could deploy cash the account does not actually have.

    Deliberately reads ``BrokerAccount.cash`` and NEVER ``buying_power``
    (guide §14): do not size from paper buying power — it may include
    margin-like capacity beyond the configured platform cash model.
    """
    if simulated_broker_mode() or not broker_mode():
        return None, None
    try:
        broker = resolve_broker()
        account = await asyncio.to_thread(broker.get_account)
    except Exception as exc:  # fail closed — ANY fault means "unverifiable"
        return None, str(exc) or type(exc).__name__
    return account.cash, None


def _statistical_shadow_detail(build) -> dict:
    """The RISK_DECISION ``shadow.statistical`` block (design contract §6).

    A COMPACT view of the SHADOW snapshot — the headline 95% 1-day
    percent-of-NAV numbers, the model-risk state, the dispersion flag and
    the per-model health ledger — not the whole payload: the audit row is a
    decision record, and the full snapshot is already persisted under
    ``snapshot_id`` for anyone who wants it (spec §44 reproducibility).

    Every value here is honest-null-capable: a model that could not be
    computed contributes ``None``, never a zero that would read as "no
    risk". The ``note`` states outright that this is the CURRENT book, so
    nobody reads it as a post-trade projection.
    """
    api = build.api

    def _pct(rows: list[dict], model: str) -> float | None:
        for row in rows:
            if row["model"] == model and row["confidence"] == 0.95:
                return row["pct_nav"]
        return None

    dispersion_block = api.get("dispersion")
    model_risk_block = api.get("model_risk")
    return {
        "snapshot_id": build.row_id,
        "as_of": api["as_of"],
        "model_risk_state": (
            model_risk_block["state"] if model_risk_block is not None else None
        ),
        "dispersion_high": (
            dispersion_block["high"] if dispersion_block is not None else None
        ),
        "historical_var_95_1d_pct_nav": _pct(api["var"], "HISTORICAL"),
        "historical_es_95_1d_pct_nav": _pct(api["es"], "HISTORICAL"),
        "gaussian_es_95_1d_pct_nav": _pct(api["es"], "GAUSSIAN"),
        "health": dict(api["model_health"]),
        "note": (
            "current-book view; the proposed-book comparison is under "
            "`comparison` / `hypothetical` (Phase C, SHADOW)"
        ),
    }


# ---------------------------------------------------------------------------
# Phase C — the pre-trade statistical layer (design contract §7.5), SHADOW
# ---------------------------------------------------------------------------
#
# THE DECISION IS ALREADY MADE before any of this runs. These helpers read
# the Tier 0 outcome and the SHADOW snapshot and produce a hypothetical
# verdict for the audit and the preview. They pass NOTHING to `assess`
# (`extra_caps` is never populated here — §70: promotion out of SHADOW is an
# explicit human step), touch no gate, and their failure mode is a `note`.


def _metric_pair_row(name: str, pair, nav: float) -> dict:
    """One §46 CURRENT-vs-AFTER-TRADE table row from a ``MetricPair``.

    Honest nulls throughout: a model that could not be computed contributes
    ``None`` on its side and a ``None`` delta (a difference of nulls is not
    a number, contract §7.1), never a zero that would read as "no change".
    Percent-of-NAV fields are FRACTIONS (house rule / UI contract).
    """
    before = pair.before
    after = pair.after

    def _value(result):
        return result.value if result is not None else None

    def _health(result):
        return str(result.health) if result is not None else None

    return {
        "metric": name,
        "before_usd": _value(before),
        "after_usd": _value(after),
        "before_pct_nav": _pct_of_nav(_value(before), nav),
        "after_pct_nav": _pct_of_nav(_value(after), nav),
        "delta_usd": pair.delta_usd,
        "delta_pct_nav": pair.delta_pct_nav,
        "before_health": _health(before),
        "after_health": _health(after),
        "reason": (after.reason if after is not None else None)
        or (before.reason if before is not None else None),
    }


def _scalar_pair_row(
    metric: str,
    before: float | None,
    after: float | None,
    nav: float,
    *,
    layer: str,
    pct_of_nav: bool,
    reason: str | None = None,
) -> dict:
    """One §46 table row built from a plain (before, after) pair.

    Used by the rows whose two sides are NOT ``ModelResult``s — net vega
    (§46, $ per IV point) — so the UI renders them with the same keys it
    already reads for every model row rather than a special case.

    ``pct_of_nav`` is False for a row whose unit is not USD: dividing
    $-per-IV-point by NAV would be arithmetic without meaning, so those
    fields stay ``None`` rather than carrying a plausible-looking number.
    ``delta_usd`` is ``after − before`` and is an honest ``None`` when
    either side is unmeasured.
    """
    delta = (
        after - before if (before is not None and after is not None) else None
    )

    def _pct(v: float | None) -> float | None:
        return _pct_of_nav(v, nav) if pct_of_nav else None

    return {
        "metric": metric,
        "layer": layer,
        "before_usd": before,
        "after_usd": after,
        "before_pct_nav": _pct(before),
        "after_pct_nav": _pct(after),
        "delta_usd": delta,
        "delta_pct_nav": _pct(delta),
        "before_health": None,
        "after_health": None,
        "reason": reason,
    }


def _stress_comparison_row(stress_block, nav: float) -> dict | None:
    """The §46 table's ``worst_stress_loss`` row (Phase D §8.5), or ``None``.

    ``None`` when the stress layer produced nothing to compare (it raised, or
    the catalogue was empty) — a row of nulls claiming to be a measurement is
    worse than no row.

    Sign: this row reports a LOSS (positive = money lost), the same
    convention every VaR/ES row above uses, so the table reads down one
    column. ``delta_usd`` is ``after − before``: positive means the trade
    would DEEPEN the worst-case loss. A side that could not be measured is an
    honest ``None``, and the delta of a null is a null (never a zero that
    would read as "no change").
    """
    if not stress_block:
        return None
    before = stress_block.get("worst_before")
    after = stress_block.get("worst_after")
    if before is None and after is None:
        return None
    before_usd = before.get("loss_usd") if before else None
    after_usd = after.get("loss_usd") if after else None
    delta = (
        after_usd - before_usd
        if (before_usd is not None and after_usd is not None)
        else None
    )
    return {
        "metric": "worst_stress_loss",
        "layer": "STRESS",
        "before_usd": before_usd,
        "after_usd": after_usd,
        "before_pct_nav": _pct_of_nav(before_usd, nav),
        "after_pct_nav": _pct_of_nav(after_usd, nav),
        "delta_usd": delta,
        "delta_pct_nav": _pct_of_nav(delta, nav),
        "before_health": before.get("health") if before else None,
        "after_health": after.get("health") if after else None,
        "before_scenario": before.get("scenario") if before else None,
        "after_scenario": after.get("scenario") if after else None,
        "reason": (after.get("reason") if after else None)
        or (before.get("reason") if before else None)
        or stress_block.get("reason"),
    }


def _pct_of_nav(value: float | None, nav: float) -> float | None:
    """``value / nav`` as a FRACTION, or an honest ``None``.

    A zero or negative NAV cannot denominate a ratio — reporting one anyway
    would fabricate a number (§44 rule 18).
    """
    if value is None or nav <= 0:
        return None
    return value / nav


def _comparison_api(comparison, nav: float) -> dict:
    """The §46 "CURRENT vs AFTER TRADE" table, serialised for the wire.

    ``rows`` is the ORDERED table the UI renders verbatim: the two Tier 0
    rows first (heat and cash — the numbers that actually decided), then the
    statistical rows. Everything below ``rows`` is the concentration view
    the §11 gate basis is measured on.
    """
    tier0_rows = [
        {
            "metric": "portfolio_heat_pct",
            "before_pct": comparison.heat_pct[0],
            "after_pct": comparison.heat_pct[1],
            "layer": "HARD_LIMIT",
        },
        {
            "metric": "cash_pct",
            "before_pct": comparison.cash_pct[0],
            "after_pct": comparison.cash_pct[1],
            "layer": "HARD_LIMIT",
        },
    ]
    rows = [
        _metric_pair_row("var_hist_95", comparison.var_hist_95, nav),
        _metric_pair_row("es_hist_95", comparison.es_hist_95, nav),
        _metric_pair_row("var_hist_99", comparison.var_hist_99, nav),
        _metric_pair_row("es_hist_99", comparison.es_hist_99, nav),
        _metric_pair_row("gaussian_es_95", comparison.gaussian_es_95, nav),
        _metric_pair_row("volatility", comparison.volatility, nav),
    ]
    before_notional, after_notional = comparison.net_delta_notional
    before_vega, after_vega = comparison.net_vega
    # §8 (ADDITIVE): incremental VaR-95 as a NAMED row, beside the
    # incremental-ES number that has always been first-class. Both sides of
    # the subtraction are the `var_hist_95` row above, so the table cannot
    # show a VaR pair whose difference disagrees with this row.
    rows.append(
        _scalar_pair_row(
            "incremental_var_95",
            comparison.var_hist_95.before.value
            if comparison.var_hist_95.before is not None
            else None,
            comparison.var_hist_95.after.value
            if comparison.var_hist_95.after is not None
            else None,
            nav,
            layer="STATISTICAL",
            pct_of_nav=True,
            reason=comparison.reason,
        )
    )
    # §46 (ADDITIVE): net vega before/after, $ per one IV POINT -- not USD
    # exposure, so `pct_of_nav=False` keeps the percent columns null rather
    # than dividing a greek by NAV.
    rows.append(
        _scalar_pair_row(
            "net_vega",
            before_vega,
            after_vega,
            nav,
            layer="STATISTICAL",
            pct_of_nav=False,
            reason=(
                None
                if after_vega is not None
                else "net vega is unmeasured on at least one side "
                "(the book greeks or the contract vega was unavailable) — "
                "a net that dropped the candidate's vega would read as "
                "'this trade adds no vol exposure'"
            ),
        )
    )
    return {
        "quantity": comparison.quantity,
        "health": str(comparison.health),
        "reason": comparison.reason,
        "n_obs": comparison.n_obs,
        "tail_size_95": comparison.tail_size_95,
        "tier0_rows": tier0_rows,
        "rows": rows,
        "incremental_es_95_usd": comparison.incremental_es_95_usd,
        "incremental_es_95_pct_nav": comparison.incremental_es_95_pct_nav,
        # §8 (ADDITIVE): first-class incremental VaR, the field the audit
        # found existed only as `MetricPair.delta_usd`.
        "incremental_var_95_usd": comparison.incremental_var_95_usd,
        "incremental_var_95_pct_nav": comparison.incremental_var_95_pct_nav,
        "marginal_es_95_per_unit": comparison.marginal_es_95_per_unit,
        "candidate_es_share_after": comparison.candidate_es_share_after,
        "max_single_es_share_before": comparison.max_single_es_share_before,
        "max_single_es_share_after": comparison.max_single_es_share_after,
        "bucket_es_share_after": dict(comparison.bucket_es_share_after),
        "net_delta_notional_before": before_notional,
        "net_delta_notional_after": after_notional,
        # §46 (ADDITIVE): net vega before/after in $ per one IV point.
        "net_vega_before": before_vega,
        "net_vega_after": after_vega,
    }


def _cap_api(cap) -> dict:
    """One hypothetical :class:`QuantityCap` — code, layer, quantity and the
    server-generated §47 sentence rendered verbatim by the UI."""
    return {
        "code": cap.code,
        "layer": cap.layer,
        "cap_qty": cap.cap_qty,
        "sentence": cap.sentence,
        "measured": dict(cap.measured),
    }


#: Position key the candidate occupies in the proposed book. The "#candidate"
#: suffix cannot collide with a real key (those are "<TICKER>#<row id>").
CANDIDATE_KEY_SUFFIX = "#candidate"


def _candidate_spec(
    ticker: str,
    *,
    instrument: str,
    chosen,
    chosen_spread,
    spot: float,
    risk_stop: float,
    risk_entry: float,
    quantity_requested: int,
) -> CandidateSpec:
    """The proposed entry as a per-unit :class:`CandidateSpec` (contract §7.5).

    The bases are THE SAME ones Tier 0 just sized against — ``risk_stop`` is
    the risk basis and ``risk_entry`` the capital basis, exactly as passed
    to ``RiskRequest`` above — so the statistical layer can never measure a
    different trade than the engine judged.

    The signed per-share delta by instrument (never re-signed here; the sign
    convention is ``greeks.py``'s and the chain's):

    - stock: ``+1`` long, ``−1`` for a Phase 3 SHORT_STOCK;
    - single option: the CHAIN's signed per-share delta AS-IS — a long put
      already carries a negative delta, so negating it would double-count
      the direction — with multiplier 100;
    - spread: the NET delta of the two legs (the short leg already negated
      by the spread builder), multiplier 100.

    OPTION LEG FIELDS (design §10.1/§10.3, ADDITIVE). An option candidate
    also carries the SELECTED contract's ``strike``, ``right``, ``t_years``
    (DTE / 365), ``iv0`` and ``mark0`` — read off the very same
    ``chosen`` / ``chosen_spread`` object ``_candidate_stress_legs`` builds
    its revaluation legs from, so the pre-trade P&L series, the pre-trade
    stress rows and the §9 selector rationale all describe ONE contract.
    The candidate's incremental/marginal ES and the ES-share caps then see
    its CONVEXITY instead of a straight delta line. A contract whose chain
    gave no IV keeps every field ``None`` and stays DELTA_LINEAR — labelled,
    never guessed. Stock candidates carry no leg fields at all.

    SPREAD: THE LONG LEG, DOCUMENTED. A spread is one candidate key (see
    ``CandidateSpec.position_at``), so it carries the LONG leg's contract
    with the spread's NET delta. The two-leg representation was rejected
    deliberately: ``pretrade.proposed_book`` writes one
    ``per_position[candidate.key]`` series and the ES-share caps call
    ``share_of(candidate.key)``, so a second key would silently drop half
    the candidate out of every one of those consumers. The consequence —
    the short leg's offsetting convexity is not modelled, making the
    measured convexity an upper bound — is stated on ``position_at`` and is
    the conservative direction for a SHADOW cap.

    NET VEGA (§46, additive). The candidate also carries ``vega0``, the
    per-share vega of the same selected contract: the chain's value for a
    single option, the spread's NET vega (vega is additive across legs, so
    unlike the revaluation fields this one need not collapse to the long
    leg), and 0.0 for stock (a measurement, not a gap). It feeds ONLY the
    §46 net-vega before/after row -- no estimator and no cap reads it.

    ``ValueError`` (from ``CandidateSpec``) when a delta is missing or a
    basis is malformed; the caller turns that into an honest note, never a
    guessed delta.
    """
    strike = right = t_years = iv0 = mark0 = vega0 = None
    if chosen is not None:
        if chosen.delta is None:
            raise ValueError(
                f"{ticker}: the selected contract carries no delta (the chain "
                "omitted it) — a candidate delta is never guessed"
            )
        delta = chosen.delta
        multiplier = OPTION_MULTIPLIER
        strike, right, iv0, mark0 = (
            chosen.strike,
            chosen.right,
            chosen.iv,
            chosen.mid,
        )
        t_years = max(chosen.dte, 0) / DAYS_PER_YEAR
        # §46: the SELECTED contract's per-share vega, AS-IS from the chain
        # (the same object `candidate_greeks` above reads for the Tier 0
        # greek limits, so the shadow net-vega row and the gate that
        # actually decides can never describe different contracts). `None`
        # when the chain omitted vega -- never a guessed 0.
        vega0 = chosen.vega
    elif chosen_spread is not None:
        if chosen_spread.net_delta is None:
            raise ValueError(
                f"{ticker}: the selected spread carries no net delta (a leg's "
                "chain delta is missing) — a candidate delta is never guessed"
            )
        delta = chosen_spread.net_delta
        multiplier = OPTION_MULTIPLIER
        long_leg = chosen_spread.long_leg
        strike, right, iv0, mark0 = (
            long_leg.strike,
            long_leg.right,
            long_leg.iv,
            long_leg.mid,
        )
        t_years = max(long_leg.dte, 0) / DAYS_PER_YEAR
        # §46: the spread's NET vega (long - short, the short leg already
        # negated by the spread builder), NOT the long leg's. Unlike the
        # revaluation fields above -- which must describe ONE contract and
        # therefore take the long leg -- vega is additive across legs, so
        # the net is both available and the correct number for a net-vega
        # row. A spread's short leg genuinely offsets vol exposure, and
        # reporting only the long leg would overstate it.
        vega0 = chosen_spread.net_vega
    else:
        delta = -1.0 if instrument == InstrumentType.SHORT_STOCK.value else 1.0
        multiplier = 1
        # Stock has NO vega. This is a measurement (0.0), not a gap -- the
        # same value `candidate_greeks` passes to the Tier 0 greek gate --
        # so a stock trade still produces a real net-vega before/after row
        # rather than an honest null.
        vega0 = 0.0
    return CandidateSpec(
        key=f"{ticker}{CANDIDATE_KEY_SUFFIX}",
        ticker=ticker,
        instrument=instrument,
        multiplier=multiplier,
        spot=spot,
        delta=delta,
        max_loss_per_unit=risk_stop,
        capital_per_unit=risk_entry,
        quantity_requested=quantity_requested,
        strike=strike,
        right=right,
        t_years=t_years,
        iv0=iv0,
        mark0=mark0,
        vega0=vega0,
    )


def _pretrade_statistical_shadow(
    build,
    candidate: CandidateSpec,
    *,
    approved_quantity: int,
    requested_quantity: int,
    nav: float,
    heat_before: float,
    heat_after: float,
    cash_before: float,
    cash_after: float | None,
    delta_notional_before: float | None,
    buckets: Mapping[str, Sequence[str]],
    net_vega_before: float | None = None,
    snapshot_now: datetime | None = None,
    limits: StatisticalLimits,
    extra_caps: Sequence[Any] = (),
    stress_block: Mapping[str, Any] | None = None,
) -> dict:
    """The Phase C ``shadow.statistical`` extension (contract §7.5), SHADOW.

    Everything is measured against the book the SHADOW snapshot just built
    (``build.book`` / ``build.returns``) — the same positions, dates and NAV
    the Tier 0 decision was made against, never a rebuilt one that could
    silently disagree (plan §21).

    Produces, all additive:

    - ``comparison``: the §46 table at the APPROVED quantity;
    - ``comparison_at_requested``: the same table at the REQUESTED quantity,
      present ONLY when the two differ (otherwise it would be a duplicate
      row set claiming to be a second measurement);
    - ``caps``: the hypothetical quantity caps and the health of the cap
      search — an UNAVAILABLE statistical view yields NO cap (fail-open in
      SHADOW, contract §7.2);
    - ``hypothetical``: what the statistical layer ALONE would have decided
      at the approved quantity;
    - ``limits``: every threshold it was measured against (research
      defaults, UNVALIDATED — house rule: never a hardcoded truth);
    - ``correlation_state``: the §19 regime of the book PLUS the candidate's
      ticker (the correlation that would exist after the trade).

    ``extra_caps`` (Phase D §8.5) are caps produced by OTHER shadow layers —
    today the STRESS cap. They join the SAME ``caps`` list and the SAME
    ``shadow_verdict``, so ``hypothetical`` is what EVERY shadow layer
    together would have decided; two layers each publishing their own final
    verdict about one trade would be two answers to one question. Each cap
    keeps its own ``layer`` (``STATISTICAL`` / ``CONCENTRATION`` / ``STRESS``)
    so the binding list still says which layer bound.

    ``stress_block`` (Phase D §8.5) contributes ONE extra row to the §46
    table — ``worst_stress_loss``, before vs after in the same
    ``before_usd`` / ``after_usd`` / ``delta_usd`` shape every other row
    uses, so the UI renders it without a special case.

    An empty book (nothing to compare against) and a missing return column
    are honest ``note``s, not errors.
    """
    book = build.book
    returns = build.returns
    if book is None or returns is None:
        # `comparison_note` — never `note`: the Phase B key means "what the
        # CURRENT-book block is", and this block is merged into it. A Phase C
        # gap must never overwrite a Phase B statement.
        return {
            "comparison_note": (
                "no priceable book to compare against (no open position has "
                "stored bars) — the proposed-book comparison needs a book"
            ),
            "limits": dataclasses.asdict(limits),
        }

    comparison = pretrade_compare(
        book,
        candidate,
        approved_quantity,
        returns=returns,
        nav=nav,
        heat_before=heat_before,
        heat_after=heat_after,
        cash_before=cash_before,
        cash_after=cash_after if cash_after is not None else cash_before,
        positions=build.positions,
        buckets=buckets,
        delta_notional_before=delta_notional_before,
        net_vega_before=net_vega_before,
        limits=limits,
    )
    at_requested = None
    if requested_quantity != approved_quantity:
        at_requested = pretrade_compare(
            book,
            candidate,
            requested_quantity,
            returns=returns,
            nav=nav,
            heat_before=heat_before,
            # The heat/cash pair is a TIER 0 measurement supplied by the
            # caller; at the requested quantity Tier 0 measured nothing (it
            # resized), so the honest thing to carry is the before-value on
            # both sides rather than a recomputed number this layer is not
            # entitled to invent. The statistical rows below ARE measured at
            # the requested quantity — those are this layer's own.
            heat_after=heat_before,
            cash_before=cash_before,
            cash_after=cash_before,
            positions=build.positions,
            buckets=buckets,
            delta_notional_before=delta_notional_before,
            net_vega_before=net_vega_before,
            limits=limits,
        )

    caps, caps_health, caps_reason = statistical_caps(
        book,
        candidate,
        returns=returns,
        nav=nav,
        positions=build.positions,
        buckets=buckets,
        limits=limits,
    )
    caps = [*caps, *extra_caps]
    # --- §55: the staleness consumer (compliance §3 Tier C) ---------------
    # `is_stale` has existed, been tested to the boundary, and been consumed
    # by NOTHING but the serialiser since Phase B. Here it finally decides
    # something: when the snapshot these caps were measured on is older than
    # its own TtlPolicy, the caps describe a book that may have moved, so the
    # shadow verdict becomes UNAVAILABLE_STALE and the caps are SUPPRESSED
    # WITH the reason rather than applied.
    #
    # SHADOW-ONLY, BY CONSTRUCTION. `assess()` above already ran and already
    # returned; `extra_caps` is not passed to it at either production call
    # site (AST-pinned by tests/test_risk_adversarial.py). Suppressing caps
    # here therefore removes a HYPOTHETICAL, never a control — a Tier 0
    # decision cannot change on this path whatever this block does.
    stale = False
    stale_reason: str | None = None
    snapshot = getattr(build, "snapshot", None)
    # `snapshot_now` defaults to NOW rather than to "never stale": a caller
    # that forgets to pass it must not silently disable the check.
    snapshot_now = snapshot_now or datetime.now(timezone.utc)
    if snapshot is not None:
        # ONE try around the whole read, not one per call: `is_stale` and
        # `age_seconds` share the same tz-awareness precondition, so a
        # mismatch that let the first through cannot trip the second — but
        # a partial success that left `stale=True` with no reason would
        # produce a suppressed verdict that could not say why, which is the
        # one outcome §55 must not allow.
        try:
            stale = bool(snapshot.is_stale(snapshot_now))
            if stale:
                age = snapshot.age_seconds(snapshot_now)
                ttl = snapshot.ttl.seconds_for(STALENESS_KIND_STATISTICAL)
                stale_reason = (
                    f"the PRE_TRADE snapshot is stale: age {age:.0f}s > ttl "
                    f"{ttl:.0f}s (statistical). Hypothetical caps were computed "
                    "but SUPPRESSED — a cap measured on a book that may have "
                    "changed must not reduce a quantity. SHADOW: no Tier 0 "
                    "decision is affected (spec §55)."
                )
        except Exception:  # noqa: BLE001 — a staleness read never vetoes
            # Fail OPEN in SHADOW, deliberately: an unreadable clock must not
            # invent a staleness verdict. The PROMOTION rule is the opposite
            # (design §7.5) and is a user decision.
            stale = False
            stale_reason = None
    verdict = shadow_verdict(
        approved_quantity, caps, stale=stale, stale_reason=stale_reason
    )

    comparison_api = _comparison_api(comparison, nav)
    stress_row = _stress_comparison_row(stress_block, nav)
    if stress_row is not None:
        comparison_api["rows"].append(stress_row)

    return {
        "comparison": comparison_api,
        "comparison_at_requested": (
            _comparison_api(at_requested, nav) if at_requested is not None else None
        ),
        "caps": {
            "health": str(caps_health),
            "reason": caps_reason,
            "rows": [_cap_api(cap) for cap in caps],
        },
        "hypothetical": {
            "decision": verdict.hypothetical_decision,
            "quantity": verdict.hypothetical_quantity,
            "binding": list(verdict.binding),
            "mode": verdict.mode,
            "approved_quantity": approved_quantity,
            # §55 (ADDITIVE): whether the snapshot behind these caps was
            # stale, and why the verdict says so. `false` / `null` on the
            # ordinary path, so a reader can tell "no cap bound" apart from
            # "the caps were not trusted".
            "stale": stale,
            "reason": verdict.reason,
            "note": (
                "SHADOW: what the STATISTICAL layer alone would have decided "
                "at the Tier 0 approved quantity. It changed nothing."
            ),
        },
        "limits": dataclasses.asdict(limits),
    }



def _sizing_v2_shadow_block(
    build,
    *,
    regime_floor_pct: float,
    tier_budget_pct: float | None,
    vol_multiplier_used: float,
    params: SizingV2Params,
) -> dict:
    """``shadow.statistical.sizing_v2`` — the §36/§37/§59 composition, SHADOW.

    Reads ONLY values the decision path already has in hand, so this block
    can never describe a different book than the one Tier 0 judged:

    - ``es95_pct_nav`` — the HISTORICAL ES-95 1-day percent-of-NAV row of the
      snapshot the shadow build just produced (the same number
      :func:`_statistical_shadow_detail` publishes as
      ``historical_es_95_1d_pct_nav``);
    - ``model_risk_state`` — ``api["model_risk"]["state"]`` (§59);
    - ``correlation_state`` — ``api["correlation_state"]["state"]`` (§19), the
      regime of the CURRENT book's tickers;
    - ``drawdown_current_pct`` — ``drawdown_api["current_pct"]``, the
      persisted SCHEDULED NAV path's drawdown (a NEGATIVE fraction);
    - ``regime_floor_pct`` — Tier 0's OWN ``limits.cash_floors[regime]``, the
      floor actually in force, passed in by the caller;
    - ``tier_budget_pct`` / ``vol_multiplier_used`` — the two factors the
      engine's budget was composed from (``limits.budget_<tier>`` for the
      strength tier the engine resolved, and the vol-targeting multiplier
      that was passed to ``assess``).

    Every one of those is honest-null-capable and the library holds the
    modifier at 1.0 with a note when a value is missing — a missing input is
    never read as "that risk is low".

    SHADOW: the returned dict is logged and mirrored to the preview. The
    budget that sized this trade is still ``tier_budget × vol_multiplier``
    hard-capped by ``abs_max_trade_risk`` inside ``assess``, and the cash
    floor still enforced is ``limits.cash_floors[regime]``. Neither is
    touched here.
    """
    api = build.api
    if tier_budget_pct is None:
        # An early REJECT resolved no strength tier, so there is no tier
        # budget to compose against — the honest answer is a note, not a
        # guessed tier (a fabricated budget would flow straight into the
        # shadow window's evidence).
        return {
            "note": (
                "no signal-strength tier was resolved (the engine rejected "
                "before sizing) — there is no tier budget to compose a v2 "
                "candidate from; a tier is never guessed"
            ),
            "mode": params.mode,
            "params": dataclasses.asdict(params),
        }

    def _es95_pct_nav() -> float | None:
        for row in api.get("es", []):
            if row.get("model") == "HISTORICAL" and row.get("confidence") == 0.95:
                return row.get("pct_nav")
        return None

    model_risk_block = api.get("model_risk")
    correlation_block = api.get("correlation_state")
    result = sizing_v2_shadow(
        es95_pct_nav=_es95_pct_nav(),
        correlation_state=(
            correlation_block.get("state") if correlation_block else None
        ),
        model_risk_state=(
            model_risk_block.get("state") if model_risk_block else None
        ),
        drawdown_current_pct=build.drawdown_api.get("current_pct"),
        regime_floor_pct=regime_floor_pct,
        tier_budget_pct=tier_budget_pct,
        vol_multiplier_used=vol_multiplier_used,
        params=params,
    )
    return {
        "es_modifier": result.es_modifier,
        "correlation_modifier": result.correlation_modifier,
        "model_health_modifier": result.model_health_modifier,
        "candidate_budget_pct": result.candidate_budget_pct,
        "budget_pct_used": result.budget_pct_used,
        "budget_delta_pct": result.budget_delta_pct,
        "risk_linked_cash_floor_pct": result.risk_linked_cash_floor_pct,
        "risk_linked_cash_floor_binds": result.risk_linked_cash_floor_binds,
        "regime_floor_pct": result.regime_floor_pct,
        "cash_floor_addons": dict(result.cash_floor_addons),
        "inputs": dict(result.inputs),
        "health": result.health,
        "reason": result.reason,
        "notes": list(result.notes),
        "mode": result.mode,
        "params": dataclasses.asdict(result.params),
        "note": (
            "SHADOW (§36/§37/§59): the ES, correlation and model-health "
            "modifiers the production budget does NOT compose, and the "
            "risk-linked cash floor. `budget_pct_used` and "
            "`regime_floor_pct` are what Tier 0 actually used; nothing here "
            "changed either."
        ),
    }



# ---------------------------------------------------------------------------
# Phase D — the pre-trade STRESS layer (design contract §8.5), SHADOW
# ---------------------------------------------------------------------------
#
# Same rules as Phase C above: the decision is already made, `extra_caps` is
# never populated, and a raise becomes a note. The only new thing is the
# instrument: the candidate is described as REVALUATION LEGS per unit of
# quantity, so the scenario engine reprices it exactly the way it reprices
# the open book — never as a delta-linear stand-in for an option.


#: Leg keys the candidate occupies in the proposed book. They carry the same
#: "#candidate" suffix the statistical layer uses (it cannot collide with a
#: real key, which is always "<TICKER>#<row id>") plus the leg role for a
#: spread's two legs, which `scenario_pnl` requires to be distinct.
CANDIDATE_LEG_KEY = f"candidate{CANDIDATE_KEY_SUFFIX}"
CANDIDATE_LONG_LEG_KEY = f"{CANDIDATE_LEG_KEY}:long"
CANDIDATE_SHORT_LEG_KEY = f"{CANDIDATE_LEG_KEY}:short"


def _candidate_stress_legs(
    ticker: str,
    *,
    instrument: str,
    chosen,
    chosen_spread,
    spot: float,
) -> tuple[list[StockLeg], list[OptionLeg]]:
    """The proposed entry as revaluation legs PER UNIT of quantity (§8.5).

    One unit is one share of stock or ONE CONTRACT of the selected option /
    spread — the same unit `assess` sized in, so scaling the legs by a
    quantity (``leg.scaled(q)``, design §8.3) reproduces the trade the engine
    approved.

    Anchoring is the chain's, exactly as the book's legs are anchored
    (risk_snapshot.stress_legs_from_book): ``mark0`` is the contract MID and
    ``iv0`` the PROVIDER's IV, so the basis ``mark0 − model(iv0)`` cancels
    and the zero scenario prices this candidate at exactly 0.0.

    Signs: stock ``+1`` share (``−1`` for a Phase 3 SHORT_STOCK); a long
    option ``+1`` contract; a spread ``+1`` long leg and ``−1`` short leg.

    A contract with no provider IV still becomes a leg, with the chain delta
    as ``delta0``: the scenario engine prices it DELTA_LINEAR and LABELS it.
    Dropping it instead would understate the proposed book's stress loss
    without saying so, which is the one thing this layer must never do.

    Raises ``ValueError`` (from the leg dataclasses) on a malformed contract;
    the caller turns that into an honest note.
    """
    if chosen_spread is not None:
        return [], [
            OptionLeg(
                key=CANDIDATE_LONG_LEG_KEY,
                ticker=ticker,
                right=chosen_spread.long_leg.right,
                strike=chosen_spread.long_leg.strike,
                t_years=max(chosen_spread.long_leg.dte, 0) / DAYS_PER_YEAR,
                quantity=1,
                spot0=spot,
                mark0=chosen_spread.long_leg.mid,
                iv0=chosen_spread.long_leg.iv,
                delta0=chosen_spread.long_leg.delta,
                multiplier=OPTION_MULTIPLIER,
                r=STRESS_RATE,
                q=STRESS_DIVIDEND_YIELD,
            ),
            OptionLeg(
                key=CANDIDATE_SHORT_LEG_KEY,
                ticker=ticker,
                right=chosen_spread.short_leg.right,
                strike=chosen_spread.short_leg.strike,
                t_years=max(chosen_spread.short_leg.dte, 0) / DAYS_PER_YEAR,
                quantity=-1,
                spot0=spot,
                mark0=chosen_spread.short_leg.mid,
                iv0=chosen_spread.short_leg.iv,
                delta0=chosen_spread.short_leg.delta,
                multiplier=OPTION_MULTIPLIER,
                r=STRESS_RATE,
                q=STRESS_DIVIDEND_YIELD,
            ),
        ]
    if chosen is not None:
        return [], [
            OptionLeg(
                key=CANDIDATE_LEG_KEY,
                ticker=ticker,
                right=chosen.right,
                strike=chosen.strike,
                t_years=max(chosen.dte, 0) / DAYS_PER_YEAR,
                quantity=1,
                spot0=spot,
                mark0=chosen.mid,
                iv0=chosen.iv,
                delta0=chosen.delta,
                multiplier=OPTION_MULTIPLIER,
                r=STRESS_RATE,
                q=STRESS_DIVIDEND_YIELD,
            ),
        ]
    quantity = -1 if instrument == InstrumentType.SHORT_STOCK.value else 1
    return [
        StockLeg(key=CANDIDATE_LEG_KEY, ticker=ticker, quantity=quantity, spot0=spot)
    ], []


def _worst_row_api(result) -> dict | None:
    """The worst scenario of a :class:`StressResult`, or ``None``.

    Only the fields a decision record needs: the audit row is not the place
    for the whole catalogue (that is persisted in ``stress_runs`` and served
    by the risk view). ``loss_usd`` is positive = money lost, the VaR/ES sign.
    """
    if result is None or result.worst is None:
        return None
    worst = result.worst
    return {
        "scenario": worst.name,
        "kind": worst.kind,
        "validated": worst.validated,
        "pnl_usd": worst.pnl_usd,
        "loss_usd": worst.loss_usd,
        "loss_pct_nav": (
            None if worst.pnl_pct_nav is None else -worst.pnl_pct_nav
        ),
        "method_coverage": dict(worst.method_coverage),
        "health": str(worst.health),
        "reason": worst.reason,
    }


def _pretrade_stress_shadow(
    build,
    *,
    ticker: str,
    instrument: str,
    chosen,
    chosen_spread,
    spot: float,
    approved_quantity: int,
    requested_quantity: int,
    nav: float,
    limits: StressLimits,
) -> tuple[dict, list]:
    """The Phase D ``shadow.statistical.stress`` block plus the STRESS caps.

    Returns ``(block, caps)``: the block goes into the audit / preview, the
    caps are merged into the SAME `shadow_verdict` the Phase C caps feed, so
    the hypothetical decision reflects EVERY shadow layer at once rather than
    two verdicts that disagree about the same trade.

    - ``worst_before``: the worst scenario over the CURRENT book — the build
      already ran the catalogue, so this is a read, not a second run;
    - ``worst_after``: the same catalogue over ``book + candidate ×
      approved_quantity``;
    - ``cap``: the STRESS_LOSS_LIMIT cap, or ``None`` when the limit is
      satisfied at the requested quantity (a satisfied limit produces no cap
      — Phase C's rule, kept);
    - ``hypothetical``: what the STRESS layer ALONE would allow.

    Honest degradation: no scenarios, a non-positive NAV, or an unpriceable
    candidate ⇒ an UNAVAILABLE block with the reason and NO cap (fail-open in
    SHADOW, the same open item Phase C recorded).
    """
    scenarios = build.scenarios
    worst_before = _worst_row_api(build.stress)
    if not scenarios:
        return (
            {
                "health": str(ModelHealth.UNAVAILABLE),
                "reason": "no stress scenarios in the catalogue for this book",
                "worst_before": worst_before,
                "worst_after": None,
                "cap": None,
                "hypothetical": None,
                "limits": dataclasses.asdict(limits),
                "mode": limits.mode,
            },
            [],
        )

    cand_stock, cand_option = _candidate_stress_legs(
        ticker,
        instrument=instrument,
        chosen=chosen,
        chosen_spread=chosen_spread,
        spot=spot,
    )

    # The proposed book at the APPROVED quantity: the book's own legs plus
    # the candidate's legs scaled to what Tier 0 allowed. `scaled(0)` is a
    # legitimate zero-quantity leg (a REJECT approves nothing), and its P&L
    # is exactly 0.0 — the honest answer, not a missing measurement.
    after = run_stress(
        [*build.stock_legs, *(leg.scaled(approved_quantity) for leg in cand_stock)],
        [*build.option_legs, *(leg.scaled(approved_quantity) for leg in cand_option)],
        scenarios,
        nav=nav if nav > 0 else None,
    )

    caps, caps_health, caps_reason = stress_caps(
        cand_stock,
        cand_option,
        build.stock_legs,
        build.option_legs,
        scenarios,
        requested_qty=requested_quantity,
        nav=nav,
        limits=limits,
    )
    verdict = shadow_verdict(approved_quantity, caps)
    return (
        {
            "health": str(caps_health),
            "reason": caps_reason,
            "worst_before": worst_before,
            "worst_after": _worst_row_api(after),
            "cap": _cap_api(caps[0]) if caps else None,
            "hypothetical": {
                "decision": verdict.hypothetical_decision,
                "quantity": verdict.hypothetical_quantity,
                "binding": list(verdict.binding),
                "mode": verdict.mode,
                "approved_quantity": approved_quantity,
                "note": (
                    "SHADOW: what the STRESS layer alone would have decided "
                    "at the Tier 0 approved quantity. It changed nothing "
                    "(spec §27's veto authority is the PRODUCTION promotion)."
                ),
            },
            "limits": dataclasses.asdict(limits),
            "mode": limits.mode,
            "n_candidate_stock_legs": len(cand_stock),
            "n_candidate_option_legs": len(cand_option),
        },
        list(caps),
    )

async def run_gate_chain(
    session: AsyncSession,
    ticker: str,
    quantity: int | None,
    direction: str = "AUTO",
    mode: Literal["research", "execution"] = "execution",
) -> GateChainResult:
    """Evaluate the §10 gate chain for a proposed entry (§33, §42).

    ``direction`` is the §9-style resolution seam: "AUTO" defers to the
    signal bias, an explicit "BULL"/"BEAR" wins (and is reported honestly in
    the DIRECTIONAL_SIGNAL detail). Places no order and never commits.
    Records exactly one SYSTEM RISK_DECISION audit event — veto or approval —
    on the session; the caller commits it in the same transaction as any
    state change (§38, rule 12).

    ``mode`` (upgrade 2026-08-12 §15/§16) — the research/execution split:

    - ``"research"`` (preview / Trade Plan): Trading Pool membership is an
      EXECUTION authorization, not a research prerequisite. The pool /
      per-symbol / kill-switch facts are still evaluated and reported in the
      preview's ``execution_authorization`` block, but they are NOT a gate
      and can NOT veto — a Watchlist-only symbol gets its full research
      plan (§15).
    - ``"execution"`` (approve): unchanged — TRADING_POOL_AUTHORIZATION is
      gate 1 and vetoes the chain (§21, §43: no Trading Pool bypass; no
      kill-switch bypass). Every order path keeps this mode.
    """
    settings = get_settings()
    limits = RiskLimits()
    regime_params = RegimeParams()
    # The §5/§8 account permissions, from Settings via the ONE factory (guide
    # §8) — the same object GET /api/config displays. The forbidden §33
    # capabilities can never arrive here True: Settings rejects them at
    # startup and AccountPermissions refuses them at construction.
    permissions = account_permissions_from_settings()

    gates: list[dict] = []

    def gate(name: str, status: str, detail: str) -> None:
        gates.append({"name": name, "status": status, "detail": detail})

    vetoed = False  # first FAIL stops evaluating later gates (§10, §42)
    sig = None
    signal_edge: float | None = None
    signal_bias: str | None = None
    strength: str | None = None
    resolved: DirectionalBias | None = None
    entry_price: float | None = None
    stop_distance: float | None = None
    assessment: RiskAssessment | None = None
    # Phase B SHADOW statistical view of the CURRENT book (contract §6). None
    # until the RISK_APPROVAL gate actually runs — an early veto measured
    # nothing, and a null says so rather than an empty object implying it did.
    statistical_shadow: dict | None = None
    # §14 EWMA vol-targeting side-by-side (Phase C contract §7.5), SHADOW.
    # None until the RISK_APPROVAL gate runs, for the same reason.
    vol_targeting_ewma_shadow: dict | None = None
    # Phase K §62-§67 event risk, SHADOW. None until the RISK_APPROVAL gate
    # runs, same reason again: an early veto sat through no print.
    event_shadow: dict | None = None
    # The SHADOW stress caps this run produced (Phase D §8.5). Declared here
    # so the §65 `stress_limit_blocks` counter at the audit site below can
    # read them on EVERY path — an early veto leaves the list empty, which is
    # the honest "the stress cap was never measured", not "it did not bind".
    stress_caps_shadow: list = []
    # §14 vol-targeting multiplier actually applied to the tier budget. None
    # until the RISK_APPROVAL gate computes it — an early veto never sized
    # anything, and a fabricated 1.0 would claim it did.
    budget_multiplier: float | None = None
    chain: list[ContractQuote] = []
    vol_regime = None  # IVRegime | None (§7; None = no chain data, honest null)
    decision = None  # InstrumentDecision | None (§8 matrix verdict)
    chosen: ContractQuote | None = None
    chosen_spread: SpreadCandidate | None = None  # §9 top-ranked candidate

    # ------------------------------------------------------------------
    # Execution authorization facts (§32, §18): pool membership, per-symbol
    # enablement, global kill switch. In EXECUTION mode they are gate 1 and
    # veto the chain; in RESEARCH mode they are reported in the preview's
    # execution_authorization block but never gate research (upgrade §15:
    # Trading Pool membership is an execution authorization, not a research
    # prerequisite — research approval ≠ execution approval, §20).
    # ------------------------------------------------------------------
    pool_row = (
        await session.execute(
            select(TradingPoolItem).where(TradingPoolItem.ticker == ticker)
        )
    ).scalar_one_or_none()
    state = await get_or_create_system_state(session)
    missing: list[str] = []
    if pool_row is None:
        missing.append(
            f"{ticker} is not in the Trading Pool — only pool symbols may trade (§32)"
        )
    elif not pool_row.trading_enabled:
        missing.append(f"trading is not enabled for {ticker} in the Trading Pool (§32)")
    if not state.trading_enabled:
        missing.append(
            "the global kill switch is engaged — trading is paused system-wide (§18)"
        )
    execution_authorization = {
        "authorized": not missing,
        "in_trading_pool": pool_row is not None,
        "symbol_trading_enabled": pool_row.trading_enabled if pool_row else False,
        "global_trading_enabled": state.trading_enabled,
        "missing": missing,
    }
    if mode == "execution":
        if missing:
            gate("TRADING_POOL_AUTHORIZATION", FAIL, "; ".join(missing))
            vetoed = True
        else:
            gate(
                "TRADING_POOL_AUTHORIZATION",
                PASS,
                f"{ticker} is in the Trading Pool with trading enabled and the "
                "global kill switch allows trading",
            )

    # ------------------------------------------------------------------
    # Gate 2 — DATA_QUALITY: stored bars exist (lazy backfill first, §4.2),
    # cover the regime engine's slow SMA, and are fresh (parameterized).
    # ------------------------------------------------------------------
    bars: list = []
    if vetoed:
        gate("DATA_QUALITY", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        problems: list[str] = []
        try:
            bars = await ensure_daily_bars(
                session, ticker, settings.market_data_provider
            )
        except HTTPException as exc:
            problems.append(f"no stored bars for {ticker}: {exc.detail}")
        age_days = 0
        if bars:
            if len(bars) < regime_params.sma_slow:
                problems.append(
                    f"only {len(bars)} stored bars; the regime engine needs at "
                    f"least {regime_params.sma_slow}"
                )
            last_ts = bars[-1].ts
            age_days = (datetime.now(timezone.utc).date() - last_ts).days
            if age_days > MAX_BAR_AGE_DAYS:
                problems.append(
                    f"last bar {last_ts.isoformat()} is {age_days} calendar days "
                    f"old (limit {MAX_BAR_AGE_DAYS})"
                )
        if problems:
            gate("DATA_QUALITY", FAIL, "; ".join(problems))
            vetoed = True
        else:
            gate(
                "DATA_QUALITY",
                PASS,
                f"{len(bars)} stored bars through {bars[-1].ts.isoformat()} "
                f"({age_days} calendar day(s) old, limit {MAX_BAR_AGE_DAYS})",
            )

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]
    if not vetoed:
        entry_price = closes[-1]
        atr_last = atr(highs, lows, closes, period=ATR_PERIOD)[-1]
        if atr_last is not None:
            stop_distance = ATR_STOP_MULTIPLE * atr_last
        # The live signal (§6.2), computed once and shared by the REGIME,
        # DIRECTIONAL_SIGNAL and INSTRUMENT gates below. The traded direction
        # resolves as in the §9 options view: explicit override wins, AUTO
        # defers to the bias.
        sig = score_direction(closes, highs, lows, volumes=volumes)
        signal_edge = sig.directional_edge
        signal_bias = sig.bias.value
        resolved = DirectionalBias(direction) if direction != "AUTO" else sig.bias

    # ------------------------------------------------------------------
    # Gate 3 — REGIME on the symbol's OWN bars (§6.1): TRANSITION defaults
    # to NO TRADE; a bear regime permits no new BULLISH exposure (§5) — but
    # a BEAR direction may trade it, via long puts only (§8; no short
    # stock exists in this system).
    # ------------------------------------------------------------------
    if vetoed:
        gate("REGIME", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        regime = classify_regime(closes, highs, lows, params=regime_params)
        cls = regime.classification
        if cls is MarketRegime.TRANSITION:
            gate("REGIME", FAIL, "regime TRANSITION defaults to NO TRADE (§6.1)")
            vetoed = True
        elif cls in _BEAR_REGIMES and resolved is not DirectionalBias.BEAR:
            gate(
                "REGIME",
                FAIL,
                f"{ticker} regime is {cls.value}: a bear regime does not permit "
                "a new long (bullish) entry (long-only account, §5); only a "
                "BEAR direction may trade here, via long puts (§8)",
            )
            vetoed = True
        elif cls in _BEAR_REGIMES:
            gate(
                "REGIME",
                PASS,
                f"{ticker} regime {cls.value} aligns with the BEAR direction — "
                "bearish exposure via long puts only (no short stock, §5)",
            )
        else:
            gate("REGIME", PASS, f"{ticker} regime {cls.value} permits a long entry")

    # ------------------------------------------------------------------
    # Gate 4 — DIRECTIONAL_SIGNAL (§6.2): under AUTO a NEUTRAL bias is NO
    # TRADE (§8 — a valid output); BULL and BEAR both proceed, because the
    # §8 matrix now maps BEAR to §5-legal long puts. An explicit direction
    # override passes with the real signal numbers reported honestly.
    # ------------------------------------------------------------------
    if vetoed:
        gate("DIRECTIONAL_SIGNAL", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        numbers = (
            f"edge {sig.directional_edge:+.1f} "
            f"(bull {sig.bull_score:.1f} / bear {sig.bear_score:.1f})"
        )
        if direction != "AUTO":
            gate(
                "DIRECTIONAL_SIGNAL",
                PASS,
                f"direction override {direction} — explicit direction wins, "
                f"as in the §9 options view (live signal: bias "
                f"{sig.bias.value}, {numbers})",
            )
        elif sig.bias is DirectionalBias.NEUTRAL:
            gate(
                "DIRECTIONAL_SIGNAL",
                FAIL,
                f"bias is NEUTRAL with {numbers}: no directional edge — "
                "NO TRADE is a valid output (§8)",
            )
            vetoed = True
        else:
            gate("DIRECTIONAL_SIGNAL", PASS, f"{sig.bias.value} bias with {numbers}")
    if not vetoed:
        # |edge| -> tier via the risk engine's single source of truth (§12.2);
        # None (below the weak threshold) makes the §8 matrix answer NO_TRADE.
        strength = strength_tier(signal_edge, limits)

    # ------------------------------------------------------------------
    # Gate 5 — VOLATILITY (§7): classify the REAL regime off today's chain
    # summary (shared helpers — one chain build for this whole chain run).
    # PASS carries the regime + features; FAIL only when the §8 matrix maps
    # this exact cell to NO_TRADE *because of vol* — detected by re-asking
    # the matrix with the no-information (treated-as-NORMAL) column: if the
    # same direction/strength would trade under NORMAL vol, vol is the veto.
    # No chain data -> PASS with an honest detail; the matrix then receives
    # vol_regime=None and documents the treated-as-NORMAL fallback itself.
    # ------------------------------------------------------------------
    if vetoed:
        gate("VOLATILITY", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        # Execution runs on LIVE data (§21/§42) — the chain cache is for
        # polling read surfaces only; research previews may reuse a
        # seconds-old build.
        chain_max_age = 0.0 if mode == "execution" else CHAIN_CACHE_TTL_SECONDS
        _, chain = build_option_chain(
            ticker, entry_price, max_age_seconds=chain_max_age
        )
        summary = chain_iv_summary(chain, entry_price, closes)
        atm_iv = summary["atm_iv"]
        rv20 = summary["rv20"]
        # atm_iv_daily (spec §24; audit §7.1): the chain read that computes
        # this ATM IV is the only place it exists — today it is discarded,
        # and empirical IV shocks / IV rank need the HISTORY. Persist it as
        # a side observation, best-effort: record_atm_iv never raises, so a
        # storage hiccup cannot touch this gate's verdict. INTERNALLY
        # CALCULATED provenance — `source` names the chain it came from, so
        # it is never mistaken for vendor IV history. No extra provider call.
        await record_atm_iv(
            session,
            ticker,
            bar_date=datetime.now(NEW_YORK).date(),
            atm_iv=atm_iv,
            spot=entry_price,
            source=f"{get_settings().market_data_provider}_chain",
        )
        if atm_iv is not None:
            vol_regime = classify_vol_regime(
                atm_iv,
                rv20 if rv20 is not None and rv20 > 0 else None,
                VOL_REGIME_PARAMS,
            ).regime
        # §8 matrix verdict — computed HERE because this gate's PASS/FAIL is
        # defined by whether vol alone turned the cell into NO_TRADE.
        decision = select_instrument(
            resolved, strength, vol_regime, permissions
        )
        if vol_regime is None:
            gate(
                "VOLATILITY",
                PASS,
                "no chain data — ATM IV unavailable (honest null); the §8 "
                "matrix treats the unknown vol regime as NORMAL "
                "(no-information column)",
            )
        else:
            vol_numbers = (
                f"atm_iv {atm_iv:.3f}, rv20 "
                + (f"{rv20:.3f}" if rv20 is not None else "n/a")
                + ", iv/rv "
                + (
                    f"{atm_iv / rv20:.2f}"
                    if rv20 is not None and rv20 > 0
                    else "n/a"
                )
            )
            normal_decision = select_instrument(
                resolved, strength, None, permissions
            )
            vol_caused_no_trade = (
                decision.instrument is InstrumentType.NO_TRADE
                and normal_decision.instrument is not InstrumentType.NO_TRADE
            )
            if vol_caused_no_trade:
                gate(
                    "VOLATILITY",
                    FAIL,
                    f"vol regime {vol_regime.value} ({vol_numbers}) maps this "
                    f"§8 cell to NO_TRADE — the same direction/strength would "
                    f"trade under NORMAL vol: {' '.join(decision.rationale)}",
                )
                vetoed = True
            else:
                gate(
                    "VOLATILITY",
                    PASS,
                    f"vol regime {vol_regime.value} ({vol_numbers})",
                )

    # ------------------------------------------------------------------
    # Gate 6 — INSTRUMENT: the §8 matrix verdict under the §5 account
    # constraints (select_instrument — never reimplemented here). FAIL when
    # NO_TRADE, with the §8 cell + degradation rationale in the detail;
    # NO_TRADE is a valid output, never an error (§8).
    # ------------------------------------------------------------------
    if vetoed:
        gate("INSTRUMENT", SKIPPED, SKIP_EARLIER_FAIL)
    elif decision.instrument is InstrumentType.NO_TRADE:
        gate(
            "INSTRUMENT",
            FAIL,
            f"§8 matrix verdict NO_TRADE: {' '.join(decision.rationale)}",
        )
        vetoed = True
    else:
        gate(
            "INSTRUMENT",
            PASS,
            f"§8 matrix selects {decision.instrument.value}: "
            f"{' '.join(decision.rationale)}",
        )

    # ------------------------------------------------------------------
    # Gate 6b — SQUEEZE_RISK, REPORT mode (auto-strategy Phase D). A PROXY:
    # real short-interest/borrow/float data is unavailable from every
    # configured provider (§33 vendor needed — execution-chains-roadmap.md),
    # so this measures the honest stand-ins computable from bars the chain
    # ALREADY holds: volume z-score, trailing-high proximity, overnight
    # gap-up. Fires only for a SHORT_STOCK candidate; always PASSes; the
    # detail names its proxy nature and the RISK_DECISION audit mirrors the
    # report under shadow.squeeze — the same promotion discipline as the
    # LIQUIDITY gate (validate on the watchlist before any veto).
    # ------------------------------------------------------------------
    squeeze_report = None
    if vetoed:
        gate("SQUEEZE_RISK", SKIPPED, SKIP_EARLIER_FAIL)
    elif decision.instrument is not InstrumentType.SHORT_STOCK:
        gate(
            "SQUEEZE_RISK",
            PASS,
            "not a short-stock candidate — squeeze hazard not applicable",
        )
    else:
        squeeze_report = assess_squeeze_proxy(
            [b.open for b in bars], closes, highs, volumes, SQUEEZE_PROXY_PARAMS
        )
        hypothetical = "ELEVATED" if squeeze_report.elevated else "NORMAL"
        z_txt = "n/a" if squeeze_report.volume_z is None else f"{squeeze_report.volume_z:.2f}"
        hi_txt = (
            "n/a" if squeeze_report.dist_from_high_pct is None
            else f"{squeeze_report.dist_from_high_pct:.2f}%"
        )
        gap_txt = (
            "n/a" if squeeze_report.overnight_gap_pct is None
            else f"{squeeze_report.overnight_gap_pct:.2f}%"
        )
        detail = (
            f"PROXY (no short-interest/borrow/float data — §33 vendor needed): "
            f"volume z {z_txt}, dist from trailing high {hi_txt}, "
            f"overnight gap {gap_txt} -> hypothetical {hypothetical}; "
            + "; ".join(squeeze_report.reasons)
        )
        # REPORT mode: never FAIL / never veto in this phase.
        gate("SQUEEZE_RISK", PASS, detail)

    # ------------------------------------------------------------------
    # Gate 7 — LIQUIDITY of the UNDERLYING, REPORT mode (risk-engine audit
    # §7.3 / §10 B0; spec §2/§5 hard limit, §70 shadow). Measured from data
    # the chain ALREADY holds — the stored daily volumes DATA_QUALITY loaded
    # (ADV20), the requested SHARE count for a stock candidate (an option
    # candidate's contracts are not translated to shares — honest null), and
    # the live stock NBBO ONLY if the in-process stream cache holds a fresh
    # one (data_source.md §5; no new provider call — absent = unmeasured).
    # The gate always PASSes here; the detail carries the numbers and the
    # HYPOTHETICAL verdict, and the RISK_DECISION audit records the same
    # report under `shadow.liquidity`, so the research defaults can be
    # validated against the watchlist before any veto is enabled (Q3).
    # Option-leg liquidity is enforced per contract by the §9.1 filters
    # inside CONTRACT_SELECTION, unchanged.
    # ------------------------------------------------------------------
    liquidity_report = None
    liquidity_order_shares: int | None = None
    liquidity_bid: float | None = None
    liquidity_ask: float | None = None
    liquidity_stock_candidate = False
    if vetoed:
        gate("LIQUIDITY", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        liquidity_stock_candidate = decision.instrument in (
            InstrumentType.LONG_STOCK,
            InstrumentType.SHORT_STOCK,
        )
        liquidity_order_shares = quantity if liquidity_stock_candidate else None
        streamed_quote = market_stream.CACHE.fresh(
            ticker, market_stream.STREAM_FRESH_SECONDS
        )
        if streamed_quote is not None:
            liquidity_bid = streamed_quote.bid
            liquidity_ask = streamed_quote.ask
        liquidity_report = evaluate_underlying_liquidity(
            volumes,
            liquidity_order_shares,
            liquidity_bid,
            liquidity_ask,
            LIQUIDITY_LIMITS,
        )
        liquidity_detail = liquidity_report_detail(liquidity_report, LIQUIDITY_LIMITS)
        if not liquidity_stock_candidate:
            liquidity_detail += (
                " (option candidate: contracts not translated to shares, so "
                "participation is unmeasured)"
            )
        elif liquidity_order_shares is None:
            liquidity_detail += (
                " (no requested share count; participation is re-measured at "
                "the approved quantity in the audit's shadow.liquidity block)"
            )
        if streamed_quote is None:
            liquidity_detail += " (no fresh streamed stock NBBO — spread unmeasured)"
        # REPORT mode: never FAIL / never veto in this phase.
        gate("LIQUIDITY", PASS, liquidity_detail)

    # ------------------------------------------------------------------
    # Gate 8 — CONTRACT_SELECTION (§9): SKIPPED for stock; for an option
    # instrument the §9 selector runs over the SAME chain and the top-ranked
    # candidate becomes the proposed contract; FAIL when nothing is eligible.
    # ------------------------------------------------------------------
    if vetoed:
        gate("CONTRACT_SELECTION", SKIPPED, SKIP_EARLIER_FAIL)
    elif decision.instrument in (
        InstrumentType.LONG_STOCK,
        InstrumentType.SHORT_STOCK,
    ):
        gate("CONTRACT_SELECTION", SKIPPED, SKIP_STOCK_ORDER)
    elif decision.instrument in (
        InstrumentType.BULL_CALL_SPREAD,
        InstrumentType.BEAR_PUT_SPREAD,
    ):
        # §9-S: long leg = §9 rank-1 unchanged; short leg width-targeted from
        # the same expiry (roadmap Phase 1).
        selection = select_vertical_spread(
            chain,
            decision.instrument.value,
            entry_price,  # last stored underlying close = the chain's spot
            SELECTOR_PARAMS,
            SPREAD_PARAMS,
        )
        if selection.candidate is None:
            gate(
                "CONTRACT_SELECTION",
                FAIL,
                f"no eligible {decision.instrument.value} in today's chain: "
                + "; ".join(selection.fail_reasons),
            )
            vetoed = True
        elif selection.candidate.net_delta is None:
            # §16 needs the NET greeks; a short leg without greeks cannot be
            # risk-checked — fail closed, never zero-fill.
            gate(
                "CONTRACT_SELECTION",
                FAIL,
                "short leg greeks not provided by the chain — §16 net "
                "exposure cannot be computed; fail closed.",
            )
            vetoed = True
        else:
            chosen_spread = selection.candidate
            gate(
                "CONTRACT_SELECTION",
                PASS,
                " ".join(chosen_spread.rationale),
            )
    else:
        side = "BULL" if decision.instrument is InstrumentType.LONG_CALL else "BEAR"
        scored = select_contracts(chain, side, SELECTOR_PARAMS)
        top = next((s for s in scored if s.rank == 1), None)
        if top is None:
            eligible = sum(1 for s in scored if s.eligible)
            gate(
                "CONTRACT_SELECTION",
                FAIL,
                f"no eligible {decision.instrument.value} contract in today's "
                f"chain ({len(chain)} contracts, {eligible} eligible after "
                "the §9.1 filters)",
            )
            vetoed = True
        else:
            chosen = top.contract
            gate(
                "CONTRACT_SELECTION",
                PASS,
                f"top-ranked §9 candidate: {chosen.right} {chosen.strike:g} "
                f"exp {chosen.expiry.isoformat()} (dte {chosen.dte}), mid "
                f"{chosen.mid:.4f}, delta {chosen.delta:+.2f}, iv "
                f"{chosen.iv:.3f}, score {top.score:.3f}",
            )

    # ------------------------------------------------------------------
    # Gate 9 — RISK_APPROVAL: build the portfolio snapshot and call the risk
    # engine (§12, §17). Risk limits have PRIORITY over strategy confidence
    # (§44 rule 20) — a REJECT here vetoes regardless of signal strength.
    #
    # §12.1 OPTIONS FORMULA: for an option entry the request carries
    # CONTRACT-level units — entry_price = stop_distance = mid * 100,
    # because the premium is FULLY at risk. approved_quantity is therefore
    # the number of CONTRACTS and every existing cap applies unchanged.
    #
    # §14 CASH-ACCOUNT SIZING: with a REAL broker configured the broker's
    # account is fetched BEFORE sizing and the snapshot's cash is clamped to
    # min(local portfolio cash, broker cash) — usable_capital =
    # min(platform_cash_available, broker_cash_compatible_amount). If the
    # fetch fails, this gate FAILS CLOSED (guide §28): deployable cash
    # cannot be verified, so nothing is sized from local cash alone.
    # ------------------------------------------------------------------
    # Fetched before the risk gate runs; (None, None) in simulated/unset
    # modes, which stay exactly as they were.
    broker_cash: float | None = None
    broker_cash_error: str | None = None
    if not vetoed:
        broker_cash, broker_cash_error = await _broker_cash_for_sizing()
    if vetoed:
        gate("RISK_APPROVAL", SKIPPED, SKIP_EARLIER_FAIL)
    elif broker_cash_error is not None:
        # FAIL CLOSED (guide §28): a real broker is configured but its
        # account could not be read — a 502-class broker fault, surfaced as
        # a risk veto because an unverifiable cash balance is a risk fact.
        gate(
            "RISK_APPROVAL",
            FAIL,
            "BROKER_ERROR (502-class fault): the broker account could not "
            f"be fetched ({broker_cash_error}) — deployable cash cannot be "
            "verified, and sizing from local cash alone is forbidden with a "
            "real broker configured (§14: usable capital = min(platform "
            "cash, broker cash)). Failing closed (§28); no order may be "
            "sized or placed until the broker answers.",
        )
        vetoed = True
    else:
        if chosen is not None:
            risk_entry = risk_stop = chosen.mid * OPTION_MULTIPLIER
        elif chosen_spread is not None:
            # §12.1 defined risk: the NET debit is fully at risk — both the
            # entry basis and the stop basis are net_debit * 100 per spread.
            risk_entry = risk_stop = chosen_spread.net_debit * OPTION_MULTIPLIER
        elif (
            decision is not None
            and decision.instrument is InstrumentType.SHORT_STOCK
        ):
            # §12.1 Phase 3: short-stock risk per share = stop_distance ×
            # gap factor (loss above the stop is possible and unbounded).
            risk_entry = entry_price
            risk_stop = stop_distance * SHORT_STOCK_GAP_RISK_FACTOR
        else:
            risk_entry, risk_stop = entry_price, stop_distance
        # §14: the cash the sizing engine may deploy is the broker's reported
        # CASH — never buying_power (which may include margin-like capacity
        # beyond the cash-account model). THE PLATFORM KEEPS NO LOCAL COPY of
        # a real account: with a real broker the live number just fetched IS
        # the usable cash. Only the dev/test simulator (no broker to ask)
        # sizes from its own ledger. NAV follows that cash so the §13 floor
        # and every %-of-NAV cap measure the capital that actually exists.
        if broker_cash is not None:
            account_cash = broker_cash
        else:
            portfolio = await get_or_create_portfolio(session)
            account_cash = portfolio.cash
        # The snapshot itself is built by the ONE shared helper (audit §10
        # Phase B0, plan §21) that the income opens also use: open positions
        # + stored closes -> market values -> PositionRisk rows -> SPY regime,
        # with collateral already pledged to open cash-secured puts NETTED
        # out of the usable cash (audit §8 item 3) — pledged cash is not
        # deployable, so neither the §13 floor nor NAV may count it.
        inputs = await build_portfolio_snapshot(
            session, cash=account_cash, trading_enabled=state.trading_enabled
        )
        snapshot = inputs.snapshot
        pairs = inputs.pairs
        usable_cash, nav = inputs.usable_cash, snapshot.nav
        # §14 vol targeting: the SAME computation the portfolio risk view
        # reports (shared helper — one implementation, plan §21). The
        # multiplier scales the §12.2 tier budget INSIDE assess, which
        # hard-caps it at abs_max_trade_risk — vol targeting can NEVER
        # override hard caps (§14, §44 rule 20). No open positions -> 1.0.
        closes_by_ticker = await stored_closes_by_ticker(
            session, (pos.ticker for pos, _price in pairs)
        )
        vol_block = vol_targeting_block(nav, pairs, closes_by_ticker)
        budget_multiplier = vol_block["multiplier"]
        # §16 portfolio greek limits: the current book's aggregated greeks
        # (shared helper) plus the candidate's per-share greeks — stock is
        # delta 1 / gamma 0 / theta 0 / vega 0; an option candidate carries
        # the proposed §9 contract's greeks. assess checks the post-trade
        # book at the APPROVED quantity; a breach REJECTs like any other
        # risk limit (§44 rule 20).
        book_greeks, _greek_rows = portfolio_greeks_read(pairs)
        if chosen is not None:
            candidate_greeks = PositionGreeksInput(
                ticker=ticker,
                instrument=decision.instrument.value,
                quantity=1,  # requested basis; assess scales by approved qty
                multiplier=OPTION_MULTIPLIER,
                spot=entry_price,
                delta=chosen.delta,
                gamma=chosen.gamma,
                theta_per_day=chosen.theta,
                vega=chosen.vega,
            )
        elif chosen_spread is not None:
            # §16: the spread's NET greeks (long − short, guaranteed present
            # by the CONTRACT_SELECTION fail-closed check above).
            candidate_greeks = PositionGreeksInput(
                ticker=ticker,
                instrument=decision.instrument.value,
                quantity=1,
                multiplier=OPTION_MULTIPLIER,
                spot=entry_price,
                delta=chosen_spread.net_delta,
                gamma=chosen_spread.net_gamma,
                theta_per_day=chosen_spread.net_theta,
                vega=chosen_spread.net_vega,
            )
        else:
            # Stock: delta ±1 per share (−1 for a Phase 3 short).
            stock_short = (
                decision is not None
                and decision.instrument is InstrumentType.SHORT_STOCK
            )
            candidate_greeks = PositionGreeksInput(
                ticker=ticker,
                instrument=(
                    InstrumentType.SHORT_STOCK.value
                    if stock_short
                    else InstrumentType.LONG_STOCK.value
                ),
                quantity=1,  # requested basis; assess scales by approved qty
                multiplier=1,
                spot=entry_price,
                delta=-1.0 if stock_short else 1.0,
                gamma=0.0,
                theta_per_day=0.0,
                vega=0.0,
            )
        assessment = assess(
            RiskRequest(
                ticker=ticker,
                entry_price=risk_entry,
                stop_distance=risk_stop,
                edge=signal_edge,
                quantity_requested=quantity,
            ),
            snapshot,
            limits,
            budget_multiplier=budget_multiplier,
            portfolio_greeks=book_greeks,
            new_position_greeks=candidate_greeks,
        )
        # §14 transparency: when vol targeting actually scaled the budget the
        # gate detail says so, with the real number.
        vt_detail = (
            f", budget multiplier {budget_multiplier:.2f} (vol targeting)"
            if abs(budget_multiplier - 1.0) > 1e-9
            else ""
        )
        # §14 transparency, cash side: with a real broker the detail names
        # the LIVE deployable cash — read from the broker moments ago, never
        # buying power, never a local copy — so the sizing basis is auditable.
        cash_detail = (
            (
                f", deployable cash ${usable_cash:,.2f} = the broker "
                "account's LIVE cash (§14: broker CASH, never buying power; "
                "the platform stores no copy)"
                + (
                    f" minus ${inputs.cash_reserved_total:,.2f} pledged to "
                    "open cash-secured puts"
                    if inputs.cash_reserved_total > 0
                    else ""
                )
            )
            if broker_cash is not None
            else ""
        )
        unit = "contracts" if chosen is not None else "shares"
        if assessment.decision is RiskDecision.REJECT:
            gate(
                "RISK_APPROVAL",
                FAIL,
                f"risk engine REJECT ({', '.join(assessment.reason_codes)}) — "
                "risk limits have priority over strategy confidence "
                f"(§44 rule 20){vt_detail}{cash_detail}",
            )
            vetoed = True
        elif assessment.decision is RiskDecision.APPROVE_WITH_RESIZE:
            gate(
                "RISK_APPROVAL",
                PASS,
                f"APPROVE_WITH_RESIZE: quantity resized to "
                f"{assessment.approved_quantity} {unit} "
                f"({', '.join(assessment.reason_codes)}){vt_detail}{cash_detail}",
            )
        else:
            gate(
                "RISK_APPROVAL",
                PASS,
                f"APPROVE: {assessment.approved_quantity} {unit}, "
                f"${assessment.trade_risk_usd:,.2f} at risk{vt_detail}{cash_detail}",
            )

        # --- Phase B statistical layer, SHADOW (design contract §6) --------
        # The CURRENT book measured with the same cash the Tier 0 snapshot
        # above used, so the shadow numbers describe the exact portfolio the
        # decision was made against, and — Phase C (§7.5) — the PROPOSED
        # book: this candidate added at the approved quantity, with the
        # hypothetical statistical caps and the verdict they imply.
        #
        # THE DECISION IS ALREADY MADE. This runs strictly AFTER `assess`,
        # touches neither `assessment` nor `vetoed` nor any gate, passes NO
        # `extra_caps` to the engine, and its failure mode is a note in the
        # audit — never a changed outcome (§70: a SHADOW model cannot veto).
        # Persisted only in execution mode: a preview is a research read that
        # should not litter the snapshot table (and its transaction is rolled
        # back anyway).
        try:
            shadow_build = await build_risk_snapshot(
                session,
                trigger=TRIGGER_PRE_TRADE,
                cash=account_cash,
                trading_enabled=state.trading_enabled,
                persist=mode == "execution",
            )
        except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
            statistical_shadow = {"note": f"{type(exc).__name__}: {exc}"}
        else:
            statistical_shadow = _statistical_shadow_detail(shadow_build)
            # --- Phase D: the STRESS layer (contract §8.5), SHADOW --------
            # Runs FIRST so its cap can join the ONE hypothetical verdict
            # Phase C computes below (a trade gets one shadow answer, not
            # two). It reads the legs and the catalogue the snapshot build
            # already assembled — the same book, the same scenarios. A raise
            # leaves `stress_caps_shadow` empty and records `stress` as a
            # note: the Phase C verdict is then exactly what it was before
            # Phase D existed, and Tier 0 is untouched either way.
            stress_shadow: dict | None = None
            stress_caps_shadow = []
            try:
                stress_shadow, stress_caps_shadow = _pretrade_stress_shadow(
                    shadow_build,
                    ticker=ticker,
                    instrument=(
                        decision.instrument.value
                        if decision is not None
                        else InstrumentType.LONG_STOCK.value
                    ),
                    chosen=chosen,
                    chosen_spread=chosen_spread,
                    # The UNDERLYING's spot anchors every leg (the option
                    # legs carry their own strike/mark from the chain).
                    spot=entry_price,
                    approved_quantity=assessment.approved_quantity,
                    requested_quantity=(
                        quantity
                        if quantity is not None
                        else assessment.approved_quantity
                    ),
                    nav=nav,
                    limits=STRESS_LIMITS,
                )
            except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
                stress_shadow = {"note": f"{type(exc).__name__}: {exc}"}
                stress_caps_shadow = []
            # --- Phase K: the EVENT-RISK layer (§62-§67), SHADOW ----------
            # The §63 state of the print this candidate would sit through,
            # and — exactly like the STRESS cap above — the HYPOTHETICAL cap
            # it implies. That cap joins the SAME `extra_caps` list and the
            # SAME single shadow verdict below; it is NEVER passed to
            # `assess`, whose `extra_caps` stays empty (§65: no backtest has
            # validated these thresholds, so promotion is a separate human
            # step). The seam reads stored rows only — no provider call, no
            # LLM, no clock beyond `now` — and a raise here leaves
            # `event_caps_shadow` empty, so the Phase C verdict is exactly
            # what it was before Phase K existed and Tier 0 is untouched
            # either way.
            event_caps_shadow: list = []
            try:
                event_shadow, event_caps_shadow = await event_risk.shadow_event_block(
                    session,
                    ticker=ticker,
                    requested_qty=(
                        quantity
                        if quantity is not None
                        else assessment.approved_quantity
                    ),
                    # The UNDERLYING's spot is the exposure basis, matching
                    # the statistical candidate: the event moves the stock,
                    # and a percent-of-NAV ceiling is measured in shares of
                    # it, never in option premium.
                    price=entry_price,
                    nav=nav,
                    # What is ALREADY held in this name and would sit through
                    # the print — the candidate itself is measured by the cap
                    # against `requested_qty`, so counting it here too would
                    # double-count it. None (never 0.0) when nothing is held.
                    position_exposure_usd=await event_risk.position_exposure_for(
                        session, ticker
                    ),
                    # §16: the SAME per-position greek rows the risk engine
                    # judged the book with (`portfolio_greeks_read` above),
                    # so the §66 sensitivity describes the position the user
                    # sees on the greeks page. `data_ok: false` rows are
                    # skipped by the seam rather than read as zeros.
                    option_greeks=event_risk.greeks_from_rows(_greek_rows, ticker),
                )
            except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
                event_shadow = {"error": f"{type(exc).__name__}: {exc}"}
                event_caps_shadow = []
            # --- Phase C: the PROPOSED-book comparison (contract §7.5) ----
            # Measured at the quantity Tier 0 actually approved (and at the
            # requested one when they differ), against the book the snapshot
            # above just built. A REJECT approves 0 units — there is no
            # proposed book to compare, so the layer says so rather than
            # comparing a zero-size trade.
            try:
                statistical_shadow.update(
                    _pretrade_statistical_shadow(
                        shadow_build,
                        _candidate_spec(
                            ticker,
                            instrument=(
                                decision.instrument.value
                                if decision is not None
                                else InstrumentType.LONG_STOCK.value
                            ),
                            chosen=chosen,
                            chosen_spread=chosen_spread,
                            # The UNDERLYING's spot drives the candidate's
                            # returns for every instrument (its P&L is
                            # delta-linear in the underlying, contract §2.9)
                            # — never the option premium.
                            spot=entry_price,
                            risk_stop=risk_stop,
                            risk_entry=risk_entry,
                            # `quantity=None` means "no user cap — size from
                            # the budget" (engine.py step 4), NOT "zero": the
                            # cap search's upper bound is then what Tier 0
                            # actually sized, never a 0 that would collapse
                            # every bisection to an empty interval.
                            quantity_requested=(
                                quantity
                                if quantity is not None
                                else assessment.approved_quantity
                            ),
                        ),
                        approved_quantity=assessment.approved_quantity,
                        requested_quantity=(
                            quantity
                            if quantity is not None
                            else assessment.approved_quantity
                        ),
                        nav=nav,
                        heat_before=assessment.heat_before_pct,
                        heat_after=assessment.heat_after_pct,
                        cash_before=usable_cash / nav if nav > 0 else 0.0,
                        cash_after=assessment.cash_after_pct,
                        delta_notional_before=book_greeks.delta_adjusted_notional,
                        # §46: the book's net vega from the SAME aggregated
                        # greeks object Tier 0 judged the book with, so the
                        # shadow row and the greek gate read one number.
                        net_vega_before=book_greeks.net_vega,
                        # §55: judged against the SAME instant the rest of
                        # this decision used, not a second `now()` that
                        # could straddle the TTL boundary.
                        snapshot_now=datetime.now(timezone.utc),
                        buckets=limits.correlation_buckets,
                        limits=STATISTICAL_LIMITS,
                        # Phase D §8.5: the STRESS cap joins the SAME caps
                        # list and the SAME verdict, carrying its own layer —
                        # and Phase K's EVENT cap joins it the same way, on
                        # the CONCENTRATION layer. This is the HYPOTHETICAL
                        # parameter of a SHADOW helper; `assess` above still
                        # receives no `extra_caps` at all.
                        extra_caps=[*stress_caps_shadow, *event_caps_shadow],
                        stress_block=stress_shadow,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
                # `comparison_note`, never `note`: the Phase B note above
                # describes the CURRENT-book block and stays true.
                statistical_shadow["comparison_note"] = f"{type(exc).__name__}: {exc}"
            # Phase D §8.5: `shadow.statistical.stress`. Attached AFTER the
            # Phase C update so a `.update()` can never drop it, and present
            # on every path — including the one where the comparison raised.
            statistical_shadow["stress"] = stress_shadow
            # --- §36/§37/§59 sizing v2 (compliance §3 Tier A), SHADOW ----
            # Composed from values ALREADY at hand: the shadow snapshot's
            # ES-95 / model-risk / correlation / drawdown blocks, Tier 0's
            # own regime cash floor, and the two factors the engine's budget
            # was built from. `assess` above already ran with the UNCHANGED
            # `budget_multiplier` and the UNCHANGED `limits.cash_floors`; a
            # raise here is a note, exactly like every other shadow layer.
            try:
                # The tier budget the engine composed against — read from the
                # SAME `limits` object `assess` used, keyed by the strength
                # tier it resolved. `None` when it rejected before sizing.
                _tier_budget_pct = (
                    {
                        "VERY_STRONG": limits.budget_very_strong,
                        "STRONG": limits.budget_strong,
                        "MODERATE": limits.budget_moderate,
                        "WEAK": limits.budget_weak,
                    }.get(assessment.signal_strength)
                    if assessment.signal_strength is not None
                    else None
                )
                statistical_shadow["sizing_v2"] = _sizing_v2_shadow_block(
                    shadow_build,
                    # Tier 0's regime cash floor — the one actually in force
                    # for this decision (engine step 5e), never a re-derived
                    # one that could silently disagree.
                    regime_floor_pct=limits.cash_floors[snapshot.regime],
                    tier_budget_pct=_tier_budget_pct,
                    vol_multiplier_used=budget_multiplier,
                    params=SIZING_V2_PARAMS,
                )
            except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
                statistical_shadow["sizing_v2"] = {
                    "note": f"{type(exc).__name__}: {exc}"
                }
            # §14 EWMA side-by-side (contract §7.5), SHADOW: the EWMA
            # forecast off the SAME book P&L series the snapshot measured,
            # logged NEXT TO the crude proxy that actually scaled the budget
            # (`budget_multiplier` above is unchanged and stays in force).
            try:
                ewma_sigma, ewma_multiplier = _ewma_vol_targeting_side_by_side(
                    nav,
                    list(shadow_build.book.total)
                    if shadow_build.book is not None
                    else None,
                )
                vol_targeting_ewma_shadow = {
                    "forecast": ewma_sigma,
                    "multiplier": ewma_multiplier,
                    "multiplier_in_force": budget_multiplier,
                    "note": (
                        "SHADOW side-by-side (§14): EWMA sigma of the book "
                        "P&L annualized as a fraction of NAV. The multiplier "
                        "that scaled the budget is `multiplier_in_force` (the "
                        "crude v0 proxy) — this one changed nothing."
                    ),
                }
            except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
                vol_targeting_ewma_shadow = {"note": f"{type(exc).__name__}: {exc}"}
            # The §19 correlation regime of the book PLUS this candidate's
            # ticker — the correlation that would exist AFTER the trade,
            # which is the one a concentration question is about.
            try:
                statistical_shadow["correlation_state"] = correlation_state_api(
                    correlation_state_for(
                        await stored_bars_by_ticker(
                            session,
                            {p.ticker for p in shadow_build.positions} | {ticker},
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001 — SHADOW must never veto
                statistical_shadow["correlation_state"] = None
                statistical_shadow["correlation_note"] = f"{type(exc).__name__}: {exc}"

    # §33: aggregate the passing/failing gate details + risk explanations;
    # both lists are ALWAYS present.
    why_trade = [f"{g['name']}: {g['detail']}" for g in gates if g["status"] == PASS]
    why_not_trade = [f"{g['name']}: {g['detail']}" for g in gates if g["status"] == FAIL]
    if assessment is not None:
        if assessment.decision is RiskDecision.REJECT:
            why_not_trade.extend(assessment.explanations)
        else:
            why_trade.extend(assessment.explanations)

    # Exactly ONE RISK_DECISION audit event per chain run — even an early veto
    # is a decision and must be auditable (§38, rule 12). Recorded on the
    # session only; the CALLER commits, so the event shares one transaction
    # with any state change (preview: none; approve: the fill).
    veto_gate = next((g["name"] for g in gates if g["status"] == FAIL), None)
    # `shadow.liquidity` (audit §7.3 / §70): the gate-7 REPORT-mode report
    # verbatim (asdict) plus the share basis it measured; for a stock
    # candidate the participation is ALSO re-measured at the APPROVED
    # quantity (the order that would actually be sent), so the shadow window
    # records what the promoted gate would have seen. None when the chain
    # vetoed before gate 7 (nothing was measured — honest null).
    liquidity_shadow: dict | None = None
    if liquidity_report is not None:
        liquidity_shadow = {
            **dataclasses.asdict(liquidity_report),
            "order_shares": liquidity_order_shares,
            "at_approved_quantity": None,
        }
        if (
            liquidity_stock_candidate
            and assessment is not None
            and assessment.decision is not RiskDecision.REJECT
            and assessment.approved_quantity > 0
        ):
            at_approved = evaluate_underlying_liquidity(
                volumes,
                assessment.approved_quantity,
                liquidity_bid,
                liquidity_ask,
                LIQUIDITY_LIMITS,
            )
            liquidity_shadow["at_approved_quantity"] = {
                **dataclasses.asdict(at_approved),
                "order_shares": assessment.approved_quantity,
            }
    # --- §65 decision-path telemetry (compliance §3 Tier A) ---------------
    #
    # Incremented HERE, at the one place a RISK_DECISION is recorded, so the
    # counters and the audit trail can never tell different stories. They are
    # pure observation: nothing below reads them, and no branch above depends
    # on them.
    #
    # - `risk_resize_count`: the engine cut the quantity it was asked for;
    # - `risk_reject_count`: no order came out of this run — either the
    #   engine REJECTed, or an earlier gate vetoed so `assess` never ran
    #   (the audit's own `decision` is then the literal "VETOED"). Both are
    #   "the chain refused", which is the rate §65 wants alertable;
    # - `stress_limit_blocks`: the SHADOW stress cap would have BOUND. The
    #   comparison basis is the quantity that actually stood at the end of
    #   the chain — the approved quantity, or the requested one when the
    #   engine never produced an approval — because a cap "binds" only
    #   against a quantity someone was going to trade.
    if assessment is not None and assessment.decision is RiskDecision.APPROVE_WITH_RESIZE:
        RISK_RESIZE_COUNT.inc()
    if assessment is None or assessment.decision is RiskDecision.REJECT:
        RISK_REJECT_COUNT.inc()
    _stress_basis_qty = (
        assessment.approved_quantity
        if assessment is not None
        else (quantity if quantity is not None else 0)
    )
    if _stress_basis_qty > 0 and any(
        cap.cap_qty < _stress_basis_qty for cap in stress_caps_shadow
    ):
        STRESS_LIMIT_BLOCKS.inc()
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.RISK_DECISION,
        entity_type="order_preview",
        entity_id=ticker,
        details={
            "decision": assessment.decision.value if assessment is not None else "VETOED",
            "mode": mode,
            "execution_authorized": execution_authorization["authorized"],
            "veto_gate": veto_gate,
            "gates": {g["name"]: g["status"] for g in gates},
            "reason_codes": (
                list(assessment.reason_codes)
                if assessment is not None
                else ([f"VETO_{veto_gate}"] if veto_gate else [])
            ),
            # ADDITIVE (Phase B contract §6): what was asked for versus what
            # the engine allowed, and the exact parameter set that decided
            # it. Existing keys above are untouched; nothing here changes a
            # decision — these make an old decision re-derivable.
            "quantity_requested": quantity,
            "approved_quantity": (
                assessment.approved_quantity if assessment is not None else None
            ),
            # ADDITIVE (Phase C contract §7.3): the engine's own view of what
            # was asked for and WHICH constraints bound, each with its LAYER
            # (HARD_LIMIT for every Tier 0 rule). A pure re-presentation of
            # `reason_codes` — no behaviour change, no new decision input.
            "requested_quantity": (
                assessment.requested_quantity if assessment is not None else None
            ),
            "binding_constraints": (
                [dataclasses.asdict(c) for c in assessment.binding_constraints]
                if assessment is not None
                else []
            ),
            "budget_multiplier": budget_multiplier,
            # Every threshold the decision was measured against (engine.py:11
            # "every threshold is a parameter, never a hardcoded truth") —
            # SCALARS only: the mapping-valued limits (cash floors, buckets)
            # are policy tables, not per-decision facts, and would bloat every
            # audit row.
            "limits": {
                name: value
                for name, value in dataclasses.asdict(limits).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            # §70 shadow block: hypothetical verdicts that did NOT influence
            # the decision (REPORT-mode gates). Reviewed before promotion.
            # `statistical` is the Phase B SHADOW snapshot of the CURRENT
            # book (contract §6); null when the gate never ran, or an object
            # carrying only `note` when the build itself failed — either way
            # the decision above is exactly what it would have been.
            "shadow": {
                "liquidity": liquidity_shadow,
                # Phase D squeeze PROXY (REPORT mode) — None unless the
                # candidate was SHORT_STOCK; mirrors the SQUEEZE_RISK gate.
                "squeeze": (
                    {
                        "volume_z": squeeze_report.volume_z,
                        "dist_from_high_pct": squeeze_report.dist_from_high_pct,
                        "overnight_gap_pct": squeeze_report.overnight_gap_pct,
                        "elevated": squeeze_report.elevated,
                        "reasons": squeeze_report.reasons,
                        "note": "PROXY — no short-interest/borrow/float data (§33)",
                    }
                    if squeeze_report is not None
                    else None
                ),
                "statistical": statistical_shadow,
                # §14 EWMA side-by-side (Phase C contract §7.5). None when
                # the gate never ran; the multiplier IN FORCE is the
                # top-level `budget_multiplier`, which this never touched.
                "vol_targeting_ewma": vol_targeting_ewma_shadow,
                # Phase K §62-§67 event risk. None when the gate never ran;
                # an object carrying only `error` when the seam raised. Its
                # `enforcement` is the literal "SHADOW" and its cap (if any)
                # bound nothing — the `approved_quantity` above is Tier 0's.
                "event": event_shadow,
            },
        },
    )

    risk_out = None
    if assessment is not None:
        risk_out = {
            "decision": assessment.decision.value,
            "approved_quantity": assessment.approved_quantity,
            "signal_strength": assessment.signal_strength,
            "risk_budget_pct": assessment.risk_budget_pct,
            "trade_risk_usd": assessment.trade_risk_usd,
            "reason_codes": assessment.reason_codes,
            "explanations": assessment.explanations,
            "heat_before_pct": assessment.heat_before_pct,
            "heat_after_pct": assessment.heat_after_pct,
            "cash_after_pct": assessment.cash_after_pct,
            # ADDITIVE (Phase C contract §7.5): the SAME content the audit's
            # `shadow.statistical` carries, mirrored onto the preview so the
            # Trade Plan panel and a stored plan's `preview.risk` render the
            # §46 table, the binding constraints and the hypothetical
            # statistical verdict without a second round trip.
            #
            # SHADOW: `decision` and `approved_quantity` above are Tier 0's
            # and are unaffected by anything below.
            "comparison": (
                statistical_shadow.get("comparison")
                if statistical_shadow is not None
                else None
            ),
            "binding_constraints": [
                dataclasses.asdict(c) for c in assessment.binding_constraints
            ],
            "shadow_statistical": statistical_shadow,
        }

    # The proposed contract (§9), null for stock / no trade (honest null).
    contract_out = None
    spread_out = None
    if chosen_spread is not None:
        try:
            long_sym = occ_option_symbol(
                ticker,
                chosen_spread.long_leg.expiry,
                chosen_spread.long_leg.strike,
                chosen_spread.long_leg.right,
            )
            short_sym = occ_option_symbol(
                ticker,
                chosen_spread.short_leg.expiry,
                chosen_spread.short_leg.strike,
                chosen_spread.short_leg.right,
            )
        except ValueError:
            long_sym = short_sym = None
        spread_out = {
            # §25: the EXACT leg identities the broker would be sent.
            "long_symbol": long_sym,
            "short_symbol": short_sym,
            "expiry": chosen_spread.long_leg.expiry.isoformat(),
            "dte": chosen_spread.long_leg.dte,
            "long_strike": chosen_spread.long_leg.strike,
            "short_strike": chosen_spread.short_leg.strike,
            "right": chosen_spread.long_leg.right,
            "long_mid": chosen_spread.long_leg.mid,
            "short_mid": chosen_spread.short_leg.mid,
            "net_debit": chosen_spread.net_debit,
            "width": chosen_spread.width,
            "max_loss_per_spread": chosen_spread.net_debit * OPTION_MULTIPLIER,
            "max_profit_per_spread": chosen_spread.max_profit * OPTION_MULTIPLIER,
            "breakeven": chosen_spread.breakeven,
            "net_delta": chosen_spread.net_delta,
            "net_theta": chosen_spread.net_theta,
            "net_vega": chosen_spread.net_vega,
            "multiplier": OPTION_MULTIPLIER,
        }
    if chosen is not None:
        try:
            symbol_out = occ_option_symbol(
                ticker, chosen.expiry, chosen.strike, chosen.right
            )
        except ValueError:
            # Unbuildable symbol is reported as null, never guessed — the
            # approve path will 422 with the full reason if this is attempted.
            symbol_out = None
        contract_out = {
            # §25: the EXACT contract identity the broker would be sent,
            # built server-side — the UI never reconstructs OCC symbols.
            "option_symbol": symbol_out,
            "expiry": chosen.expiry.isoformat(),
            "dte": chosen.dte,
            "strike": chosen.strike,
            "right": chosen.right,
            # §25: the live quote the mid came from, so the user sees what
            # they would actually cross, not just the midpoint.
            "bid": chosen.bid,
            "ask": chosen.ask,
            "mid": chosen.mid,
            "spread_pct": chosen.spread_pct,
            "open_interest": chosen.open_interest,
            "volume": chosen.volume,
            "delta": chosen.delta,
            "iv": chosen.iv,
            "multiplier": OPTION_MULTIPLIER,
            # A long option's premium is fully at risk (§12.1).
            "max_loss_per_contract": chosen.mid * OPTION_MULTIPLIER,
        }

    # §24 — the exit plan is part of the plan BEFORE it can be applied: the
    # user must see how the position would be exited. All values come from
    # ExitParams — the SAME engine that monitors open positions — plus the
    # entry/stop numbers computed above. Honest nulls where the chain vetoed
    # before an entry existed.
    exit_params = ExitParams()
    is_option_entry = chosen is not None or chosen_spread is not None
    exit_plan = {
        "signal_invalidation": (
            "SIGNAL_FLIP: exit when the live bias flips against the position; "
            f"SIGNAL_DECAY: exit when directional edge <= "
            f"{exit_params.exit_edge_threshold}"
        ),
        "exit_edge_threshold": exit_params.exit_edge_threshold,
        "hard_stop": (
            (
                f"premium hard stop: exit at "
                f"{exit_params.premium_hard_stop_pct:.0%} premium loss "
                f"(stop level ${(chosen.mid if chosen is not None else chosen_spread.net_debit) * OPTION_MULTIPLIER * (1.0 - exit_params.premium_hard_stop_pct):,.2f} "
                f"of ${(chosen.mid if chosen is not None else chosen_spread.net_debit) * OPTION_MULTIPLIER:,.2f} entry "
                + ("net debit)" if chosen is None else "premium)")
            )
            if is_option_entry
            else (
                # Phase 3: a short's stop sits ABOVE entry (mirror image).
                (
                    f"hard stop at ${entry_price + stop_distance:,.2f} "
                    f"(entry ${entry_price:,.2f} + 2×ATR14 ${stop_distance:,.2f})"
                    if decision is not None
                    and decision.instrument is InstrumentType.SHORT_STOCK
                    else f"hard stop at ${entry_price - stop_distance:,.2f} "
                    f"(entry ${entry_price:,.2f} − 2×ATR14 ${stop_distance:,.2f})"
                )
                if entry_price is not None and stop_distance is not None
                else None
            )
        ),
        "premium_hard_stop_pct": (
            exit_params.premium_hard_stop_pct if is_option_entry else None
        ),
        "atr_trail": f"trailing stop at {exit_params.atr_trail_k}×ATR{exit_params.atr_period} below the highest close since entry",
        "atr_trail_k": exit_params.atr_trail_k,
        "time_stop": (
            f"abandon after {exit_params.time_stop_bars} bars unless the move "
            f"exceeds {exit_params.min_move_atr}×ATR in favour"
        ),
        "time_stop_bars": exit_params.time_stop_bars,
        "dte_exit_threshold": (
            exit_params.dte_exit_threshold if is_option_entry else None
        ),
        # V1 has NO fixed profit target — exits are stop/trail/decay/time
        # driven (honest null, §24 line stated rather than invented).
        "profit_target": None,
    }

    preview = {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        # §15/§16 research/execution split: which chain this evaluation ran.
        "mode": mode,
        # §20/§34: execution authorization is reported SEPARATELY from the
        # research verdict — in research mode it never gates the plan.
        "execution_authorization": execution_authorization,
        "gates": gates,
        "signal": {
            "edge": signal_edge,
            "bias": signal_bias,
            "strength": assessment.signal_strength if assessment is not None else None,
        },
        "proposed": {
            # The §8 matrix verdict — "NO_TRADE" is reported honestly when
            # the INSTRUMENT (or vol-caused VOLATILITY) gate failed; null when
            # the chain vetoed before the matrix ran (honest null).
            "instrument": decision.instrument.value if decision is not None else None,
            "vol_regime": vol_regime.value if vol_regime is not None else None,
            "instrument_rationale": (
                list(decision.rationale) if decision is not None else []
            ),
            "contract": contract_out,
            # §9-S spread verdict (roadmap Phase 1), null otherwise.
            "spread": spread_out,
            # §12.1: for an option entry BOTH numbers are the risk engine's
            # contract-level basis (mid * 100 — premium fully at risk); for
            # stock they stay the last close and the 2*ATR14 stop.
            "entry_price": (
                chosen.mid * OPTION_MULTIPLIER
                if chosen is not None
                else chosen_spread.net_debit * OPTION_MULTIPLIER
                if chosen_spread is not None
                else entry_price
            ),
            "stop_distance": (
                chosen.mid * OPTION_MULTIPLIER
                if chosen is not None
                else chosen_spread.net_debit * OPTION_MULTIPLIER
                if chosen_spread is not None
                else stop_distance
            ),
            "quantity_requested": quantity,
        },
        "risk": risk_out,
        # §24: how the position would be EXITED, visible before Apply.
        "exit_plan": exit_plan,
        "why_trade": why_trade,
        "why_not_trade": why_not_trade,
    }
    return GateChainResult(
        preview=preview,
        assessment=assessment,
        entry_price=entry_price,
        stop_distance=stop_distance,
        edge=signal_edge,
        last_bar_date=bars[-1].ts.isoformat() if bars else None,
        veto_gate=veto_gate,
        instrument=decision.instrument.value if decision is not None else None,
        vol_regime=vol_regime.value if vol_regime is not None else None,
        contract=chosen,
        spread=chosen_spread,
    )


