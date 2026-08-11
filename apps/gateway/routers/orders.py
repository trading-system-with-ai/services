"""Order preview + paper execution API — the §10 entry gate chain, fully
explainable (§33, §36), and the V1 paper fill model (plan §11), for stock
AND long-option entries (plan §8, §9, §12.1).

``POST /api/orders/preview`` walks a proposed entry through the nine §10
gates in their exact order and answers with per-gate PASS / FAIL / SKIPPED
status plus why-trade / why-not-trade narratives (§33 — both lists are
always present). Evaluation stops populating later gates after the first
FAIL ("no rejected ticker may produce an order", §42): the remaining gates
report SKIPPED with "not evaluated: earlier gate failed".

Gate semantics (options wired through, plan §7-§9):

- VOLATILITY is a REAL §7 classification off today's chain summary
  (``atm_iv`` + ``rv20`` via the shared helpers in routers/options.py —
  one chain build, never duplicated, plan §21). It PASSes with the regime
  detail and FAILs ONLY when the §8 matrix maps this exact cell to NO_TRADE
  *because of vol* (the same direction/strength under NORMAL vol would
  trade). No chain data -> PASS with an honest "no chain data" detail and
  the matrix is called with ``vol_regime=None``, which it documents as
  treated-as-NORMAL (the no-information column).
- INSTRUMENT is the §8 matrix verdict via
  ``libs.trading_core.strategies.select_instrument`` — FAIL when NO_TRADE,
  with the §8 cell + §5 degradation rationale in the detail.
- CONTRACT_SELECTION is SKIPPED for LONG_STOCK ("stock order — no contract
  selection needed"); for option instruments it runs the §9 selector over
  the same chain and PASSes with the top-ranked candidate (which becomes
  the proposed contract) or FAILs when no eligible contract exists.

Option risk sizing (§12.1, the options formula): the risk engine receives
CONTRACT-level units — ``entry_price`` and ``stop_distance`` are BOTH
``mid * 100`` because a long option's premium is FULLY at risk (max loss =
premium paid). ``approved_quantity`` is therefore the number of CONTRACTS,
and every existing risk cap (tier budget §12.2, single-name risk/capital
§12.3, buckets §12.4, heat §12.5, cash floor §13) applies unchanged.

``POST /api/orders/approve`` ALWAYS re-runs that same chain server-side —
client previews are never trusted (§42) — and only a fully passing chain may
fill. ``POST /api/orders/close`` sells an open position to close; closing is
allowed even while global trading is paused, because closing REDUCES risk and
risk protection outranks the pause (§18 risk-priority).

Paper fill model (plan §11; parameters live on Settings, §6.2): stock fills
simulate off the last STORED daily close, moved AGAINST the trader by
``paper_slippage_bps`` — BUY fills at ``close * (1 + bps/10000)``, SELL at
``close * (1 - bps/10000)`` — plus ``paper_commission_per_share * quantity``
commission charged on BOTH sides. Option fills apply the SAME slippage to
the contract MID per share: BUY debits ``qty * mid*(1+bps/10000) * 100 +
paper_commission_per_contract * qty``; SELL credits ``qty *
mid*(1-bps/10000) * 100 - paper_commission_per_contract * qty``. Closing an
option whose contract is MISSING from today's regenerated chain (e.g.
expired) falls back to INTRINSIC value against the current spot — an honest,
documented degradation, never a silent zero. The only order sides are
BUY_TO_OPEN and SELL_TO_CLOSE — for options that means sell-to-close ONLY;
Sell-to-Open does not exist in this system (§5).

For option positions the ``stop_distance`` column stores the per-share fill
premium — the §11.3 PREMIUM hard-stop basis (the exit engine stops at
``entry_premium * (1 - premium_hard_stop_pct)``), NOT an underlying price
stop; ``max_loss`` is the full premium paid (``qty * fill * 100``, §12.1).

Risk approval calls ``libs.trading_core.risk.assess`` — the risk engine is
never reimplemented here, and risk limits have PRIORITY over strategy
confidence (§44 rule 20). The call now also carries (a) the §14 vol-targeting
``budget_multiplier`` — the SAME computation the portfolio risk view reports
(shared helper in routers/portfolio.py; 1.0 when no positions are open), which
scales the tier budget but can NEVER override hard caps (§14) and is named in
the RISK_APPROVAL detail whenever it differs from 1 — and (b) the §16
portfolio greeks (current book aggregated via the shared helper, plus the
candidate's per-share greeks: stock delta 1, options the proposed contract's
greeks), so a post-trade greek-limit breach REJECTs exactly like any other
risk limit, with its reason codes and explanations surfaced in the preview.
Every chain run writes exactly ONE SYSTEM-attributed
RISK_DECISION audit event — even when vetoed at gate one — in the same
transaction, so every decision is auditable (§38, rule 12).
"""
import asyncio
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.trading_core.contracts import ContractQuote, SelectorParams, select_contracts
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
    PortfolioSnapshot,
    PositionRisk,
    RiskAssessment,
    RiskLimits,
    RiskRequest,
    assess,
)
from libs.trading_core.risk.engine import strength_tier
from libs.trading_core.signals import RegimeParams, classify_regime, score_direction
from libs.trading_core.strategies import AccountPermissions, select_instrument
from libs.trading_core.volatility import VolRegimeParams, classify_vol_regime

from .. import audit
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
from ..schemas import TickerRequest
from .analysis import ensure_daily_bars, market_regime_from_spy
from .options import build_option_chain, chain_iv_summary

# The portfolio-picture helpers live in routers/portfolio.py so the risk view
# and this order path build the identical book (plan §21). find_option_contract
# and is_option_position are also RE-EXPORTED through this module on purpose:
# routers/positions.py imports them from here.
from .portfolio import (
    find_option_contract,
    is_option_position,
    open_positions_with_prices,
    portfolio_greeks_read,
    position_market_value,
    stored_closes_by_ticker,
    vol_targeting_block,
)

router = APIRouter(prefix="/api/orders", tags=["orders"])

# The only order sides that exist in this system (§5): a long-only account
# BUYS to open and SELLS to close. Sell-to-Open does not exist, ever.
BUY_TO_OPEN = "BUY_TO_OPEN"
SELL_TO_CLOSE = "SELL_TO_CLOSE"

# Position lifecycle states (positions.status column).
POSITION_OPEN = "OPEN"
POSITION_CLOSED = "CLOSED"

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
# Stop distance for a LONG_STOCK entry = ATR_STOP_MULTIPLE * ATR14 (plan §12.1).
ATR_STOP_MULTIPLE = 2.0
ATR_PERIOD = 14
# DATA_QUALITY staleness bound: last stored bar must be within this many
# calendar days of today (stub data always passes; the check must still exist).
MAX_BAR_AGE_DAYS = 5
# Standard US equity option contract multiplier (plan §12.1: cash and max
# loss are per-share premium x this).
OPTION_MULTIPLIER = 100
# Parameter seams for the option gates (plan §6.2): module-level so tests and
# future config wiring can substitute custom thresholds without touching the
# chain logic — §7 vol-regime thresholds, §9.1/§9.2 selector thresholds, and
# the §5 account permission flags the §8 matrix degrades under.
VOL_REGIME_PARAMS = VolRegimeParams()
SELECTOR_PARAMS = SelectorParams()
ACCOUNT_PERMISSIONS = AccountPermissions()

# The §10 gate chain, in its exact order.
GATE_ORDER = (
    "TRADING_POOL_AUTHORIZATION",
    "DATA_QUALITY",
    "REGIME",
    "DIRECTIONAL_SIGNAL",
    "VOLATILITY",
    "INSTRUMENT",
    "LIQUIDITY",
    "CONTRACT_SELECTION",
    "RISK_APPROVAL",
)

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"

# Exact V1 skip details (contract-fixed strings).
SKIP_EARLIER_FAIL = "not evaluated: earlier gate failed"
SKIP_NO_OPTION_DATA = "no option/quote data yet — arrives with the Massive integration"
SKIP_STOCK_ORDER = "stock order — no contract selection needed"

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

    @property
    def failed(self) -> bool:
        """True when any gate FAILed — no order may be produced (§42)."""
        return self.veto_gate is not None


async def run_gate_chain(
    session: AsyncSession,
    ticker: str,
    quantity: int | None,
    direction: str = "AUTO",
) -> GateChainResult:
    """Evaluate the §10 gate chain for a proposed entry (§33, §42).

    ``direction`` is the §9-style resolution seam: "AUTO" defers to the
    signal bias, an explicit "BULL"/"BEAR" wins (and is reported honestly in
    the DIRECTIONAL_SIGNAL detail). Places no order and never commits.
    Records exactly one SYSTEM RISK_DECISION audit event — veto or approval —
    on the session; the caller commits it in the same transaction as any
    state change (§38, rule 12).
    """
    settings = get_settings()
    limits = RiskLimits()
    regime_params = RegimeParams()

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
    chain: list[ContractQuote] = []
    vol_regime = None  # IVRegime | None (§7; None = no chain data, honest null)
    decision = None  # InstrumentDecision | None (§8 matrix verdict)
    chosen: ContractQuote | None = None  # §9 top-ranked candidate

    # ------------------------------------------------------------------
    # Gate 1 — TRADING_POOL_AUTHORIZATION (§32, §18): the ticker must be in
    # the Trading Pool, per-symbol trading enabled, AND the global kill
    # switch must allow trading. FAIL names every missing authorization.
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
        _, chain = build_option_chain(ticker, entry_price)
        summary = chain_iv_summary(chain, entry_price, closes)
        atm_iv = summary["atm_iv"]
        rv20 = summary["rv20"]
        if atm_iv is not None:
            vol_regime = classify_vol_regime(
                atm_iv,
                rv20 if rv20 is not None and rv20 > 0 else None,
                VOL_REGIME_PARAMS,
            ).regime
        # §8 matrix verdict — computed HERE because this gate's PASS/FAIL is
        # defined by whether vol alone turned the cell into NO_TRADE.
        decision = select_instrument(
            resolved, strength, vol_regime, ACCOUNT_PERMISSIONS
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
                resolved, strength, None, ACCOUNT_PERMISSIONS
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

    # LIQUIDITY stays a V1 skip: underlying quote/depth data arrives with the
    # Massive integration (§22.1); option liquidity is already enforced
    # per-contract by the §9.1 filters inside CONTRACT_SELECTION.
    gate("LIQUIDITY", SKIPPED, SKIP_EARLIER_FAIL if vetoed else SKIP_NO_OPTION_DATA)

    # ------------------------------------------------------------------
    # Gate 8 — CONTRACT_SELECTION (§9): SKIPPED for stock; for an option
    # instrument the §9 selector runs over the SAME chain and the top-ranked
    # candidate becomes the proposed contract; FAIL when nothing is eligible.
    # ------------------------------------------------------------------
    if vetoed:
        gate("CONTRACT_SELECTION", SKIPPED, SKIP_EARLIER_FAIL)
    elif decision.instrument is InstrumentType.LONG_STOCK:
        gate("CONTRACT_SELECTION", SKIPPED, SKIP_STOCK_ORDER)
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
    # ------------------------------------------------------------------
    if vetoed:
        gate("RISK_APPROVAL", SKIPPED, SKIP_EARLIER_FAIL)
    else:
        if chosen is not None:
            risk_entry = risk_stop = chosen.mid * OPTION_MULTIPLIER
        else:
            risk_entry, risk_stop = entry_price, stop_distance
        portfolio = await get_or_create_portfolio(session)
        pairs = await open_positions_with_prices(session)
        values = [position_market_value(pos, price) for pos, price in pairs]
        nav = portfolio.cash + sum(v for v in values if v is not None)
        position_risks = [
            PositionRisk(
                ticker=pos.ticker,
                market_value=value if value is not None else 0.0,
                max_loss=pos.max_loss,
            )
            for (pos, _price), value in zip(pairs, values)
        ]
        spy_regime = (await market_regime_from_spy(session)).classification
        snapshot = PortfolioSnapshot(
            nav=nav,
            cash=portfolio.cash,
            positions=position_risks,
            regime=spy_regime,
            trading_enabled=state.trading_enabled,
        )
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
        else:
            candidate_greeks = PositionGreeksInput(
                ticker=ticker,
                instrument=InstrumentType.LONG_STOCK.value,
                quantity=1,  # requested basis; assess scales by approved qty
                multiplier=1,
                spot=entry_price,
                delta=1.0,
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
        unit = "contracts" if chosen is not None else "shares"
        if assessment.decision is RiskDecision.REJECT:
            gate(
                "RISK_APPROVAL",
                FAIL,
                f"risk engine REJECT ({', '.join(assessment.reason_codes)}) — "
                "risk limits have priority over strategy confidence "
                f"(§44 rule 20){vt_detail}",
            )
            vetoed = True
        elif assessment.decision is RiskDecision.APPROVE_WITH_RESIZE:
            gate(
                "RISK_APPROVAL",
                PASS,
                f"APPROVE_WITH_RESIZE: quantity resized to "
                f"{assessment.approved_quantity} {unit} "
                f"({', '.join(assessment.reason_codes)}){vt_detail}",
            )
        else:
            gate(
                "RISK_APPROVAL",
                PASS,
                f"APPROVE: {assessment.approved_quantity} {unit}, "
                f"${assessment.trade_risk_usd:,.2f} at risk{vt_detail}",
            )

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
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.RISK_DECISION,
        entity_type="order_preview",
        entity_id=ticker,
        details={
            "decision": assessment.decision.value if assessment is not None else "VETOED",
            "veto_gate": veto_gate,
            "gates": {g["name"]: g["status"] for g in gates},
            "reason_codes": (
                list(assessment.reason_codes)
                if assessment is not None
                else ([f"VETO_{veto_gate}"] if veto_gate else [])
            ),
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
        }

    # The proposed contract (§9), null for stock / no trade (honest null).
    contract_out = None
    if chosen is not None:
        contract_out = {
            "expiry": chosen.expiry.isoformat(),
            "dte": chosen.dte,
            "strike": chosen.strike,
            "right": chosen.right,
            "mid": chosen.mid,
            "delta": chosen.delta,
            "iv": chosen.iv,
            "multiplier": OPTION_MULTIPLIER,
            # A long option's premium is fully at risk (§12.1).
            "max_loss_per_contract": chosen.mid * OPTION_MULTIPLIER,
        }

    preview = {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
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
            # §12.1: for an option entry BOTH numbers are the risk engine's
            # contract-level basis (mid * 100 — premium fully at risk); for
            # stock they stay the last close and the 2*ATR14 stop.
            "entry_price": (
                chosen.mid * OPTION_MULTIPLIER if chosen is not None else entry_price
            ),
            "stop_distance": (
                chosen.mid * OPTION_MULTIPLIER
                if chosen is not None
                else stop_distance
            ),
            "quantity_requested": quantity,
        },
        "risk": risk_out,
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
    )


@router.post("/preview")
async def preview_order(
    req: OrderPreviewRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Evaluate the §10 gate chain for a proposed entry (§33, §42).

    Places no order. Writes exactly one SYSTEM RISK_DECISION audit event —
    veto or approval — committed in the same transaction (§38, rule 12).
    """
    result = await run_gate_chain(session, req.ticker, req.quantity, req.direction)
    await session.commit()
    return result.preview


# ---------------------------------------------------------------------------
# Paper execution (plan §11, §42)
# ---------------------------------------------------------------------------


class OrderApproveRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)
    # Same §9-style direction seam as preview — approve re-runs the chain
    # with it, so what was previewed is what is re-evaluated (§42).
    direction: Literal["AUTO", "BULL", "BEAR"] = "AUTO"
    # Idempotency key (§42): replaying the same key returns the existing
    # order — a duplicate request can never fill twice.
    client_order_id: str | None = Field(default=None, min_length=1, max_length=64)


class OrderCloseRequest(TickerRequest):
    quantity: int | None = Field(default=None, ge=1)  # default: close in full
    reason: str | None = None


def _contract_payload(row: Order | Position) -> dict | None:
    """The opt_* identity block of an order/position, null for stock."""
    if row.opt_expiry is None:
        return None
    return {
        "expiry": row.opt_expiry,
        "strike": row.opt_strike,
        "right": row.opt_right,
        "multiplier": OPTION_MULTIPLIER,
    }


def _order_payload(order: Order) -> dict:
    return {
        "id": order.id,
        "client_order_id": order.client_order_id,
        "ticker": order.ticker,
        "instrument": order.instrument,
        "side": order.side,
        "quantity": order.quantity,
        "fill_price": order.fill_price,
        "commission": order.commission,
        "status": order.status,
        "contract": _contract_payload(order),
        "created_at": order.created_at.isoformat(),
    }


def _position_payload(position: Position) -> dict:
    return {
        "id": position.id,
        "ticker": position.ticker,
        "instrument": position.instrument,
        "status": position.status,
        "quantity": position.quantity,
        "avg_price": position.avg_price,
        # Stock: the §11.3 underlying stop. Options: honest null here — the
        # §11.3 stop is PREMIUM-based (entry premium * (1 - stop pct)) and is
        # reported by the position monitor's exit-engine read.
        "stop_price": (
            position.avg_price - position.stop_distance
            if position.stop_distance > 0 and not is_option_position(position)
            else None
        ),
        "max_loss": position.max_loss,
        "realized_pnl": position.realized_pnl,
        "contract": _contract_payload(position),
        "closed_at": position.closed_at.isoformat() if position.closed_at else None,
    }


async def _open_position(session: AsyncSession, ticker: str) -> Position | None:
    """The OPEN position in `ticker`, if any (at most one — no pyramiding, V1)."""
    return (
        (
            await session.execute(
                select(Position)
                .where(Position.ticker == ticker, Position.status == POSITION_OPEN)
                .order_by(Position.id)
            )
        )
        .scalars()
        .first()
    )


async def _last_stored_close(session: AsyncSession, ticker: str) -> float | None:
    """Last stored daily close for `ticker` — the paper fill reference price."""
    return (
        (
            await session.execute(
                select(StockBarDaily.close)
                .where(StockBarDaily.ticker == ticker)
                .order_by(StockBarDaily.ts.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


def option_intrinsic_value(position: Position, spot: float) -> float:
    """Intrinsic value per share of `position`'s contract at `spot` — the
    documented close-price fallback when the contract is missing from
    today's chain (e.g. expired): max(0, S-K) for calls, max(0, K-S) for
    puts. Honest degradation, never a silent zero."""
    if position.opt_right == "C":
        return max(0.0, spot - (position.opt_strike or 0.0))
    return max(0.0, (position.opt_strike or 0.0) - spot)


async def option_close_reference(
    session: AsyncSession, position: Position
) -> tuple[float, str]:
    """Per-share reference price to sell-to-close an option position (§11).

    Regenerates today's chain via the SHARED helper (routers/options.py) and
    reads the SAME contract's mid; when the contract is missing from today's
    chain (e.g. expired) it falls back to INTRINSIC value against the current
    spot (last stored close) — documented above. Returns ``(price, source)``
    with source "chain mid" | "intrinsic (contract missing from today's
    chain)". Raises 422 when no stored bars exist (no spot — honest error,
    §44 rule 18).
    """
    last_close = await _last_stored_close(session, position.ticker)
    if last_close is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"no stored bars for {position.ticker} — the paper fill model "
                "has no reference price (honest error, §44 rule 18)"
            ),
        )
    _, chain = build_option_chain(position.ticker, last_close)
    contract = find_option_contract(chain, position)
    if contract is not None:
        return contract.mid, "chain mid"
    return (
        option_intrinsic_value(position, last_close),
        "intrinsic (contract missing from today's chain)",
    )


async def execute_sell_to_close(
    session: AsyncSession,
    position: Position,
    quantity: int,
    reference_price: float,
    *,
    reason: str | None = None,
    system_generated: bool = False,
    reference_source: str = "last stored close",
) -> tuple[Order, float]:
    """Fill a SELL_TO_CLOSE against `position` at the paper fill model (§11).

    ``reference_price`` is per share — the last stored close for stock, the
    current contract mid (or documented intrinsic fallback) for options; the
    SELL fill = ``reference * (1 - paper_slippage_bps/10000)`` (slippage
    always moves AGAINST the trader). Commission =
    ``paper_commission_per_share * quantity`` for stock,
    ``paper_commission_per_contract * quantity`` for options — charged on
    this side too. Cash proceeds and realized PnL scale by the position
    multiplier (1 stock / 100 options):
    ``realized_pnl = (fill - avg_price) * quantity * multiplier -
    commission`` — the buy-side commission is NOT re-charged here because it
    already left cash at open. Cash is credited the net proceeds; the
    position shrinks (its ``max_loss`` scales down proportionally so
    portfolio heat, plan §12.5, reflects only the risk still open) and flips
    to CLOSED with ``closed_at`` when quantity reaches 0. Records the full
    ORDER_REQUESTED (SYSTEM when ``system_generated``, else USER, plan §11)
    -> ORDER_SUBMITTED -> ORDER_FILLED audit chain on the session and NEVER
    commits — the caller owns the transaction (rule 12).

    Deliberately checks NEITHER the kill switch nor the §10 gates: closing
    REDUCES risk, and risk protection outranks the pause (§18 risk-priority).
    The side is SELL_TO_CLOSE — for options that is the ONLY closing action;
    Sell-to-Open does not exist in this system (§5).
    """
    settings = get_settings()
    is_option = is_option_position(position)
    multiplier = position.multiplier or 1
    fill = reference_price * (1.0 - settings.paper_slippage_bps / 10000.0)
    commission = (
        settings.paper_commission_per_contract
        if is_option
        else settings.paper_commission_per_share
    ) * quantity
    proceeds = quantity * fill * multiplier - commission
    realized = (fill - position.avg_price) * quantity * multiplier - commission

    portfolio = await get_or_create_portfolio(session)
    portfolio.cash += proceeds
    portfolio.updated_at = utcnow()

    order = Order(
        ticker=position.ticker,
        instrument=position.instrument,
        side=SELL_TO_CLOSE,
        quantity=quantity,
        fill_price=fill,
        commission=commission,
        status="FILLED",
        opt_expiry=position.opt_expiry,
        opt_strike=position.opt_strike,
        opt_right=position.opt_right,
    )
    session.add(order)
    await session.flush()

    remaining = position.quantity - quantity
    position.max_loss = (
        position.max_loss * remaining / position.quantity
        if position.quantity > 0
        else 0.0
    )
    position.quantity = remaining
    position.realized_pnl = (position.realized_pnl or 0.0) + realized
    if remaining <= 0:
        position.status = POSITION_CLOSED
        position.closed_at = utcnow()
        position.max_loss = 0.0

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM if system_generated else ActorType.USER,
        actor_id="" if system_generated else CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "ticker": position.ticker,
            "instrument": position.instrument,
            "side": SELL_TO_CLOSE,
            "quantity": quantity,
            "reason": reason,
            "system_generated": system_generated,
        },
    )
    submitted_details = {
        "fill_model": (
            "paper: reference * (1 - slippage_bps/10000) per share; "
            "cash = qty * fill * multiplier - commission"
        ),
        "reference_price": reference_price,
        "reference_source": reference_source,
        "multiplier": multiplier,
        "slippage_bps": settings.paper_slippage_bps,
    }
    if is_option:
        submitted_details["commission_per_contract"] = (
            settings.paper_commission_per_contract
        )
    else:
        # Kept for continuity with the pre-option audit shape.
        submitted_details["last_close"] = reference_price
        submitted_details["commission_per_share"] = (
            settings.paper_commission_per_share
        )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details=submitted_details,
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_price": fill,
            "commission": commission,
            "position_id": position.id,
            "realized_pnl": realized,
        },
    )
    return order, realized


@router.post("/approve")
async def approve_order(
    req: OrderApproveRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Approve and paper-fill a BUY_TO_OPEN entry — ONE transaction (§11, §42).

    The server ALWAYS re-runs the full §10 gate chain at approval time —
    client previews are never trusted (§42); any FAILing gate answers 422
    with the fresh preview embedded, and no order row may exist for a
    rejected ticker. Flow: idempotency lookup (a duplicate
    ``client_order_id`` returns the EXISTING order, 200, no second fill,
    §42) -> open-position check (an existing OPEN position in the ticker is
    409 — no pyramiding in V1) -> gate chain -> fill.

    Fill model (§11): stock BUY fill = last stored close *
    ``(1 + paper_slippage_bps/10000)`` plus
    ``paper_commission_per_share * quantity`` commission. Option BUY fill =
    contract mid * ``(1 + paper_slippage_bps/10000)`` PER SHARE; cash debit
    = ``qty * fill * 100 + paper_commission_per_contract * qty`` (§12.1:
    the premium is fully at risk, so the position's ``max_loss`` is
    ``qty * fill * 100`` and its ``stop_distance`` column stores the
    per-share fill premium — the §11.3 PREMIUM stop basis, not an underlying
    stop). Quantity = min(requested, risk-approved) — risk limits outrank
    strategy confidence (§44 rule 20); for options both counts are
    CONTRACTS. Cash is guarded again at fill time (INSUFFICIENT_CASH ->
    422) even though the risk chain already enforced the cash floor (§13).
    The Order + Position + cash decrement + audit chain ORDER_REQUESTED
    (USER) -> ORDER_SUBMITTED -> ORDER_FILLED (+ the chain's RISK_DECISION)
    all commit atomically (rule 12). The whole flow runs under the
    paper-execution lock so two rapid approves can never double-fill (§42;
    V1 no-pyramiding).
    """
    async with execution_lock():
        return await _approve_order_locked(req, session)


async def _approve_order_locked(
    req: OrderApproveRequest, session: AsyncSession
) -> dict:
    """The approve flow proper — caller holds the paper-execution lock."""
    ticker = req.ticker

    # --- Idempotency (§42): same client_order_id -> the EXISTING order. ----
    if req.client_order_id is not None:
        existing = (
            (
                await session.execute(
                    select(Order).where(Order.client_order_id == req.client_order_id)
                )
            )
            .scalars()
            .first()
        )
        if existing is not None:
            # Replay: no re-evaluation, no second fill, no new audit events.
            # The position reflects its CURRENT state; preview is an honest
            # null (§44 rule 18) — nothing was re-evaluated on this call.
            position = (
                (
                    await session.execute(
                        select(Position)
                        .where(Position.ticker == existing.ticker)
                        .order_by(Position.id.desc())
                    )
                )
                .scalars()
                .first()
            )
            return {
                "order": _order_payload(existing),
                "position": _position_payload(position) if position else None,
                "preview": None,
            }

    # --- No pyramiding in V1: one OPEN position per ticker. ---------------
    if await _open_position(session, ticker) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{ticker} already has an OPEN position — pyramiding is not "
                "supported in V1; close it before opening again"
            ),
        )

    # --- Re-run the FULL §10 chain (§42: client previews are never trusted).
    chain = await run_gate_chain(session, ticker, req.quantity, req.direction)
    if chain.failed or chain.assessment is None:
        # The chain already recorded its RISK_DECISION; commit it so the veto
        # stays auditable (rule 12), then answer 422 with the fresh preview.
        await session.commit()
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"approval denied: gate {chain.veto_gate} failed — no "
                    "rejected ticker may produce an order (§42)"
                ),
                "preview": chain.preview,
            },
        )

    approved = chain.assessment.approved_quantity
    quantity = min(req.quantity, approved) if req.quantity is not None else approved

    # --- Paper fill (§11): slippage against the trader + commission. -------
    # Options (§12.1): fill = contract mid * (1 + slippage) PER SHARE; the
    # cash debit carries the x100 multiplier plus the per-contract
    # commission, and quantity counts CONTRACTS.
    settings = get_settings()
    is_option = chain.contract is not None
    if is_option:
        contract = chain.contract
        fill = contract.mid * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_contract * quantity
        cost = quantity * fill * OPTION_MULTIPLIER + commission
    else:
        fill = chain.entry_price * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_share * quantity
        cost = quantity * fill + commission

    portfolio = await get_or_create_portfolio(session)
    if cost > portfolio.cash:
        # The §13 cash floor was already enforced by RISK_APPROVAL, but the
        # fill adds slippage + commission — guard anyway (risk outranks all).
        await session.commit()  # keep the RISK_DECISION auditable (rule 12)
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"INSUFFICIENT_CASH: cost ${cost:,.2f} exceeds cash "
                    f"${portfolio.cash:,.2f}"
                ),
                "preview": chain.preview,
            },
        )

    order = Order(
        client_order_id=req.client_order_id,
        ticker=ticker,
        instrument=chain.instrument or InstrumentType.LONG_STOCK.value,
        side=BUY_TO_OPEN,
        quantity=quantity,
        fill_price=fill,
        commission=commission,
        status="FILLED",
        opt_expiry=contract.expiry.isoformat() if is_option else None,
        opt_strike=contract.strike if is_option else None,
        opt_right=contract.right if is_option else None,
    )
    session.add(order)
    portfolio.cash -= cost
    portfolio.updated_at = utcnow()
    if is_option:
        # §12.1: the premium is fully at risk — max_loss is the whole debit
        # (ex commission) and stop_distance stores the per-share fill premium,
        # the §11.3 PREMIUM stop basis (NOT an underlying price stop).
        position = Position(
            ticker=ticker,
            instrument=chain.instrument,
            quantity=quantity,
            avg_price=fill,
            max_loss=quantity * fill * OPTION_MULTIPLIER,
            stop_distance=fill,
            entry_edge=chain.edge,
            entry_bar_date=chain.last_bar_date,
            status=POSITION_OPEN,
            opt_expiry=contract.expiry.isoformat(),
            opt_strike=contract.strike,
            opt_right=contract.right,
            multiplier=OPTION_MULTIPLIER,
        )
    else:
        position = Position(
            ticker=ticker,
            quantity=quantity,
            avg_price=fill,
            max_loss=quantity * chain.stop_distance,
            stop_distance=chain.stop_distance,
            entry_edge=chain.edge,
            entry_bar_date=chain.last_bar_date,
            status=POSITION_OPEN,
        )
    session.add(position)
    await session.flush()

    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "ticker": ticker,
            "instrument": order.instrument,
            "side": BUY_TO_OPEN,
            "quantity_requested": req.quantity,
            "client_order_id": req.client_order_id,
        },
    )
    if is_option:
        submitted_details = {
            "fill_model": (
                "paper: contract mid * (1 + slippage_bps/10000) per share; "
                "cash = qty * fill * 100 + commission"
            ),
            "contract_mid": contract.mid,
            "multiplier": OPTION_MULTIPLIER,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_contract": settings.paper_commission_per_contract,
        }
    else:
        submitted_details = {
            "fill_model": "paper: last stored close * (1 + slippage_bps/10000)",
            "last_close": chain.entry_price,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_share": settings.paper_commission_per_share,
        }
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details=submitted_details,
    )
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_price": fill,
            "commission": commission,
            "quantity": quantity,
            "position_id": position.id,
        },
    )
    await session.commit()

    return {
        "order": _order_payload(order),
        "position": {
            "id": position.id,
            "ticker": position.ticker,
            "instrument": position.instrument,
            "quantity": position.quantity,
            "avg_price": position.avg_price,
            # Stock: underlying stop. Options: honest null — the §11.3 stop
            # is premium-based and reported by the position monitor.
            "stop_price": (
                position.avg_price - position.stop_distance
                if not is_option
                else None
            ),
            "max_loss": position.max_loss,
            "contract": _contract_payload(position),
        },
        "preview": chain.preview,
    }


@router.post("/close")
async def close_position(
    req: OrderCloseRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Sell-to-close an OPEN paper position, fully or partially (§11).

    Closing is ALLOWED while global trading is paused: the §18 kill switch
    blocks NEW risk, and closing REDUCES risk — risk protection has priority
    (§18 risk-priority), so no gate chain and no kill-switch check runs here.
    404 when no OPEN position exists in the ticker; 422 when the requested
    quantity exceeds the open quantity.

    Reference price: stock closes off the last STORED daily close; an option
    position regenerates today's chain via the SHARED helper and closes at
    the SAME contract's current mid — or at INTRINSIC value against the
    current spot when the contract is missing from today's chain (e.g.
    expired; documented fallback, never a silent zero). Option proceeds =
    ``qty * mid*(1-slippage) * 100 - paper_commission_per_contract * qty``.
    Fill, cash credit, position update and the ORDER_REQUESTED (USER) ->
    ORDER_SUBMITTED -> ORDER_FILLED audit chain commit in ONE transaction
    (rule 12), under the paper-execution lock so two rapid closes can never
    double-credit cash (§42 analogue).
    """
    async with execution_lock():
        return await _close_position_locked(req, session)


async def _close_position_locked(
    req: OrderCloseRequest, session: AsyncSession
) -> dict:
    """The close flow proper — caller holds the paper-execution lock."""
    ticker = req.ticker
    position = await _open_position(session, ticker)
    if position is None:
        raise HTTPException(status_code=404, detail=f"no OPEN position in {ticker}")

    quantity = req.quantity if req.quantity is not None else position.quantity
    if quantity > position.quantity:
        raise HTTPException(
            status_code=422,
            detail=(
                f"cannot close {quantity} of {ticker}: only "
                f"{position.quantity} open"
            ),
        )

    if is_option_position(position):
        reference, source = await option_close_reference(session, position)
    else:
        reference = await _last_stored_close(session, ticker)
        source = "last stored close"
        if reference is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"no stored bars for {ticker} — the paper fill model has no "
                    "reference price (honest error, §44 rule 18)"
                ),
            )

    order, realized = await execute_sell_to_close(
        session,
        position,
        quantity,
        reference,
        reason=req.reason,
        system_generated=False,
        reference_source=source,
    )
    await session.commit()

    return {
        "order": _order_payload(order),
        "position": _position_payload(position),
        "realized_pnl": realized,
    }
