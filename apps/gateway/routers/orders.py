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

EXECUTION VENUE (plan §11, §44 rule 18) — ``Settings.broker_provider``:

- UNSET (the default): approve and close answer 503
  ``BROKER_NOT_CONFIGURED`` and place NOTHING. There is deliberately NO
  fallback to the internal simulator. A simulated fill reported as a broker
  fill would be an execution that never happened, presented as one that did —
  the exact class of invented data this platform refuses.
- ``"simulated"``: DEVELOPMENT / BACKTEST-COMPARISON ONLY, an explicit opt-in
  exactly like the stub market-data and LLM providers. The internal paper fill
  model documented above runs unchanged; nothing about it is byte-different
  from before the broker existed.
- ``"alpaca_paper"``: real orders against a real Alpaca PAPER account. Fills,
  quantities and prices come from the BROKER, never from the fill model, and
  partial fills are first-class — a position opens with the FILLED quantity
  and a zero-fill ACCEPTED order opens no position at all. Submission
  mechanics (network-level idempotency, bounded polling) live in
  apps/gateway/broker_exec.py and are shared with the exit sweep.

OPTIONS ARE OUT OF SCOPE FOR THE BROKER PATH in this change: Alpaca options
trading is not wired. With a real broker configured, an approval whose §8
instrument resolves to LONG_CALL/LONG_PUT answers 422 naming the limitation —
options remain fully available in ``"simulated"`` mode. Fabricating an option
order submission would be the same dishonesty in a different costume.

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
import dataclasses
import weakref
from dataclasses import dataclass
from datetime import date, datetime, timezone
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
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
from .analysis import ensure_daily_bars
from .options import CHAIN_CACHE_TTL_SECONDS, build_option_chain, chain_iv_summary

# The portfolio-picture helpers live in routers/portfolio.py so the risk view
# and this order path build the identical book (plan §21). find_option_contract
# and is_option_position are also RE-EXPORTED through this module on purpose:
# routers/positions.py imports them from here.
from .portfolio import (
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

router = APIRouter(prefix="/api/orders", tags=["orders"])

# The only order sides that exist in this system (§5): a long-only account
# BUYS to open and SELLS to close. Sell-to-Open does not exist, ever.

# ---------------------------------------------------------------------------
# THE GATE CHAIN LIVES IN ``apps/gateway/execution/gate_chain.py``.
#
# It used to live here — 2,500 lines under four endpoints — and the shape
# leaked: background loops imported from a router to reach the trading
# decision, thirty-two test modules reached into an HTTP module for business
# rules, and two sibling routers had to be imported inside functions to break
# the resulting cycles. A router validates input, calls one function, and
# turns the result into a status code.
#
# The shared execution constants moved WITH it: defining them in both places
# would give the two modules separate copies of ``_EXECUTION_LOCKS``, and an
# execution mutex that is not the same object in every caller is not a mutex.
#
# TESTS: monkeypatch ``execution.gate_chain``, never this module. A patch
# applied here rebinds only the re-exported reference while the code reading
# it keeps the original value — a silent failure that looks like a gate veto.
# ---------------------------------------------------------------------------
from ..execution import gate_chain  # noqa: F401  (the patch target)
from ..execution.gate_chain import (  # noqa: F401  (re-export)
    ATR_PERIOD,
    BUY_TO_OPEN,
    CANDIDATE_KEY_SUFFIX,
    CANDIDATE_LEG_KEY,
    CANDIDATE_LONG_LEG_KEY,
    CANDIDATE_SHORT_LEG_KEY,
    CURRENT_USER,
    FAIL,
    GATE_ORDER,
    GateChainResult,
    LIQUIDITY_LIMITS,
    MAX_BAR_AGE_DAYS,
    OPTION_MULTIPLIER,
    OrderPreviewRequest,
    PASS,
    PENDING_SUBMIT,
    POSITION_CLOSED,
    POSITION_OPEN,
    RISK_REJECT_COUNT,
    RISK_RESIZE_COUNT,
    SELECTOR_PARAMS,
    SELL_TO_CLOSE,
    SHORT_STOCK_GAP_RISK_FACTOR,
    SIZING_V2_PARAMS,
    SKIPPED,
    SKIP_EARLIER_FAIL,
    SKIP_NO_OPTION_DATA,
    SKIP_STOCK_ORDER,
    SPREAD_PARAMS,
    SQUEEZE_PROXY_PARAMS,
    STATISTICAL_LIMITS,
    STRESS_LIMITS,
    STRESS_LIMIT_BLOCKS,
    VOL_REGIME_PARAMS,
    _BEAR_REGIMES,
    _EXECUTION_LOCKS,
    _broker_cash_for_sizing,
    _candidate_spec,
    _candidate_stress_legs,
    _cap_api,
    _comparison_api,
    _metric_pair_row,
    _pct_of_nav,
    _pretrade_statistical_shadow,
    _pretrade_stress_shadow,
    _scalar_pair_row,
    _sizing_v2_shadow_block,
    _statistical_shadow_detail,
    _stress_comparison_row,
    _worst_row_api,
    execution_lock,
    run_gate_chain,
)

@router.get("/open")
async def list_open_orders(session: AsyncSession = Depends(get_session)) -> dict:
    """Non-terminal orders (§11/§26): what is still in flight at the broker.

    ``{"orders": [...]}`` — every order in PENDING_SUBMIT / ACCEPTED /
    PARTIALLY_FILLED, oldest first. The positions UI derives its
    PENDING_UPDATE reconciliation state from this list (a position whose
    order is still working is neither MATCHED nor MISMATCHED — it is
    honestly in flux until the order-sync sweep settles it). Empty list when
    nothing is in flight; this endpoint never needs a broker and never 503s —
    it reports LOCAL rows only.
    """
    rows_result = await session.execute(
        select(Order)
        .where(Order.status.in_([PENDING_SUBMIT, "ACCEPTED", "PARTIALLY_FILLED"]))
        .order_by(Order.id)
    )
    return {"orders": [_order_payload(o) for o in rows_result.scalars().all()]}


@router.post("/preview")
async def preview_order(
    req: OrderPreviewRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Evaluate the §10 gate chain for a proposed entry (§33, §42).

    Places no order. Writes exactly one SYSTEM RISK_DECISION audit event —
    veto or approval — committed in the same transaction (§38, rule 12).

    503 ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is
    configured. Every gate — data quality, regime, volatility, contract
    selection, sizing — is a judgement about current prices; running them on
    invented numbers would produce a confident recommendation about nothing.

    RESEARCH mode (upgrade §15/§16): a Watchlist symbol gets its complete
    research plan without Trading Pool membership — execution authorization
    is reported separately in ``execution_authorization`` and enforced only
    by the approve path, which re-runs this chain in execution mode (§21).
    """
    require_market_data_provider()
    result = await run_gate_chain(
        session, req.ticker, req.quantity, req.direction, mode="research"
    )
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
    """The opt_* identity block of an order/position, null for stock.

    ``option_symbol`` is the server-built OCC symbol (§27) — the same string
    the broker is addressed with, so the UI never reconstructs it. Null when
    the stored fields cannot build one (malformed row): reported honestly,
    and the close path will 422 with the full reason if a close is attempted.
    """
    if row.opt_expiry is None:
        return None
    try:
        symbol = occ_option_symbol(
            row.ticker,
            date.fromisoformat(row.opt_expiry),
            float(row.opt_strike),
            row.opt_right or "",
        )
    except (TypeError, ValueError):
        symbol = None
    return {
        "option_symbol": symbol,
        "expiry": row.opt_expiry,
        "strike": row.opt_strike,
        "right": row.opt_right,
        "multiplier": OPTION_MULTIPLIER,
    }


def _order_payload(order: Order) -> dict:
    """One order row as the API reports it.

    ``quantity`` is what was REQUESTED and ``filled_quantity`` what actually
    filled — separate facts, never conflated (§11 partial fills). The
    ``broker`` block is an honest null for internally simulated fills and
    carries the broker's own id and RAW status word otherwise.
    """
    return {
        "id": order.id,
        "client_order_id": order.client_order_id,
        "ticker": order.ticker,
        "instrument": order.instrument,
        "side": order.side,
        "quantity": order.quantity,
        "filled_quantity": order.filled_quantity,
        "fill_price": order.fill_price,
        "commission": order.commission,
        "status": order.status,
        # The local position this order opened/closes (§27) — null until a
        # BUY's first fill lands, and for pre-lifecycle rows.
        "position_id": order.position_id,
        "contract": _contract_payload(order),
        "broker": (
            {
                "broker_order_id": order.broker_order_id,
                "broker_status": order.broker_status,
            }
            if order.broker_order_id is not None or order.broker_status is not None
            else None
        ),
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
        # Stock: the §11.3 underlying stop — ABOVE entry for a Phase 3
        # short. Options: honest null here — the §11.3 stop is
        # PREMIUM-based (entry premium * (1 - stop pct)) and is reported by
        # the position monitor's exit-engine read.
        "stop_price": (
            (
                position.avg_price + position.stop_distance
                if is_short_stock_position(position)
                else position.avg_price - position.stop_distance
            )
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
    # Close pricing feeds a REAL fill — always a live chain read (§42).
    _, chain = build_option_chain(position.ticker, last_close, max_age_seconds=0.0)
    contract = find_option_contract(chain, position)
    if contract is not None:
        return contract.mid, "chain mid"
    return (
        option_intrinsic_value(position, last_close),
        "intrinsic (contract missing from today's chain)",
    )


async def spread_close_reference(
    session: AsyncSession, position: Position
) -> tuple[float, str]:
    """Per-share NET reference to close a defined-risk vertical (Phase 1).

    Net mid = long mid − short mid off a LIVE chain regeneration; when
    either leg is missing from today's chain, falls back to NET intrinsic
    against the last stored close (bounded at >= 0) — contractual
    arithmetic, documented, never a silent zero."""
    last_close = await _last_stored_close(session, position.ticker)
    if last_close is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"no stored bars for {position.ticker} — the paper fill model "
                "has no reference price (honest error, §44 rule 18)"
            ),
        )
    _, chain = build_option_chain(position.ticker, last_close, max_age_seconds=0.0)
    long_leg = find_option_contract(chain, position)
    short_leg = find_spread_short_leg(chain, position)
    if long_leg is not None and short_leg is not None:
        return max(long_leg.mid - short_leg.mid, 0.0), "chain net mid (long - short)"
    long_intr = option_intrinsic_value(position, last_close)
    short_strike = position.short_strike or 0.0
    if (position.opt_right or "C") == "C":
        short_intr = max(last_close - short_strike, 0.0)
    else:
        short_intr = max(short_strike - last_close, 0.0)
    return (
        max(long_intr - short_intr, 0.0),
        "net intrinsic (leg(s) missing from today's chain)",
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

    EXECUTION VENUE (§11): everything above describes the internal simulator,
    reached only under ``BROKER_PROVIDER=simulated``. With a real broker
    configured the sell is SUBMITTED to the broker and the local rows follow
    the broker's actual fill — see :func:`_sell_to_close_via_broker`. This is
    the ONE sell implementation for manual closes AND mechanical exits alike:
    an exit that moved local rows while the broker still held the position is
    exactly the reconciliation failure §18 warns about.
    """
    if not simulated_broker_mode():
        return await _sell_to_close_via_broker(
            session,
            position,
            quantity,
            reason=reason,
            system_generated=system_generated,
        )

    settings = get_settings()
    is_option = is_option_position(position)
    is_spread = is_spread_position(position)
    is_short_stock = is_short_stock_position(position)
    multiplier = position.multiplier or 1
    if is_short_stock:
        # Phase 3 COVER: the close is a BUY — slippage moves the fill UP
        # (against the buyer), cash is DEBITED, and P&L mirrors:
        # realized = (entry − cover) × qty − commission.
        fill = reference_price * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_share * quantity
        proceeds = -(quantity * fill * multiplier) - commission
        realized = (
            (position.avg_price - fill) * quantity * multiplier - commission
        )
    else:
        fill = reference_price * (1.0 - settings.paper_slippage_bps / 10000.0)
        if is_spread:
            # Two legs close together: per-contract commission on EACH leg.
            commission = settings.paper_commission_per_contract * quantity * 2
        elif is_option:
            commission = settings.paper_commission_per_contract * quantity
        else:
            commission = settings.paper_commission_per_share * quantity
        proceeds = quantity * fill * multiplier - commission
        realized = (
            (fill - position.avg_price) * quantity * multiplier - commission
        )

    portfolio = await get_or_create_portfolio(session)
    portfolio.cash += proceeds
    portfolio.updated_at = utcnow()

    close_side = BUY_TO_CLOSE if is_short_stock else SELL_TO_CLOSE
    order = Order(
        ticker=position.ticker,
        instrument=position.instrument,
        side=close_side,
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
            "side": close_side,
            "quantity": quantity,
            "reason": reason,
            "system_generated": system_generated,
        },
    )
    submitted_details = {
        "fill_model": (
            "paper: cover BUY at reference * (1 + slippage_bps/10000); "
            "cash DEBITED qty * fill + commission"
            if is_short_stock
            else "paper: reference * (1 - slippage_bps/10000) per share; "
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



async def _sell_to_close_via_broker(
    session: AsyncSession,
    position: Position,
    quantity: int,
    *,
    reason: str | None = None,
    system_generated: bool = False,
) -> tuple[Order, float]:
    """Sell-to-close `position` through the REAL broker (§11).

    The broker half of :func:`execute_sell_to_close`. TRANSACTIONS (§11): the
    PENDING_SUBMIT order row is committed (T1) before the submit leaves the
    process — including whatever the caller had staged, see the inline note —
    and the failure paths commit their own settled state; only the SUCCESS
    path leaves the fill/position/cash mutation uncommitted for the caller
    (rule 12: the outcome and its audits land together). The local position
    shrinks by what ACTUALLY filled at the broker — never by what was
    requested — so a partial exit leaves an honestly partial position rather
    than a local flat against a broker long.

    Realized PnL is computed against the broker's own ``filled_avg_price``. A
    zero-fill leaves the position completely untouched and returns 0.0
    realized: nothing was sold, so nothing may be booked.

    Raises 502 on a broker fault and 422 on a rejection — an exit that the
    broker refused MUST NOT quietly close the local row, or the two ledgers
    diverge exactly as §18 warns.
    """
    ticker = position.ticker

    # Close the SAME instrument that was opened: an option position is closed
    # by selling its OCC contract symbol, never the underlying ticker (that
    # would sell shares we do not hold — and this account is long-only, §5).
    close_mleg_legs: list[BrokerOrderLeg] | None = None
    if is_spread_position(position):
        # Close BOTH legs atomically: SELL_TO_CLOSE the long + BUY_TO_CLOSE
        # the short (roadmap Phase 1) — never leg-by-leg, which would leave
        # a naked short between the fills.
        try:
            long_sym = occ_option_symbol(
                ticker,
                date.fromisoformat(position.opt_expiry or ""),
                float(position.opt_strike),
                position.opt_right or "",
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"cannot build the OCC symbol to close spread position "
                    f"{position.id} ({ticker}): {exc}."
                ),
            ) from exc
        short_sym = position.short_occ_symbol
        if not short_sym:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"spread position {position.id} ({ticker}) has no stored "
                    "short leg symbol; it cannot be closed at the broker "
                    "until the row is corrected."
                ),
            )
        broker_symbol = f"{long_sym}/{short_sym}"
        close_mleg_legs = [
            BrokerOrderLeg(symbol=long_sym, side=SELL_TO_CLOSE),
            BrokerOrderLeg(symbol=short_sym, side=BUY_TO_CLOSE),
        ]
    elif is_option_position(position):
        try:
            broker_symbol = occ_option_symbol(
                ticker,
                date.fromisoformat(position.opt_expiry or ""),
                float(position.opt_strike),
                position.opt_right or "",
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"cannot build the OCC symbol to close position "
                    f"{position.id} ({ticker}): {exc}. The position row is "
                    "missing or has malformed contract fields; it cannot be "
                    "closed at the broker until they are corrected."
                ),
            ) from exc
    else:
        broker_symbol = ticker
    covering_short = is_short_stock_position(position)
    close_side = BUY_TO_CLOSE if covering_short else SELL_TO_CLOSE

    # NO SHORTING (§5), enforced at the submission boundary itself.
    #
    # Both callers already constrain the quantity — /close rejects an
    # over-close before it gets here and the exit sweep always passes
    # pos.quantity — so this is defence in depth, not a live bug. It lives
    # HERE because this is the last point before a "sell" leaves the process:
    # a sell larger than the open long would leave the account SHORT, which
    # this platform must never be able to do, and that guarantee should not
    # depend on every future caller remembering to check.
    if quantity > position.quantity:
        raise HTTPException(
            status_code=422,
            detail=(
                f"refusing to sell {quantity} of {ticker} against an OPEN "
                f"position of {position.quantity}: this account is long-only "
                "and a sell may never exceed the position it closes (§5)"
            ),
        )

    # ONE closing order in flight per position (§5/§11). Without this, every
    # exit-monitor tick would re-trigger the same exit while the previous sell
    # sits ACCEPTED-unfilled at the broker — each tick minting a fresh
    # client_order_id — and two fills would sell more than the position holds.
    # The §5 quantity guard above cannot see broker-side exposure; this can.
    in_flight_sell = (
        (
            await session.execute(
                select(Order).where(
                    Order.position_id == position.id,
                    Order.side == close_side,
                    Order.status.in_(
                        [PENDING_SUBMIT, "ACCEPTED", "PARTIALLY_FILLED"]
                    ),
                )
            )
        )
        .scalars()
        .first()
    )
    if in_flight_sell is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "CLOSE_ALREADY_IN_FLIGHT",
                "message": (
                    f"position {position.id} ({ticker}) already has a closing "
                    f"order in flight (order #{in_flight_sell.id}, status "
                    f"{in_flight_sell.status}); the order-sync sweep settles "
                    "it — nothing was re-submitted"
                ),
            },
        )

    broker = resolve_broker()
    client_order_id = new_client_order_id(f"stc-{position.id}")

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM if system_generated else ActorType.USER,
        actor_id="" if system_generated else CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=client_order_id,
        details={
            "ticker": ticker,
            "instrument": position.instrument,
            "broker_symbol": broker_symbol,
            "side": close_side,
            "quantity": quantity,
            "reason": reason,
            "system_generated": system_generated,
            "client_order_id": client_order_id,
            "venue": get_settings().broker_provider,
        },
    )

    # §11 lifecycle: the local order row exists BEFORE the submit leaves the
    # process, committed durably (T1). NOTE the deliberate mid-function
    # commit: whatever the caller had staged on this session (e.g. this
    # position's EXIT_GENERATED audit in the sweep) lands with the intent
    # row — each position's exit is semantically independent, so this never
    # tears a cross-position invariant. A crash after this point leaves a
    # PENDING_SUBMIT row the order-sync sweep resolves by client_order_id.
    # opt_* mirror the position so the order row identifies the actual
    # contract sold (§44 rule 18).
    order = Order(
        client_order_id=client_order_id,
        ticker=ticker,
        instrument=position.instrument,
        opt_expiry=position.opt_expiry,
        opt_strike=position.opt_strike,
        opt_right=position.opt_right,
        side=close_side,
        quantity=quantity,
        fill_price=0.0,
        commission=0.0,  # whatever the broker charged; Alpaca paper: none
        status=PENDING_SUBMIT,
        filled_quantity=0,
        position_id=position.id,
    )
    session.add(order)
    await session.flush()
    await session.commit()

    try:
        if close_mleg_legs is not None:
            broker_order, adopted = await submit_mleg_and_poll(
                broker, client_order_id, close_mleg_legs, quantity
            )
        elif covering_short:
            # Phase 3: buy-to-cover through the dedicated stock-gated
            # adapter method — risk-reducing, allowed under the pause.
            broker_order, adopted = await submit_stock_cover_and_poll(
                broker, client_order_id, broker_symbol, quantity
            )
        else:
            broker_order, adopted = await submit_and_poll(
                broker, client_order_id, broker_symbol, SELL_TO_CLOSE, quantity
            )
    except BrokerRejected as exc:
        order.status = "REJECTED"
        order.broker_status = "rejected_at_submit"
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_REJECTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "ticker": ticker,
                "side": close_side,
                "client_order_id": client_order_id,
                "reason": str(exc),
                "rejected_by": "broker",
                "position_unchanged": True,
            },
        )
        await session.commit()
        raise HTTPException(
            status_code=422,
            detail={
                "code": "BROKER_REJECTED",
                "message": (
                    f"the broker rejected the closing order: {exc}. The "
                    "position is UNCHANGED locally — it is still open at the "
                    "broker too."
                ),
            },
        ) from exc
    except BrokerError as exc:
        # A FAULT, not a decision: the row STAYS PENDING_SUBMIT for the
        # order-sync sweep to settle against the broker.
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_SUBMITTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "ticker": ticker,
                "side": close_side,
                "client_order_id": client_order_id,
                "error": str(exc),
                "outcome": (
                    "broker call FAILED — the closing order may or may not "
                    "exist at the broker; the local position was NOT changed "
                    "and the order row stays PENDING_SUBMIT until the "
                    "order-sync sweep resolves it."
                ),
            },
        )
        await session.commit()
        raise HTTPException(
            status_code=502,
            detail={
                "code": "BROKER_ERROR",
                "message": (
                    f"the broker call failed: {exc}. The position was left "
                    "untouched locally; the order-sync sweep will resolve the "
                    "order's true state — do not blindly retry."
                ),
            },
        ) from exc

    filled = broker_order.filled_quantity
    fill_price = broker_order.filled_avg_price

    if filled > 0 and fill_price is None:
        # PRICE-PENDING FILL (same eventual-consistency gap as the buy path):
        # contracts sold but no average price yet — the proceeds cannot be
        # credited honestly, so NOTHING is applied. filled_quantity stays 0
        # (the sweep's delta measures applied state) and the status is forced
        # non-terminal so the sweep applies the sale when the price arrives.
        # The position is deliberately untouched until then.
        order.status = "ACCEPTED"
        order.broker_order_id = broker_order.broker_order_id or None
        order.broker_status = (
            broker_order.raw_status[:24] if broker_order.raw_status else None
        )
        await session.flush()
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_SUBMITTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "outcome": (
                    f"broker reports {filled} sold but no average price yet "
                    "— the sale is NOT applied (no price, no cash credit, "
                    "position unchanged) and the row stays non-terminal; the "
                    "order-sync sweep applies it once the broker publishes "
                    "the price"
                ),
                **broker_order_details(broker_order, adopted=adopted),
            },
        )
        return order, 0.0

    # Settle the PENDING_SUBMIT row to what the broker actually did (§11).
    # INVARIANT: filled_quantity/fill_price record what has been APPLIED to
    # cash and the position — the order-sync sweep's delta depends on it.
    order.fill_price = fill_price if fill_price is not None else 0.0
    order.status = broker_order.status
    order.broker_order_id = broker_order.broker_order_id or None
    order.broker_status = (
        broker_order.raw_status[:24] if broker_order.raw_status else None
    )
    order.filled_quantity = filled
    await session.flush()

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_model": (
                "broker: quantity, price and status are the BROKER's — no "
                "internal fill model runs on this path"
            ),
            "venue": get_settings().broker_provider,
            **broker_order_details(broker_order, adopted=adopted),
        },
    )

    if broker_order.status == "REJECTED":
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_REJECTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "ticker": ticker,
                "side": close_side,
                "reason": (
                    f"the broker settled the closing order as "
                    f"{broker_order.raw_status!r}"
                ),
                "rejected_by": "broker",
                "position_unchanged": True,
                **broker_order_details(broker_order, adopted=adopted),
            },
        )
        return order, 0.0

    if filled <= 0 or fill_price is None:
        # NOTHING SOLD. The position stays exactly as it was — shrinking it
        # here would create a local flat against a real broker long, which is
        # the precise divergence the reconciliation kill switch exists for.
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_SUBMITTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "outcome": (
                    "no quantity filled — the position is UNCHANGED and no "
                    "cash moved. The closing order stands at the broker in "
                    f"status {broker_order.raw_status!r}."
                ),
                "filled_quantity": filled,
                **broker_order_details(broker_order, adopted=adopted),
            },
        )
        return order, 0.0

    # Options are quoted PER SHARE and trade in 100-share contracts, so both
    # the cash credited and the realized P&L carry the position's multiplier.
    # Dropping it here would credit an option close at 1/100th of its value.
    multiplier = position.multiplier or 1
    proceeds = filled * fill_price * multiplier
    realized = (
        (position.avg_price - fill_price) * filled * multiplier
        if covering_short
        else (fill_price - position.avg_price) * filled * multiplier
    )
    # NO LOCAL CASH LEDGER in broker mode: the proceeds landed in the REAL
    # account at the broker, and the platform stores no copy of it — cash is
    # read live from the broker wherever it is displayed or sized from.

    remaining = position.quantity - filled
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
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_FILLED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_price": fill_price,
            "filled_quantity": filled,
            "requested_quantity": quantity,
            "partial": filled < quantity,
            "commission": 0.0,
            "position_id": position.id,
            "realized_pnl": realized,
            "cash_credited": proceeds,
            **broker_order_details(broker_order, adopted=adopted),
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

    EXECUTION VENUE: with ``BROKER_PROVIDER=simulated`` everything above runs
    exactly as documented. With a real broker the QUANTITY, PRICE and STATUS
    all come from the broker instead: the order is submitted with our
    ``client_order_id``, polled briefly, and the position opens with the FILLED
    quantity — a partial fill opens a partial position, a zero-fill ACCEPTED
    order opens none. Option instruments are 422 on the broker path (module
    docstring).

    503 ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is
    configured — checked BEFORE the lock and before the idempotency lookup, so
    an unconfigured install cannot fill an order at a made-up price.

    503 ``BROKER_NOT_CONFIGURED`` when no execution venue is configured, for
    the same reason one step further along: with no broker there is nowhere to
    place the order, and quietly simulating one instead would report an
    execution that never happened. Nothing is written — no Order row, no
    Position row, no cash movement.
    """
    require_market_data_provider()
    require_broker()
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

    # --- ...and one in-flight opening order per ticker (§11/§42). ----------
    # A non-terminal BUY is exposure that has not landed yet: it may fill any
    # moment. Approving another would place TWO broker orders for one intent —
    # the position-based check above cannot see it (a zero-fill order has no
    # position). The order-sync sweep settles the in-flight row; retry after.
    in_flight = (
        (
            await session.execute(
                select(Order).where(
                    Order.ticker == ticker,
                    # Phase 3: a stock short opens with SELL_TO_OPEN — both
                    # opening sides count as in-flight exposure.
                    Order.side.in_([BUY_TO_OPEN, SELL_TO_OPEN]),
                    Order.status.in_(
                        [PENDING_SUBMIT, "ACCEPTED", "PARTIALLY_FILLED"]
                    ),
                )
            )
        )
        .scalars()
        .first()
    )
    if in_flight is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{ticker} already has an opening order in flight (order "
                f"#{in_flight.id}, status {in_flight.status}) — it may still "
                "fill at the broker. The order-sync sweep settles it; retry "
                "after it reaches a terminal state."
            ),
        )

    # --- Re-run the FULL §10 chain (§42: client previews are never trusted).
    chain = await run_gate_chain(
        session, ticker, req.quantity, req.direction, mode="execution"
    )
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

    # --- Execution venue (§11, §44 rule 18) --------------------------------
    # The chain has passed and a quantity is approved. WHERE the order goes is
    # decided here and only here: a real broker, or — only when explicitly
    # opted into — the internal simulator below. The unconfigured case never
    # reaches this line (approve_order raised 503 before taking the lock).
    if not simulated_broker_mode():
        return await _approve_via_broker(session, req, chain, quantity)

    # --- Paper fill (§11): slippage against the trader + commission. -------
    # Options (§12.1): fill = contract mid * (1 + slippage) PER SHARE; the
    # cash debit carries the x100 multiplier plus the per-contract
    # commission, and quantity counts CONTRACTS.
    settings = get_settings()
    is_option = chain.contract is not None
    is_spread = chain.spread is not None
    is_short_stock = chain.instrument == InstrumentType.SHORT_STOCK.value
    if is_short_stock:
        # Phase 3: the OPEN is a SELL — slippage moves the fill DOWN
        # (against the seller) and the net proceeds are CREDITED to cash.
        # The liability is the open position itself, carried at NEGATIVE
        # market value by the portfolio view; NAV stays honest throughout.
        fill = chain.entry_price * (1.0 - settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_share * quantity
        cost = 0.0  # nothing is debited; the guard below never trips
    elif is_option:
        contract = chain.contract
        fill = contract.mid * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_contract * quantity
        cost = quantity * fill * OPTION_MULTIPLIER + commission
    elif is_spread:
        # §12.1 defined risk: fill = NET debit * (1 + slippage) per share —
        # slippage adverse on the net; commission PER LEG (×2).
        spread = chain.spread
        fill = spread.net_debit * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_contract * quantity * 2
        cost = quantity * fill * OPTION_MULTIPLIER + commission
    else:
        fill = chain.entry_price * (1.0 + settings.paper_slippage_bps / 10000.0)
        commission = settings.paper_commission_per_share * quantity
        cost = quantity * fill + commission

    portfolio = await get_or_create_portfolio(session)
    if not is_short_stock and cost > portfolio.cash:
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
        side=SELL_TO_OPEN if is_short_stock else BUY_TO_OPEN,
        quantity=quantity,
        fill_price=fill,
        commission=commission,
        status="FILLED",
        # Spread rows: opt_* identify the LONG leg (the close path and the
        # audit trail carry the short leg separately).
        opt_expiry=(
            contract.expiry.isoformat()
            if is_option
            else chain.spread.long_leg.expiry.isoformat()
            if is_spread
            else None
        ),
        opt_strike=(
            contract.strike
            if is_option
            else chain.spread.long_leg.strike
            if is_spread
            else None
        ),
        opt_right=(
            contract.right
            if is_option
            else chain.spread.long_leg.right
            if is_spread
            else None
        ),
    )
    session.add(order)
    if is_short_stock:
        portfolio.cash += quantity * fill - commission  # short proceeds
    else:
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
    elif is_spread:
        # Defined risk (§12.1): max_loss = the whole NET debit; stop_distance
        # stores the per-share net fill — the §11.3 premium-stop basis ON THE
        # NET. opt_* = long leg; short_* = short leg (migration 015).
        spread = chain.spread
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
            opt_expiry=spread.long_leg.expiry.isoformat(),
            opt_strike=spread.long_leg.strike,
            opt_right=spread.long_leg.right,
            multiplier=OPTION_MULTIPLIER,
            short_occ_symbol=occ_option_symbol(
                ticker,
                spread.short_leg.expiry,
                spread.short_leg.strike,
                spread.short_leg.right,
            ),
            short_strike=spread.short_leg.strike,
        )
    elif is_short_stock:
        # §12.1 Phase 3: heat carries the GAP-inflated risk the chain sized
        # from; stop_distance stays the per-share 2×ATR — the exit engine
        # mirrors it ABOVE entry for the BEAR direction.
        position = Position(
            ticker=ticker,
            instrument=InstrumentType.SHORT_STOCK.value,
            quantity=quantity,
            avg_price=fill,
            max_loss=quantity * chain.stop_distance * SHORT_STOCK_GAP_RISK_FACTOR,
            stop_distance=chain.stop_distance,
            entry_edge=chain.edge,
            entry_bar_date=chain.last_bar_date,
            status=POSITION_OPEN,
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
            "side": order.side,
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
    elif is_spread:
        submitted_details = {
            "fill_model": (
                "paper: NET debit * (1 + slippage_bps/10000) per share; "
                "cash = qty * net_fill * 100 + 2-leg commission"
            ),
            "net_debit_mid": chain.spread.net_debit,
            "long_leg_mid": chain.spread.long_leg.mid,
            "short_leg_mid": chain.spread.short_leg.mid,
            "short_occ_symbol": position.short_occ_symbol,
            "multiplier": OPTION_MULTIPLIER,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_contract": settings.paper_commission_per_contract,
        }
    elif is_short_stock:
        submitted_details = {
            "fill_model": (
                "paper: last stored close * (1 - slippage_bps/10000) — a "
                "short SELL fills lower; cash CREDITED qty*fill - commission"
            ),
            "last_close": chain.entry_price,
            "slippage_bps": settings.paper_slippage_bps,
            "commission_per_share": settings.paper_commission_per_share,
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
                None
                if is_option or is_spread
                else position.avg_price + position.stop_distance
                if is_short_stock
                else position.avg_price - position.stop_distance
            ),
            "max_loss": position.max_loss,
            "contract": _contract_payload(position),
        },
        "preview": chain.preview,
    }


# ---------------------------------------------------------------------------
# Real broker execution (plan §11) — the BUY_TO_OPEN half.
# ---------------------------------------------------------------------------

# The 422 message for an instrument with no broker representation.
# LONG_STOCK, LONG_CALL, LONG_PUT, the defined-risk verticals (mleg) and the
# Phase 3 margin-backed SHORT_STOCK all execute at the broker; anything else
# (the naked shorts, forever) is refused rather than approximated.
BROKER_INSTRUMENT_UNSUPPORTED = (
    "{instrument} cannot be executed at the broker: this platform submits "
    "LONG_STOCK, LONG_CALL, LONG_PUT, defined-risk verticals and "
    "margin-backed SHORT_STOCK (§5). The §8 matrix selected {instrument} "
    "for {ticker}."
)


async def _approve_via_broker(
    session: AsyncSession, req: OrderApproveRequest, chain: GateChainResult, quantity: int
) -> dict:
    """Submit an approved BUY_TO_OPEN to the real broker and record the truth.

    Called with the §10 chain already PASSED, the risk-approved quantity fixed
    and the shared execution lock held. What happens from here is the broker's
    to decide; this function's whole job is to record what it decided without
    embellishment:

    - the order row is written with the broker's id, its RAW status and the
      ACTUALLY filled quantity;
    - a position opens ONLY for quantity that actually filled, at the broker's
      own ``filled_avg_price`` — never at a modelled price;
    - a zero-fill ACCEPTED order writes the order row, opens NO position, moves
      NO cash, and is audited as exactly that;
    - a rejection writes no position and audits ORDER_REJECTED with the
      broker's own reason.

    Idempotency spans the network (§42): our ``client_order_id`` goes to the
    broker as its client_order_id and an order it already holds under that key
    is ADOPTED rather than submitted twice. When the caller supplied no key,
    one is generated here and stored, so the order remains recognisable after a
    lost response.

    NO LOCAL FILL-TIME CASH GUARD, deliberately. The simulated path re-checks
    cash because it invents the price and could otherwise overdraw a number
    only it controls. Here the §13 cash floor was already enforced by
    RISK_APPROVAL, and the BROKER owns buying power: an order it cannot fund is
    rejected by the broker itself and arrives as a :class:`BrokerRejected`.
    Refusing locally on our own cash figure would refuse orders the broker
    would happily fill — and our cash figure is precisely the number
    reconciliation exists to check, so it is the wrong thing to arbitrate on.

    Commits in TWO transactions (§11 lifecycle): T1 — the chain's
    RISK_DECISION, the ORDER_REQUESTED audit and the order row in
    PENDING_SUBMIT — lands BEFORE the submit leaves the process, so a crash
    mid-submit leaves a durable row the order-sync sweep can resolve against
    the broker by client_order_id. T2 — the settled row, any position/cash
    mutation and the outcome ORDER_* audits — lands together after the broker
    answers (rule 12 per state change).
    """
    ticker = req.ticker

    # LONG_STOCK trades the ticker; LONG_CALL/LONG_PUT trade an OCC contract
    # symbol on the SAME endpoint. Anything else has no broker representation
    # in this long-only platform (§5) and is refused before submission, with
    # the veto's RISK_DECISION still auditable.
    contract = chain.contract
    spread = chain.spread
    mleg_legs: list[BrokerOrderLeg] | None = None
    stock_short = chain.instrument == InstrumentType.SHORT_STOCK.value
    if contract is None and spread is None and stock_short:
        # Phase 3: SELL_TO_OPEN the ticker itself — the dedicated
        # margin-attested adapter method; Alpaca enforces locate/HTB and
        # maintenance margin on its side.
        broker_symbol = ticker
        instrument = InstrumentType.SHORT_STOCK.value
    elif contract is None and spread is None:
        broker_symbol = ticker
        instrument = InstrumentType.LONG_STOCK.value
    elif spread is not None and chain.instrument in (
        InstrumentType.BULL_CALL_SPREAD.value,
        InstrumentType.BEAR_PUT_SPREAD.value,
    ):
        # Roadmap Phase 1: one ATOMIC mleg order — the adapter's shape guard
        # revalidates the defined-risk pair before any I/O.
        instrument = chain.instrument
        try:
            long_sym = occ_option_symbol(
                ticker,
                spread.long_leg.expiry,
                spread.long_leg.strike,
                spread.long_leg.right,
            )
            short_sym = occ_option_symbol(
                ticker,
                spread.short_leg.expiry,
                spread.short_leg.strike,
                spread.short_leg.right,
            )
        except ValueError as exc:
            await session.commit()
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        f"cannot build OCC symbols for the {ticker} spread: {exc}"
                    ),
                    "preview": chain.preview,
                },
            ) from exc
        broker_symbol = f"{long_sym}/{short_sym}"
        mleg_legs = [
            BrokerOrderLeg(symbol=long_sym, side=BUY_TO_OPEN),
            BrokerOrderLeg(symbol=short_sym, side=SELL_TO_OPEN),
        ]
    elif chain.instrument in (
        InstrumentType.LONG_CALL.value,
        InstrumentType.LONG_PUT.value,
    ):
        instrument = chain.instrument
        try:
            broker_symbol = occ_option_symbol(
                ticker, contract.expiry, contract.strike, contract.right
            )
        except ValueError as exc:
            await session.commit()
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        f"cannot build an OCC option symbol for {ticker}: {exc}"
                    ),
                    "preview": chain.preview,
                },
            ) from exc
    else:
        await session.commit()
        raise HTTPException(
            status_code=422,
            detail={
                "message": BROKER_INSTRUMENT_UNSUPPORTED.format(
                    instrument=chain.instrument, ticker=ticker
                ),
                "preview": chain.preview,
            },
        )

    broker = resolve_broker()
    client_order_id = req.client_order_id or new_client_order_id("bto")

    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.ORDER_REQUESTED,
        entity_type="order",
        entity_id=client_order_id,
        details={
            "ticker": ticker,
            "instrument": instrument,
            # The exact string sent to the broker — for an option this is the
            # OCC symbol, so the audit trail identifies the actual contract.
            "broker_symbol": broker_symbol,
            "side": SELL_TO_OPEN if stock_short else BUY_TO_OPEN,
            "quantity_requested": req.quantity,
            "quantity_submitted": quantity,
            "client_order_id": client_order_id,
            "venue": get_settings().broker_provider,
        },
    )

    # §11 lifecycle: the local order row exists BEFORE the submit leaves the
    # process, committed durably (T1) with the request audit and the chain's
    # RISK_DECISION. If we crash mid-submit, this row — findable at the broker
    # by its client_order_id — is what the order-sync sweep resolves, instead
    # of an invisible broker order. fill_price 0.0 / filled_quantity 0 are
    # honest ("nothing filled yet"), never placeholders.
    order = Order(
        client_order_id=client_order_id,
        ticker=ticker,
        instrument=instrument,
        # opt_* identify WHICH contract trades; None for stock. Without these
        # the position could not later be closed at the broker (the close path
        # rebuilds the OCC symbol from exactly these three fields).
        opt_expiry=(
            contract.expiry.isoformat()
            if contract is not None
            else spread.long_leg.expiry.isoformat()
            if spread is not None
            else None
        ),
        opt_strike=(
            contract.strike
            if contract is not None
            else spread.long_leg.strike
            if spread is not None
            else None
        ),
        opt_right=(
            contract.right
            if contract is not None
            else spread.long_leg.right
            if spread is not None
            else None
        ),
        side=SELL_TO_OPEN if stock_short else BUY_TO_OPEN,
        quantity=quantity,
        fill_price=0.0,
        commission=0.0,
        status=PENDING_SUBMIT,
        filled_quantity=0,
        # Approval-time risk context (migration 010): a fill that arrives
        # AFTER this request returned (surfaced by the order-sync sweep) must
        # open the position with the §10 chain's own parameters, so they are
        # captured here while the chain still exists.
        stop_distance=chain.stop_distance,
        entry_edge=chain.edge,
        entry_bar_date=chain.last_bar_date,
    )
    session.add(order)
    await session.flush()
    await session.commit()

    try:
        if mleg_legs is not None:
            broker_order, adopted = await submit_mleg_and_poll(
                broker, client_order_id, mleg_legs, quantity
            )
        elif stock_short:
            # The attestation names the §10 chain audit that sized this
            # short — the adapter refuses an unattested call outright.
            broker_order, adopted = await submit_stock_short_and_poll(
                broker,
                client_order_id,
                broker_symbol,
                quantity,
                margin_attested_by=f"gate-chain:{client_order_id}",
            )
        else:
            broker_order, adopted = await submit_and_poll(
                broker, client_order_id, broker_symbol, BUY_TO_OPEN, quantity
            )
    except BrokerRejected as exc:
        # A business rejection: the broker's answer was "no". The PENDING_SUBMIT
        # row settles to REJECTED — a terminal, audited state — with no
        # position and no cash movement.
        order.status = "REJECTED"
        order.broker_status = "rejected_at_submit"
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_REJECTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "ticker": ticker,
                "client_order_id": client_order_id,
                "reason": str(exc),
                "rejected_by": "broker",
            },
        )
        await session.commit()
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"the broker rejected the order: {exc}",
                "preview": chain.preview,
            },
        ) from exc
    except BrokerError as exc:
        # A FAULT, not a decision: the order may or may not exist at the
        # broker. We must not claim either, so the row STAYS PENDING_SUBMIT —
        # the order-sync sweep looks it up at the broker by client_order_id
        # and settles it to whatever actually happened.
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_SUBMITTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "ticker": ticker,
                "client_order_id": client_order_id,
                "error": str(exc),
                "outcome": (
                    "broker call FAILED — the order may or may not exist at "
                    "the broker; the local row stays PENDING_SUBMIT and the "
                    "order-sync sweep (or GET /api/broker/reconcile) resolves "
                    "it against the broker by client_order_id."
                ),
            },
        )
        await session.commit()
        raise HTTPException(
            status_code=502,
            detail={
                "code": "BROKER_ERROR",
                "message": (
                    f"the broker call failed: {exc}. The order may or may not "
                    "have reached the broker — the local order row stays "
                    "PENDING_SUBMIT until the order-sync sweep resolves it; "
                    "do not blindly retry."
                ),
            },
        ) from exc

    filled = broker_order.filled_quantity
    fill_price = broker_order.filled_avg_price

    if filled > 0 and fill_price is None:
        # PRICE-PENDING FILL (broker eventual consistency: filled_qty can
        # populate before filled_avg_price). The fill CANNOT be applied — no
        # price means no honest cash debit — so NOTHING is recorded as
        # applied: filled_quantity stays 0 (the order-sync sweep's
        # incremental delta measures against what was APPLIED, not what was
        # reported) and the local status is forced NON-terminal even if the
        # broker already says "filled", so the sweep keeps watching and
        # applies the fill the moment the price populates.
        order.status = "ACCEPTED"
        order.broker_order_id = broker_order.broker_order_id or None
        order.broker_status = (
            broker_order.raw_status[:24] if broker_order.raw_status else None
        )
        await session.flush()
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_SUBMITTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "outcome": (
                    f"broker reports {filled} filled but no average price "
                    "yet — the fill is NOT applied (no price, no cash "
                    "movement, no position) and the row stays non-terminal; "
                    "the order-sync sweep applies it once the broker "
                    "publishes the price"
                ),
                "venue": get_settings().broker_provider,
                **broker_order_details(broker_order, adopted=adopted),
            },
        )
        await session.commit()
        return {
            "order": _order_payload(order),
            "position": None,
            "preview": chain.preview,
        }

    # Settle the PENDING_SUBMIT row to what the broker actually did (§11).
    # No modelled price EVER lands here: an unfilled order's fill_price stays
    # 0.0 because nothing was paid, not because we guessed. Commission is
    # whatever the broker charged (Alpaca paper: none) — never the internal
    # model's number. INVARIANT: filled_quantity/fill_price record what has
    # been APPLIED to cash and positions (the branch below applies it in the
    # same transaction) — the order-sync sweep's delta arithmetic depends on
    # this never diverging.
    order.fill_price = fill_price if fill_price is not None else 0.0
    order.status = broker_order.status
    order.broker_order_id = broker_order.broker_order_id or None
    order.broker_status = (
        broker_order.raw_status[:24] if broker_order.raw_status else None
    )
    order.filled_quantity = filled
    await session.flush()

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.ORDER_SUBMITTED,
        entity_type="order",
        entity_id=str(order.id),
        details={
            "fill_model": (
                "broker: quantity, price and status are the BROKER's — no "
                "internal fill model runs on this path"
            ),
            "venue": get_settings().broker_provider,
            **broker_order_details(broker_order, adopted=adopted),
        },
    )

    position: Position | None = None
    if broker_order.status == "REJECTED":
        # A rejection discovered by POLLING (not at submit time) is a stored
        # order in a terminal state — recorded, audited, no position.
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_REJECTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "ticker": ticker,
                "reason": (
                    f"the broker settled the order as {broker_order.raw_status!r}"
                ),
                "rejected_by": "broker",
                **broker_order_details(broker_order, adopted=adopted),
            },
        )
    elif filled > 0 and fill_price is not None:
        # A REAL fill: the position opens for the FILLED quantity at the
        # BROKER's average price — never the requested quantity (§11).
        #
        # Options are priced PER SHARE but trade in 100-share contracts, so
        # cash and max loss both carry the multiplier. For a long option the
        # premium paid IS the max loss (§12.1); for stock it is the stop
        # distance per share.
        is_option_like = contract is not None or spread is not None
        multiplier = 100 if is_option_like else 1
        cost = filled * fill_price * multiplier
        # Phase 3 short stock: heat carries the GAP-inflated stop risk the
        # chain sized from (§12.1); a short's premium-free max_loss is
        # unbounded in principle, so the stop-based estimate is the honest
        # heat number, inflated for gap risk.
        short_stock_max_loss = (
            filled * (chain.stop_distance or 0.0) * SHORT_STOCK_GAP_RISK_FACTOR
        )
        # NO LOCAL CASH LEDGER in broker mode: the debit happened in the REAL
        # account at the broker; the platform stores no copy of it. For a
        # spread, fill_price is the broker-reported NET debit per share.
        position = Position(
            ticker=ticker,
            instrument=instrument,
            quantity=filled,
            avg_price=fill_price,
            multiplier=multiplier,
            opt_expiry=(
                contract.expiry.isoformat()
                if contract is not None
                else spread.long_leg.expiry.isoformat()
                if spread is not None
                else None
            ),
            opt_strike=(
                contract.strike
                if contract is not None
                else spread.long_leg.strike
                if spread is not None
                else None
            ),
            opt_right=(
                contract.right
                if contract is not None
                else spread.long_leg.right
                if spread is not None
                else None
            ),
            short_occ_symbol=(
                broker_symbol.split("/")[1] if spread is not None else None
            ),
            short_strike=spread.short_leg.strike if spread is not None else None,
            max_loss=(
                cost
                if is_option_like
                else short_stock_max_loss
                if stock_short
                else filled * (chain.stop_distance or 0.0)
            ),
            # For options/spreads this is the per-share (net) premium basis
            # the §11.3 premium stop measures against.
            stop_distance=(
                fill_price if is_option_like else (chain.stop_distance or 0.0)
            ),
            entry_edge=chain.edge,
            entry_bar_date=chain.last_bar_date,
            status=POSITION_OPEN,
        )
        session.add(position)
        await session.flush()
        # Link the order to the position it opened (migration 010): further
        # fills found by the order-sync sweep land on THIS position row.
        order.position_id = position.id
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_FILLED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "fill_price": fill_price,
                "filled_quantity": filled,
                "requested_quantity": quantity,
                "partial": filled < quantity,
                "commission": 0.0,
                "position_id": position.id,
                # For a Phase 3 short the broker CREDITED the proceeds;
                # either way the real account holds the truth (§14).
                ("cash_credited" if stock_short else "cash_debited"): cost,
                **broker_order_details(broker_order, adopted=adopted),
            },
        )
    else:
        # ZERO FILL. The order is live (or settled unfilled) at the broker and
        # nothing has happened yet: no position, no cash movement, and the
        # audit says so in as many words rather than leaving a silent gap.
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.ORDER_SUBMITTED,
            entity_type="order",
            entity_id=str(order.id),
            details={
                "outcome": (
                    "no quantity filled — NO position was opened and no cash "
                    "moved. The order stands at the broker in status "
                    f"{broker_order.raw_status!r}; whatever fills later is "
                    "surfaced by GET /api/broker/reconcile."
                ),
                "filled_quantity": filled,
                **broker_order_details(broker_order, adopted=adopted),
            },
        )

    await session.commit()

    return {
        "order": _order_payload(order),
        "position": (
            {
                "id": position.id,
                "ticker": position.ticker,
                "instrument": position.instrument,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
                "stop_price": (
                    position.avg_price + position.stop_distance
                    if stock_short
                    else position.avg_price - position.stop_distance
                ),
                "max_loss": position.max_loss,
                "contract": None,
            }
            if position is not None
            else None
        ),
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

    EXECUTION VENUE (§11): the reference-price machinery above is the internal
    simulator, reached only under ``BROKER_PROVIDER=simulated``. With a real
    broker the sell is submitted to the broker and the position shrinks by what
    ACTUALLY filled, at the broker's own average price — no reference price and
    no slippage model are involved.

    503 ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is
    configured. Closing normally outranks the §18 kill switch because it
    reduces risk — but this is not a policy pause, it is the absence of a
    price. A fill must happen AT something, and booking realized PnL against
    an invented number would corrupt the ledger permanently. Refusing is the
    conservative answer: the position stays open and honest.

    503 ``BROKER_NOT_CONFIGURED`` when no execution venue is configured, on the
    same reasoning: with nowhere to send the sell, marking the position closed
    locally would claim an exit that never happened. The position stays open.
    """
    require_market_data_provider()
    require_broker()
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

    # Phase 2 collateral law: shares pinned under OPEN covered calls cannot
    # be sold — buy the call back first. Only the FREE portion may close.
    if position.instrument == InstrumentType.LONG_STOCK.value:
        pinned = (
            (
                await session.execute(
                    select(func.coalesce(func.sum(Position.quantity), 0)).where(
                        Position.collateral_position_id == position.id,
                        Position.status == POSITION_OPEN,
                        Position.instrument == InstrumentType.COVERED_CALL.value,
                    )
                )
            ).scalar_one()
            * 100
        )
        free = position.quantity - pinned
        if pinned > 0 and quantity > free:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{pinned} of {position.quantity} {ticker} shares are "
                    "pinned as covered-call collateral — buy back the open "
                    f"covered call(s) first, or close at most {free} free "
                    "share(s)."
                ),
            )

    # The reference price is the INTERNAL fill model's input. On the broker
    # path there is nothing to reference — the broker sets the price — so it is
    # neither computed nor required (0.0 is passed and ignored downstream).
    reference, source = 0.0, "broker fill (no reference price used)"
    if simulated_broker_mode():
        if is_spread_position(position):
            reference, source = await spread_close_reference(session, position)
        elif is_option_position(position):
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
