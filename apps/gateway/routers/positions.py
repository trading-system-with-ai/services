"""Position monitor API (development plan §11, §37, §38).

``GET /api/positions`` lists paper positions with their live exit-engine read:
every OPEN stock row is evaluated by
:func:`libs.trading_core.exits.evaluate_exit` and every OPEN option row by
:func:`libs.trading_core.exits.evaluate_option_exit` — the SAME engines the
backtest validated (plan §21), never reimplemented here — and carries the
full per-rule reason list, "OK:"-prefixed for rules that did NOT fire, so the
user can always see why the system is still holding (§37). Option rows'
``exit_reasons`` therefore include the option families (§11.3 PREMIUM
hard stop, §11.7 DTE exit) alongside the shared underlying-driven rules.
CLOSED rows carry ``realized_pnl`` and honest nulls elsewhere (§44 rule 18).
Read-only: no audit events (rule 12 covers state changes and decisions).

Option rows (plan §12.1 conventions): ``quantity`` counts CONTRACTS,
``avg_price`` is the entry premium PER SHARE (mid at fill), ``market_value``
is ``qty * current_mid * 100``, ``max_loss`` the premium paid. The
``contract`` block carries the identity plus the live read — remaining
``dte`` from ``opt_expiry`` and ``current_mid`` from the SAME contract in
today's regenerated chain (shared helper in routers/options.py); both are
honest nulls when unavailable (e.g. the contract expired off the chain).

``POST /api/positions/check-exits`` runs the same evaluation for every OPEN
position and EXECUTES the triggered exits for BOTH instrument kinds through
the shared sell-to-close flow: each fires an EXIT_GENERATED audit event
(SYSTEM) and the sell (ORDER_REQUESTED actor SYSTEM) in the same
transaction. Option exits close at the current chain mid, falling back to
INTRINSIC value when the contract is missing from today's chain (documented
in routers/orders.py). Mechanical exits are NOT blocked by the §18 kill
switch: exits reduce risk, and risk protection outranks the pause (§18
risk-priority).

MECHANICAL EXITS USE THE SAME BROKER PATH AS MANUAL CLOSES (plan §11, §18).
``execute_sell_to_close`` is the single sell implementation for both, so a
triggered exit is submitted to the configured broker exactly like a
user-initiated close. There is deliberately no local-only shortcut: an exit
that flattened the local row while the broker still held the position is
precisely the divergence the §18 reconciliation kill switch exists to catch,
and it would be a silent one.

NO BROKER, NO SWEEP: when no execution venue is configured the sweep SKIPS
with a warning — like the no-market-data case — rather than closing positions
locally. Closing a row we cannot actually sell would be a fabricated exit.
"""
import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.trading_core.exits import (
    ExitDecision,
    OptionState,
    PositionState,
    evaluate_exit,
    evaluate_option_exit,
    ShortPremiumState,
    evaluate_short_premium_exit,
)
from libs.trading_core.models import ActorType, AuditAction, InstrumentType

from .. import audit
from ..db import Position, StockBarDaily, get_session
from ..deps import (
    BROKER_NOT_CONFIGURED,
    broker_configured,
    broker_unavailable_reason,
    market_data_configured,
    require_market_data_provider,
)
from .analysis import EASTERN as EASTERN_TZ
from .options import option_chain_or_none
# Phase D (design §8.5, spec §52): the option-row risk fields come from the
# persisted stress run of the newest snapshot — a READ of history, so the
# positions list stays a read view and never re-runs the catalogue.
from ..risk_snapshot import (
    latest_worst_scenario_per_position,
    worst_scenario_pnl_for_key,
)
from libs.broker.alpaca import occ_option_symbol

from .orders import (
    POSITION_OPEN,
    execute_sell_to_close,
    execution_lock,
    find_option_contract,
    find_spread_short_leg,
    is_option_position,
    is_short_stock_position,
    is_spread_position,
    option_intrinsic_value,
)

router = APIRouter(prefix="/api/positions", tags=["positions"])

# Named "position_monitor" so a skipped sweep logs on the SAME logger whether
# it was triggered by the endpoint or the background loop.
logger = logging.getLogger("position_monitor")

VALID_STATUS = ("OPEN", "CLOSED", "ALL")

# Shown in an OPEN row's ``exit_reasons`` when no market data provider is
# configured: the position is real, but nothing market-derived about it can be
# known, so the gap is NAMED rather than filled with a synthetic number.
NO_MARKET_DATA_REASON = (
    "DATA_ISSUE: no market data provider is configured — no current price, "
    "market value or exit evaluation is available for this position"
)


async def _stored_bars(session: AsyncSession, ticker: str) -> list[StockBarDaily]:
    """All stored daily bars for `ticker`, oldest first (no lazy backfill —
    the monitor only reads what execution already stored)."""
    rows = await session.execute(
        select(StockBarDaily)
        .where(StockBarDaily.ticker == ticker)
        .order_by(StockBarDaily.ts)
    )
    return list(rows.scalars().all())


def _bars_held(position: Position, bars: list[StockBarDaily]) -> int:
    """Bars since entry; the entry bar (``entry_bar_date``) is bar 0 (§11).

    A missing ``entry_bar_date`` conservatively counts 0 — the time stop can
    then never fire early, while every price-based rule still protects.
    """
    if position.entry_bar_date is None:
        return 0
    entry = date.fromisoformat(position.entry_bar_date)
    return sum(1 for b in bars if b.ts > entry)


def _dte_remaining(position: Position) -> int | None:
    """Calendar days from today to ``opt_expiry``, clamped at 0; ``None`` for
    a row with no expiry (stock).

    US-exchange trading day (EASTERN): after 20:00 ET the UTC calendar has
    already rolled over, which made this DTE differ by one from the chain
    paths every US evening. ONE clock for every options surface — which is
    why this is a shared helper and not two copies of the arithmetic.
    """
    if not position.opt_expiry:
        return None
    today = datetime.now(EASTERN_TZ).date()
    return max(0, (date.fromisoformat(position.opt_expiry) - today).days)


def _option_live_read(
    position: Position, spot: float
) -> tuple[int | None, float | None]:
    """Live ``(dte_remaining, current_mid)`` for one OPEN option position.

    ``dte`` counts calendar days from today to ``opt_expiry``, clamped at 0
    once expired (the remaining time cannot be negative); ``current_mid`` is
    the SAME contract's mid in today's regenerated chain via the SHARED
    helper (routers/options.py) — an honest ``None`` when the contract is
    missing from the chain (e.g. expired), which makes the §11.3 premium
    stop report "insufficient data" loudly rather than pretending (§44
    rule 18). The same honest ``None`` covers "no market data provider is
    configured": ``dte`` is arithmetic on a stored expiry date and stays
    real, but no mid can be known without a chain.
    """
    dte = _dte_remaining(position)
    chain = option_chain_or_none(position.ticker, spot)
    if is_spread_position(position):
        # Roadmap Phase 1: the spread's live mark is the NET mid — both legs
        # must be quoted; either one missing -> honest None (the §11.3 net
        # premium stop then reports "insufficient data").
        if chain is None:
            return dte, None
        long_leg = find_option_contract(chain, position)
        short_leg = find_spread_short_leg(chain, position)
        if long_leg is None or short_leg is None:
            return dte, None
        return dte, max(long_leg.mid - short_leg.mid, 0.0)
    if chain is None:
        return dte, None
    contract = find_option_contract(chain, position)
    return dte, contract.mid if contract is not None else None


def _option_iv0(position: Position, spot: float) -> float | None:
    """The PROVIDER's implied vol for this position's contract, or ``None``.

    Phase D §8.5 (spec §52 "Volatility sensitivity"): the same ``iv0`` the
    stress engine anchors this position's revaluation on, read through the
    SAME shared chain helper — so the number a user sees next to a position
    is the number its scenario loss was computed with.

    Honest nulls: no chain (no provider), a contract missing from today's
    chain, or a feed that omits IV on this strike ⇒ ``None``, never a solved
    or guessed vol. A spread reports its LONG leg's IV (a two-leg position
    has no single IV; the long leg is the one whose premium is at risk) and
    ``None`` when that leg is unquoted.

    PROVENANCE: this is the vendor's IV, passed through unchanged — never an
    internally solved one (data-source-architecture.md §12; the internally
    calculated solver is labelled where it is used).
    """
    chain = option_chain_or_none(position.ticker, spot)
    if chain is None:
        return None
    contract = find_option_contract(chain, position)
    return contract.iv if contract is not None else None


def _evaluate_open_position(
    position: Position,
    bars: list[StockBarDaily],
    option_read: tuple[int | None, float | None] | None = None,
) -> tuple[ExitDecision | None, str | None]:
    """Run the shared exit engine for one OPEN position.

    Stock rows go through :func:`evaluate_exit` unchanged; option rows
    through :func:`evaluate_option_exit` with ``option_read`` — the
    ``(dte, current_mid)`` pair from :func:`_option_live_read` (computed by
    the caller so the chain is built once per row). Returns ``(decision,
    None)`` when evaluable, else ``(None, reason)`` — an honest explanation
    of WHY no evaluation was possible (§44 rule 18): no stored bars, a
    legacy stock row without a positive ``stop_distance``, or an option row
    without a positive entry premium.
    """
    if not bars:
        return None, (
            f"no stored bars for {position.ticker} — exit rules cannot be "
            "evaluated (DATA_ISSUE)"
        )
    # Spreads ride the OPTION exit semantics on NET values (net entry debit
    # in avg_price, net mid from the live read) — §11.3/§11.7 transfer
    # verbatim because the net debit IS the premium at risk.
    is_option = is_option_position(position) or is_spread_position(position)
    if is_option and position.avg_price <= 0.0:
        return None, (
            "option position has no entry premium recorded — exit rules "
            "cannot be evaluated (DATA_ISSUE)"
        )
    if not is_option and position.stop_distance <= 0.0:
        return None, (
            "position has no stop_distance recorded — exit rules cannot be "
            "evaluated (DATA_ISSUE)"
        )
    entry = (
        date.fromisoformat(position.entry_bar_date)
        if position.entry_bar_date is not None
        else None
    )
    closes_since_entry = [b.close for b in bars if entry is None or b.ts >= entry]
    # Fold the current close into the peak BEFORE evaluating, exactly as the
    # backtest engine does (see PositionState docs).
    peak = max(closes_since_entry) if closes_since_entry else bars[-1].close
    # PositionState tracks the UNDERLYING (see its docs): for an option row
    # entry_price is the underlying close on the entry bar — avg_price is
    # the PREMIUM and belongs to OptionState.entry_premium instead.
    if is_option:
        underlying_entry = next(
            (b.close for b in bars if b.ts.isoformat() == position.entry_bar_date),
            closes_since_entry[0] if closes_since_entry else bars[-1].close,
        )
    else:
        underlying_entry = position.avg_price
    # Direction mirror (2026-08-17 bug fix): a LONG_PUT / BEAR_PUT_SPREAD
    # profits when the underlying FALLS — its signal exits fire on the
    # OPPOSING (BULL) bias, its decay on the bear-favorable edge, and its
    # ATR trail hangs ABOVE the running trough (the short-side standard).
    # Before this, every winning put was SIGNAL_FLIP'd on the first sweep.
    bear_position = position.instrument in (
        InstrumentType.LONG_PUT.value,
        InstrumentType.BEAR_PUT_SPREAD.value,
        # Phase 3: a stock short profits when the underlying falls; the
        # shared engine mirrors its hard stop ABOVE entry.
        InstrumentType.SHORT_STOCK.value,
    )
    trough = min(closes_since_entry) if closes_since_entry else bars[-1].close
    state = PositionState(
        entry_price=underlying_entry,
        stop_distance=position.stop_distance,  # ignored by the option engine
        entry_edge=position.entry_edge,
        bars_held=_bars_held(position, bars),
        highest_close_since_entry=max(peak, bars[-1].close),
        direction="BEAR" if bear_position else "BULL",
        lowest_close_since_entry=(
            min(trough, bars[-1].close) if bear_position else None
        ),
    )
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]
    if is_option:
        dte, current_mid = option_read if option_read is not None else (None, None)
        decision = evaluate_option_exit(
            state,
            OptionState(
                entry_premium=position.avg_price,
                current_mid=current_mid,
                dte=dte,
            ),
            closes,
            highs,
            lows,
            volumes=volumes,
        )
    else:
        decision = evaluate_exit(state, closes, highs, lows, volumes=volumes)
    return decision, None


@router.get("")
async def list_positions(
    status: str = Query(default="OPEN"),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List positions with the full exit-engine read per OPEN row (§37).

    ``status`` filters OPEN (default) | CLOSED | ALL. Read-only — no audit.
    Every OPEN row's ``exit_reasons`` lists EVERY §11 rule with real numbers,
    "OK:"-prefixed when not firing (§37/§38) — including the option families
    (§11.3 premium hard stop, §11.7 DTE exit) for option rows; CLOSED rows
    carry ``realized_pnl`` and honest nulls elsewhere (§44 rule 18).

    Option rows: ``quantity`` = contracts, ``avg_price`` = entry premium per
    share, ``market_value`` = qty * current_mid * 100, ``max_loss`` = premium
    paid; ``current_price`` stays the UNDERLYING's last stored close (the
    underlying series drives the shared exit rules) while the option's own
    quote lives in ``contract.current_mid``. ``stop_price`` on an option row
    is the §11.3 PREMIUM stop per share (from the exit engine), not an
    underlying level.

    NOT 503 when market data is unconfigured — the deliberate difference from
    every other market-facing endpoint. A position is a REAL row: the user
    holds it, and its quantity, entry price, open date and realized PnL are
    facts the DATABASE owns, not market data. Hiding them because no quote is
    available would be its own dishonesty. The row is therefore still listed
    and only the MARKET-DERIVED fields become honest nulls —
    ``current_price``, ``market_value``, ``unrealized_pnl``,
    ``unrealized_pnl_pct``, ``trail_price``, ``current_edge``, the option
    ``current_mid`` and the whole exit-engine read — with the gap named in
    ``exit_reasons``. Endpoints whose entire substance IS market data
    (overview, analysis, bars, options, order preview/approve/close,
    backtests, check-exits) answer 503 instead.
    """
    status = status.upper()
    if status not in VALID_STATUS:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {', '.join(VALID_STATUS)}",
        )

    stmt = select(Position).order_by(Position.ticker, Position.id)
    if status != "ALL":
        stmt = stmt.where(Position.status == status)
    positions = (await session.execute(stmt)).scalars().all()

    # Checked ONCE per request: with no provider the live read is skipped for
    # every OPEN row (see the docstring — real rows, null market fields).
    have_market_data = market_data_configured()

    # Phase D (spec §52): the worst persisted scenario's per-leg P&L, read
    # ONCE for the whole list. Empty when no snapshot has been built yet —
    # every row's `worst_scenario_pnl` is then an honest null, never a 0.
    worst_per_position, worst_scenario_name = await latest_worst_scenario_per_position(
        session
    )

    out: list[dict] = []
    for pos in positions:
        is_option = is_option_position(pos)
        contract_out = None
        if pos.opt_expiry is not None:
            # Server-built OCC symbol (§27) — same string the broker is
            # addressed with; null when the stored fields cannot build one.
            try:
                option_symbol = occ_option_symbol(
                    pos.ticker,
                    date.fromisoformat(pos.opt_expiry),
                    float(pos.opt_strike),
                    pos.opt_right or "",
                )
            except (TypeError, ValueError):
                option_symbol = None
            contract_out = {
                # Spread rows (Phase 1): the SHORT leg's identity; honest
                # nulls for single-leg options.
                "short_symbol": pos.short_occ_symbol,
                "short_strike": pos.short_strike,
                "option_symbol": option_symbol,
                "expiry": pos.opt_expiry,
                "strike": pos.opt_strike,
                "right": pos.opt_right,
                "multiplier": pos.multiplier or 1,
                # Live fields; filled below for OPEN rows (honest nulls on
                # CLOSED rows — no live read, §44 rule 18).
                "dte": None,
                "current_mid": None,
                "premium_pnl_pct": None,
            }
        row = {
            "id": pos.id,
            "ticker": pos.ticker,
            "instrument": pos.instrument,
            "status": pos.status,
            "quantity": pos.quantity,
            "avg_price": pos.avg_price,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
            "current_price": None,
            "market_value": None,
            "unrealized_pnl": None,
            "unrealized_pnl_pct": None,
            "realized_pnl": pos.realized_pnl,
            "max_loss": pos.max_loss,
            # Stock: the fixed §11.3 underlying stop — ABOVE entry for a
            # Phase 3 short. Options: null here — the premium stop is
            # filled from the exit-engine read below.
            "stop_price": (
                (
                    pos.avg_price + pos.stop_distance
                    if is_short_stock_position(pos)
                    else pos.avg_price - pos.stop_distance
                )
                if pos.stop_distance > 0 and not is_option
                else None
            ),
            "trail_price": None,
            "entry_edge": pos.entry_edge,
            "current_edge": None,
            "signal_decay": None,
            "bars_held": None,
            "time_stop_remaining": None,
            "exit_status": None,
            "exit_reasons": [],
            "contract": contract_out,
            # --- ADDITIVE (Phase D design §8.5; spec §52 "stock vs option
            # risk display"): the option-specific risk facts, so an option
            # row is never presented as if it were stock. All FOUR are null
            # on a stock row — a share has no premium at risk, no expiry and
            # no IV, and saying null is more honest than saying zero.
            #
            # `premium_at_risk` is the capital that CANNOT be recovered if
            # the position expires worthless: the premium PAID for a long
            # option (= max_loss, §12.1) and the NET DEBIT for a spread. It
            # is a DB fact, available with or without market data.
            # `dte` mirrors contract.dte (calendar days, Eastern clock) at
            # the top level so §52's option panel reads one object.
            # `iv0` is the PROVIDER's IV for this contract — the anchor the
            # stress reprice used. `worst_scenario_pnl` is this position's
            # own P&L (gain-positive; a loss is negative) under the worst
            # scenario of the newest persisted snapshot, with the scenario
            # NAMED so the number is never an anonymous figure.
            "premium_at_risk": (
                pos.quantity * pos.avg_price * (pos.multiplier or 1)
                if (is_option or is_spread_position(pos)) and pos.avg_price > 0
                else None
            ),
            # Calendar days to expiry on the EASTERN clock — the same one
            # every options surface uses (see _option_live_read). Filled for
            # spreads too, which never enter the single-leg live read.
            "dte": _dte_remaining(pos),
            "iv0": None,
            "worst_scenario_pnl": worst_scenario_pnl_for_key(
                worst_per_position, f"{pos.ticker}#{pos.id}"
            ),
            "worst_scenario_name": worst_scenario_name,
        }
        if pos.status == POSITION_OPEN and not have_market_data:
            # No provider configured: the row's DB facts above stand, every
            # market-derived field stays null, and the user is told why
            # instead of being shown a number nobody can vouch for.
            row["stop_price"] = None
            row["exit_reasons"] = [NO_MARKET_DATA_REASON]
        elif pos.status == POSITION_OPEN:
            bars = await _stored_bars(session, pos.ticker)
            option_read = None
            if bars:
                price = bars[-1].close
                row["current_price"] = price
                row["bars_held"] = _bars_held(pos, bars)
                if is_option:
                    # Live option read via the shared chain helper (§9);
                    # market value carries the x100 multiplier (§12.1) and is
                    # an honest null when the contract has no current mid.
                    option_read = _option_live_read(pos, price)
                    dte, current_mid = option_read
                    mult = pos.multiplier or 1
                    if contract_out is not None:
                        contract_out["dte"] = dte
                        contract_out["current_mid"] = current_mid
                    # Phase D §8.5 additive: the provider IV the stress
                    # reprice anchored on. (`dte` is filled for EVERY
                    # option-bearing row below, spreads included.)
                    row["iv0"] = _option_iv0(pos, price)
                    if current_mid is not None:
                        row["market_value"] = pos.quantity * current_mid * mult
                        row["unrealized_pnl"] = (
                            (current_mid - pos.avg_price) * pos.quantity * mult
                        )
                        if pos.avg_price > 0:
                            pnl_pct = current_mid / pos.avg_price - 1.0
                            row["unrealized_pnl_pct"] = pnl_pct
                            if contract_out is not None:
                                contract_out["premium_pnl_pct"] = pnl_pct
                elif is_short_stock_position(pos):
                    # Phase 3: a short is a LIABILITY (negative market
                    # value, matching the portfolio view) and its P&L
                    # mirrors: entry − current.
                    row["market_value"] = -pos.quantity * price
                    row["unrealized_pnl"] = (pos.avg_price - price) * pos.quantity
                    row["unrealized_pnl_pct"] = (
                        (pos.avg_price - price) / pos.avg_price
                        if pos.avg_price
                        else None
                    )
                else:
                    row["market_value"] = pos.quantity * price
                    row["unrealized_pnl"] = (price - pos.avg_price) * pos.quantity
                    row["unrealized_pnl_pct"] = (
                        (price - pos.avg_price) / pos.avg_price
                        if pos.avg_price
                        else None
                    )
            decision, why_not = _evaluate_open_position(pos, bars, option_read)
            if decision is not None:
                row["stop_price"] = decision.stop_price
                row["trail_price"] = decision.trail_price
                row["current_edge"] = decision.current_edge
                row["signal_decay"] = (
                    pos.entry_edge - decision.current_edge
                    if decision.current_edge is not None
                    else None
                )
                row["time_stop_remaining"] = decision.time_stop_remaining
                row["exit_status"] = (
                    "EXIT_SIGNALED" if decision.should_exit else "HOLD"
                )
                row["exit_reasons"] = decision.reasons
            else:
                # §37: even when unevaluable, the user must see WHY.
                row["exit_reasons"] = [why_not]
        out.append(row)
    return out


@router.post("/check-exits")
async def check_exits(session: AsyncSession = Depends(get_session)) -> dict:
    """Evaluate + EXECUTE §11 exits for every OPEN position — one transaction.

    Covers BOTH instrument kinds: stock rows via ``evaluate_exit``, option
    rows via ``evaluate_option_exit`` (§11.3 premium hard stop / §11.7 DTE
    exit in front of the shared underlying rules). Each triggered exit
    writes EXIT_GENERATED (SYSTEM, with the rule and the full reason list)
    and runs the shared sell-to-close flow (``system_generated`` ->
    ORDER_REQUESTED actor SYSTEM) in the SAME transaction (rule 12) —
    option exits fill at the current chain mid, or at intrinsic value when
    the contract is missing from today's chain (documented in
    routers/orders.py). Positions that hold answer with their full
    "OK:"-prefixed reason list (§37). Mechanical exits are deliberately NOT
    blocked by the §18 kill switch: an exit reduces risk, and risk protection
    outranks the pause (§18 risk-priority). Runs under the paper-execution
    lock shared with approve/close, so a concurrent manual close can never
    double-sell the same position (§42 analogue).

    The whole flow lives in :func:`run_exit_sweep`, shared verbatim with the
    background position monitor (apps/gateway/monitor.py) — one sweep
    implementation, two triggers (plan §21 spirit: never reimplement).

    503 ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is
    configured: every exit rule is a comparison against a current price, so a
    sweep without market data could only ever act on invented numbers. Refusing
    to sweep is the safe answer — it changes nothing (§44 rule 18).

    With no BROKER configured the sweep answers 200 with ``"skipped":
    "BROKER_NOT_CONFIGURED"`` and changes nothing. It is a 200 rather than a
    503 on purpose: unlike the market-data case, the caller asked "is anything
    ready to exit, and if so exit it" and the honest answer is "nothing was
    swept, here is why" — the same shape the background monitor reports. What
    matters is what it does NOT do: close positions locally that it cannot sell
    at the broker.
    """
    require_market_data_provider()
    return await run_exit_sweep(session)


@router.get("/monitor")
async def monitor_status() -> dict:
    """Status of the automated position monitor (plan §26, §44 rule 18).

    ``enabled`` is honest: True only when the configured interval is > 0 AND
    the background task actually started (lifespan ran). Under test
    transports (httpx ASGITransport) lifespan never runs, so this reports
    ``enabled: false`` with null sweep fields rather than pretending a
    monitor is running.
    """
    # Local import: monitor.py imports run_exit_sweep from this module, so a
    # module-level import here would be circular.
    from .. import monitor

    interval = get_settings().position_monitor_interval_seconds
    state = monitor.STATE
    return {
        "enabled": interval > 0 and state.enabled,
        "interval_seconds": interval,
        "last_sweep_at": (
            state.last_sweep_at.isoformat() if state.last_sweep_at else None
        ),
        "sweeps_total": state.sweeps_total,
        "last_result": state.last_result,
    }


async def run_exit_sweep(session: AsyncSession) -> dict:
    """One full exit sweep: evaluate + execute §11 exits, then COMMIT.

    The single sweep implementation shared by POST /check-exits and the
    background monitor loop (apps/gateway/monitor.py). Takes the
    paper-execution lock (never double-sells against a concurrent manual
    close/approve) and commits the transaction itself — triggered exits,
    their EXIT_GENERATED + ORDER_* audit events and the position/cash
    mutations land atomically (rule 12). Returns
    ``{"checked": int, "exits_triggered": [...], "held": [...]}``, plus
    ``"exits_failed": [...]`` when the broker refused or could not execute a
    signalled exit — one position's failure never aborts the sweep, because
    the other positions still deserve their protective exits.

    SKIPS ENTIRELY when no execution venue is configured: no evaluation, no
    position change, one WARNING, and ``"skipped": "BROKER_NOT_CONFIGURED"``
    added to the result. An exit sweep exists to SELL; with nowhere to send
    the sell, "closing" a position would only move a local row and leave the
    broker holding the real thing (§18 reconciliation).
    """
    if not broker_configured():
        reason = broker_unavailable_reason()
        logger.warning(
            "exit_sweep_skipped_no_broker",
            extra={"extra_fields": {"reason": reason}},
        )
        return {
            "checked": 0,
            "exits_triggered": [],
            "held": [],
            "skipped": BROKER_NOT_CONFIGURED,
            "reason": reason,
        }
    async with execution_lock():
        return await _run_exit_sweep_locked(session)


async def _run_exit_sweep_locked(session: AsyncSession) -> dict:
    """The exit-sweep flow proper — caller holds the paper-execution lock."""
    positions = (
        (
            await session.execute(
                select(Position)
                .where(Position.status == POSITION_OPEN)
                .order_by(Position.ticker, Position.id)
            )
        )
        .scalars()
        .all()
    )

    exits_triggered: list[dict] = []
    held: list[dict] = []
    # Exits the engine SIGNALLED but the broker did not execute (refused,
    # unreachable, or filled nothing). Reported, never hidden: the position is
    # still open at the broker AND locally, which is the honest state, but the
    # user must be told the protective exit did not actually happen (§37).
    failed: list[dict] = []
    for pos in positions:
        bars = await _stored_bars(session, pos.ticker)

        # Phase 2 — income rows (covered call / CSP): mechanical
        # short-premium management, then BUYBACK via the income endpoint's
        # own logic is the human's; the sweep only SURFACES the verdict for
        # now (auto-buyback lands with the Phase 2 unlock). Loud, not
        # silent: the decision and reasons appear in the sweep output.
        if pos.instrument in (
            InstrumentType.COVERED_CALL.value,
            InstrumentType.CASH_SECURED_PUT.value,
        ):
            spot = bars[-1].close if bars else None
            mid = None
            dte = None
            if spot is not None:
                dte, mid = _option_live_read(pos, spot)
            decision = evaluate_short_premium_exit(
                ShortPremiumState(
                    entry_credit=pos.avg_price,
                    current_mid=mid,
                    dte=dte,
                    strike=pos.opt_strike or 0.0,
                    spot=spot,
                    right=pos.opt_right or "C",
                )
            )
            if decision.should_exit:
                await audit.record(
                    session,
                    actor_type=ActorType.SYSTEM,
                    action=AuditAction.EXIT_GENERATED,
                    entity_type="position",
                    entity_id=str(pos.id),
                    details={
                        "ticker": pos.ticker,
                        "instrument": pos.instrument,
                        "rule": decision.triggered_rule,
                        "reasons": decision.reasons,
                        "action_required": (
                            "BUY BACK via POST /api/income/"
                            f"{pos.id}/buyback — auto-buyback arrives with "
                            "the Phase 2 unlock"
                        ),
                    },
                )
                failed.append(
                    {
                        "ticker": pos.ticker,
                        "rule": decision.triggered_rule,
                        "error": (
                            "income position needs BUYBACK "
                            f"(POST /api/income/{pos.id}/buyback)"
                        ),
                    }
                )
            else:
                held.append({"ticker": pos.ticker, "reasons": decision.reasons})
            continue

        # Phase 2 collateral law: a stock row pinned under open covered
        # calls cannot be auto-sold — HOLD loudly instead.
        if pos.instrument == InstrumentType.LONG_STOCK.value:
            pinned = (
                (
                    await session.execute(
                        select(
                            func.coalesce(func.sum(Position.quantity), 0)
                        ).where(
                            Position.collateral_position_id == pos.id,
                            Position.status == POSITION_OPEN,
                            Position.instrument
                            == InstrumentType.COVERED_CALL.value,
                        )
                    )
                ).scalar_one()
                * 100
            )
            if pinned > 0:
                held.append(
                    {
                        "ticker": pos.ticker,
                        "reasons": [
                            f"HELD: {pinned} share(s) pinned as covered-call "
                            "collateral — the exit sweep will not sell them; "
                            "buy back the covered call first (risk note: the "
                            "stock exit signal is being deferred)."
                        ],
                    }
                )
                continue

        option_read = None
        if (is_option_position(pos) or is_spread_position(pos)) and bars:
            option_read = _option_live_read(pos, bars[-1].close)
        decision, why_not = _evaluate_open_position(pos, bars, option_read)
        if decision is None:
            # Unevaluable is NOT an exit — but the gap is surfaced (§44 r18).
            held.append({"ticker": pos.ticker, "reasons": [why_not]})
            continue
        if not decision.should_exit:
            held.append({"ticker": pos.ticker, "reasons": decision.reasons})
            continue

        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.EXIT_GENERATED,
            entity_type="position",
            entity_id=str(pos.id),
            details={
                "ticker": pos.ticker,
                "instrument": pos.instrument,
                "rule": decision.triggered_rule,
                "reasons": decision.reasons,
            },
        )
        if is_spread_position(pos):
            # Same reference logic as /close for spreads: live NET mid, or
            # the documented NET-intrinsic fallback (bounded at >= 0).
            _dte, current_net = option_read if option_read is not None else (None, None)
            if current_net is not None:
                reference, source = current_net, "chain net mid (long - short)"
            else:
                spot_ref = bars[-1].close
                long_intr = option_intrinsic_value(pos, spot_ref)
                short_strike = pos.short_strike or 0.0
                if (pos.opt_right or "C") == "C":
                    short_intr = max(spot_ref - short_strike, 0.0)
                else:
                    short_intr = max(short_strike - spot_ref, 0.0)
                reference = max(long_intr - short_intr, 0.0)
                source = "net intrinsic (leg(s) missing from today's chain)"
        elif is_option_position(pos):
            # Same reference logic as /close: current chain mid, or the
            # documented intrinsic fallback when the contract is gone.
            _dte, current_mid = option_read if option_read is not None else (None, None)
            if current_mid is not None:
                reference, source = current_mid, "chain mid"
            else:
                reference = option_intrinsic_value(pos, bars[-1].close)
                source = "intrinsic (contract missing from today's chain)"
        else:
            reference, source = bars[-1].close, "last stored close"
        try:
            order, _realized = await execute_sell_to_close(
                session,
                pos,
                pos.quantity,  # mechanical exits always close in full (§11)
                reference,
                reason=f"exit engine: {decision.triggered_rule}",
                system_generated=True,
                reference_source=source,
            )
        except HTTPException as exc:
            # One position's exit failed at the broker (a rejection, a fault,
            # or an option the broker cannot trade). The sweep continues: the
            # OTHER positions still deserve their protective exits, and
            # aborting here would lose them all. The audit trail already
            # carries the broker's own reason (written by the sell path).
            detail = exc.detail
            message = (
                detail.get("message") if isinstance(detail, dict) else str(detail)
            )
            logger.warning(
                "exit_execution_failed",
                extra={
                    "extra_fields": {
                        "ticker": pos.ticker,
                        "rule": decision.triggered_rule,
                        "status_code": exc.status_code,
                        "reason": message,
                    }
                },
            )
            failed.append(
                {
                    "ticker": pos.ticker,
                    "rule": decision.triggered_rule,
                    "reason": message,
                }
            )
            continue
        exits_triggered.append(
            {
                "ticker": pos.ticker,
                "rule": decision.triggered_rule,
                "order_id": order.id,
            }
        )

    await session.commit()
    result = {
        "checked": len(positions),
        "exits_triggered": exits_triggered,
        "held": held,
    }
    if failed:
        # Only present when something went wrong, so the common shape is
        # unchanged — but never suppressed when it did.
        result["exits_failed"] = failed
    return result
