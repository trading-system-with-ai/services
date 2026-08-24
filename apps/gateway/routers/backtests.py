"""Backtest API (development plan §20).

Runs Backtest Engine V1 (libs.trading_core.backtest) over a Watchlist
symbol's stored daily bars and persists the full result. Only Watchlist
symbols may be backtested — historical data exists only for them (plan §20,
§4.2) — so other tickers 404.

The engine is imported from libs.trading_core.backtest, which itself imports
its signals EXCLUSIVELY from libs.trading_core.signals, so backtest and live
run the exact same signal code (plan §21 — MANDATORY; nothing is
reimplemented here).

V1 EXECUTION MODEL — SYNCHRONOUS, NO QUEUE: POST runs the engine in-request
and returns the finished record. At V1 scale (one user, ~600 stored daily
bars per symbol) a run completes in well under a second, so a job queue and
polling endpoint are deliberately deferred; when longer histories or
parameter sweeps arrive, this endpoint becomes an enqueue.

Every run writes its audit trail in the SAME transaction as the persisted
record (rule 12): a USER-attributed BACKTEST_STARTED (with the resolved
params) plus a SYSTEM-attributed BACKTEST_COMPLETED (headline metrics) or
BACKTEST_FAILED (error message). Every threshold the engine uses is a
BacktestParams parameter (plan §6.2); invalid values 422 with the engine's
own ValueError message.
"""
import asyncio
import dataclasses

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import get_provider
from libs.trading_core.backtest.advice import assess_portfolio_result
from libs.trading_core.backtest import (
    SymbolBars,
    run_auto_backtest,
    run_portfolio_backtest,
    BacktestParams,
    BacktestResult,
    OptionLegBars,
    OptionTrade,
    SpreadLegBars,
    run_backtest,
    run_call_backtest,
    run_short_stock_backtest,
    run_covered_call_backtest,
    run_csp_backtest,
    run_spread_backtest,
)
from libs.trading_core.models import ActorType, AuditAction
from libs.trading_core.strategies import AccountPermissions

from .. import audit
from ..db import (
    AtmIvDailyRow,
    BacktestRecord,
    PortfolioBacktestRecord,
    WatchlistItem,
    get_session,
)
from ..deps import account_permissions_from_settings, require_market_data_provider
from ..schemas import TICKER_RE
from .analysis import ensure_daily_bars
from .watchlist import CURRENT_USER

router = APIRouter(prefix="/api/backtests", tags=["backtests"])

# Record status values (backtests.status column).
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

# Valid BacktestParams field names — request params must be a subset (plan §20).
_PARAM_FIELDS = frozenset(f.name for f in dataclasses.fields(BacktestParams))

# Tunable list-endpoint defaults (plan §6.2: parameters, never hardcoded truths).
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 500


class BacktestRequest(BaseModel):
    """POST /api/backtests body: ticker plus an optional subset of
    BacktestParams overrides (plan §20; every threshold a parameter, §6.2)."""

    ticker: str
    params: dict = Field(default_factory=dict)
    # AUTO runs only (Phase B, docs/auto-strategy-portfolio-design.md): the
    # user's instrument multi-select. None = every AUTO-supported instrument
    # the account permissions allow. Explicitly selecting an instrument the
    # account disallows is a 422 — the selection may RESTRICT, never exceed.
    instruments: list[str] | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.match(v):
            raise ValueError(f"invalid ticker: {v!r}")
        return v


def _flat_metrics(stored: dict) -> dict:
    """Present stored metrics in the flat full-period contract shape.

    IS/OOS segmentation was removed 2026-08-16 (user decision: manual-only
    tuning needs no split until ML-driven search exists). New rows store the
    flat dict; rows persisted before then stored {"full", "in_sample",
    "out_of_sample"} — their DB payload is untouched (records are never
    rewritten), and this presents their full-period segment.
    """
    if isinstance(stored, dict) and "full" in stored:
        return stored["full"]
    return stored


def _record_json(rec: BacktestRecord) -> dict:
    """Full backtest record in the API contract shape (plan §20)."""
    return {
        "id": rec.id,
        "ticker": rec.ticker,
        "created_at": rec.created_at.isoformat(),
        "status": rec.status,
        "params": rec.params,
        "error": rec.error or None,
        "metrics": _flat_metrics(rec.metrics),
        "trades": rec.trades,
        "equity_curve": rec.equity_curve,
    }


def _summary_json(rec: BacktestRecord) -> dict:
    """List-endpoint summary: identity plus headline full-segment metrics.

    ``fill_model`` (plan §20.2) is surfaced from the stored resolved params
    so history rows can chip the model without fetching the full record;
    records persisted before fill models existed report ``None``.
    """
    full = _flat_metrics(rec.metrics or {})
    return {
        "id": rec.id,
        "ticker": rec.ticker,
        "created_at": rec.created_at.isoformat(),
        "status": rec.status,
        "num_trades": full.get("num_trades"),
        "total_return_pct": full.get("total_return_pct"),
        "profit_factor": full.get("profit_factor"),
        "fill_model": (rec.params or {}).get("fill_model"),
        # "LONG_STOCK" | "LONG_CALL"; None for records before the leg existed.
        "instrument": (rec.params or {}).get("instrument"),
    }


def _result_payloads(result: BacktestResult) -> tuple[dict, list, dict]:
    """Serialize a BacktestResult into the JSON payloads the record stores.

    Exactly the API contract shapes (plan §20); the engine guarantees nulls
    where a metric is undefined — never NaN/Infinity (plan §44 rule 18).
    """
    metrics = dataclasses.asdict(result.metrics)
    trades = []
    for t in result.trades:
        row = {
            "entry_date": t.entry_date.isoformat(),
            "entry_price": t.entry_price,
            "exit_date": t.exit_date.isoformat() if t.exit_date is not None else None,
            "exit_price": t.exit_price,
            "bars_held": t.bars_held,
            "return_pct": t.return_pct,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
        }
        if isinstance(t, OptionTrade):
            # Option-leg extras: prices above are premiums per share (NET
            # debit per share for spread trades).
            row["contracts"] = t.contracts
            row["contract_symbol"] = t.contract_symbol
            row["strike"] = t.strike
            row["contract_expiry"] = (
                t.contract_expiry.isoformat() if t.contract_expiry else None
            )
            if t.short_symbol:
                row["short_symbol"] = t.short_symbol
                row["short_strike"] = t.short_strike
        trades.append(row)
    equity_curve = {
        "dates": [d.isoformat() for d in result.dates],
        "equity": result.equity,
        "drawdown": result.drawdown,
    }
    return metrics, trades, equity_curve


def _make_leg_resolver(ticker: str, params, contracts_fn, bars_fn, *, bear: bool):
    """One single-leg contract resolver (call or put) — the same moneyness/
    DTE selection the LONG_CALL/LONG_PUT runs use, shared by the AUTO and
    portfolio runners."""
    target_dte = (params.target_dte_min + params.target_dte_max) // 2

    def resolver(decision_date, spot):
        rows = contracts_fn(
            ticker,
            decision_date,
            params.target_dte_min,
            params.target_dte_max,
            spot,
            contract_type="put" if bear else "call",
        )
        if not rows:
            return None
        target_strike = spot * (
            (1.0 - params.strike_otm_pct) if bear else (1.0 + params.strike_otm_pct)
        )
        pick = min(
            rows,
            key=lambda r: (
                abs((r["expiration_date"] - decision_date).days - target_dte),
                abs(r["strike_price"] - target_strike),
            ),
        )
        leg_bars = bars_fn(pick["ticker"], decision_date, pick["expiration_date"])
        if not leg_bars:
            return None
        return OptionLegBars(
            symbol=pick["ticker"],
            strike=pick["strike_price"],
            expiry=pick["expiration_date"],
            bars=leg_bars,
        )

    return resolver


async def _run_auto_backtest_with_provider(
    ticker: str,
    bars: list,
    params,
    provider_name: str,
    permissions,
    session,
):
    """Run the AUTO engine (Phase B, docs/auto-strategy-portfolio-design.md).

    Builds BOTH single-leg contract resolvers (call + put) with the same
    moneyness/DTE selection the LONG_CALL/LONG_PUT runs use. A provider
    without historical-options capability does NOT fail the run — AUTO can
    still trade stock; option decisions then skip their entries and the
    stored auto_decisions trail shows exactly what was skipped (§33: no
    silent substitution — the skip is visible, never faked).

    iv_series: REAL stored ATM IV (atm_iv_daily, internally calculated from
    provider chains) aligned to the bar dates; None where no row exists —
    the engine then coerces vol to NORMAL exactly like the live chain.
    """
    provider = get_provider(provider_name)
    contracts_fn = getattr(provider, "get_option_contracts_window", None)
    bars_fn = getattr(provider, "get_option_daily_bars", None)
    have_options = contracts_fn is not None and bars_fn is not None
    call_resolver = (
        _make_leg_resolver(ticker, params, contracts_fn, bars_fn, bear=False)
        if have_options and permissions.long_call
        else None
    )
    put_resolver = (
        _make_leg_resolver(ticker, params, contracts_fn, bars_fn, bear=True)
        if have_options and permissions.long_put
        else None
    )

    # REAL stored ATM IV only (atm_iv_daily accumulates since 2026-08-18) —
    # sparse history simply means NORMAL-coerced vol on the uncovered days,
    # byte-identical to the live chain's no-data behavior.
    iv_rows = await session.execute(
        select(AtmIvDailyRow).where(AtmIvDailyRow.ticker == ticker)
    )
    iv_by_date = {r.bar_date: r.atm_iv for r in iv_rows.scalars().all()}
    iv_series = [iv_by_date.get(b.ts) for b in bars]

    return await asyncio.to_thread(
        run_auto_backtest,
        [b.ts for b in bars],
        [b.open for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
        [b.close for b in bars],
        [b.volume for b in bars],
        params,
        permissions=permissions,
        call_provider=call_resolver,
        put_provider=put_resolver,
        iv_series=iv_series,
    )


async def _run_call_backtest_with_provider(
    ticker: str, bars: list, params: BacktestParams, provider_name: str
) -> BacktestResult:
    """Run the LONG_CALL engine with contracts/bars resolved from the REAL
    provider (user mandate 2026-08-17; data-source-architecture.md).

    The resolver is a sync callback doing network I/O, so the whole replay
    runs in a worker thread — the event loop is never blocked. A provider
    without the historical-options methods (e.g. Massive today) gets an
    honest 422 naming the capability, never a silent stock fallback (§33).
    """
    provider = get_provider(provider_name)
    contracts_fn = getattr(provider, "get_option_contracts_window", None)
    bars_fn = getattr(provider, "get_option_daily_bars", None)
    if contracts_fn is None or bars_fn is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"market data provider {provider_name!r} does not serve "
                "historical option contracts/bars — the LONG_CALL backtest "
                "needs both (no cross-provider fallback, §33)."
            ),
        )

    target_dte = (params.target_dte_min + params.target_dte_max) // 2
    bear = params.instrument == "LONG_PUT"

    def resolver(decision_date, spot):
        rows = contracts_fn(
            ticker,
            decision_date,
            params.target_dte_min,
            params.target_dte_max,
            spot,
            contract_type="put" if bear else "call",
        )
        if not rows:
            return None
        # OTM is BELOW spot for puts, above for calls.
        target_strike = spot * (
            (1.0 - params.strike_otm_pct) if bear else (1.0 + params.strike_otm_pct)
        )
        pick = min(
            rows,
            key=lambda r: (
                abs((r["expiration_date"] - decision_date).days - target_dte),
                abs(r["strike_price"] - target_strike),
            ),
        )
        leg_bars = bars_fn(
            pick["ticker"], decision_date, pick["expiration_date"]
        )
        if not leg_bars:
            return None  # no real data for the pick -> entry skipped
        return OptionLegBars(
            symbol=pick["ticker"],
            strike=pick["strike_price"],
            expiry=pick["expiration_date"],
            bars=leg_bars,
        )

    return await asyncio.to_thread(
        run_call_backtest,
        [b.ts for b in bars],
        [b.open for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
        [b.close for b in bars],
        [b.volume for b in bars],
        params,
        contract_provider=resolver,
    )


async def _run_spread_backtest_with_provider(
    ticker: str, bars: list, params: BacktestParams, provider_name: str
) -> BacktestResult:
    """Run the BULL_CALL_SPREAD engine with both legs resolved from the REAL
    provider (roadmap Phase 1). Long leg: nearest to strike_otm_pct of spot
    in the DTE window (as the LONG_CALL leg); short leg: nearest REAL strike
    to long + spread_width_pct×spot in the SAME expiry. Historical greeks/OI
    do not exist, so selection is moneyness/width-based — deterministic over
    the real grid, stated here rather than pretended otherwise."""
    provider = get_provider(provider_name)
    contracts_fn = getattr(provider, "get_option_contracts_window", None)
    bars_fn = getattr(provider, "get_option_daily_bars", None)
    if contracts_fn is None or bars_fn is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"market data provider {provider_name!r} does not serve "
                "historical option contracts/bars — the BULL_CALL_SPREAD "
                "backtest needs both (no cross-provider fallback, §33)."
            ),
        )

    target_dte = (params.target_dte_min + params.target_dte_max) // 2
    bear = params.instrument == "BEAR_PUT_SPREAD"

    def resolver(decision_date, spot):
        rows = contracts_fn(
            ticker,
            decision_date,
            params.target_dte_min,
            params.target_dte_max,
            spot,
            contract_type="put" if bear else "call",
        )
        if not rows:
            return None
        target_strike = spot * (
            (1.0 - params.strike_otm_pct) if bear else (1.0 + params.strike_otm_pct)
        )
        long_pick = min(
            rows,
            key=lambda r: (
                abs((r["expiration_date"] - decision_date).days - target_dte),
                abs(r["strike_price"] - target_strike),
            ),
        )
        # Short leg OTM-ward: BELOW the long for puts, ABOVE for calls.
        offset = params.spread_width_pct * spot
        short_target = long_pick["strike_price"] + (-offset if bear else offset)
        same_expiry_otm = [
            r
            for r in rows
            if r["expiration_date"] == long_pick["expiration_date"]
            and (
                r["strike_price"] < long_pick["strike_price"]
                if bear
                else r["strike_price"] > long_pick["strike_price"]
            )
        ]
        if not same_expiry_otm:
            return None
        short_pick = min(
            same_expiry_otm,
            key=lambda r: (abs(r["strike_price"] - short_target), r["strike_price"]),
        )
        long_bars = bars_fn(
            long_pick["ticker"], decision_date, long_pick["expiration_date"]
        )
        short_bars = bars_fn(
            short_pick["ticker"], decision_date, short_pick["expiration_date"]
        )
        if not long_bars or not short_bars:
            return None
        return SpreadLegBars(
            long=OptionLegBars(
                symbol=long_pick["ticker"],
                strike=long_pick["strike_price"],
                expiry=long_pick["expiration_date"],
                bars=long_bars,
            ),
            short=OptionLegBars(
                symbol=short_pick["ticker"],
                strike=short_pick["strike_price"],
                expiry=short_pick["expiration_date"],
                bars=short_bars,
            ),
        )

    return await asyncio.to_thread(
        run_spread_backtest,
        [b.ts for b in bars],
        [b.open for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
        [b.close for b in bars],
        [b.volume for b in bars],
        params,
        spread_provider=resolver,
    )


async def _run_income_backtest_with_provider(
    ticker: str, bars: list, params: BacktestParams, provider_name: str
) -> BacktestResult:
    """COVERED_CALL / CASH_SECURED_PUT backtests over real contract bars
    (Phase 2). Historical greeks do not exist, so the leg to SELL is chosen
    by moneyness (strike_otm_pct OTM-ward of spot: calls above, puts below)
    in the DTE window — stated, never pretended otherwise."""
    provider = get_provider(provider_name)
    contracts_fn = getattr(provider, "get_option_contracts_window", None)
    bars_fn = getattr(provider, "get_option_daily_bars", None)
    if contracts_fn is None or bars_fn is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"market data provider {provider_name!r} does not serve "
                "historical option contracts/bars — income backtests need "
                "both (no cross-provider fallback, §33)."
            ),
        )
    covered = params.instrument == "COVERED_CALL"
    target_dte = (params.target_dte_min + params.target_dte_max) // 2

    def resolver(decision_date, spot):
        rows = contracts_fn(
            ticker,
            decision_date,
            params.target_dte_min,
            params.target_dte_max,
            spot,
            contract_type="call" if covered else "put",
        )
        if not rows:
            return None
        # OTM-ward of spot: calls ABOVE, puts BELOW.
        otm = abs(params.strike_otm_pct) or 0.05  # income default: 5% OTM
        target_strike = spot * ((1.0 + otm) if covered else (1.0 - otm))
        candidates = [
            r
            for r in rows
            if (r["strike_price"] > spot if covered else r["strike_price"] < spot)
        ]
        if not candidates:
            return None
        pick = min(
            candidates,
            key=lambda r: (
                abs((r["expiration_date"] - decision_date).days - target_dte),
                abs(r["strike_price"] - target_strike),
            ),
        )
        leg_bars = bars_fn(pick["ticker"], decision_date, pick["expiration_date"])
        if not leg_bars:
            return None
        return OptionLegBars(
            symbol=pick["ticker"],
            strike=pick["strike_price"],
            expiry=pick["expiration_date"],
            bars=leg_bars,
        )

    runner = run_covered_call_backtest if covered else run_csp_backtest
    return await asyncio.to_thread(
        runner,
        [b.ts for b in bars],
        [b.open for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
        [b.close for b in bars],
        [b.volume for b in bars],
        params,
        contract_provider=resolver,
    )


def _auto_permissions(req_instruments, permissions) -> AccountPermissions:
    """Build an AUTO run's permissions: account flags ∩ the user's
    multi-select (restrict-only; explicit selection of a disabled instrument
    is a 422 by name; spreads/income arrive Phase D)."""
    supported = ("LONG_STOCK", "LONG_CALL", "LONG_PUT", "SHORT_STOCK")
    if req_instruments is not None:
        unknown_sel = sorted(set(req_instruments) - set(supported))
        if unknown_sel:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"AUTO does not support instrument(s) {unknown_sel} yet "
                    f"(Phase D); selectable now: {sorted(supported)}"
                ),
            )
        selected = set(req_instruments)
        if not selected:
            raise HTTPException(
                status_code=422,
                detail="instruments must not be empty for an AUTO run",
            )
        for name, ok, msg in (
            ("LONG_STOCK", permissions.long_stock, "allow_long_stock=false"),
            ("LONG_CALL", permissions.long_call, "allow_long_call=false"),
            ("LONG_PUT", permissions.long_put, "allow_long_put=false"),
            (
                "SHORT_STOCK",
                permissions.short_stock and permissions.margin,
                "requires BOTH allow_short_stock and allow_margin",
            ),
        ):
            if name in selected and not ok:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{name} is selected but disabled in account "
                        f"permissions ({msg}); the AUTO multi-select may "
                        "restrict, never exceed"
                    ),
                )
    else:
        selected = {
            name
            for name, ok in (
                ("LONG_STOCK", permissions.long_stock),
                ("LONG_CALL", permissions.long_call),
                ("LONG_PUT", permissions.long_put),
                ("SHORT_STOCK", permissions.short_stock and permissions.margin),
            )
            if ok
        }
    return AccountPermissions(
        long_stock="LONG_STOCK" in selected,
        long_call="LONG_CALL" in selected,
        long_put="LONG_PUT" in selected,
        defined_risk_spreads=False,  # Phase D
        short_stock="SHORT_STOCK" in selected,
        margin="SHORT_STOCK" in selected,
    )


@router.post("")
async def create_backtest(
    req: BacktestRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Run a backtest for one Watchlist symbol and persist the result (plan §20).

    Synchronous V1 (no queue — see module docstring): the response is the
    finished record. 422s (unknown param keys, invalid values with the
    engine's own message) happen before any state change or audit write.

    503 ``MARKET_DATA_NOT_CONFIGURED`` when no market data provider is
    configured. A backtest is a claim about how a strategy WOULD have
    performed; run over synthetic bars it produces a Sharpe ratio, a win rate
    and an equity curve that look exactly like evidence and are worth nothing.
    Checked before any state change or audit write.
    """
    require_market_data_provider()
    # --- Param validation first: a 422 must not write state or audit. -------
    unknown = sorted(set(req.params) - _PARAM_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown backtest param(s) {unknown}; "
                f"valid params: {sorted(_PARAM_FIELDS)}"
            ),
        )
    try:
        params = BacktestParams(**req.params)
    except (TypeError, ValueError) as exc:
        # Surface the engine's own validation message (plan §6.2).
        raise HTTPException(status_code=422, detail=str(exc))

    # --- User-level instrument permissions gate the BACKTEST too (user
    # mandate 2026-08-17): the replayed leg must be one the user's live
    # permissions would allow — display, live gating and backtest read the
    # SAME Settings factory, so they can never disagree.
    permissions = account_permissions_from_settings()
    if params.instrument == "LONG_STOCK" and not permissions.long_stock:
        raise HTTPException(
            status_code=422,
            detail=(
                "long stock is disabled in account permissions "
                "(allow_long_stock=false); re-enable it in Settings or "
                "backtest instrument='LONG_CALL' instead."
            ),
        )
    if params.instrument == "LONG_PUT" and not permissions.long_put:
        raise HTTPException(
            status_code=422,
            detail=(
                "long puts are disabled in account permissions "
                "(allow_long_put=false); re-enable them in Settings to "
                "backtest the LONG_PUT leg."
            ),
        )
    if params.instrument == "LONG_CALL" and not permissions.long_call:
        raise HTTPException(
            status_code=422,
            detail=(
                "long calls are disabled in account permissions "
                "(allow_long_call=false); re-enable them in Settings to "
                "backtest the LONG_CALL leg."
            ),
        )
    if params.instrument == "COVERED_CALL" and not permissions.covered_call:
        raise HTTPException(
            status_code=422,
            detail=(
                "covered calls are disabled in account permissions "
                "(roadmap Phase 2 — the toggle unlocks with the full chain); "
                "the COVERED_CALL backtest follows the same permission."
            ),
        )
    if params.instrument == "CASH_SECURED_PUT" and not permissions.cash_secured_put:
        raise HTTPException(
            status_code=422,
            detail=(
                "cash-secured puts are disabled in account permissions "
                "(roadmap Phase 2); the CASH_SECURED_PUT backtest follows "
                "the same permission."
            ),
        )
    if params.instrument == "SHORT_STOCK" and not (
        permissions.short_stock and permissions.margin
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "short stock requires BOTH allow_short_stock and "
                "allow_margin (roadmap Phase 3 — margin exists to support "
                "shorting); enable both in Settings to backtest the "
                "SHORT_STOCK leg."
            ),
        )
    if (
        params.instrument in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD")
        and not permissions.defined_risk_spreads
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "defined-risk spreads are disabled in account permissions "
                "(allow_defined_risk_spreads=false); enable them in Settings "
                "to backtest the spread legs (research+backtest "
                "scope — live spread execution is still under construction, "
                "roadmap Phase 1)."
            ),
        )

    # --- AUTO (Phase B): build the run's permissions from the account flags
    # ∩ the user's multi-select (restrict-only). Spreads/income legs join in
    # Phase D — selecting them under AUTO is refused, not silently dropped.
    auto_permissions = None
    if params.instrument == "AUTO":
        auto_permissions = _auto_permissions(req.instruments, permissions)
    elif req.instruments is not None:
        raise HTTPException(
            status_code=422,
            detail="instruments is only valid with instrument='AUTO'",
        )

    # --- Watchlist gate: only Watchlist symbols may be backtested (§20/§4.2).
    row = await session.execute(
        select(WatchlistItem).where(WatchlistItem.ticker == req.ticker)
    )
    if row.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{req.ticker} is not on the watchlist; only Watchlist "
                "symbols may be backtested"
            ),
        )

    settings = get_settings()
    bars = await ensure_daily_bars(session, req.ticker, settings.market_data_provider)
    params_json = dataclasses.asdict(params)

    # Record + BACKTEST_STARTED + outcome event share ONE transaction (rule 12).
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.BACKTEST_STARTED,
        entity_type="backtests",
        entity_id=req.ticker,
        details={"params": params_json},
    )

    auto_decisions = None
    try:
        if params.instrument == "AUTO":
            result, auto_decisions = await _run_auto_backtest_with_provider(
                req.ticker,
                bars,
                params,
                settings.market_data_provider,
                auto_permissions,
                session,
            )
        elif params.instrument in ("LONG_CALL", "LONG_PUT"):
            result = await _run_call_backtest_with_provider(
                req.ticker, bars, params, settings.market_data_provider
            )
        elif params.instrument in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"):
            result = await _run_spread_backtest_with_provider(
                req.ticker, bars, params, settings.market_data_provider
            )
        elif params.instrument in ("COVERED_CALL", "CASH_SECURED_PUT"):
            result = await _run_income_backtest_with_provider(
                req.ticker, bars, params, settings.market_data_provider
            )
        elif params.instrument == "SHORT_STOCK":
            result = run_short_stock_backtest(
                [b.ts for b in bars],
                [b.open for b in bars],
                [b.high for b in bars],
                [b.low for b in bars],
                [b.close for b in bars],
                [b.volume for b in bars],
                params,
            )
        else:
            result = run_backtest(
                [b.ts for b in bars],
                [b.open for b in bars],
                [b.high for b in bars],
                [b.low for b in bars],
                [b.close for b in bars],
                [b.volume for b in bars],
                params,
            )
    except HTTPException:
        # Pre-engine refusals (e.g. provider lacks historical options) are
        # API answers, not engine failures — no FAILED record for them.
        raise
    except Exception as exc:  # engine failure -> persisted FAILED record
        record = BacktestRecord(
            ticker=req.ticker,
            status=STATUS_FAILED,
            params=params_json,
            metrics={},
            trades=[],
            equity_curve={},
            oos_start_date=None,
            error=str(exc),
        )
        session.add(record)
        await session.flush()
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.BACKTEST_FAILED,
            entity_type="backtests",
            entity_id=req.ticker,
            details={"backtest_id": record.id, "error": str(exc)},
        )
        await session.commit()
        await session.refresh(record)
        return _record_json(record)

    metrics, trades, equity_curve = _result_payloads(result)
    if auto_decisions is not None:
        # Additive key in the metrics JSONB: the §8 audit trail of every
        # AUTO entry decision (edge, tier, vol regime, instrument, rationale).
        metrics["auto_decisions"] = [
            {
                "date": d.day.isoformat(),
                "edge": d.edge,
                "tier": d.tier,
                "vol_regime": d.vol_regime,
                "instrument": d.instrument,
                "rationale": d.rationale,
            }
            for d in auto_decisions
        ]
    record = BacktestRecord(
        ticker=req.ticker,
        status=STATUS_COMPLETED,
        params=params_json,
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        oos_start_date=None,
        error="",
    )
    session.add(record)
    await session.flush()
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.BACKTEST_COMPLETED,
        entity_type="backtests",
        entity_id=req.ticker,
        details={
            "backtest_id": record.id,
            "num_trades": metrics["num_trades"],
            "profit_factor": metrics["profit_factor"],
            "total_return_pct": metrics["total_return_pct"],
        },
    )
    await session.commit()
    await session.refresh(record)
    return _record_json(record)


class PortfolioBacktestRequest(BaseModel):
    """POST /api/backtests/portfolio body (auto-strategy Phase C): the whole
    watchlist by default, or an explicit subset; params must resolve to
    instrument='AUTO' (the portfolio replay is AUTO by definition)."""

    tickers: list[str] | None = None
    params: dict = Field(default_factory=dict)
    instruments: list[str] | None = None
    # portfolio capital controls (docs/auto-strategy-portfolio-design.md §C)
    cash_floor_pct: float = 0.0
    max_positions: int | None = None
    max_gross_pct: float = 1.0


def _validate_portfolio_controls(req: "PortfolioBacktestRequest") -> None:
    if not (0.0 <= req.cash_floor_pct < 1.0):
        raise ValueError(f"cash_floor_pct must be in [0, 1), got {req.cash_floor_pct!r}")
    if req.max_positions is not None and req.max_positions < 1:
        raise ValueError(f"max_positions must be >= 1, got {req.max_positions!r}")
    if req.max_gross_pct <= 0.0:
        raise ValueError(f"max_gross_pct must be > 0, got {req.max_gross_pct!r}")


def _portfolio_record_json(r: PortfolioBacktestRecord) -> dict:
    return {
        "id": r.id,
        "tickers": r.tickers,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "status": r.status,
        "params": r.params,
        "metrics": r.metrics,
        "trades": r.trades,
        "equity_curve": r.equity_curve,
        "allocations": r.allocations,
        "decisions": r.decisions,
        "journal": r.journal,
        "advice": r.advice,
        "error": r.error,
    }


@router.post("/portfolio")
async def create_portfolio_backtest(
    req: PortfolioBacktestRequest, session: AsyncSession = Depends(get_session)
) -> dict:
    """Replay the watchlist (or a subset) against ONE shared cash ledger
    (docs/auto-strategy-portfolio-design.md Phase C): per-day §8 decisions
    per symbol, LIVE §12 tier-budget sizing, |edge|-priority contention, and
    the per-day signed allocation table the user asked for verbatim."""
    require_market_data_provider()
    unknown = sorted(set(req.params) - _PARAM_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown backtest param(s) {unknown}; "
                f"valid params: {sorted(_PARAM_FIELDS)}"
            ),
        )
    merged = {"instrument": "AUTO", **req.params}
    if merged["instrument"] != "AUTO":
        raise HTTPException(
            status_code=422,
            detail="the portfolio backtest is AUTO by definition — omit "
            "params.instrument or set it to 'AUTO'",
        )
    try:
        params = BacktestParams(**merged)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    permissions = account_permissions_from_settings()
    auto_permissions = _auto_permissions(req.instruments, permissions)

    # Tickers: explicit subset (each must be a member) or the whole watchlist.
    rows = (await session.execute(select(WatchlistItem))).scalars().all()
    watchlist = {w.ticker for w in rows}
    if req.tickers is not None:
        # normalize + validate + DEDUPE (verifier catch: duplicate slots
        # against one ledger broke the allocation identity)
        raw = [t.strip().upper() for t in req.tickers]
        bad = sorted({t for t in raw if not TICKER_RE.fullmatch(t)})
        if bad:
            raise HTTPException(status_code=422, detail=f"invalid ticker(s): {bad}")
        tickers = list(dict.fromkeys(raw))
        missing = sorted(set(tickers) - watchlist)
        if missing:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"{missing} not on the watchlist; only Watchlist "
                    "symbols may be backtested"
                ),
            )
        if not tickers:
            raise HTTPException(status_code=422, detail="tickers must not be empty")
    else:
        tickers = sorted(watchlist)
        if not tickers:
            raise HTTPException(
                status_code=422,
                detail="the watchlist is empty — add symbols before running "
                "a portfolio backtest",
            )

    settings = get_settings()
    provider_name = settings.market_data_provider
    bars_by_ticker = {}
    for tk in tickers:
        bars_by_ticker[tk] = await ensure_daily_bars(session, tk, provider_name)

    # Shared calendar = the INTERSECTION of the symbols' trading dates —
    # a symbol missing a shared date would otherwise need an invented mark
    # (§44 rule 18). The intersection is stated in the stored params echo.
    common = set.intersection(*(set(b.ts for b in bars) for bars in bars_by_ticker.values()))
    dates = sorted(common)
    if len(dates) < 2:
        raise HTTPException(
            status_code=422,
            detail="fewer than 2 shared trading dates across the selected "
            "symbols — nothing honest to replay",
        )
    symbols = []
    iv_by_ticker = {}
    call_providers = {}
    put_providers = {}
    provider = get_provider(provider_name)
    contracts_fn = getattr(provider, "get_option_contracts_window", None)
    bars_fn = getattr(provider, "get_option_daily_bars", None)
    for tk in tickers:
        by_date = {b.ts: b for b in bars_by_ticker[tk]}
        aligned = [by_date[d] for d in dates]
        symbols.append(SymbolBars(
            ticker=tk,
            opens=[b.open for b in aligned],
            highs=[b.high for b in aligned],
            lows=[b.low for b in aligned],
            closes=[b.close for b in aligned],
            volumes=[b.volume for b in aligned],
        ))
        iv_rows = await session.execute(
            select(AtmIvDailyRow).where(AtmIvDailyRow.ticker == tk)
        )
        iv_by_date = {r.bar_date: r.atm_iv for r in iv_rows.scalars().all()}
        iv_by_ticker[tk] = [iv_by_date.get(d) for d in dates]
        if contracts_fn is not None and bars_fn is not None:
            if auto_permissions.long_call:
                call_providers[tk] = _make_leg_resolver(
                    tk, params, contracts_fn, bars_fn, bear=False
                )
            if auto_permissions.long_put:
                put_providers[tk] = _make_leg_resolver(
                    tk, params, contracts_fn, bars_fn, bear=True
                )

    try:
        _validate_portfolio_controls(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    params_json = dataclasses.asdict(params)
    # the stored run states the capital constraints it actually ran under
    params_json["cash_floor_pct"] = req.cash_floor_pct
    params_json["max_positions"] = req.max_positions
    params_json["max_gross_pct"] = req.max_gross_pct
    await audit.record(
        session,
        actor_type=ActorType.USER,
        actor_id=CURRENT_USER,
        action=AuditAction.BACKTEST_STARTED,
        entity_type="portfolio_backtests",
        entity_id=",".join(tickers)[:64],
        details={"params": params_json, "tickers": tickers},
    )

    try:
        result = await asyncio.to_thread(
            run_portfolio_backtest,
            dates,
            symbols,
            params,
            permissions=auto_permissions,
            iv_series_by_ticker=iv_by_ticker,
            call_providers=call_providers,
            put_providers=put_providers,
            cash_floor_pct=req.cash_floor_pct,
            max_positions=req.max_positions,
            max_gross_pct=req.max_gross_pct,
        )
    except HTTPException:
        raise
    except Exception as exc:
        record = PortfolioBacktestRecord(
            tickers=tickers, status=STATUS_FAILED, params=params_json,
            metrics={}, trades=[], equity_curve={}, allocations={},
            decisions=[], journal=[], advice=[], error=str(exc),
        )
        session.add(record)
        await session.flush()
        await audit.record(
            session,
            actor_type=ActorType.SYSTEM,
            action=AuditAction.BACKTEST_FAILED,
            entity_type="portfolio_backtests",
            entity_id=str(record.id),
            details={"error": str(exc), "tickers": tickers},
        )
        await session.commit()
        await session.refresh(record)
        return _portfolio_record_json(record)

    metrics = dataclasses.asdict(result.metrics)
    trades = []
    for tk, t in result.trades:
        row = {
            "ticker": tk,
            "entry_date": t.entry_date.isoformat(),
            "entry_price": t.entry_price,
            "exit_date": t.exit_date.isoformat() if t.exit_date is not None else None,
            "exit_price": t.exit_price,
            "bars_held": t.bars_held,
            "return_pct": t.return_pct,
            "pnl": t.pnl,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
        }
        if isinstance(t, OptionTrade):
            row["contracts"] = t.contracts
            row["contract_symbol"] = t.contract_symbol
            row["strike"] = t.strike
        trades.append(row)
    equity_curve = {
        "dates": [d.isoformat() for d in result.dates],
        "equity": result.equity,
        "drawdown": result.drawdown,
    }
    allocations = {
        "dates": [d.isoformat() for d in result.dates],
        "by_symbol": result.allocations,
        "cash_pct": result.cash_pct,
    }
    decisions = [
        {
            "ticker": tk,
            "date": d.day.isoformat(),
            "edge": d.edge,
            "tier": d.tier,
            "vol_regime": d.vol_regime,
            "instrument": d.instrument,
            "rationale": d.rationale,
        }
        for tk, d in result.decisions
    ]
    journal = [
        {
            "date": ev.day.isoformat(),
            "ticker": ev.ticker,
            "action": ev.action,
            "instrument": ev.instrument,
            "quantity": ev.quantity,
            "price": ev.price,
            "reason": ev.reason,
            "sizing": ev.sizing,
            "cash_after": round(ev.cash_after, 2),
            "equity_prev": round(ev.equity_prev, 2),
        }
        for ev in result.journal
    ]
    advice = [
        {
            "severity": a.severity,
            "code": a.code,
            "finding": a.finding,
            "evidence": a.evidence,
            "suggestion": a.suggestion,
            "rationale": a.rationale,
        }
        for a in assess_portfolio_result(
            result.dates,
            result.equity,
            result.allocations,
            result.cash_pct,
            {sb.ticker: sb.closes for sb in symbols},
        )
    ]
    record = PortfolioBacktestRecord(
        tickers=tickers, status=STATUS_COMPLETED, params=params_json,
        metrics=metrics, trades=trades, equity_curve=equity_curve,
        allocations=allocations, decisions=decisions,
        journal=journal, advice=advice, error="",
    )
    session.add(record)
    await session.flush()
    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.BACKTEST_COMPLETED,
        entity_type="portfolio_backtests",
        entity_id=str(record.id),
        details={
            "portfolio_backtest_id": record.id,
            "num_trades": metrics["num_trades"],
            "total_return_pct": metrics["total_return_pct"],
            "tickers": tickers,
        },
    )
    await session.commit()
    await session.refresh(record)
    return _portfolio_record_json(record)


@router.get("/portfolio")
async def list_portfolio_backtests(
    limit: int = DEFAULT_LIST_LIMIT, session: AsyncSession = Depends(get_session)
) -> list[dict]:
    limit = max(1, min(limit, MAX_LIST_LIMIT))
    rows = (
        await session.execute(
            select(PortfolioBacktestRecord)
            .order_by(PortfolioBacktestRecord.id.desc())
            .limit(limit)
        )
    ).scalars().all()
    return [_portfolio_record_json(r) for r in rows]


@router.get("/portfolio/{portfolio_backtest_id}")
async def get_portfolio_backtest(
    portfolio_backtest_id: int, session: AsyncSession = Depends(get_session)
) -> dict:
    row = (
        await session.execute(
            select(PortfolioBacktestRecord).where(
                PortfolioBacktestRecord.id == portfolio_backtest_id
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"portfolio backtest {portfolio_backtest_id} not found"
        )
    return _portfolio_record_json(row)


@router.get("")
async def list_backtests(
    session: AsyncSession = Depends(get_session),
    ticker: str | None = Query(default=None, description="filter by ticker"),
    limit: int = Query(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
) -> list[dict]:
    """Newest-first backtest summaries (id, identity, headline full-segment
    metrics) — the detail endpoint serves the heavy payloads (plan §20)."""
    stmt = select(BacktestRecord).order_by(BacktestRecord.id.desc()).limit(limit)
    if ticker:
        stmt = stmt.where(BacktestRecord.ticker == ticker.strip().upper())
    rows = await session.execute(stmt)
    return [_summary_json(rec) for rec in rows.scalars().all()]


@router.get("/{backtest_id}")
async def get_backtest(
    backtest_id: int, session: AsyncSession = Depends(get_session)
) -> dict:
    """Full stored backtest record — metrics, trades, equity curve (plan §20)."""
    record = await session.get(BacktestRecord, backtest_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"backtest {backtest_id} not found")
    return _record_json(record)
