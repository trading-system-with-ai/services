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
import dataclasses

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.trading_core.backtest import BacktestParams, BacktestResult, run_backtest
from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import BacktestRecord, WatchlistItem, get_session
from ..deps import require_market_data_provider
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

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.match(v):
            raise ValueError(f"invalid ticker: {v!r}")
        return v


def _record_json(rec: BacktestRecord) -> dict:
    """Full backtest record in the API contract shape (plan §20)."""
    return {
        "id": rec.id,
        "ticker": rec.ticker,
        "created_at": rec.created_at.isoformat(),
        "status": rec.status,
        "params": rec.params,
        "error": rec.error or None,
        "oos_start_date": rec.oos_start_date,
        "metrics": rec.metrics,
        "trades": rec.trades,
        "equity_curve": rec.equity_curve,
    }


def _summary_json(rec: BacktestRecord) -> dict:
    """List-endpoint summary: identity plus headline full-segment metrics.

    ``fill_model`` (plan §20.2) is surfaced from the stored resolved params
    so history rows can chip the model without fetching the full record;
    records persisted before fill models existed report ``None``.
    """
    full = (rec.metrics or {}).get("full") or {}
    return {
        "id": rec.id,
        "ticker": rec.ticker,
        "created_at": rec.created_at.isoformat(),
        "status": rec.status,
        "num_trades": full.get("num_trades"),
        "total_return_pct": full.get("total_return_pct"),
        "profit_factor": full.get("profit_factor"),
        "oos_start_date": rec.oos_start_date,
        "fill_model": (rec.params or {}).get("fill_model"),
    }


def _result_payloads(result: BacktestResult) -> tuple[dict, list, dict, str | None]:
    """Serialize a BacktestResult into the JSON payloads the record stores.

    Exactly the API contract shapes (plan §20); the engine guarantees nulls
    where a metric is undefined — never NaN/Infinity (plan §44 rule 18).
    """
    metrics = {name: dataclasses.asdict(seg) for name, seg in result.metrics.items()}
    trades = [
        {
            "entry_date": t.entry_date.isoformat(),
            "entry_price": t.entry_price,
            "exit_date": t.exit_date.isoformat() if t.exit_date is not None else None,
            "exit_price": t.exit_price,
            "bars_held": t.bars_held,
            "return_pct": t.return_pct,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
        }
        for t in result.trades
    ]
    equity_curve = {
        "dates": [d.isoformat() for d in result.dates],
        "equity": result.equity,
        "drawdown": result.drawdown,
    }
    oos = result.oos_start_date.isoformat() if result.oos_start_date else None
    return metrics, trades, equity_curve, oos


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

    try:
        result = run_backtest(
            [b.ts for b in bars],
            [b.open for b in bars],
            [b.high for b in bars],
            [b.low for b in bars],
            [b.close for b in bars],
            [b.volume for b in bars],
            params,
        )
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

    metrics, trades, equity_curve, oos = _result_payloads(result)
    record = BacktestRecord(
        ticker=req.ticker,
        status=STATUS_COMPLETED,
        params=params_json,
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        oos_start_date=oos,
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
            "num_trades": metrics["full"]["num_trades"],
            "profit_factor": metrics["full"]["profit_factor"],
            "oos_total_return_pct": metrics["out_of_sample"]["total_return_pct"],
        },
    )
    await session.commit()
    await session.refresh(record)
    return _record_json(record)


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
