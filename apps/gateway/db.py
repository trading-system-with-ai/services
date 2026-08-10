"""Async SQLAlchemy setup and ORM models for the gateway (Phase 0/1 scope).

High-volume time-series tables (option chains, features) live in Timescale and
are managed by raw SQL migrations — they are intentionally NOT ORM-mapped here.
Exception: daily stock bars (stock_bars_daily) ARE ORM-mapped, because the lazy
backfill path (plan §4.2) writes them in the same transaction as its audit
event and daily granularity stays small at V1 scale.
"""
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from libs.common.config import get_settings


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    added_by: Mapped[str] = mapped_column(String(64))  # user id; only USER actors may add
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TradingPoolItem(Base):
    __tablename__ = "trading_pool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    trading_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    allowed_strategies: Mapped[list] = mapped_column(JSON, default=list)  # e.g. ["LONG_STOCK","LONG_CALL"]
    promoted_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StockBarDaily(Base):
    """One stored daily OHLCV bar (plan §4.2).

    Historical bars are stored only for Watchlist symbols (plan §4.2) plus the
    system reference indices SPY/QQQ/VIX (ADR-005). Rows are written by the
    lazy backfill path together with a SYSTEM DATA_BACKFILL audit event in the
    same transaction; (ticker, ts) is unique so a backfill can never duplicate
    a bar.
    """

    __tablename__ = "stock_bars_daily"
    __table_args__ = (
        UniqueConstraint("ticker", "ts", name="uq_stock_bars_daily_ticker_ts"),
        Index("ix_stock_bars_daily_ticker", "ticker"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16))
    ts: Mapped[date] = mapped_column(Date)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class BacktestRecord(Base):
    """One persisted run of Backtest Engine V1 (plan §20).

    The engine's full output — resolved params, per-segment metrics, trades and
    the equity curve — is stored as JSON exactly in the API response shape, so
    reads never recompute anything. Rows are written by the backtests router in
    the SAME transaction as their BACKTEST_* audit events (rule 12). ``status``
    is COMPLETED | FAILED; a FAILED row keeps the engine's error message and
    empty result payloads (honest nulls, plan §44 rule 18).
    """

    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16))  # COMPLETED | FAILED
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    trades: Mapped[list] = mapped_column(JSON, default=list)
    equity_curve: Mapped[dict] = mapped_column(JSON, default=dict)
    oos_start_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True, default=None
    )  # YYYY-MM-DD
    error: Mapped[str] = mapped_column(Text, default="")


class SystemState(Base):
    """Singleton row (id=1) backing the global kill switch (plan §18).

    Persisted so the pause/resume state survives restarts; trading is
    disabled by default and only an explicit USER resume enables it.
    """

    __tablename__ = "system_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trading_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str] = mapped_column(Text, default="startup default: trading disabled")
    updated_by: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )


# The kill switch is global, so exactly one system_state row exists.
SYSTEM_STATE_ID = 1


async def get_or_create_system_state(session: AsyncSession) -> SystemState:
    """Lazily get-or-create the singleton system_state row (id=1).

    Flushes (never commits) so callers control the transaction and can group
    the row with their audit event, per the audit-in-same-transaction rule.
    """
    state = await session.get(SystemState, SYSTEM_STATE_ID)
    if state is None:
        state = SystemState(id=SYSTEM_STATE_ID)
        session.add(state)
        await session.flush()
    return state


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor_type: Mapped[str] = mapped_column(String(16))  # USER | SYSTEM | LLM
    actor_id: Mapped[str] = mapped_column(String(64), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    correlation_id: Mapped[str] = mapped_column(String(64), default="")


_settings = get_settings()
engine = create_async_engine(_settings.database_url, echo=False)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Dev convenience: create tables if absent. Production uses migrations/."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    async with SessionLocal() as session:
        yield session
