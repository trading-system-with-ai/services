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


class Portfolio(Base):
    """Singleton row (id=1) holding the paper account's cash (plan §11).

    V1 runs one paper account, so exactly one portfolio row exists; cash is
    seeded from ``settings.paper_initial_cash`` on first access (a parameter,
    never a truth). NAV is always DERIVED as cash + open-position market
    value — it is deliberately not stored, so it can never go stale.
    """

    __tablename__ = "portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# One paper account in V1, so exactly one portfolio row exists.
PORTFOLIO_ID = 1


async def get_or_create_portfolio(session: AsyncSession) -> Portfolio:
    """Lazily get-or-create the singleton portfolio row (id=1, plan §11).

    Cash seeds from ``settings.paper_initial_cash``. Flushes (never commits)
    so callers control the transaction — same pattern as
    :func:`get_or_create_system_state`.
    """
    portfolio = await session.get(Portfolio, PORTFOLIO_ID)
    if portfolio is None:
        portfolio = Portfolio(id=PORTFOLIO_ID, cash=get_settings().paper_initial_cash)
        session.add(portfolio)
        await session.flush()
    return portfolio


class Position(Base):
    """One paper position (plan §11, §12.5).

    ``max_loss`` is the position-level maximum loss in dollars, fixed at open
    as ``quantity * stop_distance`` — the unit portfolio heat is measured in
    (plan §12.5). ``status`` is OPEN | CLOSED; only OPEN positions count
    toward NAV and heat.

    Exit-engine state (plan §11): ``stop_distance`` is the per-share dollar
    risk fixed at open (2 * ATR14 via the §10 chain), ``entry_edge`` the
    directional edge at entry, and ``entry_bar_date`` the last stored bar
    date at entry (YYYY-MM-DD) — the entry bar is bar 0 when counting
    ``bars_held``. ``realized_pnl`` accumulates over partial closes and
    holds the final figure once CLOSED (null until any close happens —
    honest nulls, plan §44 rule 18).
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    instrument: Mapped[str] = mapped_column(String(16), default="LONG_STOCK")
    quantity: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    max_loss: Mapped[float] = mapped_column(Float)  # dollars = quantity * stop_distance at open
    status: Mapped[str] = mapped_column(String(8), default="OPEN", index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    entry_edge: Mapped[float] = mapped_column(Float, default=0.0)
    stop_distance: Mapped[float] = mapped_column(Float, default=0.0)
    entry_bar_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True, default=None
    )  # YYYY-MM-DD of the last stored bar at entry (bars_held anchor)
    realized_pnl: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )


class Order(Base):
    """One executed paper order (plan §11, §42).

    ``side`` is ``BUY_TO_OPEN`` or ``SELL_TO_CLOSE`` ONLY — Sell-to-Open does
    not exist anywhere in this system, for options or stock (plan §5).
    ``client_order_id`` is the caller's optional idempotency key (§42):
    UNIQUE when present, so replaying the same key can only ever return the
    existing order — a duplicate request can never fill twice. V1 paper
    orders fill instantly, so ``status`` defaults to FILLED.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, default=None
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(16))  # BUY_TO_OPEN | SELL_TO_CLOSE (§5)
    quantity: Mapped[int] = mapped_column(Integer)
    fill_price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="FILLED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


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
