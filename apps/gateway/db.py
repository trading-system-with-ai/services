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
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    inspect,
    String,
    Text,
    text,
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

    Bars may be stored for ANY symbol a user browses (2026-08-20, §4.2
    amended: lazy backfill on read) — the presence of rows does NOT imply
    watchlist membership, and rows are never pruned. Rows are written by the
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


class StockBar1mRow(Base):
    """One stored 1-minute OHLCV bar (Catalyst Phase C, §17, §20). Mirrors
    migrations/002_system_state_and_bars.sql EXACTLY (mirror rule).

    THE TABLE PREDATES ITS ORM BY TWENTY MIGRATIONS. ``stock_bars_1m`` was
    created as Phase-1 groundwork and stayed empty because nothing needed a
    minute until event replay did: §17 asks what a stock did in the five,
    thirty and sixty minutes after a release, and a DAILY bar cannot answer
    that — its open already contains the whole overnight reaction. This class
    is the mirror that finally lets the gateway write it.

    ``ts`` IS THE BAR'S OPENING INSTANT IN UTC, and the primary key is
    ``(ticker, ts)`` — a COMPOSITE natural key, not a surrogate ``id``,
    unlike :class:`StockBarDaily`. That difference is the migration's, not a
    choice made here: the raw table is a Timescale hypertable partitioned on
    ``ts`` and declares the composite PK itself. Mirroring it exactly is what
    makes the backfill's ON CONFLICT upsert land on the same key the database
    enforces, so refetching an event window can only overwrite a minute, never
    duplicate one.

    ``volume`` is BIGINT in the migration and ``int`` here, deliberately
    unlike ``StockBarDaily.volume`` (a float): a minute's volume is a whole
    share count, and the adapters DROP a provider bar that reported no volume
    field rather than storing 0 — so a stored 0 always means "the provider
    said zero trades", never "we did not know" (§44 rule 18).

    Rows are written ONLY by ``event_replay.ensure_event_window_bars``, on an
    explicit USER backfill, together with a DATA_BACKFILL audit event in the
    same transaction (rule 12, ADR-003). Nothing lazily fills this table on a
    GET: a minute-bar window is thousands of rows per event, and a page
    showing twelve past earnings would fetch twelve windows nobody asked for.
    """

    __tablename__ = "stock_bars_1m"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)


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


class TradePlanRow(Base):
    """One research trade plan (upgrade §39/§40/§41, migration 013).

    ``preview`` stores the COMPLETE research-chain output (the §16 preview
    payload, gates and all) exactly as generated — the plan the user reviewed
    is the plan that gets applied, reproducible from this row alone.
    ``versions`` records the §41 configuration identifiers active at
    generation time. Lifecycle (§40) lives in ``status``; a superseded row
    points at its successor (``superseded_by``) so the chain of user
    decisions stays walkable. Applying NEVER places an order (§19).
    """

    __tablename__ = "trade_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # PlanStatus
    direction: Mapped[str] = mapped_column(String(8), default="AUTO")
    quantity_requested: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preview: Mapped[dict] = mapped_column(JSON)
    versions: Mapped[dict] = mapped_column(JSON, default=dict)
    market_data_as_of: Mapped[str | None] = mapped_column(String(10), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    applied_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    superseded_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))


class NewsArticleRow(Base):
    """One REAL news article, verbatim from the provider (migration 012).

    ``source_id`` is the provider's own article id — the DEDUPLICATION key
    (UNIQUE): re-ingesting the same feed inserts nothing. LLM recommendation
    evidence must cite rows of this table; the enrichment path drops any
    draft citing an article that is not stored here (no fabricated sources).
    """

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(128), unique=True)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(String(200), default="")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    url: Mapped[str] = mapped_column(Text)
    tickers: Mapped[list] = mapped_column(JSON, default=list)
    description: Mapped[str] = mapped_column(Text, default="")
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # --- Phase D news evidence (migration 023, appended in that order) ------
    # AS-OF-INDEPENDENT FIELDS ONLY (audit §7.1, §96). These four describe the
    # ARTICLE — its own text, its own publisher, the story it belongs to — and
    # mean the same thing at every instant. The as-of-DEPENDENT factors
    # (novelty, decay and the composite evidence score) are deliberately NOT
    # columns: they are functions of the as_of instant and of which other
    # articles shared the window, so freezing one request's viewpoint onto the
    # row would let the next read at a different as_of inherit it — a
    # look-ahead leak wearing a cache's clothes. apps/gateway/event_news.py is
    # the only writer, and it recomputes them per request.
    #
    # NULL means "not yet through the evidence pipeline", which is the honest
    # state of every row the recommendations ingest path stores (that path
    # mirrors articles, it does not analyse them). A default of 0.0/'OTHER'
    # would make an un-analysed article look like an immaterial one.
    cluster_id: Mapped[str | None] = mapped_column(String(64), default=None)
    materiality: Mapped[str | None] = mapped_column(String(32), default=None)
    materiality_score: Mapped[float | None] = mapped_column(Float, default=None)
    source_quality: Mapped[float | None] = mapped_column(Float, default=None)
    #: §22 relevance is per (article, TICKER) — a piece tagged AAPL and MSFT
    #: is 1.0 to one and 0.7 to the other — so this is an object
    #: ``{"AAPL": 1.0}`` rather than a float. A ticker the pipeline has not
    #: scored this article for is ABSENT, never 0.0 (§44 rule 18).
    relevance: Mapped[dict] = mapped_column(JSON, default=dict)


class RuntimeConfig(Base):
    """One UI-managed provider setting (migration 011).

    Provider selection and credentials are set from the Settings UI and
    stored here, overriding .env: rows are loaded into the process
    environment at startup and on every change, then the cached Settings
    object is rebuilt (apps/gateway/runtime_config.py owns that flow).

    SECRETS LIVE HERE. Values are never returned by any API, never logged,
    and never audited by value — CONFIG_CHANGED audit events record which
    keys changed, nothing else.
    """

    __tablename__ = "runtime_config"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


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

    Option positions (plan §8/§9, §12.1): ``instrument`` is LONG_CALL or
    LONG_PUT, ``opt_expiry``/``opt_strike``/``opt_right`` identify the
    contract, ``multiplier`` is 100 (1 for stock), ``quantity`` counts
    CONTRACTS, ``avg_price`` is the entry premium PER SHARE, ``max_loss`` is
    the full premium paid (``quantity * avg_price * multiplier`` — a long
    option's premium is fully at risk, §12.1), and ``stop_distance`` stores
    the per-share entry premium — the §11.3 PREMIUM hard-stop basis, NOT an
    underlying price stop. The opt_* columns are honest nulls for stock.
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
    # Option contract identity (null for stock — honest nulls, §44 rule 18).
    opt_expiry: Mapped[str | None] = mapped_column(
        String(10), nullable=True, default=None
    )  # YYYY-MM-DD
    opt_strike: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    opt_right: Mapped[str | None] = mapped_column(
        String(1), nullable=True, default=None
    )  # "C" | "P"
    multiplier: Mapped[int] = mapped_column(Integer, default=1)  # 100 for options
    # Defined-risk spread rows (roadmap Phase 1): instrument is then
    # BULL_CALL_SPREAD, opt_* identify the LONG leg, these two the SHORT leg
    # (bare OCC + strike); avg_price is the NET debit per share and max_loss
    # = quantity * net_debit * multiplier (defined risk, §12.1). Honest
    # nulls for every non-spread row.
    short_occ_symbol: Mapped[str | None] = mapped_column(
        String(24), nullable=True, default=None
    )
    short_strike: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    # Phase 2 — collateralized short premium (roadmap): COVERED_CALL rows
    # link the LONG_STOCK row whose shares back the short call (100 per
    # contract); CASH_SECURED_PUT rows lock strike*100*qty in cash_reserved.
    # For both, avg_price is the CREDIT received per share and opt_*
    # identify the SHORT contract. Honest NULLs on every other row.
    collateral_position_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    cash_reserved: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )


class Order(Base):
    """One executed paper order (plan §11, §42).

    ``side`` is one of ``libs.broker.provider.MLEG_LEG_SIDES`` — BUY_TO_OPEN /
    SELL_TO_CLOSE for longs, SELL_TO_OPEN / BUY_TO_CLOSE for the collateralized
    or margin-backed shorts the execution-chains program unlocked (roadmap
    Phases 1–3); migration 017 pins the same list as the DB CHECK and
    tests/test_migration_parity.py keeps the two from drifting. A NAKED short
    option is still unconstructable at the adapter (plan §5) — the ledger
    vocabulary is not the permission.
    ``client_order_id`` is the caller's optional idempotency key (§42):
    UNIQUE when present, so replaying the same key can only ever return the
    existing order — a duplicate request can never fill twice. V1 paper
    orders fill instantly, so ``status`` defaults to FILLED.

    Option orders (plan §8/§9): ``instrument`` is LONG_CALL or LONG_PUT with
    the contract identified by ``opt_expiry``/``opt_strike``/``opt_right``;
    ``quantity`` counts CONTRACTS and ``fill_price`` is the premium PER
    SHARE (x100 multiplier applies to cash). The opt_* columns are honest
    nulls for stock orders (§44 rule 18).

    Broker execution (plan §11): when the order went to a real broker,
    ``broker_order_id`` is the broker's own id (UNIQUE when set — one local row
    per broker order, so a reconciliation can never match two rows to one
    order), ``broker_status`` its RAW status string preserved verbatim, and
    ``filled_quantity`` how much ACTUALLY filled. All three are honest nulls /
    0 for internally simulated fills.

    PARTIAL FILLS ARE FIRST-CLASS. ``quantity`` is what we ASKED for and
    ``filled_quantity`` what happened; they are not the same fact and the code
    never conflates them. A position opens with the FILLED quantity, and a
    zero-fill ACCEPTED order opens no position at all.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, default=None
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    instrument: Mapped[str] = mapped_column(String(16), default="LONG_STOCK")
    side: Mapped[str] = mapped_column(String(16))  # MLEG_LEG_SIDES (migration 017)
    quantity: Mapped[int] = mapped_column(Integer)
    fill_price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="FILLED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Option contract identity (null for stock — honest nulls, §44 rule 18).
    opt_expiry: Mapped[str | None] = mapped_column(
        String(10), nullable=True, default=None
    )  # YYYY-MM-DD
    opt_strike: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    opt_right: Mapped[str | None] = mapped_column(
        String(1), nullable=True, default=None
    )  # "C" | "P"
    # Broker execution (plan §11) — honest nulls for simulated fills.
    broker_order_id: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, default=None
    )
    broker_status: Mapped[str | None] = mapped_column(
        String(24), nullable=True, default=None
    )  # the broker's RAW status string, preserved verbatim
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    # Order lifecycle (§11, migration 010). The row is written BEFORE the
    # broker submit as PENDING_SUBMIT and updated after; a crash mid-submit
    # leaves a row the order-sync sweep resolves by client_order_id. The
    # position link plus the approval-time risk context let a fill that
    # arrives AFTER the request returned (found by the sweep) open or shrink
    # the right position with the §10 chain's own parameters — never guessed.
    position_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None, index=True
    )
    stop_distance: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    entry_edge: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    entry_bar_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True, default=None
    )  # YYYY-MM-DD of the last stored bar at approval


class Recommendation(Base):
    """One LLM-proposed candidate (plan §4.1) — an INFORMATION row only.

    CENTRAL SAFETY RULE (plan §4.1, §44 rule 5, §46): the LLM proposes, the
    user curates. A recommendation row carries zero execution authority —
    nothing may move it into the Watchlist, Trading Pool, or orders except an
    explicit USER API action (POST /api/recommendations/{id}/promote).

    Scores follow the §4.1 schema: sentiment in [-1, 1]; impact, novelty and
    source_reliability in [0, 1]. ``evidence`` is a list of citation dicts
    ({"source", "published_at", "snippet"}), each published strictly before
    the generation as-of time (news timestamp integrity, plan §20.3).
    ``status`` is PENDING | DISMISSED | PROMOTED.
    """

    __tablename__ = "recommendations"
    __table_args__ = (Index("ix_recommendations_status_ts", "status", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ticker: Mapped[str] = mapped_column(String(16))
    company: Mapped[str | None] = mapped_column(String(128), nullable=True, default=None)
    sentiment: Mapped[float] = mapped_column(Float)
    impact: Mapped[float] = mapped_column(Float)
    novelty: Mapped[float] = mapped_column(Float)
    source_reliability: Mapped[float] = mapped_column(Float)
    horizon: Mapped[str] = mapped_column(String(16))
    catalyst_type: Mapped[str] = mapped_column(String(64))
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    #: §38/§41 — which provider/model generated this interpretation, recorded
    #: at generation time ("" on pre-upgrade rows: honest unknown, never
    #: backfilled from current settings, which may have changed since).
    llm_model: Mapped[str] = mapped_column(String(128), default="")


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



class RiskSnapshotRow(Base):
    """One BUILD of the statistical risk snapshot (Risk Engine Upgrade Phase
    B; spec §44/§45/§55/§56). Mirrors migrations/018_risk_snapshots.sql
    EXACTLY (mirror rule).

    ``trigger`` is SCHEDULED (the once-per-trading-day background build after
    the bar refresh — these rows are the NAV series live drawdown is measured
    on), ON_DEMAND (a risk-view request) or PRE_TRADE (built for an order
    decision). Scalars are typed columns; only diagnostics-shaped data
    (``data_quality`` reasons, the per-model ``model_health`` map) is JSON.
    Honest NULLs when no account / no data. SHADOW: nothing here alters a
    Tier 0 decision.
    """

    __tablename__ = "risk_snapshots"
    __table_args__ = (
        Index("ix_risk_snapshots_as_of", "as_of"),
        Index("ix_risk_snapshots_trigger", "trigger", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    snapshot_version: Mapped[str] = mapped_column(String(16))
    trigger: Mapped[str] = mapped_column(String(16))  # SCHEDULED | ON_DEMAND | PRE_TRADE
    nav: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    cash: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    cash_reserved: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    gross_exposure: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    delta_adjusted_exposure: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    heat_pct: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    heat_state: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    n_positions: Mapped[int] = mapped_column(Integer, default=0)
    n_obs: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    window_start: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)
    window_end: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)
    pnl_method: Mapped[str | None] = mapped_column(String(24), nullable=True, default=None)
    data_quality_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    data_quality: Mapped[dict] = mapped_column(JSON, default=dict)
    model_health: Mapped[dict] = mapped_column(JSON, default=dict)
    model_risk_state: Mapped[str | None] = mapped_column(
        String(16), nullable=True, default=None
    )
    dispersion_ratio: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    dispersion_high: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    distribution_primary: Mapped[str | None] = mapped_column(
        String(16), nullable=True, default=None
    )
    gaussian_trust: Mapped[str | None] = mapped_column(String(8), nullable=True, default=None)
    drawdown_current_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    drawdown_max_pct: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    risk_state: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RiskMetricRow(Base):
    """One (metric, model, confidence, horizon) result of a snapshot build,
    carrying the FULL §44 model identity inline (model_name, model_version,
    params, distribution, sample_size, diagnostics; the data window is on the
    parent snapshot) so every stored number is reproducible without a join.
    Mirrors migrations/018_risk_snapshots.sql EXACTLY. ``value`` is a USD
    LOSS (positive = money lost) for VAR/ES, USD/day for VOLATILITY; NULL
    when the model was UNAVAILABLE/FAILED (health + reason say why).
    """

    __tablename__ = "risk_metrics"
    __table_args__ = (
        Index("ix_risk_metrics_snapshot", "snapshot_id"),
        Index("ix_risk_metrics_model", "metric", "model_name", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("risk_snapshots.id", ondelete="CASCADE")
    )
    metric: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(48))
    model_version: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    horizon_days: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    distribution: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    value: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    value_pct_nav: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    health: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnostics: Mapped[dict] = mapped_column(JSON, default=dict)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RiskContributionRow(Base):
    """Per-position risk contribution (VOL component σ or Euler ES) of a
    snapshot build — capital weight vs risk weight with history (spec §10,
    §49). Mirrors migrations/018_risk_snapshots.sql EXACTLY."""

    __tablename__ = "risk_contributions"
    __table_args__ = (Index("ix_risk_contributions_snapshot", "snapshot_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("risk_snapshots.id", ondelete="CASCADE")
    )
    method: Mapped[str] = mapped_column(String(8))  # VOL | ES
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    position_key: Mapped[str] = mapped_column(String(64))
    ticker: Mapped[str] = mapped_column(String(16))
    instrument: Mapped[str] = mapped_column(String(24))
    contribution: Mapped[float] = mapped_column(Float)
    share: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    capital_weight: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PortfolioBacktestRecord(Base):
    """One persisted PORTFOLIO backtest run (auto-strategy Phase C,
    docs/auto-strategy-portfolio-design.md). Mirrors
    migrations/028_portfolio_backtests.sql EXACTLY (mirror rule).

    Same philosophy as BacktestRecord: full engine output stored as JSONB in
    the API response shape; rows written in the SAME transaction as their
    BACKTEST_* audit events (rule 12); FAILED rows keep the error message and
    empty payloads (§44 rule 18)."""

    __tablename__ = "portfolio_backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tickers: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16))  # COMPLETED | FAILED
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    trades: Mapped[list] = mapped_column(JSON, default=list)
    equity_curve: Mapped[dict] = mapped_column(JSON, default=dict)
    allocations: Mapped[dict] = mapped_column(JSON, default=dict)
    decisions: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str] = mapped_column(Text, default="")
    # 029 (ALTER, appended in migration order): the rebalance journal and
    # the risk-model advice — see migrations/029_portfolio_journal_advice.sql.
    journal: Mapped[list] = mapped_column(JSON, default=list)
    advice: Mapped[list] = mapped_column(JSON, default=list)


class AtmIvDailyRow(Base):
    """ATM implied volatility per underlying per day, persisted from the
    chain read that computes and discards it today (spec §24 empirical IV
    shocks / IV rank need history; audit §7.1). INTERNALLY CALCULATED from
    the provider chain — ``source`` labels it; never presented as vendor IV
    history. Mirrors migrations/018_risk_snapshots.sql EXACTLY."""

    __tablename__ = "atm_iv_daily"
    __table_args__ = (
        UniqueConstraint("ticker", "bar_date", name="uq_atm_iv_daily_ticker_date"),
        Index("ix_atm_iv_daily_ticker", "ticker"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16))
    bar_date: Mapped[date] = mapped_column(Date)
    atm_iv: Mapped[float] = mapped_column(Float)
    spot: Mapped[float] = mapped_column(Float)
    expiry: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)
    dte: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    source: Mapped[str] = mapped_column(String(24))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StressRunRow(Base):
    """One SCENARIO of one stress run (Risk Engine Upgrade Phase D; spec §25,
    §26, §51, §56; design §8.4). Mirrors migrations/019_stress_runs.sql
    EXACTLY (mirror rule).

    Every snapshot build runs the whole default catalogue and persists one row
    per scenario, so the table IS the scenario history: what the book of that
    day would have lost under that scenario stays answerable after the book
    has changed. ``snapshot_id`` is NULL for a USER row (POST
    /api/risk/stress/run) — a user-defined hypothesis is a READ of the current
    book under a hypothesis, not a snapshot build, and it writes no audit
    event while still keeping its history (spec §56).

    ``pnl_usd`` is GAIN-POSITIVE (a stress LOSS is negative) and NULLABLE: a
    scenario whose window falls outside the stored history is stored as an
    UNAVAILABLE row with its ``reason``, never as a fabricated 0 (§44 rule
    18). ``method_full_reval`` / ``method_delta_linear`` count the legs priced
    each way, so a book that degraded to delta-linear pricing (no IV on a
    leg) is visible in history rather than silently equivalent to a full
    revaluation. SHADOW: nothing here alters a Tier 0 decision.
    """

    __tablename__ = "stress_runs"
    __table_args__ = (
        Index("ix_stress_runs_snapshot", "snapshot_id"),
        Index("ix_stress_runs_scenario", "scenario", "as_of"),
        Index("ix_stress_runs_kind", "kind", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("risk_snapshots.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
    )
    scenario: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    pnl_pct_nav: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    method_full_reval: Mapped[int] = mapped_column(Integer, default=0)
    method_delta_linear: Mapped[int] = mapped_column(Integer, default=0)
    health: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    per_position: Mapped[dict] = mapped_column(JSON, default=dict)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RiskModelBacktestRow(Base):
    """One MODEL VIEW of one VaR/ES validation run (Risk Engine Upgrade Phase
    E; spec §42, §43, §56, §57, §63, §68; design §9.4). Mirrors
    migrations/020_risk_model_backtests.sql EXACTLY (mirror rule).

    A run scores the whole view grid — historical VaR 95/99, Gaussian VaR
    95/99, EWMA-filtered VaR 95 and the RESEARCH GARCH-filtered VaR 95 — and
    persists one row each, so the table IS the calibration history: whether
    the 99% VaR was actually breaching ~1% of days last quarter stays
    answerable after the book that produced it has changed.

    WALK-FORWARD ONLY (spec §43): every forecast counted in a row was
    produced on a rolling window of observations STRICTLY BEFORE the day it
    forecasts. ``window_obs`` is that window's length (the design doc calls
    it ``window``; that word is RESERVED in PostgreSQL, so both the column
    and this attribute carry the unit in the name).

    ``verdict`` is the Basel-style traffic light on the Kupiec p-value, and
    ``UNAVAILABLE`` when there were too few usable pairs to test. An
    UNAVAILABLE row is PERSISTED with its ``reason`` and NULL statistics
    rather than skipped — "not yet validatable" is a fact, and a missing row
    would later read as "never run" (§44 rule 18). Every statistic column is
    nullable for the same reason. ``rate`` / ``expected_rate`` are FRACTIONS.

    ``snapshot_id`` is the SCHEDULED build that triggered the run, NULL for an
    on-demand ``POST /api/risk/validation/run`` — validating a model is a READ
    of the book's P&L history, not a snapshot build, and it writes no audit
    event while still keeping its history (spec §56).

    SHADOW/RESEARCH: nothing here alters a Tier 0 decision.
    """

    __tablename__ = "risk_model_backtests"
    __table_args__ = (
        Index("ix_risk_model_backtests_snapshot", "snapshot_id"),
        Index("ix_risk_model_backtests_as_of", "as_of"),
        Index("ix_risk_model_backtests_model", "model_name", "confidence", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    snapshot_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("risk_snapshots.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
    )
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(16))
    distribution: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    horizon_days: Mapped[int] = mapped_column(Integer)
    window_obs: Mapped[int] = mapped_column(Integer)
    n_forecasts: Mapped[int] = mapped_column(Integer)
    exceedances: Mapped[int] = mapped_column(Integer)
    rate: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    expected_rate: Mapped[float] = mapped_column(Float)
    kupiec_lr: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    kupiec_p: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    christoffersen_lr: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    christoffersen_p: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    es_severity_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    verdict: Mapped[str] = mapped_column(String(16))
    health: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EventRow(Base):
    """One real-world catalyst in the event registry (Catalyst & Event
    Intelligence Phase B; event spec §5, §6, §7, §11; audit §5.2). Mirrors
    migrations/021_events.sql EXACTLY (mirror rule).

    ``event_key`` is the deterministic natural key (``EARNINGS:NVDA:2026-08-27``,
    ``CPI:2026-07``, ``FOMC_DECISION:2026-09-16``) and is UNIQUE: an event is
    ONE fact that several sources describe with differing authority, so a
    better source UPDATES the row (source-precedence merge) instead of
    inserting a rival copy. That uniqueness is also what makes ingestion
    idempotent at the DATABASE level, which ADR-007 (no leader election)
    requires — a second replica can only collide, never double-insert.

    ``status`` carries EventStatus. ESTIMATED is a DERIVED date (filing
    cadence), never presented as a confirmed fact and never alerted on (§7);
    the value travels into every downstream payload so the UI can label it.
    ``source`` is the EventSourceKind PRECEDENCE TIER and ``source_name`` the
    concrete adapter that wrote the row ("sec_edgar", "alpaca_calendar",
    "fed_fomc", "derived_cadence", "user") — priority is data, not code (§78).

    Instants are UTC, and ``event_timezone`` additionally stores the event's
    own zone: "08:30 ET" is the fact a macro release asserts, and its UTC
    instant shifts across DST. ``previous_event_id`` is the self-FK to the
    previous comparable event with ``comparison_reason`` recording why that
    row was chosen (§15). ``importance`` is NULL when not yet scored — never
    a fabricated 0. Only ``revision_history`` (the append-only trail of date
    moves) is JSON; every scalar the UI reads is a typed column.
    """

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_scheduled_at", "scheduled_at"),
        Index("ix_events_ticker", "ticker", "scheduled_at"),
        Index("ix_events_type", "event_type", "scheduled_at"),
        Index("ix_events_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(String(200), unique=True)
    event_type: Mapped[str] = mapped_column(String(32))  # EventType
    title: Mapped[str] = mapped_column(String(300))
    ticker: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    company_id: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    event_timezone: Mapped[str] = mapped_column(String(64), default="America/New_York")
    session: Mapped[str] = mapped_column(String(16), default="UNKNOWN")  # EventSession
    status: Mapped[str] = mapped_column(String(16))  # EventStatus
    source: Mapped[str] = mapped_column(String(32))  # EventSourceKind (precedence tier)
    source_name: Mapped[str] = mapped_column(String(64))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    source_event_id: Mapped[str | None] = mapped_column(
        String(200), nullable=True, default=None
    )
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    previous_event_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("events.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    comparison_reason: Mapped[str | None] = mapped_column(
        String(200), nullable=True, default=None
    )
    importance: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    series_id: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    agency: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    release_period: Mapped[str | None] = mapped_column(
        String(32), nullable=True, default=None
    )
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    speaker: Mapped[str | None] = mapped_column(String(120), nullable=True, default=None)
    topic: Mapped[str | None] = mapped_column(String(300), nullable=True, default=None)
    revision_history: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MarketCalendarRow(Base):
    """One exchange session day (event spec §10; audit §5.2). Mirrors
    migrations/021_events.sql EXACTLY (mirror rule).

    The real session grid from Alpaca ``/v2/calendar`` and Massive
    ``/v1/marketstatus/upcoming``, keyed by ``session_date``. It fills the
    hole admitted in routers/analysis.py::_last_expected_trading_date
    ("Holidays are not modeled") and is what classifies an event as
    BEFORE_MARKET / DURING_MARKET / AFTER_MARKET on a HALF DAY, where the
    09:30-16:00 default would put an early-close release in the wrong
    session. ``open_utc``/``close_utc`` are the regular session;
    ``session_open_utc``/``session_close_utc`` the extended session and
    NULLABLE — a provider that does not report extended hours leaves them
    NULL rather than a guessed 04:00/20:00.
    """

    __tablename__ = "market_calendar"
    __table_args__ = (
        Index("ix_market_calendar_exchange", "exchange", "session_date"),
    )

    session_date: Mapped[date] = mapped_column(Date, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(16), default="US")
    open_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    close_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    session_open_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    session_close_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    is_early_close: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EventIngestStateRow(Base):
    """One calendar provider's ingestion watermark (event spec §8 "calendar
    ingestion should survive individual provider failures"; audit §5.2).
    Mirrors migrations/021_events.sql EXACTLY (mirror rule).

    ``key`` is the adapter, or adapter:ticker where the fetch is per-symbol
    ("sec_edgar:NVDA", "fed_fomc", "alpaca_calendar"). ``last_ok_at`` is the
    last SUCCESS and gates the re-fetch cadence (SEC/Fed daily, calendars
    weekly); ``last_fetched_at`` is the last ATTEMPT, so a provider that has
    been failing for a week is distinguishable from one never tried.
    ``last_error`` keeps the last honest failure string (403
    SUBSCRIPTION_DENIED, timeout, HTML parse failure) — a dead adapter stays
    visible instead of silently contributing nothing — and ``meta`` carries
    its last capability report. Failure is per-adapter: a failing provider
    updates its own row and leaves every other provider's rows committed.
    """

    __tablename__ = "event_ingest_state"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    last_fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    last_ok_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class FundamentalStatementRow(Base):
    """One filed financial statement period, mirrored point-in-time (Catalyst
    & Event Intelligence Phase E2; event spec §16, §28, §85, §96; audit §7.1,
    §11.3). Mirrors migrations/022_fundamental_statements.sql EXACTLY (mirror
    rule).

    ``acceptance_datetime`` IS THE AS-OF KEY — the instant the filing became
    public, which is weeks after ``end_date``, the instant the PERIOD closed.
    Every consumer filters ``acceptance_datetime <= as_of`` and nothing else;
    an as-of written against ``end_date`` would let a quarter inform an
    analysis run before that quarter was ever filed, which is exactly the
    look-ahead §96 requires a planted sentinel to catch. It is NULLABLE
    because a provider row may omit it: such a row is still STORED (it is
    real, and dropping it would hide that the feed is degraded) but the pure
    layer excludes it from every as-of answer with a stated reason rather
    than inferring publication from ``filing_date``.

    The natural key is ``(ticker, timeframe, fiscal_year, fiscal_period,
    end_date)``, so re-fetching is idempotent at the DATABASE level and a
    refresh can only collide, never accumulate duplicate quarters. A
    RESTATEMENT has that same key with a LATER acceptance instant and
    therefore OVERWRITES: the vendor serves only its current XBRL view and
    does not retain the superseded original (audit §7.3), so claiming to
    store both would be a fiction — the moved acceptance instant is itself
    the flag that a period was restated.

    ``values`` is the provider's flattened ``"income_statement.revenues" ->
    number`` mapping and is the one justified JSON column: the field NAME SET
    differs per filer and per period, so a fixed column list would either
    truncate filers or invent NULLs that read as "reported nothing". Fields
    the filer did not report are ABSENT from it, never stored as 0 (§44 rule
    18 — a missing capex and a capex of zero are different facts).
    ``raw_fields_count`` is the provider row's field count BEFORE the numeric
    filter, so a mostly-unparseable filing is visible without re-fetching,
    and ``fetched_at`` is when THIS platform stored the row — a different
    fact from every other date here, all of which are the filer's.
    """

    __tablename__ = "fundamental_statements"
    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "timeframe",
            "fiscal_year",
            "fiscal_period",
            "end_date",
            name="uq_fundamental_statements_period",
        ),
        Index(
            "ix_fundamental_statements_ticker_acceptance",
            "ticker",
            "acceptance_datetime",
        ),
        Index(
            "ix_fundamental_statements_ticker_period",
            "ticker",
            "timeframe",
            "end_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16))
    cik: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    timeframe: Mapped[str] = mapped_column(String(16))  # quarterly | annual | ttm
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    fiscal_period: Mapped[str] = mapped_column(String(16))  # Q1..Q4 | FY | TTM
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)  # period end — NEVER the as-of key
    filing_date: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)
    acceptance_datetime: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    source_filing_url: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    values: Mapped[dict] = mapped_column(JSON, default=dict)
    raw_fields_count: Mapped[int] = mapped_column(Integer, default=0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EventAnalysisRow(Base):
    """One stored event analysis package: the evidence bundle that was
    assembled and the LLM synthesis produced from EXACTLY that bundle
    (Catalyst & Event Intelligence Phase F; event spec §16, §46-§52, §69-§71,
    §99; audit §7.2, §9.3, §11.6). Mirrors
    migrations/024_event_analyses.sql EXACTLY (mirror rule).

    THE BUNDLE IS STORED WITH THE ANALYSIS, and that is the point of the
    table. §47 forbids the model from computing anything: every number in
    ``analysis`` must be QUOTED from ``bundle``, and the ``numbers_quoted``
    validator checks each one against the bundle's fact index. That check is
    only meaningful against the document the model actually saw. Re-deriving
    the bundle at read time would rebuild it from TODAY's stored filings,
    prices and articles — a different document — so a later reader would be
    validating the wrong evidence and could not tell. ``bundle`` is therefore
    a snapshot, NOT NULL: a row without its evidence is not an analysis, it is
    an assertion.

    ``bundle_digest`` (sha256 of the bundle's canonical JSON) is the CACHE
    KEY, and ``(event_id, bundle_digest, prompt_version, model)`` is UNIQUE so
    the dedupe is a DATABASE fact rather than a race between two request
    handlers — the same idempotence-by-unique-key ADR-007 relies on, there
    being no distributed lock. ``model`` and ``prompt_version`` are in the key
    because the same evidence read by a different model, or under revised
    instructions, is a different answer that must be allowed to coexist with
    the old one; both are NULL on a failed attempt, and NULLs compare distinct
    in a UNIQUE index, so a failure never blocks the later success.

    ``status`` is the honest outcome vocabulary, deliberately not a boolean:
    OK (returned, every quoted number checked out), INVALID (returned, but the
    validator found violations — the text is STILL stored with its
    ``violations`` list, because hiding a model that quoted an invented number
    destroys the evidence that it did, §99), FAILED (the provider raised;
    ``error`` keeps the honest string and ``analysis`` stays NULL rather than
    a placeholder narrative, §44 rule 18) and BUNDLE_ONLY (evidence assembled,
    no synthesis asked for). SUPERSEDED (was OK, until a
    FORCED re-run on the same evidence produced a newer good answer — the text
    is untouched and stays readable, because deleting it is what would make a
    regression between two model versions undiagnosable). ``violations`` is
    NOT NULL default ``[]`` because an empty list ("checked, nothing wrong")
    is a different fact from NULL ("never checked").

    ``as_of`` is the instant the bundle was assembled AS OF and is NOT
    ``created_at``: a historical analysis re-run for 2025-10-30 is written
    today and is as-of then. Every look-ahead question is answered against
    ``as_of``; ``created_at`` only orders the trail.
    """

    __tablename__ = "event_analyses"
    __table_args__ = (
        # PARTIAL unique index (``WHERE status = 'OK'``), not a table
        # constraint: at most ONE cached good answer per (event, evidence,
        # prompt, model), while FAILED and INVALID attempts on that same
        # evidence stay storable — a retry is exactly what those statuses
        # invite, and a total UNIQUE would forbid the one action they suggest.
        # ``sqlite_where`` mirrors ``postgresql_where`` so the test harness
        # builds the same predicate the production DDL declares.
        Index(
            "uq_event_analyses_cache",
            "event_id",
            "bundle_digest",
            "prompt_version",
            "model",
            unique=True,
            postgresql_where=text("status = 'OK'"),
            sqlite_where=text("status = 'OK'"),
        ),
        Index("ix_event_analyses_event_created", "event_id", "created_at"),
        Index("ix_event_analyses_as_of", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE")
    )
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kind: Mapped[str] = mapped_column(String(16), default="PRE_EVENT")
    bundle: Mapped[dict] = mapped_column(JSON, default=dict)
    bundle_digest: Mapped[str] = mapped_column(String(64))
    analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True, default=None)
    prompt_version: Mapped[str | None] = mapped_column(
        String(32), nullable=True, default=None
    )
    usage: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    violations: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OptionDailyBarRow(Base):
    """One stored DAILY OHLCV bar for ONE OPTION CONTRACT (Catalyst Phase I;
    event spec §18, §36, §37; audit §7.3). Mirrors
    migrations/025_event_options.sql EXACTLY (mirror rule).

    A SEPARATE TABLE FROM ``stock_bars_daily``, and the reason is the key.
    An option bar's identity is the CONTRACT — ``O:AAPL250801C00210000``,
    which already encodes underlying, expiry, right and strike — not a
    ticker. Folding these rows into the equity table would put a 21-character
    OCC symbol into a ``VARCHAR(16)`` column shared with watchlist symbols,
    and every existing query that means "the stock's bars" would start
    matching option contracts.

    ``volume`` IS BIGINT AND NULLABLE, unlike either equity bar table. A
    daily option aggregate reports contracts traded, a whole number; and an
    illiquid contract's bar can arrive with no volume field at all, which is
    a different fact from "zero contracts traded" (§44 rule 18). The
    adapter's ``get_option_history_bars`` normalises a missing ``v`` to 0.0
    for its own value type, so what actually lands here is provider truth —
    but the column stays nullable so a future adapter that preserves the
    absence has somewhere honest to put it.

    THE PRIMARY KEY IS ``(option_ticker, bar_date)`` — composite and natural,
    like :class:`StockBar1mRow` and unlike :class:`StockBarDaily`. A daily
    bar for one contract on one session is ONE fact, so a refetch can only
    overwrite it, never duplicate it, and the backfill's idempotence is a
    database property rather than a race between two request handlers
    (ADR-007, there being no distributed lock).

    ``provider`` records WHO served the bar. Massive is the only provider
    with dated option aggregates today (Alpaca raises
    ``CapabilityNotAvailable``), but a stored premium whose source is
    unknown cannot be audited, and the stub writes ``"stub"`` here so a test
    fixture is never mistaken for market data.

    Rows are written ONLY by ``event_options.backfill_event_options`` on an
    explicit USER backfill, with a ``DATA_BACKFILL`` audit row in the same
    transaction (rule 12, ADR-003). Nothing lazily fills this table on a GET.
    """

    __tablename__ = "option_daily_bars"
    __table_args__ = (
        Index("ix_option_daily_bars_date", "bar_date"),
    )

    option_ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    bar_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True, default=None)
    provider: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EventOptionMetricRow(Base):
    """The implied-move picture computed for ONE event under ONE basis
    (Catalyst Phase I; event spec §18, §36, §37, §66; audit §7.3). Mirrors
    migrations/025_event_options.sql EXACTLY (mirror rule).

    WHY THE BASIS IS IN THE UNIQUE KEY AND NOT A FLAG. §37's whole point is
    that an implied move read off a LIVE chain and one RECONSTRUCTED from
    daily closes are different claims with different confidence: the live
    snapshot is a real bid/ask midpoint at a known instant, the historical
    one is a settlement close standing in for a mark nobody observed. Storing
    them under ``UNIQUE(event_id, basis)`` means the two can coexist for the
    same event — an upcoming print's live number today, its historical
    reconstruction after it happens — and neither can silently overwrite the
    other. ``basis`` is ``LIVE_CHAIN_SNAPSHOT`` or
    ``HISTORICAL_DAILY_CLOSE_APPROXIMATION``, the two constants
    ``libs/trading_core/events/implied_move.py`` defines.

    EVERY PRICE COLUMN IS NULLABLE, and that is the §44-rule-18 contract
    rather than laziness. If the put leg never traded on the pre-event
    session there is no straddle, and the honest row carries
    ``pre_put_close = NULL``, ``implied_move_pct = NULL`` and
    ``status = 'NO_DATA'`` with the reason in ``notes``. A zero would read as
    "the put was free", which is the single most dangerous fabrication this
    table could store — it would halve every implied move that touched it.

    ``status`` is the honest outcome vocabulary: OK (a pre-event straddle and
    a usable post side), PARTIAL (the implied move computed but a downstream
    metric — the crush, the ratio — did not) and NO_DATA (no implied move at
    all). The UI treats a finite number arriving beside NO_DATA as the server
    retracting its own computation and suppresses it, so PARTIAL is the value
    to use whenever the straddle itself is real.

    ``as_of`` is the instant the metrics were computed AS OF and is NOT
    ``created_at``: a historical reconstruction for 2025-10-30 is written
    today and is as-of then. ``notes`` is the library's own reason map
    (JSON), so a reader can always answer "why is this column NULL" from the
    row alone.

    ``ON DELETE CASCADE`` from ``events(id)``: an implied move is a statement
    about one event and there is nothing to preserve in an orphan.
    """

    __tablename__ = "event_option_metrics"
    __table_args__ = (
        UniqueConstraint("event_id", "basis", name="uq_event_option_metrics_basis"),
        Index("ix_event_option_metrics_event", "event_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE")
    )
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    basis: Mapped[str] = mapped_column(String(48))
    expiry: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)
    strike: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    spot: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    call_ticker: Mapped[str | None] = mapped_column(
        String(32), nullable=True, default=None
    )
    put_ticker: Mapped[str | None] = mapped_column(
        String(32), nullable=True, default=None
    )
    pre_call_close: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    pre_put_close: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    post_call_close: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    post_put_close: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    implied_move_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    implied_move_points: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    iv_before: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    iv_after: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    iv_crush_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    actual_move_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    implied_realized_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    classification: Mapped[str | None] = mapped_column(
        String(16), nullable=True, default=None
    )
    status: Mapped[str] = mapped_column(String(16))
    notes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MacroObservationRow(Base):
    """One published macro statistic for one REFERENCE PERIOD (Phase G, §8,
    §38). Mirrors migrations/026_macro_data.sql EXACTLY (mirror rule).

    THE PRIMARY KEY IS (series_id, period) — the identity of the fact is
    "CPI-U all items for 2026-07", not "the number printed on 2026-08-12". An
    agency revises the same period repeatedly (BLS restates SA series every
    February; BEA revises a quarter three times), so a re-fetch OVERWRITES.
    Accumulating a row per fetch would double every MoM computed over the
    series. The revision HISTORY is deliberately not modelled: this table
    holds the current vintage, and the point-in-time question is answered by
    ``release_at`` rather than by a vintage key.

    ``value`` is NULLABLE with no default: a withheld observation stores NULL,
    because 0.0 for an index level implies a -100% month-over-month (§44 rule
    18). ``release_at``/``release_basis`` travel together — BLS's data API
    carries no timestamps at all, so the instant comes either from the
    agency's release schedule (SCHEDULED) or from period-end + 45 days
    (ESTIMATED), and those are different claims about what was knowable when.
    """

    __tablename__ = "macro_observations"
    __table_args__ = (
        Index("ix_macro_observations_release_at", "release_at"),
    )

    series_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    period: Mapped[str] = mapped_column(String(16), primary_key=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    release_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    release_basis: Mapped[str | None] = mapped_column(
        String(16), nullable=True, default=None
    )
    provider: Mapped[str] = mapped_column(String(16))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class TreasuryYieldRow(Base):
    """One session of the Treasury par yield curve (Phase G, §39). Mirrors
    migrations/026_macro_data.sql EXACTLY (mirror rule).

    THE WHOLE CURVE IS ONE ROW. Treasury publishes thirteen tenors for one
    business day and this platform reads them together (the 2Y and the 10Y
    across the same release), so ``tenors`` is a JSON object keyed by the
    CSV's own tenor spelling — ``"2 Yr"``, ``"10 Yr"``. Normalising to
    (curve_date, tenor) rows would make every read a pivot for no query this
    platform issues, and would put the tenor vocabulary in a column that has
    to be migrated the next time Treasury adds one.

    A TENOR TREASURY DID NOT QUOTE IS ABSENT FROM THE OBJECT, never 0.0 — the
    20Y was unpublished for years and the 30Y for a decade, and zero percent
    is a claim nobody made.
    """

    __tablename__ = "treasury_yields"

    curve_date: Mapped[date] = mapped_column(Date, primary_key=True)
    tenors: Mapped[dict] = mapped_column(JSON, default=dict)
    provider: Mapped[str] = mapped_column(String(16))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class FedDocumentRow(Base):
    """One Federal Reserve document stored VERBATIM (Phase H, §9, §42-§45).
    Mirrors migrations/027_fed_documents.sql EXACTLY (mirror rule).

    THE TEXT IS STORED BECAUSE THE DIFF MUST BE REPRODUCIBLE. §44 makes the
    source document authoritative: the sentence-level diff that says what the
    Committee changed at its last meeting is computed over the statement's own
    paragraphs, and a diff whose inputs were re-fetched on every read would
    change under the reader the moment federalreserve.gov edited a page or
    answered 403. Storing the words is also what lets the read endpoint hold
    no network handle at all, and what lets an as-of replay show the June
    statement as it stood in June.

    ``url`` IS THE IDENTITY. One Fed URL is one document forever, so the
    backfill's upsert lands on it. A (doc_type, meeting_date) key would be
    wrong for speeches, which have no meeting and several of which share a day.

    ``meeting_date`` IS THE MEETING, ``released_at`` IS THE PUBLICATION. They
    are twenty-one days apart for minutes and that gap is the point: the
    as-of gate reads ``released_at`` and the join onto the FOMC_* event rows
    reads ``meeting_date``. ``released_at`` is NULLABLE and NULL means
    UNKNOWN — the minutes' instant comes only from the press_monetary RSS
    feed, which reaches back a finite distance, and "we do not know when this
    became public" is a different claim from midnight (§44 rule 18).

    ``parsed`` holds ONLY what the document states outright — the vote, the
    target range, a speech's speaker — each carrying the sentence it was read
    from. Nothing inferred or scored is stored: §43 forbids a single
    hawkish/dovish label anywhere, and a column that could hold one is a
    column someone eventually fills.

    ``event_id`` is ``ON DELETE SET NULL``, not CASCADE: a statement is a fact
    published by the Federal Reserve and the events row is merely this
    platform's index of it. Deleting a mis-ingested calendar row must not
    delete the Fed's words.

    ``id`` is ``Integer`` against a ``SERIAL`` column, matching every other
    surrogate key in this schema — including ``events.id``, which this table's
    foreign key targets. A BIGINT referencing a SERIAL is a mismatch Postgres
    accepts and then indexes badly, and the Fed publishes eight statements a
    year.
    """

    __tablename__ = "fed_documents"
    __table_args__ = (
        Index("ix_fed_documents_type_meeting", "doc_type", "meeting_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_type: Mapped[str] = mapped_column(String(24))
    meeting_date: Mapped[date | None] = mapped_column(Date, nullable=True, default=None)
    event_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="SET NULL"), nullable=True,
        default=None,
    )
    url: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text, default="")
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    text: Mapped[str] = mapped_column(Text, default="")
    paragraphs: Mapped[list] = mapped_column(JSON, default=list)
    parsed: Mapped[dict] = mapped_column(JSON, default=dict)
    provider: Mapped[str] = mapped_column(String(16))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class EventSearchRunRow(Base):
    """One external WEB SEARCH refresh for one event (Catalyst research
    upgrade, plan §4). Mirrors migrations/030_web_research.sql EXACTLY
    (mirror rule).

    THE RUN IS THE AUDITABLE UNIT: what was asked (``plan``, stored
    verbatim — deterministic code's output, never the LLM's), under which
    point-in-time window (``window_start``/``window_end``/``window_basis``
    with ``previous_event_id``/``fallback_reason`` so a fallback window can
    never masquerade as one anchored on a real previous event), what it cost
    (``queries_executed`` — plan Phase 12 cost transparency is a column
    read, not a scan), and what it admitted/suppressed. Rows are written
    ONLY by the explicit USER research backfill, never by a GET.

    ``status`` is the honest outcome vocabulary (OK | PARTIAL | FAILED,
    defined in code): a run where one query failed and four answered is
    PARTIAL with the failures named in ``skipped``, never silently OK.
    """

    __tablename__ = "event_search_runs"
    __table_args__ = (
        Index("ix_event_search_runs_event", "event_id", "as_of"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE")
    )
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_basis: Mapped[str] = mapped_column(String(48))
    previous_event_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="SET NULL"), nullable=True,
        default=None,
    )
    fallback_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    provider: Mapped[str] = mapped_column(String(16))
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    queries_executed: Mapped[int] = mapped_column(Integer, default=0)
    results_considered: Mapped[int] = mapped_column(Integer, default=0)
    results_accepted: Mapped[int] = mapped_column(Integer, default=0)
    suppressed_suspicious: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SearchEvidenceRow(Base):
    """One search result CONSIDERED for one event's research run (accepted or
    rejected — plan §4). Mirrors migrations/030_web_research.sql EXACTLY
    (mirror rule).

    RAW AND SAFE TEXT BOTH LIVE HERE. ``title``/``snippet`` are the
    provider's verbatim words — UNTRUSTED third-party text kept for
    display/provenance; ``safe_title``/``safe_snippet`` are the
    ``sanitize_for_llm`` outputs and are the ONLY forms the evidence bundle
    may hand the model. ``suspicious_instruction`` marks injection-shaped
    text: flagged rows stay visible in diagnostics and are excluded from
    model-facing text (the §81 news discipline, applied to the web).

    ``published_at`` is NULLABLE and never faked (§44 rule 18): a provider
    that stated no publication time produced a row the as-of gate treats
    conservatively, with the exclusion counted — never coerced to
    ``retrieved_at``, which is the platform's fetch clock, not the
    document's.

    REJECTED CANDIDATES ARE STORED TOO (``accepted=False`` +
    ``reject_reason``): "what the platform refused to admit and why" is the
    Evidence tab's transparency promise, and it is how tests prove the
    as-of/dedup gates fired rather than merely observing an absence.

    ``evidence_key`` is the STABLE citation id (derived from
    ``canonical_url``, e.g. ``web:3f9c2ab1e4d0``) that the LLM's
    ``evidence_refs`` cite and the validator resolves — derived, not the
    surrogate ``id``, so the same document cites identically across runs.
    """

    __tablename__ = "search_evidence"
    __table_args__ = (
        UniqueConstraint("run_id", "canonical_url", name="uq_search_evidence_run_url"),
        Index("ix_search_evidence_event", "event_id", "accepted"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("event_search_runs.id", ondelete="CASCADE")
    )
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE")
    )
    evidence_key: Mapped[str] = mapped_column(String(64))
    query: Mapped[str] = mapped_column(Text, default="")
    purpose: Mapped[str] = mapped_column(String(64), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    safe_title: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    domain: Mapped[str] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    snippet: Mapped[str] = mapped_column(Text, default="")
    safe_snippet: Mapped[str] = mapped_column(Text, default="")
    suspicious_instruction: Mapped[bool] = mapped_column(Boolean, default=False)
    source_tier: Mapped[str | None] = mapped_column(
        String(24), nullable=True, default=None
    )
    topic: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    relevance: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    result_type: Mapped[str] = mapped_column(String(8))
    provider: Mapped[str] = mapped_column(String(16))
    accepted: Mapped[bool] = mapped_column(Boolean)
    reject_reason: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PredictionMarketRow(Base):
    """One prediction market's METADATA, provider-independent (Catalyst
    research upgrade, plan §4; Phases 3/18). Mirrors
    migrations/031_prediction_markets.sql EXACTLY (mirror rule).

    KEYED ``UNIQUE(provider, provider_market_id)`` so a Kalshi ticker can
    never collide with a Polymarket condition id and adding KalshiProvider
    is new ROWS, not new columns. ``provider_event_id`` is the PROVIDER'S
    own grouping — it is NOT ``events.id`` and deliberately carries no FK:
    the explicit match table (:class:`EventPredictionMarketRow`) owns that
    association, exactly as option bars carry no event FK.

    READ-ONLY SUBSYSTEM: no column here (or in the snapshot/history/match
    tables) can hold an order, wallet, credential or position — a column
    that could hold one is a column someone eventually fills.
    """

    __tablename__ = "prediction_markets"
    __table_args__ = (
        UniqueConstraint(
            "provider", "provider_market_id",
            name="uq_prediction_markets_provider_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(16))
    provider_market_id: Mapped[str] = mapped_column(String(128))
    provider_event_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None
    )
    question: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    outcomes: Mapped[list] = mapped_column(JSON, default=list)
    resolution_criteria: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    end_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    market_status: Mapped[str] = mapped_column(String(16))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict] = mapped_column(JSON, default=dict)


class PredictionMarketSnapshotRow(Base):
    """One market's PRICING at one observed instant — append-only (plan §4).
    Mirrors migrations/031_prediction_markets.sql EXACTLY (mirror rule).

    ``observed_at`` is when THIS platform saw the prices — the point-in-time
    identity every as-of read filters on. A later observation never
    overwrites an earlier one (``UNIQUE(market_id, observed_at)``): "what
    was the market pricing when the analysis ran" must survive the market
    moving on.

    EVERY LIQUIDITY-ADJACENT COLUMN IS NULLABLE (§44 rule 18): a market
    with unreported depth is NOT a market with zero depth. The
    interpretation layer weights price confidence BY liquidity, so zeroing
    an absence here would silently upgrade the thinnest markets to
    "confidently priced at zero depth" — the most dangerous fabrication
    this schema could hold.
    """

    __tablename__ = "prediction_market_snapshots"
    __table_args__ = (
        # The unique constraint's backing index already serves the read seam's
        # (market_id, observed_at) range query — a separate index would be an
        # exact duplicate maintained twice.
        UniqueConstraint(
            "market_id", "observed_at", name="uq_prediction_market_snapshots"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("prediction_markets.id", ondelete="CASCADE")
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    outcome_prices: Mapped[dict] = mapped_column(JSON, default=dict)
    best_bid: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    best_ask: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    midpoint: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    last_trade_price: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    volume: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    liquidity: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    open_interest: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    provider: Mapped[str] = mapped_column(String(16))


class PredictionMarketPricePointRow(Base):
    """One dated price for ONE outcome of one market (plan §4). Mirrors
    migrations/031_prediction_markets.sql EXACTLY (mirror rule).

    NATURAL COMPOSITE PK ``(market_id, outcome, ts)`` like
    :class:`StockBar1mRow`: one outcome's price at one instant is ONE fact,
    so a refetch can only overwrite, never duplicate — backfill idempotence
    as a database property (ADR-007). Points are provider truth verbatim;
    no interpolated row is ever written (plan §3).
    """

    __tablename__ = "prediction_market_history"

    market_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prediction_markets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    outcome: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    price: Mapped[float] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(16))


class EventPredictionMarketRow(Base):
    """One event↔market MATCH decision under one as-of (plan Phase 4).
    Mirrors migrations/031_prediction_markets.sql EXACTLY (mirror rule).

    THE MATCH IS A CLAIM THIS PLATFORM MAKES, so it carries its own
    provenance: ``relation`` (DIRECT | DERIVED | CONTEXT — vocab in code),
    a deterministic ``relevance`` score, the ``reason`` in words,
    ``matched_by`` naming the matcher version (DETERMINISTIC_V1 today, so a
    future LLM-assisted classifier's rows stay distinguishable), and
    rejected candidates kept with ``reject_reason`` — the LLM can never
    cite a market that has no accepted row here.

    ``UNIQUE(event_id, market_id, as_of)`` lets the match be re-decided
    under a later as-of without rewriting history.
    """

    __tablename__ = "event_prediction_markets"
    __table_args__ = (
        UniqueConstraint(
            "event_id", "market_id", "as_of", name="uq_event_prediction_markets"
        ),
        Index("ix_event_prediction_markets_event", "event_id", "accepted"),
        Index("ix_event_prediction_markets_series", "event_id", "series_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE")
    )
    market_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("prediction_markets.id", ondelete="CASCADE")
    )
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    relation: Mapped[str] = mapped_column(String(16))
    relevance: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, default="")
    ambiguity: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    matched_by: Mapped[str] = mapped_column(String(32))
    accepted: Mapped[bool] = mapped_column(Boolean)
    reject_reason: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    #: Which BRACKET SERIES this contract belongs to (migration 032). Venues
    #: publish a distribution as one contract per range; siblings share this
    #: key so a partial distribution can be recognised as partial.
    series_key: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )
    #: True when the accept cap cut this contract's series in half. A partial
    #: distribution must be LABELLED, never drawn as if it were whole.
    series_truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_settings = get_settings()
engine = create_async_engine(_settings.database_url, echo=False)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class SchemaDriftError(RuntimeError):
    """The live database does not match the ORM. Startup refuses to continue."""


def _schema_drift(inspector, *, dialect: str) -> list[str]:
    """Every way the live schema differs from :data:`Base.metadata`.

    Compares TABLES and COLUMNS only. Indexes, constraint names and types are
    deliberately out of scope: they vary legitimately between Postgres and
    SQLite and across Timescale's hypertable rewriting, and a checker that
    cries wolf is a checker people disable.
    """
    problems: list[str] = []
    live_tables = set(inspector.get_table_names())
    for table_name, table in Base.metadata.tables.items():
        if table_name not in live_tables:
            problems.append(f"missing table: {table_name}")
            continue
        live_columns = {c["name"] for c in inspector.get_columns(table_name)}
        for column in table.columns:
            if column.name not in live_columns:
                problems.append(f"missing column: {table_name}.{column.name}")
    return problems


async def init_db(*, create: bool = False) -> None:
    """VERIFY the live schema against the ORM — do not silently create it.

    WHY THIS IS NOT ``create_all`` ANY MORE. It used to be, and that quietly
    admitted two different failures:

    - A NEW TABLE in the ORM with no migration was auto-created at startup, so
      everything worked here and nowhere else. The divergence only surfaced
      when somebody built a database from ``migrations/`` and the table was
      absent — long after the change, and far from its author.
    - A NEW COLUMN on an EXISTING table was NOT created (``create_all`` never
      alters a live table), so the process started happily and then 500'd on
      the first query that touched it.

    ``tests/test_migration_parity.py`` already pins the ORM against the SQL
    FILES. This closes the remaining gap: whether the SQL was ever APPLIED to
    the database this process is about to serve. Refusing to boot is the point
    — a wrong schema discovered at startup costs a restart, whereas the same
    schema discovered at request time can corrupt a trading decision.

    ``create=True`` is for test fixtures and a genuinely empty dev database.
    """
    async with engine.begin() as conn:
        if create:
            await conn.run_sync(Base.metadata.create_all)
            return
        problems = await conn.run_sync(
            lambda sync_conn: _schema_drift(
                inspect(sync_conn), dialect=sync_conn.dialect.name
            )
        )
    if problems:
        raise SchemaDriftError(
            "The database schema does not match the ORM. Apply the pending "
            "migration(s) from migrations/ and restart.\n  "
            + "\n  ".join(sorted(problems))
        )


async def get_session():
    async with SessionLocal() as session:
        yield session
