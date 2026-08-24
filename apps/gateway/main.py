"""API gateway — the only backend surface the front end talks to.

V1 runs as a modular monolith: watchlist, trading pool, and audit modules live
in one process but keep service-shaped boundaries (own routers + service logic)
so they can be split into standalone containers later without API changes.

Observability (plan §41): every request gets an X-Request-ID (honored when the
client sends one) bound to :data:`libs.common.telemetry.request_id_var` for
the request's duration, one structured log line on completion, and
counter/histogram samples labeled by the ROUTE TEMPLATE — never the concrete
URL — so label cardinality stays bounded. ``GET /metrics`` renders the whole
in-process registry in Prometheus text format.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.common.logging import setup_logging
from libs.common.telemetry import REGISTRY, request_id_var

from . import (
    event_calendar,
    market_stream,
    monitor,
    order_sync,
    risk_snapshot,
    runtime_config,
)
from .db import (
    SessionLocal,
    StockBarDaily,
    WatchlistItem,
    get_session,
    init_db,
)
from .routers import (
    alerts,
    analysis,
    audit_log,
    backtests,
    broker,
    config,
    events,
    health,
    market,
    options,
    orders,
    plans,
    portfolio,
    positions,
    recommendations,
    risk,
    trading_control,
    trading_pool,
    watchlist,
    income,
)

# ---------------------------------------------------------------------------
# Metrics (plan §41). Registered once at import; the middleware and /metrics
# endpoint below only ever record into / render this registry.
# ---------------------------------------------------------------------------
_PROCESS_START_MONOTONIC = time.monotonic()

#: The WIRE CONTRACT's version, sent as ``X-API-Version`` on every response.
#:
#: Not the application's version — the shape of the JSON. Bump the minor when
#: a field is ADDED (old clients keep working) and the major when one is
#: removed, renamed or changes meaning (old clients break, and should be told
#: so rather than left rendering an empty panel).
#:
#: This exists because the frontend ships from its own repository and updates
#: independently. A field that silently disappears reads as ``undefined`` in
#: TypeScript — the type checker cannot catch it, so the symptom of a version
#: mismatch is a blank panel rather than an error. A client that compares this
#: header against what it was built for can say what actually went wrong.
API_VERSION = "1.0"

HTTP_REQUESTS_TOTAL = REGISTRY.counter(
    "http_requests_total",
    "Total HTTP requests handled, labeled by method, route template and status (plan §41).",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION_MS = REGISTRY.histogram(
    "http_request_duration_ms",
    "HTTP request latency in milliseconds, labeled by method and route template (plan §41).",
    ("method", "path"),
)
PROCESS_UPTIME_SECONDS = REGISTRY.gauge(
    "process_uptime_seconds",
    "Seconds since this gateway process imported its main module (plan §41).",
)
PROCESS_UPTIME_SECONDS.set_callback(
    lambda: time.monotonic() - _PROCESS_START_MONOTONIC
)
WATCHLIST_BARS_MAX_AGE_DAYS = REGISTRY.gauge(
    "watchlist_bars_max_age_days",
    "Max age in days of the NEWEST stored daily bar across watchlist tickers "
    "(data freshness, plan §41); 0 when no watchlist ticker has stored bars.",
)
OPTION_CHAIN_AGE_SECONDS = REGISTRY.gauge(
    "option_chain_age_seconds",
    "Age of the served option chain snapshot. Fixed at 0 because the stub "
    "provider GENERATES chains on demand at request time — there is no stored "
    "snapshot to go stale yet; a real feed must replace this with true "
    "snapshot age (plan §41 honesty).",
)
OPTION_CHAIN_AGE_SECONDS.set(0.0)

# One structured line per completed request (plan §41), via the shared JSON
# formatter — request_id makes it joinable with audit rows (§38).
_http_logger = logging.getLogger("http")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # SCHEMA IS VERIFIED, NOT CREATED (see db.init_db). A mismatch raises
    # SchemaDriftError and the process refuses to start: a wrong schema found
    # at startup costs a restart, the same schema found at request time can
    # corrupt a trading decision.
    #
    # DB_AUTO_CREATE=true restores the old create-if-absent behaviour for a
    # genuinely empty development database. It is deliberately opt-in and
    # deliberately not the default — that default is what let a table exist
    # here and nowhere else.
    await init_db(create=get_settings().db_auto_create)
    # UI-managed provider settings (runtime_config table) override .env —
    # loaded before anything reads Settings, so the first request already
    # sees what the user configured in the UI last session.
    async with SessionLocal() as _rc_session:
        await runtime_config.apply_overrides(_rc_session)
    # Automated position monitor (plan §26): a background task, started only
    # when the configured interval is > 0 (0 disables it). NOTE: test suites
    # drive the app through httpx ASGITransport, which does NOT run lifespan —
    # the monitor task never starts under tests, GET /api/positions/monitor
    # honestly reports enabled=false there, and the sweep core is tested
    # directly (tests/test_position_monitor_auto.py).
    background_tasks: list[asyncio.Task] = []
    settings = get_settings()
    if settings.position_monitor_interval_seconds > 0:
        background_tasks.append(asyncio.create_task(monitor.monitor_loop()))
    # Order-sync sweep (guide §11 Iteration C): settles non-terminal broker
    # orders against the broker's own state. Cheap no-op ticks when no real
    # broker is configured.
    if settings.order_sync_interval_seconds > 0:
        background_tasks.append(asyncio.create_task(order_sync.order_sync_loop()))
    # Periodic reconciliation (guide §13 Iteration D): compares whole ledgers
    # on a cadence; a material mismatch pauses trading via the §18 kill
    # switch — the same code path as GET /api/broker/reconcile.
    if settings.reconciliation_interval_seconds > 0:
        background_tasks.append(asyncio.create_task(reconciliation_loop()))
    # Statistical risk snapshot writer (Risk Engine Upgrade Phase B, contract
    # §6): persists ONE SCHEDULED snapshot per America/New_York day — the NAV
    # series live drawdown is measured on. SHADOW; 0 disables it. Same
    # lifespan caveat as the monitor: tests drive
    # risk_snapshot.run_scheduled_snapshot() directly.
    if settings.risk_snapshot_interval_seconds > 0:
        background_tasks.append(asyncio.create_task(risk_snapshot.risk_snapshot_loop()))
    # Event calendar ingestion (Catalyst & Event Intelligence Phase B, event
    # spec §8): fetches the typed event registry from SEC EDGAR, the Federal
    # Reserve and the exchange calendars, and fires the §11 T-minus alert
    # exactly once per (event, horizon). 0 disables the background task; POST
    # /api/events/refresh always works. Same lifespan caveat as the monitor:
    # tests drive event_calendar.run_calendar_ingest() directly.
    if settings.event_calendar_interval_seconds > 0:
        background_tasks.append(
            asyncio.create_task(event_calendar.event_calendar_loop())
        )
    # Alpaca market data stream (data_source.md §5): replaces per-poll REST
    # snapshot calls with ONE websocket. The supervisor self-disables when
    # the configured provider is not "alpaca" (checked every cycle, so
    # runtime provider switches take effect without a restart).
    background_tasks.append(asyncio.create_task(market_stream.market_stream_loop()))
    try:
        yield
    finally:
        # Graceful shutdown (§26): cancel and AWAIT each task so an in-flight
        # sweep's cancellation fully unwinds before the process exits.
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


async def reconciliation_loop() -> None:
    """Sleep -> reconcile forever (guide §13 Iteration D).

    Skips the comparison when there is no real broker (unset or simulated) —
    an absent ledger is not a disagreement between ledgers. On a material
    mismatch :func:`routers.broker.run_reconciliation` itself pauses trading
    and writes the KILL_SWITCH_TRIGGERED audit; this loop only logs the
    outcome. Transient faults are logged and retried next tick — the loop
    must outlive any single failure.
    """
    from .deps import broker_configured, simulated_broker_mode
    from .routers.broker import run_reconciliation

    interval = get_settings().reconciliation_interval_seconds
    logger = logging.getLogger("apps.gateway.reconciliation")
    logger.info(
        "reconciliation_loop_started",
        extra={"extra_fields": {"interval_seconds": interval}},
    )
    try:
        while True:
            await asyncio.sleep(interval)
            if not broker_configured() or simulated_broker_mode():
                continue
            try:
                async with SessionLocal() as session:
                    result = await run_reconciliation(session)
                if result.get("mismatches"):
                    logger.error(
                        "periodic_reconciliation_mismatch",
                        extra={
                            "extra_fields": {
                                "mismatches": result["mismatches"],
                                "paused": result.get("paused", False),
                            }
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("periodic_reconciliation_failed")
    except asyncio.CancelledError:
        logger.info("reconciliation_loop_stopped")
        raise


async def _refresh_freshness_gauges(session: AsyncSession) -> None:
    """Recompute scrape-time data-freshness gauges (plan §41).

    ``watchlist_bars_max_age_days`` is the max over watchlist tickers of
    (today UTC - newest stored bar date). Tickers with no stored bars yet
    contribute nothing (they have no age to report — honest absence, not a
    guess); with no watchlist or no stored bars at all, the gauge reads 0.
    """
    tickers = (await session.execute(select(WatchlistItem.ticker))).scalars().all()
    max_age = 0.0
    if tickers:
        rows = await session.execute(
            select(StockBarDaily.ticker, func.max(StockBarDaily.ts))
            .where(StockBarDaily.ticker.in_(tickers))
            .group_by(StockBarDaily.ticker)
        )
        today = datetime.now(timezone.utc).date()
        ages = [
            float((today - newest).days) for _, newest in rows.all() if newest is not None
        ]
        if ages:
            max_age = max(ages)
    WATCHLIST_BARS_MAX_AGE_DAYS.set(max_age)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        # LOCALHOST ONLY. The gateway ships with no authentication layer, so
        # the browser origins it trusts are the local dev servers and nothing
        # else. 3001 is included because 3000 is a popular default that is
        # often already taken. Widening this — especially to "*" — would let
        # any page on the internet drive an authenticated-by-nothing API that
        # can place orders; put real auth in front of it first.
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_and_metrics(request: Request, call_next):
        """Request-ID + one log line + metrics per request (plan §41, §38).

        Honors an incoming ``X-Request-ID`` (so a client/browser can stitch
        its own traces to ours), else generates a uuid4 hex; the ID is bound
        to :data:`request_id_var` for the request's duration — audit rows
        written inside pick it up as their ``correlation_id`` — and echoed
        back in the ``X-Request-ID`` response header. Metric labels use the
        ROUTE TEMPLATE (e.g. ``/api/watchlist/{ticker}/analysis``), never the
        concrete URL, to keep label cardinality bounded (§41); ``/metrics``
        itself is excluded from the metrics so scrapes don't count themselves.
        """
        incoming = request.headers.get("X-Request-ID", "").strip()
        request_id = incoming or uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500  # what the outer ServerErrorMiddleware returns on a raise
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            # THE WIRE CONTRACT'S VERSION. The frontend ships in its own
            # repository and updates independently, so the two halves can
            # legitimately be at different versions — and without a marker the
            # symptom of a mismatch is an empty panel rather than an error,
            # because TypeScript cannot catch a field that is simply absent
            # from the JSON. A client that reads this can say "the API moved"
            # instead of rendering a blank.
            response.headers["X-API-Version"] = API_VERSION
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            # Route template when routing matched; raw path otherwise (404s).
            route = request.scope.get("route")
            path_template = getattr(route, "path", None) or request.url.path
            _http_logger.info(
                "http_request",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": path_template,
                        "status": status,
                        "duration_ms": round(duration_ms, 3),
                    }
                },
            )
            if path_template != "/metrics":
                HTTP_REQUESTS_TOTAL.inc(
                    method=request.method, path=path_template, status=str(status)
                )
                HTTP_REQUEST_DURATION_MS.observe(
                    duration_ms, method=request.method, path=path_template
                )
            request_id_var.reset(token)

    @app.get("/metrics", include_in_schema=False)
    async def metrics(session: AsyncSession = Depends(get_session)) -> PlainTextResponse:
        """Prometheus text exposition of the in-process registry (plan §41).

        Freshness gauges are recomputed at scrape time so the exported values
        are current, not cached from request traffic.
        """
        await _refresh_freshness_gauges(session)
        return PlainTextResponse(
            REGISTRY.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    app.include_router(health.router)
    app.include_router(watchlist.router)
    app.include_router(analysis.router)
    app.include_router(options.router)
    app.include_router(backtests.router)
    app.include_router(income.router)
    app.include_router(trading_pool.router)
    app.include_router(market.router)
    app.include_router(trading_control.router)
    app.include_router(portfolio.router)
    # Phase D (design §8.5): POST /api/risk/stress/run — a user-defined
    # hypothetical over the current book. SHADOW, read-only, no audit event.
    app.include_router(risk.router)
    app.include_router(orders.router)
    app.include_router(plans.router)
    app.include_router(positions.router)
    app.include_router(recommendations.router)
    app.include_router(audit_log.router)
    app.include_router(alerts.router)
    app.include_router(events.router)
    app.include_router(broker.router)
    app.include_router(config.router)
    return app


app = create_app()
