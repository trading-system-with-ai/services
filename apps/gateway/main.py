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

from .db import StockBarDaily, WatchlistItem, get_session, init_db
from .routers import (
    alerts,
    analysis,
    audit_log,
    backtests,
    config,
    health,
    market,
    options,
    orders,
    portfolio,
    positions,
    recommendations,
    trading_control,
    trading_pool,
    watchlist,
)

# ---------------------------------------------------------------------------
# Metrics (plan §41). Registered once at import; the middleware and /metrics
# endpoint below only ever record into / render this registry.
# ---------------------------------------------------------------------------
_PROCESS_START_MONOTONIC = time.monotonic()

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
    await init_db()
    yield


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
        allow_origins=["http://localhost:3000"],
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
    app.include_router(trading_pool.router)
    app.include_router(market.router)
    app.include_router(trading_control.router)
    app.include_router(portfolio.router)
    app.include_router(orders.router)
    app.include_router(positions.router)
    app.include_router(recommendations.router)
    app.include_router(audit_log.router)
    app.include_router(alerts.router)
    app.include_router(config.router)
    return app


app = create_app()
