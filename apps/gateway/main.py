"""API gateway — the only backend surface the front end talks to.

V1 runs as a modular monolith: watchlist, trading pool, and audit modules live
in one process but keep service-shaped boundaries (own routers + service logic)
so they can be split into standalone containers later without API changes.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from libs.common.config import get_settings
from libs.common.logging import setup_logging

from .db import init_db
from .routers import (
    analysis,
    audit_log,
    backtests,
    health,
    market,
    trading_control,
    trading_pool,
    watchlist,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(watchlist.router)
    app.include_router(analysis.router)
    app.include_router(backtests.router)
    app.include_router(trading_pool.router)
    app.include_router(market.router)
    app.include_router(trading_control.router)
    app.include_router(audit_log.router)
    return app


app = create_app()
