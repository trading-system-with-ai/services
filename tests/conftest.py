"""Shared test fixtures.

TWO CLIENTS, TWO WORLDS — the split matters, so it is spelled out here:

- ``client`` runs with BOTH data providers explicitly set to "stub" and the
  execution venue set to ``BROKER_PROVIDER=simulated``. All three are opt-in
  development/test values (never defaults — see libs/common/config.py and each
  stub's module docstring); setting them here is what lets the bulk of the
  suite exercise the full pipeline end to end with deterministic data.
- ``unconfigured_client`` runs with ALL THREE UNSET, which is the state a fresh
  install starts in. tests/test_no_synthetic_data.py uses it to prove the
  platform shows NOTHING rather than synthetic numbers when Massive is not
  configured.

WHY THE FIXTURE OWNS ``BROKER_PROVIDER=simulated`` (and not the call sites):
execution now has no default venue either. With ``BROKER_PROVIDER`` unset,
POST /api/orders/approve and POST /api/orders/close answer 503
BROKER_NOT_CONFIGURED and the exit sweep skips — an unconfigured install places
nothing and never falls back to the internal fill simulator. Every existing
test that approves, closes or sweeps therefore needs to opt INTO the simulator
explicitly, and it does so here, once, exactly the way MARKET_DATA_PROVIDER=stub
is opted into. Under ``simulated`` the internal §11 paper fill model runs
byte-identically to before the broker existed, so those tests' hand-computed
fill arithmetic is unchanged. A test that wants the REAL broker path overrides
the variable itself (see tests/test_broker_execution.py).

All fixtures set the environment AND clear the ``get_settings`` lru_cache
before and after the test, so a cached Settings object can never leak the wrong
provider configuration into the next test.
"""
import os

# Must be set before any app import so the engine binds to the test database.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"

from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from apps.gateway.db import Base, engine
from apps.gateway.main import app
from libs.common.config import get_settings

# Environment variables this module owns. Saved and restored around every test
# so provider configuration never leaks between them.
_PROVIDER_ENV_VARS = ("MARKET_DATA_PROVIDER", "LLM_PROVIDER", "BROKER_PROVIDER")


def _apply_provider_env(values: dict[str, str | None]) -> dict[str, str | None]:
    """Set/unset the provider env vars, clear the settings cache, return the
    previous values so the caller can restore them."""
    previous = {name: os.environ.get(name) for name in _PROVIDER_ENV_VARS}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    get_settings.cache_clear()
    return previous


@asynccontextmanager
async def _client_with_providers(values: dict[str, str | None]):
    """A fresh-schema AsyncClient bound to the app with `values` configured."""
    previous = _apply_provider_env(values)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        _apply_provider_env(previous)


@pytest.fixture
async def client():
    """The full-pipeline client: data providers "stub", execution "simulated".

    None of these three are defaults (an unconfigured install serves nothing
    and places nothing); this fixture opts into all of them so the suite can
    exercise every code path against deterministic, reproducible synthetic
    data and internal fills. See the module docstring for why the broker opt-in
    lives here rather than at the call sites.
    """
    async with _client_with_providers(
        {
            "MARKET_DATA_PROVIDER": "stub",
            "LLM_PROVIDER": "stub",
            "BROKER_PROVIDER": "simulated",
        }
    ) as c:
        yield c


@pytest.fixture
async def unconfigured_client():
    """A client with NO market data, NO LLM and NO broker — a fresh install.

    Used by tests/test_no_synthetic_data.py to prove the core guarantee: with
    Massive unconfigured, every market-facing endpoint 503s and no synthetic
    price, bar, chain, greek or recommendation reaches the response body — and
    by tests/test_broker_execution.py for its execution counterpart: with no
    broker, nothing is placed and nothing is closed.

    All three variables are set to the EMPTY STRING rather than deleted: an
    environment variable overrides ``.env``, so this pins the unconfigured
    state even on a developer machine whose real ``.env`` names a provider.
    Deleting them would let the local ``.env`` leak in and quietly turn this
    fixture into a configured one.
    """
    async with _client_with_providers(
        {"MARKET_DATA_PROVIDER": "", "LLM_PROVIDER": "", "BROKER_PROVIDER": ""}
    ) as c:
        yield c
