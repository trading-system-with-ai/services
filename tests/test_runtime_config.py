"""UI-managed provider configuration (runtime_config layer).

What these tests pin:

1. PUT /api/config/providers changes take effect IMMEDIATELY — environment +
   Settings cache rebuilt, provider status flips, no restart.
2. Secrets are write-only: no API response ever contains a stored secret
   value, and the CONFIG_CHANGED audit records changed KEYS only.
3. Empty string explicitly disconnects; invalid provider names 422.
4. Stored rows override env at startup (apply_overrides).
5. First-connect cash adoption: connecting a real broker with a completely
   EMPTY local ledger adopts the broker's cash as the local baseline
   (audited); with any local activity it never fires.
"""
import json
import os

import httpx
import pytest

from libs.common.config import get_settings

from apps.gateway import runtime_config
from apps.gateway.db import Order, Portfolio, RuntimeConfig, SessionLocal

from tests.test_broker_execution import FakeAlpaca, _broker_client, order_body, rows

pytestmark = pytest.mark.anyio

ALL_ENV = tuple(runtime_config.CONFIG_KEYS.values())


@pytest.fixture(autouse=True)
def preserve_provider_env():
    """PUTs write os.environ; every test must leave it exactly as found."""
    before = {name: os.environ.get(name) for name in ALL_ENV}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    get_settings.cache_clear()


def _walk_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


async def test_put_connects_providers_immediately(unconfigured_client):
    client = unconfigured_client
    before = (await client.get("/api/config/providers")).json()
    assert before["market_data"]["configured"] is False
    assert before["llm"]["configured"] is False
    assert before["broker"]["configured"] is False

    r = await client.put(
        "/api/config/providers",
        json={"market_data_provider": "stub", "llm_provider": "stub"},
    )
    assert r.status_code == 200
    after = r.json()
    assert after["market_data"]["configured"] is True
    assert after["llm"]["configured"] is True
    assert after["broker"]["configured"] is False  # untouched

    # The change is live for OTHER surfaces too, no restart: market overview
    # now answers 200 instead of the 503 the unconfigured client would get.
    assert (await client.get("/api/market/overview")).status_code == 200


async def test_secret_values_never_appear_in_any_response(unconfigured_client):
    client = unconfigured_client
    secret = "sk-THIS-MUST-NEVER-LEAVE-THE-SERVER"
    r = await client.put(
        "/api/config/providers",
        json={"llm_provider": "stub", "llm_api_key": secret},
    )
    assert r.status_code == 200
    assert secret not in json.dumps(r.json())
    assert r.json()["secrets_set"]["llm_api_key"] is True

    for path in ("/api/config/providers", "/api/config"):
        body = (await client.get(path)).json()
        assert all(secret not in s for s in _walk_strings(body))

    # Audit records the changed KEYS, never the value.
    events = (await client.get("/api/audit")).json()
    config_events = [e for e in events if e["action"] == "CONFIG_CHANGED"]
    assert config_events, "CONFIG_CHANGED must be audited"
    assert "llm_api_key" in config_events[0]["details"]["changed_keys"]
    assert all(secret not in s for e in config_events for s in _walk_strings(e))


async def test_empty_string_disconnects(unconfigured_client):
    client = unconfigured_client
    await client.put("/api/config/providers", json={"market_data_provider": "stub"})
    assert (await client.get("/api/config/providers")).json()["market_data"][
        "configured"
    ] is True

    r = await client.put("/api/config/providers", json={"market_data_provider": ""})
    assert r.status_code == 200
    assert r.json()["market_data"]["configured"] is False
    # Downstream surfaces honestly 503 again.
    assert (await client.get("/api/market/overview")).status_code == 503


async def test_invalid_provider_name_422(unconfigured_client):
    r = await unconfigured_client.put(
        "/api/config/providers", json={"broker_provider": "robinhood"}
    )
    assert r.status_code == 422


async def test_alpaca_is_an_accepted_market_data_provider(unconfigured_client):
    """data_source.md §1: Alpaca is the authoritative market-data source —
    selectable via the UI config layer; it resolves off the broker's key
    pair, so with keys stored the connection reports configured."""
    r = await unconfigured_client.put(
        "/api/config/providers",
        json={
            "market_data_provider": "alpaca",
            "alpaca_api_key_id": "test-key",
            "alpaca_api_secret_key": "test-secret",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["market_data"]["provider"] == "alpaca"
    assert body["market_data"]["configured"] is True


async def test_stored_rows_override_env_at_startup(unconfigured_client):
    """apply_overrides (the lifespan hook) loads DB rows over the env."""
    async with SessionLocal() as s:
        s.add(RuntimeConfig(key="market_data_provider", value="stub"))
        await s.commit()
        applied = await runtime_config.apply_overrides(s)
    assert applied == 1
    assert get_settings().market_data_provider == "stub"


async def test_broker_cash_is_live_and_never_copied_locally(monkeypatch):
    """THE PLATFORM STORES NO COPY OF THE ACCOUNT: connecting a broker makes
    /api/portfolio/risk report the broker's LIVE cash, no Portfolio row is
    ever created in broker mode, and a broker-side change shows up on the
    very next read."""
    fake = FakeAlpaca(order_body())
    fake.account_payload = dict(fake.account_payload, cash="123456.78")
    async with _broker_client(fake, monkeypatch) as client:
        r = await client.get("/api/portfolio/risk")
        assert r.json()["cash"] == pytest.approx(123456.78)
        assert await rows(Portfolio) == []  # zero copies, ever

        # The broker's number moves -> the platform's next read moves with it.
        fake.account_payload = dict(fake.account_payload, cash="99111.22")
        r = await client.get("/api/portfolio/risk")
        assert r.json()["cash"] == pytest.approx(99111.22)
        assert await rows(Portfolio) == []


async def test_disconnecting_the_broker_nulls_the_account(monkeypatch):
    """Disconnect -> no venue -> no account: every account number is null
    (never a stale copy of the last broker read)."""
    fake = FakeAlpaca(order_body())
    async with _broker_client(fake, monkeypatch) as client:
        assert (await client.get("/api/portfolio/risk")).json()["cash"] is not None

        r = await client.put("/api/config/providers", json={"broker_provider": ""})
        assert r.status_code == 200
        body = (await client.get("/api/portfolio/risk")).json()
        assert body["cash"] is None and body["nav"] is None
        assert body["venue"]["configured"] is False


async def test_llm_output_language_round_trip_and_validation(unconfigured_client):
    """Settings.llm_output_language rides the same runtime-config layer:
    "zh" applies immediately and is reported by GET /providers; an unknown
    language is a 422, and "" restores the default (reported as "en")."""
    client = unconfigured_client
    # Default: reported as "en" even while everything else is unconfigured.
    assert (await client.get("/api/config/providers")).json()["llm"][
        "output_language"
    ] == "en"

    r = await client.put(
        "/api/config/providers", json={"llm_output_language": "zh"}
    )
    assert r.status_code == 200
    assert r.json()["llm"]["output_language"] == "zh"

    # Unknown language: enum-gated like provider names, never stored.
    r = await client.put(
        "/api/config/providers", json={"llm_output_language": "fr"}
    )
    assert r.status_code == 422

    # Empty string = back to default.
    r = await client.put("/api/config/providers", json={"llm_output_language": ""})
    assert r.status_code == 200
    assert r.json()["llm"]["output_language"] == "en"


async def test_instrument_permissions_round_trip_and_effects(client):
    """User-level instrument permissions (user mandate 2026-08-17) ride the
    runtime-config layer and take effect EVERYWHERE at once: GET /api/config
    account_permissions, the §10 gate chain (same factory), and the backtest
    gate. Strict true/false — "" would make the pydantic bool field (and so
    the whole Settings object) unconstructable, so it 422s."""
    # Defaults: the three real flags on.
    perms = (await client.get("/api/config")).json()["account_permissions"]
    assert (perms["long_stock"], perms["long_call"], perms["long_put"]) == (
        True,
        True,
        True,
    )

    # Disable long calls -> reflected immediately.
    r = await client.put(
        "/api/config/providers", json={"allow_long_call": "false"}
    )
    assert r.status_code == 200
    perms = (await client.get("/api/config")).json()["account_permissions"]
    assert perms["long_call"] is False
    assert perms["long_stock"] is True

    # Disable long stock -> the V1 backtest honestly refuses (stock-only).
    r = await client.put(
        "/api/config/providers", json={"allow_long_stock": "false"}
    )
    assert r.status_code == 200
    await client.post("/api/watchlist", json={"ticker": "NVDA"})
    r = await client.post("/api/backtests", json={"ticker": "NVDA"})
    assert r.status_code == 422
    assert "allow_long_stock=false" in r.json()["detail"]

    # Strict values: empty string and junk are refused.
    for bad in ("", "yes", "1"):
        r = await client.put(
            "/api/config/providers", json={"allow_long_put": bad}
        )
        assert r.status_code == 422, bad

    # Re-enable -> backtest runs again.
    r = await client.put(
        "/api/config/providers",
        json={"allow_long_stock": "true", "allow_long_call": "true"},
    )
    assert r.status_code == 200
    r = await client.post("/api/backtests", json={"ticker": "NVDA"})
    assert r.status_code == 200


async def test_forbidden_permissions_are_not_settable(unconfigured_client):
    """§33 after the Phase 3 unlock: ONLY the naked shorts remain outside
    the config layer, forever (broker refusal + §4 charter) — unknown
    fields are silently ignored by the request model, and the stored state
    never gains such a key. short_stock/margin are REAL toggles now and
    are exercised in test_short_stock_execution.py."""
    client = unconfigured_client
    r = await client.put(
        "/api/config/providers", json={"allow_naked_short_call": "true"}
    )
    # Unknown field -> ignored, nothing changes (200 echo of current state).
    assert r.status_code == 200
    perms = (await client.get("/api/config")).json()["account_permissions"]
    assert perms["naked_short_call"] is False


def test_config_keys_and_provider_api_cannot_drift():
    """Every runtime-config key must be REACHABLE through the providers API
    (a key in CONFIG_KEYS that ProviderConfigRequest lacks is silently
    un-settable — PUTs of it no-op), except the explicitly staged set below.
    The staged set must SHRINK, never grow silently: adding a CONFIG_KEYS
    entry without wiring the request model now fails here by name."""
    from apps.gateway.routers.config import ProviderConfigRequest

    # bea_api_key predates this pin and has no Settings-UI field yet (BEA
    # actuals are configured via .env; see the trading-platform memory note).
    staged = {"bea_api_key"}
    unreachable = set(runtime_config.CONFIG_KEYS) - set(
        ProviderConfigRequest.model_fields
    )
    assert unreachable == staged, (
        f"CONFIG_KEYS not settable through PUT /api/config/providers: "
        f"{sorted(unreachable - staged)} — wire them into "
        "ProviderConfigRequest or add them to the staged set with a reason"
    )
    # Reachable fields must not invent keys either: every request field is a
    # real runtime-config key.
    assert set(ProviderConfigRequest.model_fields) <= set(
        runtime_config.CONFIG_KEYS
    )


def test_secret_presence_booleans_cover_every_reachable_secret():
    """_providers_status()['secrets_set'] must report presence for every
    secret settable through the API — a secret the UI can store but whose
    StoredMark can never light up looks like a failed save."""
    from apps.gateway.routers.config import (
        ProviderConfigRequest,
        _providers_status,
    )

    reachable_secrets = runtime_config.SECRET_KEYS & set(
        ProviderConfigRequest.model_fields
    )
    assert set(_providers_status()["secrets_set"]) == reachable_secrets
