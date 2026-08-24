"""UI-managed provider configuration (the runtime layer over Settings).

The user configures data/LLM/broker providers in the Settings UI, not .env.
Stored rows (``runtime_config`` table) OVERRIDE environment variables: on
startup and after every change they are written into ``os.environ`` and the
cached Settings object is rebuilt, so every existing ``get_settings()`` call
site — providers, gates, sweeps, background loops — picks the new values up
without knowing this layer exists. .env remains the fallback for anything
never set through the UI (and for tests, which monkeypatch env directly).

SECURITY INVARIANTS:
- values are NEVER returned by any API (presence booleans only), never
  logged, never audited by value — CONFIG_CHANGED records changed KEYS;
- only the eight whitelisted keys below can be set — nothing else in the
  environment is reachable through this table;
- the Alpaca base URL is deliberately NOT configurable here: the paper-only
  host guard (libs/broker/alpaca.py) would refuse anything else anyway, and
  offering the field would only invite pointing a UI at a live endpoint.
"""
from __future__ import annotations

import logging
import os

from sqlalchemy import select

from libs.common.config import get_settings

from .db import RuntimeConfig, utcnow

logger = logging.getLogger(__name__)

#: API field name -> environment variable it overrides. The ONLY settable keys.
CONFIG_KEYS: dict[str, str] = {
    "market_data_provider": "MARKET_DATA_PROVIDER",
    "massive_api_key": "MASSIVE_API_KEY",
    "llm_provider": "LLM_PROVIDER",
    "llm_api_key": "LLM_API_KEY",
    "llm_model": "LLM_MODEL",
    "llm_output_language": "LLM_OUTPUT_LANGUAGE",
    "broker_provider": "BROKER_PROVIDER",
    # User-level instrument permissions (§5/§8): the three REAL flags only.
    # Forbidden capabilities (short/naked/margin/covered/CSP) have NO code
    # path (§33) and are deliberately NOT settable here; defined_risk_spreads
    # stays env-only until spread execution exists.
    "allow_long_stock": "ALLOW_LONG_STOCK",
    "allow_long_call": "ALLOW_LONG_CALL",
    "allow_long_put": "ALLOW_LONG_PUT",
    # Phase 1 partial scope (roadmap): gates spread RESEARCH + BACKTEST now;
    # live §10 execution keeps its own hard veto until mleg lands, stated in
    # the gate detail — the toggle's Settings copy declares this scope.
    "allow_defined_risk_spreads": "ALLOW_DEFINED_RISK_SPREADS",
    # Phase 2 unlock (2026-08-17): collateralized short premium.
    "allow_covered_call": "ALLOW_COVERED_CALL",
    "allow_cash_secured_put": "ALLOW_CASH_SECURED_PUT",
    # Phase 3 unlock (2026-08-17): margin-backed short stock.
    "allow_short_stock": "ALLOW_SHORT_STOCK",
    "allow_margin": "ALLOW_MARGIN",
    "alpaca_api_key_id": "ALPACA_API_KEY_ID",
    "alpaca_api_secret_key": "ALPACA_API_SECRET_KEY",
    # BEA actuals (GDP/PCE); absent -> dates only, values unavailable.
    "bea_api_key": "BEA_API_KEY",
    # External research (Catalyst research upgrade): RESEARCH ONLY providers
    # feeding the event Evidence Bundle — never execution. Unset = the
    # research sections report NOT_CONFIGURED; nothing else degrades.
    "web_search_provider": "WEB_SEARCH_PROVIDER",
    "brave_api_key": "BRAVE_API_KEY",
    "prediction_markets_provider": "PREDICTION_MARKETS_PROVIDER",
}

#: Fields whose values are credentials: presence is reported, value never.
SECRET_KEYS = frozenset(
    {
        "massive_api_key",
        "llm_api_key",
        "alpaca_api_key_id",
        "alpaca_api_secret_key",
        "bea_api_key",
        "brave_api_key",
    }
)

#: Accepted values per enum-valued field ("" always allowed = unset/default).
ALLOWED_PROVIDERS: dict[str, frozenset] = {
    # "alpaca" is the authoritative market-data source (data_source.md §1);
    # it authenticates with the broker's alpaca_api_key_id/secret pair.
    "market_data_provider": frozenset({"", "alpaca", "massive", "stub"}),
    "llm_provider": frozenset({"", "openai", "stub"}),
    "broker_provider": frozenset({"", "alpaca_paper", "simulated"}),
    # Language of newly generated LLM narrative ("" = default en). Not a
    # provider, but validated through the same enum gate.
    "llm_output_language": frozenset({"", "en", "zh"}),
    # Instrument permissions: STRICT true/false — NO empty string. These env
    # vars feed pydantic bool fields, and an env value of "" would make
    # Settings() unconstructable and take the whole app down with it.
    "allow_long_stock": frozenset({"true", "false"}),
    "allow_long_call": frozenset({"true", "false"}),
    "allow_long_put": frozenset({"true", "false"}),
    "allow_defined_risk_spreads": frozenset({"true", "false"}),
    "allow_covered_call": frozenset({"true", "false"}),
    "allow_cash_secured_put": frozenset({"true", "false"}),
    "allow_short_stock": frozenset({"true", "false"}),
    "allow_margin": frozenset({"true", "false"}),
    # External research providers (RESEARCH ONLY — plan §8). "brave" /
    # "polymarket" are accepted before their adapters land (LOOP 2/4): the
    # registry then answers "unknown provider" honestly rather than this
    # gate lying about what will eventually be valid.
    "web_search_provider": frozenset({"", "brave", "stub"}),
    "prediction_markets_provider": frozenset({"", "polymarket", "stub"}),
}


def _clear_derived_caches() -> None:
    """Rebuild Settings and drop every provider-derived cache."""
    get_settings.cache_clear()
    # The §16 capability verdict belongs to the OLD provider/key. Imported
    # lazily: routers.market imports deps which sits beside this module.
    from .routers import market, options

    market._capabilities_cache.update({"at": None, "provider": None, "payload": None})
    market._rest_quotes_cache.update({"at": None, "provider": None, "quotes": None})
    # The EOD options day-cache and the short chain cache are provider-keyed,
    # but a switch should not keep serving the old vendor's payloads either.
    options._eod_cache.clear()
    options._chain_cache.clear()
    # The stream cache belongs to the OLD provider's session; the supervisor
    # re-evaluates the new settings within its next cycle.
    from . import market_stream

    market_stream.CACHE.clear()


async def apply_overrides(session) -> int:
    """Load every stored override into the environment; returns the count.

    Called from the gateway lifespan before serving and after every
    :func:`set_values`. Values land in ``os.environ`` VERBATIM (empty string
    = explicitly unset, which Settings reads as unconfigured — the honest
    503 path, never a fallback provider).
    """
    rows = (await session.execute(select(RuntimeConfig))).scalars().all()
    applied = 0
    for row in rows:
        env = CONFIG_KEYS.get(row.key)
        if env is None:
            logger.warning("runtime_config holds unknown key %r; ignored", row.key)
            continue
        os.environ[env] = row.value
        applied += 1
    if applied:
        _clear_derived_caches()
        logger.info(
            "runtime_config_applied",
            extra={"extra_fields": {"keys_applied": applied}},
        )
    return applied


async def set_values(session, changes: dict[str, str]) -> list[str]:
    """Upsert `changes` (already validated by the caller) and apply them.

    Returns the sorted list of changed KEYS (for the audit event — never the
    values). Does NOT commit: the caller owns the transaction so the audit
    event lands with the rows (rule 12).
    """
    changed: list[str] = []
    for key, value in changes.items():
        if key not in CONFIG_KEYS:  # defence in depth; API validates first
            raise ValueError(f"unknown config key {key!r}")
        row = await session.get(RuntimeConfig, key)
        if row is None:
            session.add(RuntimeConfig(key=key, value=value, updated_at=utcnow()))
            changed.append(key)
        elif row.value != value:
            row.value = value
            row.updated_at = utcnow()
            changed.append(key)
    if changed:
        await session.flush()
        # Apply the FULL stored state (not just the delta) so environment,
        # Settings cache and DB can never drift apart.
        await apply_overrides(session)
    return sorted(changed)
