"""Central configuration. All secrets come from environment variables (never committed)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "trading-platform-backend"
    environment: str = "dev"  # dev | paper | live

    # Database: async SQLAlchemy URL. Postgres/Timescale in docker, sqlite for local unit tests.
    database_url: str = "sqlite+aiosqlite:///./dev.db"
    redis_url: str = "redis://localhost:6379/0"

    # External providers.
    #
    # HONESTY RULE (§44 rule 18): both provider fields default to "" — UNSET —
    # and there is deliberately NO fallback. Massive is the only supported real
    # market data source; when MARKET_DATA_PROVIDER is unset every endpoint
    # needing market data answers 503 MARKET_DATA_NOT_CONFIGURED and the
    # platform shows NOTHING rather than synthetic numbers that look real.
    # The "stub" providers still exist in the registries but are opt-in only
    # (development/tests), never a default — an unconfigured install must never
    # silently serve made-up prices, bars, chains or recommendations.
    massive_api_key: str = ""
    market_data_provider: str = ""  # "" = unset; "massive" (real) | "stub" (dev/tests only)
    llm_provider: str = ""  # "" = unset; "openai" | "anthropic" (real) | "stub" (dev/tests only)
    llm_api_key: str = ""
    # Must match llm_provider — model ids are provider-specific and there is no
    # cross-provider translation (e.g. "gpt-5.6-sol" for openai,
    # "claude-opus-5" for anthropic). Ignored while llm_provider is unset.
    llm_model: str = "gpt-5.6-sol"

    # Global safety switches
    trading_enabled: bool = False  # kill switch default: OFF until explicitly enabled

    # Paper trading account (plan §11): starting cash seeded into the singleton
    # portfolio row on first access. A parameter, never a hardcoded truth.
    paper_initial_cash: float = 100_000.0

    # Paper fill model (plan §11): fills are simulated off the last STORED
    # daily close, moved AGAINST the trader by paper_slippage_bps (buys fill
    # higher, sells fill lower) plus a flat per-share commission both ways.
    # Parameters, never hardcoded truths (plan §6.2).
    paper_slippage_bps: float = 5.0
    paper_commission_per_share: float = 0.005
    # Option paper fills (plan §11): flat per-CONTRACT commission charged on
    # both sides, alongside the same slippage model applied to the contract
    # mid. A parameter, never a hardcoded truth (plan §6.2).
    paper_commission_per_contract: float = 0.65

    # Automated position monitor (plan §26): seconds between background exit
    # sweeps run by apps/gateway/monitor.py. 0 disables the background task
    # entirely (the manual POST /api/positions/check-exits path always
    # remains). A parameter, never a hardcoded truth (plan §6.2).
    position_monitor_interval_seconds: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
