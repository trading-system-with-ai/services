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

    # External providers (empty by default; required only when the relevant service starts)
    massive_api_key: str = ""
    market_data_provider: str = "stub"  # "stub" until the MASSIVE integration lands (plan §22.1)
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    # Global safety switches
    trading_enabled: bool = False  # kill switch default: OFF until explicitly enabled

    # Paper trading account (plan §11): starting cash seeded into the singleton
    # portfolio row on first access. A parameter, never a hardcoded truth.
    paper_initial_cash: float = 100_000.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
