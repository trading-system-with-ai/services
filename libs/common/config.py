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

    # Broker / real execution (plan §11).
    #
    # SAME HONESTY RULE, APPLIED TO FILLS (§44 rule 18). ``broker_provider``
    # defaults to "" — UNSET — and there is NO fallback: with no broker
    # configured, POST /api/orders/approve and POST /api/orders/close answer
    # 503 BROKER_NOT_CONFIGURED and the mechanical exit sweep SKIPS. An
    # unconfigured install places nothing and closes nothing.
    #
    #   ""             unset — no execution at all (the default)
    #   "alpaca_paper" the real Alpaca PAPER account (libs.broker registry)
    #   "simulated"    DEVELOPMENT / BACKTEST-COMPARISON ONLY — the internal
    #                  paper fill model in apps/gateway/routers/orders.py.
    #                  Handled in the GATEWAY, deliberately absent from the
    #                  libs.broker registry: a simulated fill is not a broker
    #                  fill and must never be reachable through the broker
    #                  interface (see libs/broker/__init__.py).
    #
    # LIVE TRADING IS NOT SUPPORTED. There is no provider name and no base URL
    # in this configuration that reaches a live account: the Alpaca adapter
    # refuses any host but paper-api.alpaca.markets at construction and
    # re-verifies the account's paper flag with the broker before every order.
    broker_provider: str = ""
    alpaca_api_key_id: str = ""
    alpaca_api_secret_key: str = ""
    # The paper endpoint, exposed as a parameter (§6.2) only so an operator can
    # point it at a local mock. It is NOT a live-trading seam: the adapter
    # rejects any host other than paper-api.alpaca.markets with a ValueError
    # naming the host, so a live URL is a hard startup failure, never an order.
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    # Submission is synchronous-then-poll (plan §11): after submitting we poll
    # the broker for at most this many SECONDS TOTAL, with a short bounded
    # backoff, waiting for a terminal-or-partial state. Whatever state has been
    # reached when the budget runs out is recorded honestly — an ACCEPTED order
    # with no fill stays ACCEPTED and opens NO position. A parameter, never a
    # hardcoded truth (§6.2); 0 disables polling entirely.
    broker_fill_poll_seconds: float = 2.0

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
