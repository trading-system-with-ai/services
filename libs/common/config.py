"""Central configuration. All secrets come from environment variables (never committed)."""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ALLOW_* env flags (guide §8) whose True value the platform HARD-REJECTS at
# startup (guide §33, non-negotiable rules), with the capability each names.
# The flags exist to make the cash-account restriction EXPLICIT in the
# environment file — never to lift it: there is no code path for any of these
# (no Sell-to-Open exists in this system, no margin model exists), and Alpaca
# Paper capability does not override platform permissions (§2).
# 2026-08-17 Phase 2 UNLOCK: covered calls and cash-secured puts left this
# set — their collateralized Sell-to-Open chain now exists end to end
# (selection, collateral locking, venue fills, mechanical management,
# sweep, §16, §18, backtests). Naked shorts remain (broker refusal + §4);
# short stock remains until the Phase 3 margin model exists.
# 2026-08-17 Phase 3 UNLOCK: allow_short_stock left this set (the
# margin-backed short chain exists). ONLY the naked shorts remain, forever.
_FORBIDDEN_ALLOW_FLAGS: dict[str, str] = {
    "allow_naked_short_call": "naked short calls",
    "allow_naked_short_put": "naked short puts",
}


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
    # Language of NEWLY GENERATED LLM narrative (summary, evidence snippets):
    # "en" | "zh". Machine-read fields (horizon, catalyst_type, reason_codes)
    # stay English regardless. Stored rows keep the language they were
    # generated in — records are never rewritten.
    llm_output_language: str = "en"
    # HTTP read timeout, in SECONDS, for the pre-event event-analysis call
    # ONLY (libs.llm provider.analyze_event). Deliberately separate from the
    # 60s the discovery calls use: an analysis prompt carries the whole §46
    # evidence bundle and asks for a long structured note, and a live run on
    # gpt-5.6-sol took 51s to answer — close enough to 60s that the next one
    # hit httpx.ReadTimeout and was stored as a FAILED analysis, having paid
    # for the inference. Raising the shared timeout instead would let a hung
    # RECOMMENDATIONS refresh sit for four minutes, which is a different and
    # worse trade. A parameter, never a hardcoded truth (plan §6.2).
    llm_analysis_timeout_seconds: float = 240.0

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

    # TEST-DETERMINISM SEAM for the SYNTHETIC stub providers (market data,
    # web search, prediction markets). The stubs' synthetic output slides
    # with "today", so signal/regime verdicts for a given ticker roll daily
    # and date-sensitive tests rot overnight. When set (ISO date, e.g.
    # "2026-08-11") every stub anchors its series/window end at this date and
    # the whole synthetic universe is frozen forever. Empty (default) = today.
    # Real providers NEVER read this.
    stub_anchor_date: str = ""


    # Order-sync sweep (guide §11 Iteration C): how often the background loop
    # re-asks the broker about non-terminal local orders (PENDING_SUBMIT /
    # ACCEPTED / PARTIALLY_FILLED) and applies what actually happened. Only
    # meaningful with a real broker configured; 0 disables the loop (the
    # manual POST /api/broker/sync-orders always works).
    order_sync_interval_seconds: float = 30.0

    # Periodic reconciliation (guide §13 Iteration D): how often the whole
    # broker ledger (account + positions) is compared against local rows. A
    # material mismatch pauses trading through the §18 kill switch — same
    # code path as GET /api/broker/reconcile. 0 disables the loop.
    reconciliation_interval_seconds: float = 300.0

    # ------------------------------------------------------------------
    # Account permissions (guide §2, §8) — the env-configurable side of
    # libs.trading_core.strategies.AccountPermissions, wired through the ONE
    # factory apps.gateway.deps.account_permissions_from_settings().
    #
    # The first four are REAL flags: flipping one re-derives the §8
    # instrument matrix (spreads stay a deferred capability, default OFF
    # until the account is explicitly approved).
    allow_long_stock: bool = True
    allow_long_call: bool = True
    allow_long_put: bool = True
    allow_defined_risk_spreads: bool = False
    # The five below are DISPLAY-AND-REFUSE flags (guide §2, §33): they
    # default False and a True value is REJECTED at startup by the validator
    # underneath — this platform cannot execute any of them (no Sell-to-Open,
    # no margin model), so the flag exists to make the restriction explicit,
    # not to lift it. Alpaca Paper capability does not override platform
    # permissions (§2), and there is deliberately NO ALLOW_MARGIN flag at all
    # (§33 rule 7: nothing margin-dependent exists to enable).
    allow_short_stock: bool = False
    allow_naked_short_call: bool = False
    allow_naked_short_put: bool = False
    allow_covered_call: bool = False
    allow_cash_secured_put: bool = False
    # Phase 3 (2026-08-17): margin exists to SUPPORT SHORTING — the broker
    # enforces buying power/maintenance on its side; levered LONG sizing
    # stays off by charter (§12 sizes from cash, never buying power).
    allow_margin: bool = False

    @field_validator(*_FORBIDDEN_ALLOW_FLAGS)
    @classmethod
    def _reject_forbidden_permissions(cls, value: bool, info) -> bool:
        """Hard-reject forbidden ALLOW_* flags set true (guide §33).

        Failing at startup — not at order time — is the point: an operator
        who writes ALLOW_SHORT_STOCK=true must learn immediately that the
        platform cannot honor it, rather than discovering it on a refused
        order (or worse, never discovering it at all).
        """
        if value:
            capability = _FORBIDDEN_ALLOW_FLAGS[info.field_name]
            raise ValueError(
                f"{info.field_name.upper()}=true violates guide §33 "
                f"(non-negotiable rules): this platform cannot execute "
                f"{capability} — no code path for it exists (no Sell-to-Open, "
                "no margin). The flag exists to make the restriction "
                "explicit, not to lift it; Alpaca Paper capability does not "
                "override platform permissions (§2)."
            )
        return value

    # Global safety switches
    trading_enabled: bool = False  # kill switch default: OFF until explicitly enabled

    #: Create missing tables at startup instead of VERIFYING the schema.
    #:
    #: Off by default, and that default is the point. When this was implicitly
    #: on, an ORM table with no migration was auto-created at startup — so it
    #: existed on the developer's machine and nowhere else, and the divergence
    #: surfaced only when someone built a database from migrations/ much later.
    #: Turn it on for a genuinely empty development database; never in an
    #: install whose schema came from migrations/.
    db_auto_create: bool = False

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

    # Statistical risk snapshot loop (Risk Engine Upgrade Phase B, design
    # contract §6): seconds between background ticks of
    # apps/gateway/risk_snapshot.py, which persists ONE SCHEDULED snapshot
    # per America/New_York calendar day (the NAV series live drawdown is
    # measured on). 0 disables the background task entirely; the ON_DEMAND
    # build behind GET /api/portfolio/risk always runs. A parameter, never a
    # hardcoded truth (plan §6.2). SHADOW: nothing it computes can alter a
    # Tier 0 decision.
    risk_snapshot_interval_seconds: int = 1800

    # ------------------------------------------------------------------
    # Catalyst & Event Intelligence (prompts/event_analy_system.md §7-§11;
    # audit §5.1/§11.1). Phase B.
    #
    # Seconds between background ticks of apps/gateway/event_calendar.py,
    # which ingests the event registry (SEC EDGAR 8-K earnings history, the
    # Fed's own FOMC/speech feeds, exchange sessions and holidays) and fires
    # the T-minus alert. 0 disables the background task entirely; POST
    # /api/events/refresh always works. A parameter, never a hardcoded truth
    # (plan §6.2). The per-provider re-fetch cadence is separate and lives in
    # event_calendar.py: a short interval here does NOT hammer SEC or the
    # Fed, it only re-checks whether their daily cadence has come due.
    event_calendar_interval_seconds: int = 3600

    # Spec §11: "the first alert should generally become available roughly
    # one week before the event". How many days ahead a CONFIRMED/REVISED
    # event raises its EVENT_APPROACHING alert — written exactly once per
    # (event_key, horizon). ESTIMATED events NEVER alert (§11: do not
    # fabricate an exact event date when only an estimate exists).
    event_horizon_alert_days: int = 7

    # SEC EDGAR requires a contact User-Agent on every request and rate-limits
    # (or 403s) anonymous traffic. The default is honest about being a
    # placeholder AND names the env var to set — an operator running real
    # ingestion must put a real contact address here, exactly as SEC's fair
    # access policy requires.
    sec_user_agent: str = (
        "trading-system-with-ai/0.1 (catalyst research; set SEC_USER_AGENT)"
    )

    # Which event calendar providers to use, CSV, e.g.
    # "sec_edgar,fed,alpaca_calendar". EMPTY (default) means AUTO: the keyless
    # primary sources (SEC EDGAR, the Federal Reserve) plus every vendor
    # adapter whose credentials exist — so an install with no vendor keys
    # still gets a real calendar rather than an empty page. The synthetic
    # "stub" provider is reachable ONLY by naming it here explicitly (same
    # opt-in discipline as MARKET_DATA_PROVIDER=stub): an unconfigured install
    # must never serve invented events.
    event_calendar_providers: str = ""

    # BEA's statistics API (GDP/PCE ACTUALS) needs a free key from
    # https://apps.bea.gov/API/signup/. Absent it, GDP/PCE release DATES
    # still come from BEA's keyless schedule page and only the values are
    # reported unavailable — never estimated (§8, audit §6).
    bea_api_key: str = ""

    # ------------------------------------------------------------------
    # External research capabilities (Catalyst research upgrade,
    # SEARCH_PREDICTION_MARKET_UPGRADE_PLAN.md §8). RESEARCH ONLY: these
    # providers feed the event Evidence Bundle and nothing else — no code
    # path from either reaches instrument selection, risk, gates or orders
    # (enforced by tests/test_research_safety_adversarial.py).
    #
    # SAME HONESTY RULE (§44 rule 18): both selectors default to "" — UNSET —
    # with NO fallback. Unconfigured means the web-research / prediction-
    # market sections of event research report themselves honestly
    # unavailable (NOT_CONFIGURED) while every other event surface keeps
    # working — capability-by-capability degradation, never all-or-nothing.
    web_search_provider: str = ""  # "" = unset; "brave" (real) | "stub" (dev/tests only)
    brave_api_key: str = ""
    # Polymarket's read APIs are public and keyless, but keyless does NOT
    # mean default-on: an outbound network dependency is a conscious opt-in,
    # and there are no wallet/trading credentials anywhere — the subsystem
    # is read-only by construction (no trading surface exists to configure).
    prediction_markets_provider: str = ""  # "" = unset; "polymarket" (real) | "stub" (dev/tests only)


@lru_cache
def get_settings() -> Settings:
    return Settings()
