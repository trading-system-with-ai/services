# Catalyst & Event Intelligence — Phase A Audit (2026-08-19)

> **STATUS NOTE (2026-08-21) — superseded in part by the Catalyst research
> upgrade.** This audit describes the platform as of 2026-08-19. Two of its
> findings have since been addressed and its §9.1 field list has grown:
>
> - The **`options_analysis` evidence-bundle gap** (bundle section hardcoded
>   unavailable while the options subsystem was live) is **FIXED**. The section
>   is populated from `event_options.build_event_options_payload()` and tiered
>   QUANT. See `event-research-orchestration.md`.
> - **LLM retry/backoff** (recorded here as absent) now exists, scoped to
>   `analyze_event` only: one bounded retry on transport/429/5xx. Discovery
>   calls stay fail-fast deliberately.
> - The §46 bundle is now **`f1-evidence-v2`**, adding `web_research` and
>   `prediction_markets` sections; the analysis contract is
>   **`event-analysis-v2`**. See `search-architecture.md` and
>   `prediction-market-architecture.md`.
>
> Everything else below stands as written and is still the reference for what
> existed before that program.

## 1. Purpose & scope (Phase A of spec §93; audit-only, no code)

This document is the deliverable of **Phase A — "Architecture / capability audit"** (spec §93), executing the §1 mandate to inspect the platform *before* implementing anything and the §100.A/§100.B final-deliverable obligations ("what existed before" + a per-feature Data Coverage Matrix). **No source file is modified by this audit.** Scope of Part 1: purpose (§1), reuse inventory (§1, §4 "do NOT create duplicate infrastructure"), data coverage (§2, §75–§77, §100.B), and the gap matrix against §5–§92. Skill decomposition (§4/§100.C), pipeline design (§100.D), UI (§100.E), LLM architecture (§100.F), risk integration (§100.G), validation plan (§95/§96/§100.H) and deferrals (§100.I) are Parts 2–4.

Two ground-truth inputs constrain everything below: the repository itself (every claim cites an absolute path), and the live entitlement probe of 2026-08-19 ~10:30 ET, which is treated as authoritative for what Massive and Alpaca can serve *today* — not vendor documentation (§2: "Do NOT assume that an endpoint documented by Massive is included in the current subscription").

Repo layout note used throughout: the repo root holds only `prompts/`, `services/`, `ui/`. `docker-compose.yml` and `docs/ARCHITECTURE.md` live under `services/`, not the root.

## 2. What exists today — reuse inventory

### 2.1 Gateway & infrastructure
- **Modular-monolith FastAPI gateway; one lifespan owns every background loop** — `services/apps/gateway/main.py::lifespan` (line 104) builds `background_tasks: list[asyncio.Task]` (line 118), gates four loops on `settings.*_interval_seconds > 0` (`monitor.monitor_loop`, `order_sync.order_sync_loop`, `reconciliation_loop`, `risk_snapshot.risk_snapshot_loop`), always creates `market_stream.market_stream_loop` (which self-disables per cycle via `market_stream.py::_stream_wanted`, lines 156-175), and on shutdown cancels **and awaits** each task (lines 149-156). **Five tasks total.** This is the insertion point for §8 calendar ingestion and §11 event alerting. Caveat documented in-code: httpx `ASGITransport` does not run lifespan, so no loop starts under the test suite — hence every loop splits its tick into a directly callable function (`monitor.run_sweep_and_update:87`, `risk_snapshot.run_scheduled_snapshot:2006`).
- **Copy-ready resilient scheduler template** — `services/apps/gateway/risk_snapshot.py`: `NEW_YORK = ZoneInfo("America/New_York")` (142), `new_york_today()` (1979), `_scheduled_exists_today()` (1984) computing the NY day by `as_of.astimezone(NEW_YORK).date()` (2001) rather than a stored date column, and named skips `{"skipped": "MARKET_DATA_NOT_CONFIGURED" | "ALREADY_BUILT_TODAY" | "BROKER_UNREADABLE" | "NO_ACCOUNT"}` instead of fabricated rows. `CancelledError` re-raised, all else logged and swallowed (`market_stream.py:176-178`) — exactly §8's "calendar ingestion should survive individual provider failures".
- **Migrations** — raw SQL only, `services/migrations/001..020_*.sql`; no Alembic, no runner. They execute only via per-file `:ro` mounts into `/docker-entrypoint-initdb.d/` in `services/docker-compose.yml:19-39`, i.e. **only on a fresh Postgres volume**. `stock_bars_1m` (`migrations/002_system_state_and_bars.sql:30`) is the only Timescale hypertable and is **never written to**; `stock_bars_daily` (007) is a plain table.
- **Audit** — `services/apps/gateway/audit.py::record` (14-47) adds the row to the *caller's* session and never commits (atomic with state), auto-filling `correlation_id` from `libs/common/telemetry.py::request_id_var`. `AuditAction` (`services/libs/trading_core/models/enums.py:14-42`) has 28 members incl. `NEWS_INGESTED`, `DATA_BACKFILL`, `PLAN_GENERATED/APPLIED/SUPERSEDED`; `ActorType` (line 5) is `USER|SYSTEM|LLM`. ADR-003, `services/docs/ARCHITECTURE.md`.
- **Alerts as a classified read over the audit trail** — `services/apps/gateway/alerts.py`: `ALERT_RULES` (152-163, 8 entries), `AlertRule(severity, title_builder, predicate)`, `classify()` (181-190), predicate example `_risk_decision_is_alert` (99); served by `routers/alerts.py:43`. ADR-006 at `services/docs/ARCHITECTURE.md:64`.
- **Metrics** — zero-dependency registry in `services/libs/common/telemetry.py`: `Counter` (120), `Histogram` (151, ms buckets, `DEFAULT_BUCKETS_MS`:34), `Gauge` (233), `render_prometheus` (315), `REGISTRY` (327). `main.py` serves `GET /metrics` (290) after `_refresh_freshness_gauges` (204) — the scrape-time recompute precedent; `risk_snapshot.py:220 RISK_SNAPSHOT_AGE_SECONDS` uses the callback gauge.
- **Runtime config / feature flags** — `services/apps/gateway/runtime_config.py`: `CONFIG_KEYS` (34), `SECRET_KEYS` (64), `ALLOWED_PROVIDERS` (69), `_clear_derived_caches` (92), `apply_overrides` (112, called from lifespan before Settings is read), `set_values` (138). Secrets never returned/logged/audited by value. In-code gotcha at line 78: bool env values must be strictly `true`/`false`.
- **Honest-absence guards** — `services/apps/gateway/deps.py`: `market_data_unavailable_reason` (86), `llm_unavailable_reason` (115), `broker_unavailable_reason` (195), `require_market_data_provider` (144), `require_llm_provider` (161), `require_broker` (228), `market_data_status` (275), `broker_status_block` (260). This is §97/§98 already in house style.

### 2.2 Providers
- `services/libs/market_data/__init__.py::_PROVIDERS` — registry by name, **no default**, `ProviderNotConfigured` on empty, `ValueError` on unknown, no cross-provider fallback. Adapters: `massive.py`, `alpaca.py`, `alpaca_stream.py`, `stub.py`.
- `services/libs/market_data/provider.py` — `MarketDataProvider` Protocol (107) exposes only `get_quotes` (110), `get_daily_bars` (114), `get_option_chain` (118); `get_news`/`get_option_contracts`/`probe_capabilities` exist on the concrete classes but are **not in the Protocol**. Dataclasses `Quote` (69), `NewsArticle` (79), `Bar` (96); `CapabilityNotAvailable` (50) raised on 403.
- **Capability probing** — `routers/market.py::market_capabilities` (160), `CAPABILITIES_TTL_SECONDS = 300.0` (155), `_capabilities_cache` (157) keyed on provider with a `refresh=true` bypass (161). Both `massive.py::probe_capabilities` (971-976) and `alpaca.py::probe_capabilities` (928-933) return the **same fixed key set** `{stock_history, stock_realtime, option_chain, option_contracts, news}` with tri-state `true` / `false` (403) / error-string — a probe fault is deliberately distinguished from proven absence. Adding an `earnings_calendar` key means editing **both** adapters and waiting out the 300 s cache.
- **News ingestion** — `db.py::NewsArticleRow` (154) / `migrations/012_news_articles.sql:13` (`source_id VARCHAR(128) NOT NULL UNIQUE` = the dedup key); `routers/recommendations.py::_refresh_locked` (`NEWS_FETCH_LIMIT=50`, `NEWS_ENRICH_LIMIT=20`, `_refresh_lock()`, the `NEWS_INGESTED` audit, and the drop of any LLM draft citing an unstored article — §27/§79 already enforced). **Asymmetry:** `massive.py:768 get_news(limit=50, published_after: datetime|None=None)` vs `alpaca.py:872 get_news(limit=50)` — Alpaca has no watermark parameter.
- **Bars with point-in-time hygiene** — `routers/analysis.py::ensure_daily_bars`, `_complete_days_only`, `_last_expected_trading_date`, `_stale_trading_days`, `_refresh_attempts`, `REFRESH_ATTEMPT_SECONDS=1800`; today-Eastern bars dropped as provisional; ADR-005 reference symbols `routers/market.py:37 INDEX_SYMBOLS = ["SPY","QQQ","VIX"]`.

### 2.3 Quant libraries (pure, no I/O — the §47 "backend calculates" layer)
`services/libs/trading_core/features/indicators.py` — `sma` (37), `ema` (56), `rsi` (77), `true_range` (114), `atr` (136), `macd` (163), `realized_vol` (205), `pivot_highs/lows` (236/257): §31 needs no new indicator code. `volatility.py::classify_vol_regime` (84); `options/bs.py`, `options/iv.py`, `options/reval.py`, `greeks.py`; `contracts/selector.py::ContractQuote`; `risk/engine.py::assess` (388) with `extra_caps: Sequence[ExtraCap] = ()` (396, Protocol at 206); `risk/models/stress.py`, `risk/snapshot.py`, `risk/validation.py`, `correlation.py`, `backtest/engine.py`.

### 2.4 LLM
`services/libs/llm/{__init__,provider,openai,anthropic,stub}.py` — registry by name, no default (unset → 503 `LLM_NOT_CONFIGURED`), `llm_model` must match provider. `db.py:459 Recommendation.llm_model` (migration 014) is the never-backfilled model-attribution precedent, defaulting to `""` = honest unknown. Note `runtime_config.ALLOWED_PROVIDERS` whitelists only `{"", "openai", "stub"}` — `anthropic.py` is reachable via `.env` but not the Settings UI. No token accounting anywhere.

### 2.5 Risk / trade flow
`routers/orders.py` RISK_APPROVAL chain; `apps/gateway/risk_snapshot.py::build_risk_snapshot`; `risk_validation.py`; `risk_inputs.py`; `broker_exec.py`; `order_sync.py`; `monitor.py`. Persisted-analysis precedent: `db.py::TradePlanRow` (112-139, migration 013) — full payload JSON `preview`, `versions` JSON, `market_data_as_of`, `generated_at`, `status` (`enums.py:45-60 PlanStatus`), `superseded_by`, `created_by`. §67 lifecycle and §84 auditability map onto this shape almost one-to-one. Existing shadow blocks (`shadow.statistical`, `shadow.vol_targeting_ewma`) in the `RISK_DECISION` audit details are the §65 SHADOW-mode template; tests assert via AST that **no** `assess()` call in `apps/` passes `extra_caps`.

### 2.6 UI
Next.js app at `ui/`. Nav is a flat 10-entry list — `ui/components/shared/Nav.tsx:9-19` (`/`, `/guide`, `/recommendations`, `/watchlist`, `/trading-pool`, `/positions`, `/backtests`, `/risk`, `/activity`, `/settings`); **no `/catalysts` destination exists** (§53). Reusable primitives: `ui/components/shared/{Modal,Toast,ConfirmDialog,NotConfigured,Placeholder,Term,FlowNav,RiskMethodModal}.tsx` — `Term.tsx` is the §90 ⓘ explainer, backed by the 827-line bilingual `ui/lib/glossary.ts`; `ui/components/charts/CandlestickChart.tsx` is the only chart component (§60/§61 need more); `ui/components/risk/{StatisticalRisk,StressScenarios,ModelValidation,TradeComparison}.tsx`; `ui/lib/{api.ts,types.ts,i18n.tsx,use-capabilities.ts}`.
**Existing catalyst surface (not greenfield):** `services/apps/gateway/routers/analysis.py::get_symbol_catalyst` (590-687) on router prefix `/api/watchlist` (`analysis.py:55`) → `GET /api/watchlist/{ticker}/catalyst`, labelled "upgrade §11/§38 — Phase E". Read-only over stored data, never calls the LLM, 404s off-watchlist, returns `{ticker, generated: true, llm, articles, latest_source_published_at}`. Consumed at `ui/lib/api.ts:366-369`, typed `SymbolCatalyst` at `ui/lib/types.ts:1637`, pinned by `services/tests/test_catalyst.py`. `Recommendation.catalyst_type` (`db.py:451`, `migrations/001_initial.sql:49`) is a free-form LLM string.

### 2.7 Tests & docs
`services/tests/` (~70 files). Load-bearing for Phase B: `test_migration_parity.py` — **four** tests, incl. `test_every_migration_is_mounted_in_docker_compose`, `test_migration_numbers_are_contiguous` (so a new migration must be exactly `021_*.sql`), and `test_orm_columns_mirror_single_create_migrations`, whose coverage is **opt-in** via the allowlist `_SINGLE_CREATE_TABLES` (83-93: 007, 018, 019, 020 only) and checks column **names and order** only. `test_no_synthetic_data.py` is a genuine property test (`MARKET_DATA_ENDPOINTS`:44, walker at 444, meta-test at 459). Also `test_catalyst.py`, `test_observability.py`, `test_risk_adversarial.py`, `test_news_phase8.py`. Docs: `services/docs/{ARCHITECTURE.md,DEVLOG.md,data-source-architecture.md,risk-engine-*.md}`.

## 3. Data Coverage Matrix (spec §100.B)

Probe of 2026-08-19 ~10:30 ET is authoritative. "PIT-safe?" = does the source carry a publication/acceptance timestamp permitting §14/§85 as-of filtering.

| Feature | Provider | Endpoint | Sub status | Historical | Realtime | PIT-safe? | Fallback |
|---|---|---|---|---|---|---|---|
| Earnings dates (upcoming) | Massive | `/benzinga/v1/earnings` | **403 not entitled** | none | none | n/a | **No authoritative source.** Estimate from filing cadence (§7 `status=ESTIMATED`); user-confirmed IR/SEC; optional 3rd-party adapter behind its own key |
| Earnings dates (past) | SEC EDGAR (free) | `data.sec.gov/submissions/CIK*.json`; efts full-text 8-K Item 2.02 | 200, no sub | full | n/a | **Yes** (`acceptanceDateTime`) | Massive `financials.filing_date` (later than press release) |
| Earnings actuals (EPS/rev/margins/FCF/capex) | Massive | `/vX/reference/financials` | **200** | quarterly + TTM | n/a | **Yes** (`filing_date` + `acceptance_datetime`) | SEC XBRL companyfacts |
| Consensus (EPS/revenue) | — | Massive `/benzinga/v1/earnings` | **403** | none | none | n/a | **None.** Render "CONSENSUS DATA UNAVAILABLE" (§33/§98); surprise % uncomputable |
| Estimate revisions (30/60/90D, PT) | — | Massive `/benzinga/v1/ratings`, `/analyst-insights` | **403** | none | none | n/a | None; §34 not deliverable |
| Guidance | — | Massive `/benzinga/v1/guidance` | **403** | none | none | n/a | LLM extraction from news/8-K, labelled LLM-EXTRACTED (§16), never as fact |
| News | Massive / Alpaca | `/v2/reference/news`; `/v1beta1/news?symbols=` | **200 / 200** | yes | yes | **Yes** (published_at) | Massive supports `published_after` watermark; Alpaca does not (over-fetch + dedupe); Alpaca `content` sometimes empty; `/benzinga/v1/news` 403 |
| Fundamentals / statements | Massive | `/vX/reference/financials` | **200** | quarterly + TTM | n/a | **Yes** | SEC XBRL |
| Ratios / valuation | Massive (derived) | from financials + daily bars | **200** (inputs) | yes | n/a | Yes | Compute in `libs/trading_core`; skip any ratio missing an input (§28) |
| Daily bars (stocks/ETFs) | Massive / Alpaca | `/v2/aggs/ticker/{t}/range`, `/prev`; Alpaca bars | **200 / 200** | full | yes | Yes (complete-days-only rule already applied) | Stored in `stock_bars_daily` via `ensure_daily_bars` |
| Intraday bars (1m → 5m/30m/1h) | Alpaca | `/v2/stocks/{t}/bars?timeframe=1Min` (`feed=iex`) | **200** | yes | yes | Yes | IEX ≠ consolidated tape — must be labelled; target the unused `stock_bars_1m` hypertable |
| Options live IV / greeks | Alpaca | `/v1beta1/options/snapshots/{t}` (`feed=indicative`) | **200** | n/a | yes | n/a | **Massive `/v3/snapshot/options` returns `greeks={}`, `implied_volatility=None`** — Alpaca is the only IV source |
| Options historical bars | Massive / Alpaca | `/v2/aggs/ticker/O:{occ}/range/1/day`; Alpaca option bars (~Feb 2024+) | **200 / 200** | daily only | n/a | Yes | Basis for approximate prior straddle-implied move |
| Historical IV / greeks / OI | — | none | **not sold by either** | none | n/a | n/a | Reconstruct approx. from ATM call+put daily closes, flagged as approximation; `atm_iv_daily` (018, `source` col at line 115) accrues forward only |
| Macro actuals (CPI etc.) | Massive / BLS / BEA | `/fed/v1/inflation` (CPI idx since 1947); `/fed/v1/inflation-expectations`; BLS v2 timeseries; BEA API | **200 / 200 / 200** | long | monthly | **Index values only — no release timestamps** | BLS series ids for PPI/payrolls/JOLTS/unemployment; `/fed/v1/{fed-funds-rate,unemployment,gdp,pce,retail-sales,fomc}` all **404** |
| Macro release calendar | BLS / BEA / Census | `bls.gov/schedule/*` HTML; `bea.gov/news/schedule`; `census.gov/economic-indicators/calendar-listview.html` | 200, no sub | yes | yes | Yes (date + 08:30 ET) | **Not in either paid provider** — needs primary-source adapters (§8) |
| Fed calendar / speeches / statements | Federal Reserve | `fomccalendars.htm`; `feeds/speeches.xml`; `feeds/press_monetary.xml` | 200, no sub | yes | yes | Yes (RSS timestamps) | No provider substitute; §9 requires distinguishing meeting/decision/minutes/speech |
| Treasury yields | Massive / Treasury | `/fed/v1/treasury-yields`; `daily-treasury-rates.csv` | **200 / 200** | 1962+ / full | daily | Yes | **Massive has 1y/5y/10y only — no 2y**; Treasury CSV supplies 2Y/10Y/30Y, else SHY proxy |
| Multi-asset proxies (§39) | Alpaca | daily/intraday bars for SPY,QQQ,GLD,USO,TLT,IEF,SHY,UUP | **200** | full | yes | Yes | Only SPY/QQQ/VIX are ADR-005-exempt today — list must be extended; no DXY index → UUP proxy |
| Peers | Massive | `/v1/related-companies/{t}` | **200** | n/a | yes | No (current-state only) | Sector via `/v3/reference/tickers/{t}` (sic/market/exchange) |
| Corporate actions | Massive / Alpaca | `/v3/reference/dividends`; `/v1/corporate-actions`; `/vX/reference/ipos` | **200** | yes | yes | Partly (declaration/ex/pay dates) | `/vX/reference/tickers/{t}/events` returns **ticker_change only — not earnings** |
| Trading calendar / holidays | Alpaca / Massive | paper-api `/v2/calendar`; `/v1/marketstatus/upcoming` | **200 / 200** | yes | yes | Yes (open/close session times) | Neither is called today; holidays currently unmodelled |

## 4. Gap matrix vs spec §5–§92

| Section group | Gap | Severity | Approach |
|---|---|---|---|
| §5, §6 | No event entity anywhere — no events table, ORM model or enum. Nearest thing is `Recommendation.catalyst_type` (`db.py:451`), a free-form LLM `String(64)` = the generic-string anti-pattern §5 forbids. No `scheduled_at`, `event_status`, `source_url`, `previous_event_id`, `importance`, `series_id`, `fiscal_quarter`, before/after-market. | blocking | `migrations/021_events.sql` (must be exactly 021 — contiguity test) + mirrored ORM in `db.py` following the 018/019 shape; `EventType/EventStatus/EventSession` StrEnums in `libs/trading_core/models/enums.py`; `scheduled_at` as `DateTime(timezone=True)` UTC + separate `event_timezone` (§10). Add the `:ro` mount in `services/docker-compose.yml` and add the table to `_SINGLE_CREATE_TABLES` (`tests/test_migration_parity.py:83-93`) in the same commit. |
| §7, §8, §9 | No calendar provider abstraction, no calendar adapter. Nothing calls `/v1/marketstatus/upcoming`, `/v3/reference/dividends`, `/vX/reference/ipos`, `/fed/v1/*`, `/v1/related-companies`, Alpaca `/v2/calendar` or `/v1/corporate-actions`. Benzinga earnings is 403 → **no authoritative upcoming-earnings source at all**; no BLS/BEA/Census/FOMC adapter. | blocking | New `libs/event_calendar/` package mirroring `libs/market_data/__init__.py::_PROVIDERS` exactly (no default, no cross-provider fallback); `EventCalendarProvider` Protocol; every adapter gets `probe_capabilities()` with the same tri-state so the 403 is a displayable fact. Any third-party earnings key gated behind its own `runtime_config.CONFIG_KEYS` entry (§2 "do not silently add"). |
| §16, §28, §29, §30 | **No fundamentals ingestion of any kind** — grep for financials/fundamental across `libs/` and `apps/` returns only comments. Massive `/vX/reference/financials` is entitled (200) and carries `filing_date` + `acceptance_datetime`. No snapshot, no prev-vs-current comparison, no ratios. | blocking | A workstream of its own: adapter method on `MassiveProvider` + fundamentals table keyed `(ticker, fiscal_period, filing_date, acceptance_datetime)`; derive every §28 ratio in a pure `libs/trading_core` module; store `acceptance_datetime` so §85 filters on "published ≤ as_of", never on fiscal period end. Skip any ratio with a missing input. |
| §33, §34 | Consensus, estimate revisions, guidance, ratings all **403 not entitled**; Alpaca has none. EPS/revenue surprise % (§16), beat frequency (§19) and revision trend (§32) are therefore uncomputable. | blocking (external) | Treat as a first-class documented absence: add the Benzinga keys to **both** `probe_capabilities` implementations, return honest nulls with a machine-readable reason per the `deps.py` convention, render "CONSENSUS DATA UNAVAILABLE" (§33/§98). `tests/test_no_synthetic_data.py` is the guard against anyone later filling these with an estimate. |
| §14, §85, §96 | No as-of primitive and no query-level enforcement. Precedents are per-site conventions only (`get_news(published_after=)`, `_complete_days_only`, walk-forward risk backtests in 020). Nothing can answer "the research that would have existed at T", and nothing stops a new code path calling a current API inside an as-of analysis. | blocking | One `AsOf` type (UTC instant + exchange tz) threaded through every evidence collector; each collector a pure function of `(as_of)` filtering on stored publication/acceptance timestamps only. Prove it with the adversarial style of `tests/test_risk_adversarial.py` (mutate only the last observation, assert earlier outputs bit-identical) to satisfy §96. |
| §10 | Holidays explicitly not modelled — `routers/analysis.py::_last_expected_trading_date` does Mon–Fri arithmetic and says so. No exchange-calendar table, no session times → before/after/during-market (§6) and next open/close (§17) are wrong around holidays and half-days. Both providers can serve a calendar; neither is called. | major | `market_calendar` table (date, open_utc, close_utc, session_type) ingested from Alpaca `/v2/calendar` by the same scheduler; route every next-trading-day computation through one helper (also fixes `_stale_trading_days`). Reuse the existing `America/New_York` constant rather than defining a fourth. |
| §17 | Only daily bars are stored (`StockBarDaily`, 007). `stock_bars_1m` (002) exists as a hypertable and **has never been written to**. No intraday method on `MarketDataProvider` or either adapter, though Alpaca 1Min bars probe 200. | major | Add an intraday adapter method and write into the existing `stock_bars_1m` hypertable rather than a new table. Scope ingestion to event windows only or volume dwarfs the DB. Label the `iex` feed — not consolidated tape. |
| §18, §19, §36, §60 | No historical IV: `atm_iv_daily` (018) records only what the platform computes going forward, and neither provider sells historical quotes/greeks/OI. So ATM IV before/after, IV crush and implied-vs-realized history have no implied side for past events. | major | Reconstruct approximate prior straddle-implied moves from option **daily bars** (ATM call close + put close the day before the event) and label as approximation in payload and UI, following the `atm_iv_daily.source` / stress `RV_PROXY` labelling vocabulary. Current implied move from the live Alpaca snapshot is fully available — only history degrades. |
| §21, §22, §23, §26 | Dedup is exact-match on `news_articles.source_id` — it cannot catch the same story syndicated under different ids, precisely the §23 failure. No `cluster_id`, `canonical_article`, `article_count`, materiality, novelty, source-quality or entity-relevance. Migration 012 indexes only `published_at`, and `tickers` is a JSON list, so "articles citing NVDA between last earnings and as_of" is not an indexed query. | major | Extend `news_articles` with cluster/materiality/score columns (or a sibling `news_clusters`), plus an index serving the `(ticker, published_at)` window — GIN on JSONB tickers or a normalized join table. Clustering must degrade to title+timestamp+entity similarity because Alpaca `content` is sometimes empty. |
| §11 | Alerts are purely retrospective: `classify()` reads stored `AuditEvent` rows keyed by `row.action`; nothing fires on the passage of time, and a T-minus-7-days crossing produces no row. No push/delivery channel. | major | Keep ADR-006 — no alerts table. A scheduler tick writes an audit row (e.g. `EVENT_APPROACHING`, T-minus in `details`) once per `(event_id, horizon_bucket)`, idempotent the way `_scheduled_exists_today` does it (DB query, not in-process state, so restarts cannot double-fire); then one `ALERT_RULES` entry with a predicate on `details`. Horizon is a `Settings` knob, never a literal. |
| §62–§66 | Risk engine has no discrete-jump/event concept: `risk/models/stress.py` scenarios are spot/IV/time shocks with no event trigger, `RiskSnapshotRow` has no time-to-event or event-move columns, the `routers/orders.py` gate chain has no event gate, no `LOW/MODERATE/HIGH/EXTREME` vocabulary. | major | Copy the existing shadow machinery rather than reinventing: `risk/engine.py::assess(extra_caps=())` (388/396) with the `ExtraCap` Protocol (206) is the structural seam, and the AST test asserting no `apps/` caller passes `extra_caps` is the mechanical guarantee of §65 "do not initially block". Add EventRisk as another shadow block in the `RISK_DECISION` audit details and mirror it into the orders response. |
| §72, §73, §82 | Every cache is an in-process dict (`options._chain_cache`, `market._capabilities_cache`, `market_stream.CACHE`, `alpaca._OI_DAY_CACHE`, `analysis._refresh_attempts`). Redis is provisioned (`docker-compose.yml:46-51`, `REDIS_URL` injected at :72) and `Settings.redis_url` exists (`libs/common/config.py:34`) but is referenced **nowhere else in any Python file** — dead weight. No `(event_id, as_of_bucket, data_version, analysis_version)` key concept, no checkpoint/merge for incremental refresh. | major | Do not reach for Redis (§74 simplest architecture; ADR-001). Model the §72 cache as an `event_analyses` table shaped like `TradePlanRow` (payload JSON + versions JSON + generated_at). Any in-process layer **must** register in `runtime_config._clear_derived_caches()`. Heed DEVLOG 2026-08-18 (18): a 15 s-polling UI against a build-on-read endpoint wrote thousands of rows/day — the risk view fixed it with a 15-minute persist dedupe; the catalyst page needs the same guard on day one. |
| §93, §94 | No migration runner: `migrations/*.sql` run only on a fresh volume, so `021_events.sql` will never self-apply on the live deployment (018/019/020 were applied by hand per DEVLOG). No rollback path, no `schema_version` table. | major | Plan an explicit manual apply per phase and record it in the DEVLOG entry; every catalyst migration uses `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS` so re-application is safe. The parity regex requires the exact `CREATE TABLE IF NOT EXISTS <t> (` … `\n);` form. |
| §53, §55 | **Naming collision:** a catalyst surface already ships — `GET /api/watchlist/{ticker}/catalyst` (`routers/analysis.py:590-687`), typed in `ui/lib/types.ts:1637`, pinned by `tests/test_catalyst.py` — but it is read-only LLM interpretation + citations with no event object, no `scheduled_at` and no previous-event linkage. There is also no `/catalysts` nav destination (`Nav.tsx:9-19`). | major | Decide supersede-vs-coexist deliberately. Name the new router `events` (prefix `/api/events`) to avoid a second divergent catalyst concept, keep the existing endpoint's contract until its consumers migrate, and add the §53 nav entry as a new `SECTIONS` row. |
| §39 | Stored bars are limited to watchlist symbols plus ADR-005's `INDEX_SYMBOLS = ["SPY","QQQ","VIX"]` (`routers/market.py:37`), so GLD/USO/TLT/IEF/SHY/UUP cannot be stored. Massive treasury yields have **no 2y**. | minor | Extend the reference-symbol list and write a new ADR recording why. Use Treasury's daily CSV for a true 2Y, else SHY labelled as a proxy — never report a 2y the platform does not have. |
| §60, §61 | UI has exactly one chart component (`ui/components/charts/CandlestickChart.tsx`); §60/§61 need implied-vs-actual-move, surprise, reaction-distribution and macro-reaction charts, plus the §57 timeline. | minor | Backend returns structured chart data only (§61); add chart components beside the existing one and reuse `Term.tsx`+`glossary.ts` for every §90 ⓘ. |
| §4, §75 | No "skills" concept, no plugin registry, no worker/queue. LLM logic lives inline in `routers/recommendations.py` rather than a reusable engine. | minor | Do not invent a skill framework: pure deterministic computation in `libs/trading_core/<domain>/` (Event Impact / Market Reaction / Evidence engines) plus one thin gateway seam module per concern under `apps/gateway/`, exactly the contract `risk_snapshot.py`'s own docstring states. |
| §83, §82 | Metric primitives all exist but no §83 metric does, and there is **no token accounting anywhere** — no counter, no field on `Recommendation`, nothing in `libs/llm`. | minor | Declare counters/gauges/histograms at module import beside the code that increments them (the `monitor.py`/`risk_snapshot.py` convention). Token usage first needs the `libs/llm` provider return shape extended before it can be counted at all. |
| §72, §11 (scaling) | ADR-007: correctness assumes exactly one gateway process. The `asyncio.Lock` in `recommendations.py::_refresh_lock` only serializes in-process; a second replica would double-ingest calendars and double-fire alerts. | minor | Make every scheduled catalyst write idempotent at the **database** level — unique constraints on `(event_id, source)` and `(event_id, alert_horizon)` — the same discipline `news_articles.source_id UNIQUE` already applies. |
| §80, §81 | Structured-output discipline exists but there is no JSON-schema validation/reject path for a malformed LLM response, and no prompt-injection isolation of untrusted article text (news bodies are passed as evidence today with no delimiting/stripping). | minor | Schema-validate and reject malformed responses at the `libs/llm` seam (§80); wrap retrieved article text in an explicit untrusted-data envelope and never let it reach a system role (§81). The existing "drop any draft citing an unstored article" rule in `_refresh_locked` is the precedent to generalize. |

## 5. Proposed decomposition (§4): skills + shared core

**Verdict: do NOT invent a "skill framework."** There is no plugin registry, worker or queue in this
codebase — everything is either a router, a *gateway seam module* (`apps/gateway/risk_snapshot.py`,
`order_sync.py`, `monitor.py`, `risk_validation.py`), or a pure `libs/trading_core/` module. The
existing two-layer split already satisfies §4's "prefer shared reusable infrastructure": pure,
I/O-free computation in `libs/`, one thin seam per concern in `apps/gateway/` that fetches inputs,
calls the pure library, persists, audits and emits metrics. The spec's five "skills" become
**capability groupings over one shared core**, not five runtimes.

### 5.1 Shared core → concrete modules

| Spec core (§4) | New module | Nature | Reuses |
|---|---|---|---|
| Event Intelligence Core | `libs/trading_core/events/models.py` (Event dataclass, `previous_event` matcher + `comparison_reason` §15), `events/taxonomy.py` | pure | `libs/trading_core/models/enums.py` (add `EventType`/`EventStatus`/`EventSession`, beside `PlanStatus:45`) |
| Market Reaction Engine | `libs/trading_core/events/reaction.py` — 1D/3D/5D/10D returns, gap, drift, IV-crush proxy (§17–19, §36) | pure | daily bars via `routers/analysis.py::ensure_daily_bars`; `libs/trading_core/greeks.py`; `volatility.py` |
| Evidence Engine | `libs/trading_core/events/evidence.py` — assembles `EventEvidenceBundle` from already-stored rows only (§46, §74) | pure | `AsOf` (§7 below); `NewsArticleRow`; new fundamentals rows |
| News Intelligence Engine | `libs/trading_core/events/news_intel.py` — clustering, materiality, novelty, source-quality (§22–26) | pure | ingest stays in `routers/recommendations.py::_refresh_locked` (dedup on `news_articles.source_id UNIQUE`, `migrations/012_news_articles.sql:13` / `db.py:154`) |
| Fundamental Snapshot Engine | `libs/trading_core/events/fundamentals.py` — ratios, prev-vs-current deltas (§28–30) | pure | new `MassiveProvider` financials method (`/vX/reference/financials`, entitled 200) |

Gateway seams (one per concern, modelled on `risk_snapshot.py`'s own docstring contract):

- `apps/gateway/event_calendar.py` — the ingestion loop. Copy `risk_snapshot.py` verbatim in shape:
  `NEW_YORK = ZoneInfo("America/New_York")` (`risk_snapshot.py:142`), `new_york_today()` (`:1979`),
  `_scheduled_exists_today()` (`:1984`) computing the day by `as_of.astimezone(NEW_YORK).date()`
  (`:2001`) rather than a stored date column, named skips `{"skipped": "..."}` instead of fabricated
  rows, `CancelledError` re-raised / everything else logged-and-swallowed. Split the tick into
  `run_calendar_ingest()` so tests can drive it — lifespan does not run under `httpx ASGITransport`
  (`main.py:104`), which is exactly why `monitor.run_sweep_and_update:87` and
  `risk_snapshot.run_scheduled_snapshot:2006` exist.
- `apps/gateway/event_analysis.py` — build/persist the research package; the §72 cache writer.
- `apps/gateway/routers/events.py` — **name it `events`, not `catalyst`.** `GET /api/analysis/{ticker}/catalyst`
  already ships (`routers/analysis.py:590-687::get_symbol_catalyst`, pinned by `tests/test_catalyst.py`);
  it is a read-only view over the latest `Recommendation` + cited articles, returns `{ticker, generated:true,
  llm, articles, latest_source_published_at}`, and 404s off-watchlist. Keep it; the new surface supersedes
  it later. `Recommendation.catalyst_type` (`db.py:451`, `migrations/001_initial.sql:49`) is a free-form
  `String(64)` — the §5 anti-pattern; the typed taxonomy replaces its role, it is not extended.

Wiring is two lines: append the task to `background_tasks` in `main.py::lifespan` (line 118, gated on a
new `settings.event_calendar_interval_seconds > 0` like the four existing interval-gated loops) and one
`app.include_router(events.router)` in `create_app()`.

### 5.2 New DB tables (migration **021** exactly — `test_migration_numbers_are_contiguous` forbids gaps)

- `021_events.sql` → **`events`**: `id`, `event_key` (natural dedup key), `event_type`, `title`, `ticker`,
  `company_id`, `scheduled_at TIMESTAMPTZ`, `event_timezone VARCHAR`, `session` (BEFORE/AFTER/DURING/UNKNOWN),
  `event_status` (CONFIRMED/ESTIMATED/REVISED/CANCELED), `source`, `source_url`, `last_verified_at`,
  `previous_event_id` (self-FK) + `comparison_reason`, `importance`, `series_id`, `agency`, `release_period`,
  `fiscal_quarter`, `fiscal_year`, `created_at`, `updated_at`. UNIQUE `(event_key, source)` — §5, §6, §15.
- **`event_analyses`**: `event_id`, `as_of TIMESTAMPTZ`, `as_of_bucket`, `data_version`, `analysis_version`,
  `payload JSONB` (complete package), `versions JSONB`, `llm_model`, `prompt_version`, `generated_at`,
  `status`, `superseded_by`. This is `TradePlanRow` (`db.py:112-139`, migration 013) re-shaped: same
  full-payload-JSON + versions-JSON + as-of + status + self-pointer identity — §72, §84.
- **`market_calendar`**: `session_date`, `open_utc`, `close_utc`, `session_type` — from Alpaca
  `/v2/calendar` (200) + Massive `/v1/marketstatus/upcoming` (200). Fixes the admitted hole in
  `routers/analysis.py::_last_expected_trading_date` ("Holidays are not modeled") — §10, §17.
- **`fundamental_snapshots`**: `(ticker, fiscal_period, filing_date, acceptance_datetime)` UNIQUE, plus
  the raw statement JSON and derived ratios — §16, §28–30, §85.
- Additive on `news_articles`: `cluster_id`, `materiality`, `novelty_score`, `source_quality`, plus a
  `(ticker, published_at)`-capable index (GIN on `tickers JSONB` or a `news_article_tickers` join table) —
  today only `idx_news_articles_published_at` exists (`012:24`) — §22, §23, §21.

Mandatory house rules, all four enforced by `tests/test_migration_parity.py`: mount each new file as an
individual `:ro` line in `services/docker-compose.yml:19-39`; keep numbering contiguous; mirror the ORM in
`db.py` in the same commit; and **add the new tables to `_SINGLE_CREATE_TABLES` (`test_migration_parity.py:83-93`)**
— that tripwire is opt-in per table and checks column names *and order*, so a table left out gets no
coverage at all. Use `CREATE TABLE IF NOT EXISTS` everywhere: `docker-entrypoint-initdb.d` runs only on a
fresh volume, so 021 is a **manual apply** on the live deployment, as 018–020 were.

Also: new `AuditAction` members (`enums.py:14`, 28 today) — `CALENDAR_INGESTED`, `EVENT_DISCOVERED`,
`EVENT_UPDATED`, `EVENT_APPROACHING`, `EVENT_ANALYSIS_GENERATED` — written through the single writer
`audit.py::record()` inside the same transaction. UI: one new `/catalysts` entry in
`ui/components/shared/Nav.tsx:9` `SECTIONS`, an `app/catalysts/` list page and `app/catalysts/[eventId]/`
detail page (detailed in Part 3 §10).

---

## 6. Provider abstraction (§75)

New package `libs/event_calendar/` with its own registry cloning `libs/market_data/__init__.py:63`
`_PROVIDERS` exactly: `get_provider(name)`, `ProviderNotConfigured` on empty, `ValueError` on unknown,
**no default, no cross-provider fallback**. Interfaces are `Protocol`s (like
`libs/market_data/provider.py::MarketDataProvider`), each with a `probe_capabilities() -> dict[str, bool | str]`:

- `EventCalendarProvider` — `get_earnings_events(tickers, window)`, `get_corporate_events(...)`
- `MacroDataProvider` — `get_release_schedule(series)`, `get_actuals(series, as_of)`
- `FedEventProvider` — `get_fomc_calendar()`, `get_speeches()`
- `FundamentalsProvider` — `get_statements(ticker, as_of)` (on `MassiveProvider`, §76)
- `MarketCalendarProvider` — `get_sessions(range)` (Alpaca `/v2/calendar`)

The research layer imports only these Protocols and the normalized `Event`/`Release` dataclasses — never a
Massive or Alpaca response shape (§75, §76).

**SUBSCRIPTION_DENIED as a first-class verdict.** Both `probe_capabilities` implementations
(`libs/market_data/massive.py:971-976`, `alpaca.py:928-933`) return a *fixed, documented* key set
`{stock_history, stock_realtime, option_chain, option_contracts, news}` and a genuine tri-state:
`True` (works) / `False` (HTTP 403 — not in the plan) / the error string (fault, availability unknown).
Add `earnings_calendar` to **both** implementations to keep the shape uniform, and let Massive's
`/benzinga/v1/earnings` **403** land as `False`. Note the coupling: `routers/market.py:155-208` caches the
payload for `CAPABILITIES_TTL_SECONDS = 300.0` keyed on the provider with a `refresh=true` bypass, so the
old shape is served until TTL. The UI then renders "CONSENSUS DATA UNAVAILABLE" (§33, §98) from a probed
fact rather than a guess. `CapabilityNotAvailable` (`provider.py:50`) is the existing exception for this;
`deps.py`'s `*_unavailable_reason` / `require_*` / `*_status()` trio is the shape for a new
`require_event_calendar_provider`.

**Earnings-date fallback chain (deterministic, ordered, never silent).** Per the probe, *no* entitled
source supplies upcoming earnings dates:

1. `CONFIRMED` — subscribed calendar provider (Massive Benzinga add-on) **when the probe says `True`**. Today: `False`.
2. `CONFIRMED` (past) — SEC EDGAR `data.sec.gov/submissions/CIK##########.json`, 8-K **Item 2.02**
   `acceptanceDateTime`. Authoritative, free, point-in-time-safe. Requires a contact `User-Agent`.
3. `ESTIMATED` — derived from filing cadence: Massive `/vX/reference/financials` `filing_date` +
   `acceptance_datetime` for the last 4 quarters → same fiscal quarter last year + ~52 weeks, cross-checked
   against the trailing quarter gaps. Written with `event_status = ESTIMATED`, `source = "derived:cadence"`,
   and a confidence window — never presented as a date.
4. `CONFIRMED` — user-confirmed (IR/SEC), which overwrites 3 and sets `last_verified_at`.
5. Optional keyed third-party adapter (Finnhub/AlphaVantage/FMP) — **must not be silently added** (§2).
   Structurally enforced: it is unreachable unless its key is added to `runtime_config.py::CONFIG_KEYS:34`
   (+ `SECRET_KEYS:64`, + `ALLOWED_PROVIDERS:69` if enum-valued) and a matching `Settings` field exists.
   Absent the key, the registry raises `ProviderNotConfigured` and the chain stops at 3.

Priority is data, not code: an `EVENT_SOURCE_PRIORITY` tuple (§78), so a later `CONFIRMED` row supersedes an
`ESTIMATED` one by rule rather than by whichever adapter ran last.

**Macro / Fed primary-source adapters** — all free, all deterministic, all in `libs/event_calendar/`:
`bls.py` (API v2 `api.bls.gov/publicAPI/v2/timeseries/data/{series}` for actuals; `bls.gov/schedule/2026/MM_sched.htm`
HTML for the 08:30 ET release times), `bea.py` (API + `bea.gov/news/schedule`), `fomc.py`
(`federalreserve.gov/monetarypolicy/fomccalendars.htm` + `feeds/speeches.xml` + `feeds/press_monetary.xml`),
`treasury.py` (daily yield-curve CSV — supplies the **2Y** that Massive `/fed/v1/treasury-yields` lacks),
`census.py`, `sec_edgar.py`. Massive stays the source for macro *actuals* it does serve
(`/fed/v1/inflation`, treasury yields, inflation expectations — all 200).

**Failure isolation is per-adapter, not per-loop.** One tick fans out over adapters; each is wrapped so a
403/timeout/HTML-parse failure increments a `calendar_provider_failures{provider=...}` counter
(`libs/common/telemetry.py::REGISTRY.counter`), logs with traceback, and leaves every other adapter's rows
committed — §8's "calendar ingestion should survive individual provider failures". HTML scrapers must fail
*loudly and empty* (zero rows + a failure metric), never partially: a changed page layout must not silently
yield an event at the wrong time. All writes idempotent at the DB level (UNIQUE `(event_key, source)`),
because ADR-007 gives no leader election and correctness must not ride on single-process assumptions.

---

## 7. As-of / point-in-time design (§14, §85, §96)

### 7.1 The publication timestamps that already exist

| Field | Location | Meaning |
|---|---|---|
| `news_articles.published_at` | `migrations/012_news_articles.sql:16`, `db.py` `NewsArticleRow` | publication instant; `massive.py:768 get_news(limit, published_after)` filters server-side |
| `financials.acceptance_datetime` / `filing_date` | Massive `/vX/reference/financials` (entitled, 200) → new `fundamental_snapshots` | when the statement became public — the true §85 key, **not** period end |
| SEC `acceptanceDateTime` | EDGAR submissions JSON | authoritative 8-K/10-Q/10-K publication instant |
| `stock_bars_daily` bar date | `migrations/007`, `routers/analysis.py::_complete_days_only` | today-Eastern bars are dropped as provisional — already a point-in-time discipline |
| `event_analyses.as_of` | new (021) | the instant the package claims to represent |

### 7.2 One primitive, threaded end-to-end

Define a single frozen `AsOf` dataclass in `libs/trading_core/events/asof.py` carrying the UTC instant plus
the exchange timezone. Today's guarantees are *per-site conventions* (`published_after`,
`_complete_days_only`, "evidence must cite stored news") — there is no shared mechanism and nothing stops a
new code path calling a live API inside an as-of scope. Enforcement, in order:

1. **Every collector is `f(as_of, ...)`** and reads **stored rows only**, filtering
   `published_at <= as_of` / `acceptance_datetime <= as_of` / `bar_date < as_of.date()`. No collector takes
   a provider handle — ingestion (writes rows, no as-of) and analysis (reads rows, as-of-scoped) are
   separate call graphs. This is the single structural rule the whole contract rests on.
2. **`as_of` is required, never defaulted to `now()`** at the seam boundary — a missing as-of is an error,
   not a silent "current".
3. `event_analyses.as_of` + `data_version` + `analysis_version` are part of the cache key, so a package
   generated at T is never re-served as if it were current (§72). Staleness is surfaced, per §71.
4. §84 fields (`llm_model`, `prompt_version`, evidence ids, `versions`) are recorded at generation and
   **never backfilled** — the precedent is `Recommendation.llm_model` (`db.py:459`, migration 014,
   default `""` meaning honest-unknown).
5. Beware write amplification: a 15s-polling UI against a build-on-read endpoint wrote thousands of rows in
   the risk view; the fix there was a 15-minute persist dedupe. `event_analyses` needs the same guard on day one.

### 7.3 NOT BACKTESTABLE (§85) — must be labelled in the payload and the UI

- **Historical ATM IV / greeks / OI** — neither provider sells historical quotes or greeks. `atm_iv_daily`
  (migration 018) only accumulates forward from the platform's own live-chain calculation. Prior implied
  moves are reconstructable *approximately* from option **daily bars** (ATM call close + put close the day
  before) — label as an approximation, exactly as migration 019 labels `RV_PROXY` and 018 uses a `source` column.
- **Consensus EPS/revenue, estimate revisions, guidance, analyst ratings** — Massive Benzinga 403 across the
  board. Not merely un-backtestable: unavailable at *any* time. Surface via the capability probe; return
  honest nulls with a reason code. `tests/test_no_synthetic_data.py` (property walker at `:444`, meta-test
  at `:459`) is the standing guard against anyone later filling these with an estimate.
- **Restated fundamentals** — Massive serves the current XBRL view; a restatement rewrites history.
  Mitigated by keying on `acceptance_datetime`, but flag any period with multiple acceptance instants.
- **Intraday 1-min bars** — Alpaca `feed=iex` is not consolidated tape; label the source (§17 5m/30m/1h).
- **`ESTIMATED` earnings dates** — derived, not observed; carry `event_status` into every downstream payload.

### 7.4 Mandatory look-ahead test (§96) — build on the existing fixture base

`tests/test_events_lookahead.py`, modelled on the two strongest precedents in the suite:

- **Sentinel-injection**, copying `tests/test_risk_phase_b_invariants.py:297::test_walk_forward_never_sees_the_day_it_forecasts`:
  seed the DB with the real fixture set, then plant a sentinel **after** T — an article at `T+1h`, a
  financial with `acceptance_datetime = T+1d`, a bar dated `T+1d`, and the event's own actual result.
  Assert `analysis(as_of=T)` is **bit-identical** to the run without the sentinel, **and** that
  `analysis(as_of=T+2d)` *differs* — the second assertion is what proves the sentinel is not inert. That
  paired assertion is the whole reason the risk test is trustworthy and must be reproduced here.
- **Prefix-truncation**, copying `tests/test_backtest.py:97::test_no_look_ahead_closed_trades_identical_on_truncated_series`:
  generate at as_of=T against a full DB and against a DB truncated at T; dataclass equality on the entire
  evidence bundle, not approximate comparison.
- **Static guard**, copying `test_property_e_no_code_path_passes_statistical_caps_into_assess`
  (`test_risk_adversarial.py:1659`): AST-walk `libs/trading_core/events/` and assert **no module imports
  `libs.market_data` or `libs.event_calendar`** — mechanically proving the pure analysis layer cannot reach a
  live API. This is the cheapest and most durable of the three.
- Fixtures: `tests/conftest.py`'s `client` (both providers `stub`, `STUB_ANCHOR_DATE=2025-11-03` freezing the
  synthetic universe) and `unconfigured_client` (all providers unset) carry over unchanged. Caveat: the suite
  runs on **sqlite**, so JSONB operators, GIN indexes and CHECK constraints are invisible — any JSONB-specific
  event query needs a parity pin, the lesson migration 017 was written to fix.

Spec §62–66, §46–52 / §79–84 / §97, §53–61 / §89–91. Paths are relative to the repository root.

## 8. Risk integration plan (§62–66) — SHADOW mode

### 8.1 The pattern to copy (exact symbols)

There is **no `apps/gateway/pretrade.py`**: shadow orchestration lives in
`services/apps/gateway/routers/orders.py`; the pure math in `services/libs/trading_core/risk/pretrade.py`.
The shipped three-layer stack is what EventRisk joins rather than reinvents:

- `routers/orders.py:446 _statistical_shadow_detail(build)` — Phase B compact current-book block.
- `routers/orders.py:722 _pretrade_statistical_shadow(...)` — Phase C proposed-book comparison + caps.
- `routers/orders.py:1007 _pretrade_stress_shadow(...)` — Phase D stress caps, merged into the *same* verdict.
- `libs/trading_core/risk/pretrade.py:1119 shadow_verdict(approved_qty, caps, *, mode=MODE_SHADOW)` → `ShadowVerdict(hypothetical_decision, hypothetical_quantity, binding, caps, mode)`; a cap only ever REDUCES (`quantity = min(approved_qty, min(cap.cap_qty))`).
- `libs/trading_core/risk/pretrade.py:700 QuantityCap(code, layer, cap_qty, sentence, measured)` satisfies the structural `ExtraCap` Protocol at `libs/trading_core/risk/engine.py:206`.
- Emission: `routers/orders.py:2075` writes the RISK_DECISION audit `"shadow"` dict (`liquidity`/`statistical`/`vol_targeting_ewma`), assembled at `orders.py:1834,1846,1879,1935`.

**The SHADOW guarantee is mechanical.** `libs/trading_core/risk/engine.py:388 assess(..., extra_caps: Sequence[ExtraCap] = ())` is the only binding seam, and `services/tests/test_risk_adversarial.py:1664–1730` AST-walks every module in `apps/` asserting no `assess`/`assess_income` call passes `extra_caps`, and that the only `extra_caps=` keyword in `apps/` targets `_pretrade_statistical_shadow`. **EventRisk must never pass event caps to `assess`** — that makes §65 ("do not initially block trades … start in SHADOW mode") true by construction, and the adversarial test keeps it true.

### 8.2 EventRisk snapshot fields (§63)

New pure module `libs/trading_core/risk/event_risk.py` (Tier-0 style: no I/O, no `apps/` imports), returning a frozen dataclass mirrored 1:1 into audit/preview JSON:

| Field | Source | Today |
|---|---|---|
| `event_type`, `event_id`, `scheduled_at`, `event_status` | events table (Parts 1–2, §5–6) | new |
| `time_to_event_days` | `as_of`→`scheduled_at`, NY session-aware | new (needs §10 calendar) |
| `historical_event_move` median-abs / p75 / p90 / max + **`sample_size`** | `StockBarDaily` via `routers/analysis.py::ensure_daily_bars` | ✅ |
| `current_implied_move` | Alpaca `/v1beta1/options/snapshots` (greeks+IV per probe) | ✅ |
| `historical_implied_move` | option DAILY bars only — **approximation, label it** | ⚠ |
| `position_exposure_usd`, `portfolio_exposure_pct_nav` | `Position` rows + `apps/gateway/risk_snapshot.py:945 build_risk_snapshot` | ✅ |
| `option_gamma` / `option_vega` / `option_theta` | greeks already in the snapshot build | ✅ |
| `expected_iv_crush`, `historical_iv_crush` | **not computable** — no historical IV/quotes (probe) | ❌ null |
| `event_risk_state` ∈ LOW/MODERATE/HIGH/EXTREME | deterministic from the above | new |

§63 "do not let LLM alone assign this state" is enforced structurally: the state is computed in
`event_risk.py` from `Settings` thresholds (house rule: a parameter, never a hardcoded truth), and the LLM
schema (§9.2) **has no field for it** — the model may cite the state, never emit it. §64 honesty:
`sample_size` is required and rendered inline ("based on 8 events"); below `N_MIN` the percentiles return
`None`, not a number — the honest-null discipline of `_statistical_shadow_detail` (`orders.py:446`).

### 8.3 Pre-trade event gate — WARN-only (§65)

Add `_pretrade_event_shadow(...)` beside the three existing helpers, emitting a fourth key into the
`"shadow"` dict at `orders.py:2075`: `{"liquidity":…, "statistical":…, "vol_targeting_ewma":…, "event":…}`.
It builds `QuantityCap(code="EVENT_RISK", layer="EVENT", cap_qty=…, sentence="Earnings in 1.3 days; historical median move 7.1% vs current implied 8.8% …")` and folds it through the SAME `shadow_verdict(...)` so one hypothetical reflects every shadow layer at once (Phase D precedent, `orders.py:1023–1024`) — passing **nothing** to `assess`. `"EVENT"` must be added to `CAP_LAYERS` in `pretrade.py` or `QuantityCap` raises on the unknown layer. Failure mode is the established one: an exception becomes
`{"note": f"{type(exc).__name__}: {exc}"}` (`orders.py:1832,1869`), so the real decision is byte-identical
whether EventRisk ran or crashed. Promotion to RESIZE/REJECT is a later human step gated on §86 backtesting; it means populating `extra_caps` and updating `test_risk_adversarial.py:1664` — that test failing is the intended tripwire.

Alerting needs a **producer**: `classify()` (`apps/gateway/alerts.py:181`) only classifies audit rows that already exist and cannot fire on the passage of time. The scheduler writes `EVENT_APPROACHING` with T-minus in `details`; one `ALERT_RULES` entry (`alerts.py:152`) with a predicate modeled on `_risk_decision_is_alert` (`alerts.py:99`) makes it a WARNING and feeds the §83 `event_risk_warnings` gauge.

### 8.4 Trade Plan surfacing (§65, §66)

`orders.py:2101+` already mirrors shadow content onto the preview (`risk_out["comparison"]`) so the Trade
Plan panel renders without a second round trip, and `preview` is stored verbatim in `TradePlanRow.preview` (`apps/gateway/db.py:112`). EventRisk takes the identical path (`risk_out["event"] = event_shadow`), so a stored plan reproduces the event context that existed at generation. The panel renders the §65 block (EVENT RISK / Earnings in / Historical median move / Current implied move / Position sensitivity) plus the §66 options row — Gamma / Vega / Theta / Event IV, with Expected and Historical IV crush as **"Unavailable"** — and a `<Term>` card (§10.4) explaining that a long call can lose money on a correct direction when realized move < priced implied move.

## 9. LLM architecture plan (§46–52, §79–84, §97)

### 9.1 EventEvidenceBundle (§46) + no-recalculation (§47)

Frozen dataclass in `libs/trading_core/events/evidence.py` (pure), assembled by one thin gateway seam — the split `apps/gateway/risk_snapshot.py` states for itself. §46 fields: `event`, `as_of`, `previous_event`,
`previous_event_results`, `previous_market_reaction`, `fundamentals`, `price_analysis`, `options_analysis`,
`news_clusters`, `macro_context`, `peer_context` (Massive `/v1/related-companies`, entitled per probe),
`source_metadata`. §47 is enforced by *shape*: every number arrives pre-computed and pre-formatted (value +
unit + `as_of` + `source`), so the bundle carries no raw series to re-derive from. Unavailable numbers are explicit `null` + `reason` code — the `deps.py` machine-readable-code convention (`market_data_unavailable_reason:86`, `llm_unavailable_reason:115`) — which is what makes §79 ("No verified guidance data available") followable rather than hoped-for.

### 9.2 Schema-validated structured output (§80)

Reuse `libs/llm` as-is, adding an `EventAnalysisProvider` Protocol beside `RecommendationProvider` (`libs/llm/provider.py`). OpenAI is the reference: `libs/llm/openai.py:191` sends
`{"type":"json_schema","schema": _OUTPUT_SCHEMA}` in strict mode (`openai.py:44`), and
`openai.py:378 _parse_entry` validates/coerces every field and **drops** malformed entries with a log line
rather than raising; `RecommendationDraft.__post_init__` (`libs/llm/provider.py`) is the second ring (range checks). Mirror all three for `EventAnalysisDraft` with §80's fields — `executive_summary`,
`positive_catalysts[]`, `negative_catalysts[]`, `key_changes[]`, `market_expectations[]`, `key_unknowns[]`,
`scenario_upside/base/downside` (§51: each with `what_would_have_to_happen`, `conditions`,
`why_market_reacts`, `evidence_refs`), `confidence` ∈ {HIGH,MODERATE,LOW} (§50), §50's
`what_would_invalidate_this`, and `evidence_refs[]`. §92: no numeric-probability field exists in the schema,
so "82.47%" is unrepresentable.

**Blocking provider caveat:** `libs/llm/anthropic.py` implements only `generate()` — there is **no
`enrich()`** (`__init__:105`, `generate:130`, `_parse_entry:214`). Any Anthropic event path must be written,
not assumed. Also `apps/gateway/runtime_config.py::ALLOWED_PROVIDERS` whitelists only `{"", "openai", "stub"}` for `llm_provider`, so anthropic is reachable via `.env` but not the Settings UI.

### 9.3 Prompt-injection isolation (§81)

Grounding is currently a flat labelled block (`libs/llm/openai.py:128 _format_articles`) with no delimiter discipline. Harden: (1) render each article in a fenced, indexed envelope with a nonce delimiter and a preamble declaring the region **evidence, never instructions**, stripping that delimiter from article text; (2) never place article text in the system prompt — user role only; (3) rely on post-hoc enforcement, which already exists — `apps/gateway/routers/recommendations.py:286–295` DROPS any draft citing a URL absent from the stored-news map, or whose ticker is absent from the cited articles' ticker lists, so an injected "recommend XYZ" cannot survive; extend the same drop loop to `evidence_refs[]` against the bundle's evidence-ID set; (4) `news_articles.source_id` UNIQUE (`migrations/012_news_articles.sql:13`, mirrored
`db.py:154`) stops an injected duplicate from multiplying its own apparent corroboration.

### 9.4 Layered summarization + token accounting (§82)

Four persisted layers keyed by content hash so refresh re-sends only what changed:
`article → cluster_summary → theme_summary → event_research`. Layers 1–3 in the news/cluster tables; layer 4
in `event_analyses` (TradePlanRow-shaped: full payload JSON + `versions` JSON + `generated_at`). §73 incremental refresh re-summarizes only clusters whose member set changed — Massive
`get_news(limit, published_after)` (`libs/market_data/massive.py:768`) supports a watermark, Alpaca
`get_news(limit)` (`libs/market_data/alpaca.py:872`) does **not**, so that path over-fetches and de-dupes
on `source_id`.

Token accounting needs a real change: **nothing in `libs/llm` returns usage today.** Add
`prompt_tokens`/`completion_tokens` to the provider return shape, then declare metrics beside the
incrementing code (the `monitor.py`/`risk_snapshot.py` convention) via `libs/common/telemetry.py::REGISTRY`:
`llm_token_usage` counter (labels `provider`,`model`,`layer`); `llm_analysis_latency` and
`evidence_refresh_latency` histograms (**ms** buckets — `telemetry.py:34 DEFAULT_BUCKETS_MS`);
`stale_analysis_count` gauge recomputed at scrape via `main.py:204 _refresh_freshness_gauges`.

Cost guard: the DEVLOG 2026-08-18 (18) lesson applies — a 15s-polling UI against a build-on-read endpoint writes thousands of rows/day. The event-analysis endpoint must be READ-ONLY over `event_analyses` (precedent:
`routers/analysis.py:590 get_symbol_catalyst`, which never calls the LLM); generation happens only on the
scheduler tick or an explicit user action.

### 9.5 Prompt versioning + audit fields (§84)

`PROMPT_VERSION` as a module constant per prompt builder, bumped on any prompt-text change and written into
the payload, never backfilled — the `Recommendation.llm_model` precedent (`apps/gateway/db.py:459`, migration 014, default `""` = honest-unknown). The §84 record (`event_id`, `as_of`, sources, source timestamps, quant model versions, `llm_model`, `prompt_version`, evidence IDs, `analysis_version`) lands in
`event_analyses.versions` plus an `EVENT_ANALYSIS_GENERATED` audit row written by
`apps/gateway/audit.py:14 record()` in the **same transaction** (ADR-003), auto-filling `correlation_id`
from `libs/common/telemetry.py::request_id_var` — so a package is joinable to the exact `X-Request-ID` that produced it. New `AuditAction` members are required (`libs/trading_core/models/enums.py:14` — 28 members, none event-related); `ActorType.LLM` already exists.

### 9.6 Failure behaviour (§97)

Already house style: `require_llm_provider` → 503 `LLM_NOT_CONFIGURED` (`apps/gateway/deps.py:161`), and
`test_portfolio_risk_degrades_instead_of_503` (`tests/test_no_synthetic_data.py:205`) pins degrade-don't-503
for composite views. The event detail endpoint must **not** 503 on LLM failure: return DATA + QUANT blocks with `analysis: null` and `analysis_unavailable_reason: "LLM_CALL_FAILED"`; the UI renders **AI ANALYSIS UNAVAILABLE**. Add every new event endpoint to `tests/test_no_synthetic_data.py::MARKET_DATA_ENDPOINTS` (line 44) or the property walker (`test_no_response_contains_a_market_number:444`) silently stops covering it.

## 10. UI plan (§53–61, §89–91)

### 10.1 Catalysts nav destination (§53)

Two coordinated edits to bilingual arrays — no new i18n machinery:

- `ui/components/shared/Nav.tsx:9 SECTIONS` — insert `{ href: "/catalysts", en: "Catalysts", zh: "催化剂" }` between Recommendations and Watchlist (research-stage adjacency).
- `ui/components/shared/FlowNav.tsx:27 FLOW_STAGES` — an 8-stage strip whose `FlowStageId` (`FlowNav.tsx:15`) is a closed union. Cleanest: keep 8 stages and let stage 2 "Research" cover the catalyst surface; adding a 9th chip renumbers every literal label (`"1 Connect"…"8 Audit"`). The trailing link auto-targets `/guide#stage-<id>`, so a matching Guide anchor is required.
- Routes: `ui/app/catalysts/page.tsx` (grouped **Today / Next 7 Days / Next 30 Days**) and `ui/app/catalysts/[eventId]/page.tsx`. Within-group ordering follows §12 POSITION > TRADING POOL > WATCHLIST > MARKET-WIDE, all already queryable rows.
- **Collision:** `GET /api/analysis/{ticker}/catalyst` exists (`apps/gateway/routers/analysis.py:590 get_symbol_catalyst`, pinned by `tests/test_catalyst.py`). Use `/api/events` for the new registry and link the old per-ticker view from the event page rather than forking a second "catalyst" concept.

### 10.2 Calendar cards (§54), event detail tabs (§55–56)

Card = existing `.card` + `.badge` vocabulary (`ui/app/globals.css:160–169`, `230–234`): ticker, event type, date + BEFORE/AFTER MARKET, POSITION EXPOSURE, TRADING POOL (`badge.on`/`badge.off`), Historical Event Move, Current Implied Move, Analysis Status (`badge.green` READY / `.badge.stale` amber STALE), `[Open Research]`. §54 "do not overload": nine fields max, no narrative on the card.

Detail tabs per §55 (Overview / Previous Event / Since Last Event / Fundamentals / Price / Options / News / Scenarios / Risk / Evidence). The tabbed-panel + honest-null idiom exists in
`ui/app/watchlist/[ticker]/page.tsx` and `ui/components/risk/*` (`StatisticalRisk.tsx`,
`StressScenarios.tsx`, `ModelValidation.tsx`, `TradeComparison.tsx`) — the Risk tab should reuse
`TradeComparison.tsx`'s shadow-verdict presentation for §8.3. §56 hero: `NVDA — Qx Earnings`, `T-2 DAYS`,
scheduled datetime **with exchange tz**, freshness `as_of`, status badge CONFIRMED / **ESTIMATED** (the probe means upcoming earnings dates are derived until confirmed — that must be visible), exposure, risk status.

### 10.3 DATA / QUANT / AI ANALYSIS visual language (§49, §91)

**Do not invent a palette.** `ui/app/globals.css:633–639` defines the provenance tag —
`.provenance.data-driven` (accent blue) / `.provenance.llm-generated` (amber) — used at
`ui/app/watchlist/[ticker]/page.tsx:791,904,1107`, `ui/components/options/OptionChainTable.tsx:745`,
`ui/components/shared/RiskMethodModal.tsx:158`. §91 needs a **third** tier: add one rule
`.provenance.quant-derived` in a distinct low-chroma token validated against the `#161b22` panel surface
(per the dataviz note at `ui/components/charts/CandlestickChart.tsx:7–19`), and render section headers as
`DATA` / `QUANT` / `AI ANALYSIS`. `.badge.llm` (amber, `globals.css:165`) stays the row-level AI marker.

### 10.4 ⓘ explainability (§90)

`ui/components/shared/Term.tsx` + `ui/lib/glossary.ts` (65 bilingual entries, `{en,zh} × {name, short,
read}`) is exactly §90's mechanism — portaled fixed-position cards that escape `overflow` clipping, one open at a time, unknown key renders children unchanged. **No new component**: add keys (`event_risk_state`,
`historical_event_move`, `implied_move`, `iv_crush`, `surprise_threshold`, `event_status_estimated`,
`sample_size`) and wrap labels in `<Term k="…">`. The content policy at `glossary.ts:1–8` ("never promise
predictive power") already aligns with §37 (implied move is not a forecast) and §92.

### 10.5 Timeline (§57) and charts (§58–61)

**There is no charting library** — `ui/package.json` dependencies are only `@tanstack/react-query`, `next`,
`react`, `react-dom`. Every chart is hand-authored inline SVG: `ui/components/charts/CandlestickChart.tsx`
(960×420 viewBox, explicit geometry, tooltip, legend, CVD-aware polarity carried by *fill state* as well as hue), plus inline SVG in `ui/app/backtests/page.tsx` and `ui/app/watchlist/[ticker]/page.tsx`.

To build in `ui/components/charts/`, following those conventions (design tokens not literals, viewBox + padding, legend/tooltip naming every mark, never hue-only encoding): `EventTimeline.tsx` (§57 — LAST EARNINGS → dated developments → TODAY → NEXT EARNINGS; the spec's highest-value component),
`ImpliedVsActualChart.tsx` (§60), `EventReactionDistribution.tsx` (§60, `sample_size` printed on-chart), and
`MacroReactionChart.tsx` (§39). §58 prefers the PREVIOUS/CURRENT/CHANGE comparison table over a raw table —
plain markup, no chart. §61: the backend returns structured chart data only; the LLM never draws. §59 News UI = KEY THEMES with per-theme development counts, expandable to source articles, backed by §9.4's layers.

### 10.6 i18n zh/en (§89) and modal/toast conventions

`ui/lib/i18n.tsx` provides `LanguageProvider`/`useLang`/`useT`, persisted to `localStorage["lang"]`, default
`en`. The mechanism is a **call-site `t(en, zh)` pair**, not a key dictionary — each new string is written
bilingually in place (`ui/lib/i18n-labels.ts` holds shared label sets). Documented scope boundary at
`i18n.tsx:5–9`: UI chrome is bilingual, **server-generated strings stay verbatim English** as audit-worthy
exact records. For catalysts: tabs, card labels and glossary entries → bilingual; LLM narrative, reason codes, audit text and `event_status` enums → verbatim. `Nav.tsx:31–44` also retargets `llm_output_language` on a language switch (fire-and-forget), so newly generated research follows the interface language while stored analysis keeps the language it was generated in.

§89: browser-native `alert`/`confirm`/`prompt` are **CI-blocked** by `ui/scripts/check-native-dialogs.sh` (`npm run check:dialogs`, grep over `app/ components/ lib/`). Use `ui/components/shared/Modal.tsx` (focus trap, ESC, ARIA dialog, focus return), `ConfirmDialog.tsx` for consent, `Toast.tsx` for outcomes — severity INFO / SUCCESS / WARNING / CRITICAL matching the badge maps, CRITICAL persisting until dismissed.
`NotConfigured.tsx` / `Placeholder.tsx` are the existing honest-absence surfaces for §97 **AI ANALYSIS
UNAVAILABLE**, §98 **Implied Move: Unavailable** and **CONSENSUS DATA UNAVAILABLE** — render the event, never hide it.

## 11. Adjusted phase plan (§93)

### 11.0 Deviations from the spec's phase order, with rationale

The spec's order is B → C → D → E → F → G → H → I → J → K → L. Five deviations:

1. **E (Fundamental + Price Context) moves ahead of C and D.** §15/§16/§17 ("previous comparable event", "what changed since") are *defined in terms of* fundamentals and price history. There is no fundamentals ingestion at all today — the only occurrence of the word is a docstring in `services/libs/market_data/__init__.py`. Building C before E would mean building replay against a data substrate that does not exist. Price context is cheaper: `services/apps/gateway/routers/analysis.py::ensure_daily_bars` already stores complete-days-only daily bars with a `DATA_BACKFILL` audit, so E splits into E1 (price, mostly wiring) and E2 (fundamentals, a full workstream matching the user's own standing next-step note).
2. **J (UI) is not a single terminal phase — it is incremental, one slice per backend phase, with a usable Catalysts page landing at the end of B.** §53/§54 describe a calendar that is useful the moment events exist. `ui/components/shared/Nav.tsx::SECTIONS` is a flat ten-entry array; adding `/catalysts` is one line, and shipping it early gives every later phase a visible acceptance surface instead of a nine-phase blind build.
3. **I (Options / Implied Move) moves ahead of G and H.** The live probe makes current implied move fully available (Alpaca `/v1beta1/options/snapshots` carries greeks + IV, already consumed by `services/apps/gateway/routers/options.py`), while G/H require net-new primary-source adapters (BLS/BEA/FOMC HTML+RSS) that nothing in the repo resembles. I is high-value/low-risk; G/H are low-certainty/high-effort. Ship the certain thing first.
4. **A dedicated B0 sub-phase precedes everything: naming + time infrastructure.** `GET /api/analysis/{ticker}/catalyst` already exists (`routers/analysis.py:590` `get_symbol_catalyst`, pinned by `services/tests/test_catalyst.py`) and `Recommendation.catalyst_type` already exists (`services/apps/gateway/db.py:451`). The new router must be named `events` and the collision resolved deliberately, not discovered in phase J. B0 also lands the market-calendar table, because `routers/analysis.py::_last_expected_trading_date` does Mon-Fri arithmetic and its docstring admits "Holidays are not modeled" — §6's before/after-market classification cannot be correct without it.
5. **K (risk shadow) is bounded to SHADOW-only and is explicitly *not* allowed to touch `assess()`.** `services/tests/test_risk_adversarial.py:815` is an AST-level test proving no `assess()` call in `apps/` passes `extra_caps` — it fires even on an empty tuple. That test is the mechanical guarantee of §65 and must stay green through K; K therefore delivers a shadow block in the `RISK_DECISION` audit details, nothing more.

Standing obligations for **every** phase below (§94): a raw-SQL migration numbered contiguously (`services/tests/test_migration_parity.py::test_migration_numbers_are_contiguous` forces exactly `021_*.sql` next, no gaps); an individual `:ro` mount added to `services/docker-compose.yml` in the same commit (`test_every_migration_is_mounted_in_docker_compose`); the ORM model in `services/apps/gateway/db.py` mirroring it, plus the table added to `_SINGLE_CREATE_TABLES` at `services/tests/test_migration_parity.py:83-93` so parity is actually checked (the tripwire is opt-in per table); every new market-facing endpoint added to `MARKET_DATA_ENDPOINTS` in `services/tests/test_no_synthetic_data.py:44`; and a manual live apply step, because migrations only auto-run on a fresh volume via `/docker-entrypoint-initdb.d` — 018/019/020 were applied by hand per DEVLOG.

### 11.1 Phase B — Event Registry + Calendar Providers (§5-§11, §75)

- **Deliverables.** `events` table + typed taxonomy; `market_calendar` table (date, open_utc, close_utc, session_type) from Alpaca `/v2/calendar` and Massive `/v1/marketstatus/upcoming`; ingestion scheduler; `GET /api/events` (horizon + relevance filters); T-minus-7 alert.
- **Modules.** `services/migrations/021_events.sql` + `022_market_calendar.sql`; ORM rows in `services/apps/gateway/db.py` beside `RiskSnapshotRow`; `EventType/EventStatus/EventSession` StrEnums in `services/libs/trading_core/models/enums.py` beside `PlanStatus`; new `services/libs/event_calendar/` package replicating the registry contract of `services/libs/market_data/__init__.py::_PROVIDERS` (no default, `ProviderNotConfigured` on empty, `ValueError` on unknown, no cross-provider fallback); `services/apps/gateway/event_calendar_sync.py` as the gateway seam, modelled line-for-line on `services/apps/gateway/risk_snapshot.py` (`NEW_YORK` at :142, `new_york_today` :1979, `_scheduled_exists_today` :1984, `run_scheduled_snapshot` :2006 returning named `{"skipped": ...}` reasons); new router `services/apps/gateway/routers/events.py`.
- **Integration.** One task appended to `background_tasks` in `services/apps/gateway/main.py::lifespan` (:118) behind a new `settings.event_calendar_interval_seconds` knob (0 disables), matching the four interval-gated loops there; one `app.include_router(events.router)` line in `create_app`; new `AuditAction` members `CALENDAR_INGESTED / EVENT_DISCOVERED / EVENT_UPDATED / EVENT_APPROACHING`; one `ALERT_RULES` entry in `services/apps/gateway/alerts.py:152-163` with a T-minus predicate modelled on `_risk_decision_is_alert` (:99).
- **Tests (§95).** event-date normalization; timezone conversion; duplicate events (DB-level unique on `(source, source_event_id)` — correctness must not rely on ADR-007's single-process assumption); rescheduled events; canceled events; before-market / after-market earnings classification via the calendar table; provider outage; subscription denied (Benzinga 403 surfaces as a capability `false`, not a crash); stale data.
- **DEVLOG (§99).** Purpose; Existing Capability (the scheduler/audit/alert precedents reused, named by path); Architecture Decision (new ADR-008: event calendar is a separate provider registry from market data — and ADR-009 if the `catalyst` endpoint is superseded); Data Providers (Alpaca calendar 200, Massive holidays 200, Benzinga earnings 403); New Models; Implementation; Tests; Known Limitations (upcoming earnings are ESTIMATED); Subscription Dependencies; Next Step.
- **Done.** `GET /api/events` returns real, deduped, typed events with UTC + exchange-tz for at least the watchlist; alert fires exactly once per (event, horizon) across a restart; unconfigured-provider mode returns 503 with a machine-readable code, never a fabricated date.

### 11.2 Phase E1 — Price context (§17, §31, §32, §39)

- **Deliverables.** Pre-event run-up / drawdown / distance-from-highs; post-event daily reaction for past events; ADR-005 reference-symbol list extended to the §39 macro proxies (QQQ/GLD/USO/TLT/IEF/SHY/UUP) with a new ADR recording why.
- **Modules.** Pure computation in `services/libs/trading_core/` (no I/O, the house split); reads go through `routers/analysis.py::ensure_daily_bars`. Intraday (5m/30m/1h) writes into the **existing, never-used** `stock_bars_1m` hypertable from `services/migrations/002_system_state_and_bars.sql:30`, scoped to event windows only.
- **Tests.** as-of enforcement on every price read; stale data; provider outage; missing bars → named exclusion, not a zero.
- **Done.** Price-cycle numbers render for a past event with an explicit `source`/`feed` label (iex is not consolidated tape).

### 11.3 Phase E2 — Fundamentals (§16, §28, §29, §30, §35)

- **Deliverables.** Massive `/vX/reference/financials` adapter; `fundamentals` table keyed `(ticker, fiscal_period, filing_date, acceptance_datetime)`; derived §28 ratios; previous-vs-current delta.
- **Modules.** New method on `services/libs/market_data/massive.py::MassiveProvider`; ratios as a pure module in `services/libs/trading_core/`.
- **Tests.** missing fundamentals; as-of enforcement filtering on `acceptance_datetime <= as_of` (never on fiscal period end); future-data leakage.
- **Done.** "What changed since the last event" is answerable from stored point-in-time rows alone; consensus-dependent fields (EPS surprise %) return honest nulls with a reason code, per §33/§98.

### 11.4 Phase C — Previous event / replay (§15, §19, §20)

- **Deliverables.** `previous_event_id` linkage; SEC EDGAR submissions adapter for authoritative past 8-K Item 2.02 `acceptanceDateTime`; multi-event history view.
- **Tests.** previous-event matching; historical replay; look-ahead (§96) — mirror the mutate-only-the-last-observation style of `services/tests/test_risk_adversarial.py`, asserting every earlier output is bit-identical.
- **Done.** For a watchlist name, the last four earnings events carry a real timestamp and a real market reaction.

### 11.5 Phase D — News evidence engine (§21-§27, §79, §81)

- **Deliverables.** cluster_id / canonical article / materiality / novelty / source-quality on top of the existing store; a `(ticker, published_at)` index.
- **Modules.** Extend `services/apps/gateway/routers/recommendations.py::_refresh_locked` rather than duplicating it — it already dedupes on `news_articles.source_id` (UNIQUE, `services/migrations/012_news_articles.sql:13`, mirrored `db.py:154`), audits `NEWS_INGESTED`, and **drops LLM drafts citing unstored articles**, which is §27/§79 already in code.
- **Constraint.** `massive.py:768 get_news(limit, published_after)` supports watermarking; `alpaca.py:872 get_news(limit)` does **not** — the incremental loop must over-fetch and dedupe against Alpaca.
- **Tests.** news deduplication (syndicated same-story, different ids); story clustering with empty bodies (probe: Alpaca content sometimes empty); LLM evidence mismatch; prompt-injection safety.

### 11.6 Phase F — Earnings Intelligence (§16-§19, §33-§35, §52)

Assembles B/C/D/E into one evidence bundle + structured LLM output. Modules: pure bundle assembly in `libs/trading_core/`, one gateway seam that persists an `event_analyses` row shaped like `TradePlanRow` (`db.py:112-139` — full payload JSON, `versions` JSON, `market_data_as_of`, status enum, `superseded_by`, `created_by`) with `llm_model` recorded at generation and never backfilled (`db.py:459` precedent). Tests: LLM malformed response; LLM evidence mismatch; §97 — analysis unavailable must still render every deterministic block. Done: a cached research package with a `(event_id, as_of_bucket, data_version, analysis_version)` identity and a persist-dedupe guard (DEVLOG 2026-08-18 (18): a 15s-polling UI against a build-on-read endpoint writes thousands of rows/day).

### 11.7 Phase I — Options / implied move (§36, §37, §66)

Current implied move from the live Alpaca snapshot; historical straddle-implied move reconstructed approximately from option **daily bars** and labelled as an approximation, exactly as `services/migrations/019_stress_runs.sql` labels `RV_PROXY` and 018's `atm_iv_daily.source` labels internally-computed IV. Tests: option data unavailable → "Implied Move: Unavailable", event still rendered.

### 11.8 Phase J — Catalyst UI (incremental)

`ui/app/catalysts/page.tsx` + `ui/app/catalysts/[eventId]/page.tsx`, one Nav entry, bilingual per the existing `SECTIONS` en/zh shape. §91's data-vs-LLM visual language reuses the existing `generated: true` flag convention from `get_symbol_catalyst`. Ships in slices: calendar at end of B, price/fundamental blocks at E, news at D, package at F, implied move at I.

### 11.9 Phases G / H — Macro and Fed (§38-§45)

Net-new primary-source adapters (BLS schedule HTML + API v2, BEA, FOMC calendar HTML, Fed RSS, Treasury CSV for the missing 2y), each behind its own `runtime_config.CONFIG_KEYS` entry. Tests: macro release parsing; Fed event classification; provider outage. Highest schedule risk in the plan (HTML scraping of third-party pages).

### 11.10 Phases K / L — Risk shadow, then replay validation (§62-§65, §85, §86)

K: an event-risk block inside the `RISK_DECISION` audit details alongside `shadow.statistical` / `shadow.vol_targeting_ewma`, mirrored into the orders response; `extra_caps` stays `()`. L: the §96 look-ahead suite plus a §86 measurement of whether event features are predictive — measured, never assumed.

---

## 12. Open questions for the user

| # | Question | Autonomous default if unanswered |
|---|---|---|
| 1 | **Massive Benzinga Earnings add-on** — purchase it? `/benzinga/v1/earnings` is 403 today, and it is the single authoritative source for upcoming earnings dates *and* the only source for consensus/surprise/guidance/revisions (§33, §34, §16 surprise %). | Do not purchase. Implement ESTIMATED dates from filing cadence (SEC 8-K history + Massive `filing_date`), and mark §33/§34 permanently as CONSENSUS DATA UNAVAILABLE via the capability probe. Add `earnings_calendar` as a probed key in **both** `massive.py::probe_capabilities` (:971-976) and `alpaca.py::probe_capabilities` (:928-933) so the key set stays uniform. |
| 2 | **Optional third-party earnings-calendar key** (Finnhub / AlphaVantage / FMP)? §2 forbids silently adding a dependency. | Build the adapter seam but ship it **disabled**: one `CONFIG_KEYS` entry in `services/apps/gateway/runtime_config.py:34` (+ `SECRET_KEYS` :64), empty by default, so no network call is ever made until a key is entered in Settings. |
| 3 | **Alert channel for the T-minus-7 alert** (§11) — in-app only, or push/email? There is no delivery mechanism today; `GET /api/alerts` (`routers/alerts.py:43`) is pull-only. | In-app only: write an `EVENT_APPROACHING` audit row and add one `ALERT_RULES` entry. ADR-006 already names push as a *future* consumer of the same classification, so this choice is not load-bearing. |
| 4 | **Default display timezone.** | `America/New_York` — already the de-facto platform constant (`risk_snapshot.py:142 NEW_YORK`, `analysis.EASTERN`, `massive.EASTERN`). Store UTC, render NY, expose the event's own exchange tz per §10. Reuse the existing constant; do not define a fourth. |
| 5 | **May ESTIMATED earnings dates appear on the calendar by default?** | Yes, but never silently: `event_status=ESTIMATED`, `source=derived`, a visible UI badge, and excluded from the T-minus-7 alert (an alert on a guessed date is worse than no alert). A `settings` flag lets the user hide them entirely. |

Secondary, lower-stakes: (6) whether `GET /api/analysis/{ticker}/catalyst` should be superseded or kept — default is **keep and coexist**, since `services/tests/test_catalyst.py` pins its contract and the UI consumes it; (7) `anthropic` is reachable via `.env` but not whitelisted in `runtime_config.ALLOWED_PROVIDERS` — default is to leave that unchanged, as it is out of scope for this spec.

---

## 13. Known limitations & NOT BACKTESTABLE list

**Marked NOT BACKTESTABLE per §85** (the provider cannot serve point-in-time history at all — these fields must carry the label in the payload, not be silently null):

- **Historical ATM IV / IV crush / implied move around any past event** (§18, §19, §36). Neither provider sells historical option quotes, greeks or OI. `atm_iv_daily` (migration 018) only accumulates forward from the day the platform started computing it. Daily-bar reconstruction is an *approximation*, labelled as such.
- **Historical analyst consensus, estimate revisions, guidance, ratings** (§33, §34, §35). Benzinga endpoints are 403; no historical vintage exists even with the add-on's current tier. EPS/revenue **surprise %** (§16) is therefore not computable for any period.
- **Upcoming earnings dates as *confirmed* facts.** Only ESTIMATED (cadence-derived) or user-confirmed. A backtest at time T cannot reproduce "the confirmed date as known at T" because that vintage was never stored.
- **Point-in-time news for any period before ingestion started.** `news_articles` only contains what the platform fetched; there is no archival backfill of the vendor's own past index.
- **Intraday reaction before the ingestion window.** `stock_bars_1m` has never been written to; anything outside the deliberately narrow event-window backfill is absent.
- **2-year Treasury yield from Massive** (§39). `/fed/v1/treasury-yields` serves 1y/5y/10y only. Treasury's daily CSV fills it; until that adapter exists, SHY is a labelled *proxy*, never reported as a 2y yield.

**Structural limitations of the platform that constrain this work:**

- **No migration runner.** `services/migrations/*.sql` execute only on a fresh volume; every catalyst migration needs a manual live apply and must use `IF NOT EXISTS` throughout so re-application is safe.
- **Test harness is SQLite, production is Postgres+Timescale.** CHECK constraints, JSONB operators, GIN indexes and hypertables are invisible to the suite — migration 017 exists precisely because that gap hid a live constraint violation. Any JSONB-dependent event query needs a parity pin.
- **Lifespan does not run under tests** (httpx ASGITransport), so the ingestion loop must split its tick into a directly callable function, as `monitor.run_sweep_and_update:87` and `risk_snapshot.run_scheduled_snapshot:2006` both do.
- **Single-process assumption (ADR-007).** No leader election, no distributed lock. All scheduled catalyst writes must be idempotent at the *database* level or a second replica double-ingests and double-alerts.
- **Holidays are not modeled today**, which is why B0 lands the calendar table before anything computes "next open".
- **No token accounting** (§82). `services/libs/llm/` providers do not return usage, so cost cannot be measured until the return shape changes.
- **Redis is provisioned and completely unused** — `Settings.redis_url` (`services/libs/common/config.py:34`) is referenced nowhere else. Per §74 and ADR-001, the analysis cache belongs in Postgres; do not adopt Redis merely because it is running.

**Subscription dependencies summary:** Massive base plan — news, financials (with `filing_date` + `acceptance_datetime`, the point-in-time key the whole as-of contract rests on), stock and option daily bars, dividends, IPOs, holidays, peers, treasury yields, CPI. Alpaca Algo Trader Plus — trading calendar, corporate actions, news, live option snapshots **with greeks and IV**, 1-minute historical bars (iex feed). Not entitled — all Benzinga endpoints (earnings, consensus, ratings, guidance, analyst insights), Alpaca logos, and any historical option quote/greek/OI data (not sold at any tier probed). Free primary sources required for G/H/C — BLS, BEA, Census, Federal Reserve, SEC EDGAR (needs a contact `User-Agent`), Treasury; each is rate-limited or HTML-scraped and must be treated as a provider that can fail, with the same honest-absence handling as the paid ones.
