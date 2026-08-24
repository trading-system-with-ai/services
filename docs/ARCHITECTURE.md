# Architecture Notes

## ADR-001: Modular monolith for V1 (2026-08-10)

The development plan (§24) lists ~20 candidate services but explicitly warns:
"Do not deploy every directory as an independent service on day one."

**Decision:** V1 runs one FastAPI process (`apps/gateway`) containing watchlist,
trading-pool, and audit modules. Each module owns its router and service logic, and
modules communicate only through explicit function boundaries — no reaching into each
other's tables. Higher-volume components (market-data ingestion, backtest workers) will
become separate containers when they arrive (Phase 1+), since they have genuinely
different scaling and lifecycle needs.

**Consequence:** docker-compose stays small (db, redis, gateway) and the split into real
microservices later is a packaging change, not an API change.

## ADR-002: Repo naming adaptation

The plan suggests a repo named `trading-platform-backend/` with a `services/` directory.
Our repo itself is named `services/`, so deployable apps live in `apps/` and shared
libraries in `libs/` (same shapes as the plan's `libs/trading_core` + `libs/common`).

## ADR-003: Audit events share the mutation transaction

`apps/gateway/audit.py` adds audit rows to the same SQLAlchemy session as the state
change. Commit is atomic: state and audit trail cannot diverge. When the event bus
arrives (Phase 4+), audit events will additionally be published to `audit.event`, but
the DB row remains the source of truth.

## ADR-004: Authorization is enforced in the write path, not the UI

- `POST /api/trading-pool` refuses tickers not currently on the Watchlist (422).
- `DELETE /api/watchlist/{t}` cascades Trading Pool removal (and audits the cascade) —
  mirrored by a DB-level `ON DELETE CASCADE` FK in `migrations/001_initial.sql`.
- Promotion always starts `trading_enabled=false`; enabling trading is a separate,
  separately-audited user action.
- Strategy allowlist is validated against account constraints (long-only V1); short
  strategies are rejected at the API boundary.

## Database

- Dev/test: SQLite (aiosqlite) via SQLAlchemy async — zero-dependency local runs.
- Docker/prod: PostgreSQL + TimescaleDB; schema in `migrations/*.sql`.
- Time-series (OHLCV, chains, features) will be Timescale hypertables managed by raw
  SQL migrations only — deliberately not ORM-mapped.

## ADR-005: System reference symbols bypass the watchlist-only data rule (2026-08-10)

> **AMENDED 2026-08-20 (user decision, DEVLOG 40):** the watchlist-only READ
> rule below is superseded — research surfaces (analysis, bars, options,
> catalyst, plan generation) serve ANY ticker via lazy backfill, and stored
> bars no longer imply membership. Watchlist membership now means continuous
> tracking + backtest eligibility; the Trading-Pool membership precondition
> is enforced on BOTH promote paths (direct and plan-apply). ADR-005's
> INDEX_SYMBOLS carve-out remains meaningful for the always-maintained
> dashboard quote set.

Plan §4.2 restricts stored historical data to Watchlist symbols —
`GET /api/watchlist/{ticker}/analysis` 404s for anything else. The Market Regime
Engine (plan §6.1), however, must read broad-market index data (SPY daily bars today;
QQQ/VIX as further inputs later) regardless of what the user watches.

**Decision:** SPY, QQQ and VIX are *system reference symbols*, exempt from the
watchlist-only rule for exactly that reason: `GET /api/market/overview` may lazily
backfill and store their daily bars without Watchlist membership. The exemption is
limited to this fixed list (`INDEX_SYMBOLS` in `routers/market.py`); every other
symbol's history remains watchlist-gated. Reference backfills go through the same
`ensure_daily_bars` path as watchlist analysis, so they are SYSTEM-attributed
`DATA_BACKFILL` audit events committed in the same transaction as the inserted bars
(ADR-003).

## ADR-006: Alerts are a classified view over the audit trail (2026-08-10)

The Dashboard alerts feed (§18/§29/§38) could have been its own table with its own
writers — a second event stream that every mutation path would have to remember to feed.

**Decision:** there is no alerts table. The audit log (ADR-003) is the single event
source; `GET /api/alerts` is a severity-graded *read* of it. A declarative
`ALERT_RULES` table (`apps/gateway/alerts.py`) maps an `AuditAction` to
`(severity, title builder, optional keep-predicate)` — any audit action absent from the
table is simply not an alert, and predicates filter routine rows (an approving risk
preview is not an alert; a rejection or veto is).

**Consequence:** alerts can never diverge from the audit record — if it alerted, it is
in the audit trail with the same id, timestamp, and correlation id, and there is no
second writer to drift. Adding or retiring an alert is a one-line rule change, not a new
write path. Future push notifications (§38) subscribe to the same classification: they
consume classified audit rows rather than inventing another event source.

## ADR-007: In-process position monitor (2026-08-10)

Mechanical exits (§11) must fire even when nobody is clicking, which requires a
periodic sweep of open positions (§26, §37). The obvious shapes are a separate
monitor service or a cron job.

**Decision:** the monitor is an asyncio background task (`apps/gateway/monitor.py`)
started and cancelled by the gateway lifespan — not a separate service or cron. This
follows the V1 modular monolith rule (ADR-001, plan §24): one process until scaling
needs prove otherwise. The task runs the SAME `run_exit_sweep` as
`POST /api/positions/check-exits` (never a reimplementation, §21 spirit), and every
sweep executes under the shared paper-execution lock, so a background sweep and a
user-triggered check can never double-sell the same position. The interval is
configuration (`POSITION_MONITOR_INTERVAL_SECONDS`, default 300); `0` disables the
task entirely, and `GET /api/positions/monitor` honestly reports whether the loop is
actually running.

**Consequence:** correctness currently rides on there being one gateway process —
horizontal scaling later requires moving the sweep to a single-owner worker (or leader
election) before running multiple gateway replicas. Because the sweep logic is already
a shared function behind a lock, that move is a packaging change, not a logic change.

## ADR-008: Event calendar is its own provider registry; one event = one row with source precedence (2026-08-19)

**Context.** The Catalyst & Event Intelligence program (`prompts/event_analy_system.md`,
audit `docs/catalyst-event-audit.md`) needs authoritative dates for earnings, macro
releases and Federal Reserve events. Neither subscribed vendor supplies an earnings
calendar (Massive `/benzinga/v1/*` = 403 "not entitled"); free primary sources (SEC
EDGAR, federalreserve.gov, BLS/BEA) do, with different shapes and failure modes.

**Decision.**
1. `libs/event_calendar/` is a **separate registry** from `libs/market_data`
   (same contract: no default, `ProviderNotConfigured` on empty, no cross-provider
   fallback), because calendar facts and price data have different authority,
   cadence and failure semantics. Providers are sync (httpx, injectable transport)
   and are called from the gateway via `asyncio.to_thread`.
2. **One event is one row** (`events.event_key UNIQUE`), described by several
   sources with different authority. A fixed source-precedence table
   (`libs/trading_core/events/models.py::SOURCE_RANK`: USER < COMPANY_IR_SEC <
   GOVERNMENT_AGENCY = FEDERAL_RESERVE < STRUCTURED_PROVIDER < DERIVED < NEWS;
   LLM may never write a date) decides who may move `scheduled_at`/`status`.
   ESTIMATED rows (deterministic cadence derivation) are promoted to CONFIRMED by
   any authoritative source, are visibly badged, and **never alert**; a confirmed
   date that moves becomes REVISED with the prior value kept in
   `revision_history`.
3. The natural key embeds the ET date; EARNINGS rows additionally reconcile by a
   ±21-day same-ticker window (`same_event`) so estimate drift and follow-up
   Item 2.02 8-Ks do not create duplicate cards. The SEC provider keeps the
   **earliest** 2.02 filing per window as the release.
4. Audit rows for events use the numeric `events.id` as `entity_id`
   (`audit_events.entity_id` is VARCHAR(64); FED_SPEECH natural keys overflowed it
   live) and carry `event_key` in `details`. The T-minus alert is written exactly
   once per (event, horizon) by checking the audit table, so it survives restarts
   and a second replica (ADR-007 has no leader election).
5. Existing `GET /api/analysis/{ticker}/catalyst` is kept; the new surface is
   `/api/events` (router `events`, not `catalyst`).

**Consequences.** Upcoming earnings dates are honest ESTIMATEs until the user (or a
future subscribed calendar) confirms them; §33/§34 consensus fields remain
"UNAVAILABLE". `SEC_USER_AGENT` must carry a contact e-mail or `www.sec.gov`
answers 403 and earnings history is skipped (named skip, never a crash).

## ADR-009: Government sources are keyless first-class adapters; macro reference symbols are a second reference set (2026-08-19)

**Context.** Phase G (§8, §38–§41, §46) needs three things the platform did not have:
published macro statistics (CPI, PPI, payrolls, JOLTS), the release SCHEDULE those
statistics arrive on, and a cross-asset view of how markets moved around a release.
No vendor in the subscription serves any of them. BLS, BEA and the Treasury all
publish theirs for free — BLS's data API and every schedule page are **keyless**,
BEA's statistics API needs a free key, Treasury's yield-curve CSV needs nothing —
and, unlike a paid feed, none of them can be switched off by a billing decision.

**Decision.**

1. **`bls` and `bea` join `_PROVIDERS` and `KEYLESS_PROVIDERS`** beside `sec_edgar`
   and `fed`, so `configured_provider_names()` returns four keyless sources by
   default and a fresh install has a macro calendar with no credentials at all.
   Every government request carries the contact `SEC_USER_AGENT` — SEC's fair-access
   policy requires it and BLS/BEA/Treasury are sent the same courtesy from the same
   single setting, because a second contact address is a second thing to forget.
2. **Statistical VALUES and calendar DATES are different registries.**
   `macro_data_provider()` is separate from `get_provider()`: they answer different
   questions and fail differently, and a BEA key that is missing must make the
   ACTUALS unavailable without touching the release dates, which stay CONFIRMED.
   `bea_macro_data_provider()` raises `CapabilityNotAvailable` until `BEA_API_KEY`
   is set — proven absence, never an estimate.
3. **`MACRO_REFERENCE_SYMBOLS` is a SECOND reference set, not an extension of
   ADR-005's `INDEX_SYMBOLS`.** `INDEX_SYMBOLS` (SPY/QQQ/VIX) is what the dashboard
   QUOTES live; `MACRO_REFERENCE_SYMBOLS` (SPY, QQQ, TLT, IEF, SHY, GLD, USO, UUP)
   is what the macro seam stores daily BARS for. Merging them would put eight extra
   REST quote calls on every dashboard poll and would ask the bar backfill for VIX,
   which the equity providers do not serve. Both sets are exempt from the
   watchlist-only data rule for the same ADR-005 reason. Every macro symbol is an
   ETF **proxy** for the exposure named and is flagged `is_proxy` in the payload —
   TLT is a fund holding long Treasuries, not "long rates".
4. **A macro observation's release INSTANT is stored, with its basis.** BLS's data
   API returns no timestamps, so the instant is joined at write time from this
   platform's own stored schedule rows (basis `SCHEDULED`, the release date at its
   published ET time) or, failing that, period end + 45 days (basis `ESTIMATED`).
   Both travel together in `macro_observations`, because "published 08:30 on the
   12th" and "probably out by mid-September" are different claims and the as-of gate
   reads both. The join key is the **reference period** (`2026-07`), never the
   release date — a release that slips a day is still the same month's print.
5. **Reads never fetch macro data.** `GET /api/events/{id}/macro` holds no provider
   handle; `POST /api/events/{id}/macro/backfill` is the only path that spends
   requests. BLS allows roughly **25 requests per day** unregistered and serves only
   the latest three years, so a lazily-topping-up read would exhaust the day's budget
   on one page load and then fail the backfill that could have repaired it.

**Consequences.** Macro release dates work with zero configuration; GDP and PCE
*actuals* stay honestly unavailable until someone registers a free BEA key, and
RETAIL_SALES actuals until a Census adapter exists (their release DATES are tracked
either way). Every macro consensus and surprise field is the fixed
`CONSENSUS DATA UNAVAILABLE` string — no code path in the macro stack computes a
surprise. `SEC_USER_AGENT` is now load-bearing for four agencies rather than one.

## ADR-010: External research is two more provider registries, admitted into the bundle, with zero execution authority (2026-08-21)

**Context.** The Catalyst research upgrade adds two external evidence
capabilities: public web search (Brave) and prediction-market pricing
(Polymarket). Both are qualitatively different from every existing data
source — search results are attacker-influenceable text, and market prices are
easily misread as forecasts — and both are metered or rate-limited in ways the
existing free-GET philosophy would abuse.

**Decision.**

1. **Two more sibling registries**, not a new framework: `libs/web_search/` and
   `libs/prediction_markets/` clone the `libs/market_data` shape exactly (no
   default provider, no cross-provider fallback, `ProviderNotConfigured` vs
   `ValueError`, stub opt-in only). Pure judgement lives in
   `libs/trading_core/events/{web_research,prediction_intel}.py`; I/O
   orchestration in `apps/gateway/event_{research,prediction_markets}.py`,
   pairing a write seam with a read seam exactly as `event_news.py` does.

2. **No agent.** No ReAct loop, no tool-calling, no browser. The orchestrator
   is a linear deterministic function — window, plan, search, evaluate,
   persist — and the model is called once over evidence the platform already
   admitted. The LLM may not control date boundaries, provider auth, network
   destinations, rate limits, source-quality rules, persistence, numeric truth,
   or execution.

3. **The read/write split is the cost boundary.** Every GET reads stored rows
   and is poll-safe; the two POST backfills are the only paths that spend.
   Bounds are named constants (`MAX_QUERIES_PER_EVENT`, `MAX_RESULTS_PER_QUERY`,
   `MAX_UNIQUE_DOCUMENTS`, `MAX_ACCEPTED_EVIDENCE`, `MAX_MARKETS_PER_QUERY`,
   `MAX_ACCEPTED_MARKETS`), plus a per-event throttle. What a press bought is
   recorded in the run row and the audit row — cost is auditable, not estimated.

4. **Bundle version bumped to `f1-evidence-v2`**, analysis contract to
   `event-analysis-v2`. The contract materially changed, so old and new
   analyses are not comparable answers and the cache key must separate them.
   The long-standing `options_analysis` gap is fixed in the same bump.

5. **Prediction-market pricing is pricing.** The field is
   `market_implied_probability`, every surface says "market-implied", and the
   DIRECT/DERIVED/CONTEXT relation travels beside every price — a DERIVED
   contract at 63c is not a 63% forecast of the catalyst. Depth facts (spread,
   liquidity, volume) are exposed unfolded; there is no composite score
   anywhere.

6. **Research-only, enforced twice.** No import path exists between the
   research stack and instrument selection, risk, signals, strategies, the
   broker, or the order/trading-pool routers (`test_research_safety_adversarial.py`,
   AST, both directions). And the whole chain run under injection writes zero
   rows to the watchlist, trading pool, orders and positions tables
   (`test_research_e2e_adversarial.py`, row diff).

**Consequences.** Adding Kalshi is new rows, not new columns
(`UNIQUE(provider, provider_market_id)`); adding a Reddit/social layer is a new
`SOCIAL` tier that already exists in the vocabulary. `BRAVE_API_KEY` joins the
write-only secret set. Search text is untrusted input on the news discipline —
sanitized, injection-flagged, withheld from model-facing text but counted, and
fenced in `<untrusted_web_research>`. A fix to `sanitize_for_llm` (re-strip
tags after entity decoding) hardens every existing fence, not just the new one.
See `docs/search-architecture.md`, `docs/prediction-market-architecture.md`,
`docs/event-research-orchestration.md`.
