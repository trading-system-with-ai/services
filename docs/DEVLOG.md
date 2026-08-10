# Development Log — Backend

Newest entries first. Each loop iteration appends one entry: what was built, key
decisions, test/audit status, and what's next.

---

## 2026-08-10 — Iteration 2: Feature engine slice, market-data stub, global kill switch

**Built (via 3 parallel implementation agents + 2 adversarial verify agents):**
- `libs/trading_core/features/indicators.py` — pure, dependency-free, deterministic:
  SMA, EMA (SMA-seeded), RSI (Wilder), True Range/ATR (Wilder), MACD (12/26/9,
  parameterized), realized vol (close-to-close log returns, annualization param),
  pivot highs/lows (§6.3; final `window` bars always unconfirmed — no look-ahead,
  §20.3). All outputs input-length with None warmup padding; all periods parameters.
  34 new tests including hand-computed reference values with arithmetic in comments
  and a backtest/live parity check for pivot stability.
- `libs/market_data/` — provider abstraction (Quote + MarketDataProvider Protocol,
  name-based registry) with deterministic StubProvider (SPY/QQQ/VIX, minute-keyed
  wiggle) until the Massive integration lands. `GET /api/market/overview` returns
  provider/as_of/stale/market_regime (NEUTRAL_RANGE placeholder)/indices.
- Global kill switch (§18): persistent `system_state` singleton row (default:
  trading disabled), `GET /api/trading/status`, `POST /api/trading/pause` (reason
  required), `POST /api/trading/resume`; both mutations USER-attributed and audited
  (TRADING_PAUSED/TRADING_RESUMED) in the same transaction.
- `migrations/002_system_state_and_bars.sql` — system_state seed + `stock_bars_1m`
  Timescale hypertable (Phase 1 groundwork).

**Verified:** full suite 51/51 green (was 11); independent verify agent booted the
app and confirmed overview/status/pause flows + audit records via curl; indicator
spot checks passed. Zero fixes needed.

**Next (iteration 3):**
1. Historical OHLCV ingestion into stock_bars (stub-generated series for dev) and
   `GET /api/watchlist/{ticker}/analysis` computing indicators over stored bars.
2. Market Regime Engine v0 (§6.1) using SPY/QQQ features — replace the
   NEUTRAL_RANGE placeholder in /api/market/overview.
3. Directional signal engine v0 (§6.2): parameterized bull/bear scores over features.
4. UI: symbol analysis page skeleton (tabs per §33) showing computed indicators.

---

## 2026-08-10 — Iteration 1: Phase 0 skeleton + Watchlist/Trading Pool core

**Built:**
- Project skeleton: `pyproject.toml`, `libs/common` (config via pydantic-settings,
  structured JSON logging with secret redaction), `libs/trading_core/models` (domain
  enums: regimes, instruments, risk decisions, audit actions, actor types).
- `apps/gateway` FastAPI modular monolith:
  - `/healthz`, `/readyz` (DB-checked)
  - Watchlist API: `GET/POST /api/watchlist`, `DELETE /api/watchlist/{ticker}`
  - Trading Pool API: `GET/POST /api/trading-pool`, toggle `POST /{ticker}/trading`,
    `DELETE /{ticker}`
  - Audit API: `GET /api/audit` (filter by entity, read-only)
- `migrations/001_initial.sql` — Postgres/Timescale DDL with FK cascade
  (trading_pool → watchlist) and audit indexes.
- `docker-compose.yml` — timescaledb + redis + gateway; migrations auto-applied on
  first db boot. Gateway Dockerfile.
- Test suite: 11 tests, all passing. The important ones encode plan rules:
  - non-Watchlist symbol cannot enter Trading Pool (rule 6);
  - promotion starts with trading disabled (authorization ≠ order);
  - short strategies rejected by account constraints (rules 7/8);
  - every mutation audited with USER attribution (rule 12);
  - Watchlist removal cascades out of Trading Pool, cascade itself audited.

**Verified:** `pytest` 11/11 green; live smoke test via uvicorn+curl confirmed the
422 rejection path, disabled-by-default promotion, and audit trail contents.

**Decisions:** see ARCHITECTURE.md ADR-001…004 (modular monolith, repo naming,
transactional audit, write-path authorization).

**Deliberately deferred:** auth-service (single fixed `local-user` identity for now),
event bus, Massive market data adapter, kill-switch API, recommendations endpoints.

**Next (iteration 2):**
1. Massive market-data adapter interface + stub provider (so UI can show prices
   without a real key), `GET /api/market/overview`.
2. Feature engine start: SMA/RSI/ATR/MACD in `libs/trading_core/features` with
   deterministic unit tests (Phase 2 groundwork).
3. Kill switch API: `POST /api/trading/pause` + `/resume` with audit + UI banner wiring.
4. Historical OHLCV storage schema (Timescale hypertable migration).
