# Development Log — Backend

Newest entries first. Each loop iteration appends one entry: what was built, key
decisions, test/audit status, and what's next.

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
