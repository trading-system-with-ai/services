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
