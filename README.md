# Trading Platform — Backend Services

Backend for the systematic options trading platform. See
`../prompts/systematic_options_trading_platform_development_plan.md` for the full
product/engineering specification.

## Core principle

```text
LLM proposes. User curates. Quant measures. Strategy selects.
Portfolio allocates. Risk decides. Execution obeys. Exit protects. Audit explains.
```

- Only **user actions** can add symbols to the Watchlist or promote them to the Trading Pool.
- Only **Trading Pool** symbols can ever reach execution, and every trade must pass the
  independent Risk Engine.
- Every state change writes an **audit event** in the same transaction.

## Layout

```text
libs/
  common/         config, structured logging, telemetry (shared plumbing)
  trading_core/   domain models + the full quant core: features, signals,
                  backtest, risk, exits, options, contracts, strategies,
                  volatility, greeks, correlation, allocation, health
                  (shared verbatim between backtest and live — mandatory rule)
apps/
  gateway/        FastAPI modular monolith: watchlist, trading-pool, audit modules.
                  Modules keep service-shaped boundaries so they can be split into
                  standalone containers later without API changes.
migrations/       raw SQL for PostgreSQL + TimescaleDB
tests/            pytest suite (authorization rules are the most important tests)
```

## Quick start

```bash
make install   # python venv + deps
make test      # run test suite
make dev       # uvicorn on :8000 (sqlite dev db)
make up        # full stack: timescaledb + redis + gateway via docker compose
```

Copy `.env.example` to `.env` for configuration. Never commit `.env`.

## Verification

- `make test` — the full pytest suite (726 tests), sqlite-backed, no services needed.
- `make verify` — tests + `docker compose config` validation + a YAML parse of
  `.github/workflows/ci.yml`, i.e. everything CI checks that can run locally.
- Docker acceptance: `make up`, then hit `http://localhost:8000/health` and the UI on
  `http://localhost:3000`; `make down` when done.

What was built and why lives in [docs/DEVLOG.md](docs/DEVLOG.md) (per-iteration log)
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (ADRs).

## Migrations

`migrations/*.sql` own the production schema and run automatically the first
time the `db` container initializes its volume (to re-run them from scratch:
`docker compose down -v`). Two rules:

- **Compose mounts each migration file individually** into
  `/docker-entrypoint-initdb.d/` — never the whole directory, which would
  shadow the timescale image's own init scripts (`CREATE EXTENSION
  timescaledb` + tuning) and break `create_hypertable()` in 002. When adding
  `migrations/00X_*.sql`, add the matching volume line in `docker-compose.yml`.
- `007_stock_bars_daily.sql` mirrors the ORM model
  `apps/gateway/db.py::StockBarDaily` exactly, and `009_broker.sql` mirrors the
  broker columns on `apps/gateway/db.py::Order` (the gateway's `init_db()`
  `create_all` is a dev convenience only). If a model changes, change the
  migration in the same commit.

### Frontend container

The `frontend` compose service builds the Next.js UI from `../ui`
(`docker compose build frontend`) and serves it on
[http://localhost:3000](http://localhost:3000), depending on `gateway`. It
bakes `NEXT_PUBLIC_API_BASE=http://localhost:8000` into the page — the
browser, not the container, calls the API, so the base URL points at the
host-published gateway port. No migrations run in this container; it is
stateless.

## Development log

See [docs/DEVLOG.md](docs/DEVLOG.md) — every iteration records what was built, decisions
made, and what's next. Architecture decisions live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
