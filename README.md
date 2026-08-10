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
  common/         config, structured logging (shared plumbing)
  trading_core/   domain models, features, signals, strategies, risk, exits
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

## Development log

See [docs/DEVLOG.md](docs/DEVLOG.md) — every iteration records what was built, decisions
made, and what's next. Architecture decisions live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
