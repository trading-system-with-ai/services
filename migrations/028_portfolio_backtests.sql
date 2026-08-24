-- 028_portfolio_backtests.sql — persisted PORTFOLIO backtest runs
-- (auto-strategy program Phase C, docs/auto-strategy-portfolio-design.md,
-- user mandate 2026-08-20: replay the whole watchlist against one shared
-- cash ledger and store the per-day allocation table).
--
-- Mirrors apps/gateway/db.py::PortfolioBacktestRecord EXACTLY, column order
-- included (mirror rule, README) — tests/test_migration_parity.py pins the
-- sequence against the ORM.
--
-- Same storage philosophy as 003_backtests.sql: the engine's full output is
-- stored as JSONB in the exact API response shape, reads never recompute.
-- `tickers` is the run's symbol set (JSONB array); `allocations` holds the
-- per-bar signed allocation percents plus cash percent (the user's ask
-- verbatim); `decisions` is the ticker-attributed §8 audit trail.

CREATE TABLE IF NOT EXISTS portfolio_backtests (
    id             SERIAL PRIMARY KEY,
    tickers        JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    status         VARCHAR(16) NOT NULL,          -- COMPLETED | FAILED
    params         JSONB NOT NULL,
    metrics        JSONB NOT NULL,
    trades         JSONB NOT NULL,
    equity_curve   JSONB NOT NULL,
    allocations    JSONB NOT NULL,
    decisions      JSONB NOT NULL,
    error          TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ix_portfolio_backtests_created
    ON portfolio_backtests (created_at DESC);
