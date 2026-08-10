-- 003_backtests.sql — persisted backtest runs (plan §20).
--
-- One row per run of Backtest Engine V1. The engine's full output (resolved
-- params, per-segment metrics, trades, equity curve) is stored as JSONB in
-- the exact API response shape, so reads never recompute anything. Rows are
-- inserted in the SAME transaction as their BACKTEST_* audit events (rule 12).

CREATE TABLE IF NOT EXISTS backtests (
    id             SERIAL PRIMARY KEY,
    ticker         VARCHAR(16) NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    status         VARCHAR(16) NOT NULL,          -- COMPLETED | FAILED
    params         JSONB NOT NULL DEFAULT '{}',   -- resolved BacktestParams
    metrics        JSONB NOT NULL DEFAULT '{}',   -- {"full","in_sample","out_of_sample"}
    trades         JSONB NOT NULL DEFAULT '[]',
    equity_curve   JSONB NOT NULL DEFAULT '{}',   -- {"dates","equity","drawdown"}
    oos_start_date VARCHAR(10),                   -- YYYY-MM-DD; NULL if unavailable
    error          TEXT NOT NULL DEFAULT ''       -- engine error message on FAILED
);

CREATE INDEX IF NOT EXISTS ix_backtests_ticker ON backtests (ticker);
