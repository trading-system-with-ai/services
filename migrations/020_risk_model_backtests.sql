-- 020_risk_model_backtests.sql — Risk Engine Upgrade Phase E: persisted
-- VaR/ES model-validation (walk-forward backtest) history. Spec
-- risk_engine.md §42 (backtest the risk models), §43 (walk-forward only),
-- §57 (model validation is a SEPARATE concern from model output), §56 (never
-- store only the latest value), §63 (required model comparison), §68 (model
-- validation acceptance); Phase B/E design contract §9.4.
--
-- Mirrors apps/gateway/db.py::RiskModelBacktestRow EXACTLY (mirror rule, README).
--
-- Design decisions (design §9.4):
-- * ONE row per (validation run, model view). A run covers the whole view
--   grid — historical VaR 95/99, Gaussian VaR 95/99, EWMA-filtered VaR 95 and
--   the RESEARCH GARCH-filtered VaR 95 — so the table IS the calibration
--   history: "was our 99% VaR actually breaching 1% of days back in March"
--   stays answerable after the book that produced it is gone.
-- * WALK-FORWARD ONLY (§43). Every forecast counted in a row was produced on
--   a rolling window of observations STRICTLY BEFORE the day it forecasts.
--   `window_obs` records how many, `n_forecasts` how many days were scored.
--   Nothing in this table is a full-period (hindsight) statistic.
--   NAMING: the design doc calls this column `window`; `window` is a RESERVED
--   word in PostgreSQL (SQL:2003 window functions) and cannot be an unquoted
--   column name, so it is `window_obs` here and in the ORM. The unit is in the
--   name, which is the better name anyway.
-- * `verdict` is the Basel-style traffic light on the KUPIEC p-value
--   (GREEN / YELLOW / RED), and `UNAVAILABLE` when there were fewer than the
--   run's `min_forecasts` usable pairs. An UNAVAILABLE row is PERSISTED with
--   its `reason` and NULL statistics rather than skipped: "we could not yet
--   validate this model" is a fact worth keeping, and a missing row would
--   later read as "never run" (§44 rule 18 — honest nulls, never a
--   fabricated 0).
-- * Every statistic column is NULLABLE for exactly that reason. `rate` is a
--   FRACTION (0.012 = 1.2 % of days breached), as is `expected_rate`
--   (= 1 - confidence). `es_severity_ratio` is the mean realized loss on
--   exceedance days divided by the mean forecast ES on the same days —
--   > 1 means the tail was worse than ES said.
-- * `params` is the run's reproducible parameter set (window, min_forecasts,
--   the traffic-light cut-offs, the filter's lambda / GARCH refit stride, the
--   estimator's own meta) plus the EWMA-vs-GARCH comparison for the row that
--   carries it — diagnostic-shaped, hence JSONB; every scalar the UI reads is
--   a typed column.
-- * `snapshot_id` is the SCHEDULED build that triggered the run, or NULL for
--   an on-demand `POST /api/risk/validation/run` — a validation run is a READ
--   of the book's P&L history, not a snapshot build, and it writes NO audit
--   event (read views write no audit events) while still keeping its history.
--
-- Everything here is SHADOW/RESEARCH: nothing in this table alters a Tier 0
-- decision. The `backtest_red_triggers` model-risk parameter READS these
-- verdicts, and model risk is itself a SHADOW display.

CREATE TABLE IF NOT EXISTS risk_model_backtests (
    id                   SERIAL PRIMARY KEY,
    as_of                TIMESTAMPTZ NOT NULL,
    snapshot_id          INTEGER REFERENCES risk_snapshots (id) ON DELETE CASCADE,
    model_name           VARCHAR(64) NOT NULL,          -- historical_var | gaussian_var | conditional_var | garch_var
    model_version        VARCHAR(16) NOT NULL,
    distribution         VARCHAR(32) NOT NULL,          -- EMPIRICAL | NORMAL | EMPIRICAL_VOL_SCALED | EMPIRICAL_GARCH_SCALED
    confidence           DOUBLE PRECISION NOT NULL,     -- 0.95 | 0.99
    horizon_days         INTEGER NOT NULL,
    window_obs           INTEGER NOT NULL,              -- rolling estimation window (observations)
    n_forecasts          INTEGER NOT NULL,              -- usable forecast/realized pairs scored
    exceedances          INTEGER NOT NULL,              -- days with realized loss STRICTLY > forecast VaR
    rate                 DOUBLE PRECISION,              -- FRACTION exceedances/n_forecasts; NULL when n=0
    expected_rate        DOUBLE PRECISION NOT NULL,     -- FRACTION 1 - confidence
    kupiec_lr            DOUBLE PRECISION,
    kupiec_p             DOUBLE PRECISION,
    christoffersen_lr    DOUBLE PRECISION,
    christoffersen_p     DOUBLE PRECISION,
    es_severity_ratio    DOUBLE PRECISION,              -- mean realized loss / mean forecast ES on exceedance days
    verdict              VARCHAR(16) NOT NULL,          -- GREEN | YELLOW | RED | UNAVAILABLE
    health               VARCHAR(16) NOT NULL,
    reason               TEXT,
    params               JSONB NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_risk_model_backtests_snapshot ON risk_model_backtests (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_risk_model_backtests_as_of    ON risk_model_backtests (as_of);
CREATE INDEX IF NOT EXISTS ix_risk_model_backtests_model    ON risk_model_backtests (model_name, confidence, as_of);
