-- 018_risk_snapshots.sql — Risk Engine Upgrade Phase B: persisted statistical
-- risk (spec risk_engine.md §44 model versioning, §45 snapshot, §55 as_of,
-- §56 "do not store only the latest value"; audit §7.1 "Persistence").
--
-- Mirrors apps/gateway/db.py::RiskSnapshotRow / RiskMetricRow /
-- RiskContributionRow / AtmIvDailyRow EXACTLY (mirror rule, README).
--
-- Design decisions (audit §10 Phase B):
-- * risk_snapshots — one row per snapshot BUILD (scheduled daily after the
--   bar refresh, on demand, or pre-trade). The typed scalars of the
--   §45 PortfolioRiskSnapshot live in columns; only diagnostics-shaped data
--   (data-quality reasons, per-model health map) is JSONB. The daily
--   SCHEDULED rows are the NAV time series live drawdown is measured on —
--   real drawdown accrues from the first row, nothing is back-filled.
-- * risk_metrics — one row per (metric, model, confidence, horizon) per
--   snapshot, carrying the FULL §44 model identity inline (model_name,
--   model_version, params, distribution, data window via the parent
--   snapshot, sample_size, diagnostics). The audit's separate
--   `risk_model_runs` entity is folded in here: at this platform's scale a
--   metric row IS the model run, and one table keeps every number
--   reproducible without a join.
-- * risk_contributions — per-position risk contribution rows (VOL / ES
--   Euler) so "capital weight vs risk weight" (§10, §49) has history.
-- * atm_iv_daily — the ATM implied vol the chain read computes today and
--   discards, persisted per underlying per day so IV rank / empirical IV
--   shocks (§24) become possible over time. Internally calculated from the
--   provider chain — labelled by `source`, never presented as vendor IV
--   history (data-source-architecture provenance rule).
--
-- Everything is SHADOW/RESEARCH: nothing here alters a Tier 0 decision.
-- Plain tables (daily granularity — same argument as 007), sqlite-mirrored
-- by the ORM for the test harness; JSONB only where the shape is diagnostic.

CREATE TABLE IF NOT EXISTS risk_snapshots (
    id                       SERIAL PRIMARY KEY,
    as_of                    TIMESTAMPTZ NOT NULL,
    snapshot_version         VARCHAR(16) NOT NULL,
    trigger                  VARCHAR(16) NOT NULL,          -- SCHEDULED | ON_DEMAND | PRE_TRADE
    nav                      DOUBLE PRECISION,              -- honest NULLs when no account
    cash                     DOUBLE PRECISION,
    cash_reserved            DOUBLE PRECISION,
    gross_exposure           DOUBLE PRECISION,
    delta_adjusted_exposure  DOUBLE PRECISION,
    heat_pct                 DOUBLE PRECISION,
    heat_state               VARCHAR(16),
    n_positions              INTEGER NOT NULL DEFAULT 0,
    n_obs                    INTEGER,                       -- aligned return observations
    window_start             DATE,
    window_end               DATE,
    pnl_method               VARCHAR(24),                   -- DELTA_LINEAR (Phase B) | FULL_REVAL (Phase D)
    data_quality_valid       BOOLEAN NOT NULL DEFAULT FALSE,
    data_quality             JSONB NOT NULL DEFAULT '{}',   -- {reasons, tickers_missing, keys_excluded, ...}
    model_health             JSONB NOT NULL DEFAULT '{}',   -- {model_name: ACTIVE|DEGRADED|UNAVAILABLE|FAILED}
    model_risk_state         VARCHAR(16),                   -- LOW | ELEVATED | HIGH
    dispersion_ratio         DOUBLE PRECISION,
    dispersion_high          BOOLEAN,
    distribution_primary     VARCHAR(16),                   -- NORMAL_LIKE | HEAVY_TAIL | LEFT_SKEWED | UNSTABLE
    gaussian_trust           VARCHAR(8),                    -- HIGH | REDUCED | LOW
    drawdown_current_pct     DOUBLE PRECISION,
    drawdown_max_pct         DOUBLE PRECISION,
    risk_state               VARCHAR(16),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_risk_snapshots_as_of   ON risk_snapshots (as_of);
CREATE INDEX IF NOT EXISTS ix_risk_snapshots_trigger ON risk_snapshots (trigger, as_of);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id             SERIAL PRIMARY KEY,
    snapshot_id    INTEGER NOT NULL REFERENCES risk_snapshots (id) ON DELETE CASCADE,
    metric         VARCHAR(32) NOT NULL,                    -- VAR | ES | VOLATILITY | COND_VAR | COND_ES | ...
    model_name     VARCHAR(48) NOT NULL,                    -- historical_var, gaussian_es, ...
    model_version  VARCHAR(16) NOT NULL,
    confidence     DOUBLE PRECISION,
    horizon_days   INTEGER,
    distribution   VARCHAR(32),
    value          DOUBLE PRECISION,                        -- USD loss (positive = money lost); NULL when unavailable
    value_pct_nav  DOUBLE PRECISION,
    health         VARCHAR(16) NOT NULL,
    reason         TEXT,
    sample_size    INTEGER NOT NULL DEFAULT 0,
    params         JSONB NOT NULL DEFAULT '{}',
    diagnostics    JSONB NOT NULL DEFAULT '{}',
    as_of          TIMESTAMPTZ NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_risk_metrics_snapshot ON risk_metrics (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_risk_metrics_model    ON risk_metrics (metric, model_name, as_of);

CREATE TABLE IF NOT EXISTS risk_contributions (
    id             SERIAL PRIMARY KEY,
    snapshot_id    INTEGER NOT NULL REFERENCES risk_snapshots (id) ON DELETE CASCADE,
    method         VARCHAR(8)  NOT NULL,                    -- VOL | ES
    confidence     DOUBLE PRECISION,                        -- ES only
    position_key   VARCHAR(64) NOT NULL,
    ticker         VARCHAR(16) NOT NULL,
    instrument     VARCHAR(24) NOT NULL,
    contribution   DOUBLE PRECISION NOT NULL,               -- USD, same units as the total
    share          DOUBLE PRECISION,                        -- contribution / total; NULL if total <= 0
    capital_weight DOUBLE PRECISION,                        -- |market value| / NAV
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_risk_contributions_snapshot ON risk_contributions (snapshot_id);

CREATE TABLE IF NOT EXISTS atm_iv_daily (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(16) NOT NULL,
    bar_date    DATE NOT NULL,
    atm_iv      DOUBLE PRECISION NOT NULL,                  -- annualized fraction, e.g. 0.28
    spot        DOUBLE PRECISION NOT NULL,
    expiry      DATE,
    dte         INTEGER,
    source      VARCHAR(24) NOT NULL,                       -- e.g. alpaca_chain (INTERNAL DETERMINISTIC read of provider IV)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_atm_iv_daily_ticker_date UNIQUE (ticker, bar_date)
);

CREATE INDEX IF NOT EXISTS ix_atm_iv_daily_ticker ON atm_iv_daily (ticker);
