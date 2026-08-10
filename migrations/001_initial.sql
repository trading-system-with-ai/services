-- 001_initial.sql — Phase 0 schema (PostgreSQL + TimescaleDB)
-- Relational core tables. Time-series hypertables arrive with Phase 1 data ingestion.

CREATE TABLE IF NOT EXISTS watchlist (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(16) NOT NULL UNIQUE,
    added_by    VARCHAR(64) NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trading_pool (
    id                 SERIAL PRIMARY KEY,
    ticker             VARCHAR(16) NOT NULL UNIQUE,
    trading_enabled    BOOLEAN NOT NULL DEFAULT FALSE,
    allowed_strategies JSONB NOT NULL DEFAULT '[]',
    promoted_by        VARCHAR(64) NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- a symbol may only be in the trading pool while it is on the watchlist
    CONSTRAINT fk_trading_pool_watchlist FOREIGN KEY (ticker)
        REFERENCES watchlist (ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_events (
    id             SERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_type     VARCHAR(16) NOT NULL,   -- USER | SYSTEM | LLM
    actor_id       VARCHAR(64) NOT NULL DEFAULT '',
    action         VARCHAR(64) NOT NULL,
    entity_type    VARCHAR(32) NOT NULL,
    entity_id      VARCHAR(64) NOT NULL,
    details        JSONB NOT NULL DEFAULT '{}',
    correlation_id VARCHAR(64) NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_events (ts);
CREATE INDEX IF NOT EXISTS ix_audit_action ON audit_events (action);
CREATE INDEX IF NOT EXISTS ix_audit_entity ON audit_events (entity_type, entity_id);

CREATE TABLE IF NOT EXISTS recommendations (
    id             SERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    ticker         VARCHAR(16) NOT NULL,
    sentiment      DOUBLE PRECISION,
    impact         DOUBLE PRECISION,
    novelty        DOUBLE PRECISION,
    source_reliability DOUBLE PRECISION,
    horizon        VARCHAR(16),
    catalyst_type  VARCHAR(64),
    reason_codes   JSONB NOT NULL DEFAULT '[]',
    summary        TEXT NOT NULL DEFAULT '',
    status         VARCHAR(16) NOT NULL DEFAULT 'PENDING'  -- PENDING | DISMISSED | PROMOTED
);
-- NOTE: there is intentionally no trigger or job that moves recommendations
-- into watchlist. Only an explicit USER API action can do that (plan rule 5).
