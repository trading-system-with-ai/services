-- 013_trade_plans.sql — UPGRADE Phase D: research trade plan lifecycle
-- (upgrade 2026-08-12 §39/§40/§41).
--
-- Mirrors apps/gateway/db.py::TradePlanRow EXACTLY (mirror rule, README).
--
-- One row per generated research plan. `preview` is the complete §16
-- research-chain output the user reviewed; `versions` the §41 configuration
-- identifiers active at generation. `status` follows the §40 lifecycle
-- (GENERATED/REVIEWED/APPLIED/ACTIVE/SUPERSEDED/CANCELLED/...); a
-- superseded plan points at its successor. Applying a plan NEVER places an
-- order (§19).

CREATE TABLE IF NOT EXISTS trade_plans (
    id                 SERIAL PRIMARY KEY,
    ticker             VARCHAR(16)  NOT NULL,
    status             VARCHAR(16)  NOT NULL,
    direction          VARCHAR(8)   NOT NULL DEFAULT 'AUTO',
    quantity_requested INTEGER,
    preview            JSONB        NOT NULL,
    versions           JSONB        NOT NULL DEFAULT '{}',
    market_data_as_of  VARCHAR(10),
    generated_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    applied_at         TIMESTAMPTZ,
    superseded_by      INTEGER,
    created_by         VARCHAR(64)  NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_plans_ticker ON trade_plans (ticker);
CREATE INDEX IF NOT EXISTS idx_trade_plans_status ON trade_plans (status);
