-- 005_orders.sql — paper order execution + position exit state (plan §11, §42, §5).
--
-- Orders record every simulated fill. side is BUY_TO_OPEN | SELL_TO_CLOSE
-- ONLY — Sell-to-Open does not exist anywhere in this system, for options or
-- stock (plan §5); the CHECK constraint enforces it at the database level.
-- client_order_id is the caller's idempotency key (§42): UNIQUE when present,
-- so a replayed request can never produce a second fill.
--
-- Positions gain the exit-engine state fixed at open (plan §11):
--   stop_distance  — per-share dollar risk (2 * ATR14 via the §10 chain)
--   entry_edge     — directional edge at entry (decay context, §38)
--   entry_bar_date — last stored bar date at entry (bars_held anchor; bar 0)
--   realized_pnl   — accumulated over partial closes; final once CLOSED

CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    client_order_id VARCHAR(64) UNIQUE,             -- idempotency key (§42)
    ticker          VARCHAR(16) NOT NULL,
    side            VARCHAR(16) NOT NULL
                    CHECK (side IN ('BUY_TO_OPEN', 'SELL_TO_CLOSE')),  -- §5
    quantity        INTEGER NOT NULL,
    fill_price      DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION NOT NULL,
    status          VARCHAR(16) NOT NULL DEFAULT 'FILLED',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_orders_ticker ON orders (ticker);

ALTER TABLE positions ADD COLUMN IF NOT EXISTS entry_edge     DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS stop_distance  DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS entry_bar_date VARCHAR(10);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS realized_pnl   DOUBLE PRECISION;
