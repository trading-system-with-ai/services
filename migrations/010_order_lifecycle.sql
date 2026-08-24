-- 010_order_lifecycle.sql — PENDING_SUBMIT lifecycle + late-fill risk context
-- on orders (guide §11 Iteration C).
--
-- Mirrors the ORM model apps/gateway/db.py::Order EXACTLY. If you change that
-- model, change this file in the same commit (the mirror rule, README).
--
-- The order row is now written BEFORE the broker submit (status
-- 'PENDING_SUBMIT') and updated after — so a crash mid-submit leaves a row
-- the order-sync sweep (apps/gateway/order_sync.py) can resolve against the
-- broker by client_order_id, instead of an invisible broker order.
--
--   position_id    — the local position this order opened (BUY, set when the
--                    first fill opens it) or closes (SELL, set at creation).
--                    Plain integer, no FK: an order outlives its position row
--                    semantics and reconciliation must never cascade.
--   stop_distance  — the risk-approved per-share stop distance captured at
--                    approval time, so a fill that arrives AFTER the approve
--                    request returned (via the sweep) can still open the
--                    position with the §10 chain's own risk parameters
--                    instead of a guessed zero.
--   entry_edge     — directional edge at approval (same late-fill purpose).
--   entry_bar_date — last stored bar date at approval (bars_held anchor).
--
-- All four are honest NULLs for rows that predate this migration and for
-- simulated fills (which settle synchronously and never need late-fill
-- reconstruction).

ALTER TABLE orders ADD COLUMN IF NOT EXISTS position_id INTEGER;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS stop_distance DOUBLE PRECISION;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS entry_edge DOUBLE PRECISION;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS entry_bar_date VARCHAR(10);

CREATE INDEX IF NOT EXISTS idx_orders_position_id ON orders (position_id);
-- The order-sync sweep's working set: non-terminal statuses only.
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
