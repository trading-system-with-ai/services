-- 009_broker.sql — real broker execution columns on orders (plan §11, §44
-- rule 18).
--
-- Mirrors the ORM model apps/gateway/db.py::Order EXACTLY. If you change that
-- model, change this file in the same commit (the mirror rule, README).
--
-- The platform can now route a fill through a REAL broker (Alpaca PAPER only —
-- live trading is deliberately unreachable, see libs/broker/alpaca.py) instead
-- of the internal simulator. What comes back from a broker is not a guaranteed
-- fill, so these three columns record what ACTUALLY happened rather than what
-- was requested:
--
--   broker_order_id — the broker's own order id. UNIQUE when set, so one local
--                     row can never be matched to two broker orders (and vice
--                     versa) during reconciliation. NULL for simulated fills.
--   broker_status   — the broker's RAW status string, preserved verbatim
--                     ('accepted', 'partially_filled', 'filled', 'rejected',
--                     ...). Kept alongside the normalised lifecycle so an
--                     unrecognised broker state stays visible instead of being
--                     flattened away. NULL for simulated fills.
--   filled_quantity — how much actually filled. DISTINCT FROM quantity, which
--                     is what we asked for: a PARTIALLY_FILLED order has
--                     0 < filled_quantity < quantity and opens a position for
--                     the FILLED amount only; a zero-fill ACCEPTED order has
--                     filled_quantity = 0 and opens NO position. Defaults to 0
--                     so pre-existing rows are not retroactively claimed to
--                     have filled at the broker.

ALTER TABLE orders ADD COLUMN IF NOT EXISTS broker_order_id VARCHAR(64);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS broker_status   VARCHAR(24);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS filled_quantity INTEGER NOT NULL DEFAULT 0;

-- UNIQUE only over NON-NULL values: every simulated fill leaves this NULL, and
-- Postgres UNIQUE treats NULLs as distinct, so many null rows coexist happily
-- while two rows can never claim the same broker order.
CREATE UNIQUE INDEX IF NOT EXISTS uq_orders_broker_order_id
    ON orders (broker_order_id)
    WHERE broker_order_id IS NOT NULL;
