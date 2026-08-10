-- 008_option_execution.sql — option paper execution columns (plan §8, §9,
-- §11.3, §12.1).
--
-- Mirrors the ORM models apps/gateway/db.py::Order / Position EXACTLY. If you
-- change those models, change this file in the same commit.
--
-- Orders gain the instrument + contract identity of an option fill:
--   instrument  — LONG_STOCK (default) | LONG_CALL | LONG_PUT
--   opt_expiry  — contract expiry, YYYY-MM-DD (null for stock — honest nulls)
--   opt_strike  — contract strike (null for stock)
--   opt_right   — 'C' | 'P' (null for stock)
-- For option orders quantity counts CONTRACTS and fill_price is the premium
-- PER SHARE; the x100 multiplier applies to cash.
--
-- Positions gain the same contract identity plus the multiplier:
--   multiplier  — 1 for stock (default), 100 for options. For option rows
--                 avg_price is the entry premium per share, max_loss the full
--                 premium paid (quantity * avg_price * multiplier, §12.1) and
--                 stop_distance stores the per-share entry premium — the
--                 §11.3 PREMIUM hard-stop basis, NOT an underlying stop.

ALTER TABLE orders ADD COLUMN IF NOT EXISTS instrument VARCHAR(16) NOT NULL DEFAULT 'LONG_STOCK';
ALTER TABLE orders ADD COLUMN IF NOT EXISTS opt_expiry VARCHAR(10);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS opt_strike DOUBLE PRECISION;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS opt_right  VARCHAR(1)
    CHECK (opt_right IN ('C', 'P') OR opt_right IS NULL);

ALTER TABLE positions ADD COLUMN IF NOT EXISTS opt_expiry VARCHAR(10);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS opt_strike DOUBLE PRECISION;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS opt_right  VARCHAR(1)
    CHECK (opt_right IN ('C', 'P') OR opt_right IS NULL);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS multiplier INTEGER NOT NULL DEFAULT 1;
