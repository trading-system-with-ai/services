-- 017_orders_side_vocabulary.sql — orders.side CHECK catches up with the
-- execution-chains program (roadmap Phases 1–3; risk-engine audit §8 item 1).
--
-- 005 pinned side to BUY_TO_OPEN | SELL_TO_CLOSE when "Sell-to-Open does not
-- exist anywhere in this system". Since 2026-08-17 the platform legitimately
-- records SELL_TO_OPEN (covered call / cash-secured put short legs, margin
-- short stock, the short leg of a defined-risk spread) and BUY_TO_CLOSE
-- (buybacks, covers). The code wrote those sides while the live constraint
-- still forbade them — a latent INSERT failure that the sqlite test harness
-- (ORM create_all, no CHECK) could not see. The vocabulary below is EXACTLY
-- libs.broker.provider.MLEG_LEG_SIDES; tests/test_migration_parity.py pins
-- the two lists together so they can never drift again.
--
-- Naked short options remain unconstructable at the ADAPTER level (an OCC
-- symbol on a single-leg short raises) — this constraint is not the guard for
-- that and never was; the side vocabulary is a ledger fact, not a permission.
ALTER TABLE orders DROP CONSTRAINT IF EXISTS orders_side_check;
ALTER TABLE orders ADD CONSTRAINT orders_side_check
    CHECK (side IN ('BUY_TO_OPEN', 'SELL_TO_OPEN', 'SELL_TO_CLOSE', 'BUY_TO_CLOSE'));
