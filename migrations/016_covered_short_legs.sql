-- Phase 2 (execution-chains roadmap): collateralized short-premium positions.
-- COVERED_CALL rows: opt_* identify the SHORT call; collateral_position_id
-- links the LONG_STOCK row whose shares back it (100/contract).
-- CASH_SECURED_PUT rows: opt_* identify the SHORT put; cash_reserved is the
-- strike*100*qty locked until the position closes. Honest NULLs elsewhere.
ALTER TABLE positions ADD COLUMN IF NOT EXISTS collateral_position_id INTEGER;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS cash_reserved DOUBLE PRECISION;
