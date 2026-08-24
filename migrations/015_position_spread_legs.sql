-- Defined-risk spread positions (execution-chains roadmap Phase 1):
-- instrument BULL_CALL_SPREAD rows identify the LONG leg via the existing
-- opt_* columns and the SHORT leg via these two. Honest NULLs elsewhere.
ALTER TABLE positions ADD COLUMN IF NOT EXISTS short_occ_symbol VARCHAR(24);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS short_strike DOUBLE PRECISION;
