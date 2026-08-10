-- 006_recommendations_evidence.sql — recommendation evidence + company (plan §4.1).
--
-- The recommendations table (001) gains the two columns the recommendation
-- API serves: company (display name, nullable) and evidence — the list of
-- citation objects {"source", "published_at", "snippet"} backing the scores.
-- Plan §20.3 (news timestamp integrity): every evidence item's published_at
-- is strictly before the generation as-of time.
--
-- SAFETY (plan §4.1, §44 rule 5, §46): recommendations remain information
-- rows with zero execution authority — as in 001, there is intentionally no
-- trigger or job that moves them into watchlist/trading_pool/orders; only an
-- explicit USER API action (promote) can do that.

ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS company  VARCHAR(128);
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '[]';

-- The list endpoint filters by status and orders by recency.
CREATE INDEX IF NOT EXISTS ix_recommendations_status_ts ON recommendations (status, ts);
