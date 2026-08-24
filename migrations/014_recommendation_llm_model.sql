-- 014_recommendation_llm_model.sql — UPGRADE Phase E (§38/§41).
--
-- Mirrors apps/gateway/db.py::Recommendation.llm_model EXACTLY.
--
-- Records WHICH provider/model generated each LLM interpretation, at
-- generation time. Pre-upgrade rows stay '' — an honest unknown; they are
-- never backfilled from current settings, which may have changed since.

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS llm_model VARCHAR(128) NOT NULL DEFAULT '';
