-- 011_runtime_config.sql — UI-managed provider configuration (runtime layer).
--
-- Mirrors apps/gateway/db.py::RuntimeConfig EXACTLY (mirror rule, README).
--
-- Provider selection and credentials (Massive / LLM / Alpaca paper) are now
-- set from the Settings UI and stored HERE, overriding any .env value: rows
-- are loaded into the process environment at startup and on every change,
-- then the cached Settings object is rebuilt. .env remains a fallback for
-- infra values (DATABASE_URL etc.) and for headless deployments.
--
-- SECRETS: values in this table are credentials. They are NEVER returned by
-- any API response (GET /api/config/providers reports presence booleans
-- only), never logged, and never audited by value — the CONFIG_CHANGED
-- audit event records which KEYS changed, nothing else.

CREATE TABLE IF NOT EXISTS runtime_config (
    key        VARCHAR(64) PRIMARY KEY,
    value      TEXT        NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
