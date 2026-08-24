-- 021_events.sql — Catalyst & Event Intelligence Phase B: the event registry,
-- the market session calendar and the per-provider ingestion watermarks
-- (event spec §5 taxonomy, §6 event object, §7 event date sources, §8 macro
-- sources / "calendar ingestion should survive individual provider failures",
-- §10 market calendar, §11 alerting; audit catalyst-event-audit.md §5.2).
--
-- Mirrors apps/gateway/db.py::EventRow / MarketCalendarRow /
-- EventIngestStateRow EXACTLY (mirror rule, README).
--
-- Design decisions (audit §5.2, §6, §7):
-- * events — ONE row per real-world catalyst, identified by the deterministic
--   natural key `event_key` (EARNINGS:NVDA:2026-08-27, CPI:2026-07,
--   FOMC_DECISION:2026-09-16, MARKET_HOLIDAY:US:2026-11-26, ...). The key is
--   UNIQUE, not `(event_key, source)`: an event is ONE fact that several
--   sources describe with differing authority, so a better source UPDATES the
--   row (source-precedence merge, libs/trading_core/events/models.py) rather
--   than inserting a rival copy. This is also what makes ingestion idempotent
--   at the DATABASE level, which ADR-007 (no leader election, no distributed
--   lock) requires: a second replica re-ingesting the same tick can only
--   collide on the unique key, never double-insert.
-- * `event_status` vs `status`: the column is `status` here and in the ORM —
--   `status` is not reserved in PostgreSQL and every other table in this
--   schema (trade_plans, orders, backtests) already names its lifecycle
--   column `status`. It carries EventStatus (ESTIMATED | CONFIRMED | REVISED
--   | CANCELED). ESTIMATED is a DERIVED date (filing cadence) and is never
--   presented as a confirmed fact nor alerted on (§7, audit §13) — the value
--   travels into every downstream payload so the UI can label it.
-- * `source` is the EventSourceKind PRECEDENCE TIER (USER | COMPANY_IR_SEC |
--   GOVERNMENT_AGENCY | FEDERAL_RESERVE | STRUCTURED_PROVIDER | DERIVED |
--   NEWS | LLM) and `source_name` the concrete adapter that wrote it
--   ("sec_edgar", "alpaca_calendar", "fed_fomc", "derived_cadence", "user").
--   Priority is DATA, not code (§78): a CONFIRMED SEC row supersedes an
--   ESTIMATED cadence row by rule, not by whichever adapter ran last.
-- * ALL instants are TIMESTAMPTZ (UTC) AND `event_timezone` stores the
--   event's own zone string, because "08:30 ET" is the fact a macro release
--   asserts — the UTC instant of an ET wall-clock time changes across DST and
--   a stored offset alone cannot be re-derived. America/New_York for display.
-- * `previous_event_id` is a SELF-FK to the previous comparable event (§15)
--   with `comparison_reason` recording WHY that row was chosen ("prior
--   quarterly earnings", "prior release period") — an unexplained comparison
--   is not auditable. ON DELETE SET NULL: deleting a stale prior event must
--   not cascade-delete the upcoming one.
-- * `revision_history` is the append-only trail of date moves
--   ({scheduled_at, status, source_name, at}) — diagnostic-shaped, hence
--   JSONB; every scalar the UI reads is a typed column. Same rule as
--   risk_snapshots.data_quality / stress_runs.params.
-- * `importance` is NULLABLE (0-100 when scored): "not yet scored" is not 0
--   (§44 rule 18 — honest nulls, never a fabricated zero). The transparent
--   component breakdown lives with the scorer, not in this table.
-- * market_calendar — the real session grid from Alpaca /v2/calendar (200)
--   and Massive /v1/marketstatus/upcoming (200), keyed by `session_date`.
--   It fixes the hole admitted in routers/analysis.py::_last_expected_trading_date
--   ("Holidays are not modeled") and is what classifies an event as
--   BEFORE_MARKET / DURING_MARKET / AFTER_MARKET on a HALF DAY, where the
--   09:30-16:00 default would be wrong. `open_utc`/`close_utc` are the
--   regular session; `session_open_utc`/`session_close_utc` the extended
--   session (04:00 / 20:00 ET) and NULLABLE because not every provider
--   reports them — an absent extended session is NULL, never a guess.
-- * event_ingest_state — one row per provider (or per provider:ticker for
--   SEC), the WATERMARK that makes per-adapter failure isolation observable:
--   `last_ok_at` gates re-fetch (SEC/Fed daily, calendars weekly),
--   `last_error` keeps the last honest failure string (403 SUBSCRIPTION_DENIED,
--   timeout, HTML parse) so a dead adapter is visible instead of silently
--   contributing nothing, and `meta` carries the provider's last capability
--   report. A failing provider updates its row and leaves every other
--   provider's rows committed (§8).
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS events (
    id                 SERIAL PRIMARY KEY,
    event_key          VARCHAR(200) NOT NULL UNIQUE,   -- deterministic natural key, e.g. EARNINGS:NVDA:2026-08-27
    event_type         VARCHAR(32)  NOT NULL,          -- EventType
    title              VARCHAR(300) NOT NULL,
    ticker             VARCHAR(16),                    -- NULL for macro / Fed / market-wide events
    company_id         VARCHAR(32),                    -- SEC CIK when known
    scheduled_at       TIMESTAMPTZ  NOT NULL,          -- UTC instant
    event_timezone     VARCHAR(64)  NOT NULL DEFAULT 'America/New_York',
    session            VARCHAR(16)  NOT NULL DEFAULT 'UNKNOWN',  -- EventSession
    status             VARCHAR(16)  NOT NULL,          -- EventStatus: ESTIMATED | CONFIRMED | REVISED | CANCELED
    source             VARCHAR(32)  NOT NULL,          -- EventSourceKind (precedence tier)
    source_name        VARCHAR(64)  NOT NULL,          -- concrete adapter: sec_edgar | alpaca_calendar | fed_fomc | derived_cadence | user
    source_url         TEXT,
    source_event_id    VARCHAR(200),                   -- provider's own id (SEC accessionNumber, ...)
    last_verified_at   TIMESTAMPTZ,                    -- last time the writing source re-asserted this date
    previous_event_id  INTEGER REFERENCES events (id) ON DELETE SET NULL,
    comparison_reason  VARCHAR(200),                   -- WHY previous_event_id was chosen (§15)
    importance         INTEGER,                        -- 0-100; NULL = not scored yet (never a fabricated 0)
    series_id          VARCHAR(64),                    -- macro: BLS/BEA series
    agency             VARCHAR(64),                    -- macro: BLS | BEA | Census | Federal Reserve
    release_period     VARCHAR(32),                    -- macro: the period released, e.g. 2026-07
    fiscal_quarter     INTEGER,                        -- corporate
    fiscal_year        INTEGER,                        -- corporate
    speaker            VARCHAR(120),                   -- FED_SPEECH
    topic              VARCHAR(300),
    revision_history   JSONB        NOT NULL DEFAULT '[]',  -- append-only [{scheduled_at, status, source_name, at}]
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_events_scheduled_at ON events (scheduled_at);
CREATE INDEX IF NOT EXISTS ix_events_ticker       ON events (ticker, scheduled_at);
CREATE INDEX IF NOT EXISTS ix_events_type         ON events (event_type, scheduled_at);
CREATE INDEX IF NOT EXISTS ix_events_status       ON events (status);

CREATE TABLE IF NOT EXISTS market_calendar (
    session_date       DATE PRIMARY KEY,
    exchange           VARCHAR(16)  NOT NULL DEFAULT 'US',
    open_utc           TIMESTAMPTZ  NOT NULL,          -- regular session open (09:30 ET normally)
    close_utc          TIMESTAMPTZ  NOT NULL,          -- regular session close (16:00 ET; 13:00 ET on a half day)
    session_open_utc   TIMESTAMPTZ,                    -- extended session (04:00 ET); NULL when the provider omits it
    session_close_utc  TIMESTAMPTZ,                    -- extended session (20:00 ET); NULL when the provider omits it
    is_early_close     BOOLEAN      NOT NULL DEFAULT FALSE,
    source             VARCHAR(32)  NOT NULL,          -- alpaca_calendar | massive_calendar
    fetched_at         TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_market_calendar_exchange ON market_calendar (exchange, session_date);

CREATE TABLE IF NOT EXISTS event_ingest_state (
    key                VARCHAR(120) PRIMARY KEY,       -- "sec_edgar:NVDA" | "fed_fomc" | "alpaca_calendar"
    last_fetched_at    TIMESTAMPTZ,                    -- last attempt (success or failure)
    last_ok_at         TIMESTAMPTZ,                    -- last SUCCESS — gates the re-fetch cadence
    last_error         TEXT,                           -- last honest failure string; NULL after a success
    meta               JSONB        NOT NULL DEFAULT '{}'  -- last capability report / provider cursors
);
