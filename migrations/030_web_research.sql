-- 030_web_research.sql — Catalyst research upgrade: external WEB SEARCH runs
-- and their accepted evidence (SEARCH_PREDICTION_MARKET_UPGRADE_PLAN.md §4).
--
-- Mirrors apps/gateway/db.py::EventSearchRunRow and ::SearchEvidenceRow
-- EXACTLY, column order included (mirror rule, README) —
-- tests/test_migration_parity.py pins the sequence against the ORM.
--
-- Design decisions:
--
-- * TWO TABLES, RUN + EVIDENCE, because provenance is the product. A search
--   run is the auditable unit ("what did we ask, when, under which window,
--   what did it cost") and the evidence rows are what the run admitted. A
--   single flat table would repeat the plan/window on every row and make
--   "how many queries did this refresh spend" a DISTINCT-scan instead of a
--   column read. Cost transparency (plan Phase 12) is a first-class query.
--
-- * `event_search_runs.plan` STORES THE WHOLE SEARCH PLAN VERBATIM (JSONB).
--   The plan is deterministic code's output (query text, purpose, priority,
--   bounds) and the LLM never controls it — storing it is what lets an
--   auditor answer "why did the platform search for this" from the row
--   alone, without re-running the planner against code that may have moved.
--
-- * `window_start`/`window_end`/`window_basis`/`previous_event_id`/
--   `fallback_reason` are the research window CONTRACT (plan Phase 1): end
--   is always the request as_of, start is the previous comparable event or
--   an explicitly named per-type fallback. Storing basis and reason means a
--   window that used the fallback can never masquerade as one anchored on a
--   real previous event. previous_event_id is ON DELETE SET NULL: the run's
--   provenance outlives a re-ingested calendar row.
--
-- * `search_evidence.published_at` IS NULLABLE AND THAT IS THE POINT (§44
--   rule 18): search providers frequently omit publication times, and a row
--   without one is a DIFFERENT FACT from any invented timestamp. The as-of
--   gate treats NULL conservatively (excluded from point-in-time-sensitive
--   views, with the exclusion counted) — never coerced to retrieved_at,
--   which is the platform's fetch clock, not the document's.
--
-- * RAW AND SAFE TEXT BOTH STORED. `title`/`snippet` are the provider's
--   verbatim words (untrusted, for display/provenance); `safe_title`/
--   `safe_snippet` are the sanitize_for_llm outputs (markup/URL-stripped,
--   truncated) and are the ONLY forms the evidence bundle may hand the
--   model. `suspicious_instruction` flags injection-shaped text — flagged
--   rows stay visible in diagnostics and are EXCLUDED from model-facing
--   text, counted in the run's suppressed_suspicious (the §81 news
--   discipline, applied to the web).
--
-- * `source_tier` / `topic` / `relevance` are UNCONSTRAINED (no CHECK) and
--   NULLABLE: the vocabulary lives in libs/trading_core/events/
--   web_research.py (migration-017 lesson — a CHECK is a second copy that
--   drifts), and NULL means "not classified", which is a different fact
--   from UNKNOWN (a classification whose answer is "we cannot place this
--   source").
--
-- * `accepted` + `reject_reason` keep the REJECTED candidates too. A row
--   the pipeline dropped (duplicate, out of window, off-topic) is stored
--   with its reason rather than discarded, because "what the platform
--   refused to admit and why" is exactly what the Evidence tab's
--   transparency promise covers — and it is also how tests prove the as-of
--   gate fired rather than merely observing an absence.
--
-- * UNIQUE (run_id, canonical_url): within one run, one canonical document
--   is one row — the dedup layer's output is a database property. Across
--   runs the same URL may legitimately reappear (a later refresh re-admits
--   a still-relevant document under a new as_of).
--
-- * `evidence_key` is the STABLE citation id (e.g. "web:3f9c2ab1e4d0"),
--   derived from the canonical URL — the id the LLM's evidence_refs cite
--   and the validator resolves. Derived, not SERIAL, so the same document
--   cites identically across runs and analyses.
--
-- * ON DELETE CASCADE from events(id): a search run is a statement about
--   one event's research; there is nothing to preserve in an orphan.
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS event_search_runs (
    id                    SERIAL PRIMARY KEY,
    event_id              INTEGER     NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    as_of                 TIMESTAMPTZ NOT NULL,   -- the research as-of; window_end == as_of always
    window_start          TIMESTAMPTZ NOT NULL,
    window_end            TIMESTAMPTZ NOT NULL,
    window_basis          VARCHAR(48) NOT NULL,   -- PREVIOUS_COMPARABLE_EVENT | TYPE_DEFAULT_LOOKBACK (vocab in code)
    previous_event_id     INTEGER     REFERENCES events (id) ON DELETE SET NULL,
    fallback_reason       TEXT,                   -- NULL when anchored on a real previous event
    provider              VARCHAR(16) NOT NULL,   -- brave | stub — who served the results
    plan                  JSONB       NOT NULL DEFAULT '{}',  -- the deterministic search plan, verbatim
    queries_executed      INTEGER     NOT NULL DEFAULT 0,
    results_considered    INTEGER     NOT NULL DEFAULT 0,
    results_accepted      INTEGER     NOT NULL DEFAULT 0,
    suppressed_suspicious INTEGER     NOT NULL DEFAULT 0,     -- injection-shaped rows excluded from model text
    skipped               JSONB       NOT NULL DEFAULT '[]',  -- named skips (query -> reason), never silent
    status                VARCHAR(16) NOT NULL,   -- OK | PARTIAL | FAILED (vocab in code)
    error                 TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- "The latest run for this event at/under an as-of" — the read seam's query.
CREATE INDEX IF NOT EXISTS ix_event_search_runs_event
    ON event_search_runs (event_id, as_of);

CREATE TABLE IF NOT EXISTS search_evidence (
    id                     SERIAL PRIMARY KEY,
    run_id                 INTEGER          NOT NULL REFERENCES event_search_runs (id) ON DELETE CASCADE,
    event_id               INTEGER          NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    evidence_key           VARCHAR(64)      NOT NULL,   -- stable citation id, derived from canonical_url
    query                  TEXT             NOT NULL DEFAULT '',
    purpose                VARCHAR(64)      NOT NULL DEFAULT '',  -- the plan concept that asked for it
    title                  TEXT             NOT NULL DEFAULT '',  -- provider-verbatim; UNTRUSTED
    safe_title             TEXT             NOT NULL DEFAULT '',  -- sanitize_for_llm output; the model-facing form
    url                    TEXT             NOT NULL,
    canonical_url          TEXT             NOT NULL,
    publisher              TEXT,                        -- NULL = the provider stated none
    domain                 VARCHAR(255)     NOT NULL,
    published_at           TIMESTAMPTZ,                 -- NULL = provider omitted it; NEVER faked
    retrieved_at           TIMESTAMPTZ      NOT NULL,   -- the platform's fetch clock, not the document's
    snippet                TEXT             NOT NULL DEFAULT '',  -- provider-verbatim; UNTRUSTED
    safe_snippet           TEXT             NOT NULL DEFAULT '',
    suspicious_instruction BOOLEAN          NOT NULL DEFAULT FALSE,
    source_tier            VARCHAR(24),                 -- OFFICIAL..UNKNOWN (vocab in code); NULL = unclassified
    topic                  VARCHAR(64),
    relevance              DOUBLE PRECISION,            -- deterministic score; NULL = unscored, never 0
    rank                   INTEGER,                     -- provider ordering: retrieval provenance, not reliability
    result_type            VARCHAR(8)       NOT NULL,   -- web | news (vocab in code)
    provider               VARCHAR(16)      NOT NULL,
    accepted               BOOLEAN          NOT NULL,
    reject_reason          VARCHAR(64),                 -- NULL on accepted rows; named reason on rejected ones
    created_at             TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_search_evidence_run_url UNIQUE (run_id, canonical_url)
);

-- "Accepted evidence for this event" — the bundle composer's query.
CREATE INDEX IF NOT EXISTS ix_search_evidence_event
    ON search_evidence (event_id, accepted);
