-- 022_fundamental_statements.sql — Catalyst & Event Intelligence Phase E2:
-- the point-in-time mirror of filed company financial statements (event spec
-- §16, §28, §29, §30, §85, §96; audit catalyst-event-audit.md §7.1, §11.3).
--
-- Mirrors apps/gateway/db.py::FundamentalStatementRow EXACTLY (mirror rule,
-- README) — column order included, which tests/test_migration_parity.py pins.
--
-- Design decisions (audit §7.1, §11.3):
-- * WHY THIS TABLE EXISTS AT ALL. The as-of contract (audit §7.2 rule 1) says
--   every analysis collector reads STORED rows and filters them, and never
--   holds a provider handle. A fundamentals view built by calling the vendor
--   at read time could not answer "what did we know on 2025-10-30" — the
--   vendor serves its CURRENT XBRL view and nothing else. Mirroring the
--   filings locally is what makes the question answerable, and what makes it
--   answerable identically twice.
-- * `acceptance_datetime` IS THE POINT-IN-TIME KEY (§85, §96; audit §7.1).
--   It is the instant the filing became public — weeks after `end_date`, the
--   instant the PERIOD closed. Every consumer filters on
--   `acceptance_datetime <= as_of` and on nothing else: a Q3 ending
--   2025-09-27 was not knowable on 2025-09-28, and an as-of written against
--   `end_date` is precisely the look-ahead leak §96 requires a test to plant.
--   It is NULLABLE because a provider row may omit it; such a row is stored
--   (it is real, and dropping it would hide that the provider is degraded)
--   but the pure layer EXCLUDES it from every as-of answer with a stated
--   reason rather than guessing publication from `filing_date`.
-- * UNIQUE (ticker, timeframe, fiscal_year, fiscal_period, end_date) — the
--   natural key of a filed PERIOD, so re-fetching is idempotent and a nightly
--   refresh can only collide, never accumulate duplicate quarters (the same
--   database-level idempotence ADR-007 relies on for the event registry).
--   `end_date` is in the key because a fiscal year can be re-labelled by a
--   filer changing its year end, and `timeframe` because the same quarter is
--   served again inside the `ttm` and `annual` roll-ups.
--   RESTATEMENTS OVERWRITE (audit §7.3 "Massive serves the current XBRL
--   view; a restatement rewrites history"): a restated Q3 has the same
--   natural key and a LATER acceptance instant, so the upsert replaces the
--   values and moves `acceptance_datetime` forward. This is honest about what
--   the vendor actually sells — it does not retain the superseded original —
--   and the moved acceptance instant is itself the flag audit §7.3 asks for.
-- * `values` is JSONB and is the ONE justified JSON column here: it is the
--   provider's own flattened statement fields ("income_statement.revenues" ->
--   number), whose NAME SET differs per filer and per period, so it cannot be
--   a fixed column list without either truncating filers or inventing NULLs
--   that read as "reported nothing". Fields the filer did not report are
--   ABSENT from the object — never present as 0 (§44 rule 18: a missing capex
--   and a capex of zero are different facts). Every scalar the UI reads
--   (dates, the acceptance instant, the fiscal labels) is a typed column.
-- * `raw_fields_count` is how many fields the provider row carried BEFORE the
--   numeric filter, so a mostly-unparseable filing is detectable from the
--   database without re-fetching the vendor.
-- * `fetched_at` is when THIS platform stored the row, which is a different
--   fact from every date above (all of which are the filer's or the SEC's).
--   It is what the freshness line reports and what throttles the refresh.
-- * INDEXES: (ticker, acceptance_datetime DESC) serves the only hot query —
--   "the statements for T that were public at as_of, newest first" — and
--   (ticker, timeframe, end_date DESC) serves the fiscal-series read the
--   year-over-year comparison walks.
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS fundamental_statements (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(16)  NOT NULL,
    cik                 VARCHAR(32),                    -- SEC CIK when the provider reports it
    timeframe           VARCHAR(16)  NOT NULL,          -- quarterly | annual | ttm (provider vocabulary)
    fiscal_year         INTEGER,                        -- NULL when the provider omits it (never a guessed year)
    fiscal_period       VARCHAR(16)  NOT NULL,          -- Q1..Q4 | FY | TTM
    start_date          DATE         NOT NULL,          -- period start (the filer's fact)
    end_date            DATE         NOT NULL,          -- period end — NEVER the as-of key
    filing_date         DATE,                           -- the filing's own date; coarser than acceptance
    acceptance_datetime TIMESTAMPTZ,                    -- THE §85 as-of key: when the filing became public
    source_filing_url   TEXT,                           -- citation back to the filing (§91 DATA provenance)
    values              JSONB        NOT NULL DEFAULT '{}',  -- "statement.field" -> number; unreported fields ABSENT, never 0
    raw_fields_count    INTEGER      NOT NULL DEFAULT 0,     -- fields in the provider row BEFORE the numeric filter
    fetched_at          TIMESTAMPTZ  NOT NULL DEFAULT now(), -- when THIS platform stored it
    CONSTRAINT uq_fundamental_statements_period UNIQUE (ticker, timeframe, fiscal_year, fiscal_period, end_date)
);

CREATE INDEX IF NOT EXISTS ix_fundamental_statements_ticker_acceptance
    ON fundamental_statements (ticker, acceptance_datetime DESC);
CREATE INDEX IF NOT EXISTS ix_fundamental_statements_ticker_period
    ON fundamental_statements (ticker, timeframe, end_date DESC);
