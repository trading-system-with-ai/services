-- 026_macro_data.sql — Catalyst & Event Intelligence Phase G: MACRO
-- OBSERVATIONS and the TREASURY par yield curve (event spec §8, §38-§41, §39;
-- audit catalyst-event-audit.md §6 macro rows, §11.9).
--
-- Mirrors apps/gateway/db.py::MacroObservationRow and ::TreasuryYieldRow
-- EXACTLY, column order included (mirror rule, README) —
-- tests/test_migration_parity.py pins the sequence against the ORM.
--
-- Design decisions:
--
-- * WHY macro_observations IS KEYED (series_id, period) AND NOT (series_id,
--   release_date). An agency publishes ONE number per reference period and
--   then REVISES it: BLS restates seasonally-adjusted CPI every February and
--   BEA revises GDP three times for the same quarter. The identity of the
--   fact is "CUSR0000SA0 for 2026-07", not "the value printed on Aug 12", so
--   a re-fetch must OVERWRITE rather than accumulate a second row that would
--   silently double every MoM computed over the series. The revision history
--   is deliberately NOT modelled here: this platform stores the CURRENT
--   vintage and states the as-of question through `release_at`, which is the
--   honest scope for Phase G (a true vintage database is a different table
--   and a different promise).
--
-- * `period` IS THE REFERENCE PERIOD ('2026-07' / '2026-Q2'), NEVER THE
--   RELEASE DATE. It is the join key onto the release schedule and onto the
--   pure library's MacroPrint; keying prints on the release date is exactly
--   how a CPI that slips a day loses its history.
--
-- * `value` IS NULLABLE and there is NO DEFAULT. A series with a suppressed
--   or withheld observation must store NULL, because 0.0 for an index level
--   would produce a -100% MoM — the single most dangerous fabrication this
--   table could hold (§44 rule 18).
--
-- * `release_at` IS NULLABLE AND CARRIES `release_basis` BESIDE IT. BLS's own
--   data API returns no timestamps at all: the instant comes either from the
--   agency's published release schedule (basis 'SCHEDULED' — the release date
--   at 08:30 ET, the actual published time) or, when no schedule row covers
--   the period, from period-end + 45 days (basis 'ESTIMATED'). Those are
--   different claims about what was knowable when, and the as-of gate in
--   libs/trading_core/events/macro.py reads BOTH: a bare timestamp with no
--   basis would let an estimate be quoted as a published instant. Unconstrained
--   VARCHAR(16), no CHECK — the vocabulary lives in
--   macro.RELEASE_BASIS_SCHEDULED/RELEASE_BASIS_ESTIMATED and a CHECK here
--   would be a second copy that drifts (migration 017 exists because exactly
--   that happened to `orders.side`).
--
-- * `provider` records WHO served the number (bls | bea | stub). A stored
--   statistic whose source is unknown cannot be audited, and the stub writes
--   'stub' so a fixture can never be mistaken for a government publication on
--   a live database.
--
-- * treasury_yields IS KEYED ON curve_date ALONE, with the whole curve in one
--   JSONB `tenors` object. The Treasury publishes the curve as a ROW — thirteen
--   tenors for one business day — and the platform reads it that way (the 2Y
--   and the 10Y on the same session). Normalising to (curve_date, tenor) rows
--   would turn every read into a thirteen-way pivot for no query this platform
--   makes, and the tenor labels are the CSV's own spelling ('2 Yr', '10 Yr'),
--   which belong in the payload rather than in a column vocabulary that would
--   have to be migrated the next time Treasury adds a tenor.
--
-- * A MISSING TENOR IS ABSENT FROM THE JSONB, never 0.0. Treasury publishes an
--   empty cell for a tenor it did not quote (the 20Y was absent for years, the
--   30Y for a decade), and 0.0 basis points of yield is a claim nobody made.
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS macro_observations (
    series_id       VARCHAR(32)      NOT NULL,   -- the agency's own id, e.g. 'CUSR0000SA0'
    period          VARCHAR(16)      NOT NULL,   -- REFERENCE period '2026-07' | '2026-Q2'
    value           DOUBLE PRECISION,            -- NULL = withheld/suppressed; never 0
    release_at      TIMESTAMPTZ,                 -- when this became public; NULL = unknown
    release_basis   VARCHAR(16),                 -- SCHEDULED | ESTIMATED (see macro.py)
    provider        VARCHAR(16)      NOT NULL,   -- bls | bea | stub
    fetched_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, period)
);

-- "What was released in this window" — the §40 related-evidence read, which
-- scans by instant rather than by series.
CREATE INDEX IF NOT EXISTS ix_macro_observations_release_at
    ON macro_observations (release_at);

CREATE TABLE IF NOT EXISTS treasury_yields (
    curve_date      DATE             NOT NULL,   -- the publication session
    tenors          JSONB            NOT NULL DEFAULT '{}',  -- {'2 Yr': 4.21, ...}; a missing tenor is ABSENT
    provider        VARCHAR(16)      NOT NULL,   -- treasury | stub
    fetched_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (curve_date)
);
