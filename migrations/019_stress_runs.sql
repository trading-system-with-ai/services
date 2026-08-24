-- 019_stress_runs.sql — Risk Engine Upgrade Phase D: persisted stress-scenario
-- history (spec risk_engine.md §25 historical stress, §26 hypothetical stress,
-- §51 stress UI, §56 "do not store only the latest value"; Phase B/D design
-- contract §8.4).
--
-- Mirrors apps/gateway/db.py::StressRunRow EXACTLY (mirror rule, README).
--
-- Design decisions (design §8.3–§8.5):
-- * ONE row per (snapshot build, scenario). Every snapshot build runs the
--   whole default catalogue, so the table is the scenario history: "what
--   would the book of that day have lost under that scenario" is answerable
--   after the fact without re-deriving a book that no longer exists.
-- * `scenario` is the catalogue NAME and `kind` its family (HISTORICAL |
--   HYPOTHETICAL | IV_GRID | USER). `validated` is FALSE for the whole
--   research grid (spec §11 / §24: unvalidated parameterisations are marked,
--   never silently promoted).
-- * `pnl_usd` is GAIN-POSITIVE (a stress LOSS is negative) and NULLABLE:
--   a scenario whose window falls outside the stored history is persisted as
--   an UNAVAILABLE row with its `reason`, never as a fabricated 0 (§44
--   rule 18). `pnl_pct_nav` is a FRACTION of NAV, NULL when NAV is unknown.
-- * `method_full_reval` / `method_delta_linear` count the legs priced each
--   way (spec §21/§22: an option priced DELTA_LINEAR because it has no IV is
--   labelled as such, and the count makes the degradation visible in
--   history).
-- * `params` is the scenario's reproducible parameter set (spot shock, the
--   per-ticker overrides, IV shock + its source, days forward, uniform-beta-1
--   flag); `per_position` is the per-leg P&L map. Both diagnostic-shaped,
--   hence JSONB — every scalar the UI reads is a typed column.
-- * A USER row (POST /api/risk/stress/run) has `snapshot_id` NULL: a
--   user-defined hypothesis is a read of the current book, not a snapshot
--   build, and it writes NO audit event (a read is not a decision) while
--   still keeping its history (spec §56).
--
-- Everything is SHADOW/RESEARCH: nothing here alters a Tier 0 decision.

CREATE TABLE IF NOT EXISTS stress_runs (
    id                   SERIAL PRIMARY KEY,
    snapshot_id          INTEGER REFERENCES risk_snapshots (id) ON DELETE CASCADE,
    scenario             VARCHAR(64) NOT NULL,
    kind                 VARCHAR(16) NOT NULL,          -- HISTORICAL | HYPOTHETICAL | IV_GRID | USER
    validated            BOOLEAN NOT NULL DEFAULT FALSE,
    pnl_usd              DOUBLE PRECISION,              -- gain-positive; NULL when unavailable
    pnl_pct_nav          DOUBLE PRECISION,              -- FRACTION of NAV; NULL when NAV unknown
    method_full_reval    INTEGER NOT NULL DEFAULT 0,
    method_delta_linear  INTEGER NOT NULL DEFAULT 0,
    health               VARCHAR(16) NOT NULL,
    reason               TEXT,
    params               JSONB NOT NULL DEFAULT '{}',
    per_position         JSONB NOT NULL DEFAULT '{}',
    as_of                TIMESTAMPTZ NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_stress_runs_snapshot ON stress_runs (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_stress_runs_scenario ON stress_runs (scenario, as_of);
CREATE INDEX IF NOT EXISTS ix_stress_runs_kind     ON stress_runs (kind, as_of);
