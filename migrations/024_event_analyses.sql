-- 024_event_analyses.sql — Catalyst & Event Intelligence Phase F: the stored
-- EVENT ANALYSIS PACKAGE — the evidence bundle that was assembled, and the
-- LLM synthesis that was produced from exactly that bundle (event spec §16,
-- §46-§52, §69-§71, §99; audit catalyst-event-audit.md §7.2, §9.3, §11.6).
--
-- Mirrors apps/gateway/db.py::EventAnalysisRow EXACTLY (mirror rule, README)
-- — column order included, which tests/test_migration_parity.py pins.
--
-- Design decisions:
-- * THE BUNDLE IS STORED WITH THE ANALYSIS, NOT DERIVED AGAIN LATER. §47 says
--   the model never computes a number; every figure in `analysis` must be
--   quoted from `bundle`. That claim is only checkable if the exact bundle the
--   model saw is on disk beside the text it wrote. Re-deriving the bundle at
--   read time would rebuild it from TODAY's stored filings, prices and
--   articles — a different document — and the numbers_quoted validation would
--   then be verifying the wrong evidence. So `bundle` is a snapshot, not a
--   cache, and it is NOT NULL: a row without its evidence is not an analysis,
--   it is an assertion.
-- * `bundle_digest` is the sha256 of the bundle's canonical JSON and is the
--   CACHE KEY. Same event, same evidence, same prompt version, same model =>
--   the same answer, so re-pressing "Analyse" returns the stored row rather
--   than spending another call.
-- * THE UNIQUE INDEX IS PARTIAL — `WHERE status = 'OK'` — and that predicate
--   is the whole design, not an optimisation. The dedupe this table wants is
--   "do not pay twice for the same GOOD answer", and a plain four-column
--   UNIQUE would enforce something else and something wrong: it would also
--   forbid a SECOND ATTEMPT after a failure. A provider that 403s writes a
--   FAILED row carrying this event, this digest, this prompt version and this
--   model; under a total UNIQUE the user's retry — the one action the FAILED
--   status is inviting — could not be stored at all, and the same applies to
--   re-running after an INVALID answer and to an explicit `force`, all three
--   of which must INSERT so the older attempt stays readable as event memory
--   (§69) and a regression between model versions stays diagnosable.
--   Restricting the constraint to OK rows keeps the property that actually
--   matters — at most ONE cached good answer per (event, evidence, prompt,
--   model), so two concurrent handlers can only collide, never double-write,
--   which is the idempotence-by-unique-key ADR-007 relies on (no distributed
--   lock exists) — while letting the FAILED/INVALID trail accumulate.
--   `model` and `prompt_version` are IN the key because the same evidence
--   read by a different model, or under revised instructions, is a different
--   answer that must coexist with the old one rather than collide with it.
-- * `status` is the honest outcome vocabulary and it is NOT a boolean:
--     OK          - analysis returned and every quoted number checked out;
--     INVALID     - analysis returned but the validator found violations. The
--                   text IS STILL STORED, with its `violations` list, because
--                   hiding a model that quoted an invented number destroys the
--                   evidence that it did (§99 transparency); the UI badges it.
--     FAILED      - the provider raised (HTTP, transport, refusal). `error`
--                   keeps the honest string and `analysis` stays NULL — never
--                   a placeholder narrative (§44 rule 18).
--     BUNDLE_ONLY - the evidence was assembled and no synthesis was asked for;
--     SUPERSEDED  - was OK, until a FORCED re-run on the SAME evidence
--                   produced a newer good answer. Not a failure and not a
--                   correction: the text is untouched and stays readable as
--                   §69 event memory. It exists so that "re-ask the model"
--                   can INSERT rather than either colliding with the partial
--                   cache index or DELETING the previous answer — and
--                   deleting is precisely what would make a regression
--                   between two model versions undiagnosable.
-- * `analysis`, `usage`, `latency_ms` are NULLABLE for exactly that reason: a
--   failed call has no output, no token counts and no meaningful duration, and
--   writing 0 would read as "it answered instantly with nothing".
-- * `violations` is NOT NULL DEFAULT '[]' — an empty list ("checked, nothing
--   wrong") is a different fact from NULL ("never checked"), and every row
--   this platform writes HAS been checked.
-- * `kind` separates PRE_EVENT (the §46 preview, written before the release)
--   from POST_EVENT (the §71 retrospective). They are different documents
--   about the same event and both must be retainable — §69 event memory reads
--   the series.
-- * `as_of` is the instant the bundle was assembled AS OF, which is NOT
--   `created_at` (when the row was written): a historical analysis re-run for
--   2025-10-30 is created today and is as-of then. Every look-ahead question
--   is answered against `as_of`; `created_at` only orders the audit trail.
-- * ON DELETE CASCADE from events(id): an analysis is meaningless without the
--   event it analyses, and there is nothing to preserve in an orphan.
-- * INDEXES: (event_id, created_at DESC) serves the only hot reads — "the
--   latest analysis for this event" and "this event's history" — and
--   (ticker-less) chronology is covered by the event join.
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS event_analyses (
    id              SERIAL PRIMARY KEY,
    event_id        INTEGER      NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    as_of           TIMESTAMPTZ  NOT NULL,          -- the instant the bundle was assembled AS OF (never created_at)
    kind            VARCHAR(16)  NOT NULL DEFAULT 'PRE_EVENT',  -- PRE_EVENT | POST_EVENT
    bundle          JSONB        NOT NULL,          -- the EXACT evidence the model saw (DATA/QUANT tiers)
    bundle_digest   VARCHAR(64)  NOT NULL,          -- sha256 of the bundle's canonical JSON — the cache key
    analysis        JSONB,                          -- the LLM tier; NULL on FAILED / BUNDLE_ONLY
    provider        VARCHAR(32),                    -- openai | anthropic | stub
    model           VARCHAR(128),
    prompt_version  VARCHAR(32),                    -- "event-analysis-v1"
    usage           JSONB,                          -- {input_tokens, output_tokens}; NULL when the provider omits it
    latency_ms      INTEGER,                        -- NULL on failure (0 would read as "instant")
    violations      JSONB        NOT NULL DEFAULT '[]',  -- [] = checked and clean; never NULL
    status          VARCHAR(16)  NOT NULL,          -- OK | INVALID | FAILED | BUNDLE_ONLY | SUPERSEDED
    error           TEXT,                           -- the honest provider failure string
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- At most ONE cached good answer per (event, evidence, prompt, model). PARTIAL
-- on purpose: FAILED and INVALID attempts on the same evidence must still be
-- storable, because a retry is exactly what those statuses invite (see the
-- header). Expressed as a partial UNIQUE INDEX rather than a table CONSTRAINT
-- because PostgreSQL supports the predicate only on the index form.
CREATE UNIQUE INDEX IF NOT EXISTS uq_event_analyses_cache
    ON event_analyses (event_id, bundle_digest, prompt_version, model)
    WHERE status = 'OK';

CREATE INDEX IF NOT EXISTS ix_event_analyses_event_created
    ON event_analyses (event_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_event_analyses_as_of
    ON event_analyses (as_of DESC);
