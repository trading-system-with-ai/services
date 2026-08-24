-- 027_fed_documents.sql — Catalyst & Event Intelligence Phase H: FEDERAL
-- RESERVE DOCUMENTS, stored VERBATIM (event spec §9, §42-§45; audit
-- catalyst-event-audit.md §11.9).
--
-- Mirrors apps/gateway/db.py::FedDocumentRow EXACTLY, column order included
-- (mirror rule, README) — tests/test_migration_parity.py pins the sequence
-- against the ORM.
--
-- Design decisions:
--
-- * WHY THE TEXT IS STORED AT ALL, when the URL is right there. §44 makes the
--   SOURCE DOCUMENT authoritative: the sentence-level diff that says what the
--   Committee changed at its last meeting is computed over the statement's
--   own words, and a diff whose inputs are re-fetched at read time would
--   change under the reader whenever federalreserve.gov edits a page, retires
--   a URL, or answers 403. Storing the paragraphs is what makes the diff
--   REPRODUCIBLE and what lets the read endpoint hold no network handle at
--   all. It is also the only way an as-of replay can show the June statement
--   as it stood in June.
--
-- * `url` IS THE UNIQUE KEY, and there is no (doc_type, meeting_date) unique
--   constraint beside it. One Fed URL is one document forever — the Fed never
--   republishes monetary20260729a.htm as a different statement — so the URL is
--   the natural identity and the backfill's upsert lands on it. Keying on
--   (doc_type, meeting_date) instead would be wrong for SPEECH rows, which
--   have no meeting date and several of which share a day.
--
-- * `meeting_date` IS THE MEETING, NEVER THE RELEASE. A statement's meeting
--   date is its decision day; MINUTES carry the meeting's END date even though
--   they are published three weeks later. That separation is the whole point:
--   `released_at` answers "when could a reader have seen this" (the as-of
--   gate) and `meeting_date` answers "which meeting is this about" (the join
--   onto the FOMC_* event rows), and collapsing them would make an as-of
--   replay show June's minutes on the day of the June meeting — twenty-one
--   days before they existed.
--
-- * `released_at` IS NULLABLE, and NULL means UNKNOWN rather than "now" or
--   "the meeting date". The minutes' publication instant comes only from the
--   press_monetary RSS feed, which reaches back a finite distance; when the
--   feed does not cover a meeting the honest answer is that this platform does
--   not know when the document became public. The as-of gate in
--   libs/trading_core/events/fed_intel.py treats an unknown instant as
--   ungateable and the payload says so, which is a different claim from
--   "released at midnight" (§44 rule 18).
--
-- * `paragraphs` IS JSONB AND `text` IS ITS JOIN, stored together on purpose.
--   The diff consumes the paragraph ARRAY (paragraph boundaries are where the
--   Committee's structure lives) and a human — or the LLM — reads `text`.
--   They are derived from one parse in the same INSERT so they cannot drift,
--   and re-deriving `text` in SQL at read time would put a second definition
--   of "the document" in the query layer.
--
-- * `parsed` HOLDS ONLY WHAT THE DOCUMENT STATES OUTRIGHT — the vote
--   ({for, against, dissenters, text}), the target range ({low_pct, high_pct,
--   text}) and, for a speech, the speaker. Each carries the SENTENCE it was
--   read from, so the UI can show the source line beside the number. Nothing
--   inferred, scored or summarised is stored here: §43 forbids a single
--   hawkish/dovish label anywhere in this platform, and a column that could
--   hold one is a column someone will eventually fill.
--
-- * `event_id` IS `ON DELETE SET NULL`, unlike the CASCADE that event_analyses
--   and event_option_metrics use. A statement is a FACT PUBLISHED BY THE
--   FEDERAL RESERVE; the events row is this platform's calendar entry pointing
--   at it. Deleting a mis-ingested calendar row must not delete the Fed's
--   words — the document outlives our index of it, and the next ingest tick
--   re-links it.
--
-- * `provider` records WHO served the document ('fed_docs' | 'stub'), so a
--   fixture can never be mistaken for federalreserve.gov on a live database.
--
-- * `id` IS `SERIAL`, NOT `BIGSERIAL`, and `event_id` is INTEGER to match.
--   Every surrogate key in this schema is int4 (migrations 021, 024, 025) and
--   `events.id` — the target of this foreign key — is one of them; a BIGINT
--   referencing a SERIAL is a type mismatch Postgres accepts and then indexes
--   badly. The Fed publishes eight statements a year, so int4 outlives the
--   platform by several centuries.

CREATE TABLE IF NOT EXISTS fed_documents (
    id              SERIAL,
    doc_type        VARCHAR(24)      NOT NULL,   -- STATEMENT | MINUTES | SPEECH
    meeting_date    DATE,                        -- decision day / meeting END; NULL for a speech
    event_id        INTEGER          REFERENCES events(id) ON DELETE SET NULL,
    url             TEXT             NOT NULL,   -- the Fed's own URL; the identity
    title           TEXT             NOT NULL DEFAULT '',
    released_at     TIMESTAMPTZ,                 -- when it became public; NULL = unknown
    text            TEXT             NOT NULL DEFAULT '',   -- paragraphs joined by \n\n
    paragraphs      JSONB            NOT NULL DEFAULT '[]', -- verbatim, in document order
    parsed          JSONB            NOT NULL DEFAULT '{}', -- vote | target_range | speaker
    provider        VARCHAR(16)      NOT NULL,   -- fed_docs | stub
    fetched_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    UNIQUE (url)
);

-- "The statement for the meeting that ended on this date" and "the minutes of
-- that meeting" — the two lookups the §44 diff makes, both keyed on the
-- meeting rather than on the release.
CREATE INDEX IF NOT EXISTS ix_fed_documents_type_meeting
    ON fed_documents (doc_type, meeting_date);
