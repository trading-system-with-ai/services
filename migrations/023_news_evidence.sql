-- 023_news_evidence.sql — Catalyst & Event Intelligence Phase D: the news
-- EVIDENCE columns on the existing article mirror (event spec §21-§27, §59,
-- §81, §96; audit catalyst-event-audit.md §5.1, §7, §9.3, §11.5).
--
-- ADDITIVE ONLY. `news_articles` already exists (migration 012) and is the
-- verbatim provider mirror the grounding rule cites: source_id is the
-- provider's own article id and the UNIQUE dedupe key, and every row is real.
-- Nothing here rewrites that contract — this file adds five nullable/defaulted
-- columns and one index, so a live database that already holds a hundred
-- articles gains the columns without a rewrite and without losing a row.
--
-- Mirrors apps/gateway/db.py::NewsArticleRow EXACTLY, in this order appended
-- after `fetched_at` (mirror rule, README) — tests/test_migration_parity.py
-- pins the 012-CREATE + 023-ALTER column sequence against the ORM.
--
-- Design decisions:
-- * WHY STORE THE ANALYSIS AT ALL when apps/gateway/event_news.py recomputes
--   it on every read. Two reasons, both about the ARTICLE rather than the
--   answer. (a) `cluster_id` is the identity of a STORY, and a story outlives
--   one request: two events, two as-ofs and the Phase F package must all name
--   the same development the same way, and a deterministic id that is never
--   written down is one nobody else can join on. (b) The columns make the
--   store queryable — "which stored articles were REGULATION" is a question
--   the database can now answer without re-running the pipeline over the
--   whole mirror.
-- * ONLY AS-OF-INDEPENDENT FIELDS ARE PERSISTED (audit §7.1, §96). This is
--   the rule that decides the column list, and it is why `novelty`, `decay`
--   and the composite evidence `score` are NOT here. Those three are
--   functions of the as-of instant and of which OTHER articles were in the
--   window: novelty measures a headline against EARLIER clusters, decay is
--   age relative to as_of, and the score multiplies both in. Writing them to
--   a row would freeze one request's viewpoint into the mirror, and the next
--   read at a different as_of would silently inherit it — a look-ahead leak
--   wearing a cache's clothes. `materiality`, `materiality_score` and
--   `source_quality` are properties of the ARTICLE ITSELF (its own text, its
--   own publisher) and mean the same thing at every instant, so they are safe
--   to store. `cluster_id` is derived from the canonical article's source_id
--   and is likewise stable.
-- * `relevance` IS JSONB, NOT A FLOAT, and defaults to '{}'. §22 relevance is
--   per (article, TICKER): a syndicated piece tagged AAPL and MSFT is 1.0 to
--   one and 0.7 to the other, so a single column would have to pick a subject
--   and would then be wrong for the other. The object is
--   {"AAPL": 1.0, "MSFT": 0.7} — one entry per ticker the pipeline has
--   actually scored this article for, ABSENT rather than 0.0 for tickers it
--   has not (§44 rule 18: an unscored ticker and an irrelevant one are
--   different facts). NOT NULL DEFAULT '{}' so an existing row reads as "no
--   ticker scored yet" instead of NULL-meaning-unknown.
-- * THE OTHER FOUR ARE NULLABLE, deliberately. NULL means "this article has
--   not been through the evidence pipeline", which is the true state of every
--   one of the hundred rows migration 012's ingest path stored, and of every
--   row the recommendations refresh will store tomorrow (that path writes the
--   mirror, not the analysis). A DEFAULT of 0.0 or 'OTHER' would make an
--   un-analysed article indistinguishable from one classified as immaterial.
-- * `materiality` IS UNCONSTRAINED VARCHAR(32), no CHECK. The §24 category
--   vocabulary lives in libs/trading_core/events/news_intel.py::CATEGORY_ORDER
--   and is expected to grow; a CHECK here would be a second copy of it that
--   drifts, and migration 017 exists precisely because that happened to
--   `orders.side`. The writer is a single seam function, and the vocabulary
--   is pinned in the library's own tests.
-- * `materiality_score` and `source_quality` are DOUBLE PRECISION in [0,1] —
--   the §24 category weight and the §22 publisher weight as the library
--   computed them, stored so a reader can see WHY a story ranked where it did
--   without re-deriving it.
-- * GIN INDEX ON `tickers`. The hot query is "every stored article tagged T
--   published in [window_start, as_of]" — a containment test on a JSONB array
--   (`tickers @> '["AAPL"]'`), which without a GIN index is a sequential scan
--   of the whole mirror on every Catalyst page open. The existing
--   idx_news_articles_published_at (012) serves the range half. Postgres
--   only: the SQLite test harness has no JSONB, so the seam falls back to a
--   Python filter there (apps/gateway/event_news.py::_articles_for_ticker).
-- * NO INDEX ON cluster_id yet. It is written far more often than it is
--   queried — today nothing reads by cluster id, the payload carries clusters
--   in memory — and an index that no query uses is pure write cost.
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS cluster_id        VARCHAR(64);
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS materiality       VARCHAR(32);
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS materiality_score DOUBLE PRECISION;
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS source_quality    DOUBLE PRECISION;
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS relevance         JSONB NOT NULL DEFAULT '{}';

-- "Every article tagged T" — the containment half of the window query. The
-- published_at range half is already served by idx_news_articles_published_at
-- (migration 012).
CREATE INDEX IF NOT EXISTS idx_news_articles_tickers_gin
    ON news_articles USING GIN (tickers);
