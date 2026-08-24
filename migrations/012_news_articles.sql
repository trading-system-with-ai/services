-- 012_news_articles.sql — Phase 8 news ingestion (REAL articles only).
--
-- Mirrors apps/gateway/db.py::NewsArticleRow EXACTLY (mirror rule, README).
--
-- Every row is one article fetched VERBATIM from the market data provider
-- (Massive /v2/reference/news). source_id is the provider's own article id
-- and the DEDUPLICATION key: re-fetching the same feed inserts nothing.
-- LLM recommendations must ground their evidence in rows of this table —
-- the enrichment path rejects any draft citing an article that is not here.

CREATE TABLE IF NOT EXISTS news_articles (
    id           SERIAL PRIMARY KEY,
    source_id    VARCHAR(128) NOT NULL UNIQUE,
    title        TEXT         NOT NULL,
    publisher    VARCHAR(200) NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ  NOT NULL,
    url          TEXT         NOT NULL,
    tickers      JSONB        NOT NULL DEFAULT '[]',
    description  TEXT         NOT NULL DEFAULT '',
    fetched_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_news_articles_published_at
    ON news_articles (published_at DESC);
