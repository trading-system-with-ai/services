-- 031_prediction_markets.sql — Catalyst research upgrade: PREDICTION-MARKET
-- registry, snapshots, history and the event↔market match join
-- (SEARCH_PREDICTION_MARKET_UPGRADE_PLAN.md §4; plan Phases 3-4).
--
-- Mirrors apps/gateway/db.py::PredictionMarketRow, ::PredictionMarketSnapshotRow,
-- ::PredictionMarketPricePointRow and ::EventPredictionMarketRow EXACTLY,
-- column order included (mirror rule, README) — tests/test_migration_parity.py
-- pins the sequence against the ORM.
--
-- Design decisions:
--
-- * READ-ONLY SUBSYSTEM. Nothing in this schema can hold an order, a wallet,
--   a credential or a position: prediction markets are an OBSERVED layer of
--   market expectations (research input), and the absence of any such column
--   is deliberate — a column that could hold one is a column someone
--   eventually fills.
--
-- * `prediction_markets` IS PROVIDER-INDEPENDENT and keyed
--   UNIQUE(provider, provider_market_id): a Kalshi ticker can never collide
--   with a Polymarket condition id, and adding KalshiProvider (plan Phase 18)
--   is new rows, not new columns. `provider_event_id` is the PROVIDER'S own
--   grouping (a Polymarket "event") — it is NOT events.id and no FK joins
--   them: the explicit match table below owns that association.
--
-- * `resolution_criteria` IS STORED because the contract's exact wording
--   decides what the price means ("GDP above 2.5%" vs "… in the advance
--   estimate" are different claims). NULL = the provider did not state it —
--   an absence the matching layer must weigh, never paper over.
--
-- * SNAPSHOTS ARE APPEND-ONLY OBSERVATIONS keyed UNIQUE(market_id,
--   observed_at). `observed_at` is when THIS platform saw the prices — the
--   point-in-time identity every as-of read filters on. A later observation
--   never overwrites an earlier one: "what was the market pricing when the
--   analysis ran" must survive the market moving on.
--
-- * EVERY LIQUIDITY-ADJACENT COLUMN IS NULLABLE (volume, liquidity,
--   open_interest, best_bid/ask, spread) and that is the §44-rule-18
--   contract: a market with unreported depth is NOT a market with zero
--   depth. Zeroing an absent liquidity would silently upgrade the thinnest
--   markets to "confidently priced at zero depth" — the single most
--   dangerous fabrication this schema could hold, because the
--   interpretation layer weights price confidence BY liquidity.
--
-- * `prediction_market_history` HAS A NATURAL COMPOSITE PK
--   (market_id, outcome, ts) like stock_bars_1m: one outcome's price at one
--   instant is ONE fact, so a refetch can only overwrite it, never
--   duplicate it — backfill idempotence as a database property (ADR-007).
--   Points are provider truth verbatim; no interpolated row is ever written
--   (plan §3 "no invented interpolation").
--
-- * `event_prediction_markets` IS THE MATCH, NOT THE MARKET. relation is
--   DIRECT | DERIVED | CONTEXT (vocab in code, migration-017 lesson);
--   `matched_by` records WHICH matcher version decided (DETERMINISTIC_V1
--   today) so a future LLM-assisted classifier's rows are distinguishable
--   from rule-based ones; `accepted` keeps rejected candidates WITH their
--   reason — "what the platform refused to admit and why" is the Evidence
--   tab's transparency promise, and the LLM can never cite a market that
--   has no accepted row here. UNIQUE(event_id, market_id, as_of) lets the
--   match be re-decided under a later as_of without rewriting history.
--
-- * CASCADE RULES: event_prediction_markets cascades from BOTH parents (a
--   match is meaningless without either side); snapshots/history cascade
--   from their market. `prediction_markets` rows themselves have no FK to
--   events — a market is a fact about the prediction-market venue, not
--   about any one catalyst (the option_daily_bars precedent).
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS prediction_markets (
    id                  SERIAL PRIMARY KEY,
    provider            VARCHAR(16)  NOT NULL,   -- polymarket | stub — who serves this market
    provider_market_id  VARCHAR(128) NOT NULL,   -- the provider's own id, verbatim
    provider_event_id   VARCHAR(128),            -- the PROVIDER'S grouping id; never events.id
    question            TEXT         NOT NULL,
    url                 TEXT,
    outcomes            JSONB        NOT NULL DEFAULT '[]',  -- [{name, price}] provider-verbatim
    resolution_criteria TEXT,                    -- NULL = the provider did not state it
    end_date            TIMESTAMPTZ,
    market_status       VARCHAR(16)  NOT NULL,   -- ACTIVE | CLOSED | RESOLVED | UNKNOWN (vocab in code)
    first_seen_at       TIMESTAMPTZ  NOT NULL,
    last_seen_at        TIMESTAMPTZ  NOT NULL,
    raw                 JSONB        NOT NULL DEFAULT '{}',  -- provider payload, provenance only
    CONSTRAINT uq_prediction_markets_provider_id UNIQUE (provider, provider_market_id)
);

CREATE TABLE IF NOT EXISTS prediction_market_snapshots (
    id               SERIAL PRIMARY KEY,
    market_id        INTEGER          NOT NULL REFERENCES prediction_markets (id) ON DELETE CASCADE,
    observed_at      TIMESTAMPTZ      NOT NULL,   -- when THIS platform saw these prices
    outcome_prices   JSONB            NOT NULL DEFAULT '{}',  -- outcome name -> price; unpriced = absent, never 0
    best_bid         DOUBLE PRECISION,            -- NULL = unreported; never 0
    best_ask         DOUBLE PRECISION,
    midpoint         DOUBLE PRECISION,
    spread           DOUBLE PRECISION,
    last_trade_price DOUBLE PRECISION,
    volume           DOUBLE PRECISION,            -- NULL = unreported depth, NOT zero depth
    liquidity        DOUBLE PRECISION,
    open_interest    DOUBLE PRECISION,
    provider         VARCHAR(16)      NOT NULL,
    -- The UNIQUE constraint's backing index already serves "the latest
    -- snapshot at/under an as-of" (market_id, observed_at) — no separate
    -- index, which would be an exact duplicate Postgres maintains twice.
    CONSTRAINT uq_prediction_market_snapshots UNIQUE (market_id, observed_at)
);

CREATE TABLE IF NOT EXISTS prediction_market_history (
    market_id INTEGER          NOT NULL REFERENCES prediction_markets (id) ON DELETE CASCADE,
    outcome   VARCHAR(64)      NOT NULL,
    ts        TIMESTAMPTZ      NOT NULL,
    price     DOUBLE PRECISION NOT NULL,   -- a point with no price is not stored, never zeroed
    provider  VARCHAR(16)      NOT NULL,
    PRIMARY KEY (market_id, outcome, ts)
);

CREATE TABLE IF NOT EXISTS event_prediction_markets (
    id         SERIAL PRIMARY KEY,
    event_id   INTEGER          NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    market_id  INTEGER          NOT NULL REFERENCES prediction_markets (id) ON DELETE CASCADE,
    as_of      TIMESTAMPTZ      NOT NULL,   -- the research as-of the match was decided under
    relation   VARCHAR(16)      NOT NULL,   -- DIRECT | DERIVED | CONTEXT (vocab in code)
    relevance  DOUBLE PRECISION NOT NULL,   -- deterministic matcher score in [0, 1]
    reason     TEXT             NOT NULL DEFAULT '',
    ambiguity  TEXT,                        -- NULL = none noted
    matched_by VARCHAR(32)      NOT NULL,   -- DETERMINISTIC_V1 today; records the matcher version
    accepted   BOOLEAN          NOT NULL,
    reject_reason VARCHAR(64),              -- NULL on accepted rows
    -- BRACKET SERIES identity. A venue publishes a distribution as one
    -- contract per range ("GDP <0.5%", "0.5-1.0%", ... ">3.0%"); siblings
    -- share series_key. series_truncated marks a series the accept cap cut in
    -- half, so a PARTIAL distribution is labelled rather than drawn as whole.
    series_key VARCHAR(256),
    series_truncated BOOLEAN    NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_event_prediction_markets UNIQUE (event_id, market_id, as_of)
);

-- "Accepted matches for this event" — the bundle composer's query.
CREATE INDEX IF NOT EXISTS ix_event_prediction_markets_event
    ON event_prediction_markets (event_id, accepted);

-- "Every bracket of this series for this event" — how the UI decides whether
-- a distribution is complete.
CREATE INDEX IF NOT EXISTS ix_event_prediction_markets_series
    ON event_prediction_markets (event_id, series_key);
