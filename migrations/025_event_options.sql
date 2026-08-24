-- 025_event_options.sql — Catalyst & Event Intelligence Phase I: OPTION daily
-- bars and the per-event IMPLIED MOVE (event spec §18, §36, §37, §66; audit
-- catalyst-event-audit.md §7.3, §9.3, options section).
--
-- Mirrors apps/gateway/db.py::OptionDailyBarRow and ::EventOptionMetricRow
-- EXACTLY, column order included (mirror rule, README) —
-- tests/test_migration_parity.py pins the sequence against the ORM.
--
-- Design decisions:
--
-- * WHY OPTION BARS ARE THEIR OWN TABLE rather than rows in
--   `stock_bars_daily`. An option bar's identity is the CONTRACT — the OCC
--   symbol `O:AAPL250801C00210000` already encodes underlying, expiry, right
--   and strike — not a ticker. `stock_bars_daily.ticker` is VARCHAR(16) and
--   is joined by every equity query in the platform; a 21-character contract
--   symbol would not fit it, and if it did, "the bars for AAPL" would start
--   returning premiums. Separate identity, separate table.
--
-- * PRIMARY KEY (option_ticker, bar_date) — composite and NATURAL, like
--   `stock_bars_1m` and unlike `stock_bars_daily`'s surrogate id. One
--   contract's session is ONE fact, so a refetch can only overwrite it and
--   never duplicate it. That makes the backfill idempotent at the DATABASE
--   level, which is the property ADR-007 relies on in the absence of any
--   leader election: two concurrent presses of the button can collide, but
--   they cannot double-write.
--
-- * `volume` IS BIGINT AND NULLABLE. A daily option aggregate counts
--   CONTRACTS traded, a whole number — but an illiquid contract's bar can
--   arrive with no volume field at all, which is a different fact from "zero
--   contracts changed hands" (§44 rule 18). NULL is where that absence goes.
--   OHLC are NOT NULL by contrast: a bar without a close is not a bar, and
--   the adapter SKIPS such rows rather than storing a hole.
--
-- * `provider` records WHO served the premium. Massive is the only vendor
--   with dated option aggregates today (Alpaca's plan has none and its
--   adapter raises CapabilityNotAvailable), but a stored price whose source
--   is unknown cannot be audited — and the stub writes 'stub' here so a test
--   fixture can never be mistaken for market data on a live database.
--
-- * NO FOREIGN KEY FROM option_daily_bars TO events. A contract's bars are a
--   fact about the OPTION MARKET, not about any one event: the same
--   AAPL 210 call window feeds this quarter's earnings straddle and next
--   quarter's comparison. Keying them to an event would duplicate the same
--   premium per event and make a cascade delete destroy market data.
--
-- * `event_option_metrics` — WHY `basis` IS IN THE UNIQUE KEY rather than
--   being a flag. §37's whole point is that an implied move read off a LIVE
--   chain and one RECONSTRUCTED from daily closes are different claims with
--   different confidence: the live snapshot is a real bid/ask midpoint at a
--   known instant; the historical one is a settlement close standing in for a
--   mark nobody observed. UNIQUE(event_id, basis) lets both coexist for one
--   event — the live number computed while the print is still upcoming, the
--   historical reconstruction written after it happens — so neither can
--   silently overwrite the other, and a reader can compare what was priced
--   with what the reconstruction says was priced.
--
-- * EVERY PRICE AND METRIC COLUMN IS NULLABLE, and that is the §44 rule 18
--   contract, not laziness. If the put leg never traded on the pre-event
--   session there is no straddle, and the honest row carries
--   pre_put_close NULL, implied_move_pct NULL and status 'NO_DATA' with the
--   reason in `notes`. A DEFAULT of 0 would read as "the put was free" — the
--   single most dangerous fabrication this table could hold, because it would
--   halve every implied move that touched it.
--
-- * `status` is the honest outcome vocabulary, deliberately not a boolean:
--     OK      - a pre-event straddle AND a usable post side;
--     PARTIAL - the implied move computed but a downstream metric (the IV
--               crush, the realized ratio) did not. This is the value to use
--               whenever the straddle itself is real, because the UI treats a
--               finite number arriving beside NO_DATA as the server retracting
--               its own computation and SUPPRESSES it;
--     NO_DATA - no implied move at all. Nothing numeric is trustworthy.
--   Unconstrained VARCHAR(16), no CHECK: the vocabulary lives in
--   libs/trading_core/events/implied_move.py (STATUS_OK/PARTIAL/NO_DATA) and a
--   CHECK here would be a second copy that drifts — migration 017 exists
--   because exactly that happened to `orders.side`.
--
-- * `as_of` is the instant the metrics were computed AS OF and is NOT
--   `created_at` (when the row was written): a historical reconstruction for
--   2025-10-30 is created today and is as-of then. Every look-ahead question
--   is answered against `as_of`; `created_at` only orders the trail.
--
-- * `notes` is JSONB NOT NULL DEFAULT '{}' — the pure library's own reason
--   map, so a reader can answer "why is iv_after NULL" from the row alone
--   without re-running the pipeline. An empty object ("computed, nothing to
--   report") is a different fact from a row that was never analysed, and
--   every row this table holds HAS been through the builder.
--
-- * ON DELETE CASCADE from events(id): an implied move is a statement about
--   one event, and there is nothing to preserve in an orphan.
--
-- No migration runner exists (audit §13): this file runs only on a fresh
-- volume via the docker-compose :ro mount, so the live apply is manual and
-- every statement is IF NOT EXISTS / re-runnable.

CREATE TABLE IF NOT EXISTS option_daily_bars (
    option_ticker   VARCHAR(32)      NOT NULL,   -- provider-verbatim OCC, e.g. 'O:AAPL250801C00210000'
    bar_date        DATE             NOT NULL,   -- the bar's EASTERN session date
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT,                      -- contracts traded; NULL = the provider omitted it
    provider        VARCHAR(16)      NOT NULL,   -- massive | stub — who served the premium
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (option_ticker, bar_date)
);

-- "Every option bar on this session" — the only range read that is not already
-- served by the PK's leading column.
CREATE INDEX IF NOT EXISTS ix_option_daily_bars_date
    ON option_daily_bars (bar_date);

CREATE TABLE IF NOT EXISTS event_option_metrics (
    id                      SERIAL PRIMARY KEY,
    event_id                INTEGER          NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    as_of                   TIMESTAMPTZ      NOT NULL,   -- the instant computed AS OF (never created_at)
    basis                   VARCHAR(48)      NOT NULL,   -- LIVE_CHAIN_SNAPSHOT | HISTORICAL_DAILY_CLOSE_APPROXIMATION
    expiry                  DATE,
    strike                  DOUBLE PRECISION,
    spot                    DOUBLE PRECISION,
    call_ticker             VARCHAR(32),
    put_ticker              VARCHAR(32),
    pre_call_close          DOUBLE PRECISION,           -- NULL = the leg had no knowable mark; never 0
    pre_put_close           DOUBLE PRECISION,
    post_call_close         DOUBLE PRECISION,
    post_put_close          DOUBLE PRECISION,
    implied_move_pct        DOUBLE PRECISION,           -- FRACTION of spot (0.062 = 6.2%)
    implied_move_points     DOUBLE PRECISION,           -- the same move in underlying dollars
    iv_before               DOUBLE PRECISION,
    iv_after                DOUBLE PRECISION,
    iv_crush_pct            DOUBLE PRECISION,           -- fraction, despite the suffix
    actual_move_pct         DOUBLE PRECISION,           -- signed realized move; fraction
    implied_realized_ratio  DOUBLE PRECISION,           -- |actual| / implied, a bare multiple
    classification          VARCHAR(16),                -- UNDER_PRICED | FAIR | OVER_PRICED
    status                  VARCHAR(16)      NOT NULL,  -- OK | PARTIAL | NO_DATA
    notes                   JSONB            NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_event_option_metrics_basis UNIQUE (event_id, basis)
);

CREATE INDEX IF NOT EXISTS ix_event_option_metrics_event
    ON event_option_metrics (event_id);
