-- 007_stock_bars_daily.sql — stored daily OHLCV bars (plan §4.2, ADR-005).
--
-- Mirrors the ORM model apps/gateway/db.py::StockBarDaily EXACTLY. The table
-- previously existed only via the dev-convenience Base.metadata.create_all in
-- init_db(); production schema is owned by migrations, so it is defined here.
-- If you change StockBarDaily, change this file in the same commit.
--
-- Historical bars are stored only for Watchlist symbols (plan §4.2) plus the
-- system reference indices SPY/QQQ/VIX (ADR-005). Rows are written by the
-- lazy backfill path together with a SYSTEM DATA_BACKFILL audit event in the
-- same transaction; (ticker, ts) is UNIQUE so a backfill can never duplicate
-- a bar. Daily granularity stays small at V1 scale, so this is a plain table,
-- not a hypertable (unlike stock_bars_1m in 002).

CREATE TABLE IF NOT EXISTS stock_bars_daily (
    id      SERIAL PRIMARY KEY,
    ticker  VARCHAR(16) NOT NULL,
    ts      DATE NOT NULL,
    open    DOUBLE PRECISION NOT NULL,
    high    DOUBLE PRECISION NOT NULL,
    low     DOUBLE PRECISION NOT NULL,
    close   DOUBLE PRECISION NOT NULL,
    volume  DOUBLE PRECISION NOT NULL,
    CONSTRAINT uq_stock_bars_daily_ticker_ts UNIQUE (ticker, ts)
);

CREATE INDEX IF NOT EXISTS ix_stock_bars_daily_ticker ON stock_bars_daily (ticker);
