-- 002_system_state_and_bars.sql — global kill switch state (plan §18)
-- plus Phase-1 market-data groundwork (1-minute OHLCV hypertable).

-- Singleton row (id=1) backing the global kill switch. Trading is DISABLED by
-- default and only an explicit USER resume enables it; state survives restarts.
CREATE TABLE IF NOT EXISTS system_state (
    id              INTEGER PRIMARY KEY,
    trading_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    reason          TEXT NOT NULL DEFAULT 'startup default: trading disabled',
    updated_by      VARCHAR(64) NOT NULL DEFAULT '',
    updated_at      TIMESTAMPTZ
);

INSERT INTO system_state (id, trading_enabled, reason, updated_by)
VALUES (1, FALSE, 'startup default: trading disabled', '')
ON CONFLICT (id) DO NOTHING;

-- Phase-1 groundwork: 1-minute stock bars as a TimescaleDB hypertable.
CREATE TABLE IF NOT EXISTS stock_bars_1m (
    ticker  VARCHAR(16) NOT NULL,
    ts      TIMESTAMPTZ NOT NULL,
    open    DOUBLE PRECISION NOT NULL,
    high    DOUBLE PRECISION NOT NULL,
    low     DOUBLE PRECISION NOT NULL,
    close   DOUBLE PRECISION NOT NULL,
    volume  BIGINT NOT NULL,
    PRIMARY KEY (ticker, ts)
);

SELECT create_hypertable('stock_bars_1m', 'ts', if_not_exists => TRUE);
