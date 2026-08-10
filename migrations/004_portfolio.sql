-- 004_portfolio.sql — paper portfolio state (plan §11, §12.5).
--
-- Singleton portfolio row (id=1) holding the paper account's cash, seeded at
-- the paper_initial_cash default. NAV is always DERIVED (cash + open-position
-- market value), never stored. Positions carry max_loss — the dollar maximum
-- loss fixed at open (quantity * stop_distance) — the unit portfolio heat is
-- measured in (plan §12.5).

CREATE TABLE IF NOT EXISTS portfolio (
    id         INTEGER PRIMARY KEY,
    cash       DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO portfolio (id, cash)
VALUES (1, 100000.0)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS positions (
    id         SERIAL PRIMARY KEY,
    ticker     VARCHAR(16) NOT NULL,
    instrument VARCHAR(16) NOT NULL DEFAULT 'LONG_STOCK',
    quantity   INTEGER NOT NULL,
    avg_price  DOUBLE PRECISION NOT NULL,
    max_loss   DOUBLE PRECISION NOT NULL,   -- dollars = quantity * stop_distance at open
    status     VARCHAR(8) NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED
    opened_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_positions_ticker ON positions (ticker);
CREATE INDEX IF NOT EXISTS ix_positions_status ON positions (status);
