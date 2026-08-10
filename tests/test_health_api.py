"""GET /api/health/strategy tests (plan §19).

Positions are seeded by DIRECT inserts (no execution chain) so every realized
PnL is hand-picked and every expected statistic below is hand-computed.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from apps.gateway.db import Position, SessionLocal
from apps.gateway.routers.health import HEALTH_PARAMS

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Hand-computed fixture (chronological order):
#   wins  : 100 + 200 + 30 + 80 + 60 + 150            = 620  (6 wins)
#   losses: 50 + 25 + 10 + 40 + 5 + 30                = 160  (6 losses)
#   cumulative curve: 100, 50, 250, 225, 255, 245, 325, 285, 345, 340, 490, 460
#   -> max drawdown 50 (peak 100 -> trough 50), current drawdown 490-460 = 30.
PNLS = [100.0, -50.0, 200.0, -25.0, 30.0, -10.0, 80.0, -40.0, 60.0, -5.0, 150.0, -30.0]


def closed_position(pnl: float, closed_at: datetime, ticker: str = "SEED") -> Position:
    return Position(
        ticker=ticker,
        quantity=10,
        avg_price=100.0,
        max_loss=100.0,
        status="CLOSED",
        closed_at=closed_at,
        realized_pnl=pnl,
    )


async def seed(positions: list[Position]) -> None:
    async with SessionLocal() as s:
        s.add_all(positions)
        await s.commit()


def assert_no_nan(body: dict) -> None:
    """Contract: undefined stats are null — never NaN/Infinity."""
    for key, value in body.items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{key} is {value}"


async def test_empty_book_is_insufficient_data(client):
    r = await client.get("/api/health/strategy")
    assert r.status_code == 200
    body = r.json()

    assert body["trade_count"] == 0
    assert body["min_trades_for_judgement"] == HEALTH_PARAMS.min_trades_for_judgement
    assert body["status"] == "INSUFFICIENT_DATA"
    # No trades: every per-trade statistic is undefined -> null.
    for key in ("win_rate", "profit_factor", "expectancy_usd", "avg_win_usd", "avg_loss_usd"):
        assert body[key] is None
    for key in ("gross_profit_usd", "gross_loss_usd", "cumulative_pnl_usd",
                "max_drawdown_usd", "current_drawdown_usd"):
        assert body[key] == 0.0
    datetime.fromisoformat(body["as_of"])
    assert body["explanations"]
    assert_no_nan(body)


async def test_seeded_closed_positions_match_hand_computed_stats(client):
    # Insert in REVERSED order with chronological closed_at stamps: the stats
    # only match the hand-computed values if the endpoint orders by closed_at
    # (reversed insertion order would give current_drawdown 0, not 30).
    rows = [
        closed_position(pnl, BASE + timedelta(days=i))
        for i, pnl in enumerate(PNLS)
    ]
    rows.reverse()
    # Excluded rows: an OPEN position with a partial-close realized_pnl, and a
    # CLOSED position with realized_pnl null (nulls excluded).
    rows.append(
        Position(ticker="OPEN", quantity=1, avg_price=10.0, max_loss=10.0,
                 status="OPEN", realized_pnl=9999.0)
    )
    rows.append(
        Position(ticker="NULL", quantity=1, avg_price=10.0, max_loss=10.0,
                 status="CLOSED", closed_at=BASE + timedelta(days=99),
                 realized_pnl=None)
    )
    await seed(rows)

    r = await client.get("/api/health/strategy")
    assert r.status_code == 200
    body = r.json()

    assert body["trade_count"] == 12
    assert body["win_rate"] == pytest.approx(0.5)
    assert body["gross_profit_usd"] == pytest.approx(620.0)
    assert body["gross_loss_usd"] == pytest.approx(160.0)
    assert body["profit_factor"] == pytest.approx(620.0 / 160.0)  # 3.875
    assert body["expectancy_usd"] == pytest.approx(460.0 / 12.0)
    assert body["avg_win_usd"] == pytest.approx(620.0 / 6.0)
    assert body["avg_loss_usd"] == pytest.approx(160.0 / 6.0)
    assert body["cumulative_pnl_usd"] == pytest.approx(460.0)
    assert body["max_drawdown_usd"] == pytest.approx(50.0)
    assert body["current_drawdown_usd"] == pytest.approx(30.0)  # proves closed_at ordering
    assert body["status"] == "HEALTHY"  # 12 >= 10 trades, PF 3.875, drawdown 30 < 310
    assert_no_nan(body)


async def test_no_losses_yet_gives_null_profit_factor(client):
    # Enough all-winning trades to pass the judgement threshold: profit factor
    # and avg_loss are undefined -> null, never Infinity/NaN.
    n = HEALTH_PARAMS.min_trades_for_judgement
    await seed([closed_position(50.0, BASE + timedelta(days=i)) for i in range(n)])

    r = await client.get("/api/health/strategy")
    assert r.status_code == 200
    body = r.json()

    assert body["trade_count"] == n
    assert body["win_rate"] == pytest.approx(1.0)
    assert body["profit_factor"] is None
    assert body["avg_loss_usd"] is None
    assert body["gross_loss_usd"] == 0.0
    assert body["avg_win_usd"] == pytest.approx(50.0)
    assert body["cumulative_pnl_usd"] == pytest.approx(50.0 * n)
    assert body["max_drawdown_usd"] == 0.0
    assert body["current_drawdown_usd"] == 0.0
    assert body["status"] == "HEALTHY"
    assert_no_nan(body)
