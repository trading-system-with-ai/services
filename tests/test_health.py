import math
from datetime import datetime, timezone

import pytest

from libs.trading_core.health import HealthParams, compute_health

# ---------------------------------------------------------------------------
# Gateway liveness/readiness endpoints (pre-existing)
# ---------------------------------------------------------------------------


async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_readyz_checks_database(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"


# ---------------------------------------------------------------------------
# Strategy Health Monitor v0 (plan §19) — pure-function tests
# ---------------------------------------------------------------------------

AS_OF = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)

# Hand-computed reference case, 12 mixed trades:
#   pnls        = [100, -50, 200, -100,  50, -25, 150, -75, 300, -50, 100, -200]
#   wins        = 100+200+50+150+300+100          = 900   (6 wins)
#   losses(mag) = 50+100+25+75+50+200             = 500   (6 losses)
#   win_rate    = 6/12                            = 0.5
#   profit_fac  = 900/500                         = 1.8
#   expectancy  = (900-500)/12                    = 33.3333...
#   avg_win     = 900/6                           = 150
#   avg_loss    = 500/6                           = 83.3333...
#   cumulative  = 100, 50, 250, 150, 200, 175, 325, 250, 550, 500, 600, 400
#   peaks       = 100,100, 250, 250, 250, 250, 325, 325, 550, 550, 600, 600
#   drawdowns   =   0, 50,   0, 100,  50,  75,   0,  75,   0,  50,   0, 200
#   max_dd      = 200; current_dd = 600 - 400 = 200
#   status: pf 1.8 >= 1.2 and dd 200 <= 0.5*900=450  -> HEALTHY
MIXED_12 = [100.0, -50.0, 200.0, -100.0, 50.0, -25.0, 150.0, -75.0, 300.0, -50.0, 100.0, -200.0]


def test_hand_computed_mixed_case():
    stats = compute_health(MIXED_12, as_of=AS_OF)
    assert stats.as_of == AS_OF.isoformat()
    assert stats.trade_count == 12
    assert stats.min_trades_for_judgement == 10
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.profit_factor == pytest.approx(1.8)
    assert stats.expectancy_usd == pytest.approx(400.0 / 12.0)
    assert stats.avg_win_usd == pytest.approx(150.0)
    assert stats.avg_loss_usd == pytest.approx(500.0 / 6.0)
    assert stats.gross_profit_usd == pytest.approx(900.0)
    assert stats.gross_loss_usd == pytest.approx(500.0)
    assert stats.cumulative_pnl_usd == pytest.approx(400.0)
    assert stats.max_drawdown_usd == pytest.approx(200.0)
    assert stats.current_drawdown_usd == pytest.approx(200.0)
    assert stats.status == "HEALTHY"
    assert stats.explanations  # always populated


def test_no_losses_yet_profit_factor_none_and_healthy():
    # 12 winners: profit factor undefined -> None (never Infinity), HEALTHY.
    stats = compute_health([50.0] * 12, as_of=AS_OF)
    assert stats.profit_factor is None
    assert stats.win_rate == pytest.approx(1.0)
    assert stats.avg_loss_usd is None
    assert stats.gross_loss_usd == 0.0
    assert stats.max_drawdown_usd == 0.0
    assert stats.status == "HEALTHY"


def test_below_min_trades_is_insufficient_data_regardless_of_numbers():
    # 5 catastrophic losses — pf 0.0 would scream PAUSE, but §19 withholds
    # judgement below the minimum sample of completed trades.
    stats = compute_health([-500.0] * 5, as_of=AS_OF)
    assert stats.trade_count == 5
    assert stats.status == "INSUFFICIENT_DATA"
    assert any("withheld" in e for e in stats.explanations)


def test_empty_history_is_insufficient_data_with_nulls():
    stats = compute_health([], as_of=AS_OF)
    assert stats.status == "INSUFFICIENT_DATA"
    assert stats.trade_count == 0
    assert stats.win_rate is None
    assert stats.profit_factor is None
    assert stats.expectancy_usd is None
    assert stats.avg_win_usd is None
    assert stats.avg_loss_usd is None
    assert stats.cumulative_pnl_usd == 0.0


def test_pause_threshold():
    # [50, -100] * 6: gross profit 300, gross loss 600 -> pf 0.5 < 1.0
    stats = compute_health([50.0, -100.0] * 6, as_of=AS_OF)
    assert stats.profit_factor == pytest.approx(0.5)
    assert stats.status == "PAUSE_RECOMMENDED"
    assert any("pause threshold" in e for e in stats.explanations)


def test_warning_threshold_on_profit_factor():
    # [110, -100] * 6: gross profit 660, gross loss 600 -> pf 1.1 in [1.0, 1.2)
    stats = compute_health([110.0, -100.0] * 6, as_of=AS_OF)
    assert stats.profit_factor == pytest.approx(1.1)
    assert stats.status == "WARNING"


def test_warning_on_current_drawdown_despite_healthy_profit_factor():
    # 10 wins of 100 then 2 losses of 300:
    #   gross profit 1000, gross loss 600 -> pf 1.6667 >= 1.2 (no pf warning)
    #   peak 1000, final 400 -> current dd 600 > 0.5 * 1000 = 500 -> WARNING
    stats = compute_health([100.0] * 10 + [-300.0, -300.0], as_of=AS_OF)
    assert stats.profit_factor == pytest.approx(1000.0 / 600.0)
    assert stats.current_drawdown_usd == pytest.approx(600.0)
    assert stats.max_drawdown_usd == pytest.approx(600.0)
    assert stats.status == "WARNING"
    assert any("drawdown" in e.lower() for e in stats.explanations)


def test_custom_params_change_thresholds():
    pnls = [100.0, -50.0, 200.0, -40.0]  # pf = 300/90 = 3.3333, only 4 trades

    # Default params: below min trades -> judgement withheld.
    assert compute_health(pnls, as_of=AS_OF).status == "INSUFFICIENT_DATA"

    # Lower the sample requirement and raise thresholds: same data, new verdicts.
    healthy = compute_health(
        pnls,
        HealthParams(min_trades_for_judgement=3, profit_factor_warning=3.0, profit_factor_pause=1.0),
        as_of=AS_OF,
    )
    assert healthy.status == "HEALTHY"

    paused = compute_health(
        pnls,
        HealthParams(min_trades_for_judgement=3, profit_factor_warning=5.0, profit_factor_pause=4.0),
        as_of=AS_OF,
    )
    assert paused.profit_factor == pytest.approx(300.0 / 90.0)
    assert paused.status == "PAUSE_RECOMMENDED"


def test_determinism_same_inputs_same_stats():
    assert compute_health(MIXED_12, as_of=AS_OF) == compute_health(MIXED_12, as_of=AS_OF)


@pytest.mark.parametrize(
    "pnls",
    [[], [50.0] * 12, [-500.0] * 12, MIXED_12, [0.0] * 12],
)
def test_no_nan_or_infinity_anywhere(pnls):
    stats = compute_health(pnls, as_of=AS_OF)
    numeric_fields = [
        stats.win_rate,
        stats.profit_factor,
        stats.expectancy_usd,
        stats.avg_win_usd,
        stats.avg_loss_usd,
        stats.gross_profit_usd,
        stats.gross_loss_usd,
        stats.cumulative_pnl_usd,
        stats.max_drawdown_usd,
        stats.current_drawdown_usd,
    ]
    for value in numeric_fields:
        assert value is None or math.isfinite(value)
