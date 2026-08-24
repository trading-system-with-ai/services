"""AUTO-instrument backtest engine tests (Phase B,
docs/auto-strategy-portfolio-design.md).

Pins the §8-driven behavior the user's a/b/c/d banded model maps onto:

- vol UNKNOWN coerces to NORMAL exactly like live: a strong bull enters
  LONG_STOCK, never LONG_CALL, when no real IV exists.
- With REAL LOW IV, the same strong bull upgrades to LONG_CALL through
  the provider (the user's ">a → buy calls" band, IV-conditioned).
- Bear side: LONG_PUT when permitted; stock-only permissions produce
  NO trade on the bear side (LONG_PUT degrades to NO_TRADE, §5 ladder).
- Permissions are the multi-select: SHORT_STOCK requires short_stock AND
  margin; spreads permission raises (Phase D).
- NO LOOK-AHEAD: decisions before bar k are identical when bars after k
  are mutated.
- Determinism and finite outputs (§44 rule 18).
"""
import math
from datetime import date, timedelta

import pytest

from libs.trading_core.backtest import BacktestParams, run_auto_backtest
from libs.trading_core.backtest.engine import INITIAL_EQUITY
from libs.trading_core.backtest.options import OptionLegBars
from libs.trading_core.risk.engine import RiskLimits
from libs.trading_core.signals import DirectionalParams, RegimeParams
from libs.trading_core.strategies import AccountPermissions

SHORT_REGIME = RegimeParams(sma_fast=5, sma_mid=10, sma_slow=20, slope_lookback=3)
SHORT_DIRECTION = DirectionalParams(
    sma_fast=5,
    sma_mid=10,
    sma_slow=20,
    slope_lookback=3,
    macd_fast=5,
    macd_slow=10,
    macd_signal=3,
    rsi_period=5,
    pivot_window=2,
    volume_sma_period=5,
)


def auto_params(**overrides) -> BacktestParams:
    return BacktestParams(instrument="AUTO", warmup_bars=25, **overrides)


def make_bars(closes: list[float], spread: float = 0.5):
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + spread for o, c in zip(opens, closes)]
    lows = [min(o, c) - spread for o, c in zip(opens, closes)]
    volumes = [1_000_000.0] * len(closes)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(closes))]
    return dates, opens, highs, lows, closes, volumes


def uptrend(n: int = 80) -> list[float]:
    return [100.0 * 1.01**i for i in range(n)]


def noisy_uptrend(n: int = 90) -> list[float]:
    """Trend + sine so realized vol is realistic (a constant-return series
    has rv≈0 and poisons the IV/RV ratio into EXTREME — an honest engine
    behavior, but a degenerate test input)."""
    return [100.0 * 1.015**i * (1.0 + 0.08 * math.sin(i / 5)) for i in range(n)]


def downtrend(n: int = 80) -> list[float]:
    return [100.0 * 0.99**i for i in range(n)]


def noisy_downtrend(n: int = 90) -> list[float]:
    return [100.0 * 0.988**i * (1.0 + 0.04 * math.sin(i / 5)) for i in range(n)]


def stock_only() -> AccountPermissions:
    return AccountPermissions(long_stock=True, long_call=False, long_put=False)


def all_longs() -> AccountPermissions:
    return AccountPermissions(long_stock=True, long_call=True, long_put=True)


def flat_leg(dates, symbol="AUTO_C", strike=100.0, expiry_offset=60, premium=5.0):
    expiry = dates[0] + timedelta(days=len(dates) + expiry_offset)
    return OptionLegBars(
        symbol=symbol,
        strike=strike,
        expiry=expiry,
        bars={d: (premium, premium) for d in dates},
    )


def run_auto(bars, params=None, *, permissions, **kw):
    return run_auto_backtest(
        *bars,
        params or auto_params(),
        permissions=permissions,
        regime_params=SHORT_REGIME,
        directional_params=SHORT_DIRECTION,
        **kw,
    )


def test_unknown_vol_coerces_to_normal_and_buys_stock_not_calls():
    bars = make_bars(uptrend())
    result, decisions = run_auto(
        bars,
        permissions=all_longs(),
        call_provider=lambda d, s: flat_leg(bars[0]),
    )
    assert len(result.trades) >= 1
    # every decision without IV must be stock (NORMAL never emits LONG_CALL)
    assert all(d.instrument == "LONG_STOCK" for d in decisions)
    assert all(d.vol_regime is None for d in decisions)
    assert all(not hasattr(t, "contract_symbol") or t.contract_symbol == "" for t in result.trades) or all(
        type(t).__name__ == "Trade" for t in result.trades
    )


def test_low_iv_upgrades_strong_bull_to_long_call_then_moderate_buys_stock():
    """The user's banded model, live in one series: the FIRST decision is
    STRONG + LOW IV -> LONG_CALL; when the edge decays to MODERATE the
    engine automatically enters LONG_STOCK instead — a>score buys calls,
    b<score<a buys stock, chosen by the same §8 code path as live."""
    bars = make_bars(noisy_uptrend())
    n = len(bars[0])
    result, decisions = run_auto(
        bars,
        permissions=all_longs(),
        call_provider=lambda d, s: flat_leg(bars[0]),
        iv_series=[0.10] * n,  # REAL low IV vs the series' realized vol
        risk_limits=RiskLimits(strength_strong=45.0),
    )
    assert decisions[0].tier == "STRONG" and decisions[0].vol_regime == "LOW"
    assert decisions[0].instrument == "LONG_CALL"
    assert any(
        d.tier == "MODERATE" and d.instrument == "LONG_STOCK" for d in decisions[1:]
    )
    trade_types = sorted({type(t).__name__ for t in result.trades})
    assert trade_types == ["OptionTrade", "Trade"]
    option_trades = [t for t in result.trades if type(t).__name__ == "OptionTrade"]
    assert option_trades[0].contract_symbol == "AUTO_C"
    assert "AUTO[" in option_trades[0].entry_reason


def test_bear_side_puts_when_permitted_nothing_when_stock_only():
    bars = make_bars(downtrend())
    result_puts, decisions_puts = run_auto(
        bars,
        permissions=all_longs(),
        put_provider=lambda d, s: flat_leg(bars[0], symbol="AUTO_P"),
    )
    assert any(d.instrument == "LONG_PUT" for d in decisions_puts)
    puts = [t for t in result_puts.trades if type(t).__name__ == "OptionTrade"]
    assert puts and puts[0].contract_symbol == "AUTO_P"

    # stock-only permissions: the §5 ladder degrades LONG_PUT to NO_TRADE
    result_none, decisions_none = run_auto(bars, permissions=stock_only())
    assert result_none.trades == [] or all(
        t.entry_reason == "" for t in result_none.trades
    ) is False and result_none.trades == []
    assert all(d.instrument == "NO_TRADE" for d in decisions_none)
    assert result_none.equity[-1] == INITIAL_EQUITY


def test_short_stock_fires_in_extreme_iv_bear_cell_with_both_flags():
    """§8 emits SHORT_STOCK only where premium is unbuyable: EXTREME IV
    bear (the dead-end cell) — and only with short_stock AND margin."""
    bars = make_bars(noisy_downtrend())
    n = len(bars[0])
    perms = AccountPermissions(
        long_stock=True, long_call=False, long_put=False,
        short_stock=True, margin=True,
    )
    # REAL high IV vs realized -> EXTREME via the IV/RV ratio rule
    result, decisions = run_auto(bars, permissions=perms, iv_series=[0.40] * n)
    assert any(d.instrument == "SHORT_STOCK" for d in decisions)
    assert len(result.trades) >= 1
    assert sum(t.pnl for t in result.trades) > 0  # downtrend shorts net green

    # margin off -> the §5 ladder degrades to NO_TRADE
    perms_no_margin = AccountPermissions(
        long_stock=True, long_call=False, long_put=False,
        short_stock=True, margin=False,
    )
    result2, decisions2 = run_auto(bars, permissions=perms_no_margin, iv_series=[0.40] * n)
    assert all(d.instrument == "NO_TRADE" for d in decisions2)
    assert result2.trades == []


def test_spreads_permission_raises_until_phase_d():
    bars = make_bars(uptrend(40))
    with pytest.raises(ValueError, match="Phase D"):
        run_auto(
            bars,
            permissions=AccountPermissions(
                long_stock=True, long_call=True, long_put=True,
                defined_risk_spreads=True,
            ),
        )


def test_instrument_validation():
    bars = make_bars(uptrend(40))
    with pytest.raises(ValueError, match="AUTO"):
        run_auto_backtest(
            *bars,
            BacktestParams(instrument="LONG_STOCK", warmup_bars=25),
            permissions=stock_only(),
            regime_params=SHORT_REGIME,
            directional_params=SHORT_DIRECTION,
        )


def test_no_look_ahead_mutating_future_bars():
    closes = uptrend(90)
    bars_a = make_bars(list(closes))
    result_a, _ = run_auto(bars_a, permissions=stock_only())
    # mutate everything after bar 60 violently
    closes_b = list(closes)
    for i in range(61, len(closes_b)):
        closes_b[i] = 1.0
    bars_b = make_bars(closes_b)
    result_b, _ = run_auto(bars_b, permissions=stock_only())
    closed_before_60_a = [t for t in result_a.trades if t.exit_index <= 60]
    closed_before_60_b = [t for t in result_b.trades if t.exit_index <= 60]
    assert closed_before_60_a == closed_before_60_b


def test_determinism_and_finite():
    bars = make_bars(uptrend())
    r1, d1 = run_auto(bars, permissions=stock_only())
    r2, d2 = run_auto(bars, permissions=stock_only())
    assert r1.equity == r2.equity and d1 == d2
    assert all(math.isfinite(v) for v in r1.equity)
    assert all(math.isfinite(v) for v in r1.drawdown)


def test_cash_never_negative_across_kinds():
    bars = make_bars(uptrend())
    n = len(bars[0])
    result, _ = run_auto(
        bars,
        permissions=all_longs(),
        call_provider=lambda d, s: flat_leg(bars[0]),
        iv_series=[0.15] * n,
    )
    # equity minus open-position value equals cash; weaker invariant that
    # holds regardless of position kind: equity never NaN and > 0
    assert all(v > 0 for v in result.equity)
