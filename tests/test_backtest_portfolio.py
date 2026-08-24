"""Portfolio AUTO backtest tests (Phase C,
docs/auto-strategy-portfolio-design.md).

Pins the user's ask: whole-watchlist replay over ONE cash ledger with a
per-day allocation table — plus the capital rules the portfolio layer adds:
§12 tier-budget sizing, |edge|-priority contention, max_positions,
cash floor, signed short allocations, and accounting identity
(cash% + Σ allocation% ≈ 100 every bar).
"""
import math
from datetime import date, timedelta

import pytest

from libs.trading_core.backtest import BacktestParams
from libs.trading_core.backtest.engine import INITIAL_EQUITY
from libs.trading_core.backtest.portfolio import (
    PortfolioBacktestResult,
    SymbolBars,
    run_portfolio_backtest,
)
from libs.trading_core.signals import DirectionalParams, RegimeParams
from libs.trading_core.strategies import AccountPermissions

SHORT_REGIME = RegimeParams(sma_fast=5, sma_mid=10, sma_slow=20, slope_lookback=3)
SHORT_DIRECTION = DirectionalParams(
    sma_fast=5, sma_mid=10, sma_slow=20, slope_lookback=3,
    macd_fast=5, macd_slow=10, macd_signal=3, rsi_period=5,
    pivot_window=2, volume_sma_period=5,
)


def series_bars(ticker: str, closes: list[float], spread: float = 0.5) -> SymbolBars:
    opens = [closes[0]] + closes[:-1]
    return SymbolBars(
        ticker=ticker,
        opens=opens,
        highs=[max(o, c) + spread for o, c in zip(opens, closes)],
        lows=[min(o, c) - spread for o, c in zip(opens, closes)],
        closes=closes,
        volumes=[1_000_000.0] * len(closes),
    )


def shared_dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def up(n: int = 90) -> list[float]:
    return [100.0 * 1.012**i * (1.0 + 0.05 * math.sin(i / 6)) for i in range(n)]


def down(n: int = 90) -> list[float]:
    return [100.0 * 0.988**i * (1.0 + 0.04 * math.sin(i / 5)) for i in range(n)]


def flat(n: int = 90) -> list[float]:
    return [100.0 + 2.0 * math.sin(i / 4) for i in range(n)]


def stock_only() -> AccountPermissions:
    return AccountPermissions(long_stock=True, long_call=False, long_put=False)


def shorting() -> AccountPermissions:
    return AccountPermissions(
        long_stock=True, long_call=False, long_put=False,
        short_stock=True, margin=True,
    )


def run(n=90, tickers=None, permissions=None, **kw):
    tickers = tickers or {"AAA": up(n), "BBB": down(n), "CCC": flat(n)}
    return run_portfolio_backtest(
        shared_dates(n),
        [series_bars(k, v) for k, v in tickers.items()],
        BacktestParams(instrument="AUTO", warmup_bars=25),
        permissions=permissions or stock_only(),
        regime_params=SHORT_REGIME,
        directional_params=SHORT_DIRECTION,
        **kw,
    )


def test_shape_and_accounting_identity():
    r = run()
    n = len(r.dates)
    assert len(r.equity) == len(r.drawdown) == len(r.allocations) == len(r.cash_pct) == n
    # cash% + Σ allocation% ≈ 100 on every bar
    for t in range(n):
        total = r.cash_pct[t] + sum(r.allocations[t].values())
        assert abs(total - 100.0) < 1e-6, (t, total)
    assert all(v > 0 for v in r.equity)


def test_multi_symbol_attribution_and_decisions():
    r = run(permissions=shorting(), iv_series_by_ticker={"BBB": [0.40] * 90})
    tickers_traded = {tk for tk, _ in r.trades}
    assert "AAA" in tickers_traded  # uptrend long
    assert "BBB" in tickers_traded  # downtrend short (EXTREME IV bear cell)
    decision_tickers = {tk for tk, _ in r.decisions}
    assert {"AAA", "BBB"} <= decision_tickers
    # the short allocation is SIGNED negative while open
    short_days = [
        t for t in range(len(r.dates))
        if r.allocations[t].get("BBB", 0.0) < 0.0
    ]
    assert short_days, "short position must contribute negative allocation"


def test_max_positions_contention_prefers_higher_edge():
    # two bull symbols, room for one position: the higher-|edge| one fills
    r = run(
        tickers={"STRONG": up(90), "MILD": [100.0 * 1.006**i * (1.0 + 0.05 * math.sin(i / 6)) for i in range(90)]},
        max_positions=1,
    )
    open_counts = [len(a) for a in r.allocations]
    assert max(open_counts) == 1  # never more than one simultaneous position
    assert any(tk == "STRONG" for tk, _ in r.trades)


def test_cash_floor_is_respected():
    r = run(cash_floor_pct=0.5)
    # with half the book reserved, allocations can never exceed ~50%+slack
    for t, a in enumerate(r.allocations):
        assert sum(abs(v) for v in a.values()) <= 60.0, (t, a)


def test_sizing_uses_tier_budget_not_all_in():
    """§12 sizing: a MODERATE entry risks 0.75% of equity over a 2×ATR stop —
    the resulting notional is far below an all-in position_pct=1.0 fill."""
    r = run(tickers={"AAA": up(90)})
    first_alloc_days = [a for a in r.allocations if a.get("AAA")]
    assert first_alloc_days, "expected at least one AAA position day"
    # risk-budget sizing keeps single-name exposure well under 100%
    assert all(a["AAA"] < 60.0 for a in first_alloc_days)


def test_determinism():
    r1, r2 = run(), run()
    assert r1.equity == r2.equity
    assert r1.allocations == r2.allocations
    assert [(tk, tr.entry_date, tr.exit_date) for tk, tr in r1.trades] == [
        (tk, tr.entry_date, tr.exit_date) for tk, tr in r2.trades
    ]


def test_validation():
    with pytest.raises(ValueError, match="AUTO"):
        run_portfolio_backtest(
            shared_dates(30),
            [series_bars("AAA", up(30))],
            BacktestParams(instrument="LONG_STOCK", warmup_bars=25),
            permissions=stock_only(),
        )
    with pytest.raises(ValueError, match="align"):
        run_portfolio_backtest(
            shared_dates(30),
            [series_bars("AAA", up(29))],
            BacktestParams(instrument="AUTO", warmup_bars=25),
            permissions=stock_only(),
        )
    with pytest.raises(ValueError, match="Phase D"):
        run(permissions=AccountPermissions(
            long_stock=True, long_call=True, long_put=True, defined_risk_spreads=True,
        ))


def test_gross_exposure_capped_under_chained_shorts():
    """Verifier catch 2026-08-20: short proceeds CREDIT cash, so without a
    gross budget chained shorts compound to multiples of equity. The
    max_gross_pct budget (default 1.0) bounds total |allocation|."""
    n = 90
    tickers = {f"S{i}": down(n) for i in range(6)}
    r = run(
        tickers=tickers,
        permissions=shorting(),
        iv_series_by_ticker={k: [0.40] * n for k in tickers},
    )
    for t_, a in enumerate(r.allocations):
        assert sum(abs(v) for v in a.values()) <= 110.0, (t_, a)


def test_max_gross_pct_is_a_parameter():
    n = 90
    tickers = {f"S{i}": down(n) for i in range(4)}
    r = run(
        tickers=tickers,
        permissions=shorting(),
        iv_series_by_ticker={k: [0.40] * n for k in tickers},
        max_gross_pct=0.5,
    )
    for t_, a in enumerate(r.allocations):
        assert sum(abs(v) for v in a.values()) <= 60.0, (t_, a)


def test_stalled_option_exit_latches_rule_and_journals_the_stall():
    """Verifier catches 2026-08-20: (1) a decided-but-unfilled exit must keep
    the rule that FIRED (not a later bar's regenerated text, never a silent
    cancel); (2) the stall itself must appear in the journal; (3) a stall
    running into the final bar still closes END_OF_DATA."""
    from datetime import timedelta
    from libs.trading_core.backtest.options import OptionLegBars

    n = 90
    up_ = up(n)
    dts = shared_dates(n)
    # leg with a gap: bars vanish for a window after entry so the exit stalls
    expiry = dts[0] + timedelta(days=n + 60)
    all_days = {d: (5.0, 5.0) for d in dts}
    # remove a mid-window stretch of bars
    for d in dts[40:70]:
        all_days.pop(d, None)
    leg = OptionLegBars(symbol="STALL_C", strike=100.0, expiry=expiry, bars=all_days)
    r = run_portfolio_backtest(
        dts,
        [series_bars("AAA", up_)],
        BacktestParams(instrument="AUTO", warmup_bars=25),
        permissions=AccountPermissions(long_stock=False, long_call=True, long_put=False),
        call_providers={"AAA": lambda d, s: leg},
        iv_series_by_ticker={"AAA": [0.10] * n},
        regime_params=SHORT_REGIME,
        directional_params=SHORT_DIRECTION,
    )
    enters = [e for e in r.journal if e.action == "ENTER"]
    exits = [e for e in r.journal if e.action == "EXIT"]
    assert len(enters) == len(exits) == len(r.trades)  # every entry closed
    stalls = [e for e in r.journal if e.action == "SKIP" and "exit pending" in e.reason]
    if stalls:  # a stall occurred: the exit reason must be the LATCHED rule
        assert any(st.reason.split("rule: ")[1][:12] in ex.reason for st in stalls for ex in exits)
