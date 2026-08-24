"""Portfolio advice tests (user mandate 2026-08-20): every item is
deterministic, method-labelled, evidenced, and honest about what it
cannot measure."""
from datetime import date, timedelta

from libs.trading_core.backtest.advice import AdviceParams, assess_portfolio_result


def days(n):
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def test_insufficient_data_is_honest():
    items = assess_portfolio_result([date(2024, 1, 1)], [100_000.0], [{}], [100.0], {})
    assert [i.code for i in items] == ["INSUFFICIENT_DATA"]
    assert items[0].evidence["bars"] == 1


def test_var_carries_method_label_and_fires_on_fat_tail():
    n = 100
    equity = [100_000.0]
    # mostly quiet days with ~6 violent ones: k = ceil(99×0.05) = 5 lands
    # inside the big-loss tail, so VaR95 = $6,000 ≈ 8% of final equity
    for i in range(1, n):
        step = -6_000.0 if i % 15 == 0 else 100.0
        equity.append(equity[-1] + step)
    items = assess_portfolio_result(days(n), equity, [{}] * n, [100.0] * n, {})
    tail = next(i for i in items if i.code == "TAIL_RISK")
    assert tail.severity == "SUGGESTION"
    assert "HISTORICAL" in tail.evidence["method"]  # §6: never a bare number
    # scale-free contract (verifier catch): VaR is native percent-of-equity,
    # USD is secondary and labelled at-final-equity; warn level echoed
    assert tail.evidence["var_pct_of_equity"] > 3.0
    assert tail.evidence["warn_level_pct"] == 3.0
    assert "of equity" in tail.finding["en"]
    assert "净值" in tail.finding["zh"]


def test_drawdown_warning_with_peak_and_trough_dates():
    n = 100
    equity = [100_000.0 + 100.0 * i for i in range(40)]
    equity += [equity[-1] * (1.0 - 0.01 * i) for i in range(1, 31)]  # -30% slide
    equity += [equity[-1]] * (n - len(equity))
    items = assess_portfolio_result(days(n), equity, [{}] * n, [100.0] * n, {})
    dd = next(i for i in items if i.code == "DRAWDOWN")
    assert dd.severity == "WARNING"
    assert dd.evidence["peak_date"] and dd.evidence["trough_date"]
    assert "max_gross_pct" in dd.suggestion["en"]  # actionable parameter named
    assert "max_gross_pct" in dd.suggestion["zh"]


def test_concentration_and_correlation_and_cash_drag():
    n = 100
    equity = [100_000.0] * n
    allocations = [{} for _ in range(n)]
    allocations[50] = {"AAA": 45.0}
    cash = [100.0] * n
    closes_a = [100.0 * 1.01**i * (1 + 0.02 * ((i % 7) - 3)) for i in range(n)]
    closes_b = [50.0 * 1.01**i * (1 + 0.02 * ((i % 7) - 3)) for i in range(n)]  # same shape -> high rho
    items = assess_portfolio_result(
        days(n), equity, allocations, cash, {"AAA": closes_a, "BBB": closes_b}
    )
    codes = {i.code for i in items}
    assert "CONCENTRATION" in codes
    conc = next(i for i in items if i.code == "CONCENTRATION")
    assert conc.evidence["ticker"] == "AAA" and conc.evidence["peak_abs_allocation_pct"] == 45.0
    assert "CORRELATION" in codes
    corr = next(i for i in items if i.code == "CORRELATION")
    assert corr.evidence["pairs"][0]["spearman"] >= 0.7
    assert "CASH_DRAG" in codes
    drag = next(i for i in items if i.code == "CASH_DRAG")
    assert "NOT loosening entry" in drag.suggestion["en"]  # anti-overfitting stance
    assert "放松入场阈值" in drag.suggestion["zh"]


def test_determinism():
    n = 80
    equity = [100_000.0 + (i % 9) * 50.0 for i in range(n)]
    a = assess_portfolio_result(days(n), equity, [{}] * n, [100.0] * n, {})
    b = assess_portfolio_result(days(n), equity, [{}] * n, [100.0] * n, {})
    assert a == b


def test_wiped_out_book_gets_loud_drawdown_warning():
    """Verifier catch: a NAV path through zero must produce an explicit
    WARNING, never a silently missing DRAWDOWN item."""
    n = 100
    equity = [100_000.0 - 1_500.0 * i for i in range(n)]  # crosses zero
    items = assess_portfolio_result(days(n), equity, [{}] * n, [0.0] * n, {})
    dd = next(i for i in items if i.code == "DRAWDOWN")
    assert dd.severity == "WARNING"
    assert "zero or below" in dd.finding["en"]
    assert "净值触及零" in dd.finding["zh"]
    assert dd.evidence["min_equity"] < 0


def test_misaligned_zero_close_does_not_crash_correlation():
    """Verifier catch: a single zero close in one symbol used to desync the
    return series and raise inside spearman — pairs now align by date."""
    n = 100
    equity = [100_000.0] * n
    a = [100.0 * 1.01**i for i in range(n)]
    b = [50.0 * 1.01**i for i in range(n)]
    b[40] = 0.0  # bad provider bar
    items = assess_portfolio_result(
        days(n), equity, [{}] * n, [100.0] * n, {"AAA": a, "BBB": b}
    )
    assert isinstance(items, list)  # no exception is the contract


def test_short_concentration_reports_signed_side():
    n = 100
    equity = [100_000.0] * n
    allocations = [{} for _ in range(n)]
    allocations[50] = {"AAA": -45.0}
    items = assess_portfolio_result(days(n), equity, allocations, [100.0] * n, {})
    conc = next(i for i in items if i.code == "CONCENTRATION")
    assert conc.evidence["side"] == "SHORT"
    assert conc.evidence["peak_signed_allocation_pct"] == -45.0
    assert "-45.0%" in conc.finding["en"] and "SHORT" in conc.finding["en"]
    assert "空头" in conc.finding["zh"]
