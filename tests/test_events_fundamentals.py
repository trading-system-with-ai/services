"""Point-in-time fundamentals — pure arithmetic (event spec §16, §28, §29,
§30, §33, §35, §85, §96; audit §7, §11.3; Phase E2 unit U2).

Every number here is hand-checkable: revenues are round hundreds so a 46%
gross margin reads off the page, and no test asserts a value the module
computed for itself. Five contracts are pinned:

1. **The as-of gate is on ``acceptance_datetime``, never on the fiscal
   period** (§7/§85/§96) — a quarter that ENDED before as_of but was ACCEPTED
   one hour after it is invisible, and a row with no acceptance instant is
   excluded rather than admitted on its end date. This is the audit's
   sentinel leak, tested from both directions.
2. **Do not compute a ratio if the inputs are unavailable** (§28) — a missing
   field, a zero revenue and a zero equity each yield ``None`` plus a reason,
   and never a zero, a NaN or an ``inf``.
3. **The provider's structural gaps are named** — capex, cash, EBITDA,
   receivables are absent from the feed, so FCF, net debt, ROIC, the quick
   ratio, EV/EBITDA and the FCF yield carry a permanent reason rather than a
   value. ``total_debt`` is present but long-term only, and says so.
4. **YoY needs the SAME fiscal quarter one year back** (§28) — four rows back
   is not a substitute, and a gap in the history refuses the comparison.
5. **§29 change math** — bps for the ratio metrics only, arrows, trend over
   the stored history, and a reason whenever either side is missing.
"""
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Mapping

import pytest

from libs.trading_core.events import (
    BPS_METRICS,
    FUNDAMENTALS_MODEL_VERSION,
    METRIC_ORDER,
    NOT_REPORTED_REASONS,
    PEER_CONTEXT_REASON,
    VALUATION_METRICS,
    FundamentalSnapshot,
    Statement,
    StatementLike,
    build_snapshot,
    coerce_statement,
    expectations_gap_inputs,
    percentile_of,
    select_statements_as_of,
    snapshot_change,
    valuation_context,
)
from libs.trading_core.events.fundamentals import (
    DEFAULT_TREND_WINDOW,
    METRIC_NOTES,
    MetricChange,
    StatementRef,
    _classify_trend,
)

UTC = timezone.utc

# Flattened keys, spelled once so a fixture edit cannot drift from the module.
REV = "income_statement.revenues"
GP = "income_statement.gross_profit"
OPINC = "income_statement.operating_income_loss"
NI = "income_statement.net_income_loss"
NI_PARENT = "income_statement.net_income_loss_attributable_to_parent"
EPS = "income_statement.diluted_earnings_per_share"
SHARES = "income_statement.diluted_average_shares"
OCF = "cash_flow_statement.net_cash_flow_from_operating_activities"
ASSETS = "balance_sheet.assets"
CA = "balance_sheet.current_assets"
CL = "balance_sheet.current_liabilities"
EQ = "balance_sheet.equity"
EQ_PARENT = "balance_sheet.equity_attributable_to_parent"
LTD = "balance_sheet.long_term_debt"


# ---------------------------------------------------------------------------
# Fixtures — two row shapes, both duck-typed, neither importing the provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderRow:
    """Stands in for ``libs.market_data.provider.FinancialStatement``.

    The pure module may not import that class (audit §7.4), so the test does
    not either: what is pinned is that ANY object with these attributes works,
    which is the whole point of the structural protocol.
    """

    fiscal_year: int | None
    fiscal_period: str
    end_date: date | None
    acceptance_datetime: datetime | None
    values: Mapping[str, float]
    timeframe: str = "quarterly"
    filing_date: date | None = None


def full_quarter(
    year: int = 2025,
    period: str = "Q3",
    *,
    accepted: datetime | None = None,
    revenue: float = 1_000.0,
    **overrides: float,
) -> ProviderRow:
    """A complete quarterly statement: revenue 1000, gross profit 460, ...

    Hand-checkable by construction — gross margin 46%, operating margin 30%,
    net margin 25%, current ratio 2.0, D/E 0.5.
    """
    values = {
        REV: revenue,
        GP: 460.0,
        OPINC: 300.0,
        NI: 250.0,
        EPS: 2.0,
        SHARES: 100.0,
        OCF: 350.0,
        ASSETS: 5_000.0,
        CA: 2_000.0,
        CL: 1_000.0,
        EQ: 2_000.0,
        LTD: 1_000.0,
    }
    values.update(overrides)
    return ProviderRow(
        fiscal_year=year,
        fiscal_period=period,
        end_date=date(year, 9, 30),
        acceptance_datetime=accepted or datetime(year, 10, 31, 20, 30, tzinfo=UTC),
        timeframe="quarterly",
        values=values,
    )


def ttm_row(
    year: int = 2025,
    *,
    accepted: datetime | None = None,
    revenue: float = 4_000.0,
    eps: float = 8.0,
    net_income: float = 1_000.0,
    ocf: float = 1_400.0,
) -> ProviderRow:
    """A TTM statement — revenue 4000, EPS 8.00, net income 1000."""
    return ProviderRow(
        fiscal_year=year,
        fiscal_period="TTM",
        end_date=date(year, 9, 30),
        acceptance_datetime=accepted or datetime(year, 10, 31, 20, 30, tzinfo=UTC),
        timeframe="ttm",
        values={REV: revenue, EPS: eps, NI: net_income, OCF: ocf},
    )


AS_OF = datetime(2025, 11, 15, 12, 0, tzinfo=UTC)


def snap(rows, *, as_of=AS_OF, ticker="AAPL", price=None, market_cap=None):
    return build_snapshot(
        rows, as_of=as_of, ticker=ticker, price=price, market_cap=market_cap
    )


# ---------------------------------------------------------------------------
# 1. The as-of gate — acceptance_datetime, and nothing else (§7/§85/§96)
# ---------------------------------------------------------------------------


# The no-I/O and no-numerics invariants for THIS module are enforced
# package-wide by tests/test_pure_layer_boundary.py, which walks every
# module under libs/trading_core/ — a per-file copy here protected this
# one file and left sixty-six others to the habit of copying a test.


def test_statement_accepted_before_as_of_is_visible():
    row = full_quarter(accepted=datetime(2025, 10, 31, 20, 30, tzinfo=UTC))
    quarter, _ttm, rows = select_statements_as_of([row], AS_OF)
    assert quarter is not None
    assert quarter.fiscal_period == "Q3"
    assert len(rows) == 1


def test_statement_accepted_one_hour_after_as_of_is_excluded():
    """The audit §7.4 sentinel: a filing one hour past as_of must not leak."""
    as_of = datetime(2025, 10, 31, 20, 0, tzinfo=UTC)
    row = full_quarter(accepted=as_of + timedelta(hours=1))
    quarter, ttm, rows = select_statements_as_of([row], as_of)
    assert quarter is None
    assert ttm is None
    assert rows == []


def test_statement_accepted_exactly_at_as_of_is_visible():
    """``<=``, not ``<`` — the instant a filing is accepted, it is public."""
    as_of = datetime(2025, 10, 31, 20, 30, tzinfo=UTC)
    quarter, _ttm, _rows = select_statements_as_of(
        [full_quarter(accepted=as_of)], as_of
    )
    assert quarter is not None


def test_period_end_before_as_of_does_not_make_a_late_filing_visible():
    """The whole §85 point: a September quarter was NOT public in October 1."""
    row = full_quarter(accepted=datetime(2025, 10, 31, 20, 30, tzinfo=UTC))
    assert row.end_date == date(2025, 9, 30)
    as_of = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)
    quarter, _ttm, _rows = select_statements_as_of([row], as_of)
    assert quarter is None, "filtered on end_date instead of acceptance"


def test_row_without_acceptance_datetime_is_excluded_with_a_reason():
    row = ProviderRow(
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        acceptance_datetime=None,
        values={REV: 1_000.0},
    )
    quarter, _ttm, rows = select_statements_as_of([row], AS_OF)
    assert quarter is None and rows == []
    result = snap([row])
    assert result.available is False
    assert "rows_without_acceptance_datetime" in result.reasons


def test_naive_acceptance_datetime_is_refused_not_assumed_utc():
    """Guessing the zone shifts the boundary by hours — exactly the leak."""
    row = ProviderRow(
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        acceptance_datetime=datetime(2025, 10, 31, 20, 30),
        values={REV: 1_000.0},
    )
    assert coerce_statement(row).acceptance_datetime is None
    quarter, _ttm, _rows = select_statements_as_of([row], AS_OF)
    assert quarter is None


def test_naive_as_of_is_refused():
    with pytest.raises(ValueError, match="as_of"):
        select_statements_as_of([full_quarter()], datetime(2025, 11, 15, 12, 0))


def test_non_utc_as_of_is_normalised_not_rejected():
    """Any aware zone is fine — it is converted, like ``require_utc`` does."""
    eastern = timezone(timedelta(hours=-5))
    as_of = datetime(2025, 10, 31, 16, 0, tzinfo=eastern)  # == 21:00Z
    quarter, _ttm, _rows = select_statements_as_of(
        [full_quarter(accepted=datetime(2025, 10, 31, 20, 30, tzinfo=UTC))], as_of
    )
    assert quarter is not None


def test_quarterly_rows_come_back_newest_first():
    rows = [
        full_quarter(2024, "Q3", accepted=datetime(2024, 10, 31, tzinfo=UTC)),
        full_quarter(2025, "Q3", accepted=datetime(2025, 10, 31, tzinfo=UTC)),
        full_quarter(2025, "Q2", accepted=datetime(2025, 7, 31, tzinfo=UTC)),
    ]
    quarter, _ttm, ordered = select_statements_as_of(rows, AS_OF)
    assert quarter.fiscal_year == 2025 and quarter.fiscal_period == "Q3"
    assert [(s.fiscal_year, s.fiscal_period) for s in ordered] == [
        (2025, "Q3"),
        (2025, "Q2"),
        (2024, "Q3"),
    ]


def test_ttm_and_quarterly_are_split_by_timeframe():
    rows = [full_quarter(), ttm_row()]
    quarter, ttm, quarterly = select_statements_as_of(rows, AS_OF)
    assert quarter is not None and quarter.fiscal_period == "Q3"
    assert ttm is not None and ttm.fiscal_period == "TTM"
    assert len(quarterly) == 1, "TTM row leaked into the quarterly list"


def test_annual_rows_are_not_treated_as_quarterly():
    annual = ProviderRow(
        fiscal_year=2025,
        fiscal_period="FY",
        end_date=date(2025, 9, 30),
        acceptance_datetime=datetime(2025, 10, 31, tzinfo=UTC),
        timeframe="annual",
        values={REV: 4_000.0},
    )
    quarter, ttm, quarterly = select_statements_as_of([annual], AS_OF)
    assert (quarter, ttm, quarterly) == (None, None, [])


def test_future_rows_are_counted_in_the_snapshot_reasons():
    visible = full_quarter(accepted=datetime(2025, 10, 31, tzinfo=UTC))
    future = full_quarter(2026, "Q1", accepted=datetime(2026, 1, 31, tzinfo=UTC))
    result = snap([visible, future])
    assert result.available is True
    assert "rows_accepted_after_as_of" in result.reasons


# ---------------------------------------------------------------------------
# 2. Row coercion — the duck-typed seam
# ---------------------------------------------------------------------------


def test_plain_mapping_row_is_accepted():
    stmt = coerce_statement(
        {
            "fiscal_year": 2025,
            "fiscal_period": "q3",
            "end_date": "2025-09-27",
            "acceptance_datetime": "2025-10-31T20:31:00Z",
            "timeframe": "QUARTERLY",
            "values": {REV: 1_000.0},
        }
    )
    assert stmt.fiscal_period == "Q3"
    assert stmt.timeframe == "quarterly"
    assert stmt.end_date == date(2025, 9, 27)
    assert stmt.acceptance_datetime == datetime(2025, 10, 31, 20, 31, tzinfo=UTC)
    assert stmt.get(REV) == 1_000.0


def test_provider_dataclass_satisfies_the_structural_protocol():
    assert isinstance(full_quarter(), StatementLike)


def test_non_numeric_values_are_dropped_not_coerced():
    stmt = coerce_statement(
        {
            "fiscal_year": 2025,
            "fiscal_period": "Q3",
            "acceptance_datetime": datetime(2025, 10, 31, tzinfo=UTC),
            "values": {REV: "n/a", GP: None, OPINC: True, NI: 250.0},
        }
    )
    assert stmt.get(REV) is None
    assert stmt.get(GP) is None
    assert stmt.get(OPINC) is None, "a boolean was read as a number"
    assert stmt.get(NI) == 250.0


def test_nan_and_inf_values_never_survive_coercion():
    stmt = coerce_statement(
        {
            "fiscal_year": 2025,
            "fiscal_period": "Q3",
            "acceptance_datetime": datetime(2025, 10, 31, tzinfo=UTC),
            "values": {REV: float("nan"), GP: float("inf"), NI: float("-inf")},
        }
    )
    assert stmt.get(REV) is None and stmt.get(GP) is None and stmt.get(NI) is None


def test_coerce_statement_is_idempotent_on_a_statement():
    stmt = coerce_statement(full_quarter())
    assert coerce_statement(stmt) is stmt


def test_statement_label_degrades_without_a_fiscal_year():
    assert Statement(fiscal_period="Q3", fiscal_year=2025).label == "FY2025 Q3"
    assert Statement(fiscal_period="Q3").label == "Q3"


# ---------------------------------------------------------------------------
# 3. §28 — margins, growth, ratios; and the refusal to fabricate
# ---------------------------------------------------------------------------


def test_margins_are_computed_from_the_quarterly_statement():
    result = snap([full_quarter()])
    assert result.available is True
    assert result.metrics["revenue"] == 1_000.0
    assert result.metrics["gross_margin"] == pytest.approx(0.46)
    assert result.metrics["operating_margin"] == pytest.approx(0.30)
    assert result.metrics["net_margin"] == pytest.approx(0.25)


def test_zero_revenue_makes_margins_none_with_a_reason_never_zero():
    result = snap([full_quarter(revenue=0.0)])
    for name in ("gross_margin", "operating_margin", "net_margin"):
        assert result.metrics[name] is None
        assert result.reasons[name] == "revenue_not_positive"


def test_negative_revenue_makes_margins_none_with_a_reason():
    result = snap([full_quarter(revenue=-50.0)])
    assert result.metrics["gross_margin"] is None
    assert result.reasons["gross_margin"] == "revenue_not_positive"


def test_missing_gross_profit_is_none_with_the_field_named():
    row = full_quarter()
    values = dict(row.values)
    values.pop(GP)
    result = snap([ProviderRow(**{**row.__dict__, "values": values})])
    assert result.metrics["gross_margin"] is None
    assert result.reasons["gross_margin"] == f"input_unavailable:{GP}"
    assert result.metrics["operating_margin"] == pytest.approx(0.30)


def test_missing_revenue_names_revenue_not_the_derived_metric():
    row = full_quarter()
    values = {k: v for k, v in row.values.items() if k != REV}
    result = snap([ProviderRow(**{**row.__dict__, "values": values})])
    assert result.metrics["revenue"] is None
    assert result.reasons["revenue"] == f"input_unavailable:{REV}"
    assert result.reasons["gross_margin"] == f"input_unavailable:{REV}"


def test_current_ratio_and_debt_to_equity_are_hand_checkable():
    result = snap([full_quarter()])
    assert result.metrics["current_ratio"] == pytest.approx(2.0)  # 2000 / 1000
    assert result.metrics["debt_to_equity"] == pytest.approx(0.5)  # 1000 / 2000


def test_zero_current_liabilities_refuses_the_current_ratio():
    result = snap([full_quarter(**{CL: 0.0})])
    assert result.metrics["current_ratio"] is None
    assert result.reasons["current_ratio"] == f"input_unavailable:{CL}"
    assert not math.isinf(result.metrics["current_ratio"] or 0.0)


def test_zero_equity_refuses_debt_to_equity_and_roe():
    result = snap([full_quarter(**{EQ: 0.0}), ttm_row()])
    assert result.metrics["debt_to_equity"] is None
    assert result.reasons["debt_to_equity"] == f"input_unavailable:{EQ}"
    assert result.metrics["roe_ttm"] is None


def test_parent_equity_is_preferred_over_total_equity():
    result = snap([full_quarter(**{EQ_PARENT: 1_000.0})])
    # D/E = long-term debt 1000 / parent equity 1000 = 1.0, not 0.5.
    assert result.metrics["debt_to_equity"] == pytest.approx(1.0)


def test_roe_and_roa_use_ttm_net_income_over_quarter_balance_sheet():
    result = snap([full_quarter(), ttm_row()])
    assert result.metrics["roe_ttm"] == pytest.approx(0.5)  # 1000 / 2000
    assert result.metrics["roa_ttm"] == pytest.approx(0.2)  # 1000 / 5000


def test_roe_without_a_ttm_statement_is_none_with_a_reason():
    result = snap([full_quarter()])
    assert result.metrics["roe_ttm"] is None
    assert "TTM" in result.reasons["roe_ttm"]


def test_parent_net_income_is_preferred_for_ttm_returns():
    row = ttm_row()
    row = ProviderRow(
        **{**row.__dict__, "values": {**row.values, NI_PARENT: 800.0}}
    )
    result = snap([full_quarter(), row])
    assert result.metrics["roe_ttm"] == pytest.approx(0.4)  # 800 / 2000


def test_ttm_flows_come_from_the_ttm_statement():
    result = snap([full_quarter(), ttm_row()])
    assert result.metrics["revenue_ttm"] == 4_000.0
    assert result.metrics["eps_diluted_ttm"] == 8.0
    assert result.metrics["operating_cash_flow_ttm"] == 1_400.0
    assert result.metrics["operating_cash_flow"] == 350.0  # the quarter's own


def test_total_debt_carries_the_long_term_only_note():
    result = snap([full_quarter()])
    assert result.metrics["total_debt"] == 1_000.0
    assert result.notes["total_debt"] == METRIC_NOTES["total_debt"]
    assert "long-term only" in result.notes["total_debt"]


@pytest.mark.parametrize(
    "metric",
    ["free_cash_flow", "capex", "cash", "net_debt", "roic", "quick_ratio", "debt_to_ebitda"],
)
def test_provider_gaps_are_permanently_unavailable_with_their_reason(metric):
    """§28's "do not compute" for fields the feed simply does not carry."""
    result = snap([full_quarter(), ttm_row()])
    assert result.metrics[metric] is None
    assert result.reasons[metric] == NOT_REPORTED_REASONS[metric]


def test_capex_reason_is_named_in_the_free_cash_flow_reason():
    result = snap([full_quarter()])
    assert "capex" in result.reasons["free_cash_flow"]
    assert "cash" in result.reasons["net_debt"]


def test_every_metric_in_the_canonical_order_is_present():
    result = snap([full_quarter(), ttm_row()], price=100.0)
    assert set(result.metrics) >= set(METRIC_ORDER)


def test_every_none_metric_has_a_reason_and_no_value_is_nan_or_inf():
    result = snap([full_quarter(), ttm_row()], price=100.0)
    for name in METRIC_ORDER:
        value = result.metrics[name]
        if value is None:
            assert result.reasons.get(name), f"{name} is None with no reason"
        else:
            assert math.isfinite(value), f"{name} is not finite"


def test_no_statements_at_all_yields_an_unavailable_snapshot():
    result = snap([])
    assert result.available is False
    assert result.quarterly is None and result.ttm is None
    assert result.quarters_available == 0
    assert all(result.metrics[name] is None for name in METRIC_ORDER)
    assert all(result.reasons.get(name) for name in METRIC_ORDER)


def test_unavailable_snapshot_still_names_the_provider_gaps_specifically():
    result = snap([])
    assert result.reasons["capex"] == NOT_REPORTED_REASONS["capex"]


def test_snapshot_records_the_statements_it_used():
    result = snap([full_quarter(), ttm_row()])
    assert isinstance(result.quarterly, StatementRef)
    assert result.quarterly.label == "FY2025 Q3"
    assert result.quarterly.end_date == date(2025, 9, 30)
    assert result.quarterly.acceptance_datetime == datetime(
        2025, 10, 31, 20, 30, tzinfo=UTC
    )
    assert result.ttm.fiscal_period == "TTM"
    assert result.model_version == FUNDAMENTALS_MODEL_VERSION


def test_snapshot_unavailable_property_lists_missing_metrics_in_order():
    result = snap([full_quarter(), ttm_row()])
    missing = result.unavailable
    assert "capex" in missing and "free_cash_flow" in missing
    assert list(missing) == [m for m in METRIC_ORDER if m in set(missing)]


def test_snapshot_helper_accessors():
    result = snap([full_quarter()])
    assert result.value("gross_margin") == pytest.approx(0.46)
    assert result.reason("gross_margin") is None
    assert result.value("capex") is None
    assert result.reason("capex") == NOT_REPORTED_REASONS["capex"]


# ---------------------------------------------------------------------------
# 4. Year-over-year — the SAME fiscal quarter, one year back
# ---------------------------------------------------------------------------


def test_yoy_matches_the_same_fiscal_quarter_one_year_earlier():
    prior = full_quarter(2024, "Q3", accepted=datetime(2024, 10, 31, tzinfo=UTC))
    prior = ProviderRow(
        **{**prior.__dict__, "values": {**prior.values, REV: 800.0, EPS: 1.60}}
    )
    result = snap([full_quarter(2025, "Q3"), prior])
    assert result.metrics["revenue_growth_yoy"] == pytest.approx(0.25)  # 1000/800
    assert result.metrics["eps_growth_yoy"] == pytest.approx(0.25)  # 2.00/1.60


def test_yoy_refuses_a_different_quarter_even_when_four_rows_back():
    """Q3 vs Q2 is seasonality, not growth — the match is on the label."""
    rows = [full_quarter(2025, "Q3")]
    for i, period in enumerate(["Q2", "Q1"]):
        rows.append(
            full_quarter(
                2025, period, accepted=datetime(2025, 4 + 3 * i, 30, tzinfo=UTC)
            )
        )
    rows.append(
        full_quarter(2024, "Q2", accepted=datetime(2024, 7, 31, tzinfo=UTC))
    )
    result = snap(rows)
    assert result.metrics["revenue_growth_yoy"] is None
    assert "prior_year_same_quarter" in result.reasons["revenue_growth_yoy"]


def test_yoy_reason_reports_how_many_quarters_were_public():
    result = snap([full_quarter(2025, "Q3")])
    assert "1 quarterly statement(s) public at as_of" in (
        result.reasons["revenue_growth_yoy"]
    )


def test_yoy_prior_year_accepted_after_as_of_is_not_used():
    """A backfilled prior-year row filed late must not resurrect the YoY."""
    prior = full_quarter(
        2024, "Q3", accepted=datetime(2025, 12, 1, tzinfo=UTC)
    )
    result = snap([full_quarter(2025, "Q3"), prior])
    assert result.metrics["revenue_growth_yoy"] is None


def test_yoy_from_a_loss_flips_sign_correctly():
    """EPS -0.50 -> +0.25 is +150%, not -150%: the base is |earlier|."""
    prior = full_quarter(2024, "Q3", accepted=datetime(2024, 10, 31, tzinfo=UTC))
    prior = ProviderRow(**{**prior.__dict__, "values": {**prior.values, EPS: -0.50}})
    current = full_quarter(2025, "Q3", **{EPS: 0.25})
    result = snap([current, prior])
    assert result.metrics["eps_growth_yoy"] == pytest.approx(1.5)


def test_yoy_with_a_zero_base_is_none_with_a_reason():
    prior = full_quarter(2024, "Q3", accepted=datetime(2024, 10, 31, tzinfo=UTC))
    prior = ProviderRow(**{**prior.__dict__, "values": {**prior.values, REV: 0.0}})
    result = snap([full_quarter(2025, "Q3"), prior])
    assert result.metrics["revenue_growth_yoy"] is None
    assert result.reasons["revenue_growth_yoy"] == f"input_unavailable:{REV}"


def test_yoy_needs_five_quarters_and_gets_them():
    """The contract's ">= 5 quarterly rows" case, end to end."""
    rows = []
    for i, (year, period) in enumerate(
        [(2025, "Q3"), (2025, "Q2"), (2025, "Q1"), (2024, "Q4"), (2024, "Q3")]
    ):
        rows.append(
            full_quarter(
                year,
                period,
                accepted=datetime(2025, 10, 31, tzinfo=UTC) - timedelta(days=91 * i),
                revenue=1_000.0 - 50.0 * i,
            )
        )
    result = snap(rows)
    assert result.quarters_available == 5
    # current 1000 vs the 2024 Q3 row at 1000 - 200 = 800.
    assert result.metrics["revenue_growth_yoy"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 5. §30 — valuation multiples need a price
# ---------------------------------------------------------------------------


def test_multiples_are_none_without_a_price_and_say_so():
    result = snap([full_quarter(), ttm_row()])
    for name in VALUATION_METRICS + ("earnings_yield",):
        assert result.metrics[name] is None
        assert result.reasons[name] == "input_unavailable:price"


def test_market_cap_is_derived_from_price_and_diluted_shares():
    result = snap([full_quarter(), ttm_row()], price=100.0)
    assert result.market_cap == pytest.approx(10_000.0)  # 100 * 100 shares
    assert "derived" in result.notes["market_cap"]


def test_pe_ps_pb_and_earnings_yield_are_hand_checkable():
    result = snap([full_quarter(), ttm_row()], price=100.0)
    assert result.metrics["pe_ttm"] == pytest.approx(12.5)  # 100 / 8.00
    assert result.metrics["ps_ttm"] == pytest.approx(2.5)  # 10000 / 4000
    assert result.metrics["pb"] == pytest.approx(5.0)  # 10000 / 2000
    assert result.metrics["earnings_yield"] == pytest.approx(0.08)  # 8 / 100


def test_explicit_market_cap_overrides_the_derivation():
    result = snap([full_quarter(), ttm_row()], price=100.0, market_cap=20_000.0)
    assert result.market_cap == 20_000.0
    assert result.metrics["ps_ttm"] == pytest.approx(5.0)
    assert "market_cap" not in result.notes


def test_negative_ttm_eps_refuses_a_pe_rather_than_reporting_a_negative_one():
    result = snap([full_quarter(), ttm_row(eps=-2.0)], price=100.0)
    assert result.metrics["pe_ttm"] is None
    assert "negative" in result.reasons["pe_ttm"]
    # The earnings yield IS meaningful for a loss and stays.
    assert result.metrics["earnings_yield"] == pytest.approx(-0.02)


def test_price_without_shares_leaves_ps_and_pb_unavailable():
    row = full_quarter()
    values = {k: v for k, v in row.values.items() if k != SHARES}
    result = snap(
        [ProviderRow(**{**row.__dict__, "values": values}), ttm_row()], price=100.0
    )
    assert result.metrics["pe_ttm"] == pytest.approx(12.5), "P/E needs no shares"
    assert result.metrics["ps_ttm"] is None
    assert "market cap not derivable" in result.reasons["ps_ttm"]


def test_a_zero_price_is_refused_rather_than_producing_a_zero_multiple():
    """A zero quote is a missing price, not a company worth nothing."""
    result = snap([full_quarter(), ttm_row()], price=0.0)
    assert result.price is None
    assert result.market_cap is None
    for name in VALUATION_METRICS + ("earnings_yield",):
        assert result.metrics[name] is None, f"{name} was fabricated from a zero price"
        assert result.reasons[name] == "input_unavailable:price"


def test_a_negative_price_is_refused_the_same_way():
    result = snap([full_quarter(), ttm_row()], price=-100.0)
    assert result.price is None
    assert result.metrics["pe_ttm"] is None


def test_a_zero_market_cap_is_refused_and_not_re_derived_as_zero():
    result = snap([full_quarter(), ttm_row()], price=100.0, market_cap=0.0)
    # The explicit zero is dropped, then derived honestly from price x shares.
    assert result.market_cap == pytest.approx(10_000.0)
    assert result.metrics["ps_ttm"] == pytest.approx(2.5)


def test_zero_diluted_shares_leaves_the_market_cap_underivable():
    result = snap([full_quarter(**{SHARES: 0.0}), ttm_row()], price=100.0)
    assert result.market_cap is None
    assert result.metrics["ps_ttm"] is None
    assert "market cap not derivable" in result.reasons["ps_ttm"]


def test_ev_ebitda_and_fcf_yield_are_always_unavailable_with_a_reason():
    result = snap([full_quarter(), ttm_row()], price=100.0)
    for name in ("ev_ebitda", "fcf_yield"):
        assert result.metrics[name] is None
        assert result.reasons[name] == NOT_REPORTED_REASONS[name]


# ---------------------------------------------------------------------------
# 6. §29 — the change table
# ---------------------------------------------------------------------------


def _two_snapshots(*, previous_gm: float, current_gm: float):
    """Two snapshots whose gross margins are exactly the given fractions."""
    prev = snap([full_quarter(2025, "Q2", **{GP: previous_gm * 1_000.0})])
    curr = snap([full_quarter(2025, "Q3", **{GP: current_gm * 1_000.0})])
    return prev, curr


def _by_metric(changes) -> dict[str, MetricChange]:
    return {c.metric: c for c in changes}


def test_change_reports_delta_and_bps_for_a_margin():
    """§29's own example: 72.4% -> 73.1% is +70 bps."""
    prev, curr = _two_snapshots(previous_gm=0.724, current_gm=0.731)
    change = _by_metric(snapshot_change(prev, curr))["gross_margin"]
    assert change.previous == pytest.approx(0.724)
    assert change.current == pytest.approx(0.731)
    assert change.delta == pytest.approx(0.007)
    assert change.delta_bps == pytest.approx(70.0)
    assert change.direction == "up"
    assert change.arrow == "↑"


def test_change_bps_is_absent_for_a_dollar_metric():
    prev = snap([full_quarter(2025, "Q2", revenue=900.0)])
    curr = snap([full_quarter(2025, "Q3", revenue=1_000.0)])
    change = _by_metric(snapshot_change(prev, curr))["revenue"]
    assert change.delta == pytest.approx(100.0)
    assert change.delta_bps is None, "bps on a dollar figure is meaningless"
    assert change.pct_change == pytest.approx(0.1111111, rel=1e-4)


def test_bps_is_emitted_for_exactly_the_ratio_metrics():
    prev, curr = _two_snapshots(previous_gm=0.40, current_gm=0.46)
    for change in snapshot_change(prev, curr):
        if change.delta is None:
            continue
        assert (change.delta_bps is not None) == (change.metric in BPS_METRICS)


def test_change_direction_down_and_flat():
    prev, curr = _two_snapshots(previous_gm=0.50, current_gm=0.40)
    down = _by_metric(snapshot_change(prev, curr))["gross_margin"]
    assert down.direction == "down" and down.arrow == "↓"
    assert down.delta_bps == pytest.approx(-1_000.0)

    prev, curr = _two_snapshots(previous_gm=0.46, current_gm=0.46)
    flat = _by_metric(snapshot_change(prev, curr))["gross_margin"]
    assert flat.direction == "flat" and flat.arrow == "→"
    assert flat.delta == pytest.approx(0.0)


def test_change_with_no_previous_snapshot_still_lists_every_metric():
    curr = snap([full_quarter(), ttm_row()])
    changes = snapshot_change(None, curr)
    assert [c.metric for c in changes] == list(METRIC_ORDER)
    gm = _by_metric(changes)["gross_margin"]
    assert gm.current == pytest.approx(0.46)
    assert gm.previous is None and gm.delta is None
    assert gm.reason == "no_previous_snapshot"


def test_change_reason_names_which_side_was_missing():
    prev = snap([full_quarter(2025, "Q2", revenue=0.0)])
    curr = snap([full_quarter(2025, "Q3")])
    change = _by_metric(snapshot_change(prev, curr))["gross_margin"]
    assert change.delta is None
    assert change.reason.startswith("previous unavailable: ")
    assert "revenue_not_positive" in change.reason

    change = _by_metric(snapshot_change(curr, prev))["gross_margin"]
    assert change.reason.startswith("current unavailable: ")


def test_change_for_a_metric_missing_on_both_sides_carries_the_gap_reason():
    prev = snap([full_quarter(2025, "Q2")])
    curr = snap([full_quarter(2025, "Q3")])
    change = _by_metric(snapshot_change(prev, curr))["capex"]
    assert change.previous is None and change.current is None
    assert change.reason == NOT_REPORTED_REASONS["capex"]


def test_change_never_reports_a_zero_delta_for_a_missing_metric():
    prev = snap([full_quarter(2025, "Q2")])
    curr = snap([full_quarter(2025, "Q3")])
    for change in snapshot_change(prev, curr):
        if change.previous is None or change.current is None:
            assert change.delta is None, f"{change.metric} fabricated a delta"
            assert change.reason


def test_change_carries_the_long_term_debt_note_forward():
    prev = snap([full_quarter(2025, "Q2")])
    curr = snap([full_quarter(2025, "Q3")])
    assert _by_metric(snapshot_change(prev, curr))["total_debt"].note == (
        METRIC_NOTES["total_debt"]
    )


def test_change_can_be_restricted_to_a_metric_subset():
    prev, curr = _two_snapshots(previous_gm=0.40, current_gm=0.46)
    changes = snapshot_change(prev, curr, metrics=["gross_margin", "revenue"])
    assert [c.metric for c in changes] == ["gross_margin", "revenue"]


def test_change_rejects_a_non_positive_trend_window():
    prev, curr = _two_snapshots(previous_gm=0.40, current_gm=0.46)
    with pytest.raises(ValueError, match="trend_window"):
        snapshot_change(prev, curr, trend_window=0)


# ---------------------------------------------------------------------------
# 7. Trend classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "series,expected",
    [
        ([0.40, 0.42, 0.44, 0.46], "improving"),
        ([0.46, 0.44, 0.42, 0.40], "deteriorating"),
        ([0.46, 0.4601, 0.4599, 0.46], "flat"),
        ([0.46], "flat"),
        ([], "flat"),
        ([0.0, 0.0], "flat"),
    ],
)
def test_trend_classification(series, expected):
    label, n = _classify_trend(series)
    assert label == expected
    assert n == len(series)


def test_trend_tolerance_ignores_a_three_bps_drift():
    """0.4600 -> 0.4603 is 0.065% — noise, not a direction."""
    assert _classify_trend([0.4600, 0.4603])[0] == "flat"


def test_trend_uses_the_history_and_ends_at_the_current_snapshot():
    history = [
        snap([full_quarter(2024, "Q3", **{GP: gm * 1_000.0})])
        for gm in (0.40, 0.42, 0.44)
    ]
    prev = snap([full_quarter(2025, "Q2", **{GP: 0.44 * 1_000.0})])
    curr = snap([full_quarter(2025, "Q3", **{GP: 0.50 * 1_000.0})])
    change = _by_metric(snapshot_change(prev, curr, history=history))["gross_margin"]
    assert change.trend == "improving"
    assert change.trend_points == 4  # three history points plus current


def test_trend_is_none_with_a_single_point():
    curr = snap([full_quarter()])
    change = _by_metric(snapshot_change(None, curr))["gross_margin"]
    assert change.trend is None
    assert change.trend_points == 1


def test_trend_window_takes_only_the_trailing_points():
    """A long improving run that recently turned down reads as the turn."""
    history = [
        snap([full_quarter(2024, "Q1", **{GP: gm * 1_000.0})])
        for gm in (0.20, 0.30, 0.40, 0.50, 0.60, 0.55, 0.50, 0.45, 0.40)
    ]
    curr = snap([full_quarter(2025, "Q3", **{GP: 0.35 * 1_000.0})])
    change = _by_metric(
        snapshot_change(None, curr, history=history, trend_window=4)
    )["gross_margin"]
    assert change.trend == "deteriorating"
    assert change.trend_points == 4
    assert DEFAULT_TREND_WINDOW == 8


# ---------------------------------------------------------------------------
# 8. §30 — valuation context
# ---------------------------------------------------------------------------


def test_percentile_of_is_the_empirical_rank():
    assert percentile_of([10.0, 20.0, 30.0, 40.0], 30.0) == pytest.approx(75.0)
    assert percentile_of([10.0, 20.0], 5.0) == pytest.approx(0.0)
    assert percentile_of([], 5.0) is None


def test_valuation_context_reports_own_history_stats():
    history = [
        snap([full_quarter(2025, "Q2"), ttm_row()], price=price)
        for price in (60.0, 80.0, 120.0)  # P/E 7.5, 10.0, 15.0
    ]
    current = snap([full_quarter(), ttm_row()], price=100.0)  # P/E 12.5
    block = valuation_context(current, history)["multiples"]["pe_ttm"]
    assert block["available"] is True
    assert block["current"] == pytest.approx(12.5)
    assert block["median"] == pytest.approx(10.0)
    assert block["min"] == pytest.approx(7.5)
    assert block["max"] == pytest.approx(15.0)
    assert block["history_n"] == 3
    assert block["percentile"] == pytest.approx(200.0 / 3.0)  # 2 of 3 at/below


def test_valuation_context_without_a_price_marks_each_multiple_unavailable():
    current = snap([full_quarter(), ttm_row()])
    context = valuation_context(current, [])
    for name in VALUATION_METRICS:
        block = context["multiples"][name]
        assert block["available"] is False
        assert block["reason"] == "input_unavailable:price"
    assert context["own_history"]["available"] is False
    assert context["own_history"]["reason"]


def test_valuation_context_keeps_the_current_multiple_without_history():
    current = snap([full_quarter(), ttm_row()], price=100.0)
    block = valuation_context(current, [])["multiples"]["pe_ttm"]
    assert block["available"] is True
    assert block["current"] == pytest.approx(12.5)
    assert block["history_n"] == 0
    assert block["median"] is None and block["percentile"] is None
    assert "history_reason" in block


def test_valuation_context_declares_sector_and_peers_unavailable():
    context = valuation_context(snap([full_quarter()], price=100.0), [])
    assert context["sector"] == {"available": False, "reason": PEER_CONTEXT_REASON}
    assert context["peers"] == {"available": False, "reason": PEER_CONTEXT_REASON}
    assert "Phase G/J" in PEER_CONTEXT_REASON


def test_valuation_context_marks_ev_ebitda_and_fcf_yield_unavailable():
    context = valuation_context(snap([full_quarter(), ttm_row()], price=100.0), [])
    for name in ("ev_ebitda", "fcf_yield"):
        block = context["multiples"][name]
        assert block["available"] is False and block["current"] is None
        assert block["reason"] == NOT_REPORTED_REASONS[name]


def test_valuation_context_is_labelled_quant_and_versioned():
    context = valuation_context(snap([full_quarter()], price=100.0), [])
    assert context["provenance"] == "QUANT"
    assert context["model_version"] == FUNDAMENTALS_MODEL_VERSION
    assert context["price"] == 100.0


# ---------------------------------------------------------------------------
# 9. §35 — the deterministic expectations-gap inputs
# ---------------------------------------------------------------------------


def test_expectations_gap_labels_a_broad_improvement():
    """Revenue, every margin, EPS and cash flow all up — one direction."""
    prev = snap(
        [
            full_quarter(
                2025,
                "Q2",
                revenue=800.0,
                **{GP: 300.0, OPINC: 200.0, NI: 150.0, EPS: 1.2, OCF: 250.0},
            ),
            ttm_row(revenue=3_000.0, eps=6.0, net_income=800.0, ocf=1_000.0),
        ]
    )
    curr = snap([full_quarter(2025, "Q3", revenue=1_000.0), ttm_row()])
    result = expectations_gap_inputs(snapshot_change(prev, curr))
    assert result["label"] == "fundamentals_improving"
    assert result["improved"] >= 2
    assert result["improved"] > result["weakened"] * 2
    assert result["provenance"] == "QUANT"


def test_expectations_gap_labels_a_broad_weakening():
    """The same fixtures the other way round — every line down."""
    curr = snap(
        [
            full_quarter(
                2025,
                "Q3",
                revenue=800.0,
                **{GP: 300.0, OPINC: 200.0, NI: 150.0, EPS: 1.2, OCF: 250.0},
            ),
            ttm_row(revenue=3_000.0, eps=6.0, net_income=800.0, ocf=1_000.0),
        ]
    )
    prev = snap([full_quarter(2025, "Q2", revenue=1_000.0), ttm_row()])
    result = expectations_gap_inputs(snapshot_change(prev, curr))
    assert result["label"] == "fundamentals_weakening"
    assert result["weakened"] > result["improved"] * 2


def test_expectations_gap_labels_a_mixed_quarter():
    """Revenue up while every margin compresses — the honest "mixed"."""
    prev = snap([full_quarter(2025, "Q2", revenue=800.0, **{GP: 300.0})])
    curr = snap([full_quarter(2025, "Q3", revenue=1_000.0)])
    result = expectations_gap_inputs(snapshot_change(prev, curr))
    assert result["label"] == "fundamentals_mixed"
    assert result["improved"] and result["weakened"]


def test_expectations_gap_is_unknown_when_nothing_is_comparable():
    curr = snap([full_quarter()])
    result = expectations_gap_inputs(snapshot_change(None, curr))
    assert result["label"] == "fundamentals_unknown"
    assert result["compared"] == 0
    assert result["reason"]


def test_expectations_gap_ignores_valuation_and_leverage_metrics():
    prev = snap([full_quarter(2025, "Q2"), ttm_row()], price=50.0)
    curr = snap([full_quarter(2025, "Q3"), ttm_row()], price=100.0)
    result = expectations_gap_inputs(snapshot_change(prev, curr))
    considered = set(result["metrics_considered"])
    assert not considered & {"pe_ttm", "ps_ttm", "pb", "debt_to_equity", "total_debt"}


def test_expectations_gap_counts_unavailable_metrics_separately():
    prev = snap([full_quarter(2025, "Q2")])
    curr = snap([full_quarter(2025, "Q3")])
    result = expectations_gap_inputs(snapshot_change(prev, curr))
    # No TTM statement on either side, so the trailing metrics are unavailable.
    assert result["unavailable"] >= 3
    assert result["unavailable"] + result["compared"] == len(
        result["metrics_considered"]
    )


# ---------------------------------------------------------------------------
# 10. Purity & invariants
# ---------------------------------------------------------------------------



def test_snapshot_is_frozen_and_copies_its_mappings():
    metrics = {"revenue": 1.0}
    result = FundamentalSnapshot(
        ticker="AAPL", as_of=AS_OF, metrics=metrics, reasons={}
    )
    metrics["revenue"] = 999.0
    assert result.metrics["revenue"] == 1.0
    with pytest.raises(Exception):
        result.ticker = "MSFT"  # type: ignore[misc]


def test_build_snapshot_does_not_mutate_the_input_rows():
    rows = [full_quarter(), ttm_row()]
    before = [dict(row.values) for row in rows]
    snap(rows, price=100.0)
    assert [dict(row.values) for row in rows] == before


def test_build_snapshot_is_deterministic():
    rows = [full_quarter(), ttm_row()]
    first = snap(rows, price=100.0)
    second = snap(list(reversed(rows)), price=100.0)
    assert first.metrics == second.metrics
    assert first.reasons == second.reasons


def test_restatement_of_the_same_period_prefers_the_later_acceptance():
    original = full_quarter(
        2025, "Q3", accepted=datetime(2025, 10, 31, tzinfo=UTC), revenue=1_000.0
    )
    restated = full_quarter(
        2025, "Q3", accepted=datetime(2025, 12, 15, tzinfo=UTC), revenue=950.0
    )
    quarter, _ttm, _rows = select_statements_as_of([original, restated], AS_OF)
    assert quarter.get(REV) == 1_000.0, "restatement leaked before it was filed"

    later = datetime(2026, 1, 1, tzinfo=UTC)
    quarter, _ttm, _rows = select_statements_as_of([original, restated], later)
    assert quarter.get(REV) == 950.0


def test_a_late_dual_filing_prefers_the_later_period():
    """Two quarters accepted in one filing: the later PERIOD is the newer fact."""
    accepted = datetime(2025, 10, 31, tzinfo=UTC)
    q2 = ProviderRow(
        fiscal_year=2025,
        fiscal_period="Q2",
        end_date=date(2025, 6, 30),
        acceptance_datetime=accepted,
        values={REV: 900.0},
    )
    q3 = ProviderRow(
        fiscal_year=2025,
        fiscal_period="Q3",
        end_date=date(2025, 9, 30),
        acceptance_datetime=accepted,
        values={REV: 1_000.0},
    )
    quarter, _ttm, _rows = select_statements_as_of([q2, q3], AS_OF)
    assert quarter.fiscal_period == "Q3"


# ---------------------------------------------------------------------------
# Derived TTM (provider TTM rows carry no acceptance instant — live 2026-08-19)
# ---------------------------------------------------------------------------
from datetime import date as _date, datetime as _dt, timezone as _tz  # noqa: E402

from libs.trading_core.events.fundamentals import (  # noqa: E402
    DERIVED_TTM_PERIOD,
    Statement,
    derive_ttm_from_quarters,
    select_statements_as_of,
)


def _q(period: str, year: int, end: str, accepted: str, **vals: float) -> Statement:
    base = {
        "income_statement.revenues": 100.0,
        "income_statement.net_income_loss": 10.0,
        "income_statement.diluted_earnings_per_share": 1.0,
        "cash_flow_statement.net_cash_flow_from_operating_activities": 12.0,
        "balance_sheet.equity": 500.0,
        "income_statement.diluted_average_shares": 10.0,
    }
    base.update(vals)
    return Statement(
        fiscal_period=period,
        fiscal_year=year,
        end_date=_date.fromisoformat(end),
        acceptance_datetime=_dt.fromisoformat(accepted).replace(tzinfo=_tz.utc),
        timeframe="quarterly",
        values=base,
    )


def _four_quarters() -> list[Statement]:
    return [  # newest first
        _q("Q3", 2026, "2026-06-27", "2026-07-31T10:01:02", **{"income_statement.revenues": 109.0}),
        _q("Q2", 2026, "2026-03-28", "2026-05-01T10:00:00", **{"income_statement.revenues": 95.0}),
        _q("Q1", 2026, "2025-12-27", "2026-01-30T10:00:00", **{"income_statement.revenues": 124.0}),
        _q("Q4", 2025, "2025-09-27", "2025-10-31T10:00:00", **{"income_statement.revenues": 102.0}),
    ]


def test_derived_ttm_sums_flows_and_carries_the_newest_balance_sheet():
    ttm = derive_ttm_from_quarters(_four_quarters())
    assert ttm is not None and ttm.fiscal_period == DERIVED_TTM_PERIOD
    assert ttm.values["income_statement.revenues"] == 109.0 + 95.0 + 124.0 + 102.0
    assert ttm.values["income_statement.diluted_earnings_per_share"] == 4.0
    assert ttm.values["balance_sheet.equity"] == 500.0  # snapshot, not summed
    assert ttm.values["income_statement.diluted_average_shares"] == 10.0
    # visible exactly when the fourth (newest) quarter became visible
    assert ttm.acceptance_datetime == _dt(2026, 7, 31, 10, 1, 2, tzinfo=_tz.utc)
    assert ttm.end_date == _date(2026, 6, 27)


def test_derived_ttm_needs_all_four_quarters_for_a_flow_line():
    quarters = _four_quarters()
    quarters[2] = Statement(
        **{**quarters[2].__dict__, "values": {k: v for k, v in quarters[2].values.items() if "revenues" not in k}}
    )
    ttm = derive_ttm_from_quarters(quarters)
    assert ttm is not None
    assert "income_statement.revenues" not in ttm.values  # no partial total
    assert ttm.values["income_statement.net_income_loss"] == 40.0


def test_derived_ttm_requires_four_quarters():
    assert derive_ttm_from_quarters(_four_quarters()[:3]) is None


def test_select_statements_uses_derived_ttm_only_when_no_provider_ttm_is_visible():
    quarters = _four_quarters()
    as_of = _dt(2026, 8, 19, 17, 0, tzinfo=_tz.utc)
    _, ttm, _ = select_statements_as_of(quarters, as_of)
    assert ttm is not None and ttm.fiscal_period == DERIVED_TTM_PERIOD
    # a provider TTM row WITH an acceptance instant wins
    provider_ttm = Statement(
        fiscal_period="TTM", fiscal_year=None, end_date=_date(2026, 6, 27),
        acceptance_datetime=_dt(2026, 7, 31, 10, 1, 2, tzinfo=_tz.utc), timeframe="ttm",
        values={"income_statement.revenues": 999.0},
    )
    _, ttm2, _ = select_statements_as_of(quarters + [provider_ttm], as_of)
    assert ttm2 is not None and ttm2.fiscal_period == "TTM"
    # before the newest quarter was accepted, the derived TTM is not visible either
    _, ttm3, _ = select_statements_as_of(quarters, _dt(2026, 7, 31, 9, 0, tzinfo=_tz.utc))
    assert ttm3 is None or ttm3.acceptance_datetime <= _dt(2026, 7, 31, 9, 0, tzinfo=_tz.utc)
