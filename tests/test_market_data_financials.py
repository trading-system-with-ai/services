"""Fundamentals across the provider layer (Phase E2, audit §11.3, spec §28/§85).

The financials adapter is the source of the ONE point-in-time key the whole
as-of contract rests on (``acceptance_datetime``, audit §7.1), so what these
tests pin, in order of importance:

1. NO FABRICATION: a field the filer did not report is ABSENT from ``values``
   — never present as 0.0 — a non-numeric or null value is skipped rather than
   coerced, and an undatable row is dropped rather than guessed into place.
2. THE AS-OF KEY IS REAL: ``acceptance_datetime`` is parsed to an aware UTC
   instant (Z, offset, sub-microsecond and naive forms all land on UTC) and is
   honestly ``None`` when the provider omits it — never backfilled from
   ``filing_date``, which is a different fact.
3. CAPABILITY HONESTY (§16): Massive 403 is CapabilityNotAvailable; Alpaca,
   which sells no fundamentals at any tier, refuses explicitly instead of
   returning ``[]`` (which would read as "this company filed nothing"); both
   providers report the SAME probe key set.
4. The stub is deterministic and reports ONLY the fields the real provider
   reports, so a metric that is unavailable live is unavailable in tests too.
"""
import json
import math
from datetime import date, datetime, timezone

import httpx
import pytest

from libs.market_data.alpaca import AlpacaMarketDataProvider
from libs.market_data.massive import MassiveProvider
from libs.market_data.provider import (
    CapabilityNotAvailable,
    FinancialStatement,
    MarketDataError,
)
from libs.market_data.stub import StubProvider

FINANCIALS_PATH = "/vX/reference/financials"


def provider_with(handler, **kwargs) -> MassiveProvider:
    return MassiveProvider(
        api_key="test-key", transport=httpx.MockTransport(handler), **kwargs
    )


def alpaca_provider() -> AlpacaMarketDataProvider:
    return AlpacaMarketDataProvider(
        api_key_id="k",
        api_secret_key="s",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={})
        ),
    )


def _field(value, unit: str = "USD", label: str = "L", order: int = 100) -> dict:
    return {"value": value, "unit": unit, "label": label, "order": order}


def _row(**overrides) -> dict:
    """One financials result shaped exactly like the verified Massive sample."""
    row = {
        "start_date": "2024-03-31",
        "end_date": "2024-06-29",
        "filing_date": "2024-08-02",
        "acceptance_datetime": "2024-08-02T06:01:36Z",
        "timeframe": "quarterly",
        "fiscal_period": "Q3",
        "fiscal_year": 2024,
        "cik": "0000320193",
        "sic": "3571",
        "source_filing_url": "https://api.massive.com/v1/reference/sec/filings/x",
        "financials": {
            "income_statement": {
                "revenues": _field(85777000000.0),
                "gross_profit": _field(39678000000.0),
                "operating_income_loss": _field(25352000000.0),
                "net_income_loss": _field(21448000000.0),
                "diluted_earnings_per_share": _field(1.4),
                "diluted_average_shares": _field(15348175000.0),
            },
            "balance_sheet": {
                "assets": _field(331612000000.0),
                "equity": _field(66708000000.0),
                "long_term_debt": _field(86773000000.0),
            },
            "cash_flow_statement": {
                "net_cash_flow_from_operating_activities": _field(28858000000.0),
            },
        },
    }
    row.update(overrides)
    return row


def _body(*rows: dict) -> dict:
    return {"status": "OK", "count": len(rows), "results": list(rows)}


def _handler_for(payload: dict, status: int = 200, record: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == FINANCIALS_PATH, request.url.path
        if record is not None:
            record.append(dict(request.url.params))
        return httpx.Response(status, json=payload)

    return handler


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_carries_ticker_timeframe_limit_and_newest_first_ordering():
    seen: list[dict] = []
    provider_with(_handler_for(_body(), record=seen)).get_financials(
        "aapl", timeframe="quarterly", limit=8
    )
    assert seen == [
        {
            "ticker": "AAPL",  # upper-cased for the provider
            "timeframe": "quarterly",
            "limit": "8",
            "order": "desc",
            "sort": "filing_date",
        }
    ]


@pytest.mark.parametrize("timeframe", ["quarterly", "annual", "ttm"])
def test_every_supported_timeframe_reaches_the_provider(timeframe):
    seen: list[dict] = []
    provider_with(_handler_for(_body(), record=seen)).get_financials(
        "AAPL", timeframe=timeframe
    )
    assert seen[0]["timeframe"] == timeframe


def test_unsupported_timeframe_is_refused_before_any_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be sent for an unsupported timeframe")

    with pytest.raises(ValueError, match="unsupported financials timeframe"):
        provider_with(handler).get_financials("AAPL", timeframe="monthly")


def test_limit_is_clamped_to_the_providers_page_size():
    seen: list[dict] = []
    provider_with(_handler_for(_body(), record=seen)).get_financials("AAPL", limit=5000)
    assert seen[0]["limit"] == "100"


def test_blank_ticker_returns_empty_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a blank ticker must not reach the provider")

    assert provider_with(handler).get_financials("   ") == []


# ---------------------------------------------------------------------------
# Wire mapping
# ---------------------------------------------------------------------------


def test_row_maps_to_the_frozen_statement_verbatim():
    [statement] = provider_with(_handler_for(_body(_row()))).get_financials("AAPL")
    assert isinstance(statement, FinancialStatement)
    assert statement.ticker == "AAPL"
    assert statement.cik == "0000320193"
    assert statement.timeframe == "quarterly"
    assert statement.fiscal_year == 2024
    assert statement.fiscal_period == "Q3"
    assert statement.start_date == date(2024, 3, 31)
    assert statement.end_date == date(2024, 6, 29)
    assert statement.filing_date == date(2024, 8, 2)
    assert statement.source_filing_url.startswith("https://api.massive.com/")


def test_statement_is_frozen():
    [statement] = provider_with(_handler_for(_body(_row()))).get_financials("AAPL")
    with pytest.raises(Exception):
        statement.end_date = date(2030, 1, 1)  # type: ignore[misc]


def test_values_are_flattened_to_block_dot_field_floats():
    [statement] = provider_with(_handler_for(_body(_row()))).get_financials("AAPL")
    assert statement.values["income_statement.revenues"] == 85777000000.0
    assert statement.values["balance_sheet.long_term_debt"] == 86773000000.0
    assert (
        statement.values["cash_flow_statement.net_cash_flow_from_operating_activities"]
        == 28858000000.0
    )
    assert all(isinstance(v, float) for v in statement.values.values())
    assert all("." in key for key in statement.values)


def test_raw_fields_count_records_the_pre_filter_field_count():
    [statement] = provider_with(_handler_for(_body(_row()))).get_financials("AAPL")
    assert statement.raw_fields_count == 10
    assert len(statement.values) == 10


def test_fields_the_provider_does_not_report_are_absent_never_zero():
    """cash / capex / D&A are not in the payload — §44 rule 18 in one assert."""
    [statement] = provider_with(_handler_for(_body(_row()))).get_financials("AAPL")
    for missing in (
        "balance_sheet.cash",
        "cash_flow_statement.capital_expenditure",
        "income_statement.depreciation_and_amortization",
    ):
        assert missing not in statement.values
    assert 0.0 not in set(statement.values.values())


def test_unknown_statement_blocks_are_carried_through_not_dropped():
    row = _row()
    row["financials"]["comprehensive_income"] = {
        "comprehensive_income_loss": _field(21000000000.0)
    }
    [statement] = provider_with(_handler_for(_body(row))).get_financials("AAPL")
    assert (
        statement.values["comprehensive_income.comprehensive_income_loss"]
        == 21000000000.0
    )


def test_integer_values_become_floats():
    row = _row()
    row["financials"]["income_statement"]["revenues"] = _field(85777000000)
    [statement] = provider_with(_handler_for(_body(row))).get_financials("AAPL")
    value = statement.values["income_statement.revenues"]
    assert isinstance(value, float) and value == 85777000000.0


def test_negative_values_survive_a_loss_is_data_not_an_error():
    row = _row()
    row["financials"]["income_statement"]["net_income_loss"] = _field(-1234000.0)
    [statement] = provider_with(_handler_for(_body(row))).get_financials("AAPL")
    assert statement.values["income_statement.net_income_loss"] == -1234000.0


def test_empty_results_is_an_honest_absence_not_an_error():
    assert provider_with(_handler_for(_body())).get_financials("AAPL") == []


def test_missing_results_key_returns_empty():
    assert provider_with(_handler_for({"status": "OK"})).get_financials("AAPL") == []


# ---------------------------------------------------------------------------
# acceptance_datetime — the §85 point-in-time key
# ---------------------------------------------------------------------------


def test_acceptance_datetime_is_parsed_to_aware_utc():
    [statement] = provider_with(_handler_for(_body(_row()))).get_financials("AAPL")
    assert statement.acceptance_datetime == datetime(
        2024, 8, 2, 6, 1, 36, tzinfo=timezone.utc
    )
    assert statement.acceptance_datetime.tzinfo is not None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-08-02T06:01:36Z", datetime(2024, 8, 2, 6, 1, 36, tzinfo=timezone.utc)),
        (
            "2024-08-02T06:01:36.123456789Z",  # sub-microsecond digits are trimmed
            datetime(2024, 8, 2, 6, 1, 36, 123456, tzinfo=timezone.utc),
        ),
        (
            "2024-08-02T02:01:36-04:00",  # an offset is converted, never dropped
            datetime(2024, 8, 2, 6, 1, 36, tzinfo=timezone.utc),
        ),
        (
            "2024-08-02T06:01:36",  # naive is documented UTC, not local
            datetime(2024, 8, 2, 6, 1, 36, tzinfo=timezone.utc),
        ),
    ],
)
def test_acceptance_datetime_forms_all_land_on_utc(raw, expected):
    [statement] = provider_with(
        _handler_for(_body(_row(acceptance_datetime=raw)))
    ).get_financials("AAPL")
    assert statement.acceptance_datetime == expected


@pytest.mark.parametrize("bad", [None, "", "not-a-timestamp", 1722578496, {"x": 1}])
def test_unusable_acceptance_datetime_is_none_never_backfilled_from_filing_date(bad):
    """A row that cannot be placed in time says so; filing_date is a DIFFERENT fact."""
    [statement] = provider_with(
        _handler_for(_body(_row(acceptance_datetime=bad)))
    ).get_financials("AAPL")
    assert statement.acceptance_datetime is None
    assert statement.filing_date == date(2024, 8, 2)  # kept, but never substituted


def test_missing_filing_date_is_none():
    [statement] = provider_with(
        _handler_for(_body(_row(filing_date=None)))
    ).get_financials("AAPL")
    assert statement.filing_date is None
    assert statement.acceptance_datetime is not None


# ---------------------------------------------------------------------------
# Malformed rows: skipped, never patched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["start_date", "end_date"])
def test_row_without_period_bounds_is_skipped(field):
    """An undatable statement cannot be filtered as-of, so it is dropped."""
    rows = _body(_row(**{field: None}), _row(end_date="2024-03-30"))
    statements = provider_with(_handler_for(rows)).get_financials("AAPL")
    assert [s.end_date for s in statements] == [date(2024, 3, 30)]


def test_non_dict_rows_are_skipped():
    body = _body(_row())
    body["results"].extend(["garbage", None, 42])
    statements = provider_with(_handler_for(body)).get_financials("AAPL")
    assert len(statements) == 1


@pytest.mark.parametrize("value", [None, "85777000000", True, {"nested": 1}, [1, 2]])
def test_non_numeric_field_values_are_skipped_never_coerced(value):
    row = _row()
    row["financials"]["income_statement"]["revenues"] = _field(value)
    [statement] = provider_with(_handler_for(_body(row))).get_financials("AAPL")
    assert "income_statement.revenues" not in statement.values
    # counted as SEEN, so a mostly-unparseable row is detectable
    assert statement.raw_fields_count == 10
    assert len(statement.values) == 9


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_are_skipped(literal):
    """Python's json decoder accepts these literals; a metric never may."""
    body = json.dumps(_body(_row())).replace("85777000000.0", literal)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "application/json"}
        )

    [statement] = provider_with(handler).get_financials("AAPL")
    assert "income_statement.revenues" not in statement.values
    assert all(math.isfinite(v) for v in statement.values.values())


def test_field_without_the_value_wrapper_is_skipped():
    row = _row()
    row["financials"]["income_statement"]["revenues"] = 85777000000.0  # bare number
    [statement] = provider_with(_handler_for(_body(row))).get_financials("AAPL")
    assert "income_statement.revenues" not in statement.values


def test_missing_financials_object_yields_an_empty_but_dated_statement():
    [statement] = provider_with(
        _handler_for(_body(_row(financials=None)))
    ).get_financials("AAPL")
    assert statement.values == {}
    assert statement.raw_fields_count == 0
    assert statement.end_date == date(2024, 6, 29)  # still placeable in time


def test_non_dict_statement_block_is_skipped():
    row = _row()
    row["financials"]["balance_sheet"] = "unexpected"
    [statement] = provider_with(_handler_for(_body(row))).get_financials("AAPL")
    assert not any(k.startswith("balance_sheet.") for k in statement.values)
    assert "income_statement.revenues" in statement.values


def test_unusable_fiscal_year_is_none_not_guessed():
    [statement] = provider_with(
        _handler_for(_body(_row(fiscal_year="unknown")))
    ).get_financials("AAPL")
    assert statement.fiscal_year is None


def test_string_fiscal_year_is_parsed():
    [statement] = provider_with(
        _handler_for(_body(_row(fiscal_year="2024")))
    ).get_financials("AAPL")
    assert statement.fiscal_year == 2024


def test_missing_timeframe_falls_back_to_the_requested_one():
    [statement] = provider_with(
        _handler_for(_body(_row(timeframe=None)))
    ).get_financials("AAPL", timeframe="ttm")
    assert statement.timeframe == "ttm"


def test_missing_cik_and_source_url_are_none():
    [statement] = provider_with(
        _handler_for(_body(_row(cik=None, source_filing_url=12)))
    ).get_financials("AAPL")
    assert statement.cik is None
    assert statement.source_filing_url is None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_statements_are_returned_newest_period_first_whatever_the_wire_order():
    body = _body(
        _row(end_date="2023-12-30", filing_date="2024-02-01"),
        _row(end_date="2024-06-29", filing_date="2024-08-02"),
        _row(end_date="2024-03-30", filing_date="2024-05-02"),
    )
    statements = provider_with(_handler_for(body)).get_financials("AAPL")
    assert [s.end_date for s in statements] == [
        date(2024, 6, 29),
        date(2024, 3, 30),
        date(2023, 12, 30),
    ]


def test_an_amended_filing_does_not_reorder_the_fiscal_series():
    """A late amendment of an old quarter stays behind the newer quarter."""
    body = _body(
        _row(end_date="2023-12-30", filing_date="2024-09-30"),  # amended late
        _row(end_date="2024-06-29", filing_date="2024-08-02"),
    )
    statements = provider_with(_handler_for(body)).get_financials("AAPL")
    assert [s.end_date for s in statements] == [date(2024, 6, 29), date(2023, 12, 30)]


# ---------------------------------------------------------------------------
# Capability honesty (§16)
# ---------------------------------------------------------------------------


def test_403_raises_capability_not_available_naming_the_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "not entitled"})

    with pytest.raises(CapabilityNotAvailable) as exc:
        provider_with(handler).get_financials("AAPL")
    assert FINANCIALS_PATH in str(exc.value)


def test_transport_fault_raises_market_data_error_not_an_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(MarketDataError):
        provider_with(handler).get_financials("AAPL")


def test_non_json_body_raises_rather_than_returning_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    with pytest.raises(MarketDataError):
        provider_with(handler).get_financials("AAPL")


def test_probe_reports_financials_false_when_the_plan_excludes_it():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == FINANCIALS_PATH:
            return httpx.Response(403, json={"message": "not entitled"})
        return httpx.Response(200, json={"status": "OK", "results": []})

    report = provider_with(handler).probe_capabilities()
    assert report["financials"] is False
    assert report["stock_history"] is True  # unaffected


def test_probe_key_sets_are_identical_across_providers():
    """A caller must never have to ask WHICH provider it is holding (§16)."""

    def massive_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "OK", "results": []})

    def alpaca_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    massive_report = provider_with(massive_handler).probe_capabilities()
    alpaca_report = AlpacaMarketDataProvider(
        api_key_id="k",
        api_secret_key="s",
        transport=httpx.MockTransport(alpaca_handler),
    ).probe_capabilities()
    assert set(massive_report) == set(alpaca_report)
    assert "financials" in massive_report


# ---------------------------------------------------------------------------
# Alpaca: no fundamentals at any tier
# ---------------------------------------------------------------------------


def test_alpaca_get_financials_refuses_instead_of_returning_empty():
    """``[]`` would claim the COMPANY filed nothing; the truth is about Alpaca."""
    with pytest.raises(CapabilityNotAvailable) as exc:
        alpaca_provider().get_financials("AAPL")
    message = str(exc.value)
    assert "Alpaca" in message and "AAPL" in message
    assert "Massive" in message  # names where fundamentals do come from


def test_alpaca_financials_refusal_is_a_market_data_error_subclass():
    with pytest.raises(MarketDataError):
        alpaca_provider().get_financials("AAPL")


def test_alpaca_probe_reports_financials_false_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "financials" not in str(request.url), "no financials probe request"
        return httpx.Response(200, json={})

    report = AlpacaMarketDataProvider(
        api_key_id="k", api_secret_key="s", transport=httpx.MockTransport(handler)
    ).probe_capabilities()
    assert report["financials"] is False


# ---------------------------------------------------------------------------
# Stub: deterministic, and honest about the same gaps as the real provider
# ---------------------------------------------------------------------------


def test_stub_statements_are_deterministic_across_instances():
    first = StubProvider().get_financials("NVDA", limit=6)
    second = StubProvider().get_financials("NVDA", limit=6)
    assert first == second


def test_stub_statements_are_newest_first_and_limited():
    statements = StubProvider().get_financials("NVDA", limit=5)
    assert len(statements) == 5
    ends = [s.end_date for s in statements]
    assert ends == sorted(ends, reverse=True)


def test_stub_acceptance_is_after_the_period_end_so_asof_has_something_to_bite_on():
    for statement in StubProvider().get_financials("NVDA", limit=4):
        assert statement.acceptance_datetime is not None
        assert statement.acceptance_datetime.tzinfo is not None
        assert statement.acceptance_datetime.date() > statement.end_date


def test_stub_omits_exactly_the_fields_the_real_provider_omits():
    """capex/cash/D&A absent here for the same reason they are absent live."""
    [statement] = StubProvider().get_financials("NVDA", limit=1)
    for key in statement.values:
        assert not key.endswith((".cash", ".capital_expenditure"))
        assert "depreciation" not in key
        assert "receivable" not in key
        assert "interest_expense" not in key


def test_stub_quarters_carry_distinct_fiscal_labels():
    statements = StubProvider().get_financials("NVDA", limit=4)
    assert len({(s.fiscal_year, s.fiscal_period) for s in statements}) == 4
    assert all(s.fiscal_period.startswith("Q") for s in statements)
    assert all(s.timeframe == "quarterly" for s in statements)


def test_stub_ttm_sums_the_trailing_four_quarters():
    provider = StubProvider()
    quarters = provider.get_financials("NVDA", limit=4)
    [ttm] = provider.get_financials("NVDA", timeframe="ttm", limit=1)
    assert ttm.fiscal_period == "TTM"
    assert ttm.end_date == quarters[0].end_date
    assert ttm.values["income_statement.revenues"] == pytest.approx(
        sum(q.values["income_statement.revenues"] for q in quarters)
    )


def test_stub_ttm_keeps_balance_sheet_lines_point_in_time():
    provider = StubProvider()
    newest = provider.get_financials("NVDA", limit=1)[0]
    [ttm] = provider.get_financials("NVDA", timeframe="ttm", limit=1)
    assert ttm.values["balance_sheet.equity"] == newest.values["balance_sheet.equity"]
    assert (
        ttm.values["income_statement.diluted_average_shares"]
        == newest.values["income_statement.diluted_average_shares"]
    )


def test_stub_annual_rollup_is_labelled_fy():
    [annual] = StubProvider().get_financials("NVDA", timeframe="annual", limit=1)
    assert annual.fiscal_period == "FY"
    assert annual.timeframe == "annual"


def test_stub_values_are_finite_floats():
    for statement in StubProvider().get_financials("NVDA", limit=8):
        assert statement.values
        assert all(
            isinstance(v, float) and math.isfinite(v)
            for v in statement.values.values()
        )


def test_stub_filings_are_never_mistakable_for_a_real_citation():
    for statement in StubProvider().get_financials("NVDA", limit=3):
        assert statement.source_filing_url.startswith("stub://")


def test_stub_different_tickers_get_different_scales():
    a = StubProvider().get_financials("NVDA", limit=1)[0]
    b = StubProvider().get_financials("COST", limit=1)[0]
    assert (
        a.values["income_statement.revenues"] != b.values["income_statement.revenues"]
    )


def test_stub_rejects_an_unsupported_timeframe():
    with pytest.raises(ValueError, match="unsupported financials timeframe"):
        StubProvider().get_financials("NVDA", timeframe="monthly")


@pytest.mark.parametrize("args", [{"ticker": "", "limit": 4}, {"ticker": "NVDA", "limit": 0}])
def test_stub_returns_empty_for_a_blank_ticker_or_zero_limit(args):
    assert StubProvider().get_financials(args["ticker"], limit=args["limit"]) == []
