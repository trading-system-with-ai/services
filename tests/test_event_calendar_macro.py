"""Macro calendar adapters + macro data providers (Phase G U1).

Every test runs the REAL adapter over ``httpx.MockTransport`` against
LIVE-DERIVED fixtures (downloaded 2026-08-19 — see
``tests/fixtures/events/README.md``). The network is never touched.

Pinned, in order of importance:

1. **THE RELEASE TIME IS PARSED, NEVER ASSUMED.** The live pages disagree with
   the "all macro prints are 08:30" folklore: JOLTS drops at 10:00 ET, which
   is AFTER the open and therefore ``DURING_MARKET``. A row whose time cannot
   be parsed is DROPPED, not defaulted.
2. **NO FABRICATION** (§2, §11). A layout change yields ZERO rows plus a
   warning, never a partially-parsed release at a guessed instant. An empty
   Treasury cell is ``None``, never ``0.0``. A BLS ``REQUEST_NOT_PROCESSED``
   raises rather than reading as an empty series.
3. **ET -> UTC ACROSS DST.** A February 08:30 ET CPI is 13:30Z and an August
   one is 12:30Z. Every ``scheduled_at`` is wrong for half the year if this
   is wrong.
4. **CAPABILITY HONESTY** (audit §6). 403 -> ``False`` (proven absence);
   500/transport fault -> the error STRING (unknown); a 200 that parses ->
   ``True``; a 200 that does NOT parse -> a string, because a layout change is
   an availability problem even though HTTP succeeded.
5. **CONTACT USER-AGENT** on every government request (§8; SEC fair access).
"""
import json
import pathlib
from datetime import date, datetime, timezone

import httpx
import pytest

from libs.event_calendar import (
    CAPABILITY_KEYS,
    KEYLESS_PROVIDERS,
    bea_macro_data_provider,
    configured_provider_names,
    get_provider,
    macro_data_provider,
)
from libs.event_calendar.bea import BeaCalendarProvider
from libs.event_calendar.bea import classify_release_title
from libs.event_calendar.bea import parse_schedule_page as parse_bea_schedule
from libs.event_calendar.bea import release_period_code
from libs.event_calendar.bls import (
    HEADLINE_SERIES,
    SCHEDULE_SLUGS,
    BlsCalendarProvider,
    reference_period_code,
)
from libs.event_calendar.bls import parse_schedule_page as parse_bls_schedule
from libs.event_calendar.macro_data import (
    BeaMacroDataProvider,
    BlsMacroDataProvider,
    MacroDataError,
    MacroObservation,
    parse_bls_series_response,
    period_key,
)
from libs.event_calendar.provider import CapabilityNotAvailable
from libs.event_calendar.treasury import (
    TENOR_2Y,
    TENOR_10Y,
    TreasuryYields,
    parse_yield_curve_csv,
    yield_change_bp,
)
from libs.market_data.provider import MarketDataError
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "events"

UTC = timezone.utc
WIDE_START = datetime(2020, 1, 1, tzinfo=UTC)
WIDE_END = datetime(2030, 1, 1, tzinfo=UTC)

USER_AGENT = "trading-system-with-ai/0.1 (catalyst research; test@example.com)"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


def _bls_schedule_transport(seen: list[httpx.Request] | None = None):
    """Serves the four live BLS schedule fixtures by slug."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        slug = request.url.path.rsplit("/", 1)[-1].removesuffix(".htm")
        path = FIXTURES / f"bls_schedule_{slug}.html"
        if not path.exists():
            return httpx.Response(404, text="no such release")
        return httpx.Response(200, text=path.read_text())

    return httpx.MockTransport(handler)


def _bea_transport(seen: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, text=_fixture("bea_schedule.html"))

    return httpx.MockTransport(handler)


def _status_transport(status: int, body: str = "nope"):
    return httpx.MockTransport(lambda request: httpx.Response(status, text=body))


def _bls_provider(**kwargs) -> BlsCalendarProvider:
    kwargs.setdefault("transport", _bls_schedule_transport())
    return BlsCalendarProvider(user_agent=USER_AGENT, **kwargs)


def _bea_provider(**kwargs) -> BeaCalendarProvider:
    kwargs.setdefault("transport", _bea_transport())
    return BeaCalendarProvider(user_agent=USER_AGENT, **kwargs)


# ---------------------------------------------------------------------------
# BLS schedule parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug,heading",
    [
        ("cpi", "Consumer Price Index"),
        ("ppi", "Producer Price Index"),
        ("empsit", "Employment Situation"),
        ("jolts", "Job Openings and Labor Turnover"),
    ],
)
def test_bls_fixture_is_the_live_page_for_its_release(slug, heading):
    """Each fixture is the page it claims to be — guards a bad re-download."""
    html = _fixture(f"bls_schedule_{slug}.html")
    assert heading in html
    assert len(parse_bls_schedule(html)) >= 10


def test_bls_cpi_schedule_parses_every_live_row():
    rows = parse_bls_schedule(_fixture("bls_schedule_cpi.html"))
    assert len(rows) == 13  # the live 2026 page carries 13 releases
    first = rows[0]
    assert first["reference_period"] == "November 2025"
    assert first["release_date"] == date(2025, 12, 18)
    assert first["release_time"] == (8, 30)
    # Zero-padded days ("Jan. 09, 2026") parse identically to unpadded ones.
    empsit = parse_bls_schedule(_fixture("bls_schedule_empsit.html"))
    assert date(2026, 1, 9) in {row["release_date"] for row in empsit}


def test_bls_jolts_releases_at_ten_am_not_eight_thirty():
    """The single most load-bearing fixture fact.

    JOLTS is a 10:00 ET release. Assuming 08:30 for "all BLS macro" would put
    every JOLTS event 90 minutes early AND in the wrong session bucket.
    """
    rows = parse_bls_schedule(_fixture("bls_schedule_jolts.html"))
    assert {row["release_time"] for row in rows} == {(10, 0)}
    cpi = parse_bls_schedule(_fixture("bls_schedule_cpi.html"))
    assert {row["release_time"] for row in cpi} == {(8, 30)}


def test_bls_layout_change_yields_zero_rows_never_a_guess():
    assert parse_bls_schedule("<html><body><p>we redesigned</p></body></html>") == []
    assert parse_bls_schedule("") == []


def test_bls_row_with_unparseable_time_is_dropped_not_defaulted():
    """A row we cannot time is not a fact we may invent (§11)."""
    html = """
    <table class="release-list"><tbody>
      <tr><td>July 2026</td><td>Aug. 12, 2026</td><td>08:30 AM</td></tr>
      <tr><td>August 2026</td><td>Sep. 11, 2026</td><td>sometime</td></tr>
      <tr><td>September 2026</td><td>whenever</td><td>08:30 AM</td></tr>
    </tbody></table>
    """
    rows = parse_bls_schedule(html)
    assert [row["release_date"] for row in rows] == [date(2026, 8, 12)]


def test_reference_period_code_is_the_canonical_join_key():
    assert reference_period_code("July 2026") == "2026-07"
    assert reference_period_code("November 2025") == "2025-11"
    assert reference_period_code("not a period") is None


# ---------------------------------------------------------------------------
# BLS candidates
# ---------------------------------------------------------------------------


def test_bls_fetch_events_covers_all_four_typed_releases():
    events = _bls_provider().fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    by_type: dict[EventType, int] = {}
    for candidate in events:
        by_type[candidate.event_type] = by_type.get(candidate.event_type, 0) + 1
    assert set(by_type) == {
        EventType.CPI,
        EventType.PPI,
        EventType.EMPLOYMENT_REPORT,
        EventType.JOLTS,
    }
    assert set(SCHEDULE_SLUGS.values()) == set(by_type)
    assert all(count >= 10 for count in by_type.values())


def test_bls_candidate_shape_and_event_key_format():
    events = _bls_provider().fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    cpi = next(c for c in events if c.event_key == "CPI:2026-08-12")
    assert cpi.event_type is EventType.CPI
    assert cpi.title == "CPI — July 2026"  # the reference period is in the title
    assert cpi.status is EventStatus.CONFIRMED
    assert cpi.source is EventSourceKind.GOVERNMENT_AGENCY
    assert cpi.source_name == "bls"
    assert cpi.agency == "Bureau of Labor Statistics"
    assert cpi.event_timezone == "America/New_York"
    assert cpi.release_period == "2026-07"
    assert cpi.series_id == HEADLINE_SERIES[EventType.CPI]
    assert cpi.source_url.endswith("/schedule/news_release/cpi.htm")
    assert cpi.raw["release_time_text"] == "08:30 AM"
    assert cpi.raw["reference_period"] == "July 2026"


def test_bls_release_times_convert_et_to_utc_across_dst():
    """February 08:30 ET is 13:30Z; August 08:30 ET is 12:30Z."""
    events = _bls_provider().fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    by_key = {c.event_key: c for c in events}
    assert by_key["CPI:2026-02-13"].scheduled_at == datetime(
        2026, 2, 13, 13, 30, tzinfo=UTC
    )
    assert by_key["CPI:2026-08-12"].scheduled_at == datetime(
        2026, 8, 12, 12, 30, tzinfo=UTC
    )


def test_bls_session_bucket_follows_the_parsed_time():
    """08:30 releases are BEFORE_MARKET; the 10:00 JOLTS is DURING_MARKET."""
    events = _bls_provider().fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    for candidate in events:
        if candidate.event_type is EventType.JOLTS:
            assert candidate.session is EventSession.DURING_MARKET
        else:
            assert candidate.session is EventSession.BEFORE_MARKET


def test_bls_fetch_events_filters_to_the_requested_window():
    provider = _bls_provider()
    events = provider.fetch_events(
        tickers=[],
        start=datetime(2026, 8, 1, tzinfo=UTC),
        end=datetime(2026, 8, 31, tzinfo=UTC),
    )
    assert events
    assert all(
        datetime(2026, 8, 1, tzinfo=UTC)
        <= c.scheduled_at
        <= datetime(2026, 8, 31, tzinfo=UTC)
        for c in events
    )


def test_bls_as_of_keeps_schedule_rows_because_the_schedule_is_published_ahead():
    """BLS publishes a year ahead: a future release date was knowable at as_of.

    (Point-in-time gating applies to macro OBSERVATIONS, not to the schedule.)
    """
    provider = _bls_provider()
    events = provider.fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END, as_of=datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert any(c.scheduled_at > datetime(2026, 6, 1, tzinfo=UTC) for c in events)


def test_bls_one_dead_page_does_not_cost_the_other_releases():
    """§8 failure isolation, per schedule page."""

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.path.rsplit("/", 1)[-1].removesuffix(".htm")
        if slug == "jolts":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, text=_fixture(f"bls_schedule_{slug}.html"))

    events = _bls_provider(transport=httpx.MockTransport(handler)).fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    assert events
    assert not any(c.event_type is EventType.JOLTS for c in events)
    assert any(c.event_type is EventType.CPI for c in events)


def test_bls_sends_the_contact_user_agent_on_every_request():
    seen: list[httpx.Request] = []
    _bls_provider(transport=_bls_schedule_transport(seen)).fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    assert seen
    assert all(r.headers["User-Agent"] == USER_AGENT for r in seen)


def test_bls_capabilities_are_tri_state():
    ok = _bls_provider().capabilities()
    assert set(ok) == set(CAPABILITY_KEYS)  # the FIXED key set
    assert ok["macro_calendar"] is True
    assert ok["earnings_calendar"] is False

    denied = _bls_provider(transport=_status_transport(403)).capabilities()
    assert denied["macro_calendar"] is False  # proven absence

    faulted = _bls_provider(transport=_status_transport(500)).capabilities()
    assert isinstance(faulted["macro_calendar"], str)  # availability unknown

    # HTTP 200 that does not parse is a layout change -> a STRING, not True.
    relaid = _bls_provider(
        transport=_status_transport(200, "<html>redesigned</html>")
    ).capabilities()
    assert isinstance(relaid["macro_calendar"], str)
    assert "layout" in relaid["macro_calendar"]


def test_bls_does_not_serve_exchange_sessions():
    with pytest.raises(CapabilityNotAvailable):
        _bls_provider().fetch_market_calendar(date(2026, 1, 1), date(2026, 2, 1))


# ---------------------------------------------------------------------------
# BEA
# ---------------------------------------------------------------------------


def test_bea_titles_never_spell_out_gross_domestic_product():
    """The live page says "GDP (Advance Estimate)" — the phrase this parser
    would have matched on does not appear at all."""
    html = _fixture("bea_schedule.html")
    assert "Gross Domestic Product" not in html
    assert "GDP (Advance Estimate)" in html


def test_bea_schedule_parses_rows_and_takes_the_year_from_the_header():
    rows = parse_bea_schedule(_fixture("bea_schedule.html"))
    assert len(rows) >= 15
    first = rows[0]
    assert first["release_date"] == date(2026, 8, 26)  # year from <th>Year 2026</th>
    assert first["release_time"] == (8, 30)
    assert first["title"].startswith("GDP (Second Estimate)")


def test_bea_year_rolls_over_when_the_month_goes_backwards():
    """December -> January means the next year, not a date 11 months in the past."""
    html = """
    <table id="release-schedule-table">
      <thead><tr><th>Year 2026</th></tr></thead>
      <tbody>
        <tr><td><div class="release-date">December 23</div>
            <small class="text-muted">8:30 AM</small></td>
            <td class="release-title">GDP (Third Estimate), 3rd Quarter 2026</td></tr>
        <tr><td><div class="release-date">January 29</div>
            <small class="text-muted">8:30 AM</small></td>
            <td class="release-title">GDP (Advance Estimate), 4th Quarter 2026</td></tr>
      </tbody>
    </table>
    """
    rows = parse_bea_schedule(html)
    assert [r["release_date"] for r in rows] == [date(2026, 12, 23), date(2027, 1, 29)]


def test_bea_without_a_year_header_refuses_to_date_any_row():
    html = """
    <table id="release-schedule-table"><tbody>
      <tr><td><div class="release-date">August 26</div>
          <small class="text-muted">8:30 AM</small></td>
          <td class="release-title">GDP (Advance Estimate), 2nd Quarter 2026</td></tr>
    </tbody></table>
    """
    assert parse_bea_schedule(html) == []


def test_bea_classifies_only_headline_gdp_and_pce_releases():
    assert classify_release_title("GDP (Advance Estimate), 3rd Quarter 2026") is EventType.GDP
    assert (
        classify_release_title("Personal Income and Outlays, July 2026") is EventType.PCE
    )
    # Regional / annual products are NOT market prints and must not be typed.
    assert classify_release_title("GDP by County and Personal Income by County, 2025") is None
    assert (
        classify_release_title(
            "Real Personal Consumption Expenditures by State, 2025"
        )
        is None
    )
    assert classify_release_title("U.S. International Trade in Goods and Services, July 2026") is None


def test_bea_release_period_code_handles_quarters_and_months():
    assert release_period_code("GDP (Advance Estimate), 3rd Quarter 2026") == "2026-Q3"
    assert release_period_code("Personal Income and Outlays, July 2026") == "2026-07"
    assert release_period_code("Something Undated") is None


def test_bea_fetch_events_emits_typed_gdp_and_pce_candidates():
    events = _bea_provider().fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    assert {c.event_type for c in events} == {EventType.GDP, EventType.PCE}
    gdp = next(c for c in events if c.event_key == "GDP:2026-08-26")
    assert gdp.status is EventStatus.CONFIRMED
    assert gdp.source is EventSourceKind.GOVERNMENT_AGENCY
    assert gdp.source_name == "bea"
    assert gdp.agency == "Bureau of Economic Analysis"
    assert gdp.release_period == "2026-Q2"
    assert gdp.session is EventSession.BEFORE_MARKET
    # 08:30 ET in August is 12:30Z.
    assert gdp.scheduled_at == datetime(2026, 8, 26, 12, 30, tzinfo=UTC)
    assert gdp.raw["release_time_text"].startswith("8:30")

    pce = next(c for c in events if c.event_key == "PCE:2026-08-26")
    assert pce.event_type is EventType.PCE
    assert pce.release_period == "2026-07"
    assert "Personal Income and Outlays" in pce.title


def test_bea_skips_regional_products_present_in_the_live_fixture():
    events = _bea_provider().fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    titles = " ".join(c.title for c in events)
    assert "by County" not in titles
    assert "Multinational Enterprises" not in titles


def test_bea_capabilities_are_tri_state():
    assert _bea_provider().capabilities()["macro_calendar"] is True
    assert _bea_provider(transport=_status_transport(403)).capabilities()["macro_calendar"] is False
    assert isinstance(
        _bea_provider(transport=_status_transport(500)).capabilities()["macro_calendar"], str
    )


def test_bea_sends_the_contact_user_agent():
    seen: list[httpx.Request] = []
    _bea_provider(transport=_bea_transport(seen)).fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    assert seen and all(r.headers["User-Agent"] == USER_AGENT for r in seen)


def test_bea_fetch_events_survives_a_dead_page():
    assert (
        _bea_provider(transport=_status_transport(500)).fetch_events(
            tickers=[], start=WIDE_START, end=WIDE_END
        )
        == []
    )


# ---------------------------------------------------------------------------
# BLS macro data (API v1)
# ---------------------------------------------------------------------------


def _bls_series_transport(seen: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, text=_fixture("bls_series_cusr0000sa0.json"))

    return httpx.MockTransport(handler)


def test_bls_series_fixture_parses_into_ascending_observations():
    payload = json.loads(_fixture("bls_series_cusr0000sa0.json"))
    observations = parse_bls_series_response(payload)
    assert observations
    assert all(o.series_id == "CUSR0000SA0" for o in observations)
    periods = [o.period for o in observations]
    assert periods == sorted(periods)
    latest = observations[-1]
    assert latest.period == "2026-07"
    assert latest.value == pytest.approx(332.813)


def test_bls_series_skips_annual_averages():
    """M13 is an annual average; folded into a monthly series it corrupts MoM."""
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "CUSR0000SA0",
                    "data": [
                        {"year": "2025", "period": "M13", "value": "320.0"},
                        {"year": "2025", "period": "M12", "value": "321.0"},
                    ],
                }
            ]
        },
    }
    observations = parse_bls_series_response(payload)
    assert [o.period for o in observations] == ["2025-12"]


def test_bls_series_drops_non_numeric_values_rather_than_zeroing_them():
    """The live CPI fixture itself carries a '-' value for one month."""
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "X",
                    "data": [
                        {"year": "2026", "period": "M01", "value": "-"},
                        {"year": "2026", "period": "M02", "value": "1.5"},
                    ],
                }
            ]
        },
    }
    observations = parse_bls_series_response(payload)
    assert [(o.period, o.value) for o in observations] == [("2026-02", 1.5)]


def test_bls_request_not_processed_raises_instead_of_reading_as_empty():
    """A throttled v1 answer is HTTP 200 + REQUEST_NOT_PROCESSED.

    Treating it as an empty series would report a real macro print as
    permanently unavailable.
    """
    payload = {
        "status": "REQUEST_NOT_PROCESSED",
        "message": ["Daily threshold for Series Report exceeded"],
        "Results": {},
    }
    with pytest.raises(MacroDataError) as excinfo:
        parse_bls_series_response(payload)
    assert "threshold" in str(excinfo.value).lower()


def test_bls_macro_client_filters_to_the_requested_years():
    client = BlsMacroDataProvider(
        user_agent=USER_AGENT, transport=_bls_series_transport()
    )
    observations = client.get_series("CUSR0000SA0", start_year=2026, end_year=2026)
    assert observations
    assert {o.year for o in observations} == {2026}


def test_bls_macro_client_sends_the_contact_user_agent():
    seen: list[httpx.Request] = []
    BlsMacroDataProvider(
        user_agent=USER_AGENT, transport=_bls_series_transport(seen)
    ).get_series("CUSR0000SA0", start_year=2024, end_year=2026)
    assert seen and all(r.headers["User-Agent"] == USER_AGENT for r in seen)


def test_bls_macro_client_maps_rate_limit_and_denial_distinctly():
    rate_limited = BlsMacroDataProvider(
        user_agent=USER_AGENT, transport=_status_transport(429)
    )
    with pytest.raises(MacroDataError) as excinfo:
        rate_limited.get_series("CUSR0000SA0", start_year=2026, end_year=2026)
    assert "429" in str(excinfo.value)

    denied = BlsMacroDataProvider(user_agent=USER_AGENT, transport=_status_transport(403))
    with pytest.raises(CapabilityNotAvailable):
        denied.get_series("CUSR0000SA0", start_year=2026, end_year=2026)

    broken = BlsMacroDataProvider(
        user_agent=USER_AGENT, transport=_status_transport(200, "not json")
    )
    with pytest.raises(MacroDataError):
        broken.get_series("CUSR0000SA0", start_year=2026, end_year=2026)


def test_bls_macro_client_rejects_a_blank_series_id():
    client = BlsMacroDataProvider(user_agent=USER_AGENT, transport=_bls_series_transport())
    with pytest.raises(MacroDataError):
        client.get_series("  ", start_year=2026, end_year=2026)


def test_period_key_and_observation_period_anchors():
    assert period_key(2026, "M07") == "2026-07"
    assert period_key(2026, "Q02") == "2026-Q2"
    assert period_key(2026, "M13") is None  # annual average
    assert period_key(2026, "M99") is None

    monthly = MacroObservation(
        series_id="X", period="2026-07", value=1.0, year=2026, period_code="M07"
    )
    assert monthly.period_end == date(2026, 7, 31)
    assert monthly.estimated_release_date == date(2026, 9, 14)  # +45d, ESTIMATED
    assert monthly.is_quarterly is False

    quarterly = MacroObservation(
        series_id="X", period="2026-Q2", value=1.0, year=2026, period_code="Q02"
    )
    assert quarterly.period_end == date(2026, 6, 30)
    assert quarterly.is_quarterly is True

    december = MacroObservation(
        series_id="X", period="2026-12", value=1.0, year=2026, period_code="M12"
    )
    assert december.period_end == date(2026, 12, 31)


# ---------------------------------------------------------------------------
# BEA macro data — honest unavailability without a key
# ---------------------------------------------------------------------------


def test_bea_actuals_without_a_key_are_unavailable_never_estimated():
    provider = BeaMacroDataProvider(api_key="")
    assert provider.is_configured is False
    with pytest.raises(CapabilityNotAvailable) as excinfo:
        provider.get_series("T10101", start_year=2026, end_year=2026)
    message = str(excinfo.value)
    assert "BEA_API_KEY" in message
    # The calendar half keeps working — that distinction is the whole point.
    assert "DATES" in message or "dates" in message


def test_bea_macro_provider_factory_reads_the_settings_key(monkeypatch):
    class _Settings:
        bea_api_key = "abc123"
        sec_user_agent = USER_AGENT

    provider = bea_macro_data_provider(_Settings())
    assert provider.is_configured is True


# ---------------------------------------------------------------------------
# Treasury yield curve
# ---------------------------------------------------------------------------


def test_treasury_csv_parses_live_fixture_ascending():
    rows = parse_yield_curve_csv(_fixture("treasury_yield_curve_2026.csv"))
    assert len(rows) >= 40
    assert [r.date for r in rows] == sorted(r.date for r in rows)
    latest = rows[-1]
    assert latest.date == date(2026, 8, 18)
    assert latest.yield_for(TENOR_2Y) == pytest.approx(4.19)
    assert latest.yield_for(TENOR_10Y) == pytest.approx(4.71)


def test_treasury_columns_are_addressed_by_header_name_not_position():
    """The live 2026 file has a '1.5 Month' column mid-row that older years
    lack; positional parsing would report the 2Y as the 3Y."""
    header = 'Date,"10 Yr","2 Yr","1 Mo"\n08/18/2026,4.71,4.19,3.78\n'
    rows = parse_yield_curve_csv(header)
    assert rows[0].yield_for(TENOR_2Y) == pytest.approx(4.19)
    assert rows[0].yield_for(TENOR_10Y) == pytest.approx(4.71)


def test_treasury_missing_cell_is_none_never_zero():
    """A blank tenor is ABSENT. Zero would be a fabricated 0% yield."""
    rows = parse_yield_curve_csv('Date,"2 Yr","20 Yr"\n08/18/2026,4.19,\n')
    assert rows[0].yield_for(TENOR_2Y) == pytest.approx(4.19)
    assert rows[0].yield_for("20 Yr") is None
    assert "20 Yr" not in rows[0].tenors


def test_treasury_bad_rows_and_layouts_fail_empty():
    assert parse_yield_curve_csv("") == []
    # No Date column -> refuse to parse positionally.
    assert parse_yield_curve_csv('"2 Yr","10 Yr"\n4.19,4.71\n') == []
    # An unparseable date drops just that row.
    rows = parse_yield_curve_csv('Date,"2 Yr"\nnot-a-date,4.19\n08/18/2026,4.20\n')
    assert [r.date for r in rows] == [date(2026, 8, 18)]


def test_yield_change_bp_is_none_when_either_side_is_missing():
    assert yield_change_bp(4.19, 4.31) == pytest.approx(12.0)
    assert yield_change_bp(4.31, 4.19) == pytest.approx(-12.0)
    assert yield_change_bp(None, 4.19) is None
    assert yield_change_bp(4.19, None) is None


def test_treasury_client_fetches_and_reports_faults_in_the_shared_taxonomy():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=_fixture("treasury_yield_curve_2026.csv"))

    client = TreasuryYields(user_agent=USER_AGENT, transport=httpx.MockTransport(handler))
    rows = client.get_yield_curve(2026)
    assert rows
    assert seen[0].headers["User-Agent"] == USER_AGENT
    assert "daily_treasury_yield_curve" in str(seen[0].url)

    with pytest.raises(CapabilityNotAvailable):
        TreasuryYields(user_agent=USER_AGENT, transport=_status_transport(403)).get_yield_curve(2026)
    with pytest.raises(MarketDataError):
        TreasuryYields(user_agent=USER_AGENT, transport=_status_transport(500)).get_yield_curve(2026)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_exposes_bls_and_bea_as_keyless_providers():
    assert "bls" in KEYLESS_PROVIDERS
    assert "bea" in KEYLESS_PROVIDERS
    assert get_provider("bls").name == "bls"
    assert get_provider("bea").name == "bea"


def test_keyless_macro_providers_are_configured_without_any_credentials():
    class _Settings:
        event_calendar_providers = ""
        alpaca_api_key_id = ""
        alpaca_api_secret_key = ""
        massive_api_key = ""

    names = configured_provider_names(_Settings())
    assert "bls" in names and "bea" in names


def test_unknown_provider_name_still_raises_value_error():
    with pytest.raises(ValueError):
        get_provider("bureau_of_vibes")


def test_macro_data_provider_factory_returns_the_keyless_bls_client():
    class _Settings:
        sec_user_agent = USER_AGENT
        bea_api_key = ""

    provider = macro_data_provider(_Settings())
    assert provider.name == "bls"
    assert isinstance(provider, BlsMacroDataProvider)


def test_government_providers_default_to_a_contact_user_agent(monkeypatch):
    """Even with SEC_USER_AGENT unset, requests carry a contact-shaped UA."""

    class _Settings:
        sec_user_agent = ""

    from libs.event_calendar import _government_user_agent

    agent = _government_user_agent(_Settings())
    assert "trading-system-with-ai" in agent
    assert "SEC_USER_AGENT" in agent
