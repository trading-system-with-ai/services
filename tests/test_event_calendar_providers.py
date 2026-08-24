"""libs/event_calendar providers — every test runs the REAL adapter over
``httpx.MockTransport``; the network is never touched (the discipline
``tests/test_alpaca_market_data.py`` established).

Pinned, in order of importance:

1. NO FABRICATION. A source that cannot answer yields FEWER events, never an
   invented date: an unparseable Alpaca session row is skipped rather than
   defaulted to 09:30-16:00; a Fed page whose layout changed yields ZERO
   meetings, not a partial one at a guessed time; ``estimate_next_earnings``
   returns ``None`` below two observations.
2. CAPABILITY HONESTY (audit §6). 403 -> capability ``False`` (proven
   absence — Benzinga earnings is 403 TODAY and must never crash the loop);
   500/transport fault -> the error STRING (availability unknown); success ->
   ``True``. The three states are distinct and every provider reports the
   same FIXED key set.
3. POINT-IN-TIME (§96). ``as_of`` drops SEC filings accepted after it and RSS
   items published after it — at as_of=T the platform cannot know about a
   release that happened at T+1h.
4. ET -> UTC ACROSS DST. A January 14:00 ET decision is 19:00Z and a July one
   is 18:00Z. Every scheduled_at in this package is UTC and every one of them
   is wrong for half the year if this is wrong.
"""
import json
import pathlib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from libs.event_calendar import (
    CAPABILITY_KEYS,
    ProviderNotConfigured,
    configured_provider_names,
    configured_providers,
    get_provider,
)
from libs.event_calendar.alpaca_calendar import AlpacaCalendarProvider
from libs.event_calendar.fed import (
    FedProvider,
    is_senior_speaker,
    parse_fomc_calendar,
    parse_speeches_rss,
)
from libs.event_calendar.massive_calendar import MassiveCalendarProvider
from libs.event_calendar.provider import (
    CalendarProviderError,
    CapabilityNotAvailable,
    EventCandidate,
    classify_session_et,
)
from libs.event_calendar.sec_edgar import (
    SecEdgarProvider,
    cluster_releases,
    estimate_next_earnings,
    reset_cik_cache,
)
from libs.event_calendar.stub import StubEventCalendarProvider
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


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def _fixture_json(name: str) -> object:
    return json.loads(_fixture(name))


@pytest.fixture(autouse=True)
def _clean_cik_cache():
    """The ticker->CIK map is process-lifetime; isolate it per test."""
    reset_cik_cache()
    yield
    reset_cik_cache()


def _status_handler(status: int, body: str = "denied"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


# ===========================================================================
# Registry (clones libs/market_data/__init__.py's contract)
# ===========================================================================


def test_registry_empty_name_is_not_configured():
    with pytest.raises(ProviderNotConfigured):
        get_provider("   ")


def test_registry_unknown_name_is_a_value_error_naming_the_known_set():
    with pytest.raises(ValueError) as exc:
        get_provider("bloomberg")
    assert "sec_edgar" in str(exc.value) and "fed" in str(exc.value)


def test_keyless_primary_sources_are_always_configured():
    """No vendor key at all still yields a real calendar (SEC + Fed are free)."""

    class Settings:
        alpaca_api_key_id = ""
        alpaca_api_secret_key = ""
        massive_api_key = ""
        event_calendar_providers = ""

    names = configured_provider_names(Settings())
    # Phase G adds the two keyless government macro calendars (BLS, BEA).
    assert names == ["sec_edgar", "fed", "bls", "bea"]
    assert "stub" not in names


def test_vendor_providers_appear_only_with_their_credentials():
    class Settings:
        alpaca_api_key_id = "k"
        alpaca_api_secret_key = "s"
        massive_api_key = "m"
        event_calendar_providers = ""

    names = configured_provider_names(Settings())
    assert set(names) == {
        "sec_edgar", "fed", "bls", "bea", "alpaca_calendar", "massive_calendar",
    }


def test_stub_is_only_reachable_when_settings_name_it_explicitly():
    class Explicit:
        alpaca_api_key_id = ""
        alpaca_api_secret_key = ""
        massive_api_key = ""
        event_calendar_providers = "stub"

    assert configured_provider_names(Explicit()) == ["stub"]
    providers = configured_providers(Explicit())
    assert [p.name for p in providers] == ["stub"]


def test_configured_providers_skips_one_that_refuses_to_construct():
    """A vendor key present but blank must not take the whole calendar down."""

    class Settings:
        alpaca_api_key_id = "k"
        alpaca_api_secret_key = "s"
        massive_api_key = "  "  # whitespace: configured_provider_names says no
        event_calendar_providers = "massive_calendar,fed"

    names = [p.name for p in configured_providers(Settings())]
    assert names == ["fed"]  # massive_calendar refused; fed survived


def test_every_provider_reports_the_same_fixed_capability_key_set():
    """A provider that simply lacks a capability reports False, never omits it."""
    providers = [
        StubEventCalendarProvider(),
        AlpacaCalendarProvider(
            "k", "s", transport=httpx.MockTransport(_status_handler(403))
        ),
        MassiveCalendarProvider(
            "m", transport=httpx.MockTransport(_status_handler(403))
        ),
        SecEdgarProvider(
            "ua/1.0 (x@y.z)", transport=httpx.MockTransport(_status_handler(403))
        ),
        FedProvider(transport=httpx.MockTransport(_status_handler(403))),
    ]
    for provider in providers:
        assert set(provider.capabilities()) == set(CAPABILITY_KEYS), provider.name


# ===========================================================================
# Alpaca calendar
# ===========================================================================


def _alpaca(handler, **kwargs) -> AlpacaCalendarProvider:
    return AlpacaCalendarProvider(
        api_key_id="test-key-id",
        api_secret_key="test-secret",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.parametrize("field", ["api_key_id", "api_secret_key"])
def test_alpaca_calendar_refuses_blank_credentials(field):
    kwargs = {"api_key_id": "k", "api_secret_key": "s", field: "  "}
    with pytest.raises(CalendarProviderError, match="ALPACA_API"):
        AlpacaCalendarProvider(**kwargs)


def test_alpaca_calendar_sends_auth_headers_and_never_echoes_the_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("APCA-API-KEY-ID")
        seen["secret"] = request.headers.get("APCA-API-SECRET-KEY")
        return httpx.Response(500, text="boom")

    provider = _alpaca(handler)
    with pytest.raises(CalendarProviderError) as exc:
        provider.fetch_market_calendar(date(2026, 1, 1), date(2026, 1, 5))
    assert seen == {"key": "test-key-id", "secret": "test-secret"}
    assert "test-key-id" not in str(exc.value)
    assert "test-secret" not in str(exc.value)


def test_alpaca_calendar_maps_sessions_to_utc_across_the_dst_boundary():
    """09:30 ET is 14:30Z in January and 13:30Z in July — the same wall clock,
    two different instants. Date-only logic would get one of them wrong."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/calendar"
        return httpx.Response(200, json=_fixture_json("alpaca_calendar.json"))

    days = _alpaca(handler).fetch_market_calendar(date(2026, 1, 1), date(2026, 12, 31))
    by_date = {d.session_date: d for d in days}

    winter = by_date[date(2026, 1, 2)]
    assert winter.open_utc == datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    assert winter.close_utc == datetime(2026, 1, 2, 21, 0, tzinfo=UTC)
    assert winter.is_early_close is False

    summer = by_date[date(2026, 7, 2)]
    assert summer.open_utc == datetime(2026, 7, 2, 13, 30, tzinfo=UTC)
    assert summer.close_utc == datetime(2026, 7, 2, 20, 0, tzinfo=UTC)


def test_alpaca_calendar_flags_early_closes_and_extended_hours():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture_json("alpaca_calendar.json"))

    days = {
        d.session_date: d
        for d in _alpaca(handler).fetch_market_calendar(
            date(2026, 1, 1), date(2026, 12, 31)
        )
    }
    half_day = days[date(2026, 11, 27)]
    assert half_day.is_early_close is True
    assert half_day.close_utc == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)  # 13:00 EST
    assert half_day.session_open_utc == datetime(2026, 11, 27, 9, 0, tzinfo=UTC)
    assert half_day.session_close_utc == datetime(2026, 11, 27, 22, 0, tzinfo=UTC)


def test_alpaca_calendar_skips_unparseable_rows_instead_of_inventing_a_session():
    """A fabricated 09:30-16:00 here would silently misclassify every earnings
    release on that day, which is worse than the day being absent."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture_json("alpaca_calendar.json"))

    days = _alpaca(handler).fetch_market_calendar(date(2026, 1, 1), date(2026, 12, 31))
    assert len(days) == 4  # 6 rows in, 2 unusable
    assert date(2026, 12, 24) not in {d.session_date for d in days}
    assert days == sorted(days, key=lambda d: d.session_date)


def test_alpaca_calendar_403_reports_market_calendar_false():
    report = _alpaca(_status_handler(403, "subscription required")).capabilities()
    assert report["market_calendar"] is False
    assert report["earnings_calendar"] is False  # Alpaca never offers it


def test_alpaca_calendar_500_reports_an_error_string_not_false():
    """A fault is 'availability unknown' — distinct from 403's proven absence."""
    report = _alpaca(_status_handler(500, "gateway blew up")).capabilities()
    assert isinstance(report["market_calendar"], str)
    assert "500" in report["market_calendar"]


def test_alpaca_calendar_200_reports_market_calendar_true():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    assert _alpaca(handler).capabilities()["market_calendar"] is True


def test_alpaca_calendar_emits_no_discrete_events():
    """Holidays reach the platform from Massive; synthesising them from gaps
    here would invent names the source never sent."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture_json("alpaca_calendar.json"))

    assert _alpaca(handler).fetch_events(
        tickers=["NVDA"], start=WIDE_START, end=WIDE_END
    ) == []


# ===========================================================================
# Massive calendar (holidays entitled; Benzinga earnings 403 TODAY)
# ===========================================================================


def _massive(handler, **kwargs) -> MassiveCalendarProvider:
    return MassiveCalendarProvider(
        api_key="test-massive-key", transport=httpx.MockTransport(handler), **kwargs
    )


def test_massive_calendar_refuses_a_blank_key():
    with pytest.raises(CalendarProviderError, match="MASSIVE_API_KEY"):
        MassiveCalendarProvider(api_key="   ")


def test_massive_holidays_become_confirmed_market_holiday_candidates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/marketstatus/upcoming"
        assert request.headers.get("Authorization") == "Bearer test-massive-key"
        return httpx.Response(200, json=_fixture_json("massive_holidays.json"))

    candidates = _massive(handler)._fetch_holidays(WIDE_START, WIDE_END)
    keys = {c.event_key for c in candidates}
    # Per-exchange rows: NYSE and NASDAQ are separate events for one day.
    assert "MARKET_HOLIDAY:NYSE:2026-11-26" in keys
    assert "MARKET_HOLIDAY:NASDAQ:2026-11-26" in keys
    assert len(candidates) == 5  # the "garbage" date row is dropped
    for candidate in candidates:
        assert candidate.event_type is EventType.MARKET_HOLIDAY
        assert candidate.status is EventStatus.CONFIRMED
        assert candidate.source is EventSourceKind.STRUCTURED_PROVIDER
        assert candidate.scheduled_at.tzinfo is not None
    early = next(c for c in candidates if c.event_key.endswith("2026-11-27"))
    assert "early-close" in early.title


def test_massive_holidays_respect_the_requested_window():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture_json("massive_holidays.json"))

    candidates = _massive(handler)._fetch_holidays(
        datetime(2026, 12, 1, tzinfo=UTC), datetime(2026, 12, 31, tzinfo=UTC)
    )
    assert [c.event_key for c in candidates] == ["MARKET_HOLIDAY:NYSE:2026-12-25"]


def test_massive_benzinga_earnings_403_is_capability_false_not_a_crash():
    """The LIVE state on 2026-08-19 (audit §13): Benzinga is 403 across the
    board. It must land as a capability verdict the UI can explain."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/benzinga"):
            return httpx.Response(403, text="subscription does not include benzinga")
        return httpx.Response(200, json=[])

    report = _massive(handler).capabilities()
    assert report["earnings_calendar"] is False
    assert report["market_holidays"] is True


def test_massive_earnings_403_yields_no_events_and_never_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/benzinga"):
            return httpx.Response(403, text="denied")
        return httpx.Response(200, json=_fixture_json("massive_holidays.json"))

    candidates = _massive(handler).fetch_events(
        tickers=["NVDA", "AAPL"], start=WIDE_START, end=WIDE_END
    )
    # Holidays still flow; not one fabricated earnings date appears.
    assert candidates
    assert all(c.event_type is EventType.MARKET_HOLIDAY for c in candidates)


def test_massive_earnings_map_when_the_add_on_is_entitled():
    """Live code, not a stub: an entitled 200 must produce real candidates,
    with date_confirmed deciding CONFIRMED vs ESTIMATED."""
    rows = [
        {"id": 11, "ticker": "NVDA", "date": "2026-08-27", "time": "amc",
         "date_confirmed": True, "period": 2, "period_year": 2027},
        {"id": 12, "ticker": "NVDA", "date": "2026-11-19", "time": "amc",
         "date_confirmed": False, "period": 3, "period_year": 2027},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/benzinga"):
            return httpx.Response(200, json=rows)
        return httpx.Response(200, json=[])

    candidates = _massive(handler)._fetch_earnings(["NVDA"], WIDE_START, WIDE_END)
    assert [c.status for c in candidates] == [
        EventStatus.CONFIRMED,
        EventStatus.ESTIMATED,
    ]
    confirmed = candidates[0]
    assert confirmed.event_key == "EARNINGS:NVDA:2026-08-27"
    assert confirmed.session is EventSession.AFTER_MARKET
    assert confirmed.fiscal_quarter == 2 and confirmed.fiscal_year == 2027
    # 16:05 ET in August is EDT -> 20:05Z
    assert confirmed.scheduled_at == datetime(2026, 8, 27, 20, 5, tzinfo=UTC)


def test_massive_401_falls_back_to_the_apikey_query_parameter():
    """The Polygon-heritage fallback MassiveProvider._request performs."""
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization"):
            attempts.append("header")
            return httpx.Response(401, text="bad header auth")
        attempts.append("query")
        assert request.url.params.get("apiKey") == "test-massive-key"
        return httpx.Response(200, json=[])

    _massive(handler)._request("/v1/marketstatus/upcoming")
    assert attempts == ["header", "query"]


def test_massive_500_reports_an_error_string():
    report = _massive(_status_handler(500, "upstream failure")).capabilities()
    assert isinstance(report["market_holidays"], str)
    assert "500" in report["market_holidays"]


def test_massive_refuses_to_serve_a_session_table():
    """Returning [] would read as 'the exchange never opens'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with pytest.raises(CapabilityNotAvailable, match="alpaca_calendar"):
        _massive(handler).fetch_market_calendar(date(2026, 1, 1), date(2026, 1, 5))


# ===========================================================================
# SEC EDGAR
# ===========================================================================


def _sec_handler(
    *,
    submissions_status: int = 200,
    tickers_status: int = 200,
    submissions: object | None = None,
):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert request.headers.get("User-Agent")  # SEC 403s without one
        if path == "/files/company_tickers.json":
            if tickers_status != 200:
                return httpx.Response(tickers_status, text="denied")
            return httpx.Response(200, json=_fixture_json("sec_company_tickers.json"))
        if path.startswith("/submissions/"):
            if submissions_status != 200:
                return httpx.Response(submissions_status, text="denied")
            if "older" in path or "-submissions-" in path:
                return httpx.Response(
                    200, json=_fixture_json("sec_submissions_nvda_older.json")
                )
            return httpx.Response(
                200,
                json=submissions
                if submissions is not None
                else _fixture_json("sec_submissions_nvda.json"),
            )
        return httpx.Response(404, text="not found")

    return handler


def _sec(handler=None, **kwargs) -> SecEdgarProvider:
    return SecEdgarProvider(
        user_agent="trading-system-with-ai/0.1 (test@example.com)",
        transport=httpx.MockTransport(handler or _sec_handler()),
        min_request_interval_seconds=0.0,
        **kwargs,
    )


def test_sec_requires_a_contact_user_agent():
    """A silent default would get the platform's IP throttled, and the
    failure would look like an outage."""
    with pytest.raises(CalendarProviderError, match="User-Agent"):
        SecEdgarProvider(user_agent="   ")


def test_sec_resolves_cik_and_caches_it_for_the_process():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _sec_handler()(request)

    provider = _sec(handler)
    assert provider.resolve_cik("nvda") == "0001045810"
    assert provider.resolve_cik("AAPL") == "0000320193"
    assert calls.count("/files/company_tickers.json") == 1  # cached
    assert provider.resolve_cik("NOTLISTED") is None


def test_sec_caller_supplied_cik_map_skips_the_lookup_entirely():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _sec_handler()(request)

    provider = _sec(handler, cik_map={"NVDA": "1045810"})
    assert provider.resolve_cik("NVDA") == "0001045810"
    assert "/files/company_tickers.json" not in calls


def test_sec_selects_only_8k_item_2_02_filings():
    """10-Q, an 8-K without Item 2.02, and the 8-K/A AMENDMENT (which would
    double-count one quarter) are all excluded."""
    events = _sec().fetch_earnings_history("NVDA")
    accessions = [c.source_event_id for c in events]
    assert "0001045810-26-000200" in accessions        # 8-K item 2.02
    assert "0001045810-26-000201" not in accessions    # 10-Q
    assert "0001045810-26-000180" not in accessions    # 8-K item 5.02
    assert "0001045810-26-000150" not in accessions    # 8-K/A amendment
    assert "0001045810-25-000090" not in accessions    # bad acceptance instant


def test_sec_earnings_candidate_fields_and_archive_url():
    events = _sec().fetch_earnings_history("NVDA")
    latest = events[0]
    assert latest.event_type is EventType.EARNINGS
    assert latest.status is EventStatus.CONFIRMED
    assert latest.source is EventSourceKind.COMPANY_IR_SEC
    assert latest.source_name == "sec_edgar"
    assert latest.ticker == "NVDA"
    assert latest.company_id == "0001045810"
    assert latest.title == "NVDA earnings release (8-K Item 2.02)"
    assert latest.source_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000200/nvda-20260827.htm"
    )
    assert latest.event_key == "EARNINGS:NVDA:2026-08-27"
    assert latest.scheduled_at == datetime(2026, 8, 27, 20, 20, 15, tzinfo=UTC)


def test_sec_uses_acceptance_datetime_not_filing_date_for_the_session():
    """20:20Z in August is 16:20 EDT — AFTER the close. filingDate is a bare
    date and would have classified this as DURING_MARKET."""
    events = _sec().fetch_earnings_history("NVDA")
    assert events[0].session is EventSession.AFTER_MARKET
    # The February release is 21:20Z = 16:20 EST, also after the close.
    february = next(c for c in events if c.event_key.endswith("2026-02-26"))
    assert february.session is EventSession.AFTER_MARKET


def test_sec_history_is_newest_first_and_respects_the_limit():
    events = _sec().fetch_earnings_history("NVDA", limit=3)
    assert len(events) == 3
    assert [c.scheduled_at for c in events] == sorted(
        [c.scheduled_at for c in events], reverse=True
    )


def test_sec_as_of_drops_filings_accepted_after_it():
    """§96 look-ahead: at as_of=T the platform cannot know about a release
    that happened at T+1h."""
    as_of = datetime(2026, 5, 28, 20, 0, tzinfo=UTC)  # 20 min BEFORE the May release
    events = _sec().fetch_earnings_history("NVDA", as_of=as_of)
    assert all(c.scheduled_at <= as_of for c in events)
    assert "EARNINGS:NVDA:2026-08-27" not in {c.event_key for c in events}
    assert "EARNINGS:NVDA:2026-05-28" not in {c.event_key for c in events}
    assert "EARNINGS:NVDA:2026-02-26" in {c.event_key for c in events}


def test_sec_as_of_boundary_is_inclusive_of_the_exact_instant():
    as_of = datetime(2026, 5, 28, 20, 20, 10, tzinfo=UTC)  # the May release instant
    keys = {c.event_key for c in _sec().fetch_earnings_history("NVDA", as_of=as_of)}
    assert "EARNINGS:NVDA:2026-05-28" in keys


def test_sec_pages_into_older_filings_only_when_recent_is_short():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return _sec_handler()(request)

    provider = _sec(handler)
    provider.fetch_earnings_history("NVDA", limit=3)
    assert not any("-submissions-" in p for p in paths)  # recent sufficed

    paths.clear()
    events = provider.fetch_earnings_history("NVDA", limit=8)
    assert any("-submissions-" in p for p in paths)  # older page fetched
    assert len(events) == 8


def test_sec_403_reports_earnings_history_false_and_earnings_calendar_false():
    """EDGAR has no upcoming-earnings feed at all — the distinction the whole
    fallback chain rests on."""
    report = _sec(_sec_handler(submissions_status=403)).capabilities()
    assert report["earnings_history"] is False
    assert report["earnings_calendar"] is False


def test_sec_500_reports_an_error_string():
    report = _sec(_sec_handler(submissions_status=500)).capabilities()
    assert isinstance(report["earnings_history"], str)
    assert "500" in report["earnings_history"]


def test_sec_200_reports_earnings_history_true():
    assert _sec().capabilities()["earnings_history"] is True


def test_sec_429_rate_limit_is_an_error_not_a_crash():
    with pytest.raises(CalendarProviderError, match="429"):
        _sec(_sec_handler(submissions_status=429)).fetch_earnings_history("NVDA")


def test_sec_fetch_events_survives_one_ticker_failing():
    """§8: one source failing must not cost the tick every other ticker."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "0000320193" in request.url.path:  # AAPL
            return httpx.Response(500, text="boom")
        return _sec_handler()(request)

    events = _sec(handler).fetch_events(
        tickers=["AAPL", "NVDA"], start=WIDE_START, end=WIDE_END
    )
    assert {c.ticker for c in events} == {"NVDA"}


def test_sec_fetch_events_includes_the_estimate_only_inside_the_window():
    provider = _sec()
    narrow = provider.fetch_events(
        tickers=["NVDA"],
        start=datetime(2026, 8, 1, tzinfo=UTC),
        end=datetime(2026, 8, 31, tzinfo=UTC),
    )
    assert all(c.status is EventStatus.CONFIRMED for c in narrow)

    with_next = provider.fetch_events(
        tickers=["NVDA"],
        start=datetime(2026, 8, 1, tzinfo=UTC),
        end=datetime(2026, 8, 31, tzinfo=UTC),
        include_next=True,
    )
    estimated = [c for c in with_next if c.status is EventStatus.ESTIMATED]
    assert len(estimated) == 1
    assert estimated[0].source is EventSourceKind.DERIVED


def test_sec_refuses_to_serve_a_session_table():
    with pytest.raises(CapabilityNotAvailable, match="alpaca_calendar"):
        _sec().fetch_market_calendar(date(2026, 1, 1), date(2026, 1, 5))


# ===========================================================================
# estimate_next_earnings (pure; audit §6 step 3)
# ===========================================================================


def _past(day: str, session: EventSession = EventSession.AFTER_MARKET) -> EventCandidate:
    """One past AMC release at 16:05 ET on `day`."""
    d = date.fromisoformat(day)
    hour = {
        EventSession.AFTER_MARKET: (20, 5),
        EventSession.BEFORE_MARKET: (11, 0),
        EventSession.DURING_MARKET: (16, 0),
    }[session]
    return EventCandidate(
        event_key=f"EARNINGS:NVDA:{day}",
        event_type=EventType.EARNINGS,
        title="past",
        scheduled_at=datetime(d.year, d.month, d.day, hour[0], hour[1], tzinfo=UTC),
        status=EventStatus.CONFIRMED,
        source=EventSourceKind.COMPANY_IR_SEC,
        source_name="sec_edgar",
        ticker="NVDA",
        session=session,
    )


def test_estimate_returns_none_with_no_history():
    assert estimate_next_earnings([]) is None


def test_estimate_returns_none_with_one_observation():
    """One release says nothing about cadence; a wrong estimate is worse than
    an honest absence (§11: do not fabricate a date)."""
    assert estimate_next_earnings([_past("2026-08-27")]) is None


def test_estimate_uses_the_median_gap_with_two_or_three_observations():
    history = [_past("2026-02-26"), _past("2026-05-28")]  # 91-day gap
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.scheduled_at.date() == date(2026, 8, 27)
    assert estimate.status is EventStatus.ESTIMATED
    assert estimate.source is EventSourceKind.DERIVED
    assert estimate.source_name == "derived_cadence"
    assert estimate.raw["method"] == "median_gap"
    assert "estimated from filing cadence" in estimate.title


def test_estimate_median_ignores_one_delayed_quarter():
    """The MEDIAN, not the mean: a single 140-day gap must not drag the
    estimate 15 days out."""
    history = [
        _past("2025-08-27"),
        _past("2026-01-14"),  # a 140-day outlier gap
        _past("2026-02-26"),
        _past("2026-05-28"),
    ]
    estimate = estimate_next_earnings(history, ticker="NVDA")
    assert estimate is not None
    # Year-ago anchor (2025-08-27 + 364d) = 2026-08-26, a Wednesday.
    assert estimate.scheduled_at.date() == date(2026, 8, 26)


def test_estimate_prefers_the_same_quarter_last_year_anchor_at_four_events():
    """52 weeks preserves the WEEKDAY, which is how issuers actually schedule."""
    history = [
        _past("2025-08-27"),  # Wednesday
        _past("2025-11-19"),
        _past("2026-02-26"),
        _past("2026-05-28"),
    ]
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.raw["method"] == "same_quarter_last_year+364d"
    assert estimate.scheduled_at.date() == date(2026, 8, 26)
    assert estimate.scheduled_at.date().weekday() == date(2025, 8, 27).weekday()


def test_estimate_snaps_a_weekend_result_forward_to_monday():
    """Issuers do not report at weekends."""
    history = [_past("2026-02-28"), _past("2026-05-30")]  # +91d = Sat 2026-08-29
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.scheduled_at.date() == date(2026, 8, 31)  # Monday
    assert estimate.scheduled_at.date().weekday() == 0


def test_estimate_uses_the_modal_session_anchor_time():
    """A session claim, not a minute: BMO history anchors at 07:00 ET."""
    history = [
        _past("2026-02-26", EventSession.BEFORE_MARKET),
        _past("2026-05-28", EventSession.BEFORE_MARKET),
    ]
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.session is EventSession.BEFORE_MARKET
    # 07:00 ET in August is EDT -> 11:00Z
    assert estimate.scheduled_at == datetime(2026, 8, 27, 11, 0, tzinfo=UTC)


def test_estimate_modal_session_tie_breaks_toward_the_most_recent_habit():
    """An issuer that moved from BMO to AMC last year is estimated AMC."""
    history = [
        _past("2025-08-27", EventSession.BEFORE_MARKET),
        _past("2025-11-19", EventSession.BEFORE_MARKET),
        _past("2026-02-26", EventSession.AFTER_MARKET),
        _past("2026-05-28", EventSession.AFTER_MARKET),
    ]
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.session is EventSession.AFTER_MARKET


def test_estimate_falls_back_when_the_year_ago_anchor_is_already_past():
    """An issuer whose year-ago anchor cannot be the NEXT event falls back to
    the gap median rather than emitting a date in the past.

    The history below has NO release after the last one's year-ago twin
    (2026-05-28), so the anchor search comes back empty and the median gap
    takes over — the guard this test exists for.
    """
    history = [
        _past("2025-08-27"),
        _past("2025-11-19"),
        _past("2026-02-26"),
        _past("2026-05-28"),
        _past("2027-05-27"),  # a year-long gap: nothing follows the twin
    ]
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.raw["method"] == "median_gap"
    assert estimate.scheduled_at.date() > date(2027, 5, 27)


def test_estimate_survives_an_issuer_that_files_five_times_a_year():
    """THE HPE BUG (found live 2026-08-22, fixed here).

    HPE's real record carries BOTH 2025-09-03 and 2025-10-15, so counting
    back four positions landed on October and the estimate skipped the
    imminent September report — the platform said 2026-10-14 while HPE's own
    IR site said 2026-09-02. Anchoring by calendar proximity instead of by
    position puts it back inside the near-term window, which is the side of
    the error that actually matters: a hidden earnings date is worse than a
    slightly wrong one.
    """
    history = [
        _past("2025-03-06"),
        _past("2025-06-03"),
        _past("2025-09-03"),
        _past("2025-10-15"),  # the extra filing that broke past[-4]
        _past("2025-12-04"),
        _past("2026-03-09"),
        _past("2026-06-01"),
    ]
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.raw["method"] == "same_quarter_last_year+364d"
    # Within a day of HPE's confirmed 2026-09-02 — and decisively NOT the
    # old October answer.
    assert date(2026, 9, 1) <= estimate.scheduled_at.date() <= date(2026, 9, 3)


def test_estimate_is_always_estimated_and_never_confirmable_here():
    history = [_past("2026-02-26"), _past("2026-05-28"), _past("2026-08-27")]
    estimate = estimate_next_earnings(history)
    assert estimate is not None
    assert estimate.status is EventStatus.ESTIMATED
    assert estimate.source is EventSourceKind.DERIVED
    assert estimate.raw["history_size"] == 3


def test_estimate_from_the_real_sec_fixture_history():
    """End to end: fixture -> parsed history -> estimate, no network."""
    history = _sec().fetch_earnings_history("NVDA")
    estimate = estimate_next_earnings(history, ticker="NVDA")
    assert estimate is not None
    assert estimate.ticker == "NVDA"
    assert estimate.session is EventSession.AFTER_MARKET
    assert estimate.scheduled_at > history[0].scheduled_at


# ===========================================================================
# Federal Reserve
# ===========================================================================


def _fed_handler(
    *, fomc_status: int = 200, rss_status: int = 200, fomc_html: str | None = None
):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/monetarypolicy/fomccalendars.htm":
            if fomc_status != 200:
                return httpx.Response(fomc_status, text="denied")
            return httpx.Response(
                200,
                text=fomc_html if fomc_html is not None
                else _fixture("fomc_calendar.html"),
            )
        if path == "/feeds/speeches.xml":
            if rss_status != 200:
                return httpx.Response(rss_status, text="denied")
            return httpx.Response(200, text=_fixture("fed_speeches.xml"))
        return httpx.Response(404, text="not found")

    return handler


def _fed(handler=None) -> FedProvider:
    return FedProvider(transport=httpx.MockTransport(handler or _fed_handler()))


def test_fomc_html_parse_handles_two_day_meetings_sep_and_month_crossings():
    meetings = parse_fomc_calendar(_fixture("fomc_calendar.html"))
    spans = [(m["start"], m["end"], m["has_sep"]) for m in meetings]
    assert (date(2026, 1, 27), date(2026, 1, 28), False) in spans
    assert (date(2026, 3, 17), date(2026, 3, 18), True) in spans      # SEP asterisk
    assert (date(2026, 4, 28), date(2026, 5, 1), False) in spans      # Apr/May 28-1
    assert (date(2027, 1, 26), date(2027, 1, 27), False) in spans     # second year panel
    assert len(meetings) == 6


def test_fomc_html_parse_reads_the_released_minutes_date():
    meetings = {m["start"]: m for m in parse_fomc_calendar(_fixture("fomc_calendar.html"))}
    assert meetings[date(2026, 1, 27)]["minutes_date"] == date(2026, 2, 18)
    assert meetings[date(2026, 4, 28)]["minutes_date"] == date(2026, 5, 20)
    assert meetings[date(2026, 6, 16)]["minutes_date"] is None  # not released yet


def test_fomc_html_parse_skips_unscheduled_and_notation_vote_rows():
    """Not part of the forward calendar; dating them would be a guess."""
    meetings = parse_fomc_calendar(_fixture("fomc_calendar.html"))
    assert date(2026, 3, 2) not in {m["start"] for m in meetings}


def test_fomc_html_parse_fails_empty_on_an_unrecognised_layout():
    """Audit §6: a changed layout must yield zero rows plus a warning, never a
    partially-parsed meeting at the wrong time."""
    assert parse_fomc_calendar("<html><body><p>redesigned</p></body></html>") == []
    assert parse_fomc_calendar("") == []


def test_fomc_emits_four_distinct_typed_events_per_meeting():
    """Spec §9: 'Do not label every Fed event as an FOMC meeting.'"""
    events = _fed().fetch_fomc_events(WIDE_START, WIDE_END)
    # The Jan 27-28 meeting: the minutes carry their OWN release date
    # (2026-02-18), which is why they are matched by raw["meeting_end"] and
    # not by a date substring of the key.
    january = [c for c in events if c.raw.get("meeting_end") == "2026-01-28"
               or c.raw.get("start") == "2026-01-27"]
    assert {c.event_type for c in january} == {
        EventType.FOMC_MEETING,
        EventType.FOMC_DECISION,
        EventType.FOMC_PRESS_CONFERENCE,
        EventType.FOMC_MINUTES,
    }
    keys = {c.event_key for c in events}
    assert "FOMC_MEETING:2026-01-27" in keys       # start day
    assert "FOMC_DECISION:2026-01-28" in keys      # LAST day
    assert "FOMC_PRESS_CONFERENCE:2026-01-28" in keys
    assert "FOMC_MINUTES:2026-02-18" in keys       # its own release date


def test_fomc_decision_time_is_correct_on_both_sides_of_dst():
    """14:00 ET is 19:00Z in January (EST) and 18:00Z in June (EDT)."""
    events = _fed().fetch_fomc_events(WIDE_START, WIDE_END)
    by_key = {c.event_key: c for c in events}
    assert by_key["FOMC_DECISION:2026-01-28"].scheduled_at == datetime(
        2026, 1, 28, 19, 0, tzinfo=UTC
    )
    assert by_key["FOMC_DECISION:2026-06-17"].scheduled_at == datetime(
        2026, 6, 17, 18, 0, tzinfo=UTC
    )
    assert by_key["FOMC_PRESS_CONFERENCE:2026-06-17"].scheduled_at == datetime(
        2026, 6, 17, 18, 30, tzinfo=UTC
    )
    for candidate in events:
        assert candidate.event_timezone == "America/New_York"


def test_fomc_minutes_are_confirmed_when_released_and_estimated_otherwise():
    events = _fed().fetch_fomc_events(WIDE_START, WIDE_END)
    by_key = {c.event_key: c for c in events}
    released = by_key["FOMC_MINUTES:2026-02-18"]
    assert released.status is EventStatus.CONFIRMED
    assert released.source is EventSourceKind.FEDERAL_RESERVE

    # June's minutes are not released on the page -> decision + 21 days.
    estimated = by_key["FOMC_MINUTES:2026-07-08"]
    assert estimated.status is EventStatus.ESTIMATED
    assert estimated.source is EventSourceKind.DERIVED
    assert estimated.source_name == "derived_cadence"
    assert "estimated" in estimated.title.lower()


def test_fomc_month_crossing_meeting_decides_on_the_second_month():
    events = _fed().fetch_fomc_events(WIDE_START, WIDE_END)
    by_key = {c.event_key: c for c in events}
    assert by_key["FOMC_MEETING:2026-04-28"].event_type is EventType.FOMC_MEETING
    assert "FOMC_DECISION:2026-05-01" in by_key
    assert by_key["FOMC_DECISION:2026-05-01"].scheduled_at == datetime(
        2026, 5, 1, 18, 0, tzinfo=UTC
    )


def test_fomc_sep_meetings_are_labelled_in_the_title():
    events = _fed().fetch_fomc_events(WIDE_START, WIDE_END)
    by_key = {c.event_key: c for c in events}
    assert "Summary of Economic Projections" in by_key["FOMC_DECISION:2026-03-18"].title
    assert "Summary of Economic Projections" not in by_key["FOMC_DECISION:2026-01-28"].title


def test_fomc_events_are_filtered_to_the_requested_window():
    events = _fed().fetch_fomc_events(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    )
    assert {c.event_key for c in events} == {
        "FOMC_MEETING:2026-01-27",
        "FOMC_DECISION:2026-01-28",
        "FOMC_PRESS_CONFERENCE:2026-01-28",
    }  # the Feb 18 minutes fall outside


def test_fed_speech_rss_parses_speaker_and_topic_from_the_title():
    items = parse_speeches_rss(_fixture("fed_speeches.xml"))
    assert items[0]["speaker"] == "Chair Jerome H. Powell"
    assert items[0]["topic"] == "Economic Outlook and Monetary Policy"
    assert items[1]["speaker"] == "Vice Chair Philip N. Jefferson"
    # 10:00 -0400 -> 14:00Z
    assert items[0]["published_at"] == datetime(2026, 8, 21, 14, 0, tzinfo=UTC)


def test_fed_speech_rss_drops_items_with_no_pubdate():
    """A speech with no official timestamp cannot be placed on a timeline."""
    items = parse_speeches_rss(_fixture("fed_speeches.xml"))
    assert len(items) == 4  # 5 items in, the undated one dropped
    assert all(i["published_at"] is not None for i in items)
    # The colon-less item has no name in its title; the speaker falls back to
    # the speech URL slug (".../speech/other20260903a.htm" -> "Other") — the
    # live feed (2026-08-19) carries no author element, so the URL is the
    # last deterministic hint available.
    colonless = [i for i in items if i["title"].startswith("An item with no colon")]
    assert len(colonless) == 1
    assert colonless[0]["speaker"] == "Other"
    assert colonless[0]["topic"] == colonless[0]["title"]


def test_fed_speech_rss_parses_the_live_surname_comma_title_format():
    """Verified against the live feed on 2026-08-19: titles read
    "Cook, Outlook for the U.S. and Alaskan Economies" (surname, comma, topic)
    with no author element; a colon inside the topic must not steal the split."""
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel>'
        "<title>FRB: Speeches</title>"
        "<item><title>Cook, Outlook for the U.S. and Alaskan Economies</title>"
        "<link>https://www.federalreserve.gov/newsevents/speech/cook20260805a.htm</link>"
        "<pubDate>Wed, 5 Aug 2026 20:05:00 GMT</pubDate></item>"
        "<item><title>Jefferson, Navigating Economic Shocks: A Monetary Policymaker's Perspective</title>"
        "<link>https://www.federalreserve.gov/newsevents/speech/jefferson20260716a.htm</link>"
        "<pubDate>Thu, 16 Jul 2026 23:00:00 GMT</pubDate></item>"
        "</channel></rss>"
    )
    items = parse_speeches_rss(xml)
    assert [i["speaker"] for i in items] == ["Cook", "Jefferson"]
    assert items[0]["topic"] == "Outlook for the U.S. and Alaskan Economies"
    assert items[1]["topic"] == "Navigating Economic Shocks: A Monetary Policymaker's Perspective"
    assert items[0]["published_at"] == datetime(2026, 8, 5, 20, 5, tzinfo=UTC)


def test_fed_speech_rss_fails_empty_on_broken_xml():
    assert parse_speeches_rss("<rss><channel><item>") == []
    assert parse_speeches_rss("") == []


def test_fed_speech_candidates_carry_speaker_topic_and_session():
    events = _fed().fetch_speeches(WIDE_START, WIDE_END)
    powell = next(c for c in events if "powell" in c.event_key)
    assert powell.event_type is EventType.FED_SPEECH
    assert powell.status is EventStatus.CONFIRMED
    assert powell.source is EventSourceKind.FEDERAL_RESERVE
    assert powell.source_name == "fed_rss"
    assert powell.speaker == "Chair Jerome H. Powell"
    assert powell.topic == "Economic Outlook and Monetary Policy"
    assert powell.session is EventSession.DURING_MARKET  # 10:00 ET
    assert powell.source_url.endswith("powell20260821a.htm")
    assert powell.event_key.startswith("FED_SPEECH:2026-08-21:chair-jerome-h-powell:")
    assert powell.raw["senior_speaker"] is True


def test_fed_speeches_respect_as_of_look_ahead():
    """§96: an RSS item published after as_of was not knowable at as_of."""
    as_of = datetime(2026, 8, 25, tzinfo=UTC)
    events = _fed().fetch_speeches(WIDE_START, WIDE_END, as_of=as_of)
    assert [c.speaker for c in events] == ["Chair Jerome H. Powell"]
    assert all(c.scheduled_at <= as_of for c in events)


def test_is_senior_speaker_flags_chair_and_vice_chair_only():
    assert is_senior_speaker("Chair Jerome H. Powell") is True
    assert is_senior_speaker("Vice Chair Philip N. Jefferson") is True
    assert is_senior_speaker("Governor Michelle W. Bowman") is False
    assert is_senior_speaker(None) is False


def test_fed_403_reports_fed_events_false():
    report = _fed(_fed_handler(fomc_status=403)).capabilities()
    assert report["fed_events"] is False


def test_fed_500_reports_an_error_string():
    report = _fed(_fed_handler(fomc_status=500)).capabilities()
    assert isinstance(report["fed_events"], str)
    assert "500" in report["fed_events"]


def test_fed_200_with_an_unparseable_page_is_an_error_string_not_true():
    """HTTP succeeded but the layout changed — that is an availability
    problem, and reporting True would hide a silently empty calendar."""
    report = _fed(_fed_handler(fomc_html="<html><body>redesign</body></html>")).capabilities()
    assert isinstance(report["fed_events"], str)
    assert "layout" in report["fed_events"]


def test_fed_200_and_parseable_reports_fed_events_true():
    assert _fed().capabilities()["fed_events"] is True


def test_fed_fetch_events_survives_a_dead_rss_feed():
    """§8: a dead speeches feed must not cost the platform the FOMC calendar."""
    events = _fed(_fed_handler(rss_status=500)).fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    assert events
    assert all(c.event_type is not EventType.FED_SPEECH for c in events)


def test_fed_fetch_events_survives_a_dead_fomc_page():
    events = _fed(_fed_handler(fomc_status=500)).fetch_events(
        tickers=[], start=WIDE_START, end=WIDE_END
    )
    assert events
    assert all(c.event_type is EventType.FED_SPEECH for c in events)


def test_fed_refuses_to_serve_a_session_table():
    with pytest.raises(CapabilityNotAvailable, match="alpaca_calendar"):
        _fed().fetch_market_calendar(date(2026, 1, 1), date(2026, 1, 5))


# ===========================================================================
# Shared invariants
# ===========================================================================


def test_classify_session_et_buckets_by_eastern_wall_clock():
    assert classify_session_et(datetime(2026, 8, 27, 11, 0, tzinfo=UTC)) is (
        EventSession.BEFORE_MARKET  # 07:00 EDT
    )
    assert classify_session_et(datetime(2026, 8, 27, 16, 0, tzinfo=UTC)) is (
        EventSession.DURING_MARKET  # 12:00 EDT
    )
    assert classify_session_et(datetime(2026, 8, 27, 20, 5, tzinfo=UTC)) is (
        EventSession.AFTER_MARKET  # 16:05 EDT
    )
    # Exactly 09:30 ET is the open -> DURING; exactly 16:00 ET -> AFTER.
    assert classify_session_et(datetime(2026, 1, 28, 14, 30, tzinfo=UTC)) is (
        EventSession.DURING_MARKET
    )
    assert classify_session_et(datetime(2026, 1, 28, 21, 0, tzinfo=UTC)) is (
        EventSession.AFTER_MARKET
    )


def test_every_candidate_from_every_provider_is_utc_aware_and_keyed():
    """One invariant sweep: no naive timestamps, no blank keys, and the
    stored event timezone is always present (§10)."""
    fed_events = _fed().fetch_events(tickers=[], start=WIDE_START, end=WIDE_END)
    sec_events = _sec().fetch_events(
        tickers=["NVDA"], start=WIDE_START, end=WIDE_END, include_next=True
    )

    def massive_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/benzinga"):
            return httpx.Response(403, text="denied")
        return httpx.Response(200, json=_fixture_json("massive_holidays.json"))

    massive_events = _massive(massive_handler).fetch_events(
        tickers=["NVDA"], start=WIDE_START, end=WIDE_END
    )
    stub_events = StubEventCalendarProvider().fetch_events(
        tickers=["NVDA"],
        start=datetime(2026, 8, 19, tzinfo=UTC),
        end=datetime(2026, 10, 19, tzinfo=UTC),
    )

    all_events = fed_events + sec_events + massive_events + stub_events
    assert len(all_events) > 20
    for candidate in all_events:
        assert candidate.scheduled_at.tzinfo is not None, candidate.event_key
        assert candidate.scheduled_at.utcoffset() == timedelta(0), candidate.event_key
        assert candidate.event_key.strip()
        assert candidate.event_key.startswith(str(candidate.event_type))
        assert candidate.event_timezone == "America/New_York"
        assert candidate.title.strip()
        assert candidate.source_name.strip()


def test_estimated_candidates_always_carry_derived_provenance():
    """§11/§13: an estimate must be distinguishable from a fact at every
    layer, so the UI can never render one as the other."""
    sec_events = _sec().fetch_events(
        tickers=["NVDA"], start=WIDE_START, end=WIDE_END, include_next=True
    )
    fed_events = _fed().fetch_events(tickers=[], start=WIDE_START, end=WIDE_END)
    estimated = [
        c for c in sec_events + fed_events if c.status is EventStatus.ESTIMATED
    ]
    assert estimated
    for candidate in estimated:
        assert candidate.source is EventSourceKind.DERIVED
        assert candidate.source_name == "derived_cadence"


def test_stub_is_deterministic_and_labels_itself_synthetic():
    provider = StubEventCalendarProvider()
    window = (datetime(2026, 8, 19, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC))
    first = provider.fetch_events(tickers=["NVDA"], start=window[0], end=window[1])
    second = provider.fetch_events(tickers=["NVDA"], start=window[0], end=window[1])
    assert first == second
    assert all("SYNTHETIC" in c.title for c in first)
    assert provider.name == "stub"


def test_stub_market_calendar_is_weekdays_only():
    days = StubEventCalendarProvider().fetch_market_calendar(
        date(2026, 8, 17), date(2026, 8, 23)
    )
    assert [d.session_date.weekday() for d in days] == [0, 1, 2, 3, 4]


def test_fomc_calendar_parses_the_live_2026_markup():
    """Fixture is the 2026 + 2027 panels captured from federalreserve.gov on
    2026-08-19 (the layout the first live ingest failed on): month in
    ``fomc-meeting__month <strong>``, days in ``fomc-meeting__date`` with a
    ``*`` SEP marker, minutes as ``(Released Month DD, YYYY)``."""
    meetings = parse_fomc_calendar(_fixture("fomc_calendar_live_2026.html"))
    by_start = {m["start"]: m for m in meetings}
    assert date(2026, 9, 15) in by_start
    sep = by_start[date(2026, 9, 15)]
    assert sep["end"] == date(2026, 9, 16) and sep["has_sep"] is True
    jan = by_start[date(2026, 1, 27)]
    assert jan["end"] == date(2026, 1, 28) and jan["minutes_date"] == date(2026, 2, 18)
    assert by_start[date(2026, 7, 28)]["minutes_date"] is None  # not yet released
    assert len([m for m in meetings if m["year"] == 2026]) == 8
    assert len([m for m in meetings if m["year"] == 2027]) == 7  # page lists 7 of 8 for 2027
    assert all(m["end"] >= m["start"] for m in meetings)


def test_cluster_releases_keeps_the_earliest_202_filing_per_quarter():
    """Live 2026-08-19: SMCI furnished a second Item 2.02 8-K days after its
    release, which the registry's +-21d same_event window then read as a
    rescheduled CONFIRMED date (REVISED). The release is the FIRST 2.02 of
    the quarter; later ones inside the window are follow-ups."""
    release = _past("2026-05-05")
    follow_up = replace(_past("2026-05-12"), source_event_id="0001-follow-up")
    prior = _past("2026-02-03")
    out = cluster_releases([follow_up, release, prior])
    assert [c.scheduled_at.date().isoformat() for c in out] == ["2026-05-05", "2026-02-03"]
    assert out[0].raw["follow_ups"] == ["0001-follow-up"]
    assert "follow_ups" not in (out[1].raw or {})


def test_cluster_releases_does_not_merge_distinct_quarters():
    out = cluster_releases([_past("2026-02-03"), _past("2026-05-05"), _past("2026-08-04")])
    assert len(out) == 3


def test_estimate_rolls_a_past_year_ago_anchor_forward_to_the_future():
    """SMCI-shaped history: the same-quarter-last-year + 364d anchor lands
    before `now`, so the estimator must roll forward by the median gap
    instead of returning nothing (or, worse, a past 'upcoming' date)."""
    history = [
        _past("2025-05-06"),
        _past("2025-08-05"),
        _past("2025-11-04"),
        _past("2026-02-03"),
        _past("2026-05-05"),
    ]
    # year-ago anchor = 2025-08-05 + 364d = 2026-08-04, already behind `now`
    now = datetime(2026, 8, 19, 17, 0, tzinfo=UTC)
    est = estimate_next_earnings(history, ticker="SMCI", now=now)
    assert est is not None
    assert est.scheduled_at > now
    assert est.status is EventStatus.ESTIMATED
    assert "rolled_forward" in est.raw["method"]


# ---------------------------------------------------------------------------
# BEA archive — past GDP/PCE releases (found live 2026-08-22)
#
# BEA's /news/schedule page is FORWARD-LOOKING ONLY: it starts at today and
# never lists a release that already happened. So GDP and PCE had zero stored
# history and every §15 comparison answered "no comparable event" — true of
# the rows, false of the world, for a series published quarterly for decades.
# ---------------------------------------------------------------------------

_BEA_ARCHIVE_HTML = """
<table>
  <tr class="release-row">
    <td class="views-field views-field-title">
      <a href="/news/2026/gdp-advance">GDP (Advance Estimate), 1st Quarter 2026</a>
    </td>
    <td class="views-field views-field-created">
      <time datetime="2026-04-30T08:30:00-04:00">April 30, 2026</time>
    </td>
  </tr>
  <tr class="release-row">
    <td class="views-field views-field-title">
      <a href="/news/2026/pi">Personal Income and Outlays, March 2026</a>
    </td>
    <td class="views-field views-field-created">
      <time datetime="2026-04-30T08:30:00-04:00">April 30, 2026</time>
    </td>
  </tr>
  <tr class="release-row">
    <td class="views-field views-field-title">
      <a href="/news/2026/county">GDP by County, 2025</a>
    </td>
    <td class="views-field views-field-created">
      <time datetime="2026-04-15T08:30:00-04:00">April 15, 2026</time>
    </td>
  </tr>
</table>
"""


def test_bea_archive_parses_past_releases_with_their_real_instant():
    from libs.event_calendar.bea import parse_archive_page

    rows = parse_archive_page(_BEA_ARCHIVE_HTML)
    assert len(rows) == 3
    first = rows[0]
    assert "GDP (Advance Estimate)" in first["title"]
    # The archive carries a full ISO instant INCLUDING the 08:30 ET release
    # hour, so nothing is inferred: 08:30 EDT == 12:30 UTC.
    assert first["published_at"].isoformat() == "2026-04-30T12:30:00+00:00"


def test_bea_archive_row_without_a_parseable_instant_is_skipped():
    """No instant means no event — never a date filled in by guess."""
    from libs.event_calendar.bea import parse_archive_page

    broken = """
    <tr class="release-row">
      <td class="views-field views-field-title"><a href="/x">GDP (Advance Estimate), 1st Quarter 2026</a></td>
      <td class="views-field views-field-created"><time datetime="not-a-date">whenever</time></td>
    </tr>
    """
    assert parse_archive_page(broken) == []


def test_bea_archive_keeps_only_the_market_moving_families():
    """"GDP by County" is a regional product, not a catalyst — the existing
    title classifier must still gate archive rows."""
    from libs.event_calendar.bea import classify_release_title, parse_archive_page

    typed = [
        classify_release_title(r["title"]) for r in parse_archive_page(_BEA_ARCHIVE_HTML)
    ]
    assert typed[0] is EventType.GDP
    assert typed[1] is EventType.PCE
    assert typed[2] is None  # GDP by County


def test_bea_classifier_accepts_the_older_spelled_out_headline_form():
    """BEA renamed its headline: pre-2026 releases read "Gross Domestic
    Product, 3rd Quarter 2025 (Initial Estimate)". Matching only the modern
    "GDP (Advance Estimate)" form dropped 14 of 20 rows on the first archive
    page — which is how a series published since 1947 had no history."""
    from libs.event_calendar.bea import classify_release_title

    for title in (
        "Gross Domestic Product, 3rd Quarter 2025 (Initial Estimate) and Corporate Profits",
        "Gross Domestic Product, 3rd Quarter 2025 (Updated Estimate), GDP by Industry",
        "Gross Domestic Product, 2nd Quarter 2025 (Advance Estimate)",
        "Gross Domestic Product (Second Estimate), Corporate Profits (Preliminary)",
        "Gross Domestic Product, 4th Quarter and Year 2024 (Third Estimate)",
    ):
        assert classify_release_title(title) is EventType.GDP, title


def test_bea_classifier_still_refuses_the_regional_gdp_products():
    """Widening the headline pattern must not swallow "by State"/"by County"
    /"by Industry" — different products on a different cadence."""
    from libs.event_calendar.bea import classify_release_title

    for title in (
        "GDP by County, 2025",
        "GDP by State, 1st Quarter 2026",
        "Real GDP by Metropolitan Area, 2024",
        "GDP by Industry, 2025 (Advance Estimate)",
        "Personal Income by State, 2025",
    ):
        assert classify_release_title(title) is None, title


def test_bea_q4_release_period_survives_the_and_year_wording():
    """"4th Quarter and Year 2025" must yield 2025-Q4, not None. §15 rejects a
    candidate whose release_period EQUALS the event's — that guard is what
    stops the Second Estimate of a quarter comparing against the Advance
    Estimate of the SAME quarter, and with both NULL it cannot fire."""
    from libs.event_calendar.bea import release_period_code

    assert release_period_code("GDP (Advance Estimate), 4th Quarter and Year 2025") == "2025-Q4"
    assert release_period_code("GDP (Second Estimate), 4th Quarter and Year 2025") == "2025-Q4"
    assert release_period_code("GDP (Advance Estimate), 1st Quarter 2026") == "2026-Q1"


def test_bea_archive_covers_both_product_families():
    """The archive filter selects ONE product family, so the GDP id (451)
    carries no Personal Income release at all. Walking only 451 is why PCE had
    zero stored history while GDP had six."""
    from libs.event_calendar.bea import ARCHIVE_PATHS

    assert len(ARCHIVE_PATHS) >= 2
    assert any("451" in p for p in ARCHIVE_PATHS)  # Gross Domestic Product
    assert any("476" in p for p in ARCHIVE_PATHS)  # Personal Income (PCE)
