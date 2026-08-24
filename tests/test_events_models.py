"""Event Intelligence Core — pure domain (event spec §5-§7, §10-§13, §15,
§67, §78; Phase B unit U1).

Pins the four contracts the whole event pipeline rests on:

1. **Natural-key determinism** — the same real-world event, described by two
   different providers, must produce the same key or ingestion duplicates
   every card on every tick.
2. **Merge precedence (§78)** — a lower-authority source never overwrites a
   higher one, CONFIRMED never downgrades to ESTIMATED, a moved confirmed
   date becomes REVISED with history, and CANCELED only arrives explicitly.
3. **Time handling (§10)** — naive datetimes are refused, ET/UTC conversion
   is correct across DST, and the session split honours a half-day calendar
   row rather than a hardcoded 16:00 close.
4. **§13 transparency** — importance components add to the score and every
   point is attributable.

No I/O, no fixtures, no DB: this is the layer that must be provable by
arithmetic alone.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from libs.trading_core.events import (
    EARNINGS_DRIFT_WINDOW,
    RELEVANCE_TIERS,
    Event,
    EventCandidate,
    classify_session,
    default_relevance_tier,
    eastern_date,
    event_key,
    lifecycle,
    merge,
    previous_comparable,
    relevance_rank,
    same_event,
    score_importance,
    session_anchor_time,
    slug,
    source_rank,
    to_local,
)
from libs.trading_core.events.importance import BASE_IMPORTANCE, CHAIR_SPEAKER_BONUS
from libs.trading_core.events.taxonomy import require_utc
from libs.trading_core.models.enums import (
    EventLifecycle,
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

NOW = datetime(2026, 8, 19, 14, 0, tzinfo=UTC)


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _et(*args) -> datetime:
    return datetime(*args, tzinfo=ET)


def _event(**overrides) -> Event:
    """A CONFIRMED NVDA earnings row from SEC EDGAR — the baseline."""
    kwargs = dict(
        event_id=1,
        event_key="EARNINGS:NVDA:2026-08-26",
        event_type=EventType.EARNINGS,
        title="NVDA earnings release (8-K Item 2.02)",
        ticker="NVDA",
        scheduled_at=_et(2026, 8, 26, 16, 20).astimezone(UTC),
        status=EventStatus.CONFIRMED,
        source=EventSourceKind.COMPANY_IR_SEC,
        source_name="sec_edgar",
        session=EventSession.AFTER_MARKET,
        last_verified_at=_utc(2026, 8, 18, 12, 0),
        created_at=_utc(2026, 8, 1, 12, 0),
        updated_at=_utc(2026, 8, 18, 12, 0),
    )
    kwargs.update(overrides)
    return Event(**kwargs)


def _candidate(**overrides) -> EventCandidate:
    kwargs = dict(
        event_key="EARNINGS:NVDA:2026-08-26",
        event_type=EventType.EARNINGS,
        title="NVDA earnings",
        ticker="NVDA",
        scheduled_at=_et(2026, 8, 26, 16, 20).astimezone(UTC),
        status=EventStatus.CONFIRMED,
        source=EventSourceKind.STRUCTURED_PROVIDER,
        source_name="massive_calendar",
        session=EventSession.AFTER_MARKET,
    )
    kwargs.update(overrides)
    return EventCandidate(**kwargs)


# ---------------------------------------------------------------------------
# 1. Natural keys (§5, §6)
# ---------------------------------------------------------------------------


def test_earnings_key_uses_the_eastern_calendar_date_not_utc():
    """An AMC release at 20:15 ET is 00:15 UTC the NEXT day — keying on UTC
    would split one quarter's release across two cards."""
    amc = _et(2026, 8, 26, 20, 15).astimezone(UTC)
    assert amc.date().isoformat() == "2026-08-27"  # the trap
    assert event_key(EventType.EARNINGS, ticker="NVDA", scheduled_at=amc) == "EARNINGS:NVDA:2026-08-26"


def test_earnings_key_is_deterministic_and_normalises_the_ticker():
    moment = _et(2026, 8, 26, 16, 20).astimezone(UTC)
    first = event_key(EventType.EARNINGS, ticker=" nvda ", scheduled_at=moment)
    second = event_key(EventType.EARNINGS, ticker="NVDA", scheduled_at=moment)
    assert first == second == "EARNINGS:NVDA:2026-08-26"


def test_macro_key_is_the_release_period_not_the_date():
    """A CPI print that slips a day is still 'CPI for 2026-07' (§15)."""
    key_a = event_key(EventType.CPI, release_period="2026-07", scheduled_at=_utc(2026, 8, 12, 12, 30))
    key_b = event_key(EventType.CPI, release_period="2026-07", scheduled_at=_utc(2026, 8, 13, 12, 30))
    assert key_a == key_b == "CPI:2026-07"


def test_fomc_keys_are_typed_per_event_not_one_meeting_blob():
    """§9: do not label every Fed event an FOMC meeting."""
    day = _et(2026, 9, 17, 14, 0).astimezone(UTC)
    keys = {
        event_key(t, scheduled_at=day)
        for t in (
            EventType.FOMC_MEETING,
            EventType.FOMC_DECISION,
            EventType.FOMC_PRESS_CONFERENCE,
            EventType.FOMC_MINUTES,
        )
    }
    assert len(keys) == 4
    assert "FOMC_DECISION:2026-09-17" in keys


def test_fed_speech_key_includes_speaker_and_truncated_title_slug():
    key = event_key(
        EventType.FED_SPEECH,
        scheduled_at=_et(2026, 9, 4, 9, 0).astimezone(UTC),
        speaker="Jerome H. Powell",
        title="Monetary Policy and the Economic Outlook, With Some Further Remarks",
    )
    head, day, speaker, title_slug = key.split(":")
    assert (head, day, speaker) == ("FED_SPEECH", "2026-09-04", "jerome-h-powell")
    assert len(title_slug) <= 40


def test_holiday_and_corporate_keys_carry_their_discriminators():
    day = _et(2026, 11, 26, 0, 0).astimezone(UTC)
    assert event_key(EventType.MARKET_HOLIDAY, scheduled_at=day, exchange="us") == (
        "MARKET_HOLIDAY:US:2026-11-26"
    )
    assert event_key(
        EventType.CORPORATE_EVENT, ticker="AAPL", subtype="Ex-Dividend", scheduled_at=day
    ) == "CORPORATE_EVENT:AAPL:ex-dividend:2026-11-26"


def test_event_key_refuses_to_guess_missing_discriminators():
    with pytest.raises(ValueError):
        event_key(EventType.EARNINGS, scheduled_at=NOW)
    with pytest.raises(ValueError):
        event_key(EventType.CPI, scheduled_at=NOW)


def test_slug_collapses_punctuation_so_reworded_titles_do_not_duplicate():
    assert slug("Powell: Policy & the Outlook!") == "powell-policy-the-outlook"
    assert slug(None) == ""


# ---------------------------------------------------------------------------
# 2. Time handling (§10)
# ---------------------------------------------------------------------------


def test_naive_datetimes_are_refused_never_assumed_utc():
    with pytest.raises(ValueError):
        require_utc(datetime(2026, 8, 26, 16, 20))
    with pytest.raises(ValueError):
        Event(
            event_key="k",
            event_type=EventType.EARNINGS,
            title="t",
            scheduled_at=datetime(2026, 8, 26, 16, 20),
            status=EventStatus.CONFIRMED,
            source=EventSourceKind.USER,
            source_name="user",
        )


def test_to_local_and_eastern_date_are_correct_across_dst():
    """EDT is UTC-4, EST is UTC-5 — a fixed offset would break one of these."""
    summer = _utc(2026, 8, 26, 20, 20)
    winter = _utc(2026, 1, 28, 19, 0)
    assert to_local(summer).strftime("%Y-%m-%d %H:%M") == "2026-08-26 16:20"
    assert to_local(winter).strftime("%Y-%m-%d %H:%M") == "2026-01-28 14:00"
    assert eastern_date(summer).isoformat() == "2026-08-26"


def test_classify_session_default_regular_hours():
    assert classify_session(_et(2026, 8, 26, 7, 0).astimezone(UTC)) is EventSession.BEFORE_MARKET
    assert classify_session(_et(2026, 8, 26, 12, 0).astimezone(UTC)) is EventSession.DURING_MARKET
    assert classify_session(_et(2026, 8, 26, 16, 20).astimezone(UTC)) is EventSession.AFTER_MARKET


def test_classify_session_boundaries_are_open_inclusive_close_inclusive():
    assert classify_session(_et(2026, 8, 26, 9, 30).astimezone(UTC)) is EventSession.DURING_MARKET
    assert classify_session(_et(2026, 8, 26, 9, 29).astimezone(UTC)) is EventSession.BEFORE_MARKET
    assert classify_session(_et(2026, 8, 26, 16, 0).astimezone(UTC)) is EventSession.AFTER_MARKET


def test_classify_session_uses_a_half_day_calendar_row():
    """13:00 ET on a 13:00-close half day is AFTER_MARKET, not DURING."""
    moment = _et(2026, 11, 27, 13, 5).astimezone(UTC)
    assert classify_session(moment) is EventSession.DURING_MARKET  # regular-hours default
    early = classify_session(
        moment,
        _et(2026, 11, 27, 9, 30).astimezone(UTC),
        _et(2026, 11, 27, 13, 0).astimezone(UTC),
    )
    assert early is EventSession.AFTER_MARKET


def test_classify_session_rejects_a_half_specified_calendar_row():
    with pytest.raises(ValueError):
        classify_session(NOW, _utc(2026, 8, 19, 13, 30), None)
    with pytest.raises(ValueError):
        classify_session(NOW, _utc(2026, 8, 19, 20, 0), _utc(2026, 8, 19, 13, 30))


def test_session_anchor_times_pin_estimates_to_the_right_half_of_the_day():
    assert session_anchor_time(EventSession.BEFORE_MARKET).hour == 7
    assert session_anchor_time(EventSession.AFTER_MARKET) > session_anchor_time(
        EventSession.DURING_MARKET
    )
    assert session_anchor_time(EventSession.UNKNOWN).hour == 12


# ---------------------------------------------------------------------------
# 3. Lifecycle (§67)
# ---------------------------------------------------------------------------


def test_lifecycle_scheduled_beyond_seven_days():
    at = _utc(2026, 9, 10, 20, 20)
    assert lifecycle(at, at - timedelta(days=8)) is EventLifecycle.SCHEDULED


def test_lifecycle_pre_event_boundary_at_exactly_seven_days():
    at = _utc(2026, 9, 10, 20, 20)
    assert lifecycle(at, at - timedelta(days=7)) is EventLifecycle.PRE_EVENT
    assert lifecycle(at, at - timedelta(days=7, seconds=1)) is EventLifecycle.SCHEDULED


def test_lifecycle_live_window_is_five_minutes_before_to_sixty_after():
    at = _utc(2026, 9, 10, 20, 20)
    assert lifecycle(at, at - timedelta(minutes=5)) is EventLifecycle.LIVE
    assert lifecycle(at, at - timedelta(minutes=6)) is EventLifecycle.PRE_EVENT
    assert lifecycle(at, at) is EventLifecycle.LIVE
    assert lifecycle(at, at + timedelta(minutes=60)) is EventLifecycle.LIVE
    assert lifecycle(at, at + timedelta(minutes=61)) is EventLifecycle.POST_EVENT


def test_lifecycle_post_event_counts_trading_days_then_archives():
    """Thursday + 5 weekdays = the following Thursday; the weekend does not
    age an event out of POST_EVENT."""
    at = _et(2026, 9, 10, 16, 20).astimezone(UTC)  # a Thursday
    assert lifecycle(at, at + timedelta(days=4)) is EventLifecycle.POST_EVENT
    assert lifecycle(at, at + timedelta(days=7)) is EventLifecycle.POST_EVENT
    assert lifecycle(at, at + timedelta(days=8)) is EventLifecycle.ARCHIVED


# ---------------------------------------------------------------------------
# 4. same_event (§5, §7)
# ---------------------------------------------------------------------------


def test_same_event_matches_on_equal_key():
    assert same_event(_event(), _candidate()) is True


def test_same_event_absorbs_estimated_earnings_drift_within_21_days():
    existing = _event(
        event_key="EARNINGS:NVDA:2026-08-20",
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
        scheduled_at=_et(2026, 8, 20, 16, 5).astimezone(UTC),
    )
    incoming = _candidate()  # confirmed 2026-08-26, different key
    assert existing.event_key != incoming.event_key
    assert same_event(existing, incoming) is True


def test_same_event_does_not_merge_across_quarters():
    existing = _event(
        event_key="EARNINGS:NVDA:2026-05-27",
        scheduled_at=_et(2026, 5, 27, 16, 20).astimezone(UTC),
    )
    assert same_event(existing, _candidate()) is False


def test_same_event_earnings_window_requires_the_same_ticker():
    existing = _event(
        event_key="EARNINGS:AMD:2026-08-25",
        ticker="AMD",
        scheduled_at=_et(2026, 8, 25, 16, 20).astimezone(UTC),
    )
    assert same_event(existing, _candidate()) is False


def test_same_event_fomc_minutes_window_is_seven_days_and_types_never_cross():
    base = _et(2026, 10, 8, 14, 0).astimezone(UTC)
    existing = _event(
        event_id=9,
        event_key="FOMC_MINUTES:2026-10-08",
        event_type=EventType.FOMC_MINUTES,
        ticker=None,
        scheduled_at=base,
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_fomc",
        session=EventSession.DURING_MARKET,
    )
    near = _candidate(
        event_key="FOMC_MINUTES:2026-10-14",
        event_type=EventType.FOMC_MINUTES,
        ticker=None,
        scheduled_at=base + timedelta(days=6),
    )
    far = _candidate(
        event_key="FOMC_MINUTES:2026-10-20",
        event_type=EventType.FOMC_MINUTES,
        ticker=None,
        scheduled_at=base + timedelta(days=12),
    )
    other_type = _candidate(
        event_key="FOMC_DECISION:2026-10-09",
        event_type=EventType.FOMC_DECISION,
        ticker=None,
        scheduled_at=base + timedelta(days=1),
    )
    assert same_event(existing, near) is True
    assert same_event(existing, far) is False
    assert same_event(existing, other_type) is False


def test_same_event_never_uses_a_drift_window_for_other_types():
    existing = _event(
        event_key="CPI:2026-07",
        event_type=EventType.CPI,
        ticker=None,
        release_period="2026-07",
        scheduled_at=_utc(2026, 8, 12, 12, 30),
    )
    incoming = _candidate(
        event_key="CPI:2026-08",
        event_type=EventType.CPI,
        ticker=None,
        release_period="2026-08",
        scheduled_at=_utc(2026, 8, 13, 12, 30),
    )
    assert same_event(existing, incoming) is False


def test_earnings_drift_window_cannot_swallow_a_neighbouring_quarter():
    assert EARNINGS_DRIFT_WINDOW < timedelta(days=45)


# ---------------------------------------------------------------------------
# 5. Source precedence + merge (§78, §7)
# ---------------------------------------------------------------------------


def test_source_rank_ladder_is_the_spec_order_and_llm_is_last():
    ranks = [
        source_rank(k)
        for k in (
            EventSourceKind.USER,
            EventSourceKind.COMPANY_IR_SEC,
            EventSourceKind.GOVERNMENT_AGENCY,
            EventSourceKind.STRUCTURED_PROVIDER,
            EventSourceKind.DERIVED,
            EventSourceKind.NEWS,
            EventSourceKind.LLM,
        )
    ]
    assert ranks == sorted(ranks)
    assert source_rank(EventSourceKind.FEDERAL_RESERVE) == source_rank(
        EventSourceKind.GOVERNMENT_AGENCY
    )
    assert source_rank(EventSourceKind.LLM) > max(ranks[:-1])


@pytest.mark.parametrize(
    "existing_source,incoming_source,expect_move",
    [
        (EventSourceKind.DERIVED, EventSourceKind.COMPANY_IR_SEC, True),
        (EventSourceKind.STRUCTURED_PROVIDER, EventSourceKind.USER, True),
        (EventSourceKind.COMPANY_IR_SEC, EventSourceKind.COMPANY_IR_SEC, True),
        (EventSourceKind.COMPANY_IR_SEC, EventSourceKind.STRUCTURED_PROVIDER, False),
        (EventSourceKind.USER, EventSourceKind.COMPANY_IR_SEC, False),
        (EventSourceKind.COMPANY_IR_SEC, EventSourceKind.NEWS, False),
        (EventSourceKind.STRUCTURED_PROVIDER, EventSourceKind.LLM, False),
        # Same-rank pairs: equal authority IS authority to correct yourself,
        # so a NEWS row may be re-dated by NEWS...
        (EventSourceKind.NEWS, EventSourceKind.NEWS, True),
        (EventSourceKind.DERIVED, EventSourceKind.DERIVED, True),
        # ...but LLM is barred absolutely, so equal rank does NOT unlock the
        # date path for it (§78: "LLM never writes dates").
        (EventSourceKind.LLM, EventSourceKind.LLM, False),
        (EventSourceKind.USER, EventSourceKind.LLM, False),
    ],
)
def test_merge_date_write_follows_the_precedence_table(
    existing_source, incoming_source, expect_move
):
    """§78: a lower-authority source never overwrites a higher one, and an
    LLM never writes a date at all."""
    existing = _event(source=existing_source, source_name=existing_source.value.lower())
    moved = existing.scheduled_at + timedelta(days=1)
    incoming = _candidate(
        source=incoming_source,
        source_name=incoming_source.value.lower(),
        scheduled_at=moved,
        event_key="EARNINGS:NVDA:2026-08-27",
    )
    merged, change = merge(existing, incoming, NOW)
    if expect_move:
        assert merged.scheduled_at == moved
        assert change == "revised"
    else:
        assert merged.scheduled_at == existing.scheduled_at
        assert change == "reverified"


def test_merge_promotes_estimated_to_confirmed_even_from_a_lower_rank_source():
    """The one documented exception: a structured provider confirming a
    DERIVED estimate is new information, not an authority violation."""
    existing = _event(
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
    )
    merged, change = merge(existing, _candidate(), NOW)
    assert change == "confirmed"
    assert merged.status is EventStatus.CONFIRMED
    assert merged.source is EventSourceKind.STRUCTURED_PROVIDER
    assert merged.source_name == "massive_calendar"


def test_merge_confirming_an_estimate_on_a_new_date_records_the_replaced_value():
    existing = _event(
        event_key="EARNINGS:NVDA:2026-08-20",
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
        scheduled_at=_et(2026, 8, 20, 16, 5).astimezone(UTC),
    )
    merged, change = merge(existing, _candidate(), NOW)
    assert change == "confirmed"
    assert merged.status is EventStatus.CONFIRMED
    assert len(merged.revision_history) == 1
    entry = merged.revision_history[0]
    assert entry["status"] == "ESTIMATED"
    assert entry["source_name"] == "derived_cadence"
    assert entry["scheduled_at"] == existing.scheduled_at.isoformat()


def test_merge_never_downgrades_confirmed_to_estimated():
    """A cadence estimate arriving after the 8-K must not un-confirm it."""
    existing = _event()
    late_estimate = _candidate(
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
        scheduled_at=existing.scheduled_at + timedelta(days=3),
        event_key="EARNINGS:NVDA:2026-08-29",
    )
    merged, change = merge(existing, late_estimate, NOW)
    assert merged.status is EventStatus.CONFIRMED
    assert merged.scheduled_at == existing.scheduled_at
    assert change == "reverified"


def test_merge_marks_a_moved_confirmed_date_revised_and_appends_history():
    existing = _event()
    moved = existing.scheduled_at + timedelta(days=2)
    merged, change = merge(
        existing,
        _candidate(source=EventSourceKind.USER, source_name="user", scheduled_at=moved),
        NOW,
    )
    assert change == "revised"
    assert merged.status is EventStatus.REVISED
    assert merged.scheduled_at == moved
    assert merged.revision_history[-1]["scheduled_at"] == existing.scheduled_at.isoformat()
    assert merged.revision_history[-1]["at"] == NOW.isoformat()


def test_merge_reverification_refreshes_last_verified_without_claiming_change():
    existing = _event()
    merged, change = merge(existing, _candidate(source=EventSourceKind.COMPANY_IR_SEC), NOW)
    assert change == "reverified"
    assert merged.last_verified_at == NOW
    assert merged.updated_at == NOW
    assert merged.scheduled_at == existing.scheduled_at
    assert merged.status is existing.status


def test_merge_cancel_requires_an_explicit_canceled_candidate():
    existing = _event()
    merged, change = merge(
        existing,
        _candidate(status=EventStatus.CANCELED, source=EventSourceKind.USER, source_name="user"),
        NOW,
    )
    assert change == "canceled"
    assert merged.status is EventStatus.CANCELED
    assert len(merged.revision_history) == 1


def test_merge_never_resurrects_a_canceled_event_from_a_provider_feed():
    canceled = _event(status=EventStatus.CANCELED)
    merged, change = merge(canceled, _candidate(), NOW)
    assert merged.status is EventStatus.CANCELED
    assert change == "reverified"
    # ... but the user can.
    revived, revived_change = merge(
        canceled,
        _candidate(source=EventSourceKind.USER, source_name="user"),
        NOW,
    )
    assert revived.status is EventStatus.CONFIRMED
    assert revived_change == "confirmed"


def test_merge_fills_null_metadata_from_any_source_without_rewriting_facts():
    existing = _event(source_url=None, fiscal_quarter=None)
    merged, change = merge(
        existing,
        _candidate(
            source=EventSourceKind.NEWS,
            source_name="news",
            source_url="https://example.test/pr",
            fiscal_quarter=2,
            fiscal_year=2027,
        ),
        NOW,
    )
    assert change == "metadata"
    assert merged.source_url == "https://example.test/pr"
    assert merged.fiscal_quarter == 2
    # The authoritative fields are untouched by the low-rank source.
    assert merged.source is EventSourceKind.COMPANY_IR_SEC
    assert merged.scheduled_at == existing.scheduled_at


def test_merge_does_not_overwrite_existing_metadata_from_a_lower_rank_source():
    existing = _event(source_url="https://www.sec.gov/original")
    merged, _ = merge(
        existing,
        _candidate(source=EventSourceKind.NEWS, source_name="news", source_url="https://blog.test/x"),
        NOW,
    )
    assert merged.source_url == "https://www.sec.gov/original"


def test_merge_returns_a_new_frozen_event_and_leaves_the_original_alone():
    existing = _event()
    merged, _ = merge(existing, _candidate(), NOW)
    assert merged is not existing
    assert existing.last_verified_at == _utc(2026, 8, 18, 12, 0)
    with pytest.raises(Exception):
        merged.status = EventStatus.CANCELED  # frozen


def test_merge_fills_in_a_session_that_was_unknown():
    existing = _event(session=EventSession.UNKNOWN)
    merged, change = merge(existing, _candidate(session=EventSession.BEFORE_MARKET), NOW)
    assert merged.session is EventSession.BEFORE_MARKET
    assert change == "metadata"


def test_candidate_to_event_stamps_creation_and_verification():
    created = _candidate(status=EventStatus.ESTIMATED).to_event(now=NOW)
    assert created.created_at == created.updated_at == NOW
    assert created.last_verified_at == NOW
    assert created.event_id is None
    assert created.revision_history == ()
    assert created.is_estimated is True


# ---------------------------------------------------------------------------
# 6. previous_comparable (§15)
# ---------------------------------------------------------------------------


def test_previous_comparable_earnings_picks_the_latest_prior_confirmed():
    current = _event(event_id=10)
    q1 = _event(
        event_id=8,
        event_key="EARNINGS:NVDA:2026-02-25",
        scheduled_at=_et(2026, 2, 25, 16, 20).astimezone(UTC),
    )
    q2 = _event(
        event_id=9,
        event_key="EARNINGS:NVDA:2026-05-27",
        scheduled_at=_et(2026, 5, 27, 16, 20).astimezone(UTC),
    )
    prior, reason = previous_comparable(current, [q1, q2, current])
    assert prior is q2
    assert reason == "prior quarterly earnings"


def test_previous_comparable_never_crosses_ticker_or_type():
    current = _event(event_id=10)
    other_ticker = _event(
        event_id=8,
        event_key="EARNINGS:AMD:2026-05-27",
        ticker="AMD",
        scheduled_at=_et(2026, 5, 27, 16, 20).astimezone(UTC),
    )
    other_type = _event(
        event_id=7,
        event_key="CPI:2026-06",
        event_type=EventType.CPI,
        ticker=None,
        release_period="2026-06",
        scheduled_at=_utc(2026, 7, 14, 12, 30),
    )
    prior, reason = previous_comparable(current, [other_ticker, other_type])
    assert prior is None
    assert reason is None


def test_previous_comparable_earnings_ignores_an_estimated_predecessor():
    current = _event(event_id=10)
    estimated_prior = _event(
        event_id=9,
        event_key="EARNINGS:NVDA:2026-05-27",
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
        scheduled_at=_et(2026, 5, 27, 16, 20).astimezone(UTC),
    )
    assert previous_comparable(current, [estimated_prior]) == (None, None)


def test_previous_comparable_macro_is_the_prior_release_period():
    july = _event(
        event_id=3,
        event_key="CPI:2026-07",
        event_type=EventType.CPI,
        ticker=None,
        release_period="2026-07",
        scheduled_at=_utc(2026, 8, 12, 12, 30),
    )
    june = _event(
        event_id=2,
        event_key="CPI:2026-06",
        event_type=EventType.CPI,
        ticker=None,
        release_period="2026-06",
        scheduled_at=_utc(2026, 7, 14, 12, 30),
    )
    may = _event(
        event_id=1,
        event_key="CPI:2026-05",
        event_type=EventType.CPI,
        ticker=None,
        release_period="2026-05",
        scheduled_at=_utc(2026, 6, 10, 12, 30),
    )
    prior, reason = previous_comparable(july, [may, june])
    assert prior is june
    assert reason == "prior release of the same series"


def test_previous_comparable_fomc_decision_and_speech_rules():
    sept = _event(
        event_id=5,
        event_key="FOMC_DECISION:2026-09-17",
        event_type=EventType.FOMC_DECISION,
        ticker=None,
        scheduled_at=_et(2026, 9, 17, 14, 0).astimezone(UTC),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_fomc",
    )
    july = _event(
        event_id=4,
        event_key="FOMC_DECISION:2026-07-29",
        event_type=EventType.FOMC_DECISION,
        ticker=None,
        scheduled_at=_et(2026, 7, 29, 14, 0).astimezone(UTC),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_fomc",
    )
    prior, reason = previous_comparable(sept, [july])
    assert prior is july and reason == "prior FOMC decision"

    speech = _event(
        event_id=7,
        event_key="FED_SPEECH:2026-09-04:jerome-h-powell:outlook",
        event_type=EventType.FED_SPEECH,
        ticker=None,
        speaker="Jerome H. Powell",
        scheduled_at=_et(2026, 9, 4, 9, 0).astimezone(UTC),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_rss",
    )
    same_speaker = _event(
        event_id=6,
        event_key="FED_SPEECH:2026-06-04:jerome-h-powell:policy",
        event_type=EventType.FED_SPEECH,
        ticker=None,
        speaker="jerome h. powell",
        scheduled_at=_et(2026, 6, 4, 9, 0).astimezone(UTC),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_rss",
    )
    other_speaker = _event(
        event_id=5,
        event_key="FED_SPEECH:2026-08-04:john-c-williams:outlook",
        event_type=EventType.FED_SPEECH,
        ticker=None,
        speaker="John C. Williams",
        scheduled_at=_et(2026, 8, 4, 9, 0).astimezone(UTC),
        source=EventSourceKind.FEDERAL_RESERVE,
        source_name="fed_rss",
    )
    prior, reason = previous_comparable(speech, [same_speaker, other_speaker])
    assert prior is same_speaker
    assert "low confidence" in reason


def test_previous_comparable_returns_honest_none_for_holidays_and_empty_pools():
    holiday = _event(
        event_id=2,
        event_key="MARKET_HOLIDAY:US:2026-11-26",
        event_type=EventType.MARKET_HOLIDAY,
        ticker=None,
        scheduled_at=_et(2026, 11, 26, 0, 0).astimezone(UTC),
        source=EventSourceKind.STRUCTURED_PROVIDER,
        source_name="massive_holidays",
    )
    assert previous_comparable(holiday, [holiday]) == (None, None)
    assert previous_comparable(_event(), []) == (None, None)


# ---------------------------------------------------------------------------
# 7. Importance (§12, §13)
# ---------------------------------------------------------------------------


def test_importance_components_sum_to_the_score():
    result = score_importance(EventType.EARNINGS, relevance_tier="WATCHLIST")
    assert result.components == {"event_type": 60, "relevance": 10}
    assert result.score == sum(result.components.values()) == 70
    assert result.was_clamped is False


def test_importance_relevance_ladder_orders_by_proximity_to_the_users_money():
    scores = [
        score_importance(EventType.EARNINGS, relevance_tier=tier).score
        for tier in ("POSITION", "TRADING_POOL", "WATCHLIST", "OTHER")
    ]
    assert scores == [90, 80, 70, 60]


def test_importance_clamps_at_100_but_still_shows_the_raw_arithmetic():
    result = score_importance(EventType.FOMC_DECISION, relevance_tier="POSITION")
    assert result.raw_total == 120
    assert result.score == 100
    assert result.was_clamped is True


def test_importance_gives_the_chair_a_named_speaker_component():
    plain = score_importance(EventType.FED_SPEECH, speaker="John C. Williams")
    chair = score_importance(EventType.FED_SPEECH, speaker="Chair Jerome H. Powell")
    assert "speaker_seniority" not in plain.components
    assert chair.components["speaker_seniority"] == CHAIR_SPEAKER_BONUS
    assert chair.score == plain.score + CHAIR_SPEAKER_BONUS
    assert score_importance(EventType.FED_SPEECH, speaker="Vice Chair Philip N. Jefferson").score == (
        plain.score + CHAIR_SPEAKER_BONUS
    )


def test_importance_defaults_macro_and_fed_types_to_market_wide():
    assert default_relevance_tier(EventType.CPI) == "MARKET_WIDE"
    assert default_relevance_tier(EventType.FOMC_DECISION) == "MARKET_WIDE"
    assert default_relevance_tier(EventType.EARNINGS) == "OTHER"
    assert score_importance(EventType.CPI).relevance_tier == "MARKET_WIDE"
    assert score_importance(EventType.CPI).score == BASE_IMPORTANCE[EventType.CPI]


def test_importance_extra_components_extend_the_same_additive_dict():
    """Later phases add implied-move / news-intensity without a new signature."""
    result = score_importance(
        EventType.EARNINGS,
        relevance_tier="POSITION",
        extra_components={"implied_move": 5, "news_intensity": -3},
    )
    assert result.components["implied_move"] == 5
    assert result.score == 60 + 30 + 5 - 3
    assert result.score == result.raw_total


def test_importance_rejects_an_unknown_relevance_tier_rather_than_scoring_zero():
    with pytest.raises(ValueError):
        score_importance(EventType.EARNINGS, relevance_tier="PORTFOLIO")


def test_importance_every_event_type_has_a_base_weight():
    """A new EventType with no weight would silently score 0."""
    assert set(BASE_IMPORTANCE) == set(EventType)
    assert all(0 <= w <= 100 for w in BASE_IMPORTANCE.values())


def test_relevance_rank_orders_the_spec_ladder_and_tolerates_unknowns():
    assert [relevance_rank(t) for t in RELEVANCE_TIERS] == list(range(len(RELEVANCE_TIERS)))
    assert relevance_rank("NOT_A_TIER") == len(RELEVANCE_TIERS)


def test_merge_rekeys_the_row_when_an_accepted_date_move_changes_the_key():
    """The natural key embeds the ET date. A row that kept its estimated-date
    key after being confirmed onto another day would be re-created under the
    new key on the next tick — the duplicate the drift window exists to stop.
    """
    estimated = _event(
        event_key="EARNINGS:NVDA:2026-08-20",
        status=EventStatus.ESTIMATED,
        source=EventSourceKind.DERIVED,
        source_name="derived_cadence",
        scheduled_at=_et(2026, 8, 20, 16, 5).astimezone(UTC),
    )
    confirmed = _candidate(source=EventSourceKind.COMPANY_IR_SEC, source_name="sec_edgar")
    assert same_event(estimated, confirmed) is True
    merged, change = merge(estimated, confirmed, NOW)
    assert change == "confirmed"
    assert merged.event_key == confirmed.event_key == "EARNINGS:NVDA:2026-08-26"
    assert merged.event_key == event_key(
        EventType.EARNINGS, ticker="NVDA", scheduled_at=merged.scheduled_at
    )
    # A second tick with the same candidate is now a pure re-verification.
    again, again_change = merge(merged, confirmed, NOW + timedelta(hours=20))
    assert again_change == "reverified"
    assert again.event_key == merged.event_key


def test_merge_keeps_the_key_when_the_date_is_not_accepted():
    existing = _event()
    merged, _ = merge(
        existing,
        _candidate(
            event_key="EARNINGS:NVDA:2026-08-29",
            source=EventSourceKind.NEWS,
            source_name="news",
            scheduled_at=existing.scheduled_at + timedelta(days=3),
        ),
        NOW,
    )
    assert merged.event_key == existing.event_key


def test_llm_cannot_move_a_date_even_over_an_llm_written_row():
    """§78 is an absolute floor, not a relative comparison.

    The authority test elsewhere is ``rank(incoming) <= rank(existing)``, and
    LLM == LLM satisfies it. That would let an extracted date overwrite a
    stored date with no structured source ever involved, which is exactly the
    failure mode the rank-99 gap exists to prevent. Phase D/F adds the news /
    LLM extractor that makes this reachable; the bar is asserted now so the
    extractor lands against a merge() that already refuses it.
    """
    existing = _event(source=EventSourceKind.LLM, source_name="llm_extractor")
    moved = existing.scheduled_at + timedelta(days=3)
    incoming = _candidate(
        source=EventSourceKind.LLM,
        source_name="llm_extractor",
        scheduled_at=moved,
        event_key="EARNINGS:NVDA:2026-08-29",
    )
    merged, change = merge(existing, incoming, NOW)

    assert merged.scheduled_at == existing.scheduled_at
    assert merged.event_key == existing.event_key
    assert merged.revision_history == existing.revision_history
    assert change == "reverified"


def test_llm_cannot_promote_or_cancel_but_still_refreshes_verification():
    """The bar covers status too — not just the timestamp column.

    An LLM candidate must not confirm an estimate nor cancel a row. What it
    MAY still do is prove the row was looked at, so ``last_verified_at``
    moves; that is information, and suppressing it would make the UI claim
    the row is staler than it is.
    """
    estimated = _event(
        source=EventSourceKind.LLM,
        source_name="llm_extractor",
        status=EventStatus.ESTIMATED,
    )
    confirming = _candidate(
        source=EventSourceKind.LLM,
        source_name="llm_extractor",
        status=EventStatus.CONFIRMED,
    )
    merged, change = merge(estimated, confirming, NOW)
    assert merged.status is EventStatus.ESTIMATED
    assert change == "reverified"
    assert merged.last_verified_at == NOW

    canceling = _candidate(
        source=EventSourceKind.LLM,
        source_name="llm_extractor",
        status=EventStatus.CANCELED,
    )
    merged, change = merge(estimated, canceling, NOW)
    assert merged.status is EventStatus.ESTIMATED
    assert change == "reverified"


def test_llm_may_still_fill_a_blank_metadata_field():
    """The bar is on facts, not on enrichment.

    Writing a source_url into a NULL is not overwriting anyone's claim, so an
    LLM candidate is still allowed to do it. Barring that too would make the
    rule cost more than it buys.
    """
    existing = _event(
        source=EventSourceKind.COMPANY_IR_SEC,
        source_name="sec_edgar",
        source_url=None,
    )
    incoming = _candidate(
        source=EventSourceKind.LLM,
        source_name="llm_extractor",
        source_url="https://example.com/ir/press-release",
    )
    merged, change = merge(existing, incoming, NOW)
    assert merged.source_url == "https://example.com/ir/press-release"
    assert merged.source is EventSourceKind.COMPANY_IR_SEC
    assert change == "metadata"
