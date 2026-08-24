"""Phase H U2 — the pure Fed intelligence library (spec §9, §42-§45).

The two statement fixtures are the REAL released text of the June 17 2026 and
July 29 2026 FOMC statements (downloaded live from federalreserve.gov with the
contact User-Agent; see ``tests/fixtures/events/README.md``). Diffing them is
the whole point of §44 and the counts asserted below are the ones the actual
documents produce — the Committee changed the vote line (12-0 → 9-3), changed
"reaffirmed its policy" to "is continuing its policy", and ADDED the dissent
paragraph. Nothing else moved.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from libs.trading_core.events import fed_intel as fi
from libs.trading_core.events.macro import YieldCurveRow
from libs.trading_core.events.reaction import DailyBar
from libs.trading_core.events.replay import MinuteBar

UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")

FIXTURES = Path(__file__).parent / "fixtures" / "events"


def _paragraphs(name: str) -> list[str]:
    text = (FIXTURES / name).read_text(encoding="utf-8").strip()
    return [p.strip() for p in text.split("\n\n") if p.strip()]


@pytest.fixture(scope="module")
def june_paragraphs() -> list[str]:
    return _paragraphs("fomc_statement_2026-06-17.txt")


@pytest.fixture(scope="module")
def july_paragraphs() -> list[str]:
    return _paragraphs("fomc_statement_2026-07-29.txt")


@pytest.fixture(scope="module")
def june_statement(june_paragraphs: list[str]) -> dict:
    return {
        "doc_type": "STATEMENT",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm",
        "title": "Federal Reserve issues FOMC statement",
        "meeting_date": date(2026, 6, 17),
        "released_at": datetime(2026, 6, 17, 18, 0, tzinfo=UTC),
        "paragraphs": june_paragraphs,
        "vote": {
            "for": 12,
            "against": 0,
            "dissenters": [],
            "text": "by a 12 – 0 vote",
        },
        "target_range": fi.parse_target_range(june_paragraphs[1]),
    }


@pytest.fixture(scope="module")
def july_statement(july_paragraphs: list[str]) -> dict:
    return {
        "doc_type": "STATEMENT",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm",
        "title": "Federal Reserve issues FOMC statement",
        "meeting_date": date(2026, 7, 29),
        "released_at": datetime(2026, 7, 29, 18, 0, tzinfo=UTC),
        "paragraphs": july_paragraphs,
        "vote": {
            "for": 9,
            "against": 3,
            "dissenters": ["Beth M. Hammack", "Neel Kashkari", "Lorie K. Logan"],
            "text": "by a 9 – 3 vote",
        },
        "target_range": fi.parse_target_range(july_paragraphs[1]),
    }


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_positions_and_normalization(july_paragraphs):
    sentences = fi.split_sentences(july_paragraphs)
    assert [s.idx for s in sentences] == list(range(len(sentences)))
    # Paragraph indices are non-decreasing and match the source numbering.
    assert sentences[0].para_idx == 0
    assert [s.para_idx for s in sentences] == sorted(s.para_idx for s in sentences)
    assert sentences[0].text.startswith("The Federal Open Market Committee approved")
    assert "committee" in sentences[0].normalized
    # Normalization strips punctuation but keeps fraction structure.
    assert "," not in sentences[0].normalized


def test_split_sentences_keeps_the_dissent_paragraph_whole(july_paragraphs):
    """"Beth M. Hammack" must not split into four fragments."""
    dissent = [p for p in july_paragraphs if p.startswith("Voting against")]
    assert len(dissent) == 1
    sentences = fi.split_sentences(dissent)
    assert len(sentences) == 1
    assert "Lorie K. Logan" in sentences[0].text


def test_split_sentences_empty_paragraph_consumes_an_index():
    sentences = fi.split_sentences(["First one. Second one.", "   ", "Third one."])
    assert [s.text for s in sentences] == ["First one.", "Second one.", "Third one."]
    assert [s.para_idx for s in sentences] == [0, 0, 2]


def test_normalize_sentence_keeps_fractions_distinguishable():
    a = fi.normalize_sentence("at 3-1/2 to 3-3/4 percent")
    b = fi.normalize_sentence("at 3-1/4 to 3-1/2 percent")
    assert a != b


# ---------------------------------------------------------------------------
# §44 — the diff
# ---------------------------------------------------------------------------


def test_statement_diff_on_real_statements_counts(june_paragraphs, july_paragraphs):
    diff = fi.statement_diff(june_paragraphs, july_paragraphs)
    assert diff.counts == {
        "ADDED": 1,
        "REMOVED": 0,
        "CHANGED": 2,
        "UNCHANGED": 6,
        "TOTAL": 9,
    }


def test_statement_diff_identifies_the_vote_line_change(
    june_paragraphs, july_paragraphs
):
    diff = fi.statement_diff(june_paragraphs, july_paragraphs)
    changed = [i for i in diff.items if i.status == fi.STATUS_CHANGED]
    vote_row = next(i for i in changed if "12 – 0" in (i.previous_text or ""))
    assert "9 – 3" in vote_row.current_text
    assert fi.DIMENSION_COMMITTEE_DISPERSION in vote_row.dimensions
    assert 0.6 <= vote_row.similarity < 1.0


def test_statement_diff_identifies_the_balance_sheet_change(
    june_paragraphs, july_paragraphs
):
    diff = fi.statement_diff(june_paragraphs, july_paragraphs)
    row = next(
        i
        for i in diff.items
        if i.status == fi.STATUS_CHANGED and "reaffirmed its policy" in (i.previous_text or "")
    )
    assert "is continuing its policy" in row.current_text
    assert fi.DIMENSION_BALANCE_SHEET in row.dimensions


def test_statement_diff_reports_the_dissent_paragraph_as_added(
    june_paragraphs, july_paragraphs
):
    diff = fi.statement_diff(june_paragraphs, july_paragraphs)
    added = [i for i in diff.items if i.status == fi.STATUS_ADDED]
    assert len(added) == 1
    assert added[0].current_text.startswith("Voting against the monetary policy action")
    assert added[0].previous_text is None
    assert added[0].similarity is None


def test_statement_diff_reverse_direction_reports_a_removal(
    june_paragraphs, july_paragraphs
):
    """Diffing the other way round turns the ADDED dissent into a REMOVED one."""
    diff = fi.statement_diff(july_paragraphs, june_paragraphs)
    assert diff.counts["REMOVED"] == 1
    assert diff.counts["ADDED"] == 0
    removed = next(i for i in diff.items if i.status == fi.STATUS_REMOVED)
    assert removed.previous_text.startswith("Voting against")


def test_statement_diff_identical_input_is_all_unchanged(july_paragraphs):
    diff = fi.statement_diff(july_paragraphs, july_paragraphs)
    assert diff.counts["CHANGED"] == 0
    assert diff.counts["ADDED"] == 0
    assert diff.counts["REMOVED"] == 0
    assert diff.counts["UNCHANGED"] == diff.counts["TOTAL"]
    assert all(i.similarity == 1.0 for i in diff.items)


def test_statement_diff_is_deterministic(june_paragraphs, july_paragraphs):
    first = fi.statement_diff(june_paragraphs, july_paragraphs).to_dict()
    second = fi.statement_diff(june_paragraphs, july_paragraphs).to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_statement_diff_wholly_new_sentence_is_added_not_changed():
    """Below the similarity floor a replacement is two rows, not one."""
    diff = fi.statement_diff(
        ["The Committee will assess incoming data."],
        ["Bananas are yellow fruit grown in tropical climates."],
    )
    statuses = sorted(i.status for i in diff.items)
    assert statuses == [fi.STATUS_ADDED, fi.STATUS_REMOVED]


def test_statement_diff_reports_verbatim_text_not_normalized(july_paragraphs):
    diff = fi.statement_diff([], july_paragraphs)
    for item in diff.items:
        assert item.current_text == item.current_text.strip()
    assert any("Committee's" in (i.current_text or "") for i in diff.items)


def test_statement_diff_to_dict_carries_the_source_note(june_paragraphs, july_paragraphs):
    payload = fi.statement_diff(june_paragraphs, july_paragraphs).to_dict()
    assert payload["note"] == fi.SOURCE_AUTHORITATIVE_NOTE
    assert payload["model_version"] == fi.FED_INTEL_MODEL_VERSION
    assert len(payload["items"]) == payload["counts"]["TOTAL"]


def test_statement_diff_empty_previous_is_all_added(july_paragraphs):
    diff = fi.statement_diff([], july_paragraphs)
    assert diff.counts["ADDED"] == diff.counts["TOTAL"] > 0
    assert diff.counts["UNCHANGED"] == 0


# ---------------------------------------------------------------------------
# §43 — dimensions
# ---------------------------------------------------------------------------


def test_dimensions_constant_is_the_eight_spec_dimensions():
    assert fi.DIMENSIONS == (
        "POLICY_RATE",
        "INFLATION",
        "EMPLOYMENT",
        "GROWTH",
        "BALANCE_SHEET",
        "FORWARD_GUIDANCE",
        "RISK_BALANCE",
        "COMMITTEE_DISPERSION",
    )
    assert set(fi.DIMENSION_KEYWORDS) == set(fi.DIMENSIONS)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Inflation remains elevated relative to the 2 percent goal.", "INFLATION"),
        ("Job gains have kept pace with the workforce.", "EMPLOYMENT"),
        ("Economic activity is expanding at a solid pace.", "GROWTH"),
        (
            "The Committee is continuing its policy of maintaining ample reserves.",
            "BALANCE_SHEET",
        ),
        (
            "In determining the extent and timing of additional adjustments.",
            "FORWARD_GUIDANCE",
        ),
        ("The Committee is attentive to risks to both sides.", "RISK_BALANCE"),
        (
            "maintain the target range for the federal funds rate at 4 percent",
            "POLICY_RATE",
        ),
        ("Voting against the monetary policy action were two members.", "COMMITTEE_DISPERSION"),
    ],
)
def test_dimensions_for_tags_the_expected_dimension(sentence, expected):
    assert expected in fi.dimensions_for(sentence)


def test_dimensions_for_returns_tags_in_report_order():
    tags = fi.dimensions_for(
        "Inflation risks remain, and the target range for the federal funds rate is unchanged."
    )
    assert list(tags) == [d for d in fi.DIMENSIONS if d in tags]
    assert "POLICY_RATE" in tags and "INFLATION" in tags and "RISK_BALANCE" in tags


def test_dimensions_for_untagged_sentence_is_empty():
    assert fi.dimensions_for("The next meeting is scheduled for later this year.") == ()
    assert fi.dimensions_for("") == ()


def test_dimension_report_has_every_dimension_always(june_statement, july_statement):
    diff = fi.statement_diff(
        june_statement["paragraphs"], july_statement["paragraphs"]
    )
    report = fi.dimension_report(june_statement, july_statement, diff)
    assert set(report) == set(fi.DIMENSIONS)
    for dimension, row in report.items():
        assert row["dimension"] == dimension
        assert row["status"] in {
            fi.STATUS_CHANGED,
            fi.STATUS_UNCHANGED,
            fi.STATUS_ADDED,
            fi.STATUS_REMOVED,
            fi.STATUS_NA,
        }
        assert isinstance(row["previous"], list)
        assert isinstance(row["current"], list)


def test_dimension_report_statuses_on_the_real_statements(
    june_statement, july_statement
):
    diff = fi.statement_diff(
        june_statement["paragraphs"], july_statement["paragraphs"]
    )
    report = fi.dimension_report(june_statement, july_statement, diff)
    assert report["BALANCE_SHEET"]["status"] == fi.STATUS_CHANGED
    assert report["COMMITTEE_DISPERSION"]["status"] == fi.STATUS_CHANGED
    assert report["INFLATION"]["status"] == fi.STATUS_UNCHANGED
    assert report["EMPLOYMENT"]["status"] == fi.STATUS_UNCHANGED
    assert report["GROWTH"]["status"] == fi.STATUS_UNCHANGED
    # Neither statement carries forward-guidance boilerplate this cycle.
    assert report["FORWARD_GUIDANCE"]["status"] == fi.STATUS_NA


def test_dimension_report_committee_dispersion_uses_the_vote(
    june_statement, july_statement
):
    diff = fi.statement_diff(
        june_statement["paragraphs"], july_statement["paragraphs"]
    )
    row = fi.dimension_report(june_statement, july_statement, diff)[
        "COMMITTEE_DISPERSION"
    ]
    assert row["previous_vote"]["against"] == 0
    assert row["previous_vote"]["unanimous"] is True
    assert row["current_vote"]["against"] == 3
    assert row["current_vote"]["unanimous"] is False
    assert row["dissent_change"] == 3
    assert row["current_vote"]["dissenters"] == [
        "Beth M. Hammack",
        "Neel Kashkari",
        "Lorie K. Logan",
    ]


def test_dimension_report_unchanged_vote_is_unchanged_status(june_statement):
    diff = fi.statement_diff(june_statement["paragraphs"], june_statement["paragraphs"])
    row = fi.dimension_report(june_statement, june_statement, diff)[
        "COMMITTEE_DISPERSION"
    ]
    assert row["status"] == fi.STATUS_UNCHANGED
    assert row["dissent_change"] == 0


def test_dimension_report_policy_rate_carries_the_change(june_statement, july_statement):
    diff = fi.statement_diff(
        june_statement["paragraphs"], july_statement["paragraphs"]
    )
    row = fi.dimension_report(june_statement, july_statement, diff)["POLICY_RATE"]
    assert row["policy_rate_change"]["direction"] == fi.DIRECTION_HOLD
    assert row["policy_rate_change"]["change_bp"] == 0
    assert "3-1/2 to 3-3/4 percent" in row["notes"]


def test_dimension_report_tolerates_missing_statements():
    report = fi.dimension_report(None, None, None)
    assert set(report) == set(fi.DIMENSIONS)
    assert all(row["status"] == fi.STATUS_NA for row in report.values())


# ---------------------------------------------------------------------------
# target range + policy rate change
# ---------------------------------------------------------------------------


def test_parse_target_range_on_the_real_sentence(july_paragraphs):
    parsed = fi.parse_target_range(july_paragraphs[1])
    assert parsed == {
        "low_pct": 3.5,
        "high_pct": 3.75,
        "text": "target range for the federal funds rate at 3-1/2 to 3-3/4 percent",
    }


@pytest.mark.parametrize(
    "text, low, high",
    [
        ("maintain the target range for the federal funds rate at 4 to 4-1/4 percent.", 4.0, 4.25),
        ("lower the target range for the federal funds rate at 3-3/4 to 4 percent.", 3.75, 4.0),
        ("the target range for the federal funds rate at 5-1/4 to 5-1/2 percent", 5.25, 5.5),
    ],
)
def test_parse_target_range_reads_mixed_fractions(text, low, high):
    parsed = fi.parse_target_range(text)
    assert parsed["low_pct"] == low
    assert parsed["high_pct"] == high


@pytest.mark.parametrize("text", [None, "", "no target range phrase here at all"])
def test_parse_target_range_absent_is_none(text):
    assert fi.parse_target_range(text) is None


def test_policy_rate_change_hold_cut_hike():
    hold = fi.policy_rate_change(
        {"low_pct": 3.5, "high_pct": 3.75}, {"low_pct": 3.5, "high_pct": 3.75}
    )
    assert hold["change_bp"] == 0 and hold["direction"] == fi.DIRECTION_HOLD
    cut = fi.policy_rate_change(
        {"low_pct": 3.5, "high_pct": 3.75}, {"low_pct": 3.25, "high_pct": 3.5}
    )
    assert cut["change_bp"] == -25 and cut["direction"] == fi.DIRECTION_CUT
    hike = fi.policy_rate_change(
        {"low_pct": 3.5, "high_pct": 3.75}, {"low_pct": 4.0, "high_pct": 4.25}
    )
    assert hike["change_bp"] == 50 and hike["direction"] == fi.DIRECTION_HIKE


def test_policy_rate_change_missing_side_is_none_not_hold():
    out = fi.policy_rate_change(None, {"low_pct": 3.5, "high_pct": 3.75})
    assert out["change_bp"] is None
    assert out["direction"] is None
    assert out["reason"]


# ---------------------------------------------------------------------------
# vote dispersion
# ---------------------------------------------------------------------------


def test_vote_dispersion_split_vote():
    out = fi.vote_dispersion(
        {"for": 9, "against": 3, "dissenters": ["A", "B", "C"], "text": "by a 9 – 3 vote"}
    )
    assert out["for"] == 9
    assert out["against"] == 3
    assert out["dissenters"] == ["A", "B", "C"]
    assert out["unanimous"] is False


def test_vote_dispersion_unanimous():
    out = fi.vote_dispersion({"for": 12, "against": 0, "dissenters": []})
    assert out["unanimous"] is True


def test_vote_dispersion_unparsed_is_not_optimistically_unanimous():
    out = fi.vote_dispersion(None)
    assert out["for"] is None
    assert out["against"] is None
    assert out["unanimous"] is None
    assert out["dissenters"] == []


# ---------------------------------------------------------------------------
# §45 — the two reaction windows
# ---------------------------------------------------------------------------


def _decision_instants(day: date = date(2026, 7, 29)):
    decision = datetime(day.year, day.month, day.day, 14, 0, tzinfo=EASTERN).astimezone(UTC)
    presser = datetime(day.year, day.month, day.day, 14, 30, tzinfo=EASTERN).astimezone(UTC)
    presser_end = datetime(day.year, day.month, day.day, 15, 30, tzinfo=EASTERN).astimezone(UTC)
    return decision, presser, presser_end


def _minute_bars(decision: datetime, *, statement_step: float, presser_step: float):
    """Bars from 30 min before the decision to 90 min after.

    The statement window rises and the press-conference window falls, so a
    test that blended the two would report roughly zero — which is exactly
    the failure §45 exists to prevent.
    """
    bars: list[MinuteBar] = []
    price = 100.0
    for offset in range(-30, 91):
        ts = decision + timedelta(minutes=offset)
        if offset <= 0:
            price = 100.0
        elif offset <= 30:
            price = 100.0 + offset * statement_step
        else:
            price = 100.0 + 30 * statement_step + (offset - 30) * presser_step
        bars.append(
            MinuteBar(ts_utc=ts, open=price, high=price, low=price, close=price)
        )
    return bars


def test_fomc_reaction_windows_separates_statement_from_presser():
    decision, presser, presser_end = _decision_instants()
    bars = _minute_bars(decision, statement_step=0.01, presser_step=-0.005)
    out = fi.fomc_reaction_windows(
        {"SPY": bars},
        decision_at_utc=decision,
        press_conf_at_utc=presser,
        press_conf_end_utc=presser_end,
    )
    assert out["basis"] == fi.REACTION_BASIS_MINUTE
    assert out["separated"] is True
    assert out["unit"] == "percent"
    stmt = out["statement"]["SPY"]
    pres = out["press_conference"]["SPY"]
    assert stmt["pre_close"] == pytest.approx(100.0)
    assert stmt["post_close"] == pytest.approx(100.3)
    assert stmt["return_pct"] == pytest.approx(0.3, abs=1e-6)
    # The presser window starts where the statement window ended and moves
    # the OTHER way. A blended number would be ~0.
    assert pres["pre_close"] == pytest.approx(100.3)
    assert pres["return_pct"] < 0
    assert stmt["return_pct"] > 0


def test_fomc_reaction_windows_labels_the_et_spans():
    decision, presser, presser_end = _decision_instants()
    out = fi.fomc_reaction_windows(
        {}, decision_at_utc=decision, press_conf_at_utc=presser, press_conf_end_utc=presser_end
    )
    assert out["windows"]["statement"]["label_et"] == "14:00-14:30 ET"
    assert out["windows"]["press_conference"]["label_et"] == "14:30-15:30 ET"
    assert out["windows"]["statement"]["end"] == out["windows"]["press_conference"]["start"]


def test_fomc_reaction_windows_multiple_symbols_sorted():
    decision, presser, presser_end = _decision_instants()
    bars = _minute_bars(decision, statement_step=0.01, presser_step=-0.005)
    out = fi.fomc_reaction_windows(
        {"TLT": bars, "SPY": bars, "GLD": bars},
        decision_at_utc=decision,
        press_conf_at_utc=presser,
        press_conf_end_utc=presser_end,
    )
    assert list(out["statement"]) == ["GLD", "SPY", "TLT"]
    assert list(out["press_conference"]) == ["GLD", "SPY", "TLT"]


def test_fomc_reaction_windows_missing_bars_gives_reason_not_zero():
    decision, presser, presser_end = _decision_instants()
    out = fi.fomc_reaction_windows(
        {"SPY": []},
        decision_at_utc=decision,
        press_conf_at_utc=presser,
        press_conf_end_utc=presser_end,
    )
    assert out["statement"]["SPY"]["return_pct"] is None
    assert out["statement"]["SPY"]["reason"]
    assert out["press_conference"]["SPY"]["return_pct"] is None


def test_fomc_reaction_windows_statement_only_bars_leaves_presser_unavailable():
    decision, presser, presser_end = _decision_instants()
    bars = [
        MinuteBar(
            ts_utc=decision + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.0 + i * 0.01,
            low=100.0 + i * 0.01,
            close=100.0 + i * 0.01,
        )
        for i in range(0, 31)
    ]
    out = fi.fomc_reaction_windows(
        {"SPY": bars},
        decision_at_utc=decision,
        press_conf_at_utc=presser,
        press_conf_end_utc=presser_end,
    )
    assert out["statement"]["SPY"]["return_pct"] is not None
    assert out["press_conference"]["SPY"]["return_pct"] is None
    assert out["press_conference"]["SPY"]["reason"] == "no bar inside the window"


def test_fomc_reaction_windows_rejects_unordered_windows():
    decision, presser, presser_end = _decision_instants()
    with pytest.raises(ValueError, match="ordered"):
        fi.fomc_reaction_windows(
            {},
            decision_at_utc=presser,
            press_conf_at_utc=decision,
            press_conf_end_utc=presser_end,
        )


def test_fomc_reaction_windows_requires_utc():
    naive = datetime(2026, 7, 29, 18, 0)
    with pytest.raises(ValueError):
        fi.fomc_reaction_windows(
            {},
            decision_at_utc=naive,
            press_conf_at_utc=naive,
            press_conf_end_utc=naive,
        )


# ---------------------------------------------------------------------------
# daily fallback
# ---------------------------------------------------------------------------


def _daily_bars(start: date, closes: list[float]) -> list[DailyBar]:
    return [
        DailyBar(
            date=start + timedelta(days=i),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1_000_000.0,
        )
        for i, close in enumerate(closes)
    ]


def test_fomc_reaction_daily_says_it_cannot_separate_the_windows():
    bars = _daily_bars(date(2026, 7, 27), [100.0, 101.0, 102.0, 103.0, 104.0])
    decision = datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN).astimezone(UTC)
    out = fi.fomc_reaction_daily({"SPY": bars}, decision_at_utc=decision)
    assert out["basis"] == fi.REACTION_BASIS_DAILY
    assert out["label"] == "daily (no intraday bars)"
    assert out["separated"] is False
    assert "separate" in out["separation_reason"]


def test_fomc_reaction_daily_reports_returns_in_percent():
    bars = _daily_bars(date(2026, 7, 27), [100.0, 100.0, 101.0, 102.0, 103.0])
    decision = datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN).astimezone(UTC)
    out = fi.fomc_reaction_daily({"SPY": bars}, decision_at_utc=decision, horizons=(1,))
    asset = out["assets"]["SPY"]
    assert asset["pre_event_close"] == pytest.approx(100.0)
    # 100 -> 101 on the decision day is +1.00 PERCENT (not the 0.01 fraction
    # reaction.ReactionResult reports).
    assert asset["returns_pct"]["1"] == pytest.approx(1.0, abs=1e-6)
    assert out["unit"] == "percent"


def test_fomc_reaction_daily_missing_bars_is_unavailable_with_a_reason():
    decision = datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN).astimezone(UTC)
    out = fi.fomc_reaction_daily({"SPY": []}, decision_at_utc=decision)
    assert "SPY" not in out["assets"]
    assert out["unavailable"]["SPY"]


def test_fomc_reaction_daily_carries_the_2y_yield_change():
    bars = _daily_bars(date(2026, 7, 27), [100.0, 100.0, 101.0, 102.0])
    decision = datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN).astimezone(UTC)
    yields = [
        YieldCurveRow(curve_date=date(2026, 7, 28), tenors={"2 Yr": 3.90}),
        YieldCurveRow(curve_date=date(2026, 7, 29), tenors={"2 Yr": 3.97}),
    ]
    out = fi.fomc_reaction_daily(
        {"SPY": bars}, decision_at_utc=decision, yields=yields
    )
    assert out["yields"]["2 Yr"]["change_bp"] == pytest.approx(7.0, abs=1e-6)


# ---------------------------------------------------------------------------
# §42 — the packet
# ---------------------------------------------------------------------------


def _packet(june_statement, july_statement, **overrides):
    kwargs = {
        "current_event": {"id": 42, "event_type": "FOMC_DECISION"},
        "previous_decision": {"id": 41, "scheduled_at": "2026-07-29T18:00:00+00:00"},
        "prev_statement": july_statement,
        "prev_prev_statement": june_statement,
        "as_of": datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
    }
    kwargs.update(overrides)
    return fi.build_fed_packet(**kwargs)


def test_build_fed_packet_shape(june_statement, july_statement):
    packet = _packet(june_statement, july_statement)
    assert set(packet) == {
        "as_of",
        "event",
        "previous_statement",
        "statement_diff",
        "dimensions",
        "previous_minutes",
        "subsequent_speeches",
        "data",
        "market_pricing",
        "previous_reaction",
        "coverage",
        "tiers",
        "disclaimers",
        "model_version",
        "previous_decision",
    }
    assert packet["model_version"] == fi.FED_INTEL_MODEL_VERSION
    # JSON-ready: no datetime / date objects survive.
    json.dumps(packet)


def test_build_fed_packet_previous_statement_block(june_statement, july_statement):
    block = _packet(june_statement, july_statement)["previous_statement"]
    assert block["available"] is True
    assert block["url"].endswith("monetary20260729a.htm")
    assert block["released_at"] == "2026-07-29T18:00:00+00:00"
    assert block["vote"]["against"] == 3
    assert block["target_range"]["high_pct"] == 3.75
    assert len(block["paragraphs"]) == len(july_statement["paragraphs"])
    assert block["compared_to"]["url"].endswith("monetary20260617a.htm")


def test_build_fed_packet_diff_is_previous_vs_the_one_before(
    june_statement, july_statement
):
    packet = _packet(june_statement, july_statement)
    assert packet["statement_diff"]["counts"]["CHANGED"] == 2
    assert packet["statement_diff"]["counts"]["ADDED"] == 1
    assert packet["dimensions"]["COMMITTEE_DISPERSION"]["status"] == fi.STATUS_CHANGED


def test_build_fed_packet_market_pricing_is_unavailable_with_a_proxy(
    june_statement, july_statement
):
    reactions = fi.fomc_reaction_daily(
        {"SPY": _daily_bars(date(2026, 7, 27), [100.0, 100.0, 101.0, 102.0])},
        decision_at_utc=datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN).astimezone(UTC),
        yields=[
            YieldCurveRow(curve_date=date(2026, 7, 28), tenors={"2 Yr": 3.90}),
            YieldCurveRow(curve_date=date(2026, 7, 29), tenors={"2 Yr": 3.97}),
        ],
    )
    packet = _packet(june_statement, july_statement, reactions=reactions)
    pricing = packet["market_pricing"]
    assert pricing["status"] == fi.MARKET_PRICING_UNAVAILABLE
    assert pricing["proxy"]["2y_yield_change_bp"] == pytest.approx(7.0, abs=1e-6)
    assert "proxy" in pricing["proxy"]["label"].lower()


def test_build_fed_packet_market_pricing_proxy_absent_without_yields(
    june_statement, july_statement
):
    packet = _packet(june_statement, july_statement)
    assert packet["market_pricing"]["status"] == fi.MARKET_PRICING_UNAVAILABLE
    assert packet["market_pricing"]["proxy"] is None


def test_build_fed_packet_minute_reaction_keeps_both_windows(
    june_statement, july_statement
):
    decision, presser, presser_end = _decision_instants()
    reactions = fi.fomc_reaction_windows(
        {"SPY": _minute_bars(decision, statement_step=0.01, presser_step=-0.005)},
        decision_at_utc=decision,
        press_conf_at_utc=presser,
        press_conf_end_utc=presser_end,
    )
    packet = _packet(june_statement, july_statement, reactions=reactions)
    block = packet["previous_reaction"]
    assert block["basis"] == fi.REACTION_BASIS_MINUTE
    assert block["separated"] is True
    assert block["statement"]["SPY"]["return_pct"] > 0
    assert block["press_conference"]["SPY"]["return_pct"] < 0


def test_build_fed_packet_daily_reaction_is_flagged_not_separated(
    june_statement, july_statement
):
    reactions = fi.fomc_reaction_daily(
        {"SPY": _daily_bars(date(2026, 7, 27), [100.0, 100.0, 101.0, 102.0])},
        decision_at_utc=datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN).astimezone(UTC),
    )
    block = _packet(june_statement, july_statement, reactions=reactions)[
        "previous_reaction"
    ]
    assert block["basis"] == fi.REACTION_BASIS_DAILY
    assert block["separated"] is False
    assert block["label"] == "daily (no intraday bars)"
    assert block["statement"] == {} and block["press_conference"] == {}


def test_build_fed_packet_no_reaction_has_a_reason(june_statement, july_statement):
    block = _packet(june_statement, july_statement)["previous_reaction"]
    assert block["available"] is False
    assert block["separated"] is False
    assert block["reason"]


def test_build_fed_packet_as_of_hides_a_later_statement(june_statement, july_statement):
    """A statement released AFTER as_of is not visible (§14/§96)."""
    packet = _packet(
        june_statement,
        july_statement,
        as_of=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
    )
    assert packet["previous_statement"]["available"] is False
    assert packet["previous_statement"]["paragraphs"] == []
    assert packet["coverage"]["previous_statement"] is False
    # June is still visible and becomes the only side of the diff.
    assert packet["previous_statement"]["compared_to"]["available"] is True


def test_build_fed_packet_as_of_filters_speeches(june_statement, july_statement):
    packet = _packet(
        june_statement,
        july_statement,
        speeches_since=[
            {
                "speaker": "Governor A",
                "title": "Outlook",
                "released_at": datetime(2026, 8, 5, 16, 0, tzinfo=UTC),
                "url": "https://www.federalreserve.gov/newsevents/speech/a20260805a.htm",
            },
            {
                "speaker": "Governor B",
                "title": "Later",
                "released_at": datetime(2026, 9, 5, 16, 0, tzinfo=UTC),
                "url": "https://www.federalreserve.gov/newsevents/speech/b20260905a.htm",
            },
        ],
    )
    speeches = packet["subsequent_speeches"]
    assert [s["speaker"] for s in speeches] == ["Governor A"]
    assert speeches[0]["at"] == "2026-08-05T16:00:00+00:00"
    assert packet["coverage"]["subsequent_speeches"] == 1


def test_build_fed_packet_minutes_key_paragraphs_are_tagged(
    june_statement, july_statement
):
    minutes = {
        "doc_type": "MINUTES",
        "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260617.htm",
        "released_at": datetime(2026, 7, 8, 18, 0, tzinfo=UTC),
        "meeting_date": date(2026, 6, 17),
        "paragraphs": [
            "The meeting convened at 9:00 a.m.",
            "Participants noted that inflation remains elevated. "
            "The unemployment rate has changed little. "
            "Some participants judged that risks to growth were two-sided.",
        ],
    }
    packet = _packet(june_statement, july_statement, prev_minutes=minutes)
    block = packet["previous_minutes"]
    assert block["available"] is True
    assert block["url"].endswith("fomcminutes20260617.htm")
    assert block["key_paragraphs"], "tagged sentences must be selected"
    assert all(row["dimensions"] for row in block["key_paragraphs"])
    assert any("inflation" in row["text"].lower() for row in block["key_paragraphs"])


def test_build_fed_packet_minutes_after_as_of_are_hidden(june_statement, july_statement):
    minutes = {
        "url": "https://example.invalid/minutes",
        "released_at": datetime(2026, 12, 1, 18, 0, tzinfo=UTC),
        "paragraphs": ["Inflation remains elevated."],
    }
    packet = _packet(june_statement, july_statement, prev_minutes=minutes)
    assert packet["previous_minutes"]["available"] is False
    assert packet["coverage"]["previous_minutes"] is False


def test_build_fed_packet_data_block_passes_macro_prints_through(
    june_statement, july_statement
):
    prints = {
        "inflation": {"series_id": "CUSR0000SA0", "value": 2.9},
        "labor": {"series_id": "LNS14000000", "value": 4.2},
        "growth": None,
    }
    packet = _packet(june_statement, july_statement, macro_prints=prints)
    assert packet["data"]["inflation"]["value"] == 2.9
    assert packet["data"]["labor"]["value"] == 4.2
    assert packet["data"]["growth"] is None
    assert packet["data"]["available"] is True
    assert packet["coverage"]["macro_prints"] is True


def test_build_fed_packet_empty_data_block_is_unavailable(june_statement, july_statement):
    packet = _packet(june_statement, july_statement)
    assert packet["data"]["available"] is False
    assert packet["coverage"]["macro_prints"] is False


def test_build_fed_packet_tiers_label_every_section(june_statement, july_statement):
    tiers = _packet(june_statement, july_statement)["tiers"]
    assert tiers["previous_statement"] == "DATA"
    assert tiers["statement_diff"] == "QUANT"
    assert tiers["dimensions"] == "QUANT"
    assert tiers["previous_reaction"] == "QUANT"
    assert set(tiers.values()) <= {"DATA", "QUANT", "LLM", "LLM_PRIOR"}


def test_build_fed_packet_carries_the_disclaimers(june_statement, july_statement):
    disclaimers = _packet(june_statement, july_statement)["disclaimers"]
    assert fi.SOURCE_AUTHORITATIVE_NOTE in disclaimers
    assert fi.NO_SINGLE_SCORE_NOTE in disclaimers
    assert any("futures" in d for d in disclaimers)


def test_build_fed_packet_requires_utc_as_of(june_statement, july_statement):
    with pytest.raises(ValueError):
        _packet(june_statement, july_statement, as_of=datetime(2026, 8, 19, 12, 0))


def test_build_fed_packet_with_no_documents_still_has_every_section():
    packet = fi.build_fed_packet(as_of=datetime(2026, 8, 19, tzinfo=UTC))
    assert packet["previous_statement"]["available"] is False
    assert set(packet["dimensions"]) == set(fi.DIMENSIONS)
    assert packet["statement_diff"]["counts"]["TOTAL"] == 0
    assert packet["market_pricing"]["status"] == fi.MARKET_PRICING_UNAVAILABLE
    json.dumps(packet)


# ---------------------------------------------------------------------------
# §43 — the mechanical no-score guard
# ---------------------------------------------------------------------------

_SCORE_KEY = re.compile(r"score|hawk|dove", re.IGNORECASE)


def _walk_keys(node) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.append(str(key))
            out.extend(_walk_keys(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            out.extend(_walk_keys(item))
    return out


def test_packet_has_no_hawk_dove_score_key(june_statement, july_statement):
    """§43: dimensions are separate. No aggregate score key may exist."""
    decision, presser, presser_end = _decision_instants()
    packet = _packet(
        june_statement,
        july_statement,
        reactions=fi.fomc_reaction_windows(
            {"SPY": _minute_bars(decision, statement_step=0.01, presser_step=-0.005)},
            decision_at_utc=decision,
            press_conf_at_utc=presser,
            press_conf_end_utc=presser_end,
        ),
        macro_prints={"inflation": {"value": 2.9}},
        prev_minutes={
            "url": "u",
            "released_at": datetime(2026, 7, 8, tzinfo=UTC),
            "paragraphs": ["Inflation remains elevated."],
        },
        speeches_since=[
            {
                "speaker": "A",
                "title": "T",
                "released_at": datetime(2026, 8, 5, tzinfo=UTC),
                "url": "u",
            }
        ],
    )
    offenders = [k for k in _walk_keys(packet) if _SCORE_KEY.search(k)]
    assert offenders == [], f"§43 forbids an aggregate score key; found {offenders}"


def test_dimension_report_has_no_hawk_dove_score_key(june_statement, july_statement):
    diff = fi.statement_diff(
        june_statement["paragraphs"], july_statement["paragraphs"]
    )
    report = fi.dimension_report(june_statement, july_statement, diff)
    offenders = [k for k in _walk_keys(report) if _SCORE_KEY.search(k)]
    assert offenders == []


def test_module_source_defines_no_score_key():
    """Even a helper that is not reachable today may not introduce one."""
    source = Path(fi.__file__).read_text(encoding="utf-8")
    keys = re.findall(r'"([a-z_0-9]*(?:score|hawk|dove)[a-z_0-9]*)"\s*:', source, re.I)
    assert keys == []


def test_module_exports_no_score_symbol():
    """No exported symbol computes a score.

    ``NO_SINGLE_SCORE_NOTE`` is the disclaimer that says one must not exist,
    so it is the single allowed name matching the pattern.
    """
    offenders = [
        name
        for name in fi.__all__
        if _SCORE_KEY.search(name) and name != "NO_SINGLE_SCORE_NOTE"
    ]
    assert offenders == []


def test_no_single_score_note_is_stated_in_the_packet(june_statement, july_statement):
    packet = _packet(june_statement, july_statement)
    assert fi.NO_SINGLE_SCORE_NOTE in packet["disclaimers"]
    assert "hawkish/dovish" in fi.NO_SINGLE_SCORE_NOTE


# ---------------------------------------------------------------------------
# purity guard (audit §7.4)
# ---------------------------------------------------------------------------


def test_fed_intel_imports_no_io_layer():
    source = Path(fi.__file__).read_text(encoding="utf-8")
    for forbidden in ("libs.market_data", "libs.event_calendar", "apps."):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_package_exports_fed_intel_symbols():
    from libs.trading_core import events

    for name in (
        "DIMENSIONS",
        "StatementDiff",
        "build_fed_packet",
        "dimension_report",
        "fomc_reaction_daily",
        "fomc_reaction_windows",
        "policy_rate_change",
        "split_sentences",
        "statement_diff",
        "vote_dispersion",
    ):
        assert hasattr(events, name), name
        assert name in events.__all__, name
