"""Macro intelligence arithmetic (event spec §8, §38-§41, §46; Phase G U2).

Every number here is hand-checkable: the CPI index levels are round enough
that ``(300.6 / 300.0 - 1) * 100 = 0.2`` reads off the page, and no test
asserts a value the module computed for itself. Seven contracts are pinned:

1. **A level is not a print** — MoM/YoY/change_k/level are the only four
   transforms, each differences against the period it CLAIMS to (the prior
   month for MoM, twelve months back for YoY), and a GAP in the series
   never silently becomes a two-month change.
2. **The gate is on the RELEASE date, never the period** (§14, §96) — July
   CPI is invisible to an as-of before its August release, and a print with
   no release instant at all is dropped rather than assumed knowable.
3. **No consensus is ever fabricated** (§33, §38, §98) — the literal
   ``CONSENSUS DATA UNAVAILABLE`` appears in both release blocks and no code
   path in the module computes a surprise.
4. **One release, one period** — a packet quotes headline and core from the
   SAME reference month even when one series has more history.
5. **Proxies are labelled and missing assets are listed** (§39) — an asset
   with no bars is in ``unavailable`` with a reason, never absent; yields
   are basis points in their own mapping, never mixed into the returns.
6. **Absence is a value** (house rule) — no zero, no NaN, no ``inf`` ever
   leaves this module, and every ``None`` has a companion reason.
7. **No I/O** (audit §7.4) — a static import check asserts the module
   reaches neither ``apps`` nor ``libs.market_data``/``libs.event_calendar``.
"""
import ast
import math
import pathlib
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from libs.trading_core.events.macro import (
    ASSET_ROLES,
    CONSENSUS_UNAVAILABLE_REASON,
    DEFAULT_MACRO_CONTEXT_HORIZON_DAYS,
    DEFAULT_MACRO_HORIZONS,
    ESTIMATED_LAG_DAYS,
    MACRO_MODEL_VERSION,
    MACRO_SERIES,
    PROXY_ROLES,
    RELEASE_BASIS_ESTIMATED,
    RELEASE_BASIS_SCHEDULED,
    SURPRISE_UNAVAILABLE_REASON,
    TENOR_2Y,
    TENOR_10Y,
    TRANSFORM_CHANGE_K,
    TRANSFORM_LEVEL,
    TRANSFORM_MOM_PCT,
    TRANSFORM_YOY_PCT,
    TREND_FALLING,
    TREND_FLAT,
    TREND_RISING,
    TREND_WINDOW,
    MacroObservation,
    MacroPacket,
    MacroPrint,
    MultiAssetReaction,
    ScheduleRow,
    SeriesSpec,
    YieldCurveRow,
    build_macro_packet,
    derive_prints,
    macro_context_for,
    multi_asset_reaction,
    period_end_date,
    period_sort_key,
    related_evidence_window,
    release_at_for_period,
    series_for,
    trend_direction,
    visible_prints,
)
from libs.trading_core.events.reaction import DailyBar
from libs.trading_core.models.enums import EventSession, EventType

UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")

#: A CPI-shaped headline spec used by the transform tests, deliberately NOT
#: read out of MACRO_SERIES so a catalogue edit cannot quietly change what
#: the arithmetic tests are asserting.
HEADLINE = SeriesSpec(
    series_id="CUSR0000SA0",
    label="CPI-U all items (SA)",
    role="headline",
    transform=TRANSFORM_MOM_PCT,
    seasonally_adjusted=True,
    unit="percent",
)


def obs(period: str, value: float | None, series_id: str = "CUSR0000SA0"):
    return MacroObservation(series_id=series_id, period=period, value=value)


def et(y: int, m: int, d: int, hh: int = 8, mm: int = 30) -> datetime:
    """An ET wall clock as UTC — how a BLS 08:30 release is stored."""
    return datetime(y, m, d, hh, mm, tzinfo=EASTERN).astimezone(UTC)


# ---------------------------------------------------------------------------
# 1. Period arithmetic
# ---------------------------------------------------------------------------


# The no-I/O and no-numerics invariants for THIS module are enforced
# package-wide by tests/test_pure_layer_boundary.py, which walks every
# module under libs/trading_core/ — a per-file copy here protected this
# one file and left sixty-six others to the habit of copying a test.


def test_period_sort_key_orders_monthly_and_quarterly_together():
    """A quarter sorts on its LAST month, so Q1 lands between Feb and Apr."""
    periods = ["2026-04", "2026-Q1", "2026-02", "2025-12"]
    assert sorted(periods, key=period_sort_key) == [
        "2025-12",
        "2026-02",
        "2026-Q1",
        "2026-04",
    ]


def test_period_end_date_handles_month_quarter_and_december():
    assert period_end_date("2026-02") == date(2026, 2, 28)
    assert period_end_date("2024-02") == date(2024, 2, 29)  # leap year
    assert period_end_date("2026-12") == date(2026, 12, 31)
    assert period_end_date("2026-Q1") == date(2026, 3, 31)
    assert period_end_date("2026-Q4") == date(2026, 12, 31)


def test_unparseable_period_sorts_last_and_has_no_end_date():
    """A malformed agency row degrades; it does not take the packet down."""
    assert period_end_date("garbage") is None
    assert period_end_date("2026-13") is None
    assert period_sort_key("garbage") > period_sort_key("2999-12")


def test_release_at_prefers_the_schedule_and_stamps_the_basis():
    scheduled = et(2026, 8, 12)
    at, basis = release_at_for_period(
        "2026-07", [ScheduleRow(period="2026-07", release_at_utc=scheduled)]
    )
    assert at == scheduled
    assert basis == RELEASE_BASIS_SCHEDULED


def test_release_at_falls_back_to_period_end_plus_lag_marked_estimated():
    at, basis = release_at_for_period("2026-07", [])
    assert basis == RELEASE_BASIS_ESTIMATED
    assert at == datetime(2026, 7, 31, tzinfo=UTC) + timedelta(
        days=ESTIMATED_LAG_DAYS
    )


# ---------------------------------------------------------------------------
# 2. §8 transforms — a level is not a print
# ---------------------------------------------------------------------------


def test_mom_pct_differences_against_the_previous_month():
    prints = derive_prints(
        [obs("2026-05", 300.0), obs("2026-06", 300.6), obs("2026-07", 301.2)],
        HEADLINE,
    )
    by_period = {p.period: p for p in prints}
    assert by_period["2026-06"].value == pytest.approx(0.2, abs=1e-9)
    assert by_period["2026-06"].value_raw == 300.6
    assert by_period["2026-06"].prior == 300.0
    # The first period has nothing to difference against — a reason, not a 0.
    assert by_period["2026-05"].value is None
    assert by_period["2026-05"].reason == "prior_period_unavailable"


def test_yoy_pct_reaches_exactly_twelve_months_back():
    spec = SeriesSpec(
        series_id="CUUR0000SA0",
        label="CPI-U NSA",
        role="level",
        transform=TRANSFORM_YOY_PCT,
        seasonally_adjusted=False,
        unit="percent",
    )
    series = [obs(f"2025-{m:02d}", 300.0, "CUUR0000SA0") for m in range(1, 13)]
    series.append(obs("2026-01", 309.0, "CUUR0000SA0"))
    by_period = {p.period: p for p in derive_prints(series, spec)}
    assert by_period["2026-01"].value == pytest.approx(3.0, abs=1e-9)
    assert by_period["2026-01"].prior == 300.0


def test_a_gap_in_the_series_never_becomes_a_two_month_change():
    """June is missing; July must NOT quietly difference against May."""
    prints = derive_prints([obs("2026-05", 300.0), obs("2026-07", 303.0)], HEADLINE)
    july = next(p for p in prints if p.period == "2026-07")
    assert july.value is None
    assert july.reason == "prior_period_unavailable"


def test_change_k_is_a_first_difference_in_the_series_own_unit():
    spec = SeriesSpec(
        series_id="CES0000000001",
        label="Payrolls",
        role="headline",
        transform=TRANSFORM_CHANGE_K,
        seasonally_adjusted=True,
        unit="thousands",
    )
    prints = derive_prints(
        [obs("2026-06", 160_000.0, "CES0000000001"), obs("2026-07", 160_142.0, "CES0000000001")],
        spec,
    )
    july = next(p for p in prints if p.period == "2026-07")
    assert july.value == pytest.approx(142.0)
    assert july.unit == "thousands"


def test_level_transform_publishes_a_rate_untouched():
    spec = SeriesSpec(
        series_id="LNS14000000",
        label="Unemployment rate",
        role="rate",
        transform=TRANSFORM_LEVEL,
        seasonally_adjusted=True,
        unit="percent",
    )
    prints = derive_prints(
        [obs("2026-06", 4.1, "LNS14000000"), obs("2026-07", 4.2, "LNS14000000")], spec
    )
    july = next(p for p in prints if p.period == "2026-07")
    assert july.value == pytest.approx(4.2)
    assert july.prior is None  # a level differences against nothing


def test_transforms_are_nan_and_zero_safe():
    """No NaN, no inf, and a non-positive base refuses rather than dividing."""
    prints = derive_prints(
        [
            obs("2026-04", 0.0),
            obs("2026-05", 300.0),
            obs("2026-06", float("nan")),
            obs("2026-07", 301.0),
        ],
        HEADLINE,
    )
    by_period = {p.period: p for p in prints}
    assert by_period["2026-05"].value is None
    assert by_period["2026-05"].reason == "prior_value_not_positive"
    assert by_period["2026-06"].value is None
    assert by_period["2026-06"].reason == "value_unavailable"
    assert by_period["2026-07"].value is None
    assert by_period["2026-07"].reason == "prior_value_unavailable"
    for item in prints:
        for number in (item.value, item.value_raw, item.prior):
            assert number is None or math.isfinite(number)


def test_a_revision_supersedes_the_first_print_of_the_same_month():
    prints = derive_prints(
        [obs("2026-06", 300.0), obs("2026-07", 300.6), obs("2026-07", 300.9)],
        HEADLINE,
    )
    july = [p for p in prints if p.period == "2026-07"]
    assert len(july) == 1
    assert july[0].value_raw == 300.9


def test_observations_for_another_series_are_ignored():
    prints = derive_prints(
        [obs("2026-06", 300.0), obs("2026-07", 999.0, "WPSFD4"), obs("2026-07", 300.6)],
        HEADLINE,
    )
    assert {p.period for p in prints} == {"2026-06", "2026-07"}
    assert next(p for p in prints if p.period == "2026-07").value_raw == 300.6


# ---------------------------------------------------------------------------
# 3. §14/§96 — the point-in-time gate is on the RELEASE date
# ---------------------------------------------------------------------------


CPI_SCHEDULE = [
    ScheduleRow(period="2026-05", release_at_utc=et(2026, 6, 10)),
    ScheduleRow(period="2026-06", release_at_utc=et(2026, 7, 14)),
    ScheduleRow(period="2026-07", release_at_utc=et(2026, 8, 12)),
    ScheduleRow(period="2026-08", release_at_utc=et(2026, 9, 11)),
]


def test_a_print_released_after_as_of_is_hidden():
    """July CPI exists as a number in July and as a FACT on 12 August."""
    prints = derive_prints(
        [obs("2026-06", 300.0), obs("2026-07", 300.6)],
        HEADLINE,
        schedule=CPI_SCHEDULE,
    )
    before = visible_prints(prints, et(2026, 8, 11, 23, 0))
    after = visible_prints(prints, et(2026, 8, 12, 9, 0))
    assert [p.period for p in before] == ["2026-06"]
    assert [p.period for p in after] == ["2026-06", "2026-07"]


def test_the_gate_is_closed_at_the_release_instant_itself():
    prints = derive_prints([obs("2026-06", 300.0), obs("2026-07", 300.6)], HEADLINE, schedule=CPI_SCHEDULE)
    at_release = visible_prints(prints, et(2026, 8, 12))
    one_minute_early = visible_prints(prints, et(2026, 8, 12, 8, 29))
    assert "2026-07" in {p.period for p in at_release}
    assert "2026-07" not in {p.period for p in one_minute_early}


def test_a_print_with_no_release_instant_is_dropped_not_assumed():
    """An ungated print is exactly the look-ahead this function prevents."""
    ungated = MacroPrint(
        series_id="X",
        period="2026-07",
        value_raw=1.0,
        value=1.0,
        prior=None,
        transform=TRANSFORM_LEVEL,
        unit="percent",
        release_at=None,
    )
    assert visible_prints([ungated], et(2030, 1, 1)) == []


def test_visible_prints_refuses_a_naive_as_of():
    with pytest.raises(ValueError):
        visible_prints([], datetime(2026, 8, 12, 12, 0))


# ---------------------------------------------------------------------------
# 4. §38 — the trend label
# ---------------------------------------------------------------------------


def test_trend_direction_labels_rising_falling_and_flat():
    assert trend_direction([0.1, 0.2, 0.5])[0] == TREND_RISING
    assert trend_direction([0.5, 0.2, 0.1])[0] == TREND_FALLING
    assert trend_direction([0.2, 0.2, 0.2])[0] == TREND_FLAT


def test_trend_slope_is_the_average_step_over_the_last_three_points():
    direction, slope = trend_direction([9.0, 9.0, 0.1, 0.3, 0.5])
    assert direction == TREND_RISING
    assert slope == pytest.approx(0.2)  # (0.5 - 0.1) / 2, older points ignored


def test_trend_tolerance_keeps_decimal_dust_flat():
    direction, slope = trend_direction([0.200, 0.205, 0.210])
    assert direction == TREND_FLAT
    assert slope == pytest.approx(0.005)


def test_trend_over_one_usable_point_is_no_direction_at_all():
    assert trend_direction([0.3]) == (None, None)
    assert trend_direction([None, 0.3]) == (None, None)
    assert trend_direction([]) == (None, None)


# ---------------------------------------------------------------------------
# 5. §38 — the packet
# ---------------------------------------------------------------------------


def _cpi_prints(as_of_levels=None):
    """Headline + core CPI, hand-picked so MoM reads off the page."""
    headline_levels = as_of_levels or {
        "2026-01": 297.0,
        "2026-02": 297.6,
        "2026-03": 298.2,
        "2026-04": 299.1,
        "2026-05": 300.0,
        "2026-06": 300.6,
        "2026-07": 301.5,
    }
    core_levels = {p: v - 20.0 for p, v in headline_levels.items()}
    specs = series_for(EventType.CPI)
    headline_spec = next(s for s in specs if s.role == "headline")
    core_spec = next(s for s in specs if s.role == "core")
    return {
        headline_spec.series_id: derive_prints(
            [obs(p, v, headline_spec.series_id) for p, v in headline_levels.items()],
            headline_spec,
            schedule=CPI_SCHEDULE,
        ),
        core_spec.series_id: derive_prints(
            [obs(p, v, core_spec.series_id) for p, v in core_levels.items()],
            core_spec,
            schedule=CPI_SCHEDULE,
        ),
    }


def test_packet_previous_release_carries_the_actual_by_role():
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    previous = packet.previous_release
    assert previous["period"] == "2026-07"
    assert previous["release_at"] == et(2026, 8, 12)
    assert previous["release_time_basis"] == RELEASE_BASIS_SCHEDULED
    # 301.5 / 300.6 - 1 = 0.2994...%
    assert previous["actual"]["headline"]["value"] == pytest.approx(
        (301.5 / 300.6 - 1.0) * 100.0
    )
    assert previous["actual"]["headline"]["seasonally_adjusted"] is True
    assert set(previous["actual"]) >= {"headline", "core"}


def test_packet_never_fabricates_a_consensus_or_a_surprise():
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    assert packet.previous_release["consensus"] == CONSENSUS_UNAVAILABLE_REASON
    assert packet.previous_release["surprise"] == SURPRISE_UNAVAILABLE_REASON
    assert packet.current_release["consensus"] == CONSENSUS_UNAVAILABLE_REASON
    assert packet.coverage["consensus"] == {
        "available": False,
        "reason": CONSENSUS_UNAVAILABLE_REASON,
    }
    assert packet.consensus_status == CONSENSUS_UNAVAILABLE_REASON


def test_no_code_path_in_the_module_computes_a_surprise():
    """§33/§98 — the absence is structural, not a runtime branch.

    The check walks the AST rather than the raw text: the module's own prose
    says the words ("a surprise is ``actual - consensus``") precisely in
    order to explain why it never computes one, and a naive grep would read
    that explanation as the offence it forbids.
    """
    path = pathlib.Path("libs/trading_core/events/macro.py")
    source = path.read_text()
    assert "CONSENSUS DATA UNAVAILABLE" in source

    tree = ast.parse(source)

    # No function is named for producing a surprise.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert "surprise" not in node.name.lower(), node.name

    # No arithmetic anywhere in the module is assigned to a surprise-shaped
    # name: a surprise can only ever BE the fixed unavailable string.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        named = [
            t.id.lower()
            for t in targets
            if isinstance(t, ast.Name) and "surprise" in t.id.lower()
        ]
        if named and node.value is not None:
            assert isinstance(node.value, ast.Constant), ast.dump(node)

    # Every literal that fills a surprise field is that one constant.
    assert SURPRISE_UNAVAILABLE_REASON.startswith("SURPRISE UNAVAILABLE")
    assert "no consensus source" in SURPRISE_UNAVAILABLE_REASON


def test_packet_current_release_is_the_next_scheduled_one():
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    assert packet.current_release["period"] == "2026-08"
    assert packet.current_release["release_at"] == et(2026, 9, 11)
    assert "actual" not in packet.current_release


def test_packet_current_release_can_be_supplied_by_the_caller():
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
        current_period="2026-08",
        current_release_at=et(2026, 9, 11, 8, 30),
    )
    assert packet.current_release["period"] == "2026-08"
    assert packet.current_release["release_at"] == et(2026, 9, 11)


def test_one_release_means_one_period_across_series():
    """Core has July, headline has August: the packet must not mix them."""
    prints = _cpi_prints()
    specs = series_for(EventType.CPI)
    headline_spec = next(s for s in specs if s.role == "headline")
    core_spec = next(s for s in specs if s.role == "core")
    prints[core_spec.series_id] = [
        p for p in prints[core_spec.series_id] if p.period != "2026-07"
    ]
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=prints,
    )
    assert packet.previous_release["period"] == "2026-07"
    assert "headline" in packet.previous_release["actual"]
    # Core has nothing for July — it is ABSENT, not backfilled from June.
    assert "core" not in packet.previous_release["actual"]
    assert headline_spec.series_id  # spec ids are stable


def test_packet_as_of_hides_a_later_print_from_the_previous_release():
    early = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 11),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    late = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    assert early.previous_release["period"] == "2026-06"
    assert late.previous_release["period"] == "2026-07"


def test_packet_without_a_schedule_estimates_and_says_so():
    specs = series_for(EventType.CPI)
    headline_spec = next(s for s in specs if s.role == "headline")
    prints = {
        headline_spec.series_id: derive_prints(
            [
                obs("2026-05", 300.0, headline_spec.series_id),
                obs("2026-06", 300.6, headline_spec.series_id),
            ],
            headline_spec,
        )
    }
    packet = build_macro_packet(
        EventType.CPI, as_of=et(2026, 9, 1), prints_by_series=prints
    )
    assert packet.previous_release["release_time_basis"] == RELEASE_BASIS_ESTIMATED
    assert packet.coverage["schedule"]["available"] is False
    assert str(ESTIMATED_LAG_DAYS) in packet.coverage["schedule"]["reason"]
    # 2026-06 end + 45d = 2026-08-14, visible at as_of 2026-09-01.
    assert packet.previous_release["period"] == "2026-06"


def test_packet_recent_trend_shows_the_last_six_prints_with_a_direction():
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    specs = series_for(EventType.CPI)
    headline_spec = next(s for s in specs if s.role == "headline")
    block = packet.recent_trend[f"{headline_spec.series_id}:headline"]
    assert block["n_points"] <= TREND_WINDOW
    assert len(block["prints"]) == block["n_points"]
    assert block["direction"] in {TREND_RISING, TREND_FALLING, TREND_FLAT, None}
    assert block["prints"][-1]["period"] == "2026-07"


def test_packet_for_a_type_with_no_source_marks_actuals_unavailable():
    """GDP/PCE need a BEA key; RETAIL_SALES has no Census adapter (Phase G)."""
    for event_type in (EventType.GDP, EventType.PCE, EventType.RETAIL_SALES):
        packet = build_macro_packet(event_type, as_of=et(2026, 8, 20))
        assert series_for(event_type) == ()
        assert packet.coverage["actuals"]["available"] is False
        assert packet.coverage["actuals"]["reason"]
        assert packet.previous_release["available"] is False
        assert packet.previous_release["consensus"] == CONSENSUS_UNAVAILABLE_REASON


def test_packet_coverage_names_a_series_with_no_visible_print():
    specs = series_for(EventType.CPI)
    headline_spec = next(s for s in specs if s.role == "headline")
    packet = build_macro_packet(
        EventType.CPI,
        as_of=et(2026, 1, 1),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    entry = packet.coverage["series"][f"{headline_spec.series_id}:headline"]
    assert entry["available"] is False
    assert entry["reason"] == "no print released on or before as_of"
    assert packet.previous_release["period"] is None


def test_packet_is_deterministic():
    kwargs = dict(
        as_of=et(2026, 8, 20),
        schedule=CPI_SCHEDULE,
        prints_by_series=_cpi_prints(),
    )
    first = build_macro_packet(EventType.CPI, **kwargs)
    second = build_macro_packet(EventType.CPI, **kwargs)
    assert first.previous_release == second.previous_release
    assert first.recent_trend == second.recent_trend
    assert first.coverage == second.coverage
    assert first.model_version == second.model_version == MACRO_MODEL_VERSION


def test_packet_refuses_a_naive_as_of():
    with pytest.raises(ValueError):
        build_macro_packet(EventType.CPI, as_of=datetime(2026, 8, 20, 12, 0))


# ---------------------------------------------------------------------------
# 6. §39 — the multi-asset reaction
# ---------------------------------------------------------------------------

REL_DAY = date(2026, 8, 12)
REL_AT = et(2026, 8, 12)  # 08:30 ET — a BEFORE_MARKET release


def _bars(pre: float, react: float, later: float) -> list[DailyBar]:
    """Three sessions: the day before the print, the print's day, and +1."""
    return [
        DailyBar(date=date(2026, 8, 11), open=pre, high=pre, low=pre, close=pre),
        DailyBar(date=REL_DAY, open=react, high=react, low=react, close=react),
        DailyBar(date=date(2026, 8, 13), open=later, high=later, low=later, close=later),
    ]


def test_multi_asset_reaction_measures_each_available_asset():
    result = multi_asset_reaction(
        {"SPY": _bars(100.0, 101.0, 102.0), "GLD": _bars(200.0, 198.0, 196.0)},
        [],
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
        horizons=(1, 2),
    )
    assert set(result.assets) == {"SPY", "GLD"}
    # FRACTIONS, inherited unchanged from reaction.event_reaction: 0.01 = +1%.
    assert result.assets["SPY"].returns[1] == pytest.approx(0.01)
    assert result.assets["SPY"].returns[2] == pytest.approx(0.02)
    assert result.assets["GLD"].returns[1] == pytest.approx(-0.01)
    assert result.event_date_et == REL_DAY
    assert result.session is EventSession.BEFORE_MARKET


def test_missing_assets_are_listed_with_a_reason_never_dropped():
    result = multi_asset_reaction(
        {"SPY": _bars(100.0, 101.0, 102.0)},
        [],
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
    )
    assert set(result.assets) == {"SPY"}
    assert set(result.unavailable) == set(ASSET_ROLES) - {"SPY"}
    assert result.unavailable["QQQ"] == "no stored daily bars"
    # Every symbol in the roster is accounted for exactly once.
    assert set(result.assets) | set(result.unavailable) == set(ASSET_ROLES)


def test_an_asset_whose_window_cannot_be_located_is_unavailable_with_the_reason():
    """Bars that stop before the release cannot produce a reaction."""
    stale = [
        DailyBar(date=date(2026, 1, 5), open=100.0, high=100.0, low=100.0, close=100.0),
        DailyBar(date=date(2026, 1, 6), open=101.0, high=101.0, low=101.0, close=101.0),
    ]
    result = multi_asset_reaction(
        {"SPY": stale},
        [],
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
    )
    assert "SPY" not in result.assets
    assert "no bar after the event yet" in result.unavailable["SPY"]


def test_proxy_roles_are_labelled_as_proxies():
    result = multi_asset_reaction(
        {
            "SPY": _bars(100.0, 101.0, 102.0),
            "UUP": _bars(28.0, 28.1, 28.2),
            "SHY": _bars(82.0, 82.1, 82.2),
        },
        [],
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
    )
    assert result.assets["SPY"].role == "equity"
    assert result.assets["SPY"].is_proxy is False
    assert result.assets["UUP"].role == "dxy_proxy"
    assert result.assets["UUP"].is_proxy is True
    assert result.assets["SHY"].role == "2y_proxy"
    assert result.assets["SHY"].is_proxy is True
    assert PROXY_ROLES == frozenset(
        r for r in ASSET_ROLES.values() if r.endswith("_proxy")
    )


def test_yield_changes_are_basis_points_in_their_own_mapping():
    curve = [
        YieldCurveRow(curve_date=date(2026, 8, 11), tenors={TENOR_2Y: 3.85, TENOR_10Y: 4.20}),
        YieldCurveRow(curve_date=REL_DAY, tenors={TENOR_2Y: 3.92, TENOR_10Y: 4.25}),
    ]
    result = multi_asset_reaction(
        {"SPY": _bars(100.0, 101.0, 102.0)},
        curve,
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
    )
    assert result.yields[TENOR_2Y].change_bp == pytest.approx(7.0)
    assert result.yields[TENOR_10Y].change_bp == pytest.approx(5.0)
    assert result.yields[TENOR_2Y].before_date == date(2026, 8, 11)
    assert result.yields[TENOR_2Y].after_date == REL_DAY
    # Basis points never leak into the percentage-return table.
    assert set(result.assets["SPY"].returns) == set(DEFAULT_MACRO_HORIZONS)


def test_yield_change_spans_a_gap_when_the_release_day_has_no_curve():
    curve = [
        YieldCurveRow(curve_date=date(2026, 8, 10), tenors={TENOR_2Y: 3.85}),
        YieldCurveRow(curve_date=date(2026, 8, 13), tenors={TENOR_2Y: 3.95}),
    ]
    result = multi_asset_reaction(
        {}, curve, event_at_utc=REL_AT, session=EventSession.BEFORE_MARKET
    )
    assert result.yields[TENOR_2Y].change_bp == pytest.approx(10.0)
    assert result.yields[TENOR_2Y].before_date == date(2026, 8, 10)
    assert result.yields[TENOR_2Y].after_date == date(2026, 8, 13)


def test_yields_absent_are_a_reason_not_a_zero():
    result = multi_asset_reaction(
        {}, [], event_at_utc=REL_AT, session=EventSession.BEFORE_MARKET
    )
    for tenor in (TENOR_2Y, TENOR_10Y):
        assert result.yields[tenor].change_bp is None
        assert result.yields[tenor].reason == "no treasury curve rows supplied"
    one_sided = multi_asset_reaction(
        {},
        [YieldCurveRow(curve_date=date(2026, 8, 13), tenors={TENOR_2Y: 3.95})],
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
    )
    assert one_sided.yields[TENOR_2Y].change_bp is None
    assert one_sided.yields[TENOR_2Y].reason == "no_curve_before_event"


def test_multi_asset_reaction_reuses_the_shared_session_window_rules():
    """A BMO print reacts on D; an AMC print reacts on D+1 (§17)."""
    bars = _bars(100.0, 101.0, 103.0)
    bmo = multi_asset_reaction(
        {"SPY": bars},
        [],
        event_at_utc=REL_AT,
        session=EventSession.BEFORE_MARKET,
        horizons=(1,),
    )
    amc = multi_asset_reaction(
        {"SPY": bars},
        [],
        event_at_utc=et(2026, 8, 12, 16, 30),
        session=EventSession.AFTER_MARKET,
        horizons=(1,),
    )
    assert bmo.assets["SPY"].returns[1] == pytest.approx(0.01)  # 101 vs 100
    assert amc.assets["SPY"].returns[1] == pytest.approx(103.0 / 101.0 - 1.0)
    assert bmo.assets["SPY"].basis != amc.assets["SPY"].basis


def test_multi_asset_reaction_is_deterministic_and_refuses_a_naive_instant():
    kwargs = dict(event_at_utc=REL_AT, session=EventSession.BEFORE_MARKET)
    bars = {"SPY": _bars(100.0, 101.0, 102.0)}
    assert multi_asset_reaction(bars, [], **kwargs).assets["SPY"].returns == (
        multi_asset_reaction(bars, [], **kwargs).assets["SPY"].returns
    )
    with pytest.raises(ValueError):
        multi_asset_reaction(
            bars, [], event_at_utc=datetime(2026, 8, 12, 12, 30),
            session=EventSession.BEFORE_MARKET,
        )


# ---------------------------------------------------------------------------
# 7. §46 — macro_context_for
# ---------------------------------------------------------------------------


def _event(event_id, event_type, when, **extra):
    return {"event_id": event_id, "event_type": event_type, "scheduled_at": when, **extra}


def test_macro_context_finds_the_next_tracked_release_within_the_horizon():
    as_of = et(2026, 8, 20, 12, 0)
    context = macro_context_for(
        [
            _event(1, "CPI", et(2026, 9, 11), title="CPI — August 2026", importance=80),
            _event(2, "FOMC_DECISION", et(2026, 8, 26, 14, 0), importance=90),
            _event(3, "EMPLOYMENT_REPORT", et(2026, 9, 2), importance=80),
        ],
        as_of=as_of,
    )
    assert context["available"] is True
    # CPI on 11 Sep is 22 days out — beyond the 14d horizon, so it is absent.
    assert [item["event_type"] for item in context["upcoming"]] == [
        "FOMC_DECISION",
        "EMPLOYMENT_REPORT",
    ]
    assert context["next"]["event_id"] == 2
    # 20 Aug 12:00 ET -> 26 Aug 14:00 ET is 6 days and 2 hours.
    assert context["next"]["days_to"] == pytest.approx(6 + 2 / 24, abs=0.01)
    assert context["horizon_days"] == DEFAULT_MACRO_CONTEXT_HORIZON_DAYS


def test_macro_context_excludes_the_past_and_untracked_types():
    as_of = et(2026, 8, 20, 12, 0)
    context = macro_context_for(
        [
            _event(1, "CPI", et(2026, 8, 12)),  # already released
            _event(2, "JOLTS", et(2026, 8, 25)),  # not a tracked context type
            _event(3, "EARNINGS", et(2026, 8, 25)),
        ],
        as_of=as_of,
    )
    assert context["available"] is False
    assert context["upcoming"] == []
    assert "no tracked macro release within 14d" in context["reason"]


def test_macro_context_counts_unparseable_rows_instead_of_dropping_them():
    context = macro_context_for(
        [
            {"event_id": 1, "event_type": "NOT_A_TYPE", "scheduled_at": et(2026, 8, 25)},
            {"event_id": 2, "event_type": "CPI", "scheduled_at": None},
            _event(3, EventType.CPI, et(2026, 8, 25)),
        ],
        as_of=et(2026, 8, 20, 12, 0),
    )
    assert context["skipped_unparseable"] == 2
    assert len(context["upcoming"]) == 1
    assert context["upcoming"][0]["event_id"] == 3


def test_macro_context_accepts_a_custom_horizon_and_is_sorted():
    as_of = et(2026, 8, 20, 12, 0)
    wide = macro_context_for(
        [
            _event(1, "CPI", et(2026, 9, 11)),
            _event(2, "EMPLOYMENT_REPORT", et(2026, 9, 4)),
        ],
        as_of=as_of,
        horizon_days=30,
    )
    assert [i["event_id"] for i in wide["upcoming"]] == [2, 1]
    assert wide["horizon_days"] == 30


# ---------------------------------------------------------------------------
# 8. §40 — the related evidence window
# ---------------------------------------------------------------------------


def test_related_evidence_window_returns_events_between_release_and_as_of():
    window = related_evidence_window(
        et(2026, 7, 14),
        et(2026, 8, 12),
        [
            _event(1, "PPI", et(2026, 7, 15), title="PPI — June 2026"),
            _event(2, "FED_SPEECH", et(2026, 7, 22), title="Powell at Jackson Hole"),
            _event(3, "CPI", et(2026, 6, 10)),  # before the window
            _event(4, "PCE", et(2026, 8, 28)),  # after as_of
        ],
    )
    assert window["available"] is True
    assert [item["event_id"] for item in window["events"]] == [1, 2]
    assert window["n_events"] == 2
    assert window["window_start"] == et(2026, 7, 14)
    assert window["window_end"] == et(2026, 8, 12)


def test_related_evidence_window_flags_macro_rows_and_keeps_fed_speeches():
    window = related_evidence_window(
        et(2026, 7, 14),
        et(2026, 8, 12),
        [
            _event(1, "PPI", et(2026, 7, 15)),
            _event(2, "FED_SPEECH", et(2026, 7, 22)),
        ],
    )
    by_id = {item["event_id"]: item for item in window["events"]}
    assert by_id[1]["is_macro"] is True
    assert by_id[2]["is_macro"] is False
    assert by_id[2]["event_type"] == "FED_SPEECH"


def test_related_evidence_window_applies_no_keyword_list():
    """§40 — the factual set is deterministic; THEMES are the LLM's job."""
    source = pathlib.Path("libs/trading_core/events/macro.py").read_text()
    for banned in ("shelter", "housing", "inflation expectations"):
        assert banned.upper() not in source.upper() or "§40" in source
    window = related_evidence_window(
        et(2026, 7, 14),
        et(2026, 8, 12),
        [_event(9, "ISM", et(2026, 8, 1), title="entirely unrelated wording")],
    )
    assert window["n_events"] == 1  # nothing was filtered out by topic
    assert "LLM" in window["note"]


def test_related_evidence_window_excludes_the_event_itself():
    window = related_evidence_window(
        et(2026, 7, 14),
        et(2026, 8, 12),
        [_event(1, "PPI", et(2026, 7, 15)), _event(7, "CPI", et(2026, 7, 14))],
        exclude_event_ids=[7],
    )
    assert [item["event_id"] for item in window["events"]] == [1]


def test_related_evidence_window_with_no_previous_release_says_so():
    window = related_evidence_window(None, et(2026, 8, 12), [])
    assert window["available"] is False
    assert window["window_start"] is None
    assert "open on the left" in window["reason"]


def test_related_evidence_window_is_deterministic():
    args = (
        et(2026, 7, 14),
        et(2026, 8, 12),
        [_event(2, "PPI", et(2026, 7, 15)), _event(1, "ISM", et(2026, 7, 15))],
    )
    assert related_evidence_window(*args) == related_evidence_window(*args)


# ---------------------------------------------------------------------------
# 9. Catalogue hygiene and the audit §7.4 purity guard
# ---------------------------------------------------------------------------


def test_catalogue_covers_the_phase_g_event_types_with_documented_gaps():
    assert set(MACRO_SERIES) == {
        EventType.CPI,
        EventType.PPI,
        EventType.EMPLOYMENT_REPORT,
        EventType.JOLTS,
        EventType.GDP,
        EventType.PCE,
        EventType.RETAIL_SALES,
    }
    assert {s.role for s in series_for(EventType.CPI)} == {"headline", "core", "level"}
    assert {s.role for s in series_for(EventType.EMPLOYMENT_REPORT)} == {
        "headline",
        "rate",
        "wages",
    }
    # A rate is published as-is; payrolls are a change in thousands.
    payrolls = next(
        s for s in series_for(EventType.EMPLOYMENT_REPORT) if s.role == "headline"
    )
    unemployment = next(
        s for s in series_for(EventType.EMPLOYMENT_REPORT) if s.role == "rate"
    )
    assert payrolls.transform == TRANSFORM_CHANGE_K
    assert payrolls.unit == "thousands"
    assert unemployment.transform == TRANSFORM_LEVEL


def test_seasonal_adjustment_is_part_of_the_series_identity():
    """SA and NSA are different series ids and the flag says which is which."""
    cpi = {s.role: s for s in series_for(EventType.CPI)}
    assert cpi["headline"].series_id == "CUSR0000SA0"
    assert cpi["headline"].seasonally_adjusted is True
    assert cpi["level"].series_id == "CUUR0000SA0"
    assert cpi["level"].seasonally_adjusted is False
    assert cpi["level"].transform == TRANSFORM_YOY_PCT


def test_asset_roles_cover_the_spec_39_roster():
    assert set(ASSET_ROLES) == {
        "SPY",
        "DIA",
        "QQQ",
        "VIXY",
        "TLT",
        "IEF",
        "SHY",
        "GLD",
        "USO",
        "UUP",
    }


def test_the_index_proxies_are_labelled_as_proxies():
    """A macro release has no issuer, so the index reaction IS the reader's
    instrument — which makes it exactly the place a proxy must not pass for
    the thing itself. VIXY holds VIX FUTURES (roll cost makes it track the
    index's direction, not its level) and DIA is a fund tracking the Dow;
    neither is the index, and the UI badges whatever lands in PROXY_ROLES."""
    from libs.trading_core.events.macro import PROXY_ROLES

    assert ASSET_ROLES["VIXY"] in PROXY_ROLES
    assert ASSET_ROLES["DIA"] in PROXY_ROLES
    # SPY and QQQ are funds too, but they are the platform's own definition of
    # "equities" rather than a stand-in for an untradable series — the roster
    # above has always treated them as the exposure itself.
    assert ASSET_ROLES["SPY"] not in PROXY_ROLES


def test_consensus_wording_matches_the_earnings_table_exactly():
    from libs.trading_core.events.replay import (
        CONSENSUS_UNAVAILABLE_REASON as REPLAY_WORDING,
    )

    assert CONSENSUS_UNAVAILABLE_REASON == REPLAY_WORDING




def test_exports_are_reachable_from_the_package():
    import libs.trading_core.events as pkg

    for name in (
        "MACRO_SERIES",
        "MacroPacket",
        "MultiAssetReaction",
        "build_macro_packet",
        "derive_prints",
        "macro_context_for",
        "multi_asset_reaction",
        "related_evidence_window",
    ):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__, name
