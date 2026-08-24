"""Implied move & IV crush (event spec §18, §36, §37, §47, §66; Phase I U2).

Every number here is hand-checkable: the straddles are priced at round
values so ``(6 + 4) / 100 = 0.10`` reads off the page, and no test asserts a
value the module computed for itself. Six contracts are pinned:

1. **The straddle is the whole implied move** — ``(call + put) / spot``, no
   vol model, and a MISSING LEG NEVER HALVES IT: a one-sided straddle is
   ``None`` with a reason, never the surviving leg's price.
2. **Expiry selection respects the session** (§18) — an AFTER_MARKET or
   DURING_MARKET release skips a same-day expiry (it expires on, or before,
   the news); BEFORE_MARKET and UNKNOWN keep it.
3. **The §66 bands are closed at FAIR** — exactly 0.8 and exactly 1.2 are
   FAIR, so UNDER/OVER are strictly beyond the band.
4. **Absence is a value** (house rule) — no zero, no NaN, no ``inf`` ever
   leaves this module, and every ``None`` has a companion note.
5. **The §37 wording and the two basis labels are fixed strings** — the API
   payload and the UI both render them verbatim, so they are pinned here.
6. **No I/O** (audit §7.4) — a static import check asserts the module reaches
   neither ``apps`` nor ``libs.market_data``.
"""
import ast
import math
import pathlib
from datetime import date

import pytest

from libs.trading_core.events.implied_move import (
    BASIS_HISTORICAL,
    BASIS_LIVE,
    CLASSIFICATION_FAIR,
    CLASSIFICATION_OVER,
    CLASSIFICATION_UNDER,
    DISCLAIMER,
    METHOD_IV_SQRT_T,
    METHOD_STRADDLE,
    STATUS_NO_DATA,
    STATUS_OK,
    STATUS_PARTIAL,
    ImpliedMove,
    ImpliedMoveSummary,
    build_summary,
    event_iv,
    historical_move_stats,
    implied_move_from_iv,
    implied_vs_realized,
    iv_crush,
    nearest_strike,
    select_event_expiry,
    straddle_implied_move,
)
from libs.trading_core.events.reaction import percentile_nearest_rank
from libs.trading_core.models.enums import EventSession
from libs.trading_core.options.bs import bs_price

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "libs"
    / "trading_core"
    / "events"
    / "implied_move.py"
)


# ---------------------------------------------------------------------------
# 1. select_event_expiry (§18)
# ---------------------------------------------------------------------------


EXPIRIES = [date(2026, 8, 21), date(2026, 8, 28), date(2026, 9, 4)]


def test_select_expiry_takes_first_on_or_after_event_for_bmo():
    """A BMO print IS priced by that day's close, so same-day is valid."""
    assert (
        select_event_expiry(
            date(2026, 8, 21), EXPIRIES, session=EventSession.BEFORE_MARKET
        )
        == date(2026, 8, 21)
    )


def test_select_expiry_skips_same_day_for_after_market():
    """An AMC release lands after the same-day contract has expired (§18)."""
    assert (
        select_event_expiry(
            date(2026, 8, 21), EXPIRIES, session=EventSession.AFTER_MARKET
        )
        == date(2026, 8, 28)
    )


def test_select_expiry_skips_same_day_for_during_market():
    """By the close a DURING_MARKET print is already in the price."""
    assert (
        select_event_expiry(
            date(2026, 8, 21), EXPIRIES, session=EventSession.DURING_MARKET
        )
        == date(2026, 8, 28)
    )


def test_select_expiry_unknown_session_takes_inclusive_branch():
    """UNKNOWN is the inclusive branch; the caller labels the assumption."""
    assert select_event_expiry(date(2026, 8, 21), EXPIRIES) == date(2026, 8, 21)
    assert (
        select_event_expiry(
            date(2026, 8, 21), EXPIRIES, session=EventSession.UNKNOWN
        )
        == date(2026, 8, 21)
    )


def test_select_expiry_accepts_plain_session_string():
    """A stored VARCHAR session works without importing the enum."""
    assert (
        select_event_expiry(date(2026, 8, 21), EXPIRIES, session="after_market")
        == date(2026, 8, 28)
    )


def test_select_expiry_unsorted_and_duplicated_input():
    """Order and duplicates do not change the answer — min of qualifying."""
    messy = [date(2026, 9, 4), date(2026, 8, 28), date(2026, 8, 28)]
    assert select_event_expiry(date(2026, 8, 22), messy) == date(2026, 8, 28)


def test_select_expiry_none_when_all_expiries_are_past():
    """Absence is a value: no qualifying expiry is ``None``, not the last."""
    assert select_event_expiry(date(2026, 12, 1), EXPIRIES) is None
    assert select_event_expiry(date(2026, 8, 21), []) is None


def test_select_expiry_none_when_only_same_day_and_session_excludes_it():
    assert (
        select_event_expiry(
            date(2026, 8, 21),
            [date(2026, 8, 21)],
            session=EventSession.AFTER_MARKET,
        )
        is None
    )


def test_select_expiry_rejects_non_date_entries():
    """Bad input never silently produces a number (contract §1)."""
    with pytest.raises(TypeError):
        select_event_expiry(date(2026, 8, 21), ["2026-08-28"])


# ---------------------------------------------------------------------------
# 2. nearest_strike
# ---------------------------------------------------------------------------


def test_nearest_strike_picks_closest():
    assert nearest_strike(203.0, [195.0, 200.0, 205.0, 210.0]) == 205.0
    assert nearest_strike(201.0, [195.0, 200.0, 205.0, 210.0]) == 200.0


def test_nearest_strike_ties_resolve_to_the_lower_strike():
    """A fixed tiebreak keeps a stored metric reproducible."""
    assert nearest_strike(202.5, [200.0, 205.0]) == 200.0


def test_nearest_strike_skips_unusable_strikes_and_returns_none_when_empty():
    assert nearest_strike(100.0, [float("nan"), 0.0, -5.0, 101.0]) == 101.0
    assert nearest_strike(100.0, []) is None
    assert nearest_strike(100.0, [float("nan"), 0.0]) is None


def test_nearest_strike_none_on_bad_spot():
    assert nearest_strike(0.0, [100.0]) is None
    assert nearest_strike(float("nan"), [100.0]) is None


# ---------------------------------------------------------------------------
# 3. straddle_implied_move (§18/§36) — the missing-leg contract
# ---------------------------------------------------------------------------


def test_straddle_implied_move_hand_checkable():
    """(6 + 4) / 100 = 0.10 — 10 % implied, 10 points."""
    move = straddle_implied_move(6.0, 4.0, 100.0)
    assert move.method == METHOD_STRADDLE
    assert move.points == pytest.approx(10.0)
    assert move.pct == pytest.approx(0.10)
    assert move.spot == 100.0
    assert move.reason is None
    assert move.inputs == {"call_px": 6.0, "put_px": 4.0, "spot": 100.0}


@pytest.mark.parametrize(
    "call_px,put_px,fragment",
    [
        (None, 4.0, "call"),
        (6.0, None, "put"),
        (None, None, "call"),
        (-1.0, 4.0, "call"),
        (float("nan"), 4.0, "call"),
    ],
)
def test_straddle_missing_leg_is_none_never_the_surviving_leg(
    call_px, put_px, fragment
):
    """A one-sided straddle is NOT an implied move — no fabricated price."""
    move = straddle_implied_move(call_px, put_px, 100.0)
    assert move.points is None
    assert move.pct is None
    assert move.reason is not None
    assert fragment in move.reason


@pytest.mark.parametrize("spot", [0.0, -10.0, None, float("nan")])
def test_straddle_bad_spot_is_none_with_reason(spot):
    move = straddle_implied_move(6.0, 4.0, spot)
    assert move.pct is None and move.points is None
    assert "spot_not_positive" in (move.reason or "")


def test_straddle_zero_priced_legs_are_a_real_zero_move():
    """Zero is a legitimate mark (worthless wings), unlike a missing leg."""
    move = straddle_implied_move(0.0, 0.0, 100.0)
    assert move.pct == 0.0
    assert move.reason is None


def test_implied_move_to_dict_is_json_ready():
    payload = straddle_implied_move(6.0, 4.0, 100.0).to_dict()
    assert set(payload) == {
        "points",
        "pct",
        "method",
        "spot",
        "inputs",
        "reason",
    }
    assert payload["method"] == "ATM_STRADDLE"


# ---------------------------------------------------------------------------
# 4. implied_move_from_iv (§36 cross-check)
# ---------------------------------------------------------------------------


def test_implied_move_from_iv_is_iv_times_sqrt_t():
    """365 DTE at 30 % vol is exactly 0.30 — the sqrt(1) case."""
    move = implied_move_from_iv(0.30, 365.0)
    assert move.method == METHOD_IV_SQRT_T
    assert move.pct == pytest.approx(0.30)
    assert move.points is None  # no spot supplied -> no dollar figure


def test_implied_move_from_iv_scales_with_calendar_days_and_spot():
    move = implied_move_from_iv(0.40, 91.25, spot=200.0)
    # 91.25 / 365 = 0.25; sqrt(0.25) = 0.5; 0.40 * 0.5 = 0.20
    assert move.pct == pytest.approx(0.20)
    assert move.points == pytest.approx(40.0)


@pytest.mark.parametrize(
    "iv,dte,fragment",
    [
        (0.0, 30.0, "iv_not_positive"),
        (-0.3, 30.0, "iv_not_positive"),
        (None, 30.0, "iv_not_positive"),
        (0.3, 0.0, "dte_not_positive"),
        (0.3, -1.0, "dte_not_positive"),
        (0.3, None, "dte_not_positive"),
    ],
)
def test_implied_move_from_iv_honest_nulls(iv, dte, fragment):
    move = implied_move_from_iv(iv, dte)
    assert move.pct is None and move.points is None
    assert fragment in (move.reason or "")


def test_straddle_and_iv_methods_are_never_blended():
    """They measure different quantities, so the labels must differ."""
    assert METHOD_STRADDLE != METHOD_IV_SQRT_T
    assert straddle_implied_move(6.0, 4.0, 100.0).method == METHOD_STRADDLE
    assert implied_move_from_iv(0.3, 30.0).method == METHOD_IV_SQRT_T


# ---------------------------------------------------------------------------
# 5. event_iv — round-trip against bs_price, honest nulls
# ---------------------------------------------------------------------------


def test_event_iv_round_trips_a_priced_option():
    """Price at sigma=0.35 with the module's own r, imply it back."""
    price = bs_price(100.0, 100.0, 30.0 / 365.0, 0.35, "C", 0.04, 0.0)
    solved = event_iv(price, 100.0, 100.0, 30.0 / 365.0, "C")
    assert solved == pytest.approx(0.35, abs=1e-6)


def test_event_iv_round_trips_a_put():
    price = bs_price(100.0, 105.0, 45.0 / 365.0, 0.28, "P", 0.04, 0.0)
    solved = event_iv(price, 100.0, 105.0, 45.0 / 365.0, "P")
    assert solved == pytest.approx(0.28, abs=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"price": None},
        {"price": 0.0},
        {"price": float("nan")},
        {"spot": 0.0},
        {"spot": None},
        {"strike": -1.0},
        {"t_years": 0.0},
        {"t_years": None},
        {"right": "X"},
    ],
)
def test_event_iv_never_raises_and_returns_none(kwargs):
    """A backfill loop over vendor rows must not blow up on bad data."""
    args = {
        "price": 5.0,
        "spot": 100.0,
        "strike": 100.0,
        "t_years": 30.0 / 365.0,
        "right": "C",
    }
    args.update(kwargs)
    assert event_iv(**args) is None


def test_event_iv_none_when_price_below_the_model_floor():
    """A mark below discounted intrinsic implies no positive volatility.

    A 50-strike call on a 100 spot is worth at least ``100 - 50*e^{-rT}``
    ~= 50.16; a 45.0 mark is below the model's own floor, so no sigma
    reproduces it and the honest answer is ``None`` — never a clamped
    bracket end.
    """
    assert event_iv(45.0, 100.0, 50.0, 30.0 / 365.0, "C") is None


# ---------------------------------------------------------------------------
# 6. iv_crush (§36/§66)
# ---------------------------------------------------------------------------


def test_iv_crush_is_a_fraction_and_negative_on_a_crush():
    assert iv_crush(0.80, 0.32) == pytest.approx(-0.60)


def test_iv_crush_positive_when_vol_expands():
    assert iv_crush(0.40, 0.50) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "before,after",
    [(None, 0.3), (0.3, None), (0.0, 0.3), (-0.2, 0.3), (0.3, -0.1),
     (float("nan"), 0.3), (0.3, float("nan"))],
)
def test_iv_crush_honest_nulls(before, after):
    assert iv_crush(before, after) is None


# ---------------------------------------------------------------------------
# 7. implied_vs_realized (§66) — bands closed at FAIR
# ---------------------------------------------------------------------------


def test_implied_vs_realized_ratio_and_labels():
    ratio, label = implied_vs_realized(0.05, 0.10)
    assert ratio == pytest.approx(2.0)
    assert label == CLASSIFICATION_UNDER

    ratio, label = implied_vs_realized(0.10, 0.02)
    assert ratio == pytest.approx(0.2)
    assert label == CLASSIFICATION_OVER

    ratio, label = implied_vs_realized(0.10, 0.10)
    assert ratio == pytest.approx(1.0)
    assert label == CLASSIFICATION_FAIR


def test_implied_vs_realized_drops_the_sign_of_the_realized_move():
    """A straddle is direction-agnostic — down 10 % is the same |move|."""
    assert implied_vs_realized(0.05, -0.10) == implied_vs_realized(0.05, 0.10)


def test_implied_vs_realized_boundaries_are_fair():
    """Exactly 0.8 and exactly 1.2 are FAIR; UNDER/OVER are strictly beyond."""
    assert implied_vs_realized(0.10, 0.08) == (pytest.approx(0.8), CLASSIFICATION_FAIR)
    assert implied_vs_realized(0.10, 0.12) == (pytest.approx(1.2), CLASSIFICATION_FAIR)
    _, just_over = implied_vs_realized(0.10, 0.1201)
    _, just_under = implied_vs_realized(0.10, 0.0799)
    assert just_over == CLASSIFICATION_UNDER
    assert just_under == CLASSIFICATION_OVER


@pytest.mark.parametrize(
    "implied,actual",
    [(None, 0.1), (0.1, None), (0.0, 0.1), (-0.1, 0.1),
     (float("nan"), 0.1), (0.1, float("nan"))],
)
def test_implied_vs_realized_honest_nulls(implied, actual):
    assert implied_vs_realized(implied, actual) == (None, None)


# ---------------------------------------------------------------------------
# 8. historical_move_stats (§19/§64) — nearest rank, n counts the usable
# ---------------------------------------------------------------------------


def test_historical_move_stats_hand_checkable():
    """Five moves, ascending |.|: 0.01 0.02 0.03 0.04 0.10.

    Nearest rank: median rank ceil(0.5*5)=3 -> 0.03; p90 rank
    ceil(0.9*5)=5 -> 0.10; max 0.10.
    """
    stats = historical_move_stats([0.02, -0.03, 0.10, -0.01, 0.04])
    assert stats["n"] == 5
    assert stats["median_abs"] == pytest.approx(0.03)
    assert stats["p90_abs"] == pytest.approx(0.10)
    assert stats["max_abs"] == pytest.approx(0.10)


def test_historical_move_stats_matches_reaction_percentile_definition():
    """Same nearest-rank definition as the price tab (no drift)."""
    moves = [0.011, -0.024, 0.033, 0.047, 0.052, -0.061, 0.077, 0.099]
    ordered = sorted(abs(m) for m in moves)
    stats = historical_move_stats(moves)
    assert stats["median_abs"] == percentile_nearest_rank(ordered, 50.0)
    assert stats["p90_abs"] == percentile_nearest_rank(ordered, 90.0)


def test_historical_move_stats_keys_always_present_when_empty():
    for empty in ([], None, [None, float("nan"), float("inf")]):
        stats = historical_move_stats(empty)
        assert stats == {
            "median_abs": None,
            "p90_abs": None,
            "max_abs": None,
            "n": 0,
        }


def test_historical_move_stats_n_counts_usable_not_length():
    stats = historical_move_stats([0.05, None, float("nan"), -0.03])
    assert stats["n"] == 2
    assert stats["max_abs"] == pytest.approx(0.05)


def test_historical_move_stats_single_sample_reports_that_sample():
    stats = historical_move_stats([-0.07])
    assert stats == {
        "median_abs": pytest.approx(0.07),
        "p90_abs": pytest.approx(0.07),
        "max_abs": pytest.approx(0.07),
        "n": 1,
    }


# ---------------------------------------------------------------------------
# 9. build_summary — statuses, notes, and the crush
# ---------------------------------------------------------------------------


def _ok_summary(**overrides):
    """A complete pre/post picture; overrides poke one input at a time."""
    kwargs = dict(
        pre_call_close=6.0,
        pre_put_close=4.0,
        spot=100.0,
        post_call_close=3.4,
        post_put_close=0.4,
        post_spot=103.0,
        strike=100.0,
        expiry=date(2026, 8, 28),
        event_date=date(2026, 8, 21),
        session=EventSession.AFTER_MARKET,
        actual_move_pct=0.03,
        dte_days=7.0,
    )
    kwargs.update(overrides)
    return build_summary(
        kwargs.pop("pre_call_close"),
        kwargs.pop("pre_put_close"),
        kwargs.pop("spot"),
        kwargs.pop("post_call_close"),
        kwargs.pop("post_put_close"),
        kwargs.pop("post_spot"),
        **kwargs,
    )


def test_build_summary_ok_path():
    summary = _ok_summary()
    assert isinstance(summary, ImpliedMoveSummary)
    assert summary.status == STATUS_OK
    assert summary.basis == BASIS_HISTORICAL
    assert summary.pre is not None and summary.pre.pct == pytest.approx(0.10)
    assert summary.post is not None
    assert summary.post.pct == pytest.approx(3.8 / 103.0)
    # 0.03 realized vs 0.10 implied -> ratio 0.30 -> OVER_PRICED
    assert summary.ratio == pytest.approx(0.30)
    assert summary.classification == CLASSIFICATION_OVER
    assert summary.actual_move_pct == pytest.approx(0.03)
    assert summary.strike == 100.0
    assert summary.expiry == date(2026, 8, 28)
    assert summary.event_date == date(2026, 8, 21)
    assert summary.session == "AFTER_MARKET"
    assert summary.dte_days == pytest.approx(7.0)
    assert summary.disclaimer == DISCLAIMER


def test_build_summary_solves_both_ivs_and_a_negative_crush():
    """Rich pre-event call, cheap post-event call -> vol collapsed."""
    summary = _ok_summary()
    assert summary.iv_before is not None
    assert summary.iv_after is not None
    assert summary.iv_before > summary.iv_after
    assert summary.iv_crush_pct is not None
    assert summary.iv_crush_pct < 0.0
    assert summary.iv_crush_pct == pytest.approx(
        summary.iv_after / summary.iv_before - 1.0
    )


def test_build_summary_partial_when_post_leg_missing():
    """The pre-event implied move is still real — only the crush is gone."""
    summary = _ok_summary(post_put_close=None)
    assert summary.status == STATUS_PARTIAL
    assert summary.pre is not None and summary.pre.pct == pytest.approx(0.10)
    assert summary.post is not None and summary.post.pct is None
    assert "post" in summary.notes
    assert "put" in summary.notes["post"]


def test_build_summary_no_data_when_pre_leg_missing():
    summary = _ok_summary(pre_call_close=None)
    assert summary.status == STATUS_NO_DATA
    assert summary.pre is not None and summary.pre.pct is None
    assert summary.ratio is None and summary.classification is None
    assert "pre" in summary.notes
    assert "ratio" in summary.notes


def test_build_summary_no_data_when_spot_missing():
    summary = _ok_summary(spot=None)
    assert summary.status == STATUS_NO_DATA
    assert "spot_not_positive" in summary.notes["pre"]


def test_build_summary_without_dte_has_no_iv_pair_and_says_so():
    summary = _ok_summary(dte_days=None)
    assert summary.iv_before is None
    assert summary.iv_after is None
    assert summary.iv_crush_pct is None
    assert "dte_missing_or_not_positive" in summary.notes["iv"]
    assert "iv_crush" in summary.notes
    # The straddle move itself does NOT depend on the IV solve.
    assert summary.pre is not None and summary.pre.pct == pytest.approx(0.10)
    assert summary.status == STATUS_OK


def test_build_summary_one_dte_leaves_no_post_event_vol():
    summary = _ok_summary(dte_days=1.0)
    assert summary.iv_after is None
    assert "expired_after_event" in summary.notes["iv_after"]
    assert summary.iv_crush_pct is None


def test_build_summary_notes_missing_actual_move():
    summary = _ok_summary(actual_move_pct=None)
    assert summary.actual_move_pct is None
    assert summary.ratio is None and summary.classification is None
    assert summary.notes["actual_move_pct"] == "not_supplied"


def test_build_summary_never_leaks_nan_or_inf():
    summary = _ok_summary(
        actual_move_pct=float("nan"), post_spot=float("inf")
    )
    for value in (
        summary.pre.pct,
        summary.post.pct,
        summary.iv_before,
        summary.iv_after,
        summary.iv_crush_pct,
        summary.actual_move_pct,
        summary.ratio,
    ):
        assert value is None or math.isfinite(value)
    assert summary.actual_move_pct is None
    assert "not_finite" in summary.notes["actual_move_pct"]


def test_build_summary_rejected_strike_is_noted_not_stored():
    summary = _ok_summary(strike=0.0)
    assert summary.strike is None
    assert "strike_not_positive" in summary.notes["strike"]


def test_build_summary_is_deterministic():
    assert _ok_summary().to_dict() == _ok_summary().to_dict()


def test_summary_to_dict_shape_and_iso_dates():
    payload = _ok_summary().to_dict()
    assert set(payload) == {
        "basis",
        "status",
        "pre",
        "post",
        "iv_before",
        "iv_after",
        "iv_crush_pct",
        "actual_move_pct",
        "ratio",
        "classification",
        "strike",
        "expiry",
        "event_date",
        "session",
        "dte_days",
        "disclaimer",
        "model_version",
        "notes",
    }
    assert payload["expiry"] == "2026-08-28"
    assert payload["event_date"] == "2026-08-21"
    assert payload["basis"] == "HISTORICAL_DAILY_CLOSE_APPROXIMATION"
    assert payload["pre"]["method"] == "ATM_STRADDLE"
    assert payload["disclaimer"] == DISCLAIMER


def test_summary_to_dict_carries_none_dates_without_crashing():
    payload = build_summary(
        None, None, None, None, None, None,
        strike=None, expiry=None, event_date=None,
    ).to_dict()
    assert payload["expiry"] is None
    assert payload["event_date"] is None
    assert payload["session"] is None
    assert payload["status"] == STATUS_NO_DATA


# ---------------------------------------------------------------------------
# 10. Fixed strings (§37) and the audit §7.4 no-I/O guard
# ---------------------------------------------------------------------------


def test_disclaimer_wording_is_the_spec_wording():
    """§37: the payload and the UI both render this verbatim."""
    assert DISCLAIMER == (
        "Implied move is option-market pricing, not a forecast of the move."
    )
    assert "not a forecast" in DISCLAIMER


def test_basis_labels_are_the_two_audit_strings():
    assert BASIS_LIVE == "LIVE_CHAIN_SNAPSHOT"
    assert BASIS_HISTORICAL == "HISTORICAL_DAILY_CLOSE_APPROXIMATION"
    assert BASIS_LIVE != BASIS_HISTORICAL


def test_classification_labels_are_the_contract_strings():
    assert CLASSIFICATION_UNDER == "UNDER_PRICED"
    assert CLASSIFICATION_FAIR == "FAIR"
    assert CLASSIFICATION_OVER == "OVER_PRICED"


def test_status_labels_are_the_contract_strings():
    assert (STATUS_OK, STATUS_PARTIAL, STATUS_NO_DATA) == (
        "OK",
        "PARTIAL",
        "NO_DATA",
    )


def test_module_imports_no_provider_or_gateway(**_):
    """Audit §7.4: the pure layer cannot reach a live API."""
    tree = ast.parse(MODULE_PATH.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        assert not name.startswith("apps"), name
        assert not name.startswith("libs.market_data"), name
        assert not name.startswith("libs.event_calendar"), name
        assert not name.startswith("sqlalchemy"), name


def test_package_reexports_the_public_names():
    """The seam imports from ``libs.trading_core.events`` directly."""
    import libs.trading_core.events as events_pkg

    for name in (
        "BASIS_HISTORICAL",
        "BASIS_LIVE",
        "DISCLAIMER",
        "ImpliedMove",
        "ImpliedMoveSummary",
        "build_summary",
        "event_iv",
        "historical_move_stats",
        "implied_move_from_iv",
        "implied_vs_realized",
        "iv_crush",
        "nearest_strike",
        "select_event_expiry",
        "straddle_implied_move",
    ):
        assert hasattr(events_pkg, name), name
        assert name in events_pkg.__all__, name


def test_implied_move_is_frozen():
    move = ImpliedMove(points=1.0, pct=0.01, method=METHOD_STRADDLE)
    with pytest.raises(Exception):
        move.pct = 0.02  # type: ignore[misc]
