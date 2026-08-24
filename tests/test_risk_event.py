"""Event-risk library tests (event spec §62–§67; Phase K contract U1).

The module under test is a TABLE, not a model, and these tests are written
to hold it to that. Four properties are re-proved rather than assumed,
because each one is a user mandate rather than an implementation detail:

1.  **Determinism, no LLM** (§63) — the classifier is called twice on the
    same inputs and the payloads must be byte-equal, and the source file
    must contain no model/network call at all.
2.  **Sample size everywhere** (§64) — every payload carrying a historical
    statistic also carries ``n``, in the same mapping, and a small sample
    says so in its caveats.
3.  **UNKNOWN is not LOW** — with no implied move and no history the state
    is UNKNOWN with a reason, never a guessed LOW.
4.  **SHADOW** (§65) — the caps are ``pretrade.QuantityCap`` rows and
    nothing in this module reaches ``assess``.

Every threshold arithmetic case below is hand-checkable: the fixtures use
round numbers (NAV 200 000, price 50, moves in whole percents) so a reader
can verify a state without running the code.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

from libs.trading_core.risk import event_risk as er
from libs.trading_core.risk.event_risk import (
    BASIS_HISTORICAL_MEDIAN,
    BASIS_IMPLIED,
    BASIS_NONE,
    CODE_EVENT_EXPOSURE,
    EVENT_RISK_MODEL_VERSION,
    SENSITIVITY_HIGH,
    SENSITIVITY_LOW,
    SENSITIVITY_MODERATE,
    STATE_EXTREME,
    STATE_HIGH,
    STATE_LADDER,
    STATE_LOW,
    STATE_MODERATE,
    STATE_UNKNOWN,
    EventRiskInputs,
    EventRiskPolicy,
    EventRiskThresholds,
    classify_event_risk,
    event_risk_caps,
    historical_event_risk,
)
from libs.trading_core.risk.pretrade import (
    CAP_LAYERS,
    LAYER_CONCENTRATION,
    MODE_SHADOW,
    QuantityCap,
    shadow_verdict,
)

NAV = 200_000.0


# ---------------------------------------------------------------------------
# §64 — historical_event_risk
# ---------------------------------------------------------------------------


def test_historical_stats_nearest_rank_hand_check() -> None:
    """Absolutes of ``[-8, 4, -6, 12, 2]`` sorted are ``[2, 4, 6, 8, 12]``.

    n=5 ⇒ median rank ceil(2.5)=3 → 6; p75 rank ceil(3.75)=4 → 8;
    p90 rank ceil(4.5)=5 → 12; max 12. Nearest-rank means every reported
    percentile is a move that actually happened.
    """
    out = historical_event_risk([-8.0, 4.0, -6.0, 12.0, 2.0])
    assert out == {
        "median_abs": 6.0,
        "p75_abs": 8.0,
        "p90_abs": 12.0,
        "max_abs": 12.0,
        "n": 5,
    }


def test_historical_stats_matches_events_percentile_convention() -> None:
    """Pinned against ``events.implied_move.historical_move_stats`` — the
    two must agree on median/p90/max/n, or two screens would print two
    different "medians" for the same eight prints."""
    from libs.trading_core.events.implied_move import historical_move_stats

    moves = [-7.1, 3.2, -9.8, 12.4, 0.5, -2.2, 8.0, 4.4]
    mine = historical_event_risk(moves)
    theirs = historical_move_stats(moves)
    assert mine["n"] == theirs["n"] == 8
    assert mine["median_abs"] == theirs["median_abs"]
    assert mine["p90_abs"] == theirs["p90_abs"]
    assert mine["max_abs"] == theirs["max_abs"]


def test_historical_stats_empty_is_none_not_zero() -> None:
    """No sample ⇒ every statistic ``None`` and ``n=0``. A 0.0 median would
    read as "this name never moves", which is the opposite of the truth."""
    out = historical_event_risk([])
    assert out == {
        "median_abs": None,
        "p75_abs": None,
        "p90_abs": None,
        "max_abs": None,
        "n": 0,
    }
    assert historical_event_risk(None) == out


def test_historical_stats_n_counts_usable_not_length() -> None:
    """Holes are counted OUT of n, never imputed: 3 usable prints out of a
    6-entry list report n=3 and statistics over the three."""
    out = historical_event_risk([None, 4.0, float("nan"), -6.0, 2.0, float("inf")])
    assert out["n"] == 3
    assert out["median_abs"] == 4.0  # sorted |.| = [2, 4, 6], rank ceil(1.5)=2
    assert out["max_abs"] == 6.0


def test_historical_stats_rejects_bools() -> None:
    """``True`` is not a 1% move — a coerced flag would fabricate a print."""
    assert historical_event_risk([True, False])["n"] == 0


def test_single_observation_is_reported_with_n_1() -> None:
    """n=1 reports that one print for all four statistics. Honest *because*
    n=1 travels with it — §64's "never imply statistical certainty"."""
    out = historical_event_risk([-9.0])
    assert out == {
        "median_abs": 9.0,
        "p75_abs": 9.0,
        "p90_abs": 9.0,
        "max_abs": 9.0,
        "n": 1,
    }


# ---------------------------------------------------------------------------
# §63 — the classification table: every state reachable
# ---------------------------------------------------------------------------


def _inputs(**kw) -> EventRiskInputs:
    base = dict(
        event_type="EARNINGS",
        time_to_event_days=10.0,
        historical_moves=(),
        implied_move_pct=None,
        position_exposure_usd=None,
        portfolio_nav_usd=NAV,
    )
    base.update(kw)
    return EventRiskInputs(**base)


def test_state_low_small_move_far_away() -> None:
    """2% expected move, 10 days out, no exposure bump ⇒ level 0 = LOW."""
    snap = classify_event_risk(_inputs(implied_move_pct=2.0))
    assert snap["event_risk_state"] == STATE_LOW
    assert snap["expected_move_basis"] == BASIS_IMPLIED
    assert snap["expected_move_pct"] == 2.0


def test_state_moderate_from_move_threshold() -> None:
    """4% is exactly the MODERATE threshold — inclusive (``>=``) ⇒ level 1."""
    snap = classify_event_risk(_inputs(implied_move_pct=4.0))
    assert snap["event_risk_state"] == STATE_MODERATE


def test_state_high_from_move_threshold() -> None:
    """8% is exactly the HIGH threshold ⇒ level 2, far-dated so no bump."""
    snap = classify_event_risk(_inputs(implied_move_pct=8.0))
    assert snap["event_risk_state"] == STATE_HIGH


def test_state_extreme_from_move_threshold() -> None:
    """12% is exactly the EXTREME threshold ⇒ level 3, the ladder top."""
    snap = classify_event_risk(_inputs(implied_move_pct=12.0))
    assert snap["event_risk_state"] == STATE_EXTREME


def test_imminence_bumps_exactly_one_level() -> None:
    """Same 8% move, 1.3 days out instead of 10 ⇒ HIGH becomes EXTREME.

    This is the §65 panel case: "Earnings in 1.3 days / implied 8.8%".
    """
    far = classify_event_risk(_inputs(implied_move_pct=8.0, time_to_event_days=10.0))
    near = classify_event_risk(_inputs(implied_move_pct=8.0, time_to_event_days=1.3))
    assert far["event_risk_state"] == STATE_HIGH
    assert near["event_risk_state"] == STATE_EXTREME


def test_imminence_boundary_is_inclusive_and_past_events_do_not_bump() -> None:
    """3.0 days bumps (``<=``); 3.1 does not; a PAST event never bumps (§67
    — this module measures pre-event risk)."""
    at = classify_event_risk(_inputs(implied_move_pct=4.0, time_to_event_days=3.0))
    just_after = classify_event_risk(
        _inputs(implied_move_pct=4.0, time_to_event_days=3.1)
    )
    passed = classify_event_risk(_inputs(implied_move_pct=4.0, time_to_event_days=-1.0))
    assert at["event_risk_state"] == STATE_HIGH
    assert just_after["event_risk_state"] == STATE_MODERATE
    assert passed["event_risk_state"] == STATE_MODERATE
    assert any("day(s) ago" in d for d in passed["drivers"])


def test_exposure_share_bumps_one_and_two_levels() -> None:
    """A 4% move (base MODERATE), 10 days out, at three exposure sizes.

    NAV 200 000. 10 000 = 5% ⇒ no bump ⇒ MODERATE. 20 000 = 10% ⇒ +1 ⇒
    HIGH. 50 000 = 25% ⇒ +2 ⇒ EXTREME. Same gap, three different position
    sizes, three different answers — which is the whole point of the axis.
    """
    small = classify_event_risk(
        _inputs(implied_move_pct=4.0, position_exposure_usd=10_000.0)
    )
    mid = classify_event_risk(
        _inputs(implied_move_pct=4.0, position_exposure_usd=20_000.0)
    )
    big = classify_event_risk(
        _inputs(implied_move_pct=4.0, position_exposure_usd=50_000.0)
    )
    assert small["exposure_share"] == pytest.approx(5.0)
    assert small["event_risk_state"] == STATE_MODERATE
    assert mid["exposure_share"] == pytest.approx(10.0)
    assert mid["event_risk_state"] == STATE_HIGH
    assert big["exposure_share"] == pytest.approx(25.0)
    assert big["event_risk_state"] == STATE_EXTREME


def test_bumps_saturate_at_extreme() -> None:
    """12% + imminent + 25% of NAV = level 3+1+2 = 6, clamped to EXTREME.

    Saturation matters: an index off the end of the ladder would be a
    crash on exactly the riskiest row in the system.
    """
    snap = classify_event_risk(
        _inputs(
            implied_move_pct=12.0,
            time_to_event_days=0.5,
            position_exposure_usd=60_000.0,
        )
    )
    assert snap["event_risk_state"] == STATE_EXTREME


def test_every_state_is_reachable() -> None:
    """The whole §63 vocabulary, plus UNKNOWN, is producible — a table with
    an unreachable rung is a table with a bug."""
    reached = {
        classify_event_risk(_inputs(implied_move_pct=m, time_to_event_days=t))[
            "event_risk_state"
        ]
        for m, t in ((1.0, 30.0), (4.0, 30.0), (8.0, 30.0), (12.0, 30.0))
    }
    assert reached == set(STATE_LADDER)
    assert (
        classify_event_risk(_inputs())["event_risk_state"] == STATE_UNKNOWN
    )


def test_table_is_monotone_in_expected_move() -> None:
    """Severity never DECREASES as the expected move grows — the property
    that makes the state explainable and an eventual backtest falsifiable."""
    ladder = list(STATE_LADDER)
    seen = -1
    for move in [x / 2.0 for x in range(0, 40)]:
        state = classify_event_risk(_inputs(implied_move_pct=move))["event_risk_state"]
        idx = ladder.index(state)
        assert idx >= seen, f"severity dropped at move={move}"
        seen = idx


# ---------------------------------------------------------------------------
# Basis selection, UNKNOWN honesty
# ---------------------------------------------------------------------------


def test_implied_move_wins_over_history_and_records_basis() -> None:
    """With both available the forward-looking implied price is used, and
    ``expected_move_basis`` says so — a reader never has to guess which
    number drove the state."""
    snap = classify_event_risk(
        _inputs(implied_move_pct=8.8, historical_moves=(-7.1, 5.0, -3.0))
    )
    assert snap["expected_move_pct"] == pytest.approx(8.8)
    assert snap["expected_move_basis"] == BASIS_IMPLIED
    assert snap["implied"]["pct"] == pytest.approx(8.8)
    assert snap["historical"]["n"] == 3  # history still reported alongside


def test_historical_median_used_when_no_implied_move() -> None:
    """No straddle ⇒ fall back to the median with basis HISTORICAL_MEDIAN,
    and the driver sentence names the sample size inline."""
    snap = classify_event_risk(
        _inputs(historical_moves=(-8.0, 4.0, -6.0, 12.0, 2.0))
    )
    assert snap["expected_move_pct"] == pytest.approx(6.0)
    assert snap["expected_move_basis"] == BASIS_HISTORICAL_MEDIAN
    assert snap["event_risk_state"] == STATE_MODERATE  # 6% ≥ 4, < 8, far-dated
    assert any("based on 5 event(s)" in d for d in snap["drivers"])


def test_unknown_when_nothing_measured_and_it_is_not_low() -> None:
    """No implied move AND n=0 ⇒ UNKNOWN with a reason naming both gaps.

    The single most important assertion in this file: an unmeasured event
    must never render as LOW.
    """
    snap = classify_event_risk(_inputs())
    assert snap["event_risk_state"] == STATE_UNKNOWN
    assert snap["event_risk_state"] != STATE_LOW
    assert snap["expected_move_pct"] is None
    assert snap["expected_move_basis"] == BASIS_NONE
    assert snap["reason"] is not None
    assert "UNKNOWN is not LOW" in snap["reason"]
    assert snap["historical"]["n"] == 0


def test_unknown_state_still_carries_n_and_greeks() -> None:
    """UNKNOWN is a full snapshot, not a stub: exposure share, greeks and
    ``n`` are all present so the UI renders one layout, not two."""
    snap = classify_event_risk(
        _inputs(position_exposure_usd=50_000.0, option_vega=300.0)
    )
    assert snap["event_risk_state"] == STATE_UNKNOWN
    assert snap["historical"]["n"] == 0
    assert snap["exposure_share"] == pytest.approx(25.0)
    assert snap["option_greeks"]["vega"] == 300.0
    assert snap["sensitivity"] == SENSITIVITY_HIGH


# ---------------------------------------------------------------------------
# §64 — n carried everywhere; caveats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"implied_move_pct": 9.0},
        {"historical_moves": (-7.0,)},
        {"historical_moves": (-7.0, 3.0, 8.0), "implied_move_pct": 5.0},
        {"historical_moves": tuple(range(1, 13)), "position_exposure_usd": 90_000.0},
    ],
)
def test_n_is_present_in_every_payload(kwargs) -> None:
    """§64: ``historical`` ALWAYS carries ``n``, in the same mapping as the
    statistics, so a median can never reach a screen without it."""
    snap = classify_event_risk(_inputs(**kwargs))
    assert "n" in snap["historical"]
    assert isinstance(snap["historical"]["n"], int)
    for key in ("median_abs", "p75_abs", "p90_abs", "max_abs"):
        assert key in snap["historical"]


def test_small_sample_caveat_and_no_certainty_language() -> None:
    """8 prints get a "based on 8 event(s)" caveat; 3 prints get that PLUS
    an explicit small-sample warning (§64: never imply certainty)."""
    eight = classify_event_risk(_inputs(historical_moves=tuple(range(1, 9))))
    three = classify_event_risk(_inputs(historical_moves=(-7.0, 3.0, 8.0)))
    assert any("based on 8 event(s)" in c for c in eight["caveats"])
    assert not any("small sample" in c for c in eight["caveats"])
    assert any("based on 3 event(s)" in c for c in three["caveats"])
    assert any("small sample" in c for c in three["caveats"])


def test_zero_sample_caveat() -> None:
    """n=0 says so rather than silently omitting the historical block."""
    snap = classify_event_risk(_inputs(implied_move_pct=5.0))
    assert any("n=0" in c for c in snap["caveats"])


def test_estimated_date_adds_caveat_but_never_changes_state() -> None:
    """An ESTIMATED date is not less risky — it is less PRECISE. The state
    is byte-identical; only the caveat list differs."""
    confirmed = classify_event_risk(_inputs(implied_move_pct=9.0, is_estimated=False))
    estimated = classify_event_risk(_inputs(implied_move_pct=9.0, is_estimated=True))
    assert confirmed["event_risk_state"] == estimated["event_risk_state"]
    assert not any("ESTIMATED" in c for c in confirmed["caveats"])
    assert any("ESTIMATED" in c for c in estimated["caveats"])


def test_drivers_carry_real_numbers() -> None:
    """Every driver justifying a state prints the number behind it, so the
    UI can answer "why EXTREME?" without re-deriving anything."""
    snap = classify_event_risk(
        _inputs(
            implied_move_pct=8.8,
            time_to_event_days=1.3,
            position_exposure_usd=12_000.0,
        )
    )
    assert snap["event_risk_state"] == STATE_EXTREME  # 8.8 → HIGH, +1 imminent
    joined = " | ".join(snap["drivers"])
    assert "8.80%" in joined
    assert "1.3 day(s)" in joined


# ---------------------------------------------------------------------------
# None-safety
# ---------------------------------------------------------------------------


def test_exposure_share_none_when_nav_missing_never_zero() -> None:
    """A share divided by nothing is not 0%. Missing NAV ⇒ ``None`` plus a
    caveat, and NO exposure bump — the alternative silently makes every
    unfunded snapshot look small."""
    snap = classify_event_risk(
        EventRiskInputs(
            event_type="EARNINGS",
            implied_move_pct=4.0,
            position_exposure_usd=50_000.0,
            portfolio_nav_usd=None,
        )
    )
    assert snap["exposure_share"] is None
    assert snap["event_risk_state"] == STATE_MODERATE  # no bump from a null
    assert any("NAV" in c for c in snap["caveats"])


def test_exposure_share_none_when_nav_zero() -> None:
    """NAV 0 is a divide-by-zero, not a 100%-of-NAV position."""
    snap = classify_event_risk(
        EventRiskInputs(implied_move_pct=4.0, position_exposure_usd=1.0, portfolio_nav_usd=0.0)
    )
    assert snap["exposure_share"] is None


def test_negative_exposure_uses_magnitude() -> None:
    """A SHORT position of −50 000 on 200 000 NAV is 25% of NAV at risk in
    the gap, not −25%. The share is a magnitude."""
    snap = classify_event_risk(_inputs(implied_move_pct=4.0, position_exposure_usd=-50_000.0))
    assert snap["exposure_share"] == pytest.approx(25.0)
    assert snap["event_risk_state"] == STATE_EXTREME


def test_nan_and_none_inputs_do_not_crash_or_fabricate() -> None:
    """NaN inputs are treated as absent, not as 0.0."""
    snap = classify_event_risk(
        EventRiskInputs(
            time_to_event_days=float("nan"),
            implied_move_pct=float("nan"),
            position_exposure_usd=float("inf"),
            portfolio_nav_usd=NAV,
            option_gamma=float("nan"),
        )
    )
    assert snap["event_risk_state"] == STATE_UNKNOWN
    assert snap["time_to_event_days"] is None
    assert snap["exposure_share"] is None
    assert snap["option_greeks"] is None


def test_missing_time_to_event_adds_caveat_and_no_bump() -> None:
    snap = classify_event_risk(
        EventRiskInputs(implied_move_pct=8.0, time_to_event_days=None, portfolio_nav_usd=NAV)
    )
    assert snap["event_risk_state"] == STATE_HIGH
    assert any("time to event unknown" in c for c in snap["caveats"])


def test_negative_nav_rejected() -> None:
    with pytest.raises(ValueError, match="portfolio_nav_usd"):
        EventRiskInputs(portfolio_nav_usd=-1.0)


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        EventRiskThresholds(moderate_move_pct=10.0, high_move_pct=5.0)
    with pytest.raises(ValueError, match="exposure_one_level_pct"):
        EventRiskThresholds(exposure_one_level_pct=30.0, exposure_two_level_pct=20.0)


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="warn_from"):
        EventRiskPolicy(warn_from="NOPE")
    with pytest.raises(ValueError, match="EXTREME is stricter"):
        EventRiskPolicy(high_max_exposure_pct=5.0, extreme_max_exposure_pct=10.0)


def test_thresholds_are_parameters_not_constants() -> None:
    """The table is RESEARCH DEFAULTS — a caller can move every rung, which
    is what makes the eventual §65 backtest possible at all."""
    strict = EventRiskThresholds(moderate_move_pct=1.0, high_move_pct=2.0, extreme_move_pct=3.0)
    snap = classify_event_risk(_inputs(implied_move_pct=3.0), thresholds=strict)
    assert snap["event_risk_state"] == STATE_EXTREME
    assert classify_event_risk(_inputs(implied_move_pct=3.0))["event_risk_state"] == STATE_LOW


# ---------------------------------------------------------------------------
# §66 — sensitivity is its own axis
# ---------------------------------------------------------------------------


def test_sensitivity_low_without_options() -> None:
    """A stock holder has no option sensitivity. That is a MEASUREMENT (they
    eat the gap linearly), not a missing value — and ``option_greeks`` is
    ``None`` rather than a dict of zeros."""
    snap = classify_event_risk(_inputs(implied_move_pct=9.0))
    assert snap["sensitivity"] == SENSITIVITY_LOW
    assert snap["option_greeks"] is None


def test_sensitivity_rises_with_vega_and_gamma() -> None:
    mod = classify_event_risk(_inputs(implied_move_pct=2.0, option_vega=60.0))
    high = classify_event_risk(_inputs(implied_move_pct=2.0, option_gamma=25.0))
    assert mod["sensitivity"] == SENSITIVITY_MODERATE
    assert high["sensitivity"] == SENSITIVITY_HIGH


def test_sensitivity_uses_magnitude_short_vega_is_exposed_too() -> None:
    """A short-vega position is at least as exposed to an IV crush as a
    long one — the label is about magnitude, the sign lives in the greek."""
    short = classify_event_risk(_inputs(implied_move_pct=2.0, option_vega=-300.0))
    assert short["sensitivity"] == SENSITIVITY_HIGH
    assert short["option_greeks"]["vega"] == -300.0


def test_greeks_never_change_the_state() -> None:
    """§66's axis is separate from §63's. Identical inputs but for the
    greeks ⇒ identical ``event_risk_state``, different ``sensitivity``."""
    plain = classify_event_risk(_inputs(implied_move_pct=9.0))
    with_greeks = classify_event_risk(
        _inputs(implied_move_pct=9.0, option_gamma=50.0, option_vega=500.0, option_theta=-40.0)
    )
    assert plain["event_risk_state"] == with_greeks["event_risk_state"]
    assert plain["sensitivity"] == SENSITIVITY_LOW
    assert with_greeks["sensitivity"] == SENSITIVITY_HIGH


def test_theta_alone_reports_greeks_without_raising_sensitivity() -> None:
    """Theta is displayed (§66) but does not drive the label — time decay is
    not gap amplification."""
    snap = classify_event_risk(_inputs(implied_move_pct=2.0, option_theta=-99.0))
    assert snap["option_greeks"] == {"gamma": None, "vega": None, "theta": -99.0}
    assert snap["sensitivity"] == SENSITIVITY_LOW


# ---------------------------------------------------------------------------
# Payload shape / determinism (§63 — no LLM)
# ---------------------------------------------------------------------------


def test_payload_shape_is_stable_and_versioned() -> None:
    snap = classify_event_risk(_inputs(implied_move_pct=8.8, option_vega=100.0))
    assert set(snap) == {
        "event_type",
        "time_to_event_days",
        "historical",
        "implied",
        "expected_move_pct",
        "expected_move_basis",
        "position_exposure_usd",
        "exposure_share",
        "option_greeks",
        "event_risk_state",
        "sensitivity",
        "drivers",
        "caveats",
        "reason",
        "model_version",
    }
    assert snap["model_version"] == EVENT_RISK_MODEL_VERSION == "event-risk-1.0.0"
    assert isinstance(snap["drivers"], list)
    assert isinstance(snap["caveats"], list)


def test_classifier_is_deterministic() -> None:
    """§63: same inputs, same state, forever. Two calls, byte-equal."""
    ins = _inputs(
        implied_move_pct=8.8,
        time_to_event_days=1.3,
        historical_moves=(-7.1, 5.0, -3.0, 9.9),
        position_exposure_usd=30_000.0,
        option_vega=120.0,
    )
    assert classify_event_risk(ins) == classify_event_risk(ins)


def test_no_llm_or_network_in_the_state_assignment() -> None:
    """§63 "Do not let LLM alone assign this state" — enforced statically.

    The check runs over the module's EXECUTABLE tokens — docstrings,
    comments and string literals are stripped first, because the prose in
    this file legitimately discusses prompts and LLMs while the code must
    contain neither. A token scan is a blunt instrument, which is why it is
    paired with the determinism test above: together they say "the state
    comes from a table, and the table is in this file".
    """
    import io
    import tokenize

    source = Path(inspect.getfile(er)).read_text(encoding="utf-8")
    code_tokens = [
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE)
    ]
    code = " ".join(code_tokens).lower()
    for forbidden in (
        "anthropic",
        "openai",
        "httpx",
        "requests",
        "aiohttp",
        "urllib",
        "prompt",
        "completion",
        "llm",
        "random",
        "socket",
        "asyncio",
    ):
        assert forbidden not in code, f"{forbidden!r} must not appear in event_risk.py code"
    # …and no import of the LLM/provider or application layers.
    assert "libs.llm" not in code
    assert "apps." not in code


def test_module_does_not_import_the_engine() -> None:
    """SHADOW by construction: the arrow never points at ``assess``."""
    source = Path(inspect.getfile(er)).read_text(encoding="utf-8")
    assert "from .engine" not in source
    assert "import engine" not in source
    assert "assess(" not in source.replace("``assess(extra_caps=...)``", "")


# ---------------------------------------------------------------------------
# §65 — event_risk_caps: SHADOW, QuantityCap-shaped
# ---------------------------------------------------------------------------


def _extreme_snapshot() -> dict:
    return classify_event_risk(
        _inputs(implied_move_pct=14.0, historical_moves=(-8.0, 4.0, -6.0, 12.0, 2.0))
    )


def _high_snapshot() -> dict:
    return classify_event_risk(_inputs(implied_move_pct=9.0))


def test_cap_hand_check_extreme() -> None:
    """EXTREME, nav 50 000, price 50, requested 100.

    Budget ``5% · 50 000 = 2 500``; ``cap_qty = floor(2500/50) = 50``;
    50 < 100 so ONE cap is emitted at 50.
    """
    caps = event_risk_caps(
        _extreme_snapshot(), requested_qty=100, price=50.0, nav=50_000.0
    )
    assert len(caps) == 1
    cap = caps[0]
    assert cap.cap_qty == 50
    assert cap.measured["budget_usd"] == pytest.approx(2_500.0)
    assert cap.measured["max_exposure_pct_nav"] == pytest.approx(5.0)


def test_cap_hand_check_high_is_looser_than_extreme() -> None:
    """Same book, HIGH instead of EXTREME: budget ``10% · 50 000 = 5 000``
    ⇒ cap 100 — which EQUALS the request, so the limit is satisfied and NO
    cap is emitted (the "a satisfied limit produces no cap" rule)."""
    caps = event_risk_caps(_high_snapshot(), requested_qty=100, price=50.0, nav=50_000.0)
    assert caps == []
    tighter = event_risk_caps(
        _high_snapshot(), requested_qty=200, price=50.0, nav=50_000.0
    )
    assert len(tighter) == 1 and tighter[0].cap_qty == 100


def test_satisfied_limit_produces_no_cap() -> None:
    """EXTREME, nav 200 000, price 50, requested 100: budget 10 000 ⇒ cap
    200 ≥ 100 ⇒ no cap. Mirrors ``stress_caps``/``statistical_caps``."""
    assert event_risk_caps(_extreme_snapshot(), requested_qty=100, price=50.0, nav=NAV) == []


def test_cap_shape_matches_pretrade_quantity_cap() -> None:
    """The cap is literally a ``pretrade.QuantityCap`` — the ONE shape
    ``assess(extra_caps=...)`` understands — with a valid layer, a
    non-empty explanatory sentence and numeric ``measured`` values."""
    caps = event_risk_caps(
        _extreme_snapshot(), requested_qty=100, price=50.0, nav=50_000.0
    )
    cap = caps[0]
    assert isinstance(cap, QuantityCap)
    assert cap.code == CODE_EVENT_EXPOSURE
    assert cap.layer == LAYER_CONCENTRATION
    assert cap.layer in CAP_LAYERS
    assert isinstance(cap.cap_qty, int) and cap.cap_qty >= 0
    assert cap.sentence
    for key, value in cap.measured.items():
        assert value is None or isinstance(value, float), key


def test_cap_sentence_carries_numbers_sample_size_and_shadow_marker() -> None:
    """§64 + §65: the justification names the expected move, its sample
    size when it is historical, the budget, and the fact that it does not
    bind."""
    hist_only = classify_event_risk(
        _inputs(historical_moves=(-14.0, -13.0, 15.0, 20.0, 18.0))
    )
    assert hist_only["event_risk_state"] == STATE_EXTREME  # median 15% ≥ 12
    caps = event_risk_caps(hist_only, requested_qty=100, price=50.0, nav=50_000.0)
    sentence = caps[0].sentence
    assert "based on 5 event(s)" in sentence
    assert "SHADOW" in sentence
    assert "not enforced" in sentence
    assert "UNVALIDATED" in sentence


def test_no_cap_below_high_warn_only() -> None:
    """LOW/MODERATE are WARN territory (§65) — they never emit a cap even at
    an absurd size."""
    for move in (1.0, 5.0):
        snap = classify_event_risk(_inputs(implied_move_pct=move))
        assert snap["event_risk_state"] in (STATE_LOW, STATE_MODERATE)
        assert event_risk_caps(snap, requested_qty=10_000, price=50.0, nav=1_000.0) == []


def test_unknown_state_emits_no_cap() -> None:
    """An unmeasured event must not produce a number that looks measured.
    Fail-open in SHADOW, recorded as the same open item Phase C/D carry."""
    snap = classify_event_risk(_inputs())
    assert snap["event_risk_state"] == STATE_UNKNOWN
    assert event_risk_caps(snap, requested_qty=100, price=50.0, nav=1_000.0) == []


@pytest.mark.parametrize(
    "price,nav",
    [(None, 50_000.0), (0.0, 50_000.0), (-5.0, 50_000.0), (50.0, None), (50.0, 0.0), (float("nan"), 50_000.0)],
)
def test_missing_or_bad_view_produces_no_cap(price, nav) -> None:
    """A view that could not be computed never produces a cap — and never
    raises on the order path."""
    assert event_risk_caps(_extreme_snapshot(), requested_qty=100, price=price, nav=nav) == []


def test_zero_and_negative_requested_qty() -> None:
    assert event_risk_caps(_extreme_snapshot(), requested_qty=0, price=50.0, nav=1.0) == []
    assert event_risk_caps(_extreme_snapshot(), requested_qty=-3, price=50.0, nav=1.0) == []
    with pytest.raises(ValueError, match="requested_qty"):
        event_risk_caps(_extreme_snapshot(), requested_qty=True, price=50.0, nav=1.0)


def test_empty_snapshot_is_safe() -> None:
    """A caller handing in ``{}`` (e.g. an upstream failure) gets no cap
    rather than an exception on the order path."""
    assert event_risk_caps({}, requested_qty=10, price=50.0, nav=NAV) == []


def test_policy_thresholds_are_parameters() -> None:
    """A stricter research policy tightens the cap — the numbers are
    RESEARCH DEFAULTS, and this proves they are not baked in."""
    strict = EventRiskPolicy(high_max_exposure_pct=2.0, extreme_max_exposure_pct=1.0)
    caps = event_risk_caps(
        _extreme_snapshot(), requested_qty=100, price=50.0, nav=50_000.0, policy=strict
    )
    assert caps[0].cap_qty == 10  # 1% · 50 000 = 500 → 500/50 = 10
    assert strict.mode == MODE_SHADOW


def test_caps_fold_into_the_hypothetical_shadow_verdict_only() -> None:
    """The caps compose with ``pretrade.shadow_verdict`` exactly like the
    statistical and stress ones — a HYPOTHETICAL resize, computed and
    reported, that touches no real decision because nothing here calls
    ``assess``."""
    caps = event_risk_caps(
        _extreme_snapshot(), requested_qty=100, price=50.0, nav=50_000.0
    )
    verdict = shadow_verdict(100, caps)
    assert verdict.hypothetical_quantity == 50
    assert verdict.hypothetical_decision == "APPROVE_WITH_RESIZE"
    assert verdict.binding == (CODE_EVENT_EXPOSURE,)
    assert verdict.mode == MODE_SHADOW


def test_caps_are_deterministic() -> None:
    snap = _extreme_snapshot()
    a = event_risk_caps(snap, requested_qty=100, price=50.0, nav=50_000.0)
    b = event_risk_caps(snap, requested_qty=100, price=50.0, nav=50_000.0)
    assert [(c.code, c.cap_qty, c.sentence) for c in a] == [
        (c.code, c.cap_qty, c.sentence) for c in b
    ]


def test_exported_from_the_risk_package() -> None:
    """The public surface other units import."""
    import libs.trading_core.risk as risk

    assert risk.classify_event_risk is classify_event_risk
    assert risk.event_risk_caps is event_risk_caps
    assert risk.historical_event_risk is historical_event_risk
    assert risk.EventRiskInputs is EventRiskInputs
    assert risk.EVENT_RISK_MODEL_VERSION == "event-risk-1.0.0"
    for name in (
        "EVENT_RISK_MODEL_VERSION",
        "EventRiskInputs",
        "EventRiskPolicy",
        "EventRiskSnapshot",
        "EventRiskThresholds",
        "classify_event_risk",
        "event_risk_caps",
        "historical_event_risk",
        "STATE_UNKNOWN",
    ):
        assert name in risk.__all__, name
