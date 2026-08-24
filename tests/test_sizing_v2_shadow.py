"""Sizing v2 — the §37 budget composition, the §36 risk-linked cash floor and
the §59 model-health budget effect, all SHADOW; plus the §65 decision-path
counters (spec compliance §3 Tier A rows §37 / §36 / §59 / §65).

**What was missing and what these tests pin.** The production budget composes
exactly two factors (``engine.py:490``)::

    budget = min(tier_budget(strength) x vol_multiplier, abs_max_trade_risk)

Spec §37 asks for five. `audit.md:226` (P1) committed to building the other
three — ES, correlation and model health — in SHADOW, and `audit.md:212`
separately committed to a §59 model-risk BUDGET EFFECT ("label NOW, budget
effect SHADOW"). Only the label shipped. §36's ``Risk up -> Floor up`` rule
held on no path at all. These tests are the arithmetic of all three, hand
checked, plus the guarantee that none of it moves a Tier 0 number.

Every number below is written out longhand so a reader can verify it without
running anything.
"""
from __future__ import annotations

import math

import pytest

from apps.gateway.execution import gate_chain

from libs.trading_core.risk.pretrade import (
    MODE_SHADOW,
    SizingV2Params,
    sizing_v2_shadow,
)

# ---------------------------------------------------------------------------
# (a) The three modifiers, one at a time
# ---------------------------------------------------------------------------


def test_es_modifier_is_the_target_over_the_measurement() -> None:
    """es_mod = clamp(es_target / es95, floor, 1.0) above target; 1.0 at or
    below it (a THROTTLE, never leverage).

    Hand-check with the default target 3 %:
      es95 = 4 %  -> 0.03 / 0.04 = 0.75
      es95 = 6 %  -> 0.03 / 0.06 = 0.50   (exactly at the 0.5 floor)
      es95 = 12 % -> 0.03 / 0.12 = 0.25   -> CLAMPED up to 0.50
      es95 = 3 %  -> at target             -> 1.0
      es95 = 1 %  -> below target          -> 1.0 (NOT 3.0)
    """
    def es_mod(es: float) -> float:
        return sizing_v2_shadow(
            es95_pct_nav=es,
            correlation_state="NORMAL",
            model_risk_state="LOW",
            drawdown_current_pct=0.0,
            regime_floor_pct=0.25,
            tier_budget_pct=0.01,
            vol_multiplier_used=1.0,
        ).es_modifier

    assert es_mod(0.04) == pytest.approx(0.75)
    assert es_mod(0.06) == pytest.approx(0.50)
    assert es_mod(0.12) == pytest.approx(0.50)  # floor, not 0.25
    assert es_mod(0.03) == 1.0
    assert es_mod(0.01) == 1.0  # never sizes UP


def test_correlation_modifier_is_the_research_default_table() -> None:
    """NORMAL 1.0 / ELEVATED 0.85 / CONVERGING 0.70 (§19 regimes, §37)."""
    def corr_mod(state):
        return sizing_v2_shadow(
            es95_pct_nav=0.01,
            correlation_state=state,
            model_risk_state="LOW",
            drawdown_current_pct=0.0,
            regime_floor_pct=0.25,
            tier_budget_pct=0.01,
            vol_multiplier_used=1.0,
        ).correlation_modifier

    assert corr_mod("NORMAL") == 1.0
    assert corr_mod("ELEVATED") == 0.85
    assert corr_mod("CONVERGING") == 0.70


def test_model_health_modifier_is_the_section_59_budget_effect() -> None:
    """LOW 1.0 / ELEVATED 0.85 / HIGH 0.70.

    THIS is the §59 gap: before this function existed, a HIGH model-risk
    state changed nothing about the hypothetical quantity the shadow window
    was accumulating evidence on.
    """
    def mh_mod(state):
        return sizing_v2_shadow(
            es95_pct_nav=0.01,
            correlation_state="NORMAL",
            model_risk_state=state,
            drawdown_current_pct=0.0,
            regime_floor_pct=0.25,
            tier_budget_pct=0.01,
            vol_multiplier_used=1.0,
        ).model_health_modifier

    assert mh_mod("LOW") == 1.0
    assert mh_mod("ELEVATED") == 0.85
    assert mh_mod("HIGH") == 0.70


# ---------------------------------------------------------------------------
# (b) The composition (§37)
# ---------------------------------------------------------------------------


def test_candidate_budget_is_the_full_five_factor_product() -> None:
    """The compliance report's own worked example, to the digit.

    tier 1 % (0.01) x vol 0.8 x es_mod 0.75 x corr 0.70 x mh 0.70

        0.01 x 0.8  = 0.008        <- what Tier 0 ACTUALLY used
        0.008 x 0.75 = 0.006
        0.006 x 0.70 = 0.0042
        0.0042 x 0.70 = 0.00294    = 0.294 % of NAV

    and the delta is 0.00294 - 0.008 = -0.00506 (v2 would size SMALLER).
    """
    r = sizing_v2_shadow(
        es95_pct_nav=0.04,          # 4 % against the 3 % target -> 0.75
        correlation_state="CONVERGING",  # -> 0.70
        model_risk_state="HIGH",         # -> 0.70
        drawdown_current_pct=-0.03,
        regime_floor_pct=0.40,
        tier_budget_pct=0.01,
        vol_multiplier_used=0.8,
    )
    assert r.es_modifier == pytest.approx(0.75)
    assert r.correlation_modifier == pytest.approx(0.70)
    assert r.model_health_modifier == pytest.approx(0.70)
    assert r.budget_pct_used == pytest.approx(0.008)
    assert r.candidate_budget_pct == pytest.approx(0.00294)
    assert r.budget_delta_pct == pytest.approx(0.00294 - 0.008)
    assert r.mode == MODE_SHADOW
    assert r.health == "ACTIVE"
    assert r.reason is None


def test_all_modifiers_benign_leaves_the_budget_exactly_tier_x_vol() -> None:
    """With ES below target, NORMAL correlation and LOW model risk, v2 and
    the production budget AGREE — the composition adds nothing it was not
    asked to add. 0.0125 x 1.0 x 1 x 1 x 1 = 0.0125."""
    r = sizing_v2_shadow(
        es95_pct_nav=0.02,
        correlation_state="NORMAL",
        model_risk_state="LOW",
        drawdown_current_pct=0.0,
        regime_floor_pct=0.15,
        tier_budget_pct=0.0125,
        vol_multiplier_used=1.0,
    )
    assert r.candidate_budget_pct == pytest.approx(0.0125)
    assert r.candidate_budget_pct == pytest.approx(r.budget_pct_used)
    assert r.budget_delta_pct == pytest.approx(0.0)


def test_the_composition_can_only_ever_throttle() -> None:
    """Property: every modifier lies in (0, 1], so the v2 candidate is never
    LARGER than what Tier 0 sized. A statistical layer that could RAISE a
    budget would be granting risk the hard limits refused."""
    for es in (None, 0.005, 0.03, 0.04, 0.30):
        for corr in (None, "NORMAL", "ELEVATED", "CONVERGING", "WEIRD"):
            for mr in (None, "LOW", "ELEVATED", "HIGH", "WEIRD"):
                r = sizing_v2_shadow(
                    es95_pct_nav=es,
                    correlation_state=corr,
                    model_risk_state=mr,
                    drawdown_current_pct=-0.10,
                    regime_floor_pct=0.40,
                    tier_budget_pct=0.01,
                    vol_multiplier_used=0.9,
                )
                assert 0 < r.es_modifier <= 1.0
                assert 0 < r.correlation_modifier <= 1.0
                assert 0 < r.model_health_modifier <= 1.0
                assert r.candidate_budget_pct <= r.budget_pct_used + 1e-15
                assert r.budget_delta_pct <= 1e-15


# ---------------------------------------------------------------------------
# (c) The risk-linked cash floor (§36) — "Risk up -> Floor up"
# ---------------------------------------------------------------------------


def test_risk_linked_cash_floor_binds_above_the_regime_floor() -> None:
    """The brief's worked case, longhand.

    regime floor 0.40 (NEUTRAL_RANGE), ES 4 % vs the 3 % target,
    drawdown -3 %, model risk ELEVATED:

        addon_es    = k_es (2.0)  x max(0, 0.04 - 0.03) = 2.0 x 0.01 = 0.020
        addon_dd    = k_dd (0.5)  x |-0.03|             = 0.5 x 0.03 = 0.015
        addon_model = model_risk_floor_addons[ELEVATED]              = 0.050
        floor = 0.40 + 0.020 + 0.015 + 0.050 = 0.485

    0.485 > 0.40, so it BINDS; 0.485 <= 0.90, so the cap does not apply.
    """
    r = sizing_v2_shadow(
        es95_pct_nav=0.04,
        correlation_state="NORMAL",
        model_risk_state="ELEVATED",
        drawdown_current_pct=-0.03,
        regime_floor_pct=0.40,
        tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    assert r.cash_floor_addons["es"] == pytest.approx(0.020)
    assert r.cash_floor_addons["drawdown"] == pytest.approx(0.015)
    assert r.cash_floor_addons["model_risk"] == pytest.approx(0.050)
    assert r.risk_linked_cash_floor_pct == pytest.approx(0.485)
    assert r.risk_linked_cash_floor_binds is True
    assert r.regime_floor_pct == 0.40


def test_quiet_book_leaves_the_regime_floor_exactly_alone() -> None:
    """ES at target, no drawdown, LOW model risk -> every addon 0.0, the
    floor IS the regime floor, and `binds` is False. Tier 0's floor must not
    move because a shadow layer looked at it."""
    r = sizing_v2_shadow(
        es95_pct_nav=0.03,
        correlation_state="NORMAL",
        model_risk_state="LOW",
        drawdown_current_pct=0.0,
        regime_floor_pct=0.25,
        tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    assert r.cash_floor_addons["es"] == 0.0
    assert r.cash_floor_addons["drawdown"] == 0.0
    assert r.cash_floor_addons["model_risk"] == 0.0
    assert r.risk_linked_cash_floor_pct == pytest.approx(0.25)
    assert r.risk_linked_cash_floor_binds is False


def test_the_cash_floor_is_capped_so_addons_cannot_demand_an_all_cash_book() -> None:
    """Catastrophic inputs against a 0.60 STRONG_BEAR floor:

        addon_es    = 2.0 x (0.50 - 0.03) = 0.94
        addon_dd    = 0.5 x 0.40          = 0.20
        addon_model =                       0.10
        raw = 0.60 + 0.94 + 0.20 + 0.10 = 1.84  -> capped at 0.90

    The raw number is still reported (nothing is hidden), but the floor the
    record publishes is the capped one.
    """
    r = sizing_v2_shadow(
        es95_pct_nav=0.50,
        correlation_state="CONVERGING",
        model_risk_state="HIGH",
        drawdown_current_pct=-0.40,
        regime_floor_pct=0.60,
        tier_budget_pct=0.005,
        vol_multiplier_used=0.5,
    )
    assert r.cash_floor_addons["raw_uncapped"] == pytest.approx(1.84)
    assert r.risk_linked_cash_floor_pct == pytest.approx(0.90)
    assert r.risk_linked_cash_floor_binds is True


def test_the_floor_never_falls_below_the_regime_floor() -> None:
    """Property over the whole input grid: `Risk up -> Floor up` is monotone
    and one-directional. A shadow floor BELOW Tier 0's would be a
    recommendation to hold LESS cash than the hard limit demands."""
    for es in (None, 0.0, 0.03, 0.09):
        for dd in (None, 0.0, -0.05, -0.30):
            for mr in (None, "LOW", "ELEVATED", "HIGH"):
                for regime_floor in (0.15, 0.40, 0.60):
                    r = sizing_v2_shadow(
                        es95_pct_nav=es,
                        correlation_state="NORMAL",
                        model_risk_state=mr,
                        drawdown_current_pct=dd,
                        regime_floor_pct=regime_floor,
                        tier_budget_pct=0.01,
                        vol_multiplier_used=1.0,
                    )
                    assert r.risk_linked_cash_floor_pct >= regime_floor
                    assert r.risk_linked_cash_floor_pct <= 0.90
                    assert r.risk_linked_cash_floor_binds == (
                        r.risk_linked_cash_floor_pct > regime_floor
                    )


def test_a_positive_drawdown_magnitude_reads_the_same_as_a_negative_one() -> None:
    """The drawdown block reports a NEGATIVE fraction; a floor addon is a
    magnitude. Both spellings of "3 % below the peak" give 0.5 x 0.03."""
    kw = dict(
        es95_pct_nav=0.03, correlation_state="NORMAL", model_risk_state="LOW",
        regime_floor_pct=0.40, tier_budget_pct=0.01, vol_multiplier_used=1.0,
    )
    a = sizing_v2_shadow(drawdown_current_pct=-0.03, **kw)
    b = sizing_v2_shadow(drawdown_current_pct=0.03, **kw)
    assert a.cash_floor_addons["drawdown"] == pytest.approx(0.015)
    assert a.risk_linked_cash_floor_pct == b.risk_linked_cash_floor_pct


# ---------------------------------------------------------------------------
# (d) Honest nulls — a missing input never reads as "that risk is low"
# ---------------------------------------------------------------------------


def test_every_missing_input_holds_its_modifier_at_one_with_a_note() -> None:
    """All four inputs absent: every modifier 1.0, every floor addon 0.0,
    health DEGRADED, and a note per missing input naming it. The budget is
    then exactly the production one — which is the honest answer, because
    nothing was measured."""
    r = sizing_v2_shadow(
        es95_pct_nav=None,
        correlation_state=None,
        model_risk_state=None,
        drawdown_current_pct=None,
        regime_floor_pct=0.25,
        tier_budget_pct=0.01,
        vol_multiplier_used=0.9,
    )
    assert r.es_modifier == 1.0
    assert r.correlation_modifier == 1.0
    assert r.model_health_modifier == 1.0
    assert r.candidate_budget_pct == pytest.approx(0.009)
    assert r.candidate_budget_pct == pytest.approx(r.budget_pct_used)
    assert r.risk_linked_cash_floor_pct == pytest.approx(0.25)
    assert r.risk_linked_cash_floor_binds is False

    assert r.health == "DEGRADED"
    assert len(r.notes) == 4
    joined = " ".join(r.notes)
    for name in (
        "es95_pct_nav",
        "correlation_state",
        "model_risk_state",
        "drawdown_current_pct",
    ):
        assert name in joined, name
    assert r.reason is not None and all(n in r.reason for n in r.notes)
    # The ES note says outright what it does NOT mean.
    assert "does NOT mean ES is low" in r.notes[0]


def test_one_missing_input_degrades_but_the_others_still_apply() -> None:
    """A missing correlation state does not silence the §59 modifier.
    0.01 x 1.0 x 1.0(es, no data) x 1.0(corr, no data) x 0.70(HIGH) = 0.007"""
    r = sizing_v2_shadow(
        es95_pct_nav=None,
        correlation_state=None,
        model_risk_state="HIGH",
        drawdown_current_pct=-0.02,
        regime_floor_pct=0.40,
        tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    assert r.model_health_modifier == 0.70
    assert r.candidate_budget_pct == pytest.approx(0.007)
    assert r.health == "DEGRADED"
    assert len(r.notes) == 2
    # The floor still gets the drawdown and model addons it CAN measure:
    # 0.40 + 0 (no ES) + 0.5 x 0.02 + 0.10 = 0.51
    assert r.risk_linked_cash_floor_pct == pytest.approx(0.51)


def test_an_unknown_state_is_held_at_one_rather_than_guessed() -> None:
    """A regime/state the parameter table does not know is a note, not an
    interpolation and not a KeyError that would take the block down."""
    r = sizing_v2_shadow(
        es95_pct_nav=0.03,
        correlation_state="STAMPEDE",
        model_risk_state="CATASTROPHIC",
        drawdown_current_pct=0.0,
        regime_floor_pct=0.40,
        tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    assert r.correlation_modifier == 1.0
    assert r.model_health_modifier == 1.0
    assert r.cash_floor_addons["model_risk"] == 0.0
    assert r.health == "DEGRADED"
    assert "STAMPEDE" in r.reason and "CATASTROPHIC" in r.reason


def test_a_non_finite_measurement_degrades_rather_than_propagating_a_nan() -> None:
    """NaN/inf in, 1.0 out with a note — never a NaN budget."""
    r = sizing_v2_shadow(
        es95_pct_nav=float("nan"),
        correlation_state="NORMAL",
        model_risk_state="LOW",
        drawdown_current_pct=float("inf"),
        regime_floor_pct=0.40,
        tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    assert r.es_modifier == 1.0
    assert math.isfinite(r.candidate_budget_pct)
    assert math.isfinite(r.risk_linked_cash_floor_pct)
    assert r.health == "DEGRADED"


# ---------------------------------------------------------------------------
# (e) Everything is a parameter, and nothing here can bind
# ---------------------------------------------------------------------------


def test_the_result_echoes_its_inputs_and_its_parameters() -> None:
    """Spec §44 reproducibility: the record carries what went in and what it
    was measured against, so an old shadow row is re-derivable."""
    params = SizingV2Params()
    r = sizing_v2_shadow(
        es95_pct_nav=0.04,
        correlation_state="ELEVATED",
        model_risk_state="ELEVATED",
        drawdown_current_pct=-0.01,
        regime_floor_pct=0.50,
        tier_budget_pct=0.0075,
        vol_multiplier_used=0.6,
        params=params,
    )
    assert r.inputs == {
        "es95_pct_nav": 0.04,
        "correlation_state": "ELEVATED",
        "model_risk_state": "ELEVATED",
        "drawdown_current_pct": -0.01,
        "regime_floor_pct": 0.50,
        "tier_budget_pct": 0.0075,
        "vol_multiplier_used": 0.6,
    }
    assert r.params is params
    # Re-derivable by hand from the echoed inputs:
    # 0.0075 x 0.6 x 0.75 x 0.85 x 0.85 = 0.00243843750
    assert r.candidate_budget_pct == pytest.approx(0.0024384375)


def test_the_research_defaults_are_the_documented_ones() -> None:
    """Every threshold a documented parameter, values UNVALIDATED (§11)."""
    p = SizingV2Params()
    assert p.es_target_pct_nav == 0.03
    assert p.es_modifier_floor == 0.5
    assert p.correlation_modifiers == {
        "NORMAL": 1.0, "ELEVATED": 0.85, "CONVERGING": 0.7,
    }
    assert p.model_risk_modifiers == {
        "LOW": 1.0, "ELEVATED": 0.85, "HIGH": 0.7,
    }
    assert p.k_es == 2.0
    assert p.k_drawdown == 0.5
    assert p.model_risk_floor_addons == {
        "LOW": 0.0, "ELEVATED": 0.05, "HIGH": 0.10,
    }
    assert p.max_cash_floor_pct == 0.90
    assert p.mode == MODE_SHADOW and p.is_shadow


def test_custom_parameters_move_every_number() -> None:
    """Nothing above is a hardcoded truth: a different parameter set gives a
    different answer. 0.01 x 1.0 x (0.02/0.04 = 0.5) x 0.5 x 0.5 = 0.00125,
    and the floor becomes 0.40 + 4.0 x 0.02 + 1.0 x 0.05 + 0.20 = 0.73."""
    p = SizingV2Params(
        es_target_pct_nav=0.02,
        es_modifier_floor=0.25,
        correlation_modifiers={"CONVERGING": 0.5},
        model_risk_modifiers={"HIGH": 0.5},
        k_es=4.0,
        k_drawdown=1.0,
        model_risk_floor_addons={"HIGH": 0.20},
        max_cash_floor_pct=0.95,
    )
    r = sizing_v2_shadow(
        es95_pct_nav=0.04,
        correlation_state="CONVERGING",
        model_risk_state="HIGH",
        drawdown_current_pct=-0.05,
        regime_floor_pct=0.40,
        tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
        params=p,
    )
    assert r.es_modifier == pytest.approx(0.5)
    assert r.candidate_budget_pct == pytest.approx(0.00125)
    assert r.risk_linked_cash_floor_pct == pytest.approx(0.73)


def test_params_reject_a_modifier_that_would_raise_a_budget() -> None:
    """A modifier > 1 is a leverage rule, which §37 does not ask for and no
    hard limit would sanction — rejected at construction."""
    with pytest.raises(ValueError, match="may only THROTTLE"):
        SizingV2Params(correlation_modifiers={"NORMAL": 1.2})
    with pytest.raises(ValueError, match="may only THROTTLE"):
        SizingV2Params(model_risk_modifiers={"LOW": 2.0})
    with pytest.raises(ValueError, match="es_modifier_floor"):
        SizingV2Params(es_modifier_floor=0.0)
    with pytest.raises(ValueError, match="max_cash_floor_pct"):
        SizingV2Params(max_cash_floor_pct=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        SizingV2Params(model_risk_floor_addons={"LOW": -0.1})
    with pytest.raises(ValueError, match="mode must be"):
        SizingV2Params(mode="ENFORCING")


def test_malformed_numbers_raise_but_missing_data_does_not() -> None:
    """Contract §1: a malformed input is a CALLER BUG and raises; missing
    data is a fact about the world and degrades health instead."""
    good = dict(
        es95_pct_nav=0.03, correlation_state="NORMAL", model_risk_state="LOW",
        drawdown_current_pct=0.0, regime_floor_pct=0.40, tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    for bad in (
        {"regime_floor_pct": -0.1},
        {"regime_floor_pct": 1.5},
        {"tier_budget_pct": float("nan")},
        {"vol_multiplier_used": -1.0},
    ):
        with pytest.raises(ValueError):
            sizing_v2_shadow(**{**good, **bad})
    # ...while every missing STATE is fine.
    assert sizing_v2_shadow(
        **{**good, "es95_pct_nav": None, "model_risk_state": None}
    ).health == "DEGRADED"


def test_the_record_is_frozen_and_says_it_is_shadow() -> None:
    r = sizing_v2_shadow(
        es95_pct_nav=0.04, correlation_state="NORMAL", model_risk_state="LOW",
        drawdown_current_pct=0.0, regime_floor_pct=0.40, tier_budget_pct=0.01,
        vol_multiplier_used=1.0,
    )
    assert r.mode == MODE_SHADOW == "SHADOW"
    with pytest.raises(Exception):
        r.candidate_budget_pct = 1.0  # type: ignore[misc]


def test_sizing_v2_imports_nothing_that_could_reach_the_engine() -> None:
    """Structural SHADOW guarantee: the composition is a pure function of
    numbers. It cannot call `assess`, and no Tier 0 module imports it."""
    import inspect

    import libs.trading_core.risk.engine as engine
    import libs.trading_core.risk.pretrade as pretrade

    assert not hasattr(engine, "sizing_v2_shadow")
    src = inspect.getsource(pretrade.sizing_v2_shadow)
    assert "assess" not in src
    assert "cash_floors" not in src


# ---------------------------------------------------------------------------
# (f) On the wire: `shadow.statistical.sizing_v2` and the §65 counters
# ---------------------------------------------------------------------------
#
# Same fixtures the Phase C shadow suite uses, so these tests measure the
# real gate chain rather than a stand-in.

from apps.gateway.routers import orders as orders_router  # noqa: E402

from .test_order_preview import BULL_TICKER, authorize, preview  # noqa: E402
from .test_orders_shadow_c import (  # noqa: E402
    _book_including_candidate,
    _latest_risk_decision,
)

#: Every key the ``sizing_v2`` block carries when it computed (contract:
#: the modifiers, both budgets, the floor and the provenance).
SIZING_V2_KEYS = {
    "es_modifier",
    "correlation_modifier",
    "model_health_modifier",
    "candidate_budget_pct",
    "budget_pct_used",
    "budget_delta_pct",
    "risk_linked_cash_floor_pct",
    "risk_linked_cash_floor_binds",
    "regime_floor_pct",
    "cash_floor_addons",
    "inputs",
    "health",
    "reason",
    "notes",
    "mode",
    "params",
    "note",
}


async def test_preview_carries_sizing_v2_in_the_audit_and_on_the_wire(client):
    """`shadow.statistical.sizing_v2` reaches BOTH the RISK_DECISION audit
    detail and the preview's `risk.shadow_statistical`, with every key, and
    the two agree byte for byte."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    details = (await _latest_risk_decision(client, BULL_TICKER))["details"]
    sizing = details["shadow"]["statistical"]["sizing_v2"]
    assert set(sizing) == SIZING_V2_KEYS
    assert sizing["mode"] == "SHADOW"

    # The mirror: audit and wire are the same content (a stored plan and an
    # audit row must never disagree about what the shadow layer said).
    assert body["risk"]["shadow_statistical"]["sizing_v2"] == sizing

    # The composition is REALLY the product of its parts, recomputable from
    # the block's own fields — no hidden factor.
    assert sizing["candidate_budget_pct"] == pytest.approx(
        sizing["budget_pct_used"]
        * sizing["es_modifier"]
        * sizing["correlation_modifier"]
        * sizing["model_health_modifier"]
    )
    assert sizing["budget_delta_pct"] == pytest.approx(
        sizing["candidate_budget_pct"] - sizing["budget_pct_used"]
    )
    # ...and `budget_pct_used` is the tier budget times the multiplier the
    # engine actually applied, which the audit publishes at top level.
    assert sizing["budget_pct_used"] == pytest.approx(
        sizing["inputs"]["tier_budget_pct"] * details["budget_multiplier"]
    )
    assert sizing["inputs"]["vol_multiplier_used"] == details["budget_multiplier"]

    # The regime floor echoed is Tier 0's own, and the risk-linked floor can
    # only sit on top of it.
    assert sizing["risk_linked_cash_floor_pct"] >= sizing["regime_floor_pct"]
    assert sizing["risk_linked_cash_floor_binds"] == (
        sizing["risk_linked_cash_floor_pct"] > sizing["regime_floor_pct"]
    )

    # Every threshold shipped with the payload, research defaults, UNVALIDATED.
    assert sizing["params"]["mode"] == "SHADOW"
    assert sizing["params"]["es_target_pct_nav"] == 0.03
    assert sizing["params"]["model_risk_modifiers"]["HIGH"] == 0.7


async def test_sizing_v2_never_moves_the_tier_zero_numbers(client):
    """SHADOW (§70). With `sizing_v2_shadow` RAISING, the decision, the
    approved quantity, the gates, the reason codes AND the budget multiplier
    are byte-identical — only a note appears where the block would have been.
    """
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    good_body = await preview(client, BULL_TICKER)
    good = (await _latest_risk_decision(client, BULL_TICKER))["details"]
    assert good["shadow"]["statistical"]["sizing_v2"]["health"] in {
        "ACTIVE",
        "DEGRADED",
    }

    def boom(*args, **kwargs):
        raise RuntimeError("sizing v2 exploded")

    # Patch the DEFINITION module: routers.orders only re-exports this name,
    # so rebinding it there would leave the caller in gate_chain untouched.
    monkeypatch_target = "sizing_v2_shadow"
    original = getattr(gate_chain, monkeypatch_target)
    setattr(gate_chain, monkeypatch_target, boom)
    try:
        broken_body = await preview(client, BULL_TICKER)
        broken = (await _latest_risk_decision(client, BULL_TICKER))["details"]
    finally:
        setattr(gate_chain, monkeypatch_target, original)

    for key in (
        "decision",
        "approved_quantity",
        "requested_quantity",
        "quantity_requested",
        "gates",
        "reason_codes",
        "veto_gate",
        "binding_constraints",
        "budget_multiplier",
        "limits",
    ):
        assert broken[key] == good[key], key
    assert broken_body["risk"]["decision"] == good_body["risk"]["decision"]
    assert (
        broken_body["risk"]["approved_quantity"]
        == good_body["risk"]["approved_quantity"]
    )

    # The ONLY difference: the block reports its own failure and nothing else.
    sizing = broken["shadow"]["statistical"]["sizing_v2"]
    assert sizing == {"note": "RuntimeError: sizing v2 exploded"}
    # ...and the rest of the shadow payload is untouched.
    assert (
        broken["shadow"]["statistical"]["comparison"]
        == good["shadow"]["statistical"]["comparison"]
    )


async def test_assess_still_receives_no_extra_caps_and_the_gates_are_tier_zero(client):
    """The two structural guarantees this batch must not break: the gate
    order is unchanged, and no `sizing_v2` name reaches the engine call."""
    import inspect

    assert gate_chain.GATE_ORDER == (
        "TRADING_POOL_AUTHORIZATION",
        "DATA_QUALITY",
        "REGIME",
        "DIRECTIONAL_SIGNAL",
        "VOLATILITY",
        "INSTRUMENT",
        # Phase D 2026-08-20: SQUEEZE_RISK is a REPORT-mode proxy gate
        # (never vetoes) — same discipline as LIQUIDITY, not a statistical
        # enforcement gate joining the chain.
        "SQUEEZE_RISK",
        "LIQUIDITY",
        "CONTRACT_SELECTION",
        "RISK_APPROVAL",
    )
    src = inspect.getsource(gate_chain.run_gate_chain)
    # The one `assess(` call in the chain composes the request, the snapshot,
    # the limits, the multiplier and the greeks — and nothing else.
    call = src[src.index("assessment = assess(") :]
    call = call[: call.index("\n        )\n") + 10]
    assert "extra_caps" not in call
    assert "sizing_v2" not in call
    assert "candidate_budget" not in call


async def test_section_65_decision_path_counters_reach_metrics(client):
    """The three §65 decision-path instruments are declared, exposed by
    /metrics with their HELP/TYPE headers, and rendered with their values."""
    body = (await client.get("/metrics")).text
    for name in ("risk_resize_count", "risk_reject_count", "stress_limit_blocks"):
        assert f"# TYPE {name} counter" in body, name
        assert f"# HELP {name} " in body, name

    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    await preview(client, BULL_TICKER)
    # ...plus one run that produces no order, so at least one instrument has
    # a value to render (a counter at 0 renders headers only, by design).
    await client.post("/api/orders/preview", json={"ticker": "ZZZZ"})

    # A counter that HAS been incremented renders its sample line with the
    # value the process holds — a header-only metric would be a time series
    # that never reports. (An untouched counter legitimately renders headers
    # only; that is the telemetry module's contract, not a gap here.)
    body = (await client.get("/metrics")).text
    rendered = False
    for name, metric in (
        ("risk_resize_count", gate_chain.RISK_RESIZE_COUNT),
        ("risk_reject_count", gate_chain.RISK_REJECT_COUNT),
        ("stress_limit_blocks", gate_chain.STRESS_LIMIT_BLOCKS),
    ):
        value = metric.value()
        if value:
            assert f"{name} {value:g}" in body, name
            rendered = True
    assert rendered, (
        "no §65 decision-path counter moved on a preview — the instruments "
        "would be inert"
    )


async def test_the_decision_counters_track_the_decision_they_are_named_for(client):
    """Whatever the chain decides, the counters move in lockstep with it:
    `risk_resize_count` only on APPROVE_WITH_RESIZE, `risk_reject_count` only
    when no order came out. A counter that drifted from the audit trail would
    make the §65 evidence series unreadable."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    seen = set()
    for quantity in (None, 1, 100_000):
        resize_before = gate_chain.RISK_RESIZE_COUNT.value()
        reject_before = gate_chain.RISK_REJECT_COUNT.value()
        body = await preview(client, BULL_TICKER, quantity=quantity)
        decision = body["risk"]["decision"]
        seen.add(decision)
        # ...and the audit says the same thing, which is the invariant: the
        # counters are incremented at the audit's own call site.
        details = (await _latest_risk_decision(client, BULL_TICKER))["details"]
        assert details["decision"] == decision

        assert gate_chain.RISK_RESIZE_COUNT.value() == resize_before + (
            1 if decision == "APPROVE_WITH_RESIZE" else 0
        )
        assert gate_chain.RISK_REJECT_COUNT.value() == reject_before + (
            1 if decision == "REJECT" else 0
        )
    assert seen, "no preview reached a decision"


async def test_reject_counter_counts_a_chain_that_produced_no_order(client):
    """§65's `risk_reject_count` is "no order came out". A ticker with no
    history at all cannot be sized, so the chain refuses — and the counter
    records exactly one refusal for exactly one RISK_DECISION event."""
    reject_before = gate_chain.RISK_REJECT_COUNT.value()
    r = await client.post("/api/orders/preview", json={"ticker": "ZZZZ"})
    assert r.status_code == 200
    body = r.json()
    details = (await _latest_risk_decision(client, "ZZZZ"))["details"]
    # Either the engine rejected, or an earlier gate vetoed before `assess`
    # ran (decision "VETOED") — both are "no order", both count once.
    assert details["decision"] in {"REJECT", "VETOED"}
    assert body["risk"] is None or body["risk"]["approved_quantity"] == 0
    assert gate_chain.RISK_REJECT_COUNT.value() == reject_before + 1


async def test_stress_limit_blocks_counts_a_binding_shadow_stress_cap(client):
    """`stress_limit_blocks` fires exactly when a SHADOW stress cap's
    `cap_qty` sits BELOW the quantity that actually stood — and never when
    the stress layer produced no cap."""
    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    before = gate_chain.STRESS_LIMIT_BLOCKS.value()
    body = await preview(client, BULL_TICKER)
    after = gate_chain.STRESS_LIMIT_BLOCKS.value()

    stress = body["risk"]["shadow_statistical"]["stress"]
    cap = stress.get("cap") if stress else None
    approved = body["risk"]["approved_quantity"]
    would_bind = cap is not None and cap["cap_qty"] < approved
    assert after == before + (1 if would_bind else 0)

    # Whatever it counted, Tier 0 is untouched: the stress layer is SHADOW,
    # so the approved quantity is the engine's, cap or no cap.
    assert body["risk"]["decision"] in {"APPROVE", "APPROVE_WITH_RESIZE"}
    if stress:
        assert stress["mode"] == "SHADOW"


def test_the_tier_budget_lookup_matches_the_engines_own() -> None:
    """The shadow block reconstructs the tier budget from `limits` by the
    strength-tier NAME the engine resolved. This test pins that
    reconstruction against the engine's own private table, so a new tier or a
    renamed budget field fails HERE rather than silently logging a wrong
    `budget_pct_used` into the shadow window's evidence.
    """
    from libs.trading_core.risk.engine import RiskLimits, _tier_budget

    limits = RiskLimits()
    router_table = {
        "VERY_STRONG": limits.budget_very_strong,
        "STRONG": limits.budget_strong,
        "MODERATE": limits.budget_moderate,
        "WEAK": limits.budget_weak,
    }
    for name, value in router_table.items():
        assert _tier_budget(name, limits) == value, name
    # ...and no tier the engine knows is missing from the router's table.
    with pytest.raises(KeyError):
        _tier_budget("NOT_A_TIER", limits)
