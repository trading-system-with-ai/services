"""Compliance batch 3 (Tier C sweep) — the display, nomenclature and
structural gaps §3 of `docs/risk-engine-spec-compliance.md` enumerated.

One section per spec §, in the order the batch closed them:

- §5  a machine-readable tier taxonomy (`ModelTier`, `ModelMeta.tier`);
- §45 `stress` as a TYPED field on `PortfolioRiskSnapshot`;
- §46 net vega before/after;
- §8  first-class incremental VaR;
- §40 the worst stress loss as a dispersion view;
- §10 ES contributions at 99 % with the "noisy at 99%" warning;
- §26 `spot_shock_by_ticker` on the user stress endpoint;
- §13 the persisted `GarchFit` diagnostics row;
- §55 `is_stale` given a consumer inside the shadow layer.

EVERYTHING HERE IS SHADOW. Several tests assert that directly: the §55
suppression is pinned with the sabotage pattern (a Tier 0 decision battery
is byte-identical with the staleness rule firing and not firing), and no
test in this file expects a changed `assess()` result.
"""
import dataclasses
import math
import random
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from apps.gateway import risk_snapshot as rs
from apps.gateway.routers import risk as risk_router
from apps.gateway.db import (
    Position,
    RiskContributionRow,
    RiskMetricRow,
    SessionLocal,
    StockBarDaily,
)
from libs.trading_core.risk.models import base as base_mod
from libs.trading_core.risk.models import stress as stress_models
from libs.trading_core.risk.models.base import (
    REGISTRY,
    ModelHealth,
    ModelMeta,
    ModelTier,
    tier_for_model_name,
)
from libs.trading_core.risk.models.garch import fit_garch
from libs.trading_core.risk.pretrade import (
    DECISION_APPROVE,
    DECISION_UNAVAILABLE_STALE,
    CandidateSpec,
    QuantityCap,
    shadow_verdict,
)

from tests.test_risk_snapshot_builder import (
    INITIAL_CASH,
    build,
    seed_stock_position,
)

# ---------------------------------------------------------------------------
# §5 — the tier taxonomy
# ---------------------------------------------------------------------------


def test_every_registered_model_carries_a_tier():
    """§5's actual gap: "a reader cannot ask a model what tier it is".

    Every model in the registry must answer, on the instance AND on the
    ``ModelMeta`` it stamps — a tier that lives only on the class would not
    reach the wire or the persisted row.
    """
    assert REGISTRY, "the registry is empty — model modules did not import"
    for name, model in sorted(REGISTRY.items()):
        tier = getattr(model, "tier", None)
        assert tier is not None, f"{name} carries no tier"
        assert isinstance(tier, ModelTier), f"{name}.tier is not a ModelTier"
        assert model.metadata().tier is tier, (
            f"{name}.metadata() dropped the tier the model declares"
        )


def test_the_tier_taxonomy_classifies_the_registry_the_way_spec_5_does():
    """The four unconditional VaR/ES estimators are Tier 1; GARCH is Tier 2.

    Hardcoded rather than derived: this test is the CLASSIFICATION, and a
    test that recomputed it from the same source would assert nothing.
    """
    assert {n: str(m.tier) for n, m in REGISTRY.items()} == {
        "historical_var": "TIER_1",
        "historical_es": "TIER_1",
        "gaussian_var": "TIER_1",
        "gaussian_es": "TIER_1",
        "garch11": "TIER_2",
    }


def test_tier_is_orthogonal_to_mode():
    """§5 (tier) and §70 (mode) answer different questions.

    GARCH is the case that proves it: Tier 2 by family, RESEARCH by
    lifecycle. If tier were derived from mode the two would move together.
    """
    garch = REGISTRY["garch11"]
    hist = REGISTRY["historical_var"]
    assert garch.tier is ModelTier.TIER_2 and str(garch.mode) == "RESEARCH"
    assert hist.tier is ModelTier.TIER_1 and str(hist.mode) == "SHADOW"


def test_model_meta_tier_defaults_to_none_and_rejects_a_bad_spelling():
    """Additive means additive: an unclassified meta is ``None``, never
    TIER_0 — defaulting into the tier that DECIDES is the one dishonest
    default available. A bad spelling is malformed input, not a silent
    ``None``: the field is only useful if it can be trusted when present.
    """
    assert ModelMeta("m", "1.0.0").tier is None
    assert ModelMeta("m", "1.0.0", tier=ModelTier.TIER_2).tier is ModelTier.TIER_2
    # A valid string spelling is coerced to the enum.
    assert ModelMeta("m", "1.0.0", tier="TIER_1").tier is ModelTier.TIER_1
    with pytest.raises(ValueError):
        ModelMeta("m", "1.0.0", tier="TIER_9")


def test_tier_lookup_resolves_the_conditional_family_by_name():
    """The names that never pass through the registry still get a tier.

    The snapshot builder stamps ``garch_<name>`` on a GARCH-filtered view
    and the validation grid uses ``conditional_var`` / ``garch_var`` — none
    of which is a registry entry. The conditional-family name test is what
    keeps those TIER_2 instead of defaulting to TIER_1.
    """
    assert tier_for_model_name("historical_var") is ModelTier.TIER_1
    assert tier_for_model_name("garch11") is ModelTier.TIER_2
    for conditional in (
        "conditional_var",
        "conditional_es",
        "garch_var",
        "garch_historical_es",
        "ewma_vol",
    ):
        assert tier_for_model_name(conditional) is ModelTier.TIER_2, conditional
    # Not a model ⇒ honest null, never a guessed tier.
    assert tier_for_model_name("") is None
    assert tier_for_model_name(None) is None  # type: ignore[arg-type]


def test_stress_declares_the_first_tier_the_spec_lists_it_in():
    """§5 puts stress testing in the FIRST tier: a deterministic reprice of
    today's book, with no fitted parameter in it."""
    assert stress_models.MODEL_TIER is ModelTier.TIER_1


async def test_api_metric_rows_and_persisted_params_carry_the_tier(client):
    """The taxonomy has to REACH a reader: every served VaR/ES row carries
    its tier, and the persisted `risk_metrics.params` carries it too so a
    replayed row answers the same question without a join."""
    await seed_stock_position(bars=200)
    result = await build()

    rows = result.api["var"] + result.api["es"]
    assert rows, "no metric rows were served"
    for row in rows:
        assert row["tier"] in {"TIER_1", "TIER_2"}, row
        # The conditional (vol-scaled) views are the Tier 2 ones; every
        # unconditional estimator is Tier 1.
        expected = "TIER_2" if row["model"] == "HISTORICAL_VOL_SCALED" else "TIER_1"
        assert row["tier"] == expected, row

    async with SessionLocal() as session:
        persisted = (
            await session.execute(
                select(RiskMetricRow).where(RiskMetricRow.snapshot_id == result.row_id)
            )
        ).scalars().all()
    assert persisted
    for row in persisted:
        if row.metric in ("VAR", "ES"):
            assert row.params.get("tier") in {"TIER_1", "TIER_2"}, row.metric


# ---------------------------------------------------------------------------
# §45 — the typed stress field
# ---------------------------------------------------------------------------


async def test_typed_snapshot_and_api_stress_block_agree(client):
    """§45's remaining half. `correlation_state` was a declared field the
    builder never populated, so the typed snapshot and the API silently
    disagreed for all of Phase C. `stress` is now typed AND populated —
    this test is the regression that keeps the two in agreement.
    """
    await seed_stock_position(bars=200)
    result = await build()

    typed = result.snapshot.stress
    block = result.api["stress"]
    assert typed is not None, "the builder left the typed stress field None"

    # Worst row: same scenario, same number, same health on both surfaces.
    assert (block["worst"] is None) == (typed.worst is None)
    if typed.worst is not None:
        assert block["worst"]["name"] == typed.worst.name
        assert block["worst"]["loss_usd"] == typed.worst.loss_usd
        assert block["worst"]["health"] == str(typed.worst.health)
    assert block["health"] == str(typed.health)
    assert block["catalogue_version"] == typed.catalogue_version
    assert [r["name"] for r in block["rows"]] == [r.name for r in typed.rows]


def test_the_stress_field_is_declared_on_the_snapshot_dataclass():
    """A declared-but-unpopulated field is the §45 defect itself, so pin the
    declaration too — the type must admit a StressResult and None."""
    from libs.trading_core.risk.snapshot import PortfolioRiskSnapshot

    names = {f.name for f in dataclasses.fields(PortfolioRiskSnapshot)}
    assert "stress" in names


async def test_stress_does_not_enter_overall_health(client):
    """ADDITIVE means the existing health algebra is untouched: `stress` is
    NOT folded into `health_summary`, so `overall_health` is what it was."""
    await seed_stock_position(bars=200)
    snapshot = (await build()).snapshot
    assert "stress" not in snapshot.health_summary()


# ---------------------------------------------------------------------------
# §46 — net vega before / after
# ---------------------------------------------------------------------------


def _candidate(**kw) -> CandidateSpec:
    base = dict(
        key="AAPL#candidate",
        ticker="AAPL",
        instrument="LONG_CALL",
        multiplier=100,
        spot=100.0,
        delta=0.5,
        max_loss_per_unit=500.0,
        capital_per_unit=500.0,
        quantity_requested=3,
    )
    base.update(kw)
    return CandidateSpec(**base)


def test_candidate_vega_at_scales_by_quantity_and_multiplier():
    """Hand-checked: 3 contracts × 100 × $0.12/share = $36 per IV point."""
    cand = _candidate(vega0=0.12)
    assert cand.vega_at(3) == pytest.approx(36.0)
    assert cand.vega_at(0) == pytest.approx(0.0)


def test_candidate_without_a_vega_is_an_honest_null_not_a_zero():
    """A chain that gave no vega is UNMEASURED. Returning 0.0 would let a
    real vol exposure disappear into a net that looked complete."""
    assert _candidate(vega0=None).vega_at(5) is None
    # ...whereas an explicit 0.0 IS a measurement (stock has no vega).
    assert _candidate(vega0=0.0).vega_at(5) == 0.0


def test_a_non_finite_vega_is_malformed():
    with pytest.raises(ValueError):
        _candidate(vega0=float("nan"))


def test_net_vega_after_is_null_when_either_side_is_unmeasured():
    """Both directions of the honest null (contract §7.1)."""
    from libs.trading_core.risk.pretrade import _net_vega_after

    cand = _candidate(vega0=0.12)
    # book unreadable ⇒ null (the candidate alone is not the book's net)
    assert _net_vega_after(None, cand, 3) is None
    # candidate unreadable ⇒ null (a net that drops it reads as "no vega")
    assert _net_vega_after(500.0, _candidate(vega0=None), 3) is None
    # both measured ⇒ real arithmetic
    assert _net_vega_after(500.0, cand, 3) == pytest.approx(536.0)


def test_a_short_leg_vega_reduces_the_book_net():
    """Sign discipline: the caller negates a short leg exactly as it does
    for delta, and this module never re-signs anything."""
    from libs.trading_core.risk.pretrade import _net_vega_after

    short = _candidate(vega0=-0.12)
    assert _net_vega_after(500.0, short, 3) == pytest.approx(464.0)


async def test_preview_serves_a_net_vega_row_and_fields(client):
    """The §46 row reaches the wire in the same shape every other row uses,
    with the top-level fields beside it."""
    from tests.test_orders_shadow_c import (
        BULL_TICKER,
        _book_including_candidate,
        authorize,
        preview,
    )

    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    comparison = body["risk"]["comparison"]

    assert "net_vega_before" in comparison and "net_vega_after" in comparison
    row = next(r for r in comparison["rows"] if r["metric"] == "net_vega")
    assert row["layer"] == "STATISTICAL"
    # $ per IV point is NOT USD exposure, so the percent columns stay null
    # rather than dividing a greek by NAV.
    assert row["before_pct_nav"] is None and row["after_pct_nav"] is None
    if row["before_usd"] is not None and row["after_usd"] is not None:
        assert row["delta_usd"] == pytest.approx(
            row["after_usd"] - row["before_usd"]
        )
    else:
        assert row["delta_usd"] is None and row["reason"]


# ---------------------------------------------------------------------------
# §8 — first-class incremental VaR
# ---------------------------------------------------------------------------


def _compare_on_a_toy_book(quantity: int = 2, *, net_vega_before=250.0):
    """The pretrade suite's own hand-checked fixture, compared at ``quantity``.

    Deliberately REUSED rather than rebuilt: the §8 and §46 fields must be
    measured on the same book every other pre-trade invariant is pinned on,
    or a divergence between the two fixtures would hide a divergence
    between the two numbers.
    """
    from libs.trading_core.risk.pretrade import compare

    from tests.test_risk_pretrade import (
        LIMITS_SMALL,
        NAV,
        POSITIONS,
        SMALL,
        book,
        candidate,
    )

    book_pnl, mtx = book()
    cand = dataclasses.replace(candidate(), vega0=0.12)
    return compare(
        book_pnl,
        cand,
        quantity,
        returns=mtx,
        nav=NAV,
        heat_before=0.01,
        heat_after=0.02,
        cash_before=0.5,
        cash_after=0.49,
        positions=POSITIONS,
        delta_notional_before=30_000.0,
        net_vega_before=net_vega_before,
        limits=LIMITS_SMALL,
        contribution_params=SMALL,
    )


def test_incremental_var_is_var_after_minus_var_before_exactly():
    """§8's gap: incremental VaR existed only as `MetricPair.delta_usd`.

    The arithmetic is pinned EXACTLY (`==`, not approx) against the same
    pair the table renders, so the named field and the row can never
    disagree — that disagreement is the whole failure mode.
    """
    cmp_ = _compare_on_a_toy_book()
    assert cmp_.is_available
    before = cmp_.var_hist_95.before.value
    after = cmp_.var_hist_95.after.value
    assert cmp_.incremental_var_95_usd == after - before
    assert cmp_.incremental_var_95_usd == cmp_.var_hist_95.delta_usd


def test_incremental_var_pct_nav_divides_by_the_same_nav():
    """The percent field is the USD field over the SAME NAV the comparison
    was measured against — never a second denominator."""
    from tests.test_risk_pretrade import NAV

    cmp_ = _compare_on_a_toy_book()
    assert cmp_.incremental_var_95_pct_nav == cmp_.incremental_var_95_usd / NAV


def test_incremental_var_follows_the_same_discipline_as_incremental_es():
    """Both increments are measured on the same n, the same k and the same
    two books, so they are comparable to each other."""
    cmp_ = _compare_on_a_toy_book()
    assert cmp_.incremental_es_95_usd is not None
    assert cmp_.incremental_var_95_usd is not None
    assert cmp_.tail_size_95 is not None
    # Same n and same tail k on both sides of BOTH increments — that shared
    # discipline is what makes the two numbers comparable to each other.
    assert cmp_.var_hist_95.before.sample_size == cmp_.es_hist_95.before.sample_size
    assert cmp_.var_hist_95.after.sample_size == cmp_.es_hist_95.after.sample_size
    assert cmp_.var_hist_95.before.sample_size == cmp_.n_obs


def test_both_increments_are_null_together_when_the_view_is_unavailable():
    """A difference of nulls is not a number (contract §7.1): when the
    statistical view cannot be computed, incremental VaR is an honest null
    exactly as incremental ES already was."""
    from libs.trading_core.risk.pretrade import StatisticalLimits, compare

    from tests.test_risk_pretrade import NAV, POSITIONS, book, candidate

    book_pnl, mtx = book()
    cand = dataclasses.replace(candidate(), vega0=0.12)
    # min_obs far above the fixture's 8 observations ⇒ UNAVAILABLE.
    cmp_ = compare(
        book_pnl, cand, 2,
        returns=mtx, nav=NAV,
        heat_before=0.01, heat_after=0.02,
        cash_before=0.5, cash_after=0.49,
        positions=POSITIONS,
        delta_notional_before=30_000.0,
        net_vega_before=250.0,
        limits=StatisticalLimits(min_obs=500),
    )
    assert not cmp_.is_available
    assert cmp_.incremental_es_95_usd is None
    assert cmp_.incremental_var_95_usd is None
    assert cmp_.incremental_var_95_pct_nav is None
    # ...but net vega SURVIVES an unavailable statistical view: a book with
    # no priceable history still has a real, readable vega today.
    assert cmp_.net_vega[0] == 250.0
    assert cmp_.net_vega[1] == pytest.approx(250.0 + 2 * 8 * 0.12)


async def test_preview_serves_incremental_var_as_a_named_row(client):
    from tests.test_orders_shadow_c import (
        BULL_TICKER,
        _book_including_candidate,
        authorize,
        preview,
    )

    await _book_including_candidate()
    await authorize(client, BULL_TICKER)
    comparison = (await preview(client, BULL_TICKER))["risk"]["comparison"]

    assert "incremental_var_95_usd" in comparison
    assert "incremental_var_95_pct_nav" in comparison
    row = next(r for r in comparison["rows"] if r["metric"] == "incremental_var_95")
    var_row = next(r for r in comparison["rows"] if r["metric"] == "var_hist_95")
    # The named row and the VaR pair it is derived from must agree.
    assert row["before_usd"] == var_row["before_usd"]
    assert row["after_usd"] == var_row["after_usd"]
    assert row["delta_usd"] == var_row["delta_usd"]


# ---------------------------------------------------------------------------
# §40 — the worst stress loss as a dispersion view
# ---------------------------------------------------------------------------


def test_the_stress_pseudo_view_carries_the_worst_loss_and_no_confidence():
    """A scenario reprice has no tail PROBABILITY. Stamping 0.95 on it would
    invite a reader to compare it with the VaR views as a quantile."""
    scenario = stress_models.ScenarioResult(
        name="worst", kind=stress_models.KIND_HYPOTHETICAL, validated=False,
        pnl_usd=-500.0, pnl_pct_nav=-0.005, per_key={}, method_coverage={},
        health=ModelHealth.ACTIVE, reason=None, params={},
    )
    result = stress_models.StressResult(
        rows=(scenario,), worst=scenario, health=ModelHealth.ACTIVE,
        min_pnl_usd=-500.0,
    )
    view = rs._stress_dispersion_view(result, as_of=date(2026, 8, 19))
    assert view is not None
    assert view.value == pytest.approx(500.0)  # LOSS-positive, the VaR sign
    assert view.meta.confidence is None
    assert view.meta.tier is ModelTier.TIER_1
    assert view.health is ModelHealth.ACTIVE


def test_the_stress_pseudo_view_is_absent_rather_than_zero_when_unmeasured():
    """A gap is not a $0 loss. An absent view is simply not compared; a 0.0
    would be excluded by the ensemble anyway and only inflate n_excluded."""
    assert rs._stress_dispersion_view(None, as_of=None) is None

    gain = stress_models.ScenarioResult(
        name="gain", kind=stress_models.KIND_HYPOTHETICAL, validated=False,
        pnl_usd=+500.0, pnl_pct_nav=0.005, per_key={}, method_coverage={},
        health=ModelHealth.ACTIVE, reason=None, params={},
    )
    every_scenario_gains = stress_models.StressResult(
        rows=(gain,), worst=gain, health=ModelHealth.ACTIVE, min_pnl_usd=+500.0,
    )
    assert rs._stress_dispersion_view(every_scenario_gains, as_of=None) is None

    unavailable = stress_models.StressResult(
        rows=(), worst=None, health=ModelHealth.UNAVAILABLE,
        min_pnl_usd=None, reason="no catalogue",
    )
    assert rs._stress_dispersion_view(unavailable, as_of=None) is None


async def test_dispersion_now_spans_the_statistical_and_stress_families(client):
    """§40's own worked example spans exactly this: its widest view is a
    stress number. Excluding stress made the reported disagreement
    UNDERSTATE the spread on a stress-dominated book."""
    await seed_stock_position(bars=200)
    api = (await build()).api

    dispersion = api["dispersion"]
    assert dispersion is not None
    # The stress view is IN the comparison...
    assert dispersion["n_comparable"] >= 4
    # ...and on a real book it is the widest, which is the whole point.
    assert dispersion["max_model"] == rs.DISPERSION_STRESS_KEY
    # The statistical views are still compared among themselves.
    assert dispersion["min_model"] in rs.DISPERSION_VIEW_KEYS
    assert dispersion["ratio"] > 1.0


# ---------------------------------------------------------------------------
# §10 — ES contributions at 99 %
# ---------------------------------------------------------------------------


async def test_es99_contributions_are_served_beside_es95(client):
    """§10's gap: the 99 % RC the audit called "noisy" was ABSENT rather
    than shown with a warning."""
    await seed_stock_position(bars=520)
    api = (await build()).api

    block = api["contributions"]["es99"]
    assert block is not None
    assert block["confidence"] == 0.99
    assert api["contributions"]["es"]["confidence"] == 0.95
    assert [r["key"] for r in block["rows"]] == [
        r["key"] for r in api["contributions"]["es"]["rows"]
    ]


async def test_es99_contributions_sum_to_their_total(client):
    """The Euler identity holds at 99 % exactly as it does at 95 %."""
    await seed_stock_position(bars=520)
    block = (await build()).api["contributions"]["es99"]
    assert block["total_usd"] is not None
    assert math.fsum(r["contribution_usd"] for r in block["rows"]) == pytest.approx(
        block["total_usd"], rel=1e-12
    )


def test_the_noisy_at_99_warning_fires_on_a_thin_tail_and_never_upgrades():
    """The audit anticipated this number and called it noisy. It ships WITH
    the warning rather than absent — the honest-null rule applied to
    PRECISION instead of availability.
    """
    from libs.trading_core.risk.models.contribution import ContributionResult

    def _result(tail: int, health=ModelHealth.ACTIVE, reason=None):
        # An UNAVAILABLE result carries no total (the library's own honest-
        # null rule), so the fixture respects it rather than working around it.
        unavailable = health in (ModelHealth.UNAVAILABLE, ModelHealth.FAILED)
        return ContributionResult(
            total=None if unavailable else 100.0,
            per_position=(), method="ES", confidence=0.99,
            tail_size=tail, health=health, reason=reason, sample_size=600,
            meta=ModelMeta("es_contributions", "1.0.0"),
        )

    # Thin tail ⇒ DEGRADED with the real k in the reason.
    warned = rs._es99_noise_warning(_result(tail=6))
    assert warned.health is ModelHealth.DEGRADED
    assert "k=6" in warned.reason and "noisy at 99%" in warned.reason

    # A fat enough tail is left exactly alone.
    fat = _result(tail=rs.ES99_NOISY_TAIL_MIN)
    assert rs._es99_noise_warning(fat) is fat

    # Never upgrades: an UNAVAILABLE result keeps its worse health, and its
    # existing reason survives.
    worse = rs._es99_noise_warning(
        _result(tail=2, health=ModelHealth.UNAVAILABLE, reason="n too small")
    )
    assert worse.health is ModelHealth.UNAVAILABLE
    assert "n too small" in worse.reason

    # The threshold is a PARAMETER, not a hardcoded truth.
    assert rs._es99_noise_warning(_result(tail=6), min_tail=3).health is (
        ModelHealth.ACTIVE
    )


async def test_es99_rows_persist_at_confidence_099(client):
    """Persisted under the SAME `method="ES"` and told apart by `confidence`
    — which the row already carries — rather than a new method spelling
    every existing reader would have to learn."""
    await seed_stock_position(bars=520)
    result = await build()

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(RiskContributionRow).where(
                    RiskContributionRow.snapshot_id == result.row_id
                )
            )
        ).scalars().all()

    es99 = [r for r in rows if r.method == "ES" and r.confidence == 0.99]
    es95 = [r for r in rows if r.method == "ES" and r.confidence == 0.95]
    assert es99 and es95
    assert {r.position_key for r in es99} == {r.position_key for r in es95}


# ---------------------------------------------------------------------------
# §26 — spot_shock_by_ticker on the user stress endpoint
# ---------------------------------------------------------------------------


async def test_stress_run_accepts_per_ticker_shocks(client):
    """§26's "SPY −5% / QQQ −8%" becomes expressible. The map OVERRIDES the
    uniform shock for the tickers it names; everything else keeps it."""
    r = await client.post(
        "/api/risk/stress/run",
        json={
            "equity_shock": -0.03,
            "spot_shock_by_ticker": {"spy": -0.05, "QQQ": -0.08},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    params = body["scenario"]["params"]
    # Keys are uppercased; values pass through unclamped.
    assert params["spot_shock_by_ticker"] == {"SPY": -0.05, "QQQ": -0.08}
    assert params["spot_shock"] == -0.03
    # The row STOPS claiming the beta = 1 uniform assumption, because it no
    # longer holds for the names the map overrides — and the note says so
    # with the real numbers rather than a fixed sentence.
    assert params["uniform_beta_1"] is False
    assert "OVERRIDE" in (params["notes"] or "")
    assert "SPY" in (params["notes"] or "")


async def test_a_uniform_run_is_unchanged_by_the_new_field(client):
    """ADDITIVE: omitting the map is byte-identical to the old behaviour."""
    r = await client.post("/api/risk/stress/run", json={"equity_shock": -0.03})
    assert r.status_code == 200, r.text
    params = r.json()["scenario"]["params"]
    assert params["spot_shock_by_ticker"] == {}


@pytest.mark.parametrize(
    "shocks",
    [
        {"SPY": -0.95},                       # below EQUITY_SHOCK_MIN
        {"SPY": 2.5},                         # above EQUITY_SHOCK_MAX
        {"": -0.05},                          # not a ticker
        {"WAYTOOLONGTICKER": -0.05},          # over MAX_TICKER_LENGTH
        {"SPY": "down"},                      # not a number
        {"spy": -0.05, "SPY": -0.06},         # collides once uppercased
    ],
)
async def test_out_of_range_per_ticker_shocks_are_422_never_clamped(client, shocks):
    """Spec §26: silently running a DIFFERENT scenario than the one asked
    for is the dishonest option, so every one of these is a 422."""
    r = await client.post(
        "/api/risk/stress/run",
        json={"equity_shock": -0.03, "spot_shock_by_ticker": shocks},
    )
    assert r.status_code == 422, (shocks, r.text)


async def test_too_many_per_ticker_shocks_is_422(client):
    r = await client.post(
        "/api/risk/stress/run",
        json={
            "equity_shock": -0.03,
            "spot_shock_by_ticker": {
                f"T{i}": -0.05
                for i in range(risk_router.MAX_SPOT_SHOCK_TICKERS + 1)
            },
        },
    )
    assert r.status_code == 422


async def test_a_per_ticker_run_gets_its_own_persisted_name(client):
    """Two runs with the same headline numbers must not share a label, or
    the persisted history shows two scenarios under one name."""
    uniform = await client.post("/api/risk/stress/run", json={"equity_shock": -0.03})
    per_ticker = await client.post(
        "/api/risk/stress/run",
        json={"equity_shock": -0.03, "spot_shock_by_ticker": {"SPY": -0.05}},
    )
    assert uniform.json()["scenario"]["name"] != per_ticker.json()["scenario"]["name"]


# ---------------------------------------------------------------------------
# §13 — the persisted GarchFit diagnostics
# ---------------------------------------------------------------------------


def _garch_series(n: int, seed: int = 7) -> list[float]:
    """A GARCH(1,1) path, written out independently of the estimator."""
    rng = random.Random(seed)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    variance = omega / (1.0 - alpha - beta)
    out: list[float] = []
    previous = 0.0
    for _ in range(n):
        variance = omega + alpha * previous * previous + beta * variance
        previous = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        out.append(previous * 10_000.0)  # USD P&L scale
    return out


def test_the_garch_fit_row_carries_every_spec_13_diagnostic():
    """§13's gap: the snapshot DISCARDED the GarchFit, so omega/alpha/beta,
    persistence, half-life and Ljung-Box existed for one function call and
    were never written down."""
    fit = fit_garch(_garch_series(400))
    assert fit.is_available, fit.reason

    row = rs.garch_fit_metric_row(
        fit, snapshot_id=1, as_of=datetime(2026, 8, 19, tzinfo=timezone.utc)
    )
    assert row is not None
    assert row.metric == "COND_VOL_FIT"
    assert row.model_name == "garch11"
    assert row.confidence is None      # a vol fit carries no tail probability
    assert row.value_pct_nav is None   # USD/day, not a loss — never ÷ NAV
    assert row.sample_size == fit.n

    for key in (
        "omega", "alpha", "beta", "persistence", "half_life",
        "ljung_box_q_sq", "ljung_box_p", "converged",
    ):
        assert key in row.diagnostics, key
    # The headline numbers are the FIT's, not recomputed.
    assert row.diagnostics["persistence"] == pytest.approx(fit.persistence)
    assert row.diagnostics["omega"] == pytest.approx(fit.params.omega)
    assert row.params["tier"] == "TIER_2"


def test_no_fit_row_is_written_when_garch_produced_no_parameters():
    """An UNAVAILABLE/FAILED fit means EWMA is driving the conditional
    views. A COND_VOL_FIT row would then claim a GARCH fit was in force."""
    assert rs.garch_fit_metric_row(None, snapshot_id=1, as_of=None) is None

    too_short = fit_garch([0.01, -0.02, 0.015] * 5)
    assert not too_short.is_available
    assert rs.garch_fit_metric_row(too_short, snapshot_id=1, as_of=None) is None


async def test_a_live_garch_build_persists_exactly_one_cond_vol_fit_row(client):
    """END-TO-END, not just the writer. A GARCH-like 600-bar path clears the
    250-observation minimum, so `conditional_source` really is GARCH and the
    §13 row is really written by `_persist` — the seam the snapshot used to
    drop the fit at.
    """
    rng = random.Random(3)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    variance = omega / (1.0 - alpha - beta)
    previous = 0.0
    closes = [100.0]
    for _ in range(600):
        variance = omega + alpha * previous * previous + beta * variance
        previous = math.sqrt(variance) * rng.gauss(0.0, 1.0)
        closes.append(max(closes[-1] * (1.0 + previous), 1.0))

    async with SessionLocal() as session:
        start = date(2024, 1, 1)
        for i, close in enumerate(closes):
            session.add(
                StockBarDaily(
                    ticker="AAPL", ts=start + timedelta(days=i),
                    open=close, high=close, low=close, close=close,
                    volume=5_000_000,
                )
            )
        session.add(
            Position(
                ticker="AAPL", instrument="LONG_STOCK", quantity=100,
                avg_price=100.0, max_loss=1_000.0, status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    result = await build()
    if result.api["conditional_source"]["source"] != "GARCH":
        pytest.skip(
            "this seeded path did not produce an ACTIVE fit; the writer and "
            "the EWMA-branch suppression are covered by the unit tests above"
        )

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(RiskMetricRow).where(
                    RiskMetricRow.snapshot_id == result.row_id,
                    RiskMetricRow.metric == "COND_VOL_FIT",
                )
            )
        ).scalars().all()

    assert len(rows) == 1, "exactly one fit row per snapshot"
    row = rows[0]
    assert row.model_name == "garch11"
    assert row.value is not None and row.value > 0      # sigma_{t+1}, USD/day
    assert row.sample_size >= 250
    for key in ("omega", "alpha", "beta", "persistence", "half_life",
                "ljung_box_q_sq", "ljung_box_p", "converged"):
        assert key in row.diagnostics, key
    # alpha + beta == persistence, hand-checkable from the row alone.
    assert row.diagnostics["alpha"] + row.diagnostics["beta"] == pytest.approx(
        row.diagnostics["persistence"]
    )


def test_the_fit_row_reports_the_fits_own_health_not_a_fabricated_active():
    """A DEGRADED fit (near-integrated, or Ljung-Box rejecting) says so on
    the row rather than borrowing the confidence of the rows beside it."""
    fit = fit_garch(_garch_series(400))
    degraded = dataclasses.replace(
        fit, health=ModelHealth.DEGRADED, reason="persistence=0.999 (near-integrated)"
    )
    row = rs.garch_fit_metric_row(degraded, snapshot_id=1, as_of=None)
    assert row.health == "DEGRADED"
    assert "near-integrated" in row.reason


# ---------------------------------------------------------------------------
# §55 — is_stale finally has a consumer
# ---------------------------------------------------------------------------


def _caps() -> list[QuantityCap]:
    return [
        QuantityCap(
            code="PORTFOLIO_ES_LIMIT", layer="STATISTICAL", cap_qty=1,
            sentence="hypothetical", measured={},
        )
    ]


def test_a_stale_snapshot_suppresses_the_shadow_caps_with_a_reason():
    """§55's gap: the mechanism was complete, tested to the boundary, and
    consumed by NOTHING but the serialiser. Here it finally decides."""
    fresh = shadow_verdict(10, _caps())
    assert fresh.hypothetical_decision == "APPROVE_WITH_RESIZE"
    assert fresh.hypothetical_quantity == 1
    assert fresh.binding == ("PORTFOLIO_ES_LIMIT",)
    assert fresh.reason is None

    stale = shadow_verdict(10, _caps(), stale=True)
    assert stale.hypothetical_decision == DECISION_UNAVAILABLE_STALE
    # Falls back to what ACTUALLY happened: Tier 0's approved quantity.
    assert stale.hypothetical_quantity == 10
    # Nothing bound...
    assert stale.binding == ()
    # ...but the caps are still carried: they were computed, and hiding them
    # would lose the evidence the shadow window is accumulating.
    assert len(stale.caps) == 1
    assert stale.reason and "stale" in stale.reason.lower()


def test_the_stale_verdict_carries_the_callers_reason_when_given():
    stale = shadow_verdict(5, _caps(), stale=True, stale_reason="age 90000s > ttl 86400s")
    assert stale.reason == "age 90000s > ttl 86400s"


def test_unavailable_stale_is_not_one_of_the_tier_0_decision_words():
    """A stale view has not decided to approve, resize OR reject — it has
    failed to answer, and saying APPROVE would be the fail-OPEN this
    vocabulary exists to make visible."""
    assert DECISION_UNAVAILABLE_STALE not in {
        DECISION_APPROVE, "APPROVE_WITH_RESIZE", "REJECT",
    }
    # ...and the dataclass accepts it (it would raise on an unknown word).
    assert shadow_verdict(1, (), stale=True).hypothetical_decision == (
        DECISION_UNAVAILABLE_STALE
    )


def test_the_snapshot_staleness_boundary_is_the_one_is_stale_defines():
    """Exactly TTL old is NOT stale; TTL + 1s is. The shadow rule inherits
    that boundary rather than inventing a second one."""
    from libs.trading_core.risk.snapshot import (
        STALENESS_KIND_STATISTICAL,
        PortfolioRiskSnapshot,
        TtlPolicy,
    )

    as_of = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    snapshot = PortfolioRiskSnapshot(
        as_of=as_of, nav=100_000.0, cash=50_000.0, cash_reserved=0.0,
        gross_exposure=50_000.0, delta_adjusted_exposure=50_000.0,
        heat_pct=0.01, heat_state="NORMAL",
    )
    ttl = TtlPolicy().seconds_for(STALENESS_KIND_STATISTICAL)
    assert not snapshot.is_stale(as_of + timedelta(seconds=ttl))
    assert snapshot.is_stale(as_of + timedelta(seconds=ttl + 1))


async def test_tier_0_is_byte_identical_whether_or_not_the_stale_rule_fires(client):
    """THE SABOTAGE PIN (§70). The §55 rule is a SHADOW-only behaviour
    change, so a Tier 0 decision must be identical with the staleness rule
    firing and not firing.

    `assess()` is never passed `extra_caps` at either production call site
    (AST-pinned elsewhere), so suppressing shadow caps removes a
    HYPOTHETICAL, never a control. This test proves it end-to-end by
    forcing every snapshot to report itself stale and comparing the whole
    Tier 0 half of the preview.
    """
    from libs.trading_core.risk.snapshot import PortfolioRiskSnapshot
    from tests.test_orders_shadow_c import (
        BULL_TICKER,
        _book_including_candidate,
        authorize,
        preview,
    )

    await _book_including_candidate()
    await authorize(client, BULL_TICKER)

    def _tier0(body: dict) -> dict:
        risk = body["risk"]
        return {
            "decision": body.get("decision"),
            "approved_quantity": risk.get("approved_quantity"),
            "binding_constraints": risk.get("binding_constraints"),
            "gates": risk.get("gates"),
            "reason": risk.get("reason"),
            "comparison_tier0": risk["comparison"]["tier0_rows"],
        }

    healthy = _tier0(await preview(client, BULL_TICKER))

    real = PortfolioRiskSnapshot.is_stale
    try:
        PortfolioRiskSnapshot.is_stale = lambda self, now, kind="statistical": True
        sabotaged_body = await preview(client, BULL_TICKER)
    finally:
        PortfolioRiskSnapshot.is_stale = real

    assert _tier0(sabotaged_body) == healthy, (
        "the §55 shadow staleness rule changed a Tier 0 number"
    )
    # ...and the SHADOW half did change, which is the point of the rule.
    hypothetical = sabotaged_body["risk"]["shadow_statistical"]["hypothetical"]
    assert hypothetical["decision"] == DECISION_UNAVAILABLE_STALE
    assert hypothetical["stale"] is True
    assert hypothetical["binding"] == []
    assert hypothetical["reason"]
