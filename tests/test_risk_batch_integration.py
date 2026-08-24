"""Cross-cutting integration guards for the compliance backlog batch.

(a) The new SHADOW/RESEARCH layers cannot move Tier 0.

Every surface added in this batch — sizing-v2 shadow, the §11 factor
diagnostic and the §65 telemetry — is monkeypatched to RAISE, and the
pre-trade decision must come back BYTE-IDENTICAL. This is the property the
whole batch rests on: a statistical layer that can change a decision when it
BREAKS could also change one when it merely disagrees.
"""
import json

from libs.trading_core.risk import pretrade as pretrade_lib
from apps.gateway.routers import orders as orders_router
from apps.gateway import risk_snapshot as rs
from apps.gateway import risk_validation as rv
from libs.trading_core.risk.models import factor as factor_mod

from .test_order_preview import authorize, preview
from apps.gateway.execution import gate_chain


def _boom(*a, **k):
    raise RuntimeError("sabotaged by the integrator probe")


DECISION_KEYS = ("decision", "approved_quantity", "reason_codes")


def _tier0(body):
    risk = body["risk"]
    return json.dumps({k: risk.get(k) for k in DECISION_KEYS}, sort_keys=True)


async def test_preview_is_byte_identical_when_every_new_layer_raises(
    client, monkeypatch
):
    await authorize(client, "AAPL")

    clean = _tier0(await preview(client, "AAPL", quantity=10))

    # Sabotage all three new surfaces at once.
    monkeypatch.setattr(gate_chain, "sizing_v2_shadow", _boom)
    monkeypatch.setattr(pretrade_lib, "sizing_v2_shadow", _boom)
    monkeypatch.setattr(factor_mod, "factor_risk_share", _boom)
    monkeypatch.setattr(rs, "factor_risk_share", _boom)
    monkeypatch.setattr(rs, "_set_model_health_gauge", _boom)
    monkeypatch.setattr(rs, "_inc_garch_fit_failures", _boom)
    monkeypatch.setattr(rv, "set_model_health_gauge", _boom)
    for name in (
        "RISK_RESIZE_COUNT",
        "RISK_REJECT_COUNT",
        "STRESS_LIMIT_BLOCKS",
    ):
        counter = getattr(orders_router, name)
        monkeypatch.setattr(counter, "inc", _boom)
    for name in ("VAR_EXCEEDANCES_TOTAL", "ES_EXCEEDANCES_TOTAL",
                 "GARCH_FIT_FAILURES_TOTAL"):
        if hasattr(rv, name):
            monkeypatch.setattr(getattr(rv, name), "inc", _boom)

    # VACUITY GUARD: prove the sabotage is actually reached, so a passing
    # byte-identity assertion cannot be an artifact of never calling in.
    # Patch the DEFINITION module. routers.orders re-exports this name, and a
    # patch there would rebind only the re-export while the caller in
    # gate_chain kept the original — which this guard's own assertion would
    # then catch as "the sabotage was never invoked".
    hit = {"n": 0}
    def _count(*a, **k):
        hit["n"] += 1
        raise RuntimeError("sabotaged")
    monkeypatch.setattr(gate_chain, "sizing_v2_shadow", _count)

    sabotaged = _tier0(await preview(client, "AAPL", quantity=10))
    assert hit["n"] > 0, "the sabotage was never invoked - the probe proves nothing"
    assert sabotaged == clean, "a broken SHADOW layer moved a Tier 0 decision"


async def test_risk_view_survives_a_broken_factor_diagnostic(client, monkeypatch):
    """The read view must degrade to an honest null, never 5xx."""
    from .test_risk_snapshot_builder import seed_stock_position

    await seed_stock_position("AAPL", bars=200)
    monkeypatch.setattr(rs, "factor_risk_share", _boom)

    r = await client.get("/api/portfolio/risk")
    assert r.status_code == 200
    factor = r.json()["statistical"]["factor"]
    assert factor["portfolio_beta"] is None
    assert factor["health"] == "UNAVAILABLE"
    assert factor["reason"]


# ---------------------------------------------------------------------------
# (c) §65: all nine named metrics are actually exposed
# ---------------------------------------------------------------------------

#: The nine instruments spec §65 names, plus the two that predate it. Four
#: existed before this batch (age, latency, builds, failures); the other
#: seven were the audit's "verified absent" list — no alertable time series
#: for how often a shadow gate would have bound.
SECTION_65_METRICS = [
    "risk_snapshot_age_seconds",
    "risk_model_latency_seconds",
    "risk_snapshot_builds_total",
    "risk_snapshot_failures_total",
    "var_exceedances",
    "es_exceedances",
    "garch_fit_failures",
    "stress_limit_blocks",
    "risk_resize_count",
    "risk_reject_count",
    "model_health_state",
]


async def test_every_section_65_metric_reaches_the_scrape(client):
    """A counter with no sample line is not an alertable time series.

    Exercises all three call sites — a seeded snapshot build, a validation
    run and a preview — because several of these instruments only render a
    sample line AFTER their first increment.
    """
    from .test_risk_snapshot_builder import seed_stock_position

    await seed_stock_position("AAPL", bars=520)
    await client.get("/api/portfolio/risk")
    assert (await client.post("/api/risk/validation/run", json={})).status_code == 200
    await authorize(client, "AAPL")
    await preview(client, "AAPL", quantity=10)

    text = (await client.get("/metrics")).text
    missing = [n for n in SECTION_65_METRICS if n not in text]
    assert not missing, f"§65 instruments absent from /metrics: {missing}"
