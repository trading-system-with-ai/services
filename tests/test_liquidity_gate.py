"""Underlying LIQUIDITY gate in REPORT mode (risk-engine audit §7.3 "Liquidity
gate" row, §10 Phase B0; spec §2/§5 hard limit, §70 shadow mode).

Pure-function tests hand-check ADV20 / participation / spread arithmetic and
the verdict rules (UNAVAILABLE only when nothing is measurable; WOULD_FAIL
when any MEASURED component breaches); API tests pin that gate 7 now
reports PASS with "REPORT mode" numbers in its detail on the standard preview
fixture, that the verdict NEVER changes the gate status, and that the
RISK_DECISION audit carries ``shadow.liquidity`` in the same event.
"""
import dataclasses
import math

import pytest

from apps.gateway import market_stream
from libs.trading_core.risk import (
    LiquidityLimits,
    LiquidityReport,
    evaluate_underlying_liquidity,
)
from libs.trading_core.risk.liquidity import (
    average_daily_volume,
    liquidity_report_detail,
    quote_spread_fraction,
)

from .test_order_preview import (
    BULL_TICKER,
    LIQUIDITY_REPORT_PREFIX,
    authorize,
    get_single_risk_decision_event,
    preview,
)

# 20 volumes whose mean is hand-computable: 19 × 1,000,000 + 1 × 1,200,000
# = 20,200,000 / 20 = 1,010,000 sh.
TWENTY = [1_000_000.0] * 19 + [1_200_000.0]


# --------------------------------------------------------------------------
# LiquidityLimits — research defaults, validation
# --------------------------------------------------------------------------
def test_limits_research_defaults_and_frozen():
    lim = LiquidityLimits()
    assert (lim.min_adv20_shares, lim.max_order_pct_adv20) == (100_000, 0.01)
    assert (lim.max_quote_spread_pct, lim.adv_window) == (0.005, 20)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lim.min_adv20_shares = 1  # type: ignore[misc]
    # The docstring is the contract: research defaults, unvalidated, REPORT.
    doc = LiquidityLimits.__doc__
    assert "RESEARCH DEFAULTS" in doc and "UNVALIDATED" in doc
    assert "REPORT mode" in doc


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_adv20_shares": -1},
        {"max_order_pct_adv20": 0.0},
        {"max_order_pct_adv20": 1.5},
        {"max_quote_spread_pct": 0.0},
        {"max_quote_spread_pct": 2.0},
        {"adv_window": 0},
    ],
)
def test_limits_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        LiquidityLimits(**kwargs)


# --------------------------------------------------------------------------
# Component arithmetic, hand-checked
# --------------------------------------------------------------------------
def test_adv20_is_mean_of_last_window_only():
    # 25 volumes: five old ones at 10 sh must NOT enter the 20-day mean.
    vols = [10.0] * 5 + TWENTY
    assert average_daily_volume(vols, 20) == pytest.approx(1_010_000.0)
    # Fewer than the window -> None (never a shorter-window substitute).
    assert average_daily_volume(TWENTY[:19], 20) is None
    assert average_daily_volume([], 20) is None
    # A non-finite / negative volume makes the window unmeasurable.
    assert average_daily_volume([1.0] * 19 + [math.nan], 20) is None
    assert average_daily_volume([1.0] * 19 + [-5.0], 20) is None


def test_quote_spread_fraction_hand_checked():
    # bid 99.95 / ask 100.05: mid 100.00, spread 0.10 -> 0.001 (0.1 %).
    assert quote_spread_fraction(99.95, 100.05) == pytest.approx(0.001)
    # bid 10.00 / ask 10.10: mid 10.05, spread 0.10 -> 0.0099502...
    assert quote_spread_fraction(10.0, 10.1) == pytest.approx(0.10 / 10.05)
    # Locked market is a measured 0.0; crossed / one-sided / non-positive
    # are unmeasurable (None), never fabricated.
    assert quote_spread_fraction(50.0, 50.0) == 0.0
    assert quote_spread_fraction(50.10, 50.0) is None
    assert quote_spread_fraction(None, 50.0) is None
    assert quote_spread_fraction(50.0, None) is None
    assert quote_spread_fraction(0.0, 50.0) is None
    assert quote_spread_fraction(50.0, -1.0) is None
    assert quote_spread_fraction(math.inf, 50.0) is None


def test_full_report_all_components_pass():
    # ADV20 1,010,000; order 5,000 sh -> 5,000 / 1,010,000 = 0.495049...%
    # (<= 1 %); spread 0.1 % (<= 0.5 %) -> PASS.
    r = evaluate_underlying_liquidity(TWENTY, 5_000, 99.95, 100.05)
    assert isinstance(r, LiquidityReport)
    assert r.adv20 == pytest.approx(1_010_000.0)
    assert r.order_pct_adv20 == pytest.approx(5_000 / 1_010_000)
    assert r.quote_spread_pct == pytest.approx(0.001)
    assert r.verdict == "PASS"
    assert r.mode == "REPORT"
    assert r.reasons == (
        "ADV20 1,010,000 sh >= 100,000 minimum",
        "order 5,000 sh = 0.50% of ADV20 <= 1.00% maximum",
        "quote spread 0.100% (bid 99.95 / ask 100.05) <= 0.50% maximum",
    )
    detail = liquidity_report_detail(r, LiquidityLimits())
    assert detail.startswith(
        "underlying liquidity (REPORT mode, research limits): ADV20 1,010,000 sh; "
        "order 0.50% of ADV20; quote spread 0.100% — would PASS: "
    )


# --------------------------------------------------------------------------
# UNAVAILABLE only when NOTHING is measurable
# --------------------------------------------------------------------------
def test_unavailable_when_fewer_than_20_volumes_and_no_quote():
    r = evaluate_underlying_liquidity(TWENTY[:19], 5_000, None, None)
    assert (r.adv20, r.order_pct_adv20, r.quote_spread_pct) == (None, None, None)
    assert r.verdict == "UNAVAILABLE"
    assert r.reasons == (
        "ADV20 unmeasured: 19 stored volume(s), need 20",
        "order participation unmeasured: 5,000 sh but no ADV20",
        "quote spread unmeasured: no two-sided stock quote",
    )
    detail = liquidity_report_detail(r, LiquidityLimits())
    assert "ADV20 n/a; order n/a; quote spread n/a — verdict UNAVAILABLE" in detail
    # Nothing at all: same honest verdict.
    assert evaluate_underlying_liquidity([], None, None, None).verdict == "UNAVAILABLE"


def test_partial_measurement_is_not_unavailable():
    # Only the spread is measurable (no bars): a measured 0.1 % passes.
    r = evaluate_underlying_liquidity([], None, 99.95, 100.05)
    assert r.adv20 is None and r.order_pct_adv20 is None
    assert r.quote_spread_pct == pytest.approx(0.001)
    assert r.verdict == "PASS"
    # Only ADV20 measurable (order unknown, no quote): passes on ADV alone.
    r = evaluate_underlying_liquidity(TWENTY, None, None, None)
    assert r.verdict == "PASS"
    assert "order participation unmeasured: order size unknown" in r.reasons


# --------------------------------------------------------------------------
# WOULD_FAIL per component
# --------------------------------------------------------------------------
def test_would_fail_on_adv20_below_floor():
    # 20 × 42,000 -> ADV20 42,000 < 100,000.
    r = evaluate_underlying_liquidity([42_000.0] * 20, None, None, None)
    assert r.adv20 == pytest.approx(42_000.0)
    assert r.verdict == "WOULD_FAIL"
    assert r.reasons[0] == "ADV20 42,000 sh < 100,000 minimum"
    detail = liquidity_report_detail(r, LiquidityLimits())
    assert "— would FAIL: ADV20 42,000 sh < 100,000 minimum" in detail


def test_would_fail_on_participation_above_max():
    # ADV20 1,010,000; order 20,200 sh = exactly 2.00 % > 1 %.
    r = evaluate_underlying_liquidity(TWENTY, 20_200, None, None)
    assert r.order_pct_adv20 == pytest.approx(0.02)
    assert r.verdict == "WOULD_FAIL"
    assert "order 20,200 sh = 2.00% of ADV20 > 1.00% maximum" in r.reasons
    # Exactly AT the limit passes (<=): 10,100 / 1,010,000 = 1.00 %.
    r = evaluate_underlying_liquidity(TWENTY, 10_100, None, None)
    assert r.order_pct_adv20 == pytest.approx(0.01)
    assert r.verdict == "PASS"


def test_would_fail_on_wide_spread():
    # bid 10.00 / ask 10.10 -> 0.10 / 10.05 = 0.995 % > 0.5 %.
    r = evaluate_underlying_liquidity(TWENTY, 100, 10.0, 10.1)
    assert r.quote_spread_pct == pytest.approx(0.10 / 10.05)
    assert r.verdict == "WOULD_FAIL"
    assert any(s.startswith("quote spread 0.995%") and "> 0.50% maximum" in s
               for s in r.reasons)


def test_crossed_quote_is_unmeasured_not_a_breach():
    r = evaluate_underlying_liquidity(TWENTY, 100, 10.10, 10.0)
    assert r.quote_spread_pct is None
    assert r.verdict == "PASS"  # ADV + participation measured and fine
    assert any("unusable NBBO" in s for s in r.reasons)


def test_zero_adv_reports_breach_and_unmeasured_participation():
    r = evaluate_underlying_liquidity([0.0] * 20, 100, None, None)
    assert r.adv20 == 0.0
    assert r.order_pct_adv20 is None
    assert r.verdict == "WOULD_FAIL"
    assert "order participation unmeasured: ADV20 is 0" in r.reasons


def test_custom_limits_are_parameters_not_truths():
    lim = LiquidityLimits(min_adv20_shares=10, max_order_pct_adv20=0.5,
                          max_quote_spread_pct=0.02, adv_window=5)
    # window 5: mean of the LAST five of [1,2,3,4,5,6,7,8,9,10] = 8.0
    r = evaluate_underlying_liquidity(list(range(1, 11)), 4, 10.0, 10.1, lim)
    assert r.adv20 == pytest.approx(8.0)
    assert r.order_pct_adv20 == pytest.approx(0.5)  # 4 / 8 == max -> passes
    assert r.verdict == "WOULD_FAIL"  # 8.0 < 10 minimum
    assert r.reasons[0] == "ADV5 8 sh < 10 minimum"
    assert liquidity_report_detail(r, lim).startswith(
        "underlying liquidity (REPORT mode, research limits): ADV5 8 sh; "
        "order 50.00% of ADV5; quote spread 0.995% — would FAIL: "
    )


# --------------------------------------------------------------------------
# API: gate 7 in REPORT mode on the standard preview fixture
# --------------------------------------------------------------------------
async def test_preview_liquidity_gate_reports_pass_and_audits_shadow(client):
    """The stub provider stores 600 daily bars (volume centre 1,000,000 sh),
    so ADV20 is measured; no quantity is requested and no streamed NBBO
    exists in tests -> those components are honestly n/a; the gate PASSes
    with the numbers, and the RISK_DECISION audit carries shadow.liquidity
    (asdict of the same report) in the same event."""
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)
    gates = {g["name"]: g for g in body["gates"]}
    liq = gates["LIQUIDITY"]
    assert liq["status"] == "PASS"
    assert liq["detail"].startswith(LIQUIDITY_REPORT_PREFIX)
    assert "ADV20 n/a" not in liq["detail"]  # measured off the stored bars
    assert "quote spread n/a" in liq["detail"]
    assert "(no fresh streamed stock NBBO — spread unmeasured)" in liq["detail"]
    assert "would " in liq["detail"]
    # GATE_ORDER (Phase D 2026-08-20): SQUEEZE_RISK sits between INSTRUMENT
    # and LIQUIDITY; LIQUIDITY still precedes CONTRACT_SELECTION.
    names = [g["name"] for g in body["gates"]]
    assert names.index("INSTRUMENT") + 1 == names.index("SQUEEZE_RISK")
    assert names.index("SQUEEZE_RISK") + 1 == names.index("LIQUIDITY")
    assert names.index("LIQUIDITY") + 1 == names.index("CONTRACT_SELECTION")

    event = await get_single_risk_decision_event(client, BULL_TICKER)
    shadow = event["details"]["shadow"]["liquidity"]
    assert shadow["mode"] == "REPORT"
    assert shadow["verdict"] in {"PASS", "WOULD_FAIL"}
    assert shadow["adv20"] > 0
    assert shadow["quote_spread_pct"] is None
    assert shadow["order_shares"] is None  # no quantity requested
    assert shadow["order_pct_adv20"] is None
    assert isinstance(shadow["reasons"], list) and shadow["reasons"]
    # The gate detail and the shadow verdict tell the same story.
    assert f"would {'PASS' if shadow['verdict'] == 'PASS' else 'FAIL'}" in liq["detail"]
    # Stock candidate: participation is re-measured at the APPROVED size when
    # the risk engine approved (the order that would actually be sent).
    risk = body["risk"]
    if risk is not None and risk["decision"] != "REJECT":
        at_qty = shadow["at_approved_quantity"]
        assert at_qty["order_shares"] == risk["approved_quantity"]
        assert at_qty["order_pct_adv20"] == pytest.approx(
            risk["approved_quantity"] / shadow["adv20"]
        )
        assert at_qty["mode"] == "REPORT"


async def test_requested_quantity_feeds_participation_in_report_mode(client):
    """A requested share count is measured against ADV20 at gate 7; the
    hypothetical verdict does NOT change the gate status (REPORT mode) even
    when the participation would breach."""
    await authorize(client, BULL_TICKER)
    huge = 50_000_000  # >> 1 % of a ~1,000,000 sh ADV20
    body = await preview(client, BULL_TICKER, quantity=huge)
    liq = {g["name"]: g for g in body["gates"]}["LIQUIDITY"]
    assert liq["status"] == "PASS"  # never a veto in this phase
    assert "would FAIL" in liq["detail"]
    assert "of ADV20 > 1.00% maximum" in liq["detail"]
    event = await get_single_risk_decision_event(client, BULL_TICKER)
    shadow = event["details"]["shadow"]["liquidity"]
    assert shadow["verdict"] == "WOULD_FAIL"
    assert shadow["order_shares"] == huge
    assert shadow["order_pct_adv20"] == pytest.approx(huge / shadow["adv20"])
    # The chain was not vetoed by LIQUIDITY.
    assert event["details"]["veto_gate"] != "LIQUIDITY"


async def test_fresh_streamed_nbbo_feeds_the_spread_component(client):
    """When the in-process stream cache holds a FRESH quote for the symbol,
    gate 7 measures the spread from it (no provider call); a wide spread
    is reported as would-FAIL and still never vetoes."""
    await authorize(client, BULL_TICKER)
    market_stream.CACHE.apply(
        [{"T": "q", "S": BULL_TICKER, "bp": 100.0, "ap": 101.0, "bs": 1, "as": 1}]
    )
    try:
        body = await preview(client, BULL_TICKER)
    finally:
        market_stream.CACHE.clear()
    liq = {g["name"]: g for g in body["gates"]}["LIQUIDITY"]
    assert liq["status"] == "PASS"
    # 1.00 / 100.5 = 0.995 % > 0.5 %.
    assert "quote spread 0.995%" in liq["detail"]
    assert "would FAIL" in liq["detail"]
    assert "(no fresh streamed stock NBBO" not in liq["detail"]
    event = await get_single_risk_decision_event(client, BULL_TICKER)
    shadow = event["details"]["shadow"]["liquidity"]
    assert shadow["quote_spread_pct"] == pytest.approx(1.0 / 100.5)
    assert shadow["verdict"] == "WOULD_FAIL"


async def test_vetoed_chain_skips_liquidity_and_records_null_shadow(client):
    """Earlier veto -> LIQUIDITY is SKIPPED with the exact skip text and
    shadow.liquidity is null (nothing was measured — honest null)."""
    from .test_order_preview import SKIP_EARLIER_FAIL

    # A symbol with no stored bars and a provider fault is hard to force
    # deterministically; instead pick a ticker whose stub chain vetoes
    # before gate 7 by scanning the deterministic stub universe.
    for ticker in ("META", "AMZN", "TSLA", "MSFT", "AAPL", "AMD", "NFLX"):
        await authorize(client, ticker)
        body = await preview(client, ticker)
        gates = {g["name"]: g for g in body["gates"]}
        first_fail = next(
            (g["name"] for g in body["gates"] if g["status"] == "FAIL"), None
        )
        if first_fail in ("REGIME", "DIRECTIONAL_SIGNAL", "VOLATILITY", "INSTRUMENT"):
            assert gates["LIQUIDITY"]["status"] == "SKIPPED"
            assert gates["LIQUIDITY"]["detail"] == SKIP_EARLIER_FAIL
            event = await get_single_risk_decision_event(client, ticker)
            # `shadow` grows additively as SHADOW layers land (Phase B added
            # `statistical`), so this asserts THIS gate's slot, not the whole
            # dict: nothing was measured before gate 7, so it is null.
            assert event["details"]["shadow"]["liquidity"] is None
            # The chain vetoed before RISK_APPROVAL, so the Phase B snapshot
            # never ran either — likewise an honest null.
            assert event["details"]["shadow"]["statistical"] is None
            return
    pytest.skip("no stub ticker vetoed before gate 7 in this universe")


# ---------------------------------------------------------------------------
# QA follow-ups (Phase B0 verification findings, 2026-08-17)
# ---------------------------------------------------------------------------


def test_none_volume_entries_make_adv_unmeasurable_not_a_typeerror():
    # A None inside the window is an unmeasurable component (honest null),
    # never a TypeError leaking out of the pure function.
    assert average_daily_volume([1e6] * 19 + [None], 20) is None
    report = evaluate_underlying_liquidity([1e6] * 19 + [None], 100, None, None)
    assert report.adv20 is None
    assert report.verdict == "UNAVAILABLE"


def test_nan_adv_floor_is_rejected_as_a_parameter():
    with pytest.raises(ValueError):
        LiquidityLimits(min_adv20_shares=float("nan"))
    with pytest.raises(ValueError):
        LiquidityLimits(min_adv20_shares=float("inf"))


def test_partial_measurement_is_flagged_on_the_report():
    # Only the spread is measurable (no volumes): PASS on ONE component,
    # flagged partial so a future veto promotion can fail closed on it.
    report = evaluate_underlying_liquidity([], 100, 10.0, 10.05)
    assert report.verdict == "PASS"
    assert report.partial is True
    assert any("partial measurement: 1 of 3" in r for r in report.reasons)
    # All three measured -> not partial.
    full = evaluate_underlying_liquidity([1_000_000.0] * 20, 100, 10.0, 10.05)
    assert full.partial is False
    # Nothing measured -> UNAVAILABLE and not partial.
    none = evaluate_underlying_liquidity([], None, None, None)
    assert none.verdict == "UNAVAILABLE" and none.partial is False
    # asdict carries the flag for the RISK_DECISION shadow block.
    assert "partial" in dataclasses.asdict(report)
