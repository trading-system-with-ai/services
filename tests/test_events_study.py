"""§86 predictiveness measurement harness — GET /api/events/study and the pure
rank-correlation library (event spec §85, §86, §87, §92; audit §7.2, §7.4;
Phase L unit U2).

WHAT THESE TESTS DEFEND, in the order they appear:

1. **Spearman's rho is arithmetically right, checked BY HAND.** Every
   correlation assertion below is against a value a reader can recompute on
   paper from the four-to-six numbers in the test, not against a value
   captured from a previous run. A measurement harness that pins itself to
   its own output cannot detect having become wrong.
2. **Ties take the AVERAGE rank**, and the proof is order-independence: the
   SAME data shuffled must produce the SAME rho. With first-seen tie-breaking
   it would not, and the report would be a function of the query order.
3. **``None`` is never imputed** (§44 rule 18, §85). A missing feature drops
   the PAIR, and ``n`` reports the pairs actually used — so ``coverage_pct``
   and ``n`` are different numbers with different meanings and both are
   checked.
4. **NOT_MEANINGFUL fires below n = 12** and ``min_n`` can only RAISE the bar.
   Asking for a lower floor is a 422 at the route: the one thing that knob
   must never do is make a four-event correlation quotable.
5. **The endpoint is DB-only** (audit §7.2 rule 1) — asserted by patching the
   market-data seam the study's neighbours use to EXPLODE. A study that
   reached a vendor would raise rather than quietly succeeding, so this cannot
   rot into a no-op.
6. **Features come from the EARLIEST stored bundle, never a fresh assembly**
   (§96). A later re-run whose bundle was assembled after the print carries a
   different, contaminated run-up; the test stores both and asserts the early
   one is what the study read.
7. **LIVE option snapshots never feed the study** (§85). A
   LIVE_CHAIN_SNAPSHOT metrics row planted with an unmistakable value must be
   absent from every feature vector.
8. **No p-value, no verdict, anywhere** (§92) — asserted by a recursive scan
   of the payload for the vocabulary of false certainty.
9. **Deterministic.** Two identical requests return byte-identical reports.

Uses the shared ``client`` fixture (conftest.py).
"""
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from apps.gateway import event_study as study_seam
from apps.gateway.db import (
    EventAnalysisRow,
    EventOptionMetricRow,
    EventRow,
    SessionLocal,
    StockBarDaily,
)
from libs.trading_core.events.event_study import (
    CAVEATS,
    FEATURE_NAMES,
    MIN_MEANINGFUL_N,
    NOT_MEANINGFUL,
    average_ranks,
    collect_feature_rows,
    feature_report,
    features_from_bundle,
    spearman_rank_corr,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

#: Every scenario is anchored on one fixed instant so the seeded bar dates,
#: the analysis ``as_of`` values and the reaction windows are reproducible
#: numbers a reader can check rather than values that drift overnight.
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ===========================================================================
# Part 1 — the pure library: rank correlation
# ===========================================================================


def test_average_ranks_shares_the_tied_block():
    """``[10, 20, 20, 30]`` -> ``[1, 2.5, 2.5, 4]`` — the textbook correction.

    Hand-checked: the two 20s occupy ranks 2 and 3, so both take 2.5. The
    total of the ranks stays ``n(n+1)/2 = 10`` either way, which is what makes
    the correction safe to apply blindly.
    """
    assert average_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]
    assert average_ranks([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]
    assert sum(average_ranks([3.0, 1.0, 2.0, 2.0])) == pytest.approx(10.0)


def test_spearman_perfect_monotone_is_one_even_when_non_linear():
    """rho = 1 for ANY increasing relationship — the reason it is rho and not r.

    ``y = x**3`` is wildly non-linear; Pearson's r on the raw values is 0.918,
    Spearman's on the ranks is exactly 1. Event returns are fat-tailed enough
    that this difference is the whole argument for the rank statistic.
    """
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 8.0, 27.0, 64.0, 125.0]
    result = spearman_rank_corr(xs, ys)
    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 5
    assert spearman_rank_corr(xs, list(reversed(ys)))["rho"] == pytest.approx(-1.0)


def test_spearman_hand_computed_value_with_no_ties():
    """rho = 1 - 6*Σd²/(n(n²-1)) is checkable by hand when nothing ties.

    x ranks 1,2,3,4,5; y = [2, 1, 4, 3, 5] ranks 2,1,4,3,5.
    d = [-1, +1, -1, +1, 0], Σd² = 4, n = 5 -> 1 - 24/120 = 0.8 exactly.

    The module does NOT use that shortcut (it is wrong with ties, and ties are
    the normal case here), so agreeing with it on a tie-free sample is a real
    cross-check of the general formula rather than a tautology.
    """
    result = spearman_rank_corr([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 5.0])
    assert result["rho"] == pytest.approx(0.8)
    assert result["n"] == 5


def test_spearman_hand_computed_value_with_ties():
    """A tied block, computed by hand through the general definition.

    x = [1, 2, 2, 4]  -> ranks [1, 2.5, 2.5, 4]
    y = [10, 20, 30, 40] -> ranks [1, 2, 3, 4]
    Deviations from mean 2.5: dx = [-1.5, 0, 0, 1.5], dy = [-1.5, -0.5, 0.5, 1.5]
    cov = 2.25 + 0 + 0 + 2.25 = 4.5; var_x = 4.5; var_y = 5.0
    rho = 4.5 / sqrt(4.5 * 5.0) = 4.5 / 4.74341649... = 0.94868329...

    The shortcut formula would give 1 - 6*0.5/60 = 0.95 here — close enough to
    look right in a glance and wrong in the third digit, which is exactly the
    kind of error a hand-computed assertion catches.
    """
    result = spearman_rank_corr([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert result["rho"] == pytest.approx(0.9486832980505138)
    assert result["n"] == 4


def test_spearman_is_independent_of_input_order():
    """The SAME pairs shuffled produce the SAME rho — the tie correction's job.

    This is the property that makes the report reproducible. With first-seen
    tie-breaking the two orderings below disagree, and the harness would return
    a different answer depending on how the events came back from the database.
    """
    pairs = [(1.0, 3.0), (2.0, 1.0), (2.0, 4.0), (2.0, 2.0), (5.0, 5.0)]
    forward = spearman_rank_corr([p[0] for p in pairs], [p[1] for p in pairs])
    shuffled = list(reversed(pairs))
    backward = spearman_rank_corr(
        [p[0] for p in shuffled], [p[1] for p in shuffled]
    )
    assert forward["rho"] == pytest.approx(backward["rho"])
    assert forward["n"] == backward["n"] == 5


def test_spearman_drops_unpaired_rows_and_never_imputes():
    """A ``None`` on either side drops the PAIR; nothing is filled in.

    The surviving three pairs are (1,1), (2,2), (3,3) -> rho = 1, n = 3. If the
    missing values had been imputed as 0.0 the ranks would shift and rho would
    fall; if the ROW had been dropped from the sample entirely, n would still
    be 3 but the coverage report elsewhere would lie about it.
    """
    xs = [1.0, 2.0, None, 3.0, 9.0]
    ys = [1.0, 2.0, 5.0, 3.0, None]
    result = spearman_rank_corr(xs, ys)
    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 3


def test_spearman_refuses_nan_inf_and_bools():
    """Non-finite and boolean inputs are dropped, not ranked.

    NaN is incomparable, so a single one leaking into ``sorted`` silently
    corrupts the ENTIRE ordering rather than just its own row — which is why it
    is filtered rather than tolerated. ``True`` is an ``int`` in Python and a
    feature column that accepted it would rank a flag beside a percentage.
    """
    assert spearman_rank_corr([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])["n"] == 2
    assert spearman_rank_corr([1.0, float("inf"), 3.0], [1.0, 2.0, 3.0])["n"] == 2
    assert spearman_rank_corr([1.0, True, 3.0], [1.0, 2.0, 3.0])["n"] == 2


def test_spearman_refuses_a_constant_column_rather_than_dividing_by_zero():
    """A feature with no variation has no ranking, and says so."""
    flat = spearman_rank_corr([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])
    assert flat["rho"] is None
    assert "constant" in flat["reason"]
    assert flat["n"] == 4  # the pairs are real; the statistic is not available

    out = spearman_rank_corr([1.0, 2.0, 3.0], [7.0, 7.0, 7.0])
    assert out["rho"] is None and "outcome is constant" in out["reason"]


def test_spearman_needs_three_pairs():
    """Two points are perfectly correlated by construction — refused."""
    result = spearman_rank_corr([1.0, 2.0], [5.0, 9.0])
    assert result["rho"] is None
    assert result["n"] == 2
    assert ">= 3" in result["reason"]


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        spearman_rank_corr([1.0, 2.0], [1.0, 2.0, 3.0])


# ===========================================================================
# Part 2 — the pure library: feature extraction and the report
# ===========================================================================


def _bundle(
    *,
    material: int | None = 3,
    run_up: float | None = 0.12,
    dist_high: float | None = -0.04,
    rv20: float | None = 0.31,
    momentum: float | None = 0.5,
    top_score: float | None = 0.62,
    median_move: float | None = 0.058,
) -> dict:
    """An evidence bundle carrying exactly the keys the study reads.

    Shaped like the real one (``apps/gateway/event_evidence.py``) rather than
    flattened: the extractor's job is to walk the REAL nesting, and a flat
    fixture would pass while the production path returned nothing.
    """
    return {
        "expectations_gap_inputs": {
            "fundamental_momentum": {"score": momentum},
            "expectation_proxies": {
                "material_developments": material,
                "run_up_since_previous_event": run_up,
                "distance_from_52w_high_pct": dist_high,
                "realized_vol_20d": rv20,
            },
        },
        "news": {"clusters": [{"score": top_score}, {"score": 0.11}]},
        "previous_market_reaction": {
            "history_table": {
                "summary": {"1D": {"last8": {"median_abs": median_move}}}
            }
        },
    }


def test_features_from_bundle_reads_the_real_nesting():
    features = features_from_bundle(_bundle())
    assert features["news_materiality"] == 3.0
    assert features["price_runup_pct"] == pytest.approx(0.12)
    assert features["distance_from_52w_high"] == pytest.approx(-0.04)
    assert features["realized_vol_20d"] == pytest.approx(0.31)
    assert features["fundamental_momentum_score"] == pytest.approx(0.5)
    assert features["news_evidence_score_max"] == pytest.approx(0.62)
    assert features["historical_median_move"] == pytest.approx(0.058)
    # Option-derived features are NOT in the bundle; they come from the
    # metrics row, and their absence here is honest rather than a zero.
    assert features["implied_move_pct"] is None
    assert features["iv_before"] is None


def test_features_from_bundle_returns_every_key_even_when_empty():
    """A missing key must be distinguishable from a measured zero (§44 r18).

    Returning a partial dict would make ``features.get(name)`` answer ``None``
    for both "not measured" and "absent from this build", and ``coverage_pct``
    would be uncomputable.
    """
    for empty in (None, {}, {"news": None}, "not a mapping"):
        features = features_from_bundle(empty)
        assert set(features) == set(FEATURE_NAMES)
        assert all(value is None for value in features.values())


def test_features_from_bundle_takes_the_max_evidence_score_not_the_mean():
    """One loud story must outrank a quiet window full of filler (§25)."""
    loud = features_from_bundle(_bundle(top_score=0.9))
    assert loud["news_evidence_score_max"] == pytest.approx(0.9)
    many_small = features_from_bundle(
        {"news": {"clusters": [{"score": 0.2}] * 20}}
    )
    assert many_small["news_evidence_score_max"] == pytest.approx(0.2)


def test_collect_feature_rows_keeps_zero_and_drops_nothing():
    """A run-up of exactly 0.0 is a MEASUREMENT and survives as one.

    The single most damaging imputation this table could make is treating a
    stock that did not move as a stock nobody measured, or the reverse: both
    would move a real observation into the middle of the ranking.
    """
    rows = collect_feature_rows(
        [
            {
                "event_id": 1,
                "event_key": "EARNINGS:AAA:2026-01-01",
                "ticker": "AAA",
                "event_date": "2026-01-01",
                "bundle": _bundle(run_up=0.0, material=0),
                "outcome_1d": 0.03,
                "outcome_5d": None,
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0].features["price_runup_pct"] == 0.0
    assert rows[0].features["news_materiality"] == 0.0
    assert rows[0].outcome_5d is None
    assert rows[0].event_date == date(2026, 1, 1)


def test_collect_feature_rows_keeps_outcomeless_rows_for_coverage():
    """An event with no outcome yet still counts in ``n_events``.

    "We have the feature for 5 events and the outcome for 2" is the single
    most important sentence a sample this small can say about itself, and a
    harness that filtered those rows would report 2/2 coverage and look far
    healthier than it is.
    """
    items = [
        {
            "event_id": i,
            "event_key": f"EARNINGS:AAA:2026-0{i}-01",
            "ticker": "AAA",
            "event_date": f"2026-0{i}-01",
            "bundle": _bundle(run_up=0.01 * i),
            "outcome_1d": 0.02 if i <= 2 else None,
            "outcome_5d": None,
        }
        for i in range(1, 6)
    ]
    report = feature_report(collect_feature_rows(items))
    assert report["n_events"] == 5
    assert report["outcome_coverage"]["outcome_1d"] == 2
    assert report["outcome_coverage"]["outcome_5d"] == 0
    assert report["features"]["price_runup_pct"]["coverage_pct"] == pytest.approx(100.0)
    # 5 features present, 2 usable pairings — two different numbers, both told.
    assert report["features"]["price_runup_pct"]["n_feature"] == 5
    assert report["features"]["price_runup_pct"]["rho_1d"]["n"] == 2


def test_option_metrics_feed_implied_move_and_iv_as_magnitudes():
    rows = collect_feature_rows(
        [
            {
                "event_id": 1,
                "event_key": "EARNINGS:AAA:2026-01-01",
                "ticker": "AAA",
                "event_date": date(2026, 1, 1),
                "bundle": _bundle(),
                "option_metrics": {"implied_move_pct": -0.07, "iv_before": 0.55},
                "outcome_1d": -0.03,
            }
        ]
    )
    # An implied move is a MAGNITUDE — a stored negative is taken absolutely
    # rather than ranked as "less move than zero".
    assert rows[0].features["implied_move_pct"] == pytest.approx(0.07)
    assert rows[0].features["iv_before"] == pytest.approx(0.55)


def _rows_with(values, outcomes, *, feature="price_runup_pct"):
    """``n`` rows carrying one feature and one 1D outcome each."""
    items = []
    for i, (value, outcome) in enumerate(zip(values, outcomes), start=1):
        bundle = _bundle(run_up=None, material=None, dist_high=None, rv20=None,
                         momentum=None, top_score=None, median_move=None)
        bundle["expectations_gap_inputs"]["expectation_proxies"][
            "run_up_since_previous_event"
        ] = value
        items.append(
            {
                "event_id": i,
                "event_key": f"EARNINGS:AAA:2026-01-{i:02d}",
                "ticker": "AAA",
                "event_date": date(2026, 1, i),
                "bundle": bundle,
                "outcome_1d": outcome,
            }
        )
    return collect_feature_rows(items)


def test_not_meaningful_flag_flips_exactly_at_the_floor():
    """n = 11 is flagged, n = 12 is not — the flag is a strict ``<`` on n.

    The threshold is a house convention (the caveats say so), but its BEHAVIOUR
    must be exact: an off-by-one here would either hide a flag on an
    eleven-event sample or fire one on a twelve-event sample, and both make the
    report's own honesty label untrustworthy.
    """
    eleven = _rows_with(
        [float(i) for i in range(11)], [float(i) * 0.01 for i in range(11)]
    )
    cell = feature_report(eleven)["features"]["price_runup_pct"]["rho_1d"]
    assert cell["n"] == 11
    assert cell["not_meaningful"] is True
    assert cell["flag"] == NOT_MEANINGFUL
    assert cell["rho"] == pytest.approx(1.0)  # still computed, just not quotable

    twelve = _rows_with(
        [float(i) for i in range(12)], [float(i) * 0.01 for i in range(12)]
    )
    cell = feature_report(twelve)["features"]["price_runup_pct"]["rho_1d"]
    assert cell["n"] == MIN_MEANINGFUL_N == 12
    assert cell["not_meaningful"] is False
    assert cell["flag"] is None


def test_report_computes_both_signed_and_absolute_and_names_the_primary():
    """Both correlations always exist; ``primary`` says which one is readable.

    Computing only the primary would leave a reader who disagrees with the
    classification unable to check; computing both and reporting the LARGER
    would be the garden-of-forking-paths the caveats warn about. So both are
    present and the choice is fixed in the spec table before any data is seen.
    """
    report = feature_report(_rows_with([1.0, 2.0, 3.0, 4.0], [0.1, -0.2, 0.3, -0.4]))
    runup = report["features"]["price_runup_pct"]
    implied = report["features"]["implied_move_pct"]
    assert runup["primary"] == "signed"
    assert implied["primary"] == "absolute"
    assert runup["rho_1d"] == runup["signed"]["rho_1d"]
    assert implied["rho_1d"] == implied["absolute"]["rho_1d"]
    # The signed and absolute views genuinely differ on this data: ranking
    # [0.1, -0.2, 0.3, -0.4] signed is not ranking [0.1, 0.2, 0.3, 0.4].
    assert runup["signed"]["rho_1d"]["rho"] != pytest.approx(
        runup["absolute"]["rho_1d"]["rho"]
    )
    assert runup["absolute"]["rho_1d"]["rho"] == pytest.approx(1.0)


def test_report_names_the_two_features_it_cannot_measure():
    """§86's list is seven; two of them have no data path and are NAMED.

    Silently omitting them would read as a claim that the section was fully
    covered — the exact overstatement §102 forbids.
    """
    report = feature_report([])
    names = {item["name"] for item in report["not_measurable"]}
    assert names == {"estimate_revision", "valuation_expansion"}
    for item in report["not_measurable"]:
        assert item["reason"] and item["spec_clause"]


def test_report_carries_the_honesty_language_verbatim():
    report = feature_report([])
    assert report["caveats"] == list(CAVEATS)
    joined = " ".join(report["caveats"]).lower()
    for phrase in (
        "measured, not assumed",
        "n is small",
        "no p-value",
        "no claim of predictiveness",
    ):
        assert phrase in joined


def test_report_makes_no_claim_of_predictiveness_anywhere():
    """No verdict, no ranking, no significance — checked by walking the JSON.

    §92 is about the SHAPE of the output, not only its wording: a ``p_value``
    key or a ``verdict`` field would be read as a finding no matter what the
    caveats said, so the absence is asserted structurally.
    """
    report = feature_report(
        _rows_with([float(i) for i in range(15)], [float(i) * 0.01 for i in range(15)])
    )
    forbidden = ("p_value", "pvalue", "significance", "verdict", "confidence_level")
    found: list[str] = []

    def walk(node, path="report"):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in forbidden:
                    found.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(report)
    assert found == [], f"report carries a certainty-shaped key: {found}"


def test_report_is_deterministic_over_the_same_rows():
    rows = _rows_with([3.0, 1.0, 2.0, 2.0, 5.0], [0.1, 0.4, 0.2, 0.2, -0.3])
    assert feature_report(rows) == feature_report(rows)


# ===========================================================================
# Part 3 — the endpoint (DB-only assembly)
# ===========================================================================


async def _add_event(
    *,
    key: str,
    ticker: str,
    when: datetime,
    status: EventStatus = EventStatus.CONFIRMED,
    event_type: EventType = EventType.EARNINGS,
    session_label: EventSession = EventSession.AFTER_MARKET,
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title="Q result",
            ticker=ticker,
            scheduled_at=when,
            session=session_label.value,
            status=status.value,
            source=EventSourceKind.STRUCTURED_PROVIDER.value,
            source_name="test",
        )
        s.add(row)
        await s.commit()
        return row.id


async def _add_analysis(
    event_id: int, *, as_of: datetime, bundle: dict, status: str = "OK"
) -> int:
    async with SessionLocal() as s:
        row = EventAnalysisRow(
            event_id=event_id,
            as_of=as_of,
            kind="PRE_EVENT",
            bundle=bundle,
            bundle_digest=f"digest-{event_id}-{as_of.isoformat()}",
            status=status,
        )
        s.add(row)
        await s.commit()
        return row.id


async def _add_metrics(
    event_id: int,
    *,
    basis: str,
    implied: float | None,
    iv: float | None,
    as_of: datetime = NOW,
) -> None:
    async with SessionLocal() as s:
        s.add(
            EventOptionMetricRow(
                event_id=event_id,
                as_of=as_of,
                basis=basis,
                implied_move_pct=implied,
                iv_before=iv,
                status="OK",
            )
        )
        await s.commit()


async def _add_bars(ticker: str, start: date, closes: list[float]) -> None:
    """Consecutive weekday bars from ``start``, one per close."""
    async with SessionLocal() as s:
        day = start
        for close in closes:
            while day.weekday() >= 5:
                day += timedelta(days=1)
            s.add(
                StockBarDaily(
                    ticker=ticker,
                    ts=day,
                    open=close,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=1_000_000.0,
                )
            )
            day += timedelta(days=1)
        await s.commit()


async def _seed_one_studied_event(
    *,
    key: str = "EARNINGS:AAA:2026-03-04",
    ticker: str = "AAA",
    when: datetime = _utc(2026, 3, 4, 21, 0),
    bundle: dict | None = None,
    seed_bars: bool = True,
) -> int:
    """One CONFIRMED AMC print with an analysis, bars and a reaction.

    Bars run 2026-03-02 .. 2026-03-13 with the close jumping on the session
    AFTER the print, which is the reaction an AMC release produces: the
    pre-event close is 2026-03-04's 100.0 and the reaction bar is 2026-03-05.
    """
    event_id = await _add_event(key=key, ticker=ticker, when=when)
    await _add_analysis(
        event_id, as_of=when - timedelta(hours=6), bundle=bundle or _bundle()
    )
    if seed_bars:
        # (ticker, ts) is UNIQUE — a second event on the SAME ticker rides the
        # bars the first one seeded rather than re-inserting them.
        await _add_bars(
            ticker,
            date(2026, 3, 2),
            [98.0, 99.0, 100.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        )
    return event_id


@pytest.mark.anyio
async def test_study_is_empty_and_honest_on_a_fresh_install(client):
    """No events -> 200 with the caveats and a stated absence, never a 404.

    "Nothing has been measured yet" is a complete §86 answer, and the feature
    list plus the caveats are the useful half of the response even then.
    """
    response = await client.get("/api/events/study")
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True
    assert body["report"]["n_events"] == 0
    assert body["rows"] == []
    assert body["report"]["caveats"] == list(CAVEATS)
    assert set(body["report"]["features"]) == set(FEATURE_NAMES)
    for stats in body["report"]["features"].values():
        assert stats["coverage_pct"] == 0.0
        assert stats["rho_1d"]["flag"] == NOT_MEANINGFUL


@pytest.mark.anyio
async def test_study_assembles_a_row_from_stored_rows_only(client):
    """One seeded event -> one row with its features and its real reaction.

    The 1D outcome is hand-checkable: pre-event close 100.0 on 2026-03-04,
    reaction close 105.0 on 2026-03-05, so ``outcome_1d`` = +0.05 exactly. It
    is measured by ``event_reaction`` — the same function the replay tab uses —
    so the study and the UI can never disagree about this print.
    """
    event_id = await _seed_one_studied_event()
    await _add_metrics(
        event_id,
        basis=study_seam.METRIC_BASIS,
        implied=0.066,
        iv=0.48,
    )

    body = (await client.get("/api/events/study")).json()
    assert body["report"]["n_events"] == 1
    assert body["rows_total"] == 1
    row = body["rows"][0]
    assert row["event_id"] == event_id
    assert row["ticker"] == "AAA"
    assert row["event_date"] == "2026-03-04"
    assert row["outcome_1d"] == pytest.approx(0.05)
    assert row["features"]["price_runup_pct"] == pytest.approx(0.12)
    assert row["features"]["news_materiality"] == 3.0
    assert row["features"]["implied_move_pct"] == pytest.approx(0.066)
    assert row["features"]["iv_before"] == pytest.approx(0.48)
    assert body["provenance"]["events_with_stored_bundle"] == 1
    assert body["provenance"]["events_with_option_metrics"] == 1
    assert body["provenance"]["tickers_with_stored_bars"] == 1


@pytest.mark.anyio
async def test_study_never_fetches(client, monkeypatch):
    """A read that reached a vendor would RAISE (audit §7.2 rule 1).

    The seam holds no provider import at all, so the guard is placed on the
    lazy-backfill path the study's neighbours use — the one function a future
    edit would most plausibly reach for when "the bars are missing". If this
    endpoint ever grows a fetch, this test stops passing rather than quietly
    becoming a provider call per event.
    """
    from apps.gateway.routers import analysis as analysis_router

    async def _explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("GET /api/events/study fetched market data")

    monkeypatch.setattr(analysis_router, "ensure_daily_bars", _explode)
    monkeypatch.setattr(
        "apps.gateway.event_price.ensure_daily_bars", _explode, raising=False
    )

    await _seed_one_studied_event()
    response = await client.get("/api/events/study")
    assert response.status_code == 200
    assert response.json()["report"]["n_events"] == 1


@pytest.mark.anyio
async def test_study_reads_the_earliest_bundle_not_a_later_re_run(client):
    """§96: a re-run assembled AFTER the print must not supply the features.

    Two analyses on one event — one from six hours before the release carrying
    ``run_up_pct = 0.12``, one from a week after carrying the unmistakable
    sentinel ``0.999`` (a run-up measured through the very reaction it is
    supposed to predict). The study must read the first. If it took the latest
    row, the sentinel would appear in the payload and the strongest column in
    the table would be pure look-ahead.
    """
    event_id = await _add_event(
        key="EARNINGS:AAA:2026-03-04", ticker="AAA", when=_utc(2026, 3, 4, 21, 0)
    )
    await _add_analysis(
        event_id,
        as_of=_utc(2026, 3, 4, 15, 0),
        bundle=_bundle(run_up=0.12),
    )
    await _add_analysis(
        event_id,
        as_of=_utc(2026, 3, 11, 15, 0),
        bundle=_bundle(run_up=0.999),
    )
    await _add_bars(
        "AAA", date(2026, 3, 2), [98.0, 99.0, 100.0, 105.0, 106.0, 107.0]
    )

    body = (await client.get("/api/events/study")).json()
    assert body["rows"][0]["features"]["price_runup_pct"] == pytest.approx(0.12)
    assert "0.999" not in str(body)


@pytest.mark.anyio
async def test_live_chain_snapshots_never_feed_the_study(client):
    """§85: a LIVE basis row is excluded, whatever it claims.

    A live snapshot is written when somebody opened the options tab, which for
    a past event may be days after the print. Correlating it with the move it
    was taken after would be the largest look-ahead this payload could carry —
    and it would surface as the table's strongest column.
    """
    event_id = await _seed_one_studied_event()
    await _add_metrics(
        event_id, basis="LIVE_CHAIN_SNAPSHOT", implied=0.777, iv=0.888
    )

    body = (await client.get("/api/events/study")).json()
    row = body["rows"][0]
    assert row["features"]["implied_move_pct"] is None
    assert row["features"]["iv_before"] is None
    assert "0.777" not in str(body) and "0.888" not in str(body)
    assert body["provenance"]["option_metric_basis"] == study_seam.METRIC_BASIS


@pytest.mark.anyio
async def test_events_without_a_stored_bundle_are_out_of_sample_but_counted(client):
    """No stored analysis -> no feature vector, and the gap is REPORTED.

    Assembling one on the spot would rebuild every feature from today's data
    (the §96 leak); dropping the event silently would make ``events_in_scope``
    agree with ``events_with_stored_bundle`` and hide how thin the sample is.
    """
    await _seed_one_studied_event()
    await _add_event(
        key="EARNINGS:BBB:2026-03-05", ticker="BBB", when=_utc(2026, 3, 5, 21, 0)
    )

    body = (await client.get("/api/events/study")).json()
    assert body["provenance"]["events_in_scope"] == 2
    assert body["provenance"]["events_with_stored_bundle"] == 1
    assert body["report"]["n_events"] == 1
    assert [r["ticker"] for r in body["rows"]] == ["AAA"]


@pytest.mark.anyio
async def test_estimated_dates_are_excluded_from_the_sample(client):
    """§15: a derived date is not an observation to measure a reaction around."""
    await _add_event(
        key="EARNINGS:CCC:2026-03-04",
        ticker="CCC",
        when=_utc(2026, 3, 4, 21, 0),
        status=EventStatus.ESTIMATED,
    )
    async with SessionLocal() as s:
        event_id = (
            await s.execute(select(EventRow.id).where(EventRow.ticker == "CCC"))
        ).scalar_one()
    await _add_analysis(event_id, as_of=_utc(2026, 3, 4, 15, 0), bundle=_bundle())

    body = (await client.get("/api/events/study")).json()
    assert body["provenance"]["events_in_scope"] == 0
    assert body["report"]["n_events"] == 0


@pytest.mark.anyio
async def test_min_n_only_raises_the_bar_and_never_lowers_it(client):
    """``min_n`` below the floor is a 422; above it, more cells go dark.

    The knob exists for a reader who considers twelve events too few. It must
    not exist for a reader who wants a four-event correlation to look
    quotable, so the floor is enforced by the route's own validation rather
    than by a clamp the caller cannot see.
    """
    await _seed_one_studied_event()

    too_low = await client.get(f"/api/events/study?min_n={MIN_MEANINGFUL_N - 1}")
    assert too_low.status_code == 422
    assert (await client.get("/api/events/study?min_n=0")).status_code == 422

    raised = await client.get("/api/events/study?min_n=50")
    assert raised.status_code == 200
    body = raised.json()
    assert body["report"]["min_meaningful_n"] == 50
    assert body["report"]["min_n_override"] == 50
    cell = body["report"]["features"]["price_runup_pct"]["rho_1d"]
    assert cell["not_meaningful"] is True and cell["flag"] == NOT_MEANINGFUL
    # The alias and the nested cell are the same measurement — a raised bar
    # that showed through only one of them would let a UI quote a flagged cell.
    signed = body["report"]["features"]["price_runup_pct"]["signed"]["rho_1d"]
    assert signed["not_meaningful"] is True


@pytest.mark.anyio
async def test_min_n_cannot_change_a_rho_or_the_sample(client):
    """Raising the bar re-labels cells; it never re-computes them."""
    await _seed_one_studied_event()
    base = (await client.get("/api/events/study")).json()
    raised = (await client.get("/api/events/study?min_n=99")).json()
    assert raised["report"]["n_events"] == base["report"]["n_events"]
    for name in FEATURE_NAMES:
        for horizon in ("rho_1d", "rho_5d"):
            assert (
                raised["report"]["features"][name]["signed"][horizon]["rho"]
                == base["report"]["features"][name]["signed"][horizon]["rho"]
            )
            assert (
                raised["report"]["features"][name]["signed"][horizon]["n"]
                == base["report"]["features"][name]["signed"][horizon]["n"]
            )


@pytest.mark.anyio
async def test_event_type_filter_narrows_the_sample_and_422s_on_nonsense(client):
    await _seed_one_studied_event()
    other_id = await _add_event(
        key="CORPORATE_EVENT:AAA:2026-03-06",
        ticker="AAA",
        when=_utc(2026, 3, 6, 21, 0),
        event_type=EventType.CORPORATE_EVENT,
    )
    await _add_analysis(
        other_id, as_of=_utc(2026, 3, 6, 15, 0), bundle=_bundle(run_up=0.44)
    )

    both = (await client.get("/api/events/study")).json()
    assert both["report"]["n_events"] == 2

    earnings = (await client.get("/api/events/study?event_type=EARNINGS")).json()
    assert earnings["report"]["n_events"] == 1
    assert earnings["rows"][0]["event_key"] == "EARNINGS:AAA:2026-03-04"
    assert earnings["provenance"]["event_type"] == "EARNINGS"

    assert (await client.get("/api/events/study?event_type=NOPE")).status_code == 422


@pytest.mark.anyio
async def test_a_future_as_of_is_a_422_not_a_silent_clamp(client):
    """The same rule every other Catalyst read applies."""
    future = (
        (datetime.now(timezone.utc) + timedelta(days=3))
        .isoformat()
        .replace("+00:00", "Z")
    )
    response = await client.get(f"/api/events/study?as_of={future}")
    assert response.status_code == 422
    assert "future" in str(response.json()["detail"]).lower()


@pytest.mark.anyio
async def test_as_of_bounds_the_sample(client):
    """An event after ``as_of`` cannot have an outcome and is out of scope."""
    await _seed_one_studied_event()
    body = (
        await client.get("/api/events/study?as_of=2026-03-01T00:00:00Z")
    ).json()
    assert body["provenance"]["events_in_scope"] == 0
    assert body["report"]["n_events"] == 0


@pytest.mark.anyio
async def test_study_endpoint_is_deterministic(client):
    """Two identical requests return the same report, key for key."""
    await _seed_one_studied_event()
    first = (await client.get("/api/events/study")).json()
    second = (await client.get("/api/events/study")).json()
    assert first["report"] == second["report"]
    assert first["rows"] == second["rows"]


@pytest.mark.anyio
async def test_payload_carries_the_research_only_label_and_no_verdict(client):
    """§87/§92: research framing present, certainty-shaped keys absent."""
    await _seed_one_studied_event()
    body = (await client.get("/api/events/study")).json()
    assert "RESEARCH ONLY" in body["not_a_signal"]
    assert body["provenance_tier"] == "QUANT"

    forbidden = ("p_value", "pvalue", "significance", "verdict", "predictive_score")
    found: list[str] = []

    def walk(node, path="body"):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in forbidden:
                    found.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(body)
    assert found == [], f"study payload carries a certainty-shaped key: {found}"


@pytest.mark.anyio
async def test_rows_are_capped_but_the_report_is_not(client, monkeypatch):
    """The row listing is a transport cap; the report is over EVERYTHING.

    Asserted by lowering the cap rather than by seeding two hundred events: the
    property under test is that ``rows_total`` and ``n_events`` keep counting
    past the slice, and seeding to the real limit would make the test slow
    without making it stronger.
    """
    monkeypatch.setattr(study_seam, "ROW_LIMIT", 1)
    await _seed_one_studied_event()
    await _seed_one_studied_event(
        key="EARNINGS:AAA:2026-03-10",
        when=_utc(2026, 3, 10, 21, 0),
        bundle=_bundle(run_up=0.2),
        seed_bars=False,
    )

    body = (await client.get("/api/events/study")).json()
    assert body["rows_total"] == 2
    assert body["report"]["n_events"] == 2
    assert len(body["rows"]) == 1
    assert body["rows_limit"] == 1


# ===========================================================================
# Part 4 — purity and exports (audit §7.4)
# ===========================================================================


def test_event_study_lib_imports_no_io_layer():
    """The pure module may not reach a provider, a database or the gateway.

    This is the guard that makes every look-ahead claim in the module's
    docstring checkable rather than aspirational: a library that cannot import
    ``libs.market_data`` cannot quietly fetch today's price while measuring
    whether yesterday's feature predicted it.
    """
    from pathlib import Path

    from libs.trading_core.events import event_study as lib

    source = Path(lib.__file__).read_text(encoding="utf-8")
    for forbidden in ("libs.market_data", "libs.event_calendar", "apps."):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_gateway_seam_holds_no_market_data_import():
    """The seam is DB-only by CONSTRUCTION, not merely by current behaviour.

    ``test_study_never_fetches`` proves today's code path spends no provider
    call; this proves the import that would let a future edit do so is not even
    present, which is the difference between a passing test and a design.
    """
    from pathlib import Path

    source = Path(study_seam.__file__).read_text(encoding="utf-8")
    for forbidden in ("libs.market_data", "libs.event_calendar"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_package_exports_event_study_symbols():
    from libs.trading_core import events

    for name in (
        "CAVEATS",
        "EVENT_STUDY_MODEL_VERSION",
        "FEATURES_NOT_MEASURABLE",
        "FEATURE_NAMES",
        "FEATURE_SPECS",
        "MIN_MEANINGFUL_N",
        "NOT_MEANINGFUL",
        "FeatureRow",
        "FeatureSpec",
        "FeatureStat",
        "average_ranks",
        "collect_feature_rows",
        "feature_report",
        "features_from_bundle",
        "spearman_rank_corr",
    ):
        assert hasattr(events, name), name
        assert name in events.__all__, name
