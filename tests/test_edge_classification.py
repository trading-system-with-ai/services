"""Directional Edge classification + contribution reconciliation (upgrade
2026-08-12 §7, §8, §44).

§44 requirements pinned here:

- Directional Edge = Bull Score - Bear Score exactly.
- Displayed contribution totals reconcile to the score EXACTLY (the score IS
  the sum of contributions, by construction).
- Threshold classification matches the configured band boundaries, including
  the minimum-side-score requirement for STRONG labels.
- The §8 legend derives from the same parameters as the classifier and covers
  [-100, +100] with no gaps.
"""
import pytest

from libs.trading_core.models import DirectionalEdgeClass
from libs.trading_core.signals import (
    EdgeClassificationParams,
    classify_edge,
    edge_legend,
    score_direction,
)

C = DirectionalEdgeClass
PARAMS = EdgeClassificationParams()


# ---------------------------------------------------------------------------
# Band boundaries (§7): inclusive thresholds, mirrored bear side.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("bull", "bear", "edge", "expected"),
    [
        # Strong band, side score meets the §7 minimum.
        (77.8, 11.1, 66.7, C.STRONG_BULL),
        (70.0, 20.0, 50.0, C.STRONG_BULL),   # both thresholds exactly met
        (100.0, 0.0, 100.0, C.STRONG_BULL),
        # Strong-band edge but side score below minimum -> degrades MODERATE.
        (69.9, 19.9, 50.0, C.MODERATE_BULL),
        (45.0, -5.0 + 0.0, 50.0, C.MODERATE_BULL),
        # Moderate band.
        (60.0, 11.0, 49.0, C.MODERATE_BULL),
        (50.0, 25.0, 25.0, C.MODERATE_BULL),  # boundary inclusive
        # Weak band.
        (40.0, 16.0, 24.0, C.WEAK_BULL),
        (30.0, 15.0, 15.0, C.WEAK_BULL),      # boundary inclusive
        # Neutral.
        (30.0, 15.1, 14.9, C.NEUTRAL),
        (0.0, 0.0, 0.0, C.NEUTRAL),
        (10.0, 24.9, -14.9, C.NEUTRAL),
        # Bear mirror.
        (15.0, 30.0, -15.0, C.WEAK_BEAR),
        (16.0, 40.0, -24.0, C.WEAK_BEAR),
        (25.0, 50.0, -25.0, C.MODERATE_BEAR),
        (11.0, 60.0, -49.0, C.MODERATE_BEAR),
        (19.9, 69.9, -50.0, C.MODERATE_BEAR),  # strong edge, weak side score
        (20.0, 70.0, -50.0, C.STRONG_BEAR),
        (11.1, 77.8, -66.7, C.STRONG_BEAR),
        (0.0, 100.0, -100.0, C.STRONG_BEAR),
    ],
)
def test_band_boundaries(bull, bear, edge, expected):
    assert classify_edge(bull, bear, edge, params=PARAMS) is expected


def test_thresholds_are_parameters_not_truths():
    """Custom thresholds move the bands — nothing is hardcoded (§7)."""
    custom = EdgeClassificationParams(
        strong_edge=60.0, moderate_edge=30.0, weak_edge=10.0,
        strong_min_side_score=80.0, version="edge-class-test",
    )
    assert classify_edge(85.0, 25.0, 60.0, params=custom) is C.STRONG_BULL
    assert classify_edge(79.9, 19.9, 60.0, params=custom) is C.MODERATE_BULL
    assert classify_edge(50.0, 20.0, 30.0, params=custom) is C.MODERATE_BULL
    assert classify_edge(30.0, 20.0, 10.0, params=custom) is C.WEAK_BULL
    assert classify_edge(29.0, 20.0, 9.9, params=custom) is C.NEUTRAL


# ---------------------------------------------------------------------------
# Legend (§8): derived from params, gap-free over [-100, +100].
# ---------------------------------------------------------------------------

def test_legend_matches_classifier_and_covers_range():
    legend = edge_legend(PARAMS)
    assert [band["classification"] for band in legend] == [
        "STRONG_BULL", "MODERATE_BULL", "WEAK_BULL", "NEUTRAL",
        "WEAK_BEAR", "MODERATE_BEAR", "STRONG_BEAR",
    ]
    # Bull-to-bear adjacency: each band's floor is the next band's ceiling —
    # no gaps, no overlaps beyond the shared boundary.
    for upper, lower in zip(legend, legend[1:]):
        assert upper["edge_min"] == lower["edge_max"]
    assert legend[0]["edge_max"] == 100.0
    assert legend[-1]["edge_min"] == -100.0
    # STRONG bands (and only they) carry the §7 side-score requirement.
    assert legend[0]["requires_side_score"] == PARAMS.strong_min_side_score
    assert legend[-1]["requires_side_score"] == PARAMS.strong_min_side_score
    assert all("requires_side_score" not in band for band in legend[1:-1])


def test_legend_boundaries_agree_with_classify_edge():
    """Sampling inside every band (with a qualifying side score) reproduces
    the band's own label — the legend can never drift from the classifier."""
    for band in edge_legend(PARAMS):
        mid = (band["edge_min"] + band["edge_max"]) / 2.0
        bull = max(mid, 0.0) + PARAMS.strong_min_side_score
        bear = max(-mid, 0.0) + PARAMS.strong_min_side_score
        assert (
            classify_edge(bull, bear, mid, params=PARAMS).value
            == band["classification"]
        )


# ---------------------------------------------------------------------------
# §44 reconciliation on real scorer output.
# ---------------------------------------------------------------------------

def _trending_series(n: int = 260, up: bool = True):
    """Deterministic monotonic-ish OHLCV series that triggers real components."""
    closes, highs, lows, volumes = [], [], [], []
    price = 100.0
    for i in range(n):
        step = 0.5 if up else -0.5
        wiggle = 0.3 if i % 7 == 0 else -0.1
        price = max(price + step + (wiggle if up else -wiggle), 1.0)
        closes.append(price)
        highs.append(price * 1.01)
        lows.append(price * 0.99)
        volumes.append(1_000_000.0 + (50_000.0 if i % 3 == 0 else 0.0))
    return closes, highs, lows, volumes


@pytest.mark.parametrize("up", [True, False])
def test_scores_reconcile_exactly_with_contributions(up):
    closes, highs, lows, volumes = _trending_series(up=up)
    result = score_direction(closes, highs, lows, volumes=volumes)

    bull_sum = sum(c.contribution for c in result.components if c.side == "bull")
    bear_sum = sum(c.contribution for c in result.components if c.side == "bear")
    # EXACT equality (§44): the score is the sum of displayed contributions.
    assert result.bull_score == bull_sum
    assert result.bear_score == bear_sum
    assert result.directional_edge == result.bull_score - result.bear_score

    for c in result.components:
        if c.triggered:
            assert c.contribution == c.max_contribution > 0.0
        else:
            assert c.contribution == 0.0
            assert c.max_contribution > 0.0  # weight share still displayed

    # Each side's max contributions span the full 0-100 scale.
    for side in ("bull", "bear"):
        total = sum(
            c.max_contribution for c in result.components if c.side == side
        )
        assert total == pytest.approx(100.0)


def test_result_carries_classification_and_versions():
    closes, highs, lows, volumes = _trending_series(up=True)
    result = score_direction(closes, highs, lows, volumes=volumes)
    # A clean uptrend classifies bullish, consistently with classify_edge.
    assert result.classification is classify_edge(
        result.bull_score, result.bear_score, result.directional_edge
    )
    assert result.classification in (C.STRONG_BULL, C.MODERATE_BULL, C.WEAK_BULL)
    assert result.weights_version == "score-weights-v1-grouped"
    assert result.classification_version == EdgeClassificationParams().version


def test_no_data_is_neutral_with_zero_contributions():
    result = score_direction([], [], [])
    assert result.bull_score == 0.0
    assert result.bear_score == 0.0
    assert result.classification is C.NEUTRAL
    assert all(c.contribution == 0.0 for c in result.components)
