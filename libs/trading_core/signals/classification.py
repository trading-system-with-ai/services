"""Directional Edge classification layer (upgrade 2026-08-12 §7, §8).

Pure labeling over the directional scorer's output — deterministic, no DB, no
FastAPI, shared verbatim by backtest and live paths (plan §21). It answers
"how strong is this edge?" in the seven-band vocabulary the UI shows; it does
NOT grant permission to trade (Tradeability and Risk stay sovereign, §9/§43).

Every threshold here is a research parameter on
:class:`EdgeClassificationParams` — configurable, versioned, never a
hardcoded truth (§7: "All thresholds must be configurable and clearly
identified as research parameters").

STRONG bands additionally require a minimum same-side score (§7): a large
edge produced by a weak-but-one-sided read (e.g. bull 45 / bear 0 on thin
data) must not be labeled STRONG. When the edge clears the strong threshold
but the side score does not, the classification degrades one band to
MODERATE — the evidence is real, the conviction bar for STRONG is not met.
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.trading_core.models import DirectionalEdgeClass


@dataclass(frozen=True)
class EdgeClassificationParams:
    """Research thresholds for the seven Directional Edge bands (§7).

    Bands (inclusive boundaries, mirrored for bear):

    - ``|edge| >= strong_edge``  -> STRONG (needs the side-score minimum too)
    - ``|edge| >= moderate_edge`` -> MODERATE
    - ``|edge| >= weak_edge``     -> WEAK
    - otherwise                   -> NEUTRAL

    ``strong_min_side_score``: minimum Bull Score (mirrored: Bear Score) for
    a STRONG label; below it a strong-band edge degrades to MODERATE.

    ``version`` identifies this threshold set in API payloads and audit
    records (§6: configuration must be versioned).
    """

    strong_edge: float = 50.0
    moderate_edge: float = 25.0
    weak_edge: float = 15.0
    strong_min_side_score: float = 70.0
    version: str = "edge-class-v1"


def classify_edge(
    bull_score: float,
    bear_score: float,
    directional_edge: float,
    params: EdgeClassificationParams = EdgeClassificationParams(),
) -> DirectionalEdgeClass:
    """Classify a directional edge into the seven §7 bands.

    Symmetric by construction: the bear side is evaluated with the same
    thresholds on ``-edge`` and ``bear_score``.
    """
    edge = directional_edge
    if edge >= params.strong_edge:
        if bull_score >= params.strong_min_side_score:
            return DirectionalEdgeClass.STRONG_BULL
        return DirectionalEdgeClass.MODERATE_BULL
    if edge >= params.moderate_edge:
        return DirectionalEdgeClass.MODERATE_BULL
    if edge >= params.weak_edge:
        return DirectionalEdgeClass.WEAK_BULL
    if edge <= -params.strong_edge:
        if bear_score >= params.strong_min_side_score:
            return DirectionalEdgeClass.STRONG_BEAR
        return DirectionalEdgeClass.MODERATE_BEAR
    if edge <= -params.moderate_edge:
        return DirectionalEdgeClass.MODERATE_BEAR
    if edge <= -params.weak_edge:
        return DirectionalEdgeClass.WEAK_BEAR
    return DirectionalEdgeClass.NEUTRAL


def edge_legend(
    params: EdgeClassificationParams = EdgeClassificationParams(),
) -> list[dict]:
    """The §8 threshold legend, derived from ``params`` — the UI renders this
    verbatim so displayed bands can never drift from the classifier.

    Bands are listed bull-to-bear and jointly cover [-100, +100] with no gaps
    (pinned by test). ``requires_side_score`` is present only on the STRONG
    bands (the §7 minimum-side-score enhancement).
    """
    s, m, w = params.strong_edge, params.moderate_edge, params.weak_edge
    return [
        {
            "classification": DirectionalEdgeClass.STRONG_BULL.value,
            "edge_min": s,
            "edge_max": 100.0,
            "requires_side_score": params.strong_min_side_score,
        },
        {
            "classification": DirectionalEdgeClass.MODERATE_BULL.value,
            "edge_min": m,
            "edge_max": s,
        },
        {
            "classification": DirectionalEdgeClass.WEAK_BULL.value,
            "edge_min": w,
            "edge_max": m,
        },
        {
            "classification": DirectionalEdgeClass.NEUTRAL.value,
            "edge_min": -w,
            "edge_max": w,
        },
        {
            "classification": DirectionalEdgeClass.WEAK_BEAR.value,
            "edge_min": -m,
            "edge_max": -w,
        },
        {
            "classification": DirectionalEdgeClass.MODERATE_BEAR.value,
            "edge_min": -s,
            "edge_max": -m,
        },
        {
            "classification": DirectionalEdgeClass.STRONG_BEAR.value,
            "edge_min": -100.0,
            "edge_max": -s,
            "requires_side_score": params.strong_min_side_score,
        },
    ]
