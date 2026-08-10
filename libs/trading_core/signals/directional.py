"""Directional Signal Engine v0 (development plan §6.2).

Pure, deterministic bull/bear scoring over technical features — no DB, no
FastAPI — shared verbatim by the backtest and live paths (plan §21). All
indicator math comes exclusively from :mod:`libs.trading_core.features`;
every weight, period, zone and threshold is a parameter on
:class:`DirectionalParams`, never a hardcoded truth (plan §6.2).

Scoring model (plan §6.2): each feature is evaluated as a mirrored bull/bear
component pair. A side's score is::

    sum(weights of that side's TRIGGERED components)
    / sum(weights of ALL that side's components) * 100

``directional_edge = bull_score - bear_score``; the bias is BULL/BEAR only
when ``|edge| >= bias_threshold``, else NEUTRAL.

Insufficient-data behaviour (documented contract): a component whose inputs
are not yet defined (warmup) is listed but NOT triggered, with a detail
string starting with "insufficient data". If the series is shorter than the
longest required period, every component is untriggered, both scores are 0
and the bias is NEUTRAL — that is the correct no-information posture, not an
error (mirrors the regime engine's no-trade default, plan §6.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from libs.trading_core.features import macd, pivot_highs, pivot_lows, rsi, sma
from libs.trading_core.models import DirectionalBias


@dataclass(frozen=True)
class DirectionalParams:
    """Backtest parameters for the directional scorer (plan §6.2).

    Every value is a tunable backtest parameter — defaults are starting
    points for optimization, never truths (plan §6.2).

    Indicator periods:

    - ``sma_fast`` / ``sma_mid`` / ``sma_slow``: SMA periods for the three
      close-vs-SMA components.
    - ``slope_lookback``: bars over which the fast SMA's slope is measured.
    - ``macd_fast`` / ``macd_slow`` / ``macd_signal``: MACD periods.
    - ``rsi_period``: RSI period.
    - ``rsi_bull_zone`` / ``rsi_bear_zone``: inclusive (lo, hi) RSI bands
      counted as bull / bear continuation (an overbought/oversold RSI outside
      the band deliberately does NOT add trend confirmation).
    - ``pivot_window``: confirmation window for pivot structure (HH+HL / LH+LL).
    - ``volume_sma_period``: SMA period for the volume-expansion component.

    Decision threshold:

    - ``bias_threshold``: minimum ``|directional_edge|`` (bull_score -
      bear_score, each 0-100) required to call a BULL or BEAR bias.

    Weights (one per mirrored feature pair, applied to BOTH sides so the
    scoring stays symmetric):

    - ``weight_sma_fast`` / ``weight_sma_mid`` / ``weight_sma_slow``:
      close vs the corresponding SMA.
    - ``weight_sma_slope``: fast-SMA slope sign.
    - ``weight_macd_cross``: MACD line vs signal line.
    - ``weight_macd_zero``: MACD line vs zero.
    - ``weight_rsi_zone``: RSI inside the continuation zone.
    - ``weight_structure``: pivot structure (HH+HL bullish / LH+LL bearish).
    - ``weight_volume``: volume expansion (only scored when volumes are given).
    """

    # Indicator periods.
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200
    slope_lookback: int = 5
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_bull_zone: tuple[float, float] = (50.0, 70.0)
    rsi_bear_zone: tuple[float, float] = (30.0, 50.0)
    pivot_window: int = 5
    volume_sma_period: int = 20
    # Decision threshold.
    bias_threshold: float = 25.0
    # Feature weights (default 1.0 each).
    weight_sma_fast: float = 1.0
    weight_sma_mid: float = 1.0
    weight_sma_slow: float = 1.0
    weight_sma_slope: float = 1.0
    weight_macd_cross: float = 1.0
    weight_macd_zero: float = 1.0
    weight_rsi_zone: float = 1.0
    weight_structure: float = 1.0
    weight_volume: float = 1.0


@dataclass
class SignalComponent:
    """One evaluated feature on one side, with a human-readable explanation.

    ``detail`` always contains the actual numbers used (explainability,
    plan §6.2), e.g. ``"close 123.4500 > sma20 120.1000"``; components whose
    inputs are still in warmup say ``"insufficient data: ..."`` and are never
    triggered.
    """

    name: str
    side: str  # "bull" | "bear"
    triggered: bool
    weight: float
    detail: str


@dataclass
class DirectionalResult:
    """Directional scoring output (plan §6.2).

    - ``bull_score`` / ``bear_score``: 0-100, each = sum of that side's
      triggered component weights / sum of ALL that side's component weights
      * 100.
    - ``directional_edge``: ``bull_score - bear_score``.
    - ``bias``: BULL if ``edge >= bias_threshold``, BEAR if
      ``edge <= -bias_threshold``, else NEUTRAL.
    - ``components``: ALL evaluated components, triggered or not, so any
      score can be audited and recomputed from this list alone.
    """

    bull_score: float
    bear_score: float
    directional_edge: float
    bias: DirectionalBias
    components: list[SignalComponent] = field(default_factory=list)


def _f(value: float) -> str:
    """Uniform numeric formatting for component details."""
    return f"{value:.4f}"


def score_direction(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None = None,
    params: DirectionalParams = DirectionalParams(),
) -> DirectionalResult:
    """Score directional bull/bear evidence from aligned OHLC(V) series
    (plan §6.2).

    v0 bull features (each with an exactly mirrored bear counterpart):

    1.  ``close > sma_fast``            (bear: ``close < sma_fast``)
    2.  ``close > sma_mid``             (bear mirrored)
    3.  ``close > sma_slow``            (bear mirrored)
    4.  fast-SMA slope over ``slope_lookback`` bars ``> 0`` (bear: ``< 0``)
    5.  MACD line ``>`` signal line     (bear: ``<``)
    6.  MACD line ``> 0``               (bear: ``< 0``)
    7.  RSI inside ``rsi_bull_zone``    (bear: inside ``rsi_bear_zone``),
        inclusive bounds
    8.  bullish pivot structure: the most recent confirmed pivot high is a
        Higher High AND the most recent confirmed pivot low is a Higher Low
        (via ``pivot_highs`` / ``pivot_lows``; bear: LH + LL)
    9.  volume expansion: last volume ``>`` SMA(``volume_sma_period``) of
        volumes. Direction-agnostic participation confirmation in v0, counted
        identically on both sides. Only evaluated when ``volumes`` is given —
        otherwise the component is skipped entirely from BOTH sides'
        denominators (plan §6.2).

    Components with insufficient data are listed untriggered with an
    "insufficient data" detail. With fewer bars than the longest required
    period, all components are untriggered => both scores 0 and bias NEUTRAL —
    correct no-information behaviour, not an error.
    """
    if not (len(closes) == len(highs) == len(lows)):
        raise ValueError(
            "closes, highs and lows must have equal length, got "
            f"{len(closes)}/{len(highs)}/{len(lows)}"
        )
    if volumes is not None and len(volumes) != len(closes):
        raise ValueError(
            f"volumes must align with closes, got {len(volumes)}/{len(closes)}"
        )
    n = len(closes)
    last_close = closes[-1] if n else None
    components: list[SignalComponent] = []

    def pair(
        name: str,
        weight: float,
        bull: bool | None,
        bear: bool | None,
        bull_detail: str,
        bear_detail: str,
    ) -> None:
        """Append the mirrored bull/bear components for one feature.

        ``None`` for a trigger means insufficient data: untriggered, listed."""
        components.append(
            SignalComponent(name, "bull", bool(bull), weight, bull_detail)
        )
        components.append(
            SignalComponent(name, "bear", bool(bear), weight, bear_detail)
        )

    # --- Features 1-3: close vs SMAs ------------------------------------
    for name, period, weight in (
        ("close_vs_sma_fast", params.sma_fast, params.weight_sma_fast),
        ("close_vs_sma_mid", params.sma_mid, params.weight_sma_mid),
        ("close_vs_sma_slow", params.sma_slow, params.weight_sma_slow),
    ):
        sma_last = sma(closes, period)[-1] if n else None
        if sma_last is None or last_close is None:
            detail = f"insufficient data: sma{period} needs {period} bars, have {n}"
            pair(name, weight, None, None, detail, detail)
        else:
            cmp_bull = last_close > sma_last
            cmp_bear = last_close < sma_last
            base = f"close {_f(last_close)} {{}} sma{period} {_f(sma_last)}"
            pair(
                name,
                weight,
                cmp_bull,
                cmp_bear,
                base.format(">" if cmp_bull else "<="),
                base.format("<" if cmp_bear else ">="),
            )

    # --- Feature 4: fast-SMA slope --------------------------------------
    sma_fast_series = sma(closes, params.sma_fast) if n else []
    slope: float | None = None
    slope_idx = n - 1 - params.slope_lookback
    if n and slope_idx >= 0:
        latest, prior = sma_fast_series[-1], sma_fast_series[slope_idx]
        if latest is not None and prior is not None:
            slope = latest - prior
    if slope is None:
        need = params.sma_fast + params.slope_lookback
        detail = (
            f"insufficient data: sma{params.sma_fast} slope over "
            f"{params.slope_lookback} bars needs {need} bars, have {n}"
        )
        pair("sma_fast_slope", params.weight_sma_slope, None, None, detail, detail)
    else:
        base = (
            f"sma{params.sma_fast} slope over {params.slope_lookback} "
            f"bars = {_f(slope)} {{}} 0"
        )
        pair(
            "sma_fast_slope",
            params.weight_sma_slope,
            slope > 0.0,
            slope < 0.0,
            base.format(">" if slope > 0.0 else "<="),
            base.format("<" if slope < 0.0 else ">="),
        )

    # --- Features 5-6: MACD cross and zero line -------------------------
    macd_result = (
        macd(closes, fast=params.macd_fast, slow=params.macd_slow, signal=params.macd_signal)
        if n
        else {"macd": [], "signal": []}
    )
    macd_last = macd_result["macd"][-1] if n else None
    signal_last = macd_result["signal"][-1] if n else None

    if macd_last is None or signal_last is None:
        need = params.macd_slow + params.macd_signal - 1
        detail = (
            f"insufficient data: macd({params.macd_fast},{params.macd_slow},"
            f"{params.macd_signal}) signal needs {need} bars, have {n}"
        )
        pair("macd_cross", params.weight_macd_cross, None, None, detail, detail)
    else:
        base = f"macd {_f(macd_last)} {{}} signal {_f(signal_last)}"
        pair(
            "macd_cross",
            params.weight_macd_cross,
            macd_last > signal_last,
            macd_last < signal_last,
            base.format(">" if macd_last > signal_last else "<="),
            base.format("<" if macd_last < signal_last else ">="),
        )

    if macd_last is None:
        detail = (
            f"insufficient data: macd line needs {params.macd_slow} bars, have {n}"
        )
        pair("macd_zero", params.weight_macd_zero, None, None, detail, detail)
    else:
        base = f"macd {_f(macd_last)} {{}} 0"
        pair(
            "macd_zero",
            params.weight_macd_zero,
            macd_last > 0.0,
            macd_last < 0.0,
            base.format(">" if macd_last > 0.0 else "<="),
            base.format("<" if macd_last < 0.0 else ">="),
        )

    # --- Feature 7: RSI continuation zones ------------------------------
    rsi_last = rsi(closes, period=params.rsi_period)[-1] if n else None
    if rsi_last is None:
        detail = (
            f"insufficient data: rsi{params.rsi_period} needs "
            f"{params.rsi_period + 1} bars, have {n}"
        )
        pair("rsi_zone", params.weight_rsi_zone, None, None, detail, detail)
    else:
        bull_lo, bull_hi = params.rsi_bull_zone
        bear_lo, bear_hi = params.rsi_bear_zone
        in_bull = bull_lo <= rsi_last <= bull_hi
        in_bear = bear_lo <= rsi_last <= bear_hi
        pair(
            "rsi_zone",
            params.weight_rsi_zone,
            in_bull,
            in_bear,
            f"rsi {_f(rsi_last)} {'inside' if in_bull else 'outside'} "
            f"bull zone [{_f(bull_lo)}, {_f(bull_hi)}]",
            f"rsi {_f(rsi_last)} {'inside' if in_bear else 'outside'} "
            f"bear zone [{_f(bear_lo)}, {_f(bear_hi)}]",
        )

    # --- Feature 8: pivot structure (HH+HL / LH+LL) ---------------------
    high_pivots = [
        i for i, flag in enumerate(pivot_highs(highs, window=params.pivot_window)) if flag
    ]
    low_pivots = [
        i for i, flag in enumerate(pivot_lows(lows, window=params.pivot_window)) if flag
    ]
    if len(high_pivots) < 2 or len(low_pivots) < 2:
        detail = (
            "insufficient data: structure needs 2 confirmed pivot highs and "
            f"2 confirmed pivot lows (window {params.pivot_window}), have "
            f"{len(high_pivots)}/{len(low_pivots)}"
        )
        pair("pivot_structure", params.weight_structure, None, None, detail, detail)
    else:
        h_prev, h_last = highs[high_pivots[-2]], highs[high_pivots[-1]]
        l_prev, l_last = lows[low_pivots[-2]], lows[low_pivots[-1]]
        hh, hl = h_last > h_prev, l_last > l_prev
        lh, ll = h_last < h_prev, l_last < l_prev
        pivots_txt = (
            f"pivot highs {_f(h_prev)} -> {_f(h_last)}, "
            f"pivot lows {_f(l_prev)} -> {_f(l_last)}"
        )
        pair(
            "pivot_structure",
            params.weight_structure,
            hh and hl,
            lh and ll,
            f"{'bullish HH+HL' if hh and hl else 'no bullish HH+HL'}: {pivots_txt}",
            f"{'bearish LH+LL' if lh and ll else 'no bearish LH+LL'}: {pivots_txt}",
        )

    # --- Feature 9: volume expansion (only when volumes are given) ------
    if volumes is not None:
        vol_sma_last = sma(volumes, params.volume_sma_period)[-1] if n else None
        last_volume = volumes[-1] if volumes else None
        if vol_sma_last is None or last_volume is None:
            detail = (
                f"insufficient data: volume sma{params.volume_sma_period} needs "
                f"{params.volume_sma_period} bars, have {len(volumes)}"
            )
            pair("volume_expansion", params.weight_volume, None, None, detail, detail)
        else:
            expanding = last_volume > vol_sma_last
            detail = (
                f"volume {_f(last_volume)} "
                f"{'>' if expanding else '<='} "
                f"sma{params.volume_sma_period}(volume) {_f(vol_sma_last)}"
                " (direction-agnostic participation confirmation)"
            )
            # v0: volume expansion confirms participation in either direction,
            # so the same condition is counted on both sides (plan §6.2).
            pair("volume_expansion", params.weight_volume, expanding, expanding, detail, detail)
    # else: volumes not given -> component skipped entirely from BOTH sides'
    # denominators (plan §6.2).

    # --- Scores, edge, bias ---------------------------------------------
    def side_score(side: str) -> float:
        total = sum(c.weight for c in components if c.side == side)
        if total <= 0.0:
            return 0.0
        hit = sum(c.weight for c in components if c.side == side and c.triggered)
        return hit / total * 100.0

    bull_score = side_score("bull")
    bear_score = side_score("bear")
    edge = bull_score - bear_score

    if edge >= params.bias_threshold:
        bias = DirectionalBias.BULL
    elif edge <= -params.bias_threshold:
        bias = DirectionalBias.BEAR
    else:
        bias = DirectionalBias.NEUTRAL

    return DirectionalResult(
        bull_score=bull_score,
        bear_score=bear_score,
        directional_edge=edge,
        bias=bias,
        components=components,
    )
