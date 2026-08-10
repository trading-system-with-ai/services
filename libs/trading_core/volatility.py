"""Volatility Regime v0 (development plan §7).

Pure, deterministic classification of the option-volatility environment —
no DB, no FastAPI, no market data. The engine receives an at-the-money
implied volatility and (optionally) a realized volatility already computed
upstream, and maps them to an :class:`~libs.trading_core.models.IVRegime`.

PROVISIONAL v0 thresholds (plan §7): with no stored IV history yet, IV Rank
/ IV Percentile cannot be computed, so this version classifies on absolute
IV LEVELS plus the IV/RV ratio. These level thresholds are research
parameters to be replaced by rank-based ones once IV history accumulates —
they are documented parameters on :class:`VolRegimeParams`, never hardcoded
truths (plan §6.2 discipline applied to §7).

Plan §7 caution, honored here: "Do not assume IV > RV automatically means
options are overpriced" — the IV/RV ratio only ESCALATES the regime
(HIGH/EXTREME) or, together with a low IV level, confirms LOW; it is never
by itself a verdict that premium is rich or cheap. When RV is unavailable
the ratio is honestly ``None`` (house rule: honest nulls) and classification
falls back to IV levels alone.
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.trading_core.models import IVRegime


@dataclass(frozen=True)
class VolRegimeParams:
    """Volatility-regime thresholds (plan §7) — all tunable parameters.

    PROVISIONAL level-based thresholds: absolute-IV cut points stand in for
    IV Rank until enough IV history is stored to compute rank/percentile
    (plan §7). Per §7, "Do not assume IV > RV automatically means options
    are overpriced" — the ratio thresholds escalate or confirm, they never
    price premium on their own.

    - ``low_iv`` / ``high_iv`` / ``extreme_iv``: ATM IV levels bounding
      LOW / HIGH / EXTREME (annualized, e.g. 0.20 = 20%).
    - ``low_ratio`` / ``high_ratio`` / ``extreme_ratio``: IV/RV ratio
      bounds; the ratio can escalate to HIGH/EXTREME and must be at or
      below ``low_ratio`` (or unavailable) for a LOW verdict.
    """

    low_iv: float = 0.20
    high_iv: float = 0.35
    extreme_iv: float = 0.60
    low_ratio: float = 1.1
    high_ratio: float = 1.5
    extreme_ratio: float = 2.0

    def __post_init__(self) -> None:
        if not (0.0 < self.low_iv < self.high_iv < self.extreme_iv):
            raise ValueError(
                "IV level thresholds must satisfy 0 < low_iv < high_iv < "
                f"extreme_iv, got {self.low_iv!r}/{self.high_iv!r}/"
                f"{self.extreme_iv!r}"
            )
        if not (0.0 < self.low_ratio < self.high_ratio < self.extreme_ratio):
            raise ValueError(
                "IV/RV ratio thresholds must satisfy 0 < low_ratio < "
                f"high_ratio < extreme_ratio, got {self.low_ratio!r}/"
                f"{self.high_ratio!r}/{self.extreme_ratio!r}"
            )


@dataclass
class VolRegimeResult:
    """One fully explainable volatility-regime classification (plan §7, §37).

    - ``regime``: the :class:`IVRegime` verdict.
    - ``features``: the real numbers used — ``atm_iv``, ``rv``,
      ``iv_rv_ratio`` (``None`` when RV is unavailable: honest nulls) and
      ``thresholds_fired``, the list of every threshold comparison that was
      true (e.g. ``["extreme_iv", "high_iv"]``); the regime is the
      highest-precedence band among them.
    """

    regime: IVRegime
    features: dict


def classify_vol_regime(
    atm_iv: float,
    rv: float | None,
    params: VolRegimeParams = VolRegimeParams(),
) -> VolRegimeResult:
    """Classify the volatility regime from ATM IV and realized vol (plan §7).

    Precedence (first match wins):

    1. EXTREME — ``atm_iv >= extreme_iv`` OR the IV/RV ratio (when
       computable) ``>= extreme_ratio``.
    2. HIGH    — ``atm_iv >= high_iv`` OR ratio ``>= high_ratio``.
    3. LOW     — ``atm_iv <= low_iv`` AND (ratio unavailable OR ratio
       ``<= low_ratio``).
    4. NORMAL  — everything else.

    ``rv`` may be ``None`` when realized vol is not computable upstream; the
    ratio is then honestly ``None`` and only the IV-level thresholds apply
    (plan §7: the ratio is a supporting feature, and "Do not assume IV > RV
    automatically means options are overpriced").

    Raises ``ValueError`` on ``atm_iv <= 0`` and on a non-``None``
    ``rv <= 0`` (a degenerate realized vol; callers with no usable RV must
    pass ``None`` — honest nulls, never a fake number).
    """
    if atm_iv <= 0.0:
        raise ValueError(f"atm_iv must be > 0, got {atm_iv!r}")
    if rv is not None and rv <= 0.0:
        raise ValueError(
            f"rv must be > 0 when provided, got {rv!r}; pass None when "
            "realized vol is unavailable"
        )

    ratio = atm_iv / rv if rv is not None else None

    level_extreme = atm_iv >= params.extreme_iv
    ratio_extreme = ratio is not None and ratio >= params.extreme_ratio
    level_high = atm_iv >= params.high_iv
    ratio_high = ratio is not None and ratio >= params.high_ratio
    level_low = atm_iv <= params.low_iv
    ratio_low = ratio is not None and ratio <= params.low_ratio

    if level_extreme or ratio_extreme:
        regime = IVRegime.EXTREME
    elif level_high or ratio_high:
        regime = IVRegime.HIGH
    elif level_low and (ratio is None or ratio_low):
        regime = IVRegime.LOW
    else:
        regime = IVRegime.NORMAL

    thresholds_fired = [
        name
        for name, fired in (
            ("extreme_iv", level_extreme),
            ("extreme_ratio", ratio_extreme),
            ("high_iv", level_high),
            ("high_ratio", ratio_high),
            ("low_iv", level_low),
            ("low_ratio", ratio_low),
        )
        if fired
    ]
    return VolRegimeResult(
        regime=regime,
        features={
            "atm_iv": atm_iv,
            "rv": rv,
            "iv_rv_ratio": ratio,
            "thresholds_fired": thresholds_fired,
        },
    )
