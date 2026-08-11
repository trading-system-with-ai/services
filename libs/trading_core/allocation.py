"""Volatility-targeted exposure scaling (development plan §14).

Pure, deterministic, dependency-free. §14 sizes exposure so the portfolio
targets a constant annualized volatility: when forecast volatility is HIGH
the exposure multiplier shrinks below 1, when it is LOW the multiplier may
rise above 1 — but the upward leverage is capped (§14 caps the multiplier)
and the result only ever scales a risk BUDGET.

CRITICAL (§14, §44 rule 20): vol targeting must NEVER override hard risk
caps. The multiplier produced here feeds
:func:`libs.trading_core.risk.engine.assess` as ``budget_multiplier``, which
applies it BEFORE ``abs_max_trade_risk`` — the absolute per-trade ceiling,
single-name / bucket / heat / cash-floor clamps and the portfolio greek
limits are all enforced downstream regardless of this multiplier.

Every threshold is a parameter on :class:`VolTargetParams`, never a
hardcoded truth.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VolTargetParams:
    """Tunable vol-targeting parameters (plan §14).

    - ``target_vol``: annualized portfolio volatility target (0.12 = 12%),
      matched against the forecast realized vol
      (:func:`libs.trading_core.features.realized_vol`).
    - ``max_multiplier``: hard cap on UPWARD scaling in calm markets (§14:
      cap leverage; low forecast vol must not balloon position sizes).
    - ``min_multiplier``: floor on downward scaling so a vol spike shrinks
      sizing sanely instead of collapsing it to zero — position EXIT is the
      exit engine's job (plan §15), not the sizer's.
    """

    target_vol: float = 0.12
    max_multiplier: float = 1.2
    min_multiplier: float = 0.25

    def __post_init__(self) -> None:
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be > 0, got {self.target_vol}")
        if self.min_multiplier <= 0:
            raise ValueError(
                f"min_multiplier must be > 0, got {self.min_multiplier}"
            )
        if self.max_multiplier < self.min_multiplier:
            raise ValueError(
                f"max_multiplier {self.max_multiplier} must be >= "
                f"min_multiplier {self.min_multiplier}"
            )


def exposure_multiplier(
    forecast_vol: float | None,
    params: VolTargetParams = VolTargetParams(),
) -> float:
    """Exposure multiplier ``clamp(target_vol / forecast_vol)`` (plan §14).

    - ``forecast_vol`` is the annualized forecast (e.g. 20-day realized vol);
      ``None`` or ``<= 0`` means there is NO usable forecast, so the honest
      default is 1.0 — no adjustment — never a guessed scale-up or -down.
    - Otherwise ``target_vol / forecast_vol`` clamped to
      ``[min_multiplier, max_multiplier]``: forecast above target shrinks
      exposure, forecast below target grows it only up to ``max_multiplier``.

    The result scales a risk budget upstream of the hard caps; it can never
    raise ``abs_max_trade_risk`` or any other limit (§14, §44 rule 20).
    """
    if forecast_vol is None or forecast_vol <= 0:
        return 1.0
    raw = params.target_vol / forecast_vol
    return min(max(raw, params.min_multiplier), params.max_multiplier)
