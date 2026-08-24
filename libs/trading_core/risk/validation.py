"""Walk-forward VaR / ES backtest & volatility forecast error (risk spec
§42, §43, §68; Phase B design contract §2.10).

Pure stdlib, deterministic, no I/O. This module VALIDATES risk forecasts —
it is not a parameter search (audit §9: walk-forward here is framed as
model validation, never hindsight calibration). Everything is
SHADOW/RESEARCH: nothing here alters a Tier 0 decision.

Conventions (contract §1):

- A **P&L series** ``pnl[t] > 0`` is a gain; the **loss** is ``L_t = -pnl_t``.
  VaR / ES forecasts are **losses (positive = money lost)** in USD per
  horizon, exactly as ``risk/models/var_es.py`` reports them.
- **Walk-forward** (spec §43): the forecast for index ``t`` is computed by
  the caller-supplied ``estimator`` on ``pnl[t - window : t]`` ONLY — never
  ``pnl[t]`` nor anything later (contract §3 invariant 5). The estimator is
  any ``Callable[[Sequence[float]], ModelResult]`` (e.g. a lambda over
  ``historical_var``); this module does not import a specific estimator.
- **Exceedance** on day ``t``: realized loss STRICTLY greater than the
  forecast, ``-pnl_t > VaR_t``. Days whose forecast is ``None``
  (estimator UNAVAILABLE / FAILED) are SKIPPED and counted in
  ``n_skipped`` — they are neither hits nor misses.

Estimators (hand-checkable):

- **Kupiec POF** (unconditional coverage), ``p = 1 - α``, ``n`` forecasts,
  ``x`` exceedances::

      LR_uc = -2·ln[(1-p)^(n-x) · p^x] + 2·ln[(1-x/n)^(n-x) · (x/n)^x]

  with the ``0·ln 0 = 0`` convention (so ``x = 0`` and ``x = n`` are
  finite). ``LR_uc ~ χ²(1)``; the p-value is the closed form
  ``p = erfc(sqrt(LR/2))`` (no incomplete gamma). The test is TWO-SIDED:
  far too FEW exceedances (an over-conservative model) fails it as well as
  too many — see ``ExceedanceReport.rate`` vs ``expected_rate`` for the
  direction.
- **Christoffersen independence** (clustering): 2-state Markov chain over
  the ``n - 1`` consecutive hit/no-hit transitions ``n00, n01, n10, n11``;
  ``π01 = n01/(n00+n01)``, ``π11 = n11/(n10+n11)``,
  ``π = (n01+n11)/(n-1)``::

      LR_ind = -2·ln[(1-π)^(n00+n10) · π^(n01+n11)]
               + 2·ln[(1-π01)^n00 · π01^n01 · (1-π11)^n10 · π11^n11]

  again with ``0·ln 0 = 0`` (an undefined ``π11`` when there is no
  exceedance to transition FROM contributes 0). ``χ²(1)``, same closed
  form. ``clustered = christoffersen_p < params.cluster_p``.
- **Conditional coverage** (Christoffersen 1998) ``LR_cc = LR_uc + LR_ind
  ~ χ²(2)``, closed-form ``p = exp(-LR/2)``. Reported as a convenience;
  the verdict uses Kupiec alone (contract §2.10).
- **ES severity ratio** = mean realized loss on exceedance days ÷ mean
  forecast ES on the SAME days (only days where an ES forecast exists);
  ``None`` if there are no exceedances or no ES forecasts. ``> 1`` means
  realized tail losses were worse than the ES forecast.
- **Verdict** (Basel traffic-light style, by Kupiec p-value; parameters
  ``BacktestParams.green_p`` / ``red_p``): ``GREEN`` if
  ``p ≥ green_p`` (0.05), ``YELLOW`` if ``red_p ≤ p < green_p``, ``RED``
  if ``p < red_p`` (0.01); ``UNAVAILABLE`` when ``n < min_forecasts``.
- **Volatility forecast error** (spec §42) for a σ forecast made BEFORE the
  return it forecasts: ``MSE = mean((r_t² - σ_t²)²)`` (variance-scale
  mean squared error) and Patton's normalised
  ``QLIKE = mean(r_t²/σ_t² - ln(r_t²/σ_t²) - 1)`` — ``0`` for a perfect
  forecast, always ``≥ 0``, robust to noise in the ``r²`` proxy. QLIKE is
  undefined at ``r_t = 0`` exactly (``ln 0``): such days are EXCLUDED from
  the QLIKE mean only, counted in ``n_zero_returns`` and flagged DEGRADED
  with the real numbers.

Honest nulls (contract §1): too few forecasts ⇒ ``verdict=UNAVAILABLE``,
``health=UNAVAILABLE`` and a ``reason`` with the real numbers; statistics
that cannot be computed are ``None``, never 0. Malformed input (mismatched
lengths, non-finite numbers, ``confidence`` outside ``(0.5, 1)``,
non-positive σ) raises ``ValueError``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from libs.trading_core.risk.models.base import ModelHealth, ModelResult

#: Module (estimator) version — contract §4: arithmetic change bumps MAJOR.
VALIDATION_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _xlogy(x: float, y: float) -> float:
    """``x·ln(y)`` with the ``0·ln 0 = 0`` convention (Kupiec / Markov LR terms)."""
    if x == 0:
        return 0.0
    if y <= 0.0:
        raise ValueError(f"log of non-positive argument with non-zero weight: x={x}, y={y}")
    return x * math.log(y)


def chi2_1_sf(lr: float) -> float:
    """χ²(1) survival function ``P(X > lr) = erfc(sqrt(lr/2))`` (closed form)."""
    if lr < 0.0:
        # Rounding can produce a tiny negative LR when the two log-likelihoods
        # coincide; treat as 0 (p = 1). A materially negative LR is a bug.
        if lr < -1e-9:
            raise ValueError(f"likelihood ratio must be >= 0, got {lr}")
        lr = 0.0
    return math.erfc(math.sqrt(lr / 2.0))


def chi2_2_sf(lr: float) -> float:
    """χ²(2) survival function ``P(X > lr) = exp(-lr/2)`` (closed form)."""
    if lr < 0.0:
        if lr < -1e-9:
            raise ValueError(f"likelihood ratio must be >= 0, got {lr}")
        lr = 0.0
    return math.exp(-lr / 2.0)


def _check_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a float in (0.5, 1), got {confidence!r}")
    if not (0.5 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    return float(confidence)


def _check_finite_seq(values: Sequence[float], name: str) -> list[float]:
    out: list[float] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}[{i}] must be a finite number, got {v!r}")
        out.append(float(v))
    return out


def _check_optional_seq(values: Sequence[float | None], name: str) -> list[float | None]:
    out: list[float | None] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}[{i}] must be a finite number or None, got {v!r}")
        out.append(float(v))
    return out


# ---------------------------------------------------------------------------
# Walk-forward forecasts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastSeries:
    """Walk-forward forecasts aligned with the realized P&L they forecast.

    Position ``i`` of every tuple refers to the same target index
    ``indices[i] = t`` of the input ``pnl``:

    - ``forecasts[i]`` = ``results[i].value`` — the forecast (loss-positive
      USD for VaR/ES estimators) for ``pnl[t]``, computed on
      ``pnl[t-window:t]``; ``None`` where the estimator was
      UNAVAILABLE/FAILED;
    - ``realized[i]`` = ``pnl[t]`` (gain-positive, contract §1);
    - ``results[i]`` = the full ``ModelResult`` (health, reason, meta);
    - ``dates[i]`` = the date of ``pnl[t]`` when the caller supplied dates.

    ``confidence`` is the confidence the estimator ran at (given by the
    caller or inferred from the results' ``meta.confidence``); ``None`` if
    unknown, in which case ``exceedances`` needs it explicitly.
    """

    indices: tuple[int, ...]
    forecasts: tuple[float | None, ...]
    realized: tuple[float, ...]
    results: tuple[ModelResult, ...]
    window: int
    confidence: float | None = None
    dates: tuple[date, ...] | None = None
    estimator_name: str | None = None
    version: str = VALIDATION_VERSION

    def __post_init__(self) -> None:
        n = len(self.indices)
        if not (len(self.forecasts) == len(self.realized) == len(self.results) == n):
            raise ValueError("ForecastSeries fields must have equal length")
        if self.dates is not None and len(self.dates) != n:
            raise ValueError("dates must align with forecasts")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _check_confidence(self.confidence))

    @property
    def n_forecasts(self) -> int:
        """Number of target indices forecast (``len(pnl) - window`` when positive)."""
        return len(self.indices)

    @property
    def n_available(self) -> int:
        """Forecasts that produced a number (ACTIVE or DEGRADED)."""
        return sum(1 for f in self.forecasts if f is not None)

    @property
    def n_unavailable(self) -> int:
        return self.n_forecasts - self.n_available

    @property
    def as_of(self) -> date | None:
        return self.dates[-1] if self.dates else None


def walk_forward(
    pnl: Sequence[float],
    *,
    window: int,
    estimator: Callable[[Sequence[float]], ModelResult],
    confidence: float | None = None,
    dates: Sequence[date] | None = None,
) -> ForecastSeries:
    """Walk-forward forecasts: for every ``t`` in ``[window, len(pnl))`` call
    ``estimator(pnl[t-window:t])`` and pair the result with ``pnl[t]``.

    The estimator NEVER sees ``pnl[t]`` or later (spec §43; contract §3
    invariant 5) — the slice is a fresh ``tuple`` of exactly ``window``
    observations, so an estimator cannot reach outside it. Estimator
    results with ``value=None`` are kept (health/reason preserved) and
    show as ``None`` in ``forecasts``.

    - ``window >= 1``; ``len(pnl) <= window`` is NOT an error — it yields
      an empty ``ForecastSeries`` (``n_forecasts = 0``) which
      ``exceedances`` reports as UNAVAILABLE.
    - ``confidence`` (in ``(0.5, 1)``) is recorded on the series; when
      omitted it is inferred from the results' ``meta.confidence`` if all
      results agree (else left ``None``).
    - ``dates`` (optional) aligns 1:1 with ``pnl``.
    - ``ValueError`` on non-finite P&L, ``window < 1``, misaligned dates,
      or an estimator that does not return a ``ModelResult``.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be an int >= 1, got {window!r}")
    values = _check_finite_seq(pnl, "pnl")
    if dates is not None and len(dates) != len(values):
        raise ValueError(
            f"dates length {len(dates)} must equal pnl length {len(values)}"
        )
    conf = _check_confidence(confidence) if confidence is not None else None

    indices: list[int] = []
    forecasts: list[float | None] = []
    realized: list[float] = []
    results: list[ModelResult] = []
    for t in range(window, len(values)):
        history = tuple(values[t - window : t])  # exactly `window` obs, all < t
        result = estimator(history)
        if not isinstance(result, ModelResult):
            raise ValueError(
                f"estimator must return a ModelResult, got {type(result).__name__} at t={t}"
            )
        indices.append(t)
        forecasts.append(result.value)
        realized.append(values[t])
        results.append(result)

    if conf is None and results:
        confs = {r.meta.confidence for r in results}
        if len(confs) == 1:
            only = next(iter(confs))
            if only is not None:
                conf = float(only)
    names = {r.meta.model_name for r in results}
    estimator_name = next(iter(names)) if len(names) == 1 else None

    return ForecastSeries(
        indices=tuple(indices),
        forecasts=tuple(forecasts),
        realized=tuple(realized),
        results=tuple(results),
        window=window,
        confidence=conf,
        dates=tuple(dates[window:]) if dates is not None else None,
        estimator_name=estimator_name,
    )


# ---------------------------------------------------------------------------
# Exceedance backtest (Kupiec / Christoffersen / ES severity)
# ---------------------------------------------------------------------------


class BacktestVerdict(StrEnum):
    """Basel-style traffic light by Kupiec p-value; UNAVAILABLE = too few forecasts."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class BacktestParams:
    """Every backtest threshold is a documented parameter (house rule).

    - ``min_forecasts`` (250): fewer usable forecast/realized pairs ⇒
      ``UNAVAILABLE`` (a Kupiec test at 99% on fewer than a year of days
      has almost no power — 250 × 1% = 2.5 expected hits).
    - ``green_p`` (0.05) / ``red_p`` (0.01): Kupiec p-value cut-offs for
      GREEN / YELLOW / RED (``red_p < green_p``).
    - ``cluster_p`` (0.05): ``clustered = christoffersen_p < cluster_p``.
    """

    min_forecasts: int = 250
    green_p: float = 0.05
    red_p: float = 0.01
    cluster_p: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.min_forecasts, bool) or not isinstance(self.min_forecasts, int) or self.min_forecasts < 1:
            raise ValueError(f"min_forecasts must be an int >= 1, got {self.min_forecasts!r}")
        for name in ("green_p", "red_p", "cluster_p"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {v!r}")
        if not self.red_p < self.green_p:
            raise ValueError(
                f"red_p ({self.red_p}) must be < green_p ({self.green_p})"
            )


DEFAULT_BACKTEST_PARAMS = BacktestParams()


@dataclass(frozen=True)
class ExceedanceReport:
    """Result of ``exceedances`` (contract §2.10). ``None`` = not computable.

    - ``n``: usable forecast/realized pairs (forecast not ``None``);
      ``n_skipped``: pairs dropped because the forecast was ``None``.
    - ``x``: exceedances (``-pnl_t > VaR_t``); ``rate = x/n``;
      ``expected_rate = 1 - confidence``.
    - ``kupiec_lr`` / ``kupiec_p``: unconditional coverage (χ²(1)).
    - ``christoffersen_lr`` / ``christoffersen_p``: independence (χ²(1));
      ``clustered = christoffersen_p < params.cluster_p``.
    - ``conditional_coverage_lr`` / ``_p``: ``LR_uc + LR_ind`` (χ²(2)).
    - ``transitions``: ``(n00, n01, n10, n11)`` so the Markov LR can be
      hand-checked.
    - ``es_severity_ratio``: mean realized loss on exceedance days ÷ mean
      forecast ES on those days; ``es_n`` = days that entered that mean.
    - ``exceedance_positions``: positions ``i`` (into the forecast series)
      of the exceedances, in order.
    - ``verdict`` / ``health`` / ``reason``: traffic light + honest health.
    """

    n: int
    x: int
    rate: float | None
    expected_rate: float
    confidence: float
    kupiec_lr: float | None
    kupiec_p: float | None
    christoffersen_lr: float | None
    christoffersen_p: float | None
    clustered: bool | None
    conditional_coverage_lr: float | None
    conditional_coverage_p: float | None
    transitions: tuple[int, int, int, int] | None
    es_severity_ratio: float | None
    es_n: int
    exceedance_positions: tuple[int, ...]
    verdict: BacktestVerdict
    health: ModelHealth
    reason: str | None
    n_skipped: int = 0
    params: BacktestParams = field(default_factory=BacktestParams)
    version: str = VALIDATION_VERSION

    @property
    def is_available(self) -> bool:
        return self.verdict is not BacktestVerdict.UNAVAILABLE


def kupiec_pof(n: int, x: int, expected_rate: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test → ``(LR, p_value)``.

    ``LR = -2·[(n-x)·ln(1-p) + x·ln p] + 2·[(n-x)·ln(1-x/n) + x·ln(x/n)]``
    with ``0·ln 0 = 0``; ``p_value = erfc(sqrt(LR/2))`` (χ²(1)).
    ``ValueError`` if ``n < 1``, ``x`` outside ``[0, n]`` or ``p`` outside
    ``(0, 1)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0 <= x <= n):
        raise ValueError(f"x must be in [0, n={n}], got {x}")
    if not (0.0 < expected_rate < 1.0):
        raise ValueError(f"expected_rate must be in (0, 1), got {expected_rate}")
    p = expected_rate
    pi = x / n
    ll_null = _xlogy(n - x, 1.0 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1.0 - pi) + _xlogy(x, pi)
    lr = -2.0 * ll_null + 2.0 * ll_alt
    if lr < 0.0 and lr > -1e-9:
        lr = 0.0
    return lr, chi2_1_sf(lr)


def markov_transitions(hits: Sequence[bool]) -> tuple[int, int, int, int]:
    """Count consecutive transitions ``(n00, n01, n10, n11)`` of a hit sequence."""
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(hits, hits[1:]):
        if prev:
            if cur:
                n11 += 1
            else:
                n10 += 1
        else:
            if cur:
                n01 += 1
            else:
                n00 += 1
    return n00, n01, n10, n11


def christoffersen_independence(hits: Sequence[bool]) -> tuple[float, float, tuple[int, int, int, int]]:
    """Christoffersen (1998) independence test → ``(LR, p_value, transitions)``.

    2-state Markov LR (see module docstring) with ``0·ln 0 = 0``; χ²(1)
    closed-form p-value. Requires at least 2 observations (one
    transition); ``ValueError`` otherwise.
    """
    if len(hits) < 2:
        raise ValueError(f"independence test needs >= 2 observations, got {len(hits)}")
    n00, n01, n10, n11 = markov_transitions(hits)
    n_from0 = n00 + n01
    n_from1 = n10 + n11
    total = n_from0 + n_from1
    pi = (n01 + n11) / total
    pi01 = n01 / n_from0 if n_from0 else 0.0
    pi11 = n11 / n_from1 if n_from1 else 0.0
    ll_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (
        _xlogy(n00, 1.0 - pi01)
        + _xlogy(n01, pi01)
        + _xlogy(n10, 1.0 - pi11)
        + _xlogy(n11, pi11)
    )
    lr = -2.0 * ll_null + 2.0 * ll_alt
    if lr < 0.0 and lr > -1e-9:
        lr = 0.0
    return lr, chi2_1_sf(lr), (n00, n01, n10, n11)


def _forecast_values(
    forecasts: ForecastSeries | Sequence[float | None], name: str
) -> list[float | None]:
    if isinstance(forecasts, ForecastSeries):
        return list(forecasts.forecasts)
    return _check_optional_seq(forecasts, name)


def exceedances(
    forecasts: ForecastSeries | Sequence[float | None],
    realized_pnl: Sequence[float] | None = None,
    *,
    confidence: float | None = None,
    es_forecasts: ForecastSeries | Sequence[float | None] | None = None,
    params: BacktestParams = DEFAULT_BACKTEST_PARAMS,
) -> ExceedanceReport:
    """VaR exceedance backtest (spec §42): Kupiec, Christoffersen, ES severity.

    - ``forecasts``: a ``ForecastSeries`` from ``walk_forward`` or a plain
      sequence of VaR forecasts (loss-positive USD; ``None`` = no forecast).
    - ``realized_pnl``: gain-positive P&L aligned 1:1 with ``forecasts``;
      may be omitted when ``forecasts`` is a ``ForecastSeries`` (its
      ``realized`` is used).
    - ``confidence``: the forecasts' confidence; may be omitted when the
      ``ForecastSeries`` carries it. ``expected_rate = 1 - confidence``.
    - ``es_forecasts``: optional ES forecasts aligned with ``forecasts``
      (``ForecastSeries`` or sequence) for the severity ratio.
    - ``params``: thresholds (``BacktestParams``).

    Pairs whose VaR forecast is ``None`` are skipped (``n_skipped``).
    ``n < params.min_forecasts`` ⇒ ``verdict=UNAVAILABLE``,
    ``health=UNAVAILABLE`` and a reason with the real numbers; ``x``,
    ``rate`` and ``exceedance_positions`` are still reported on whatever
    pairs exist (they are counts, not inferences) but every test statistic
    is ``None``. ``ValueError`` on misaligned lengths, non-finite numbers,
    or a missing/invalid confidence.
    """
    var_values = _forecast_values(forecasts, "forecasts")
    if realized_pnl is None:
        if not isinstance(forecasts, ForecastSeries):
            raise ValueError("realized_pnl is required unless forecasts is a ForecastSeries")
        realized = list(forecasts.realized)
    else:
        realized = _check_finite_seq(realized_pnl, "realized_pnl")
    if len(realized) != len(var_values):
        raise ValueError(
            f"realized_pnl length {len(realized)} must equal forecasts length {len(var_values)}"
        )
    if confidence is None:
        if isinstance(forecasts, ForecastSeries) and forecasts.confidence is not None:
            confidence = forecasts.confidence
        else:
            raise ValueError("confidence is required (not carried by the forecasts)")
    conf = _check_confidence(confidence)
    expected_rate = 1.0 - conf

    es_values: list[float | None] | None = None
    if es_forecasts is not None:
        es_values = _forecast_values(es_forecasts, "es_forecasts")
        if len(es_values) != len(var_values):
            raise ValueError(
                f"es_forecasts length {len(es_values)} must equal forecasts length {len(var_values)}"
            )

    # Usable pairs and hit sequence (strict: loss > VaR).
    hits: list[bool] = []
    positions: list[int] = []
    losses_on_hits: list[float] = []
    es_on_hits: list[float] = []
    n_skipped = 0
    for i, (var_t, pnl_t) in enumerate(zip(var_values, realized)):
        if var_t is None:
            n_skipped += 1
            continue
        loss = -pnl_t
        hit = loss > var_t
        hits.append(hit)
        if hit:
            positions.append(i)
            if es_values is not None and es_values[i] is not None:
                losses_on_hits.append(loss)
                es_on_hits.append(es_values[i])  # type: ignore[arg-type]
    n = len(hits)
    x = len(positions)
    rate = x / n if n else None

    # ES severity ratio (independent of the sample-size gate: plain arithmetic).
    es_ratio: float | None = None
    es_n = len(es_on_hits)
    if es_n:
        mean_loss = math.fsum(losses_on_hits) / es_n
        mean_es = math.fsum(es_on_hits) / es_n
        es_ratio = mean_loss / mean_es if mean_es != 0.0 else None

    if n < params.min_forecasts:
        return ExceedanceReport(
            n=n, x=x, rate=rate, expected_rate=expected_rate, confidence=conf,
            kupiec_lr=None, kupiec_p=None,
            christoffersen_lr=None, christoffersen_p=None, clustered=None,
            conditional_coverage_lr=None, conditional_coverage_p=None,
            transitions=None,
            es_severity_ratio=es_ratio, es_n=es_n,
            exceedance_positions=tuple(positions),
            verdict=BacktestVerdict.UNAVAILABLE,
            health=ModelHealth.UNAVAILABLE,
            reason=f"n={n} < min_forecasts={params.min_forecasts} (skipped={n_skipped})",
            n_skipped=n_skipped, params=params,
        )

    k_lr, k_p = kupiec_pof(n, x, expected_rate)
    c_lr, c_p, transitions = christoffersen_independence(hits)
    cc_lr = k_lr + c_lr
    cc_p = chi2_2_sf(cc_lr)
    clustered = c_p < params.cluster_p

    if k_p >= params.green_p:
        verdict = BacktestVerdict.GREEN
    elif k_p >= params.red_p:
        verdict = BacktestVerdict.YELLOW
    else:
        verdict = BacktestVerdict.RED

    reason: str | None = None
    health = ModelHealth.ACTIVE
    if n_skipped:
        health = ModelHealth.DEGRADED
        reason = f"{n_skipped} of {n + n_skipped} forecasts unavailable and skipped"

    return ExceedanceReport(
        n=n, x=x, rate=rate, expected_rate=expected_rate, confidence=conf,
        kupiec_lr=k_lr, kupiec_p=k_p,
        christoffersen_lr=c_lr, christoffersen_p=c_p, clustered=clustered,
        conditional_coverage_lr=cc_lr, conditional_coverage_p=cc_p,
        transitions=transitions,
        es_severity_ratio=es_ratio, es_n=es_n,
        exceedance_positions=tuple(positions),
        verdict=verdict, health=health, reason=reason,
        n_skipped=n_skipped, params=params,
    )


# ---------------------------------------------------------------------------
# Volatility forecast error (spec §42)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityForecastError:
    """``volatility_forecast_error`` result.

    - ``mse``: ``mean((r² - σ²)²)`` over the ``n`` usable pairs;
    - ``qlike``: ``mean(r²/σ² - ln(r²/σ²) - 1)`` over the ``qlike_n``
      pairs with ``r ≠ 0`` (``None`` if none);
    - ``n``: pairs with a σ forecast; ``n_skipped``: pairs with ``σ=None``;
      ``n_zero_returns``: pairs excluded from QLIKE because ``r = 0``.
    """

    mse: float | None
    qlike: float | None
    n: int
    qlike_n: int
    n_skipped: int
    n_zero_returns: int
    health: ModelHealth
    reason: str | None
    version: str = VALIDATION_VERSION


def volatility_forecast_error(
    forecast_sigmas: Sequence[float | None],
    realized_returns: Sequence[float],
    *,
    min_obs: int = 20,
) -> VolatilityForecastError:
    """Volatility forecast error (spec §42) — MSE on variance and QLIKE.

    ``forecast_sigmas[t]`` is the σ forecast made BEFORE ``realized_returns[t]``
    (walk-forward: e.g. ``ewma_variance`` output, whose entry ``t`` uses
    returns ``< t``); the two are aligned 1:1 and share units (both USD/day
    or both return units). ``None`` forecasts are skipped and counted.
    Estimators::

        MSE   = (1/n) Σ (r_t² - σ_t²)²
        QLIKE = (1/n') Σ_{r_t ≠ 0} (r_t²/σ_t² - ln(r_t²/σ_t²) - 1)

    ``n < min_obs`` ⇒ ``UNAVAILABLE`` (values ``None``); ``ValueError`` on
    misaligned lengths, non-finite inputs or ``σ ≤ 0`` (a non-positive
    forecast σ is malformed, not a data gap).
    """
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 1:
        raise ValueError(f"min_obs must be an int >= 1, got {min_obs!r}")
    sigmas = _check_optional_seq(forecast_sigmas, "forecast_sigmas")
    rets = _check_finite_seq(realized_returns, "realized_returns")
    if len(sigmas) != len(rets):
        raise ValueError(
            f"realized_returns length {len(rets)} must equal forecast_sigmas length {len(sigmas)}"
        )
    sq_err: list[float] = []
    qlike_terms: list[float] = []
    n_skipped = 0
    n_zero = 0
    for i, (sig, r) in enumerate(zip(sigmas, rets)):
        if sig is None:
            n_skipped += 1
            continue
        if sig <= 0.0:
            raise ValueError(f"forecast_sigmas[{i}] must be > 0, got {sig}")
        var_f = sig * sig
        r2 = r * r
        sq_err.append((r2 - var_f) ** 2)
        if r2 == 0.0:
            n_zero += 1
            continue
        ratio = r2 / var_f
        qlike_terms.append(ratio - math.log(ratio) - 1.0)
    n = len(sq_err)
    qlike_n = len(qlike_terms)
    if n < min_obs:
        return VolatilityForecastError(
            mse=None, qlike=None, n=n, qlike_n=qlike_n, n_skipped=n_skipped,
            n_zero_returns=n_zero, health=ModelHealth.UNAVAILABLE,
            reason=f"n={n} < min_obs={min_obs} (skipped={n_skipped})",
        )
    mse = math.fsum(sq_err) / n
    qlike = math.fsum(qlike_terms) / qlike_n if qlike_n else None
    health = ModelHealth.ACTIVE
    reasons: list[str] = []
    if n_skipped:
        reasons.append(f"{n_skipped} of {n + n_skipped} forecasts unavailable and skipped")
    if n_zero:
        reasons.append(f"{n_zero} zero-return days excluded from QLIKE (ln 0 undefined)")
    if reasons:
        health = ModelHealth.DEGRADED
    return VolatilityForecastError(
        mse=mse, qlike=qlike, n=n, qlike_n=qlike_n, n_skipped=n_skipped,
        n_zero_returns=n_zero, health=health,
        reason="; ".join(reasons) if reasons else None,
    )


__all__ = [
    "BacktestParams",
    "BacktestVerdict",
    "DEFAULT_BACKTEST_PARAMS",
    "ExceedanceReport",
    "ForecastSeries",
    "VALIDATION_VERSION",
    "VolatilityForecastError",
    "chi2_1_sf",
    "chi2_2_sf",
    "christoffersen_independence",
    "exceedances",
    "kupiec_pof",
    "markov_transitions",
    "volatility_forecast_error",
    "walk_forward",
]
