"""NAV drawdown — current & maximum peak-to-trough decline, plus the
honestly-labelled reconstructed-book variant (risk spec §5 Tier 1, §45;
Phase B design contract §2.8).

Pure stdlib, deterministic, no I/O. SHADOW/RESEARCH: nothing here alters a
Tier 0 decision (``risk/engine.py`` stays byte-identical).

Input is a NAV path ``[(date, nav), ...]`` in USD, strictly increasing by
date (contract §1: bars/dates out of order are malformed input ⇒
``ValueError``). NAV must be ``> 0`` — a drawdown is a *ratio* to a running
peak, so a zero or negative NAV has no percentage interpretation and is
likewise malformed.

Estimator (contract §2.8 — every number hand-checkable)::

    running_max_t = max(nav_0 … nav_t)          # the peak SO FAR (inclusive)
    dd_t          = nav_t / running_max_t − 1   # ≤ 0 always, a FRACTION

- ``current_dd_pct = dd_{n−1}`` — today's decline from the highest NAV ever
  seen up to today.
- ``max_dd_pct = min_t dd_t`` — the worst such decline over the path. Ties
  are resolved by **date order: the EARLIEST date attaining the minimum
  wins** (stable and deterministic, matching the tie rule used across the
  library).
- ``peak_date`` / ``peak_nav`` are the running peak *in force at the
  trough* — i.e. the peak the max drawdown is measured FROM, not the global
  maximum of the whole path (they coincide unless a later, higher peak was
  set after the trough). ``trough_date`` is where ``max_dd_pct`` occurs.

Sign convention: drawdowns are reported as **fractions ≤ 0** (−0.20 is a
20% decline), *not* percent units and *not* flipped positive — the caller
(UI/snapshot) formats. A flat-or-rising path gives ``0.0``, which is the
true value, not a missing one.

Health (contract §1 honest nulls): ``n < min_obs`` (default 2 — one point
cannot show a decline) ⇒ ``UNAVAILABLE`` with ``value=None``-style empty
fields and a ``reason`` carrying the real numbers; ``min_obs ≤ n <
degraded_multiple × min_obs`` ⇒ ``DEGRADED`` (a two-point "max drawdown" is
arithmetic, not evidence); otherwise ``ACTIVE``. Missing data never raises;
only malformed input does.

``reconstructed_book_drawdown(pnl, nav_now)`` answers a DIFFERENT question
and says so in its label. It walks TODAY's book backwards over its own P&L
history to build a hypothetical NAV path::

    nav_T = nav_now                       # today, the anchor
    nav_t = nav_now − Σ_{u > t} pnl_u     # undo later P&L to step back

then runs the same drawdown estimator on it. This is **not** a real NAV
history: it holds today's positions fixed over the whole window, so it
ignores every trade actually made and every position since closed. It is
labelled ``method="RECONSTRUCTED_CURRENT_BOOK"`` (vs ``"NAV_PATH"``) so a
reader can never mistake it for realised account history (spec §45).

Every result carries a ``ModelMeta`` (``model_version="1.0.0"``; bump per
contract §4) so the number is reproducible (spec §44).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .base import ModelHealth, ModelMeta, ModelTier

#: Model name / version recorded in ``ModelMeta`` (contract §4).
MODEL_VERSION = "1.0.0"
MODEL_NAME = "drawdown"
MODEL_NAME_RECONSTRUCTED = "reconstructed_book_drawdown"

#: ``method`` labels (contract §2.8). The reconstructed variant is a
#: hypothetical path over today's book, never realised account history.
METHOD_NAV_PATH = "NAV_PATH"
METHOD_RECONSTRUCTED = "RECONSTRUCTED_CURRENT_BOOK"


# ---------------------------------------------------------------------------
# Parameters & result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownParams:
    """Every threshold of this module (house rule: never a magic number).

    - ``min_obs`` (2): fewer points cannot express a decline ⇒
      ``UNAVAILABLE`` (contract §2.8: ``n < 2`` ⇒ UNAVAILABLE);
    - ``degraded_multiple`` (2.0): ``n < degraded_multiple × min_obs``
      (i.e. fewer than 4 points by default) ⇒ ``DEGRADED`` — a drawdown off
      a handful of points is arithmetic, not evidence.
    """

    min_obs: int = 2
    degraded_multiple: float = 2.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_obs, bool)
            or not isinstance(self.min_obs, int)
            or self.min_obs < 2
        ):
            raise ValueError(f"min_obs must be an int >= 2, got {self.min_obs!r}")
        if not (self.degraded_multiple >= 1.0):
            raise ValueError(
                f"degraded_multiple must be >= 1, got {self.degraded_multiple}"
            )


DEFAULT_PARAMS = DrawdownParams()


@dataclass(frozen=True)
class DrawdownResult:
    """Drawdown of one NAV path (contract §2.8).

    ``current_dd_pct`` / ``max_dd_pct`` are fractions ``≤ 0`` (−0.2 = −20%),
    or ``None`` when the path is too short to have a drawdown at all. The
    date/NAV fields are ``None`` in that same case. ``method`` says which
    question was answered (``NAV_PATH`` vs ``RECONSTRUCTED_CURRENT_BOOK``).
    """

    current_dd_pct: float | None
    max_dd_pct: float | None
    peak_date: date | None
    trough_date: date | None
    peak_nav: float | None
    current_nav: float | None
    n_obs: int
    method: str
    health: ModelHealth
    reason: str | None
    meta: ModelMeta

    def __post_init__(self) -> None:
        health = ModelHealth(self.health)
        object.__setattr__(self, "health", health)
        if health is not ModelHealth.ACTIVE and not self.reason:
            raise ValueError(f"health={health} requires a non-empty reason")
        if self.method not in (METHOD_NAV_PATH, METHOD_RECONSTRUCTED):
            raise ValueError(
                f"method must be {METHOD_NAV_PATH} or {METHOD_RECONSTRUCTED}, "
                f"got {self.method!r}"
            )
        if isinstance(self.n_obs, bool) or not isinstance(self.n_obs, int) or self.n_obs < 0:
            raise ValueError(f"n_obs must be an int >= 0, got {self.n_obs!r}")
        for name in ("current_dd_pct", "max_dd_pct", "peak_nav", "current_nav"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name} must be finite or None, got {v!r}")
        # Drawdowns are declines: never positive (contract §2.8 dd_t ≤ 0).
        for name in ("current_dd_pct", "max_dd_pct"):
            v = getattr(self, name)
            if v is not None and v > 0.0:
                raise ValueError(f"{name} must be <= 0, got {v}")

    @property
    def is_available(self) -> bool:
        """True iff a drawdown was computed (ACTIVE or DEGRADED)."""
        return self.max_dd_pct is not None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _meta(
    name: str,
    *,
    params: dict[str, Any],
    n: int,
    as_of: date | None,
    method: str,
) -> ModelMeta:
    return ModelMeta(
        model_name=name,
        model_version=MODEL_VERSION,
        params={**params, "method": method},
        return_type=None,
        frequency="1D",
        lookback=n if n >= 1 else None,
        data_source=None,
        as_of=as_of,
        confidence=None,
        horizon_days=1,
        distribution=None,
        # §5: a deterministic path statistic over realised NAV.
        tier=ModelTier.TIER_1,
    )


def _check_nav_path(nav: Sequence[tuple[date, float]]) -> tuple[list[date], list[float]]:
    """Validate and split a NAV path. Malformed input ⇒ ``ValueError``."""
    dates: list[date] = []
    values: list[float] = []
    previous: date | None = None
    for i, point in enumerate(nav):
        try:
            when, value = point
        except (TypeError, ValueError):
            raise ValueError(
                f"nav[{i}] must be a (date, nav) pair, got {point!r}"
            ) from None
        if not isinstance(when, date):
            raise ValueError(f"nav[{i}] date must be a datetime.date, got {when!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"nav[{i}] value must be a float, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"nav[{i}] value must be finite, got {value!r}")
        if value <= 0.0:
            raise ValueError(
                f"nav[{i}] value must be > 0 (a drawdown is a ratio to a peak), "
                f"got {value}"
            )
        if previous is not None and when <= previous:
            raise ValueError(
                f"nav dates must be strictly increasing, got {previous} then {when} "
                f"at index {i}"
            )
        previous = when
        dates.append(when)
        values.append(value)
    return dates, values


def _unavailable(
    *,
    name: str,
    method: str,
    n: int,
    as_of: date | None,
    reason: str,
    params: DrawdownParams,
) -> DrawdownResult:
    return DrawdownResult(
        current_dd_pct=None,
        max_dd_pct=None,
        peak_date=None,
        trough_date=None,
        peak_nav=None,
        current_nav=None,
        n_obs=n,
        method=method,
        health=ModelHealth.UNAVAILABLE,
        reason=reason,
        meta=_meta(
            name,
            params={"min_obs": params.min_obs},
            n=n,
            as_of=as_of,
            method=method,
        ),
    )


def _drawdown_from_path(
    dates: Sequence[date],
    values: Sequence[float],
    *,
    name: str,
    method: str,
    params: DrawdownParams,
) -> DrawdownResult:
    """Core estimator shared by both public entry points."""
    n = len(values)
    as_of = dates[-1] if dates else None

    if n < params.min_obs:
        return _unavailable(
            name=name,
            method=method,
            n=n,
            as_of=as_of,
            reason=f"n={n} < min_obs={params.min_obs}",
            params=params,
        )

    # Running peak and the drawdown at every point (contract §2.8).
    running_max = values[0]
    running_max_date = dates[0]
    # Peak in force at the current worst point, and where that worst is.
    max_dd = 0.0
    trough_index = 0
    peak_at_trough = values[0]
    peak_date_at_trough = dates[0]
    current_dd = 0.0

    for t in range(n):
        if values[t] > running_max:
            running_max = values[t]
            running_max_date = dates[t]
        dd_t = values[t] / running_max - 1.0
        if t == n - 1:
            current_dd = dd_t
        # Strict `<` keeps the EARLIEST date on ties (documented tie rule).
        if dd_t < max_dd:
            max_dd = dd_t
            trough_index = t
            peak_at_trough = running_max
            peak_date_at_trough = running_max_date

    # A never-declining path has max_dd == 0.0 at t=0 by construction; the
    # peak/trough then both sit on the first observation, which is honest.
    health = ModelHealth.ACTIVE
    reason: str | None = None
    degraded_below = params.degraded_multiple * params.min_obs
    if n < degraded_below:
        health = ModelHealth.DEGRADED
        reason = (
            f"short NAV path: n={n} < {degraded_below:g} "
            f"({params.degraded_multiple:g} x min_obs={params.min_obs})"
        )

    return DrawdownResult(
        current_dd_pct=current_dd,
        max_dd_pct=max_dd,
        peak_date=peak_date_at_trough,
        trough_date=dates[trough_index],
        peak_nav=peak_at_trough,
        current_nav=values[-1],
        n_obs=n,
        method=method,
        health=health,
        reason=reason,
        meta=_meta(
            name,
            params={
                "min_obs": params.min_obs,
                "degraded_multiple": params.degraded_multiple,
            },
            n=n,
            as_of=as_of,
            method=method,
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def drawdown(
    nav: Sequence[tuple[date, float]],
    *,
    params: DrawdownParams = DEFAULT_PARAMS,
) -> DrawdownResult:
    """Current & maximum drawdown of a real NAV path (contract §2.8).

    ``nav`` is ``[(date, nav_usd), ...]``, strictly increasing by date with
    every NAV ``> 0`` (else ``ValueError``). Returns fractions ``≤ 0``::

        dd_t       = nav_t / max(nav_0 … nav_t) − 1
        current_dd = dd_{n−1}
        max_dd     = min_t dd_t        (earliest date wins a tie)

    ``peak_date``/``peak_nav`` are the peak the max drawdown is measured
    from (the running peak in force at the trough), so
    ``max_dd_pct == current_nav_at_trough / peak_nav − 1`` by construction.

    ``n < params.min_obs`` (2) ⇒ ``UNAVAILABLE`` with empty fields and a
    reason carrying the real numbers — never a fabricated ``0.0``.
    """
    dates, values = _check_nav_path(nav)
    return _drawdown_from_path(
        dates,
        values,
        name=MODEL_NAME,
        method=METHOD_NAV_PATH,
        params=params,
    )


def reconstructed_book_drawdown(
    pnl: Sequence[tuple[date, float]] | Sequence[float],
    nav_now: float,
    *,
    dates: Sequence[date] | None = None,
    params: DrawdownParams = DEFAULT_PARAMS,
) -> DrawdownResult:
    """Drawdown of a HYPOTHETICAL NAV path rebuilt from today's book
    (contract §2.8; labelled ``RECONSTRUCTED_CURRENT_BOOK``).

    ``pnl`` is today's book's P&L series (USD per day, gain-positive) —
    either ``[(date, pnl), ...]`` or plain floats plus a ``dates``
    sequence. ``nav_now`` (> 0) anchors the path at the LAST observation;
    earlier NAVs are recovered by undoing later P&L::

        nav_T = nav_now
        nav_t = nav_now − Σ_{u > t} pnl_u

    so that ``nav_t − nav_{t−1} == pnl_t`` exactly. The same estimator as
    :func:`drawdown` then runs on that path.

    **This is not account history.** It holds today's positions fixed across
    the whole window, ignoring every trade actually made and every position
    since closed; it answers "how would today's book have ridden the last
    N days", which is a research question, not a realised P&L statement.
    The ``method`` field and ``meta.params["method"]`` both say so.

    A reconstructed NAV that would go ``≤ 0`` (the book's cumulative P&L
    exceeds today's NAV) is malformed for a ratio-based drawdown and raises
    ``ValueError`` rather than reporting a meaningless percentage.
    """
    if isinstance(nav_now, bool) or not isinstance(nav_now, (int, float)):
        raise ValueError(f"nav_now must be a float, got {nav_now!r}")
    nav_now = float(nav_now)
    if not math.isfinite(nav_now) or nav_now <= 0.0:
        raise ValueError(f"nav_now must be finite and > 0, got {nav_now}")

    # -- normalise the two accepted input shapes --------------------------
    pnl_dates: list[date] = []
    pnl_values: list[float] = []
    if dates is not None:
        if len(dates) != len(pnl):
            raise ValueError(
                f"len(dates)={len(dates)} != len(pnl)={len(pnl)}"
            )
        for i, (when, value) in enumerate(zip(dates, pnl)):
            if not isinstance(when, date):
                raise ValueError(f"dates[{i}] must be a datetime.date, got {when!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"pnl[{i}] must be a float, got {value!r}")
            pnl_dates.append(when)
            pnl_values.append(float(value))
    else:
        for i, point in enumerate(pnl):
            try:
                when, value = point  # type: ignore[misc]
            except (TypeError, ValueError):
                raise ValueError(
                    f"pnl[{i}] must be a (date, pnl) pair when dates= is not given, "
                    f"got {point!r}"
                ) from None
            if not isinstance(when, date):
                raise ValueError(f"pnl[{i}] date must be a datetime.date, got {when!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"pnl[{i}] value must be a float, got {value!r}")
            pnl_dates.append(when)
            pnl_values.append(float(value))

    for i, value in enumerate(pnl_values):
        if not math.isfinite(value):
            raise ValueError(f"pnl[{i}] must be finite, got {value!r}")
    previous: date | None = None
    for i, when in enumerate(pnl_dates):
        if previous is not None and when <= previous:
            raise ValueError(
                f"pnl dates must be strictly increasing, got {previous} then {when} "
                f"at index {i}"
            )
        previous = when

    n = len(pnl_values)
    if n < params.min_obs:
        return _unavailable(
            name=MODEL_NAME_RECONSTRUCTED,
            method=METHOD_RECONSTRUCTED,
            n=n,
            as_of=pnl_dates[-1] if pnl_dates else None,
            reason=f"n={n} < min_obs={params.min_obs}",
            params=params,
        )

    # nav_t = nav_now − Σ_{u > t} pnl_u, built backwards from the anchor so
    # the last point is exactly nav_now (no accumulated rounding drift there).
    nav_values = [0.0] * n
    running = nav_now
    for t in range(n - 1, -1, -1):
        nav_values[t] = running
        running = running - pnl_values[t]
    for t, value in enumerate(nav_values):
        if value <= 0.0:
            raise ValueError(
                f"reconstructed nav[{t}] = {value:g} <= 0 on {pnl_dates[t]}: "
                f"cumulative P&L exceeds nav_now={nav_now:g}; a ratio drawdown "
                f"has no meaning on a non-positive NAV"
            )

    return _drawdown_from_path(
        pnl_dates,
        nav_values,
        name=MODEL_NAME_RECONSTRUCTED,
        method=METHOD_RECONSTRUCTED,
        params=params,
    )


__all__ = [
    "DEFAULT_PARAMS",
    "METHOD_NAV_PATH",
    "METHOD_RECONSTRUCTED",
    "MODEL_NAME",
    "MODEL_NAME_RECONSTRUCTED",
    "MODEL_VERSION",
    "DrawdownParams",
    "DrawdownResult",
    "drawdown",
    "reconstructed_book_drawdown",
]
