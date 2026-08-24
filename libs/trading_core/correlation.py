"""Rolling correlation and dynamic correlation buckets (plan §12.4).

Pure, deterministic, dependency-free — plain-list inputs, hand-checkable
Pearson arithmetic. The static ``RiskLimits.correlation_buckets`` config is
the v0 mechanism; this module provides the DATA-DRIVEN alternative §12.4
calls for: measure rolling correlations of daily log returns and group
tickers whose correlation exceeds a threshold into shared-risk buckets.

Every parameter is explicit and tunable (house rule; §12.4: "thresholds
require validation" — the 0.70 / 60-day defaults are starting points to be
validated against history, never hardcoded truths):

- ``window`` (default 60 trading days): the number of most-recent aligned
  log returns the Pearson correlation is computed over.
- ``threshold`` (default 0.70): tickers with pairwise correlation STRICTLY
  above this share a bucket.

Honest nulls: insufficient history or a zero-variance return series yields
``None``, never a fabricated 0.0 correlation.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

# ``log_returns`` MOVED to the standardized returns layer (risk spec §3;
# Phase B design contract §2.1) and is re-exported here, byte-identical, so
# existing callers keep importing it from ``correlation``.
from libs.trading_core.risk.returns import log_returns  # noqa: F401


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Pearson correlation of two equal-length samples; None on zero variance."""
    n = len(a)
    mean_a = math.fsum(a) / n
    mean_b = math.fsum(b) / n
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    var_a = math.fsum(x * x for x in da)
    var_b = math.fsum(y * y for y in db)
    if var_a <= 0.0 or var_b <= 0.0:
        return None  # zero variance -> correlation undefined (honest null)
    cov = math.fsum(x * y for x, y in zip(da, db))
    return cov / math.sqrt(var_a * var_b)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks ``1..n`` with TIES SHARING THEIR AVERAGE RANK (spec §18).

    ``[10, 20, 20, 30]`` ranks as ``[1, 2.5, 2.5, 4]``. The tie correction is
    a correctness requirement, not a refinement: assigning tied values ranks
    by first-seen order would make the resulting rho a function of the order
    the caller happened to pass the sample in — the same data would produce
    a different number on a different day. Averaging the tied block is the
    standard "fractional ranking" convention and is what makes
    :func:`spearman` equal Pearson-of-ranks in the presence of ties.

    Deliberately duplicated from ``events/event_study.average_ranks`` rather
    than imported: this module is a leaf whose only intra-platform import is
    ``risk.returns`` (see the import-cycle note below), and the risk layer
    must not grow an edge into the Catalyst/events package to compute a rank.
    Nine lines of arithmetic are the cheaper of the two costs, and both
    copies are pinned by hand-checked tie tests.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0  # 1-based average of the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation of two equal-length samples (spec §18).

    Definition: rank-transform BOTH samples with :func:`_average_ranks`
    (ties share their average rank), then take the ordinary Pearson
    correlation of the ranks via the SAME :func:`_pearson` estimator the
    Pearson path uses. This is the general definition, valid with ties —
    NOT the ``1 − 6Σd²/(n(n²−1))`` shortcut, which is only correct when
    there are none.

    Why §18 wants it beside Pearson: Spearman measures MONOTONE association,
    so it is invariant to any monotone re-scaling of either series and is far
    less sensitive to a single outlier day than Pearson, which measures
    LINEAR association. A pair that co-moves reliably but non-linearly reads
    low on Pearson and high on Spearman; the gap between the two is the
    diagnostic.

    Honest nulls, matching :func:`_pearson` exactly: ``None`` when either
    sample is constant (every value tied ⇒ every rank identical ⇒ zero rank
    variance ⇒ the correlation is undefined, never 0.0). ``ValueError`` on a
    length mismatch or on fewer than two observations — malformed input, not
    a data gap.

    SHADOW/RESEARCH (spec §70): a display diagnostic. Nothing here gates.
    """
    if len(a) != len(b):
        raise ValueError(
            f"spearman requires equal-length samples, got {len(a)} and {len(b)}"
        )
    if len(a) < 2:
        raise ValueError(f"spearman requires >= 2 observations, got {len(a)}")
    return _pearson(_average_ranks(a), _average_ranks(b))


def rolling_spearman(
    a_closes: Sequence[float],
    b_closes: Sequence[float],
    window: int = 60,
) -> float | None:
    """Spearman correlation of the last ``window`` daily log returns (§18).

    The rank-correlation mirror of :func:`rolling_correlation`, with the
    same contract: the two close series are assumed bar-aligned at the
    RECENT end, the trailing ``window`` returns of each are compared, and
    the result is ``None`` — honest null, never a fake 0.0 — when either
    series has fewer than ``window + 1`` closes or when either return
    window is constant (undefined rank correlation).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if len(a_closes) < window + 1 or len(b_closes) < window + 1:
        return None
    ra = log_returns(a_closes[-(window + 1):])
    rb = log_returns(b_closes[-(window + 1):])
    return spearman(ra, rb)


def rolling_correlation(
    a_closes: Sequence[float],
    b_closes: Sequence[float],
    window: int = 60,
) -> float | None:
    """Pearson correlation of the last ``window`` daily log returns (§12.4).

    The two close series are assumed bar-aligned at the RECENT end (both
    ending on the current bar); the trailing ``window`` returns of each are
    compared. Returns ``None`` — honest null, never a fake 0.0 — when either
    series has fewer than ``window + 1`` closes (insufficient data) or when
    either return window has zero variance (correlation undefined).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if len(a_closes) < window + 1 or len(b_closes) < window + 1:
        return None
    ra = log_returns(a_closes[-(window + 1):])
    rb = log_returns(b_closes[-(window + 1):])
    return _pearson(ra, rb)


def build_dynamic_buckets(
    closes_by_ticker: Mapping[str, Sequence[float]],
    threshold: float = 0.70,
    window: int = 60,
) -> list[frozenset[str]]:
    """Group tickers into correlation buckets via connected components (§12.4).

    Builds the graph whose edges connect ticker pairs with
    ``rolling_correlation > threshold`` (strict; ``None`` correlations —
    insufficient data or zero variance — never create an edge), then returns
    its connected components. Transitivity is deliberate: if A~B and B~C
    exceed the threshold, {A, B, C} share ONE bucket cap even if A~C does
    not, matching the shared-risk intent of §12.4.

    Singleton components are EXCLUDED — a ticker correlated with nothing
    needs no bucket cap beyond its single-name limits (§12.3).

    Deterministic output regardless of mapping iteration order: members are
    compared in sorted ticker order and buckets are sorted by their sorted
    member tuple.
    """
    tickers = sorted(closes_by_ticker)
    parent: dict[str, str] = {t: t for t in tickers}

    def find(t: str) -> str:
        root = t
        while parent[root] != root:
            root = parent[root]
        while parent[t] != root:  # path compression
            parent[t], t = root, parent[t]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic root choice: smaller ticker wins.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            corr = rolling_correlation(
                closes_by_ticker[a], closes_by_ticker[b], window=window
            )
            if corr is not None and corr > threshold:
                union(a, b)

    components: dict[str, set[str]] = {}
    for t in tickers:
        components.setdefault(find(t), set()).add(t)

    buckets = [
        frozenset(members)
        for members in components.values()
        if len(members) > 1  # singletons excluded
    ]
    buckets.sort(key=lambda bucket: tuple(sorted(bucket)))
    return buckets


# ---------------------------------------------------------------------------
# Correlation regime (risk spec §19; Phase B design contract §7.4) — ADDITIVE
#
# Spec §19: "Portfolio diversification tends to deteriorate in stressed
# markets. Create monitoring for normal correlation / current rolling
# correlation / stress correlation." A tech pair whose NORMAL correlation is
# 0.61 but whose CURRENT correlation is 0.84 has raised the book's
# concentration risk even though not one dollar of position changed — that is
# the state this section measures and names.
#
# SHADOW/RESEARCH (spec §70): the state is computed, logged and displayed;
# nothing here feeds a Tier 0 decision. Rolling Spearman IS now built
# (spec §18, compliance §3 row 18, this batch: ``spearman``,
# ``rolling_spearman``, ``rolling_spearman_matrix``,
# ``rolling_spearman_average`` and ``CorrelationState.current_avg_spearman``)
# and stays RESEARCH: a display diagnostic beside Pearson that enters no
# state rule and gates nothing.
#
# Import-cycle note (contract §7.4, and the note in ``risk/__init__.py``):
# this module is imported BY ``risk`` consumers and itself imports
# ``risk.returns`` for ``log_returns`` — a leaf module with no further risk
# imports. ``ReturnMatrix`` is therefore imported from that SAME leaf, which
# keeps the cycle closed in both directions (``import correlation`` first and
# ``import risk`` first both work); it is bound under TYPE_CHECKING for the
# annotations and duck-typed at runtime through its public attributes.
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import TYPE_CHECKING, ClassVar  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    from libs.trading_core.risk.returns import ReturnMatrix

#: The ``return_type`` a regime matrix must carry (contract §1: LOG for
#: correlation of returns — the convention ``_pearson`` above is written for).
REGIME_RETURN_TYPE = "LOG"

#: Regime state names (contract §7.4).
STATE_NORMAL = "NORMAL"
STATE_ELEVATED = "ELEVATED"
STATE_CONVERGING = "CONVERGING"
STATE_UNAVAILABLE = "UNAVAILABLE"

#: How many worst pairs :class:`CorrelationState` names (display budget).
WORST_PAIRS = 3


@dataclass(frozen=True)
class CorrelationRegimeParams:
    """Every threshold of the regime monitor (house rule: never a hardcoded
    truth). **RESEARCH DEFAULTS — UNVALIDATED** (audit §11 Q3).

    - ``long_window`` (250): observations defining the NORMAL correlation
      level — roughly one trading year;
    - ``short_window`` (60): observations defining the CURRENT level — the
      same 60-day window ``rolling_correlation`` uses above;
    - ``stress_quantile`` (0.10): the fraction of WORST days of the
      equal-weight portfolio return that defines the stress sample;
    - ``elevated_delta`` (0.05): ``current − normal`` at or above this is at
      least ELEVATED;
    - ``converging_level`` (0.80): CONVERGING when the CURRENT average is at
      or above this level — whether it got there by a jump or has sat there
      all along (a book persistently at ρ≈1 has NO diversification; §19
      exists to catch exactly that, so the level alone decides — QA finding
      2026-08-18: the earlier "jump AND level" rule read such a book as
      NORMAL);
    - ``converging_delta`` (0.15): a jump of at least this size is named in
      ``reason`` as a regime SHIFT (spec §19 "0.61 → 0.84") — it annotates,
      it does not gate CONVERGING;
    - ``min_pairs`` (1): fewer computable pairs than this ⇒ UNAVAILABLE;
    - ``min_stress_obs`` (10): fewer stress days than this ⇒ ``stress_avg``
      is an honest ``None`` (contract §7.4: "≥ 10 days else None").
    """

    long_window: int = 250
    short_window: int = 60
    stress_quantile: float = 0.10
    elevated_delta: float = 0.05
    converging_delta: float = 0.15
    converging_level: float = 0.80
    min_pairs: int = 1
    min_stress_obs: int = 10

    def __post_init__(self) -> None:
        for name in ("long_window", "short_window", "min_stress_obs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 2:
                raise ValueError(f"{name} must be an int >= 2, got {v!r}")
        if isinstance(self.min_pairs, bool) or not isinstance(self.min_pairs, int) or self.min_pairs < 1:
            raise ValueError(f"min_pairs must be an int >= 1, got {self.min_pairs!r}")
        if self.short_window > self.long_window:
            raise ValueError(
                f"short_window {self.short_window} must be <= long_window "
                f"{self.long_window} (CURRENT is a sub-window of NORMAL)"
            )
        if not (0.0 < self.stress_quantile < 1.0):
            raise ValueError(
                f"stress_quantile must be in (0, 1), got {self.stress_quantile}"
            )
        for name in ("elevated_delta", "converging_delta"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v!r}")
        if not (-1.0 <= self.converging_level <= 1.0):
            raise ValueError(
                f"converging_level must be in [-1, 1], got {self.converging_level}"
            )


DEFAULT_REGIME_PARAMS = CorrelationRegimeParams()


@dataclass(frozen=True)
class CorrelationState:
    """Normal / current / stress correlation and the regime they imply (§19).

    Averages are over the UPPER TRIANGLE of the pairwise Pearson matrix —
    each unordered pair counted once, pairs whose correlation is ``None``
    (zero variance) skipped, never filled with 0.0.

    - ``normal_avg``: average pairwise correlation over the last
      ``long_window`` observations;
    - ``current_avg``: same over the last ``short_window``;
    - ``current_avg_spearman`` (spec §18, a NON-FIELD attribute — see the
      note on the attribute itself): the RANK-correlation twin of
      ``current_avg`` — same SHORT window, same tickers, same
      upper-triangle convention, :func:`spearman` instead of
      :func:`_pearson`. ``None`` on insufficient data (the state would be
      UNAVAILABLE anyway) or when no pair has a defined rank correlation.
      It is a DISPLAY diagnostic: no state rule reads it, so ``state`` is
      byte-identical with or without it. The Pearson/Spearman GAP is the
      thing to read — a book whose Pearson average is low while its
      Spearman average is high is co-moving monotonically but non-linearly,
      which Pearson alone understates;
    - ``stress_avg``: same over the worst ``stress_quantile`` of days by the
      EQUAL-WEIGHT portfolio return of the same tickers (``None`` when fewer
      than ``min_stress_obs`` such days exist);
    - ``delta``: ``current_avg − normal_avg`` (``None`` if either is
      ``None``) — the §19 "0.61 → 0.84" number;
    - ``state``: NORMAL / ELEVATED / CONVERGING / UNAVAILABLE;
    - ``n_pairs``: pairs that produced a CURRENT correlation;
    - ``worst_pairs``: up to :data:`WORST_PAIRS` ``(a, b, current_rho)``
      rows, highest current correlation first, ties by ticker order;
    - ``reason``: why the state is UNAVAILABLE (real numbers), else ``None``.
    """

    normal_avg: float | None
    current_avg: float | None
    stress_avg: float | None
    delta: float | None
    state: str
    n_pairs: int
    n_obs_long: int
    n_obs_short: int
    n_obs_stress: int
    worst_pairs: tuple[tuple[str, str, float], ...]
    reason: str | None

    # ADDITIVE (spec §18, this batch) — a NON-FIELD attribute, deliberately.
    #
    # ``current_avg_spearman`` is a real, readable attribute of every
    # instance (default ``None``), but it is NOT a dataclass field, so
    # ``dataclasses.asdict`` — the one serialiser the gateway uses for this
    # object — produces the SAME eleven keys it produced before this batch.
    # That is the point: the §19 wire contract is pinned key-for-key by the
    # gateway's own tests, this batch is SHADOW, and a research display
    # diagnostic must not silently widen a published API. A later batch that
    # WANTS it on the wire promotes it to a field and updates the wire tests
    # in the same commit — an explicit step, not a side effect of computing
    # a number.
    #
    # Consequences, all intentional: it takes no part in ``==``/``hash``
    # (two states equal on the eleven fields are equal), it does not appear
    # in ``repr``, and ``dataclasses.replace`` drops it back to ``None``.
    current_avg_spearman: ClassVar[float | None] = None

    def __post_init__(self) -> None:
        if self.state == STATE_UNAVAILABLE and not self.reason:
            raise ValueError("state=UNAVAILABLE requires a non-empty reason")

    def with_spearman(self, rho: float | None) -> "CorrelationState":
        """This state carrying ``current_avg_spearman = rho`` (spec §18).

        Returns ``self`` after setting the non-field attribute — the frozen
        dataclass is not copied, because nothing a field defines has
        changed. ``rho`` must be ``None`` or a finite number in ``[-1, 1]``;
        anything else is malformed input and raises ``ValueError`` rather
        than being displayed.
        """
        if rho is not None:
            if isinstance(rho, bool) or not isinstance(rho, (int, float)):
                raise ValueError(f"current_avg_spearman must be a number or None, got {rho!r}")
            rho = float(rho)
            if not math.isfinite(rho) or not (-1.0 <= rho <= 1.0):
                raise ValueError(
                    f"current_avg_spearman must be finite in [-1, 1], got {rho}"
                )
        object.__setattr__(self, "current_avg_spearman", rho)
        return self

    @property
    def is_available(self) -> bool:
        return self.state != STATE_UNAVAILABLE


def _upper_triangle_average(
    columns: "list[list[float]]",
    tickers: "tuple[str, ...]",
    estimator: "Callable[[Sequence[float], Sequence[float]], float | None]" = _pearson,
) -> tuple[float | None, int, list[tuple[str, str, float]]]:
    """``(mean rho, n_pairs, pair rows)`` over the upper triangle.

    Pairs whose correlation is ``None`` (zero variance / constant ranks —
    undefined, never 0.0) are SKIPPED and do not enter the count;
    ``(None, 0, [])`` when no pair is computable.

    ``estimator`` defaults to :func:`_pearson`, so every pre-existing call
    behaves byte-identically; passing :func:`spearman` gives the rank
    version over the SAME columns and the SAME upper-triangle convention,
    which is what makes the two averages a like-for-like comparison
    (spec §18).
    """
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            rho = estimator(columns[i], columns[j])
            if rho is not None:
                pairs.append((tickers[i], tickers[j], rho))
    if not pairs:
        return None, 0, []
    return math.fsum(r for _, _, r in pairs) / len(pairs), len(pairs), pairs


def rolling_spearman_matrix(
    closes_by_ticker: "Mapping[str, Sequence[float]]",
    window: int = 60,
) -> "dict[tuple[str, str], float | None]":
    """Pairwise rolling Spearman over the last ``window`` log returns (§18).

    The rank mirror of the Pearson pair loop :func:`build_dynamic_buckets`
    runs: one entry per UNORDERED pair, keyed ``(a, b)`` with ``a < b`` in
    sorted ticker order so the key is canonical regardless of mapping
    iteration order. A pair whose correlation is undefined (insufficient
    history, or a constant return window) maps to ``None`` — present in the
    mapping, honestly null, never dropped and never filled with 0.0.

    Display diagnostic (RESEARCH, spec §70): nothing consumes this to gate.
    """
    tickers = sorted(closes_by_ticker)
    out: dict[tuple[str, str], float | None] = {}
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            out[(a, b)] = rolling_spearman(
                closes_by_ticker[a], closes_by_ticker[b], window=window
            )
    return out


def rolling_spearman_average(
    closes_by_ticker: "Mapping[str, Sequence[float]]",
    window: int = 60,
) -> tuple[float | None, int]:
    """``(mean Spearman rho, n_pairs)`` over the upper triangle (§18).

    Averages :func:`rolling_spearman_matrix` over the pairs that produced a
    number; ``None`` pairs are SKIPPED and excluded from ``n_pairs``, and
    ``(None, 0)`` comes back when no pair is computable — the same honest
    convention :func:`_upper_triangle_average` uses, so the Pearson and
    Spearman averages of the same book are directly comparable.
    """
    defined = [
        rho for rho in rolling_spearman_matrix(closes_by_ticker, window=window).values()
        if rho is not None
    ]
    if not defined:
        return None, 0
    return math.fsum(defined) / len(defined), len(defined)


def _stress_rows(rows: "list[tuple[float, ...]]", quantile: float) -> list[int]:
    """Indices of the worst ``quantile`` share of days by the EQUAL-WEIGHT
    portfolio return ``mean_i(r_i)`` (contract §7.4).

    ``m = floor(n × quantile)`` days, the most negative first; ties broken by
    date order (earlier first — stable and deterministic). Returns the
    indices in DATE order so the sliced columns stay chronological.
    """
    n = len(rows)
    m = math.floor(n * quantile)
    if m <= 0:
        return []
    port = [math.fsum(row) / len(row) for row in rows]
    order = sorted(range(n), key=lambda t: (port[t], t))
    return sorted(order[:m])


def correlation_regime(
    matrix: "ReturnMatrix",
    *,
    params: CorrelationRegimeParams = DEFAULT_REGIME_PARAMS,
) -> CorrelationState:
    """Normal / current / stress correlation regime of a book (spec §19;
    contract §7.4).

    ``matrix`` must be a date-aligned ``ReturnMatrix`` of **LOG** returns
    (``ValueError`` otherwise — mixing return conventions is malformed
    input, not a data gap; contract §1). The three averages use the SAME
    ``_pearson`` estimator and the SAME upper-triangle convention, so the
    ``delta`` between them is a like-for-like comparison.

    Honest nulls: fewer than ``short_window`` observations, fewer than two
    tickers, or fewer than ``min_pairs`` computable pairs ⇒ state
    UNAVAILABLE with a ``reason`` carrying the real numbers; a stress sample
    below ``min_stress_obs`` days ⇒ ``stress_avg=None`` while the rest of the
    state stands; an UNAVAILABLE state carries ``current_avg_spearman=None``
    too (there is no current window to rank). ``normal_avg`` is measured over the last ``long_window``
    observations, or over ALL of them when the history is shorter (the
    window is a maximum, and the honest ``n_obs_long`` says what was used).

    State rules (all parameters):

    - CONVERGING when ``current_avg >= converging_level`` (level alone —
      persistent or sudden);
    - ELEVATED when ``delta >= elevated_delta``;
    - NORMAL otherwise. A jump ``>= converging_delta`` is named in ``reason``.

    ``current_avg_spearman`` (spec §18) is reported beside ``current_avg``
    over the SAME short window and takes part in NO state rule — it is a
    display diagnostic, and the state this function returns is identical to
    the state it returned before the field existed.

    Hand-check (contract §7.4): with ``normal_avg = 0.61`` and
    ``current_avg = 0.84``, ``delta = 0.23 >= 0.15`` and ``0.84 >= 0.80`` ⇒
    CONVERGING — the §19 example.
    """
    return_type = getattr(matrix, "return_type", None)
    if return_type != REGIME_RETURN_TYPE:
        raise ValueError(
            f"correlation_regime requires {REGIME_RETURN_TYPE!r} returns, got "
            f"{return_type!r} (correlation of returns uses the log convention)"
        )
    tickers = tuple(matrix.tickers)
    rows = [tuple(row) for row in matrix.rows]
    n = len(rows)

    def _na(reason: str, *, n_long: int = 0, n_short: int = 0) -> CorrelationState:
        return CorrelationState(
            normal_avg=None,
            current_avg=None,
            stress_avg=None,
            delta=None,
            state=STATE_UNAVAILABLE,
            n_pairs=0,
            n_obs_long=n_long,
            n_obs_short=n_short,
            n_obs_stress=0,
            worst_pairs=(),
            reason=reason,
        )

    if len(tickers) < 2:
        return _na(f"n_tickers={len(tickers)} < 2 (a correlation needs a pair)")
    if n < params.short_window:
        return _na(
            f"n={n} < short_window={params.short_window}", n_long=n, n_short=n
        )

    long_rows = rows[-params.long_window:]
    short_rows = rows[-params.short_window:]
    n_long, n_short = len(long_rows), len(short_rows)

    def _columns(window: "list[tuple[float, ...]]") -> list[list[float]]:
        return [[row[i] for row in window] for i in range(len(tickers))]

    short_columns = _columns(short_rows)
    current_avg, n_pairs, current_pairs = _upper_triangle_average(
        short_columns, tickers
    )
    if n_pairs < params.min_pairs or current_avg is None:
        return _na(
            f"n_pairs={n_pairs} < min_pairs={params.min_pairs} over the "
            f"{n_short}-day current window (zero-variance series are skipped)",
            n_long=n_long,
            n_short=n_short,
        )
    normal_avg, _, _ = _upper_triangle_average(_columns(long_rows), tickers)
    # Spec §18 rank twin of CURRENT: same window, same columns, same
    # upper-triangle convention, ``spearman`` in place of ``_pearson``.
    # DISPLAY ONLY — it is computed after the state-deciding numbers and
    # feeds no branch below, so the state is unchanged by its presence.
    current_avg_spearman, _, _ = _upper_triangle_average(
        short_columns, tickers, spearman
    )

    stress_idx = _stress_rows(long_rows, params.stress_quantile)
    stress_avg: float | None = None
    if len(stress_idx) >= params.min_stress_obs:
        stress_rows_ = [long_rows[t] for t in stress_idx]
        stress_avg, _, _ = _upper_triangle_average(_columns(stress_rows_), tickers)

    delta = current_avg - normal_avg if normal_avg is not None else None
    if current_avg >= params.converging_level:
        state = STATE_CONVERGING
    elif delta is not None and delta >= params.elevated_delta:
        state = STATE_ELEVATED
    else:
        state = STATE_NORMAL

    worst = sorted(current_pairs, key=lambda p: (-p[2], p[0], p[1]))[:WORST_PAIRS]
    # The reason names the real numbers behind the state (spec §19 shape).
    notes: list[str] = []
    if state == STATE_CONVERGING:
        notes.append(
            f"current average correlation {current_avg:.2f} >= "
            f"{params.converging_level:.2f} converging level"
        )
    if delta is not None and delta >= params.converging_delta:
        notes.append(
            f"regime shift: normal {normal_avg:.2f} -> current {current_avg:.2f} "
            f"(jump {delta:+.2f} >= {params.converging_delta:.2f})"
        )
    elif state == STATE_ELEVATED and delta is not None:
        notes.append(
            f"normal {normal_avg:.2f} -> current {current_avg:.2f} "
            f"(jump {delta:+.2f} >= {params.elevated_delta:.2f} elevated delta)"
        )
    return CorrelationState(
        normal_avg=normal_avg,
        current_avg=current_avg,
        stress_avg=stress_avg,
        delta=delta,
        state=state,
        n_pairs=n_pairs,
        n_obs_long=n_long,
        n_obs_short=n_short,
        n_obs_stress=len(stress_idx),
        worst_pairs=tuple(worst),
        reason="; ".join(notes) if notes else None,
    ).with_spearman(current_avg_spearman)
