"""Standardized returns layer (risk spec §3; Phase B design contract §2.1).

Pure, deterministic, stdlib-only (house rule): no DB, no market data, no
numpy. Every statistical risk model in ``libs/trading_core/risk/`` operates
on RETURNS or on P&L series built from returns — never on raw prices — and
this module is the single place those returns are computed and date-aligned
so VaR, ES, volatility and risk contribution all see the same numbers.

Two return conventions, both hand-checkable, chosen per use (contract §1):

- ``SIMPLE``  ``r_t = close[t] / close[t-1] - 1`` — used for **P&L
  construction** (``pnl = exposure × r_simple`` is exact for stock).
- ``LOG``     ``r_t = ln(close[t] / close[t-1])`` — used for correlation
  and realized volatility (the pre-existing ``correlation.py`` and
  ``realized_vol`` conventions). :func:`log_returns` MOVED here from
  ``correlation.py``, which re-exports it; behaviour is byte-identical.

Return ``t`` is dated on the LATER bar (the day the return is realized).
A close ``<= 0`` is malformed input and raises ``ValueError`` — a silent
fallback would poison every downstream statistic.

Alignment rule (contract §2.1) — the ONE way several tickers become a
matrix: per-ticker returns are computed FIRST on that ticker's own
consecutive bars, THEN inner-joined on return dates. A date missing for any
ticker is dropped for all tickers, and a return is NEVER compounded across a
gap in another ticker's history (each ticker's return on a kept date is
still its own bar-to-bar return). Bars must be strictly increasing by date
(``ValueError`` otherwise — duplicated or unsorted bars are malformed).

Provenance metadata (spec §3 ``return_type, frequency, data_source``) lives
on :class:`ReturnSeries` / :class:`ReturnMatrix` as typed fields, never a
loose dict; ``lookback_window`` is ``n_obs`` and ``timestamp`` is ``as_of``
(the last return date). Model-level metadata is ``ModelMeta`` (contract
§2.2), owned by the models package.

This module imports nothing from ``risk/`` or ``correlation.py`` (import
cycle guard: ``correlation.py`` imports it).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

#: The two supported return conventions (spec §3 "support at minimum
#: simple returns, log returns").
ReturnType = Literal["SIMPLE", "LOG"]

#: Contract-fixed strings; a ``ReturnSeries``/``ReturnMatrix`` carries one.
RETURN_TYPE_SIMPLE: ReturnType = "SIMPLE"
RETURN_TYPE_LOG: ReturnType = "LOG"
_RETURN_TYPES: tuple[str, ...] = (RETURN_TYPE_SIMPLE, RETURN_TYPE_LOG)

#: Default provenance labels (spec §3 ``frequency`` / ``data_source``).
DEFAULT_FREQUENCY = "1D"
DEFAULT_SOURCE = "stock_bars_daily"


# ---------------------------------------------------------------------------
# Return arithmetic
# ---------------------------------------------------------------------------


def _check_closes(closes: Sequence[float]) -> None:
    """Every close must be > 0 (shared guard; message text is contract-fixed)."""
    for c in closes:
        if c <= 0:
            raise ValueError(f"closes must all be > 0, got {c}")


def simple_returns(closes: Sequence[float]) -> list[float]:
    """Simple (arithmetic) returns ``close[t] / close[t-1] - 1`` (spec §3).

    Returns a list one shorter than ``closes`` (no return exists for the
    first bar); an empty or single-element input yields ``[]``. Every close
    must be > 0 (``ValueError`` otherwise — a non-positive close is
    malformed input, not missing data).

    Hand-check: ``[100, 110, 99]`` → ``[110/100 - 1, 99/110 - 1]`` =
    ``[0.10, -0.10]``.
    """
    _check_closes(closes)
    return [closes[t] / closes[t - 1] - 1.0 for t in range(1, len(closes))]


def log_returns(closes: Sequence[float]) -> list[float]:
    """Daily log returns ``ln(close[t] / close[t-1])`` (plan §12.4; spec §3).

    Returns a list one shorter than ``closes`` (no return exists for the
    first bar). Every close must be > 0 — log returns are undefined
    otherwise, and a silent fallback would poison the correlations.

    Moved verbatim from ``libs/trading_core/correlation.py`` (contract §2.1,
    invariant §3.7): identical output on every input, identical
    ``ValueError`` text.
    """
    for c in closes:
        if c <= 0:
            raise ValueError(f"closes must all be > 0, got {c}")
    return [
        math.log(closes[t] / closes[t - 1]) for t in range(1, len(closes))
    ]


def _returns_of_type(closes: Sequence[float], return_type: str) -> list[float]:
    if return_type == RETURN_TYPE_SIMPLE:
        return simple_returns(closes)
    if return_type == RETURN_TYPE_LOG:
        return log_returns(closes)
    raise ValueError(
        f"return_type must be one of {_RETURN_TYPES}, got {return_type!r}"
    )


def _check_strictly_increasing(dates: Sequence[date], *, what: str) -> None:
    """Dates must be strictly increasing (no duplicates, no reordering)."""
    for i in range(1, len(dates)):
        if not dates[i] > dates[i - 1]:
            raise ValueError(
                f"{what} must be strictly increasing by date, got "
                f"{dates[i - 1]} followed by {dates[i]} at index {i}"
            )


# ---------------------------------------------------------------------------
# Typed containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnSeries:
    """One ticker's return series with provenance (spec §3; contract §2.1).

    - ``dates[t]`` is the date of the return — the LATER of the two bars it
      was computed from;
    - ``values[t]`` is the return realized on ``dates[t]`` under
      ``return_type`` (``"SIMPLE"`` or ``"LOG"``);
    - ``frequency`` / ``source`` are provenance labels carried through to
      every model that consumes the series.

    Invariants (``ValueError`` on violation — malformed input): equal
    lengths, strictly increasing dates, a known ``return_type``.
    """

    ticker: str
    dates: tuple[date, ...]
    values: tuple[float, ...]
    return_type: ReturnType
    frequency: str = DEFAULT_FREQUENCY
    source: str = DEFAULT_SOURCE

    def __post_init__(self) -> None:
        if self.return_type not in _RETURN_TYPES:
            raise ValueError(
                f"return_type must be one of {_RETURN_TYPES}, "
                f"got {self.return_type!r}"
            )
        if len(self.dates) != len(self.values):
            raise ValueError(
                f"{self.ticker}: dates and values must have equal length, "
                f"got {len(self.dates)} dates and {len(self.values)} values"
            )
        _check_strictly_increasing(self.dates, what=f"{self.ticker} return dates")

    @property
    def n_obs(self) -> int:
        """Number of returns actually held (spec §3 ``lookback_window``)."""
        return len(self.values)

    @property
    def as_of(self) -> date | None:
        """Date of the last return, or ``None`` for an empty series."""
        return self.dates[-1] if self.dates else None

    def window(self, n: int) -> ReturnSeries:
        """The LAST ``n`` returns (all of them if ``n >= n_obs``); ``n >= 0``."""
        if n < 0:
            raise ValueError(f"window n must be >= 0, got {n}")
        start = max(0, len(self.values) - n)
        return ReturnSeries(
            ticker=self.ticker,
            dates=self.dates[start:],
            values=self.values[start:],
            return_type=self.return_type,
            frequency=self.frequency,
            source=self.source,
        )


@dataclass(frozen=True)
class ReturnMatrix:
    """Date-aligned returns of several tickers (contract §2.1).

    ``rows[t][i]`` is the return of ``tickers[i]`` on ``dates[t]``. Built by
    :func:`align`; every row is complete (no missing cells — the inner join
    guarantees it), all columns share one ``return_type``.

    Invariants (``ValueError`` — malformed input): ``len(rows) ==
    len(dates)``, every row has ``len(tickers)`` cells, tickers unique,
    dates strictly increasing, known ``return_type``.
    """

    dates: tuple[date, ...]
    tickers: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    return_type: ReturnType
    frequency: str = DEFAULT_FREQUENCY
    source: str = DEFAULT_SOURCE

    def __post_init__(self) -> None:
        if self.return_type not in _RETURN_TYPES:
            raise ValueError(
                f"return_type must be one of {_RETURN_TYPES}, "
                f"got {self.return_type!r}"
            )
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError(f"tickers must be unique, got {self.tickers}")
        if len(self.rows) != len(self.dates):
            raise ValueError(
                f"rows and dates must have equal length, got "
                f"{len(self.rows)} rows and {len(self.dates)} dates"
            )
        width = len(self.tickers)
        for t, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"row {t} ({self.dates[t]}) has {len(row)} cells, "
                    f"expected {width} (one per ticker)"
                )
        _check_strictly_increasing(self.dates, what="matrix dates")

    @property
    def n_obs(self) -> int:
        """Number of aligned observation dates."""
        return len(self.dates)

    @property
    def as_of(self) -> date | None:
        """Last aligned date, or ``None`` when the matrix is empty."""
        return self.dates[-1] if self.dates else None

    def column(self, ticker: str) -> list[float]:
        """The return series of ``ticker`` in date order.

        Raises ``KeyError`` when the ticker has no column — callers that
        must degrade honestly (e.g. :func:`~libs.trading_core.risk.pnl_series
        .book_pnl_series`) test membership in ``tickers`` first.
        """
        try:
            i = self.tickers.index(ticker)
        except ValueError:
            raise KeyError(
                f"ticker {ticker!r} has no column; matrix has {self.tickers}"
            ) from None
        return [row[i] for row in self.rows]

    def window(self, n: int) -> ReturnMatrix:
        """The LAST ``n`` dates (all if ``n >= n_obs``); ``n >= 0``."""
        if n < 0:
            raise ValueError(f"window n must be >= 0, got {n}")
        start = max(0, len(self.rows) - n)
        return ReturnMatrix(
            dates=self.dates[start:],
            tickers=self.tickers,
            rows=self.rows[start:],
            return_type=self.return_type,
            frequency=self.frequency,
            source=self.source,
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def returns_from_closes(
    ticker: str,
    bars: Sequence[tuple[date, float]],
    *,
    return_type: ReturnType,
    frequency: str = DEFAULT_FREQUENCY,
    source: str = DEFAULT_SOURCE,
) -> ReturnSeries:
    """Build a :class:`ReturnSeries` from ``(date, close)`` bars (spec §3).

    Returns are computed on CONSECUTIVE bars of this ticker only —
    ``values[t-1] = f(bars[t-1].close, bars[t].close)`` dated ``bars[t]
    .date`` — with ``f`` = simple or log per ``return_type``. Bars must be
    strictly increasing by date and every close > 0 (``ValueError``
    otherwise). Zero or one bar yields an empty series (honest: nothing to
    compute), never an exception.
    """
    dates = tuple(d for d, _ in bars)
    closes = [c for _, c in bars]
    _check_strictly_increasing(dates, what=f"{ticker} bars")
    values = _returns_of_type(closes, return_type)
    return ReturnSeries(
        ticker=ticker,
        dates=dates[1:],
        values=tuple(values),
        return_type=return_type,
        frequency=frequency,
        source=source,
    )


def align(series: Sequence[ReturnSeries]) -> ReturnMatrix:
    """Inner-join several :class:`ReturnSeries` on return dates (contract §2.1).

    Rule: each series' returns are ALREADY its own bar-to-bar returns
    (built by :func:`returns_from_closes`); this function only keeps the
    dates present in EVERY series, in ascending order, and never
    recomputes or compounds a return across a gap. A date missing for any
    ticker is dropped for all tickers. Column order = input order.

    ``ValueError`` (malformed input) when the input is empty, tickers
    repeat, or the series mix ``return_type`` / ``frequency``. ``source``
    is kept when all agree, else the sorted distinct sources joined by
    ``"+"`` (provenance is never silently dropped). Zero common dates yields
    an EMPTY matrix (``n_obs == 0``) — honest degradation, not an error.
    """
    if not series:
        raise ValueError("align() needs at least one ReturnSeries")
    tickers = tuple(s.ticker for s in series)
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"tickers must be unique, got {tickers}")
    return_types = {s.return_type for s in series}
    if len(return_types) != 1:
        raise ValueError(
            f"cannot align mixed return types {sorted(return_types)}"
        )
    frequencies = {s.frequency for s in series}
    if len(frequencies) != 1:
        raise ValueError(
            f"cannot align mixed frequencies {sorted(frequencies)}"
        )
    sources = sorted({s.source for s in series})
    source = sources[0] if len(sources) == 1 else "+".join(sources)

    common: set[date] = set(series[0].dates)
    for s in series[1:]:
        common &= set(s.dates)
    dates = tuple(sorted(common))

    lookups = [dict(zip(s.dates, s.values)) for s in series]
    rows = tuple(tuple(lk[d] for lk in lookups) for d in dates)
    return ReturnMatrix(
        dates=dates,
        tickers=tickers,
        rows=rows,
        return_type=series[0].return_type,
        frequency=series[0].frequency,
        source=source,
    )
