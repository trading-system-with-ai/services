"""Point-in-time fundamentals — snapshot, change and valuation context (event
spec §16, §28, §29, §30, §33, §35, §85, §96; audit §7, §11.3; Phase E2 unit U2).

Pure stdlib, deterministic, **no I/O** — like the rest of
``libs/trading_core/events/`` this module may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4). It is handed
plain statement rows by the gateway seam (``apps/gateway/fundamentals.py``)
and hands back frozen result objects. It never learns whether a row arrived
from a provider dataclass or a database mirror: :class:`StatementLike` is a
structural protocol, so the provider's ``FinancialStatement`` and the ORM's
``FundamentalStatementRow`` both satisfy it without either importing the
other. :func:`coerce_statement` additionally accepts a plain mapping, which
is what keeps the tests free of both halves.

Four ideas carry the whole module:

1. **The as-of gate is on ``acceptance_datetime``, never on the fiscal
   period** (§7, §85, §96). A quarter ending 2025-09-30 was not public on
   2025-10-01 — it became public the instant the filing was accepted, weeks
   later. :func:`select_statements_as_of` is THE place that rule lives, and
   it filters on ``acceptance_datetime <= as_of`` alone. A row whose
   acceptance instant is unknown is EXCLUDED with a reason rather than
   admitted on its end date: an unknown publication time is unusable
   point-in-time evidence, not a permission slip.

2. **A metric is computed only when its inputs are present** (§28: "Do not
   compute a ratio if required inputs are unavailable"). Every metric is
   ``float | None`` in :attr:`FundamentalSnapshot.metrics` and every ``None``
   has a companion string in :attr:`FundamentalSnapshot.reasons`. There is no
   zero standing in for "the provider does not report capex", no NaN and no
   ``inf``: a zero revenue makes the margins ``None`` with
   ``"revenue_not_positive"``, not a division blow-up.

3. **The provider's gaps are named, not papered over.** Massive's XBRL
   financials carry no cash, no capex, no depreciation and no interest
   expense, so free cash flow, net debt, EV/EBITDA, debt/EBITDA, ROIC, the
   quick ratio and the FCF yield are structurally unavailable — they are
   emitted as ``None`` with :data:`NOT_REPORTED_REASONS` explaining which
   input is missing, so the UI prints "Unavailable — capex not reported by
   provider" rather than a blank or, far worse, a zero. ``total_debt`` IS
   reported but only as long-term debt, so it is delivered with the
   ``total_debt`` note pinned in :data:`METRIC_NOTES`.

4. **Change is a first-class object** (§29). :func:`snapshot_change` pairs a
   previous and a current snapshot metric by metric and returns
   :class:`MetricChange` values carrying the delta, a ``delta_bps`` for the
   ratio-valued metrics, an arrow direction and a trend classified over the
   stored history — with a reason whenever either side is missing, because
   "no change" and "we could not compare" are different claims.

Everything this module returns is QUANT provenance (§91): the statement
values themselves are DATA, the arithmetic over them is this platform's.
Consensus-dependent figures (EPS surprise, estimate revisions) appear
nowhere here — §33/§98 require them to be absent with a reason, and the
gateway supplies that block.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from libs.trading_core.events.taxonomy import UTC, require_utc

#: Sort floor for a statement with no acceptance instant. Such rows never
#: survive :func:`select_statements_as_of`, but the ordering key must stay
#: total for callers that sort a raw list.
_MIN_UTC = datetime.min.replace(tzinfo=UTC)

__all__ = [
    "BPS_METRICS",
    "DEFAULT_TREND_WINDOW",
    "FUNDAMENTALS_MODEL_VERSION",
    "METRIC_NOTES",
    "METRIC_ORDER",
    "NOT_REPORTED_REASONS",
    "PEER_CONTEXT_REASON",
    "TREND_TOLERANCE",
    "VALUATION_METRICS",
    "FundamentalSnapshot",
    "MetricChange",
    "Statement",
    "StatementLike",
    "StatementRef",
    "build_snapshot",
    "coerce_statement",
    "expectations_gap_inputs",
    "percentile_of",
    "select_statements_as_of",
    "snapshot_change",
    "valuation_context",
]

#: Bumped whenever the metric set or a formula changes, so a stored payload
#: can be told apart from one built by a later definition (§72 precedent:
#: ``IMPORTANCE_MODEL_VERSION``).
FUNDAMENTALS_MODEL_VERSION = "e2.1"

#: Trend is classified over at most this many trailing quarterly points (§29).
DEFAULT_TREND_WINDOW = 8

#: Relative tolerance under which a first-vs-last comparison is "flat". A
#: gross margin that moved 3 bps over eight quarters did not improve; calling
#: it "improving" would be a direction invented out of rounding noise.
TREND_TOLERANCE = 0.01

#: Metrics expressed as a ratio of revenue/assets/equity/price, whose change
#: is also reported in basis points (§29's "Δ: +70 bps" example).
BPS_METRICS: frozenset[str] = frozenset(
    {
        "gross_margin",
        "operating_margin",
        "net_margin",
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "roe_ttm",
        "roa_ttm",
        "roic",
        "fcf_yield",
        "earnings_yield",
    }
)

#: The valuation multiples §30 asks to be read against their own history.
VALUATION_METRICS: tuple[str, ...] = ("pe_ttm", "ps_ttm", "pb")

#: Structural gaps in the provider's XBRL view (contract "NOT PRESENT" list).
#: Each entry is the reason string the metric carries forever — the field is
#: not missing for this ticker or this quarter, it is missing from the feed.
NOT_REPORTED_REASONS: Mapping[str, str] = {
    "free_cash_flow": "capex not reported by provider",
    "capex": "capex not reported by provider",
    "cash": "cash and equivalents not reported by provider",
    "net_debt": "net debt needs cash, which is not reported by provider",
    "roic": "ROIC needs invested capital incl. cash, not reported by provider",
    "quick_ratio": "quick ratio needs cash and receivables, not reported by provider",
    "debt_to_ebitda": "EBITDA needs depreciation & amortisation, not reported by provider",
    "ev_ebitda": "EV/EBITDA needs cash and EBITDA, neither reported by provider",
    "fcf_yield": "FCF yield needs free cash flow, which needs capex (not reported)",
}

#: Caveats that travel WITH a computed number rather than replacing it.
METRIC_NOTES: Mapping[str, str] = {
    "total_debt": "long-term only — provider reports no short-term borrowings",
}

#: Why the §30 sector/peer comparison is absent. Peer sets and sector
#: aggregates are Phase G/J work; naming them here keeps the payload from
#: implying the comparison was attempted and came back empty.
PEER_CONTEXT_REASON = "peer/sector multiples not implemented (Phase G/J)"

#: Canonical metric order — the §28 list, then the §30 multiples. The UI
#: renders rows in this order, so it lives next to the formulas.
METRIC_ORDER: tuple[str, ...] = (
    "revenue",
    "revenue_ttm",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "eps_diluted",
    "eps_diluted_ttm",
    "eps_growth_yoy",
    "operating_cash_flow",
    "operating_cash_flow_ttm",
    "free_cash_flow",
    "capex",
    "cash",
    "total_debt",
    "net_debt",
    "roe_ttm",
    "roa_ttm",
    "roic",
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "debt_to_ebitda",
    "shares_diluted",
    "pe_ttm",
    "ps_ttm",
    "pb",
    "ev_ebitda",
    "fcf_yield",
    "earnings_yield",
)

# Flattened statement keys, spelled once so a provider rename is a one-line
# change here rather than a hunt through the formulas.
_REVENUES = "income_statement.revenues"
_GROSS_PROFIT = "income_statement.gross_profit"
_OPERATING_INCOME = "income_statement.operating_income_loss"
_NET_INCOME = "income_statement.net_income_loss"
_NET_INCOME_PARENT = "income_statement.net_income_loss_attributable_to_parent"
_EPS_DILUTED = "income_statement.diluted_earnings_per_share"
_SHARES_DILUTED = "income_statement.diluted_average_shares"
_OCF = "cash_flow_statement.net_cash_flow_from_operating_activities"
_ASSETS = "balance_sheet.assets"
_CURRENT_ASSETS = "balance_sheet.current_assets"
_CURRENT_LIABILITIES = "balance_sheet.current_liabilities"
_EQUITY = "balance_sheet.equity"
_EQUITY_PARENT = "balance_sheet.equity_attributable_to_parent"
_LONG_TERM_DEBT = "balance_sheet.long_term_debt"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@runtime_checkable
class StatementLike(Protocol):
    """The shape this module needs from one filed financial statement.

    Structural, not nominal, on purpose: the provider dataclass
    (``libs.market_data.provider.FinancialStatement``) and the ORM row
    (``apps.gateway.db.FundamentalStatementRow``) each satisfy it while this
    pure module imports neither. ``values`` is the flattened
    ``"statement.field" -> float`` mapping; anything non-numeric in it is
    ignored by :func:`_number`.
    """

    fiscal_year: int | None
    fiscal_period: str
    end_date: date | None
    acceptance_datetime: datetime | None
    values: Mapping[str, float]


@dataclass(frozen=True)
class Statement:
    """A normalised statement row — the module's internal working form.

    :func:`coerce_statement` produces these from anything :class:`StatementLike`
    or from a plain mapping, so every function below can assume aware-UTC
    acceptance instants and a float-only ``values`` mapping. ``timeframe`` is
    lower-cased (``"quarterly"`` / ``"annual"`` / ``"ttm"``) and
    ``fiscal_period`` upper-cased (``"Q3"`` / ``"FY"`` / ``"TTM"``) because
    the two providers disagree about case and a fiscal-quarter match must not
    turn on it.
    """

    fiscal_period: str
    fiscal_year: int | None = None
    end_date: date | None = None
    acceptance_datetime: datetime | None = None
    timeframe: str | None = None
    filing_date: date | None = None
    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))

    @property
    def label(self) -> str:
        """``"FY2025 Q3"``-style label, degrading to whatever is known."""
        period = self.fiscal_period or "?"
        if self.fiscal_year is None:
            return period
        return f"FY{self.fiscal_year} {period}"

    def get(self, key: str) -> float | None:
        """The finite float at ``key``, or ``None`` — never a NaN, never 0."""
        return _number(self.values.get(key))


@dataclass(frozen=True)
class StatementRef:
    """The provenance stub a snapshot carries for each statement it used.

    Not the whole statement: the payload needs to say WHICH filing produced
    the numbers ("FY2025 Q3, period ending 2025-09-27, accepted 2025-10-31
    20:31Z") and nothing more. ``acceptance_datetime`` is the §7 as-of key and
    is always aware-UTC when present.
    """

    label: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    end_date: date | None = None
    acceptance_datetime: datetime | None = None
    filing_date: date | None = None
    timeframe: str | None = None


@dataclass(frozen=True)
class FundamentalSnapshot:
    """The §28 snapshot of one ticker as of one instant.

    ``metrics`` maps every name in :data:`METRIC_ORDER` to a finite float or
    ``None``; ``reasons`` carries a string for every ``None`` (the invariant
    is checked by the tests, not merely intended). ``notes`` carries caveats
    for numbers that ARE present but mean less than their name suggests
    (``total_debt`` is long-term only). ``available`` is ``False`` when no
    statement was public at ``as_of`` at all — then every metric is ``None``
    and the reasons say so once.
    """

    ticker: str
    as_of: datetime
    available: bool = False
    quarterly: StatementRef | None = None
    ttm: StatementRef | None = None
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, str] = field(default_factory=dict)
    quarters_available: int = 0
    price: float | None = None
    market_cap: float | None = None
    model_version: str = FUNDAMENTALS_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "reasons", dict(self.reasons))
        object.__setattr__(self, "notes", dict(self.notes))

    def value(self, metric: str) -> float | None:
        """The metric's value, or ``None`` if absent or not computed."""
        return self.metrics.get(metric)

    def reason(self, metric: str) -> str | None:
        """Why ``metric`` is ``None``, or ``None`` when it has a value."""
        return self.reasons.get(metric)

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Metric names with no value, in :data:`METRIC_ORDER` order."""
        return tuple(
            name for name in METRIC_ORDER if self.metrics.get(name) is None
        )


@dataclass(frozen=True)
class MetricChange:
    """One row of the §29 previous-vs-current comparison.

    ``delta`` is absolute (``current - previous``) in the metric's own units;
    ``delta_bps`` is that delta in basis points and exists only for the ratio
    metrics in :data:`BPS_METRICS`, because "+70 bps" is meaningless for a
    revenue in dollars. ``pct_change`` is the relative move, defined only when
    the previous value is non-zero. ``direction`` is one of ``"up"`` /
    ``"down"`` / ``"flat"`` with :attr:`arrow` rendering ↑/↓/→. ``reason`` is
    set exactly when the comparison could not be made.
    """

    metric: str
    previous: float | None = None
    current: float | None = None
    delta: float | None = None
    delta_bps: float | None = None
    pct_change: float | None = None
    direction: str | None = None
    trend: str | None = None
    trend_points: int = 0
    reason: str | None = None
    note: str | None = None

    @property
    def arrow(self) -> str:
        """``↑`` / ``↓`` / ``→``, or ``""`` when there is no direction."""
        return {"up": "↑", "down": "↓", "flat": "→"}.get(self.direction or "", "")


# ---------------------------------------------------------------------------
# Small numeric helpers — every one of them refuses to fabricate
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """``None`` unless ``value`` is a finite real number (no NaN, no ±inf).

    Booleans are rejected on purpose: ``True`` is an ``int`` in Python and a
    provider that ships a flag where a figure belongs must not have it read
    as one dollar of revenue.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _positive(value: float | None) -> float | None:
    """``None`` unless the value is strictly positive.

    For prices and market caps only, where zero is not a small number but a
    missing quote: a zero price would make every multiple ``0.0``, a number
    that looks computed and is not.
    """
    if value is None or value <= 0.0:
        return None
    return value


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator`` only when the base is non-zero and finite.

    A zero denominator is a missing input, not a divide-by-zero to be papered
    over with ``inf`` — the caller records the reason instead.
    """
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return None
    return _number(numerator / denominator)


def _growth(later: float | None, earlier: float | None) -> float | None:
    """``(later - earlier) / |earlier|``, or ``None`` when the base is unusable.

    The absolute value in the DENOMINATOR is what makes a swing out of a loss
    read correctly: EPS moving from ``-0.50`` to ``+0.25`` is ``+150%``,
    whereas the signed base of the textbook ``later/earlier - 1`` reports
    ``-150%`` and inverts the sign of good news. For a positive base the two
    forms are identical. A zero base yields ``None``: growth from nothing is
    not a percentage, it is a division by zero wearing one.
    """
    if later is None or earlier is None:
        return None
    if earlier == 0.0:
        return None
    return _number((later - earlier) / abs(earlier))


def percentile_of(values: Sequence[float], target: float) -> float | None:
    """Where ``target`` sits in ``values``, 0-100, by fraction at or below it.

    The plain empirical rank rather than an interpolated one: with six or
    eight historical quarters (all §30 will ever have from stored filings) an
    interpolated percentile claims a resolution the sample does not have.
    ``None`` for an empty sample.
    """
    finite = [v for v in (_number(x) for x in values) if v is not None]
    if not finite:
        return None
    at_or_below = sum(1 for v in finite if v <= target)
    return _number(100.0 * at_or_below / len(finite))


def _median(values: Sequence[float]) -> float | None:
    """Median of the finite values (mean of the middle two when even)."""
    finite = sorted(v for v in (_number(x) for x in values) if v is not None)
    n = len(finite)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return _number(finite[mid])
    return _number((finite[mid - 1] + finite[mid]) / 2.0)


# ---------------------------------------------------------------------------
# Row coercion — the duck-typed seam
# ---------------------------------------------------------------------------


def _attr(row: Any, name: str) -> Any:
    """Read ``name`` off a mapping or an object, whichever ``row`` is."""
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _as_date(value: Any) -> date | None:
    """A plain ``date`` from a date, a datetime, or an ISO string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _as_utc(value: Any) -> datetime | None:
    """An aware-UTC instant from a datetime or an ISO-8601 string.

    A NAIVE datetime is refused (``None``), not assumed to be UTC: the
    acceptance instant is the as-of key, and quietly labelling an unknown
    zone shifts the boundary by hours in exactly the direction that leaks
    future filings into a past analysis (§96).
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return None
        return require_utc(value, name="acceptance_datetime")
    if isinstance(value, str) and value:
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        return require_utc(parsed, name="acceptance_datetime")
    return None


def coerce_statement(row: Any) -> Statement:
    """Normalise one provider dataclass / ORM row / mapping into a
    :class:`Statement`.

    Everything downstream assumes this ran: acceptance instants are aware-UTC
    or ``None``, ``values`` holds finite floats only, and the fiscal period is
    upper-cased. Non-numeric entries in ``values`` are dropped rather than
    coerced — a provider string like ``"n/a"`` must not become a number.
    """
    if isinstance(row, Statement):
        return row
    raw_values = _attr(row, "values") or {}
    values: dict[str, float] = {}
    if isinstance(raw_values, Mapping):
        for key, value in raw_values.items():
            number = _number(value)
            if number is not None:
                values[str(key)] = number
    fiscal_year = _number(_attr(row, "fiscal_year"))
    period = _attr(row, "fiscal_period")
    timeframe = _attr(row, "timeframe")
    return Statement(
        fiscal_period=str(period).upper().strip() if period else "",
        fiscal_year=int(fiscal_year) if fiscal_year is not None else None,
        end_date=_as_date(_attr(row, "end_date")),
        acceptance_datetime=_as_utc(_attr(row, "acceptance_datetime")),
        timeframe=str(timeframe).lower().strip() if timeframe else None,
        filing_date=_as_date(_attr(row, "filing_date")),
        values=values,
    )


def _ref(statement: Statement | None) -> StatementRef | None:
    if statement is None:
        return None
    return StatementRef(
        label=statement.label,
        fiscal_year=statement.fiscal_year,
        fiscal_period=statement.fiscal_period or None,
        end_date=statement.end_date,
        acceptance_datetime=statement.acceptance_datetime,
        filing_date=statement.filing_date,
        timeframe=statement.timeframe,
    )


def _is_ttm(statement: Statement) -> bool:
    return statement.timeframe == "ttm" or statement.fiscal_period == "TTM"


def _is_quarterly(statement: Statement) -> bool:
    if _is_ttm(statement):
        return False
    if statement.timeframe == "quarterly":
        return True
    if statement.timeframe in {"annual", "yearly"}:
        return False
    period = statement.fiscal_period
    return len(period) == 2 and period[0] == "Q" and period[1].isdigit()


def _sort_key(statement: Statement) -> tuple[date, datetime, str]:
    """Newest-first ordering key: period end, then acceptance instant.

    Period end leads because two quarters can be accepted in ONE filing (a
    late filer catching up): both share an acceptance instant, and the later
    PERIOD is the newer fact. Acceptance breaks ties the other way — a
    restatement of the same period accepted later supersedes the original.
    """
    accepted = statement.acceptance_datetime
    return (
        statement.end_date or date.min,
        accepted if accepted is not None else _MIN_UTC,
        statement.fiscal_period,
    )


# ---------------------------------------------------------------------------
# §7 / §85 / §96 — the as-of gate, on acceptance_datetime alone
# ---------------------------------------------------------------------------


#: Flow (period-additive) fields summed when deriving a trailing-four-quarter
#: statement; everything else (balance sheet, per-share averages) is a
#: snapshot and is carried from the latest quarter.
_TTM_FLOW_FIELDS: tuple[str, ...] = (
    "income_statement.revenues",
    "income_statement.cost_of_revenue",
    "income_statement.gross_profit",
    "income_statement.operating_expenses",
    "income_statement.research_and_development",
    "income_statement.selling_general_and_administrative_expenses",
    "income_statement.operating_income_loss",
    "income_statement.income_loss_from_continuing_operations_before_tax",
    "income_statement.income_tax_expense_benefit",
    "income_statement.net_income_loss",
    "income_statement.net_income_loss_attributable_to_parent",
    "income_statement.diluted_earnings_per_share",
    "income_statement.basic_earnings_per_share",
    "cash_flow_statement.net_cash_flow_from_operating_activities",
    "cash_flow_statement.net_cash_flow_from_investing_activities",
    "cash_flow_statement.net_cash_flow_from_financing_activities",
    "cash_flow_statement.net_cash_flow",
)
DERIVED_TTM_PERIOD = "TTM(derived)"


def derive_ttm_from_quarters(quarterly_newest_first: Sequence[Statement]) -> Statement | None:
    """A trailing-four-quarter statement summed from the four NEWEST visible
    quarterly statements (§47: deterministic backend arithmetic).

    Needed because the provider's own TTM rows carry no acceptance instant
    (live 2026-08-19) and are therefore — correctly — invisible to the as-of
    gate. The derived row is point-in-time safe by construction: its
    acceptance instant is the LATEST of the four constituents, so it becomes
    visible exactly when the fourth quarter did. Flow lines are summed only
    when ALL FOUR quarters report them (a partial sum is a fabricated total);
    balance-sheet and share-count lines are copied from the newest quarter.
    Returns None with fewer than four quarters. ``fiscal_period`` is
    :data:`DERIVED_TTM_PERIOD` so callers can label the basis.
    """
    quarters = list(quarterly_newest_first)[:4]
    if len(quarters) < 4:
        return None
    newest = quarters[0]
    values: dict[str, float] = {}
    for key in _TTM_FLOW_FIELDS:
        parts = [q.values.get(key) for q in quarters]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in parts):
            values[key] = float(sum(parts))  # type: ignore[arg-type]
    for key, value in newest.values.items():
        if key not in _TTM_FLOW_FIELDS and key not in values:
            values[key] = value
    accepted = max(
        (q.acceptance_datetime for q in quarters if q.acceptance_datetime is not None),
        default=None,
    )
    return Statement(
        fiscal_period=DERIVED_TTM_PERIOD,
        fiscal_year=newest.fiscal_year,
        end_date=newest.end_date,
        acceptance_datetime=accepted,
        timeframe="ttm",
        filing_date=newest.filing_date,
        values=values,
    )


def select_statements_as_of(
    rows: Iterable[Any], as_of: datetime
) -> tuple[Statement | None, Statement | None, list[Statement]]:
    """Statements public at ``as_of`` — ``(quarterly_latest, ttm_latest, quarterly_rows)``.

    THE point-in-time gate for fundamentals (§7, §85, §96). A statement is
    visible iff ``acceptance_datetime <= as_of``. The fiscal period end is
    NEVER consulted for this: a quarter that ended in September was not public
    in September, and filtering on ``end_date`` is precisely the look-ahead
    the audit's sentinel test plants (a filing accepted one hour after
    ``as_of`` must not appear).

    Rows with no acceptance instant are dropped — an unknown publication time
    cannot be proven to precede ``as_of``, and admitting it "because the
    period is old" reintroduces the same leak through the back door.

    ``quarterly_rows`` is every visible quarterly statement, NEWEST FIRST, so
    the year-over-year comparison can look five rows back. ``as_of`` must be
    timezone-aware; a naive instant is refused rather than assumed UTC.
    """
    moment = require_utc(as_of, name="as_of")
    quarterly: list[Statement] = []
    ttm: list[Statement] = []
    for row in rows:
        statement = coerce_statement(row)
        accepted = statement.acceptance_datetime
        if accepted is None:
            continue
        if accepted > moment:
            continue
        if _is_ttm(statement):
            ttm.append(statement)
        elif _is_quarterly(statement):
            quarterly.append(statement)
    quarterly.sort(key=_sort_key, reverse=True)
    ttm.sort(key=_sort_key, reverse=True)
    ttm_latest = ttm[0] if ttm else derive_ttm_from_quarters(quarterly)
    return (
        quarterly[0] if quarterly else None,
        ttm_latest,
        quarterly,
    )


def _excluded_reasons(rows: Iterable[Any], as_of: datetime) -> dict[str, str]:
    """Reasons for rows the as-of gate dropped, for the payload's honesty."""
    moment = require_utc(as_of, name="as_of")
    missing = 0
    future = 0
    for row in rows:
        statement = coerce_statement(row)
        if statement.acceptance_datetime is None:
            missing += 1
        elif statement.acceptance_datetime > moment:
            future += 1
    out: dict[str, str] = {}
    if missing:
        out["rows_without_acceptance_datetime"] = (
            f"{missing} statement row(s) excluded: no acceptance_datetime, so "
            "publication cannot be proven to precede as_of"
        )
    if future:
        out["rows_accepted_after_as_of"] = (
            f"{future} statement row(s) excluded: accepted after "
            f"{moment.isoformat()}"
        )
    return out


# ---------------------------------------------------------------------------
# §28 — the snapshot
# ---------------------------------------------------------------------------


def _yoy_match(
    quarterly_rows: Sequence[Statement], current: Statement
) -> Statement | None:
    """The SAME fiscal quarter one year earlier, or ``None``.

    Same quarter, not "four rows back": a company that restates or files two
    quarters at once would otherwise have Q3 compared against Q2, which is a
    seasonality artefact dressed as growth. Requires both fiscal years.
    """
    if current.fiscal_year is None or not current.fiscal_period:
        return None
    target_year = current.fiscal_year - 1
    for statement in quarterly_rows:
        if (
            statement.fiscal_year == target_year
            and statement.fiscal_period == current.fiscal_period
        ):
            return statement
    return None


def _put(
    metrics: dict[str, float | None],
    reasons: dict[str, str],
    name: str,
    value: float | None,
    reason: str,
) -> None:
    """Record a metric, attaching ``reason`` exactly when it came out ``None``."""
    metrics[name] = value
    if value is None:
        reasons[name] = reason
    else:
        reasons.pop(name, None)


def _missing(statement: Statement | None, key: str) -> str:
    """The ``input_unavailable:<field>`` reason for a missing statement field."""
    if statement is None:
        return "input_unavailable:no_statement"
    return f"input_unavailable:{key}"


def _equity_of(statement: Statement | None) -> float | None:
    """Equity attributable to the parent, falling back to total equity.

    Parent equity is the right denominator for ROE when minority interests
    exist; total equity is the honest fallback when the provider reports only
    the one line.
    """
    if statement is None:
        return None
    parent = statement.get(_EQUITY_PARENT)
    return parent if parent is not None else statement.get(_EQUITY)


def _net_income_of(statement: Statement | None) -> float | None:
    """Net income attributable to the parent, falling back to total."""
    if statement is None:
        return None
    parent = statement.get(_NET_INCOME_PARENT)
    return parent if parent is not None else statement.get(_NET_INCOME)


def build_snapshot(
    rows: Iterable[Any],
    *,
    as_of: datetime,
    ticker: str,
    price: float | None = None,
    market_cap: float | None = None,
) -> FundamentalSnapshot:
    """Build the §28 snapshot from stored statement rows as of ``as_of``.

    ``rows`` may be in any order and may mix quarterly, annual and TTM
    timeframes; :func:`select_statements_as_of` applies the point-in-time gate
    and splits them. Balance-sheet metrics come off the newest visible
    QUARTERLY statement (a balance sheet is a snapshot, so its TTM copy would
    be the same figures under a misleading label); flow metrics come off the
    quarter for the quarter columns and off the TTM statement for the trailing
    columns.

    ``price`` and ``market_cap`` are supplied by the caller (this module reads
    no bars). When ``market_cap`` is omitted it is derived as
    ``price × diluted shares`` if both are known — and when neither is
    available the multiples are ``None`` with the reason naming which leg was
    missing, never a multiple computed off a stale price.

    Every metric in :data:`METRIC_ORDER` is present in the result, with a
    reason for each ``None``. Nothing here returns a NaN, an ``inf`` or a zero
    standing in for absence.
    """
    moment = require_utc(as_of, name="as_of")
    rows = list(rows)
    quarter, ttm, quarterly_rows = select_statements_as_of(rows, moment)

    metrics: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    notes: dict[str, str] = {}
    if ttm is not None and ttm.fiscal_period == DERIVED_TTM_PERIOD:
        notes["ttm_basis"] = (
            "TTM derived as the sum of the four newest quarterly statements "
            "visible at as_of (provider TTM rows carry no acceptance instant)"
        )

    # A non-positive price or market cap is malformed input, not a valuation:
    # admitting it would hand the multiples a zero numerator and print a P/E
    # of 0.0, which is the fabricated number this whole layer exists to
    # refuse. It is dropped here, once, so every multiple below sees only a
    # usable price or none at all.
    price = _positive(_number(price))
    market_cap = _positive(_number(market_cap))

    if quarter is None and ttm is None:
        excluded = _excluded_reasons(rows, moment)
        reason = (
            "no financial statement was public at as_of"
            if not excluded
            else "; ".join(excluded.values())
        )
        for name in METRIC_ORDER:
            metrics[name] = None
            reasons[name] = NOT_REPORTED_REASONS.get(name, reason)
        return FundamentalSnapshot(
            ticker=ticker,
            as_of=moment,
            available=False,
            metrics=metrics,
            reasons={**reasons, **excluded},
            quarters_available=0,
            price=price,
            market_cap=market_cap,
        )

    # --- quarterly flow ---------------------------------------------------
    revenue = quarter.get(_REVENUES) if quarter else None
    _put(metrics, reasons, "revenue", revenue, _missing(quarter, _REVENUES))

    gross_profit = quarter.get(_GROSS_PROFIT) if quarter else None
    operating_income = quarter.get(_OPERATING_INCOME) if quarter else None
    net_income = _net_income_of(quarter)

    if revenue is None:
        margin_reason = _missing(quarter, _REVENUES)
    elif revenue <= 0.0:
        margin_reason = "revenue_not_positive"
    else:
        margin_reason = ""
    margin_base = revenue if revenue is not None and revenue > 0.0 else None

    _put(
        metrics,
        reasons,
        "gross_margin",
        _ratio(gross_profit, margin_base),
        margin_reason or _missing(quarter, _GROSS_PROFIT),
    )
    _put(
        metrics,
        reasons,
        "operating_margin",
        _ratio(operating_income, margin_base),
        margin_reason or _missing(quarter, _OPERATING_INCOME),
    )
    _put(
        metrics,
        reasons,
        "net_margin",
        _ratio(net_income, margin_base),
        margin_reason or _missing(quarter, _NET_INCOME),
    )

    eps = quarter.get(_EPS_DILUTED) if quarter else None
    _put(metrics, reasons, "eps_diluted", eps, _missing(quarter, _EPS_DILUTED))

    shares = quarter.get(_SHARES_DILUTED) if quarter else None
    _put(
        metrics,
        reasons,
        "shares_diluted",
        shares,
        _missing(quarter, _SHARES_DILUTED),
    )

    ocf = quarter.get(_OCF) if quarter else None
    _put(metrics, reasons, "operating_cash_flow", ocf, _missing(quarter, _OCF))

    # --- trailing twelve months ------------------------------------------
    ttm_reason = "no TTM statement was public at as_of"
    revenue_ttm = ttm.get(_REVENUES) if ttm else None
    _put(
        metrics,
        reasons,
        "revenue_ttm",
        revenue_ttm,
        _missing(ttm, _REVENUES) if ttm else ttm_reason,
    )
    eps_ttm = ttm.get(_EPS_DILUTED) if ttm else None
    _put(
        metrics,
        reasons,
        "eps_diluted_ttm",
        eps_ttm,
        _missing(ttm, _EPS_DILUTED) if ttm else ttm_reason,
    )
    ocf_ttm = ttm.get(_OCF) if ttm else None
    _put(
        metrics,
        reasons,
        "operating_cash_flow_ttm",
        ocf_ttm,
        _missing(ttm, _OCF) if ttm else ttm_reason,
    )
    net_income_ttm = _net_income_of(ttm)

    # --- year over year ---------------------------------------------------
    prior = _yoy_match(quarterly_rows, quarter) if quarter else None
    if prior is None:
        yoy_reason = (
            "input_unavailable:prior_year_same_quarter "
            f"({len(quarterly_rows)} quarterly statement(s) public at as_of)"
        )
        _put(metrics, reasons, "revenue_growth_yoy", None, yoy_reason)
        _put(metrics, reasons, "eps_growth_yoy", None, yoy_reason)
    else:
        prior_revenue = prior.get(_REVENUES)
        _put(
            metrics,
            reasons,
            "revenue_growth_yoy",
            _growth(revenue, prior_revenue),
            _missing(prior, _REVENUES)
            if prior_revenue in (None, 0.0)
            else _missing(quarter, _REVENUES),
        )
        prior_eps = prior.get(_EPS_DILUTED)
        _put(
            metrics,
            reasons,
            "eps_growth_yoy",
            _growth(eps, prior_eps),
            _missing(prior, _EPS_DILUTED)
            if prior_eps in (None, 0.0)
            else _missing(quarter, _EPS_DILUTED),
        )

    # --- balance sheet ----------------------------------------------------
    assets = quarter.get(_ASSETS) if quarter else None
    current_assets = quarter.get(_CURRENT_ASSETS) if quarter else None
    current_liabilities = quarter.get(_CURRENT_LIABILITIES) if quarter else None
    equity = _equity_of(quarter)
    long_term_debt = quarter.get(_LONG_TERM_DEBT) if quarter else None

    _put(
        metrics,
        reasons,
        "total_debt",
        long_term_debt,
        _missing(quarter, _LONG_TERM_DEBT),
    )
    if long_term_debt is not None:
        notes["total_debt"] = METRIC_NOTES["total_debt"]

    _put(
        metrics,
        reasons,
        "current_ratio",
        _ratio(current_assets, current_liabilities),
        _missing(quarter, _CURRENT_LIABILITIES)
        if current_liabilities in (None, 0.0)
        else _missing(quarter, _CURRENT_ASSETS),
    )
    _put(
        metrics,
        reasons,
        "debt_to_equity",
        _ratio(long_term_debt, equity),
        _missing(quarter, _EQUITY)
        if equity in (None, 0.0)
        else _missing(quarter, _LONG_TERM_DEBT),
    )
    if metrics.get("debt_to_equity") is not None:
        notes["debt_to_equity"] = METRIC_NOTES["total_debt"]

    _put(
        metrics,
        reasons,
        "roe_ttm",
        _ratio(net_income_ttm, equity),
        (_missing(ttm, _NET_INCOME) if ttm else ttm_reason)
        if net_income_ttm is None
        else _missing(quarter, _EQUITY),
    )
    _put(
        metrics,
        reasons,
        "roa_ttm",
        _ratio(net_income_ttm, assets),
        (_missing(ttm, _NET_INCOME) if ttm else ttm_reason)
        if net_income_ttm is None
        else _missing(quarter, _ASSETS),
    )

    # --- structurally unavailable (contract "NOT PRESENT") ----------------
    for name, reason in NOT_REPORTED_REASONS.items():
        if name in {"ev_ebitda", "fcf_yield"}:
            continue  # valuation block below, same reason
        _put(metrics, reasons, name, None, reason)

    # --- §30 valuation ----------------------------------------------------
    if market_cap is None and price is not None and shares is not None and shares > 0.0:
        market_cap = _number(price * shares)
        if market_cap is not None:
            notes["market_cap"] = "derived as price × diluted average shares"

    price_reason = "input_unavailable:price"
    cap_reason = (
        price_reason
        if price is None
        else "input_unavailable:diluted_average_shares (market cap not derivable)"
    )

    if price is None:
        pe = ps = pb = None
        pe_reason = ps_reason = pb_reason = price_reason
        earnings_yield = None
        ey_reason = price_reason
    else:
        pe = _ratio(price, eps_ttm)
        pe_reason = (
            (_missing(ttm, _EPS_DILUTED) if ttm else ttm_reason)
            if eps_ttm is None
            else "eps_ttm_not_positive"
        )
        if pe is not None and pe < 0.0:
            # A negative P/E is arithmetic, not a valuation: a loss-making
            # company has no earnings multiple. Report the absence.
            pe = None
            pe_reason = "eps_ttm_negative — no meaningful P/E for a loss"
        ps = _ratio(market_cap, revenue_ttm)
        ps_reason = (
            cap_reason
            if market_cap is None
            else ((_missing(ttm, _REVENUES) if ttm else ttm_reason))
        )
        pb = _ratio(market_cap, equity)
        pb_reason = (
            cap_reason if market_cap is None else _missing(quarter, _EQUITY)
        )
        earnings_yield = _ratio(eps_ttm, price)
        ey_reason = (_missing(ttm, _EPS_DILUTED) if ttm else ttm_reason)

    _put(metrics, reasons, "pe_ttm", pe, pe_reason)
    _put(metrics, reasons, "ps_ttm", ps, ps_reason)
    _put(metrics, reasons, "pb", pb, pb_reason)
    _put(metrics, reasons, "earnings_yield", earnings_yield, ey_reason)
    _put(metrics, reasons, "ev_ebitda", None, NOT_REPORTED_REASONS["ev_ebitda"])
    _put(metrics, reasons, "fcf_yield", None, NOT_REPORTED_REASONS["fcf_yield"])

    for name in METRIC_ORDER:
        if name not in metrics:  # pragma: no cover - guarded by the meta-test
            metrics[name] = None
            reasons[name] = "metric_not_computed"

    return FundamentalSnapshot(
        ticker=ticker,
        as_of=moment,
        available=True,
        quarterly=_ref(quarter),
        ttm=_ref(ttm),
        metrics=metrics,
        reasons={**reasons, **_excluded_reasons(rows, moment)},
        notes=notes,
        quarters_available=len(quarterly_rows),
        price=price,
        market_cap=market_cap,
    )


# ---------------------------------------------------------------------------
# §29 — what changed since the previous event
# ---------------------------------------------------------------------------


def _classify_trend(series: Sequence[float]) -> tuple[str, int]:
    """``("improving"|"deteriorating"|"flat", n)`` over an oldest-first series.

    Deliberately first-vs-last rather than a fitted slope: with at most eight
    quarterly points a least-squares slope reads as more machinery than the
    sample supports, and its sign agrees with first-vs-last in every case a
    reader would call a trend. The tolerance is RELATIVE to the magnitude of
    the endpoints (:data:`TREND_TOLERANCE`), so a margin drifting 3 bps over
    two years is "flat" instead of a direction invented out of rounding.

    Fewer than two points is "flat" with the count, never a direction — one
    observation has no trend.
    """
    finite = [v for v in (_number(x) for x in series) if v is not None]
    n = len(finite)
    if n < 2:
        return "flat", n
    first, last = finite[0], finite[-1]
    scale = max(abs(first), abs(last))
    if scale == 0.0:
        return "flat", n
    change = (last - first) / scale
    if change > TREND_TOLERANCE:
        return "improving", n
    if change < -TREND_TOLERANCE:
        return "deteriorating", n
    return "flat", n


def snapshot_change(
    previous: FundamentalSnapshot | None,
    current: FundamentalSnapshot,
    *,
    metrics: Sequence[str] | None = None,
    history: Sequence[FundamentalSnapshot] | None = None,
    trend_window: int = DEFAULT_TREND_WINDOW,
) -> list[MetricChange]:
    """The §29 previous-vs-current table, one :class:`MetricChange` per metric.

    ``previous`` is the snapshot taken at the previous comparable event's
    instant — ``None`` when there was no previous event, which yields a full
    table of rows carrying the current value and the reason
    ``"no_previous_snapshot"`` rather than an empty list: the UI still shows
    every metric, it just cannot show a delta.

    ``history`` is an OLDEST-FIRST list of earlier snapshots used only for the
    trend column; the last ``trend_window`` points that have a value are
    classified by :func:`_classify_trend`. ``current`` is appended to that
    series automatically, so a caller passing the stored history gets a trend
    that ends where the table does.

    Rows come out in :data:`METRIC_ORDER`. Both-sides-present rows carry the
    delta; ratio metrics additionally carry ``delta_bps``.
    """
    if trend_window < 1:
        raise ValueError(f"trend_window must be >= 1, got {trend_window}")
    names = tuple(metrics) if metrics is not None else METRIC_ORDER
    history = list(history or [])
    out: list[MetricChange] = []

    for name in names:
        current_value = current.metrics.get(name)
        previous_value = previous.metrics.get(name) if previous is not None else None

        series = [
            value
            for value in (snap.metrics.get(name) for snap in history)
            if value is not None
        ]
        if current_value is not None:
            series.append(current_value)
        trend, points = _classify_trend(series[-trend_window:])

        note = current.notes.get(name)

        if previous is None:
            out.append(
                MetricChange(
                    metric=name,
                    current=current_value,
                    trend=trend if points >= 2 else None,
                    trend_points=points,
                    reason=(
                        "no_previous_snapshot"
                        if current_value is not None
                        else current.reasons.get(name, "no_previous_snapshot")
                    ),
                    note=note,
                )
            )
            continue

        if previous_value is None or current_value is None:
            if previous_value is None and current_value is None:
                reason = current.reasons.get(name) or previous.reasons.get(
                    name, "unavailable_on_both_sides"
                )
            elif previous_value is None:
                reason = (
                    "previous unavailable: "
                    + previous.reasons.get(name, "unavailable")
                )
            else:
                reason = (
                    "current unavailable: "
                    + current.reasons.get(name, "unavailable")
                )
            out.append(
                MetricChange(
                    metric=name,
                    previous=previous_value,
                    current=current_value,
                    trend=trend if points >= 2 else None,
                    trend_points=points,
                    reason=reason,
                    note=note,
                )
            )
            continue

        delta = _number(current_value - previous_value)
        delta_bps = (
            _number(delta * 10_000.0)
            if delta is not None and name in BPS_METRICS
            else None
        )
        pct_change = _growth(current_value, previous_value)
        if delta is None:
            direction = None
        elif delta > 0.0:
            direction = "up"
        elif delta < 0.0:
            direction = "down"
        else:
            direction = "flat"
        out.append(
            MetricChange(
                metric=name,
                previous=previous_value,
                current=current_value,
                delta=delta,
                delta_bps=delta_bps,
                pct_change=pct_change,
                direction=direction,
                trend=trend if points >= 2 else None,
                trend_points=points,
                note=note,
            )
        )
    return out


# ---------------------------------------------------------------------------
# §30 — valuation in context
# ---------------------------------------------------------------------------


def valuation_context(
    current_snapshot: FundamentalSnapshot,
    history_snapshots: Sequence[FundamentalSnapshot] | None = None,
) -> dict[str, Any]:
    """§30 valuation: each multiple against its OWN history, never in isolation.

    ``history_snapshots`` are earlier :class:`FundamentalSnapshot` values,
    each already valued with the price that prevailed at ITS date (the caller
    builds them from a list of ``(date, price)`` — this module reads no
    prices). For every multiple in :data:`VALUATION_METRICS` the block reports
    the current value, the historical median/min/max, the current value's
    percentile within that history and the sample size. A multiple with no
    current value is reported as ``{"available": false, "reason": ...}``
    carrying the snapshot's own reason string, and a multiple with an empty
    history keeps its current value while saying the history is absent — "we
    have a P/E but nothing to compare it to" is a different and more useful
    statement than "unavailable".

    The sector and peer comparisons §30 also asks for are explicitly
    ``{"available": false}`` with :data:`PEER_CONTEXT_REASON`. Silently
    omitting them would read as "not applicable"; inventing a peer set would
    be a fabricated number.
    """
    history = list(history_snapshots or [])
    multiples: dict[str, Any] = {}

    for name in VALUATION_METRICS:
        current_value = current_snapshot.metrics.get(name)
        series = [
            value
            for value in (snap.metrics.get(name) for snap in history)
            if value is not None
        ]
        block: dict[str, Any] = {
            "metric": name,
            "current": current_value,
            "history_n": len(series),
            "median": _median(series),
            "min": min(series) if series else None,
            "max": max(series) if series else None,
            "percentile": (
                percentile_of(series, current_value)
                if current_value is not None and series
                else None
            ),
        }
        if current_value is None:
            block["available"] = False
            block["reason"] = current_snapshot.reasons.get(
                name, "input_unavailable"
            )
        else:
            block["available"] = True
            if not series:
                block["history_reason"] = (
                    "no historical snapshot with this multiple — needs prior "
                    "statements priced at their own dates"
                )
        multiples[name] = block

    for name in ("ev_ebitda", "fcf_yield"):
        multiples[name] = {
            "metric": name,
            "available": False,
            "current": None,
            "reason": NOT_REPORTED_REASONS[name],
        }

    return {
        "as_of": current_snapshot.as_of,
        "price": current_snapshot.price,
        "market_cap": current_snapshot.market_cap,
        "multiples": multiples,
        "own_history": {
            "available": bool(history),
            "n": len(history),
            "reason": (
                None
                if history
                else "no earlier snapshots supplied for own-history context"
            ),
        },
        "sector": {"available": False, "reason": PEER_CONTEXT_REASON},
        "peers": {"available": False, "reason": PEER_CONTEXT_REASON},
        "provenance": "QUANT",
        "model_version": FUNDAMENTALS_MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# §35 — the deterministic half of the expectations gap
# ---------------------------------------------------------------------------

#: Metrics whose INCREASE is an improvement. Leverage and the valuation
#: multiples are deliberately absent: rising debt/equity is not "improving",
#: and a rising P/E is a valuation fact, not a fundamental one.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "revenue",
        "revenue_ttm",
        "revenue_growth_yoy",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "eps_diluted",
        "eps_diluted_ttm",
        "eps_growth_yoy",
        "operating_cash_flow",
        "operating_cash_flow_ttm",
        "roe_ttm",
        "roa_ttm",
        "current_ratio",
    }
)


def expectations_gap_inputs(
    changes: Sequence[MetricChange],
) -> dict[str, Any]:
    """The DETERMINISTIC inputs to the §35 expectations gap — counts only.

    No LLM, no narrative, no forecast: this counts how many of the
    directional metrics in :data:`_HIGHER_IS_BETTER` moved the right way and
    returns one of three labels — ``"fundamentals_improving"``,
    ``"fundamentals_weakening"`` or ``"fundamentals_mixed"`` — alongside the
    raw counts that produced it. A label with its own arithmetic attached can
    be argued with; a label without one is an opinion in a number's clothing.

    ``"fundamentals_unknown"`` when nothing was comparable, with the reason.
    Labelled QUANT (§91): this platform's counting, not a model's judgement.
    """
    improved = 0
    weakened = 0
    unchanged = 0
    unavailable = 0
    considered: list[str] = []

    for change in changes:
        if change.metric not in _HIGHER_IS_BETTER:
            continue
        considered.append(change.metric)
        if change.delta is None:
            unavailable += 1
        elif change.delta > 0.0:
            improved += 1
        elif change.delta < 0.0:
            weakened += 1
        else:
            unchanged += 1

    compared = improved + weakened + unchanged
    if compared == 0:
        label = "fundamentals_unknown"
        reason = (
            "no directional metric was comparable across the two snapshots"
        )
    else:
        reason = None
        if improved > weakened * 2 and improved >= 2:
            label = "fundamentals_improving"
        elif weakened > improved * 2 and weakened >= 2:
            label = "fundamentals_weakening"
        else:
            label = "fundamentals_mixed"

    return {
        "label": label,
        "reason": reason,
        "improved": improved,
        "weakened": weakened,
        "unchanged": unchanged,
        "unavailable": unavailable,
        "compared": compared,
        "metrics_considered": tuple(considered),
        "provenance": "QUANT",
        "model_version": FUNDAMENTALS_MODEL_VERSION,
    }
