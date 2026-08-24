"""Market data provider interface (development plan §22.1).

Consumers depend only on this Protocol. Concrete providers (the MASSIVE-backed
provider; the local stub for development and tests) are selected by
configuration via :func:`libs.market_data.get_provider`, never imported
directly by callers — this keeps the provider swappable without touching
consumer code.

NO PROVIDER, NO DATA (§44 rule 18): there is no default provider. When
``MARKET_DATA_PROVIDER`` is unset the registry raises
:class:`ProviderNotConfigured` and every consumer surfaces that as an honest
503 — an unconfigured install must NEVER be served synthetic numbers that
look like real market data.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping, Protocol, Sequence

# The message every unconfigured-market-data path reports verbatim, so the API
# error, the logs and the tests all name the SAME missing configuration.
MARKET_DATA_NOT_CONFIGURED_MESSAGE = (
    "market data provider is not configured — set MARKET_DATA_PROVIDER and "
    "the corresponding credentials"
)


class ProviderNotConfigured(RuntimeError):
    """No market data provider is configured (``MARKET_DATA_PROVIDER`` unset).

    Deliberately NOT a ``ValueError``: an unknown provider name is a
    misconfiguration the operator typed, while this is the absence of any
    configuration at all — the state a fresh install starts in. Callers map it
    to HTTP 503 ``MARKET_DATA_NOT_CONFIGURED`` and show nothing (§44 rule 18),
    never a synthetic fallback.
    """

    def __init__(self, message: str = MARKET_DATA_NOT_CONFIGURED_MESSAGE) -> None:
        super().__init__(message)

class MarketDataError(RuntimeError):
    """A market data request could not be answered honestly.

    Transport faults, HTTP errors, unparseable payloads and missing provider
    credentials all land here. Callers surface it as an explicit error — NEVER
    a synthetic fallback (§44 rule 18): when the real provider cannot answer,
    the platform shows nothing rather than numbers that look real but are not.
    """


class CapabilityNotAvailable(MarketDataError):
    """The configured plan/subscription does not include the endpoint (HTTP 403).

    Distinct from :class:`MarketDataError` so callers can tell "the provider
    is broken / unreachable" apart from "the provider works but this
    capability was never purchased" (§16 capability detection). Either way
    the answer is an explicit refusal naming the endpoint — never made-up
    numbers standing in for the missing capability.
    """


# One option contract snapshot (plan §9). This is deliberately the SAME class
# as libs.trading_core.contracts.ContractQuote — imported and aliased, never
# redefined — so provider output feeds select_contracts() directly with no
# translation layer and the two shapes can never drift apart.
from libs.trading_core.contracts import ContractQuote as OptionQuote  # noqa: F401


@dataclass(frozen=True)
class Quote:
    """A point-in-time quote for one symbol."""

    symbol: str
    price: float
    change_pct: float
    ts: datetime


@dataclass(frozen=True)
class NewsArticle:
    """One REAL news article from the provider (Phase 8 news ingestion).

    ``source_id`` is the provider's own article id — the deduplication key.
    Every field is the provider's data verbatim; nothing here is generated.
    """

    source_id: str
    title: str
    publisher: str
    published_at: datetime
    url: str
    tickers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class Bar:
    """One daily OHLCV bar for one symbol (plan §4.2)."""

    ts: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class IntradayBar:
    """One INTRADAY OHLCV bar for one symbol (Phase C event replay, §20).

    Deliberately separate from :class:`Bar`: a daily bar is keyed by a trading
    ``date`` (its Eastern session), while an intraday bar is keyed by an
    INSTANT — ``ts`` is the aware-UTC start of the bar's interval, because the
    whole point of event replay is placing a price relative to a release
    timestamp, and a date cannot do that. Consumers convert to Eastern
    themselves to decide pre-market / regular / after-hours.

    ``volume`` is an ``int`` (shares actually traded in the interval), not the
    float ``Bar.volume`` carries: providers report minute volume as a whole
    share count, and the store it lands in (``stock_bars_1m.volume``) is
    BIGINT. A bar the provider reported with no volume field is dropped by the
    adapters rather than volume-zeroed, so 0 here always means "the provider
    said zero trades", never "we did not know".
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class FinancialStatement:
    """One filed financial statement period, exactly as the provider reported it.

    THE POINT-IN-TIME KEY IS ``acceptance_datetime`` (audit §7.1, spec §85/§96):
    the instant the filing became public. ``end_date`` is when the PERIOD
    ended, which is weeks earlier — an as-of filter written against it would
    let a Q3 statement inform an analysis run before Q3 was ever filed. Every
    consumer filters on ``acceptance_datetime <= as_of`` and NOTHING else.
    It is optional because a provider row may omit it; a row without it cannot
    be placed in time and callers exclude it with an explicit reason rather
    than guessing from ``filing_date``.

    ``values`` is the provider's own statement fields FLATTENED to
    ``"<statement>.<field>" -> float`` (e.g. ``"income_statement.revenues"``),
    carrying only numeric values. Fields the filer did not report are ABSENT
    from the mapping — never present as 0.0 — so a consumer can tell "reported
    zero" from "not reported" and answer the second case with a reason
    (§44 rule 18). ``raw_fields_count`` is how many fields the provider row
    carried before the numeric filter, so a mostly-unparseable row is
    detectable without re-fetching.
    """

    ticker: str
    cik: str | None
    timeframe: str
    fiscal_year: int | None
    fiscal_period: str
    start_date: date
    end_date: date
    filing_date: date | None
    acceptance_datetime: datetime | None
    source_filing_url: str | None
    values: Mapping[str, float]
    raw_fields_count: int


def require_aware_utc(value: datetime, name: str) -> datetime:
    """`value` as an aware UTC instant; ``ValueError`` if it is naive.

    Shared by every :meth:`MarketDataProvider.get_intraday_bars`
    implementation so the rejection is IDENTICAL across providers — the stub
    must refuse exactly what Alpaca refuses, or a window that works in tests
    would silently shift by hours in production. A naive datetime is a
    caller bug, not data: there is no correct zone to assume for it (UTC and
    Eastern differ by 4-5 hours, which is the difference between an
    after-hours release and the next morning's open), so it raises rather
    than guessing. An aware non-UTC input is fine and is converted.
    """
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{name} must be timezone-aware (UTC) — a naive datetime cannot be "
            "placed on the clock and will NOT be assumed to be UTC or Eastern"
        )
    return value.astimezone(timezone.utc)


class MarketDataProvider(Protocol):
    """Structural interface every market data provider must satisfy."""

    def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Return current quotes for the given symbols, one Quote per symbol."""
        ...

    def get_daily_bars(self, symbol: str, days: int) -> list[Bar]:
        """Return the last `days` daily bars for `symbol`, oldest first."""
        ...

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
    ) -> list[IntradayBar]:
        """Intraday bars for `symbol` covering ``[start, end]``, oldest first.

        The window is an INSTANT range, and `start` / `end` MUST be timezone
        AWARE — a naive datetime raises ``ValueError`` rather than being
        assumed UTC or Eastern. Event replay lives or dies on placing a bar
        relative to a release timestamp, and silently guessing a zone would
        shift every window by four or five hours depending on the season,
        producing plausible-looking but wrong reactions.

        Contract every implementation honours:

        - ascending by ``ts``, DE-DUPLICATED (a ts appearing on two pages is
          kept once — the first parse wins);
        - PAGINATED FULLY: the caller receives the whole window, never a
          silently truncated first page;
        - ``[]`` when the provider serves no bars for the window (a holiday, a
          halted symbol, a range before the symbol listed) — an honest absence,
          never synthesized minutes;
        - HTTP 403 -> :class:`CapabilityNotAvailable` naming the endpoint, and
          a provider without an intraday endpoint at all raises it too. A
          missing capability is reported, never filled in (§16, §44 rule 18).

        Bars OUTSIDE the regular session (pre-market and after-hours) are
        included when the provider serves them: an after-market earnings
        release moves the stock at 16:05 ET, so dropping extended-hours bars
        would discard the very reaction being replayed. Consumers classify
        each bar's session from its own Eastern time.
        """
        ...

    def get_option_chain(
        self, symbol: str, spot: float, as_of: date
    ) -> list[OptionQuote]:
        """Return the option chain snapshot for `symbol` as of `as_of` (plan §9).

        `spot` is the underlying reference price the chain is built around
        (the caller's last stored close). Rows are
        :class:`OptionQuote` — an alias of
        ``libs.trading_core.contracts.ContractQuote`` — ready for
        ``select_contracts``. Chain reads are read-only (house rule: no
        audit events on reads).
        """
        ...

    def get_financials(
        self, ticker: str, *, timeframe: str = "quarterly", limit: int = 12
    ) -> list["FinancialStatement"]:
        """Filed financial statements for `ticker`, NEWEST FIRST (§28, audit §11.3).

        `timeframe` is the provider's own period vocabulary — ``"quarterly"``,
        ``"annual"`` or ``"ttm"`` — and `limit` caps how many periods are
        returned. Rows are the provider's filings verbatim; nothing here is
        derived, and a provider that serves no statements for `ticker` returns
        ``[]`` (an honest absence), never a synthesized period.

        A provider whose plan does not include fundamentals raises
        :class:`CapabilityNotAvailable` naming the endpoint (§16) — the answer
        to a missing capability is that refusal, never invented numbers.
        """
        ...

    def get_news_window(
        self,
        *,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> list[NewsArticle]:
        """Every article for `tickers` published in ``[start, end]``, NEWEST FIRST.

        This is the WINDOWED sibling of :meth:`get_news`, and the difference
        is the whole point (Phase D §21-§27): ``get_news`` answers "what is on
        the wire right now", which is useless for an event whose window closed
        last week. News evidence is assembled for a SPECIFIC event window, so
        the window — not a recency cursor — is the query.

        Contract every implementation honours:

        - `start` / `end` MUST be timezone AWARE; a naive datetime raises
          ``ValueError`` via :func:`require_aware_utc` rather than being
          assumed UTC or Eastern. The same reasoning as
          :meth:`get_intraday_bars`: guessing a zone shifts the window by
          four or five hours, which is the difference between an article
          published before a release and one published after it — and that
          difference is exactly what the evidence engine reads. A reversed
          window (``end < start``) raises too: an empty window is expressed
          as ``start == end``.
        - PAGINATED TO EXHAUSTION, bounded by `limit`: the caller receives the
          whole window rather than a silently truncated first page. `limit`
          caps how many articles come back in total, and pagination stops as
          soon as it is reached.
        - DE-DUPLICATED BY ``source_id``, first parse wins. The same article
          arrives twice when a page boundary repeats it and when two of the
          requested tickers are both tagged on it, and one article is one
          article — counting it twice would inflate every downstream evidence
          count.
        - Sorted NEWEST FIRST (descending ``published_at``), so a caller that
          truncates keeps the most recent news, matching ``get_news``.
        - ``[]`` when the provider serves nothing for the window (a quiet
          ticker, a window before the symbol listed) — an honest absence,
          never synthesized articles.
        - HTTP 403 -> :class:`CapabilityNotAvailable` naming the endpoint, and
          a provider with no news endpoint at all raises it too. A missing
          capability is reported, never filled in (§16, §44 rule 18).

        Rows are the provider's data verbatim, exactly as :meth:`get_news`
        returns them; rows missing a citable field (id, title, url, timestamp)
        are SKIPPED rather than patched, because an uncitable article cannot
        ground anything. Filtering, clustering and scoring are the evidence
        engine's job, not the adapter's: an off-topic article the provider
        tagged with the ticker is returned as-is, and the caller decides.
        """
        ...


# ----------------------------------------------------------------------
# Historical option capability (Phase I §18/§36-§37 implied move)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class OptionContractRef:
    """One option contract's IDENTITY — what existed, never what it cost.

    Deliberately price-free and greek-free, unlike :class:`OptionQuote`: this
    is the answer to "which contracts were listed on `underlying` for that
    expiry, as of that date", which is the FIRST question the §36 implied-move
    pipeline asks. Prices for the contract are a separate call
    (:meth:`MarketDataProvider.get_option_history_bars`) so a provider that
    can list contracts but not price them fails at the step that is actually
    missing, instead of returning rows with zero-filled premiums.

    ``ticker`` is the provider's own contract symbol verbatim (Massive serves
    the ``O:``-prefixed OCC form, e.g. ``"O:AAPL250801C00210000"``); it is
    passed back to the bars call unmodified rather than re-derived, because a
    reconstructed symbol that differs by one zero silently fetches nothing.
    ``expiry``/``right``/``strike`` are the provider's parsed identity fields,
    with ``right`` normalised to ``"C"``/``"P"`` to match
    :class:`OptionQuote.right`.
    """

    ticker: str
    underlying: str
    expiry: date
    right: str
    strike: float


class HistoricalOptionProvider(Protocol):
    """The historical-options half of the provider interface (Phase I §36).

    Split out from :class:`MarketDataProvider` as its own Protocol because it
    is genuinely OPTIONAL: Alpaca sells option snapshots but no dated contract
    reference, so it satisfies the base interface and refuses this one. A
    consumer that needs point-in-time option history states that by typing
    against this Protocol, and the refusal it gets is
    :class:`CapabilityNotAvailable` naming the endpoint — never an empty list
    that would read as "this contract never traded".
    """

    def list_option_contracts(
        self,
        underlying: str,
        *,
        expiration_date: date,
        as_of: date,
        right: str | None = None,
        limit: int = 250,
    ) -> list[OptionContractRef]:
        """Contracts on `underlying` expiring `expiration_date`, listed AS OF `as_of`.

        ``as_of`` is the point-in-time key and the reason this method exists
        separately from the chain snapshot: the §36 pre-event straddle must be
        built from contracts that existed BEFORE the event, and asking today's
        universe would quietly admit strikes listed in reaction to the event
        itself. Providers pass it to the server rather than filtering locally,
        so the answer is the vendor's own point-in-time record.

        `right` filters to ``"C"``/``"P"`` (case-insensitive; ``"call"``/
        ``"put"`` accepted) and ``None`` returns both. `limit` is the
        server page size, not a result cap — implementations follow pagination
        to exhaustion so a caller never mistakes a first page for the whole
        expiry.

        Returns ``[]`` when no contracts were listed for that expiry as of
        that date — an honest absence (a date before the underlying had
        options, an expiry that is not a listed one). Rows missing a usable
        ticker, strike, right or expiry are SKIPPED rather than patched.
        HTTP 403, and a provider with no dated-reference endpoint at all,
        raise :class:`CapabilityNotAvailable` naming the endpoint (§16).
        """
        ...

    def get_option_history_bars(
        self, option_ticker: str, start: date, end: date
    ) -> list[Bar]:
        """DAILY bars for ONE option contract over ``[start, end]``, oldest first.

        Named ``get_option_history_bars`` rather than ``get_option_daily_bars``
        on purpose: the latter name is already taken across this package by the
        options-BACKTEST price source, whose contract is a
        ``{date: (open, close)}`` mapping keyed by BARE OCC symbols. Reusing
        that name for a different return shape would break the backtest
        resolver at runtime with no type error to catch it, so the new
        capability gets a new name and the old one is left exactly as it is.

        Rows are the full :class:`Bar` (the same dataclass daily stock bars
        use), with ``ts`` the bar's Eastern trading date. ``[]`` means the
        contract served no bars in the window — an illiquid contract that
        never traded, or a window before it was listed. That absence is the
        answer: the §36 pipeline reports ``NO_DATA`` for the straddle rather
        than pricing a missing leg at zero.

        HTTP 403, and a provider with no historical option aggregates at all,
        raise :class:`CapabilityNotAvailable` naming the endpoint (§16).
        """
        ...
