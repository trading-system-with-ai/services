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
from datetime import date, datetime
from typing import Protocol

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
class Bar:
    """One daily OHLCV bar for one symbol (plan §4.2)."""

    ts: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(Protocol):
    """Structural interface every market data provider must satisfy."""

    def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Return current quotes for the given symbols, one Quote per symbol."""
        ...

    def get_daily_bars(self, symbol: str, days: int) -> list[Bar]:
        """Return the last `days` daily bars for `symbol`, oldest first."""
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
