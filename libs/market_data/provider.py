"""Market data provider interface (development plan §22.1).

Consumers depend only on this Protocol. Concrete providers (the local stub
today, the MASSIVE-backed provider later) are selected by configuration via
:func:`libs.market_data.get_provider`, never imported directly by callers —
this keeps the provider swappable without touching consumer code.
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

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
