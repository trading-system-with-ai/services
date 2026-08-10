"""Market data provider interface (development plan §22.1).

Consumers depend only on this Protocol. Concrete providers (the local stub
today, the MASSIVE-backed provider later) are selected by configuration via
:func:`libs.market_data.get_provider`, never imported directly by callers —
this keeps the provider swappable without touching consumer code.
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


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
