"""Market data provider interface (development plan §22.1).

Consumers depend only on this Protocol. Concrete providers (the local stub
today, the MASSIVE-backed provider later) are selected by configuration via
:func:`libs.market_data.get_provider`, never imported directly by callers —
this keeps the provider swappable without touching consumer code.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Quote:
    """A point-in-time quote for one symbol."""

    symbol: str
    price: float
    change_pct: float
    ts: datetime


class MarketDataProvider(Protocol):
    """Structural interface every market data provider must satisfy."""

    def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Return current quotes for the given symbols, one Quote per symbol."""
        ...
