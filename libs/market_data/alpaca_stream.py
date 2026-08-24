"""Alpaca market data STREAM protocol + live quote cache (data_source.md §5).

The pure, testable half of the streaming layer: message parsing and the
in-memory latest-quote store. The async connection loop (websockets, auth,
subscription management) lives in ``apps/gateway/market_stream.py`` — this
module never opens a socket.

Protocol (verified against https://docs.alpaca.markets/docs/streaming-market-data):
the server speaks JSON ARRAYS of messages, each tagged ``T``:

  - ``{"T":"success","msg":"connected"|"authenticated"}``
  - ``{"T":"error","code":...,"msg":...}``
  - ``{"T":"subscription","trades":[...],"quotes":[...],...}``
  - quote:  ``{"T":"q","S":sym,"bp":..,"bs":..,"ap":..,"as":..,"t":RFC3339}``
  - trade:  ``{"T":"t","S":sym,"p":..,"s":..,"t":RFC3339}``
  - minute bar: ``{"T":"b",...}`` (ignored for now — bars stay REST/stored)

STREAM DATA IS STILL ALPACA RAW DATA (§25 provenance: transport differs,
source does not). NO fabrication: unparseable messages are dropped with a
debug log; the cache serves only what the socket actually delivered, and
consumers must check freshness — a quiet market (or closed session) yields
an honestly EMPTY/stale cache, never invented ticks.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

#: Stocks stream host; Algo Trader Plus is entitled to the SIP feed.
ALPACA_STOCK_STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
DEFAULT_STOCK_FEED = "sip"


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ts(value: object) -> datetime | None:
    """RFC-3339 (possibly nanosecond) timestamp -> aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    raw = value.replace("Z", "+00:00")
    if "." in raw:
        head, _, tail = raw.partition(".")
        frac = tail[:-6] if tail.endswith("+00:00") else tail
        offset = "+00:00" if tail.endswith("+00:00") else ""
        if len(frac) > 6:
            raw = f"{head}.{frac[:6]}{offset}"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_stream_payload(raw: str | bytes) -> list[dict]:
    """One websocket frame -> list of message dicts (Alpaca sends arrays).

    Anything unparseable returns [] with a debug log — a malformed frame is
    dropped, never guessed at.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        logger.debug("alpaca stream: unparseable frame dropped")
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    return [m for m in payload if isinstance(m, dict)]


@dataclass
class StreamQuote:
    """Latest streamed state for one symbol — Alpaca raw values only."""

    symbol: str
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    quote_ts: datetime | None = None
    last_price: float | None = None
    last_size: float | None = None
    trade_ts: datetime | None = None
    #: When OUR process last applied a message for this symbol (freshness).
    updated_at: datetime | None = None


@dataclass
class QuoteCache:
    """In-memory latest-quote store fed by the stream loop.

    Single-event-loop use (the gateway); no locking. ``messages_applied``
    counts real data messages (quotes+trades), for the status surface.
    """

    _by_symbol: dict[str, StreamQuote] = field(default_factory=dict)
    messages_applied: int = 0

    def apply(self, messages: list[dict], now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        for m in messages:
            kind = m.get("T")
            symbol = m.get("S")
            if kind not in ("q", "t") or not isinstance(symbol, str) or not symbol:
                continue
            entry = self._by_symbol.setdefault(symbol, StreamQuote(symbol=symbol))
            if kind == "q":
                entry.bid = _as_float(m.get("bp"))
                entry.ask = _as_float(m.get("ap"))
                entry.bid_size = _as_float(m.get("bs"))
                entry.ask_size = _as_float(m.get("as"))
                entry.quote_ts = _ts(m.get("t"))
            else:
                price = _as_float(m.get("p"))
                if price is None or price <= 0:
                    continue
                entry.last_price = price
                entry.last_size = _as_float(m.get("s"))
                entry.trade_ts = _ts(m.get("t"))
            entry.updated_at = now
            self.messages_applied += 1

    def get(self, symbol: str) -> StreamQuote | None:
        return self._by_symbol.get(symbol)

    def fresh(
        self, symbol: str, max_age_seconds: float, now: datetime | None = None
    ) -> StreamQuote | None:
        """The symbol's streamed state, ONLY if our process applied a message
        for it within `max_age_seconds` — else None (stale = absent)."""
        entry = self._by_symbol.get(symbol)
        if entry is None or entry.updated_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        if (now - entry.updated_at).total_seconds() > max_age_seconds:
            return None
        return entry

    def clear(self) -> None:
        self._by_symbol.clear()

    def symbols(self) -> list[str]:
        return sorted(self._by_symbol)


def auth_message(key_id: str, secret_key: str) -> str:
    return json.dumps({"action": "auth", "key": key_id, "secret": secret_key})


def subscribe_message(symbols: list[str]) -> str:
    """Subscribe to trades + quotes for `symbols` (bars stay REST/stored)."""
    return json.dumps(
        {"action": "subscribe", "trades": symbols, "quotes": symbols}
    )


def unsubscribe_message(symbols: list[str]) -> str:
    return json.dumps(
        {"action": "unsubscribe", "trades": symbols, "quotes": symbols}
    )
