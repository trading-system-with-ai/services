"""Alpaca market data streaming service (data_source.md §5).

A background asyncio task (started by the gateway lifespan, like the
position monitor) that keeps ONE websocket connection to Alpaca's stock
stream and feeds the in-memory :class:`QuoteCache`. REST remains the source
for history/backfill/recovery; the stream replaces the per-poll REST
snapshot calls for CURRENT prices (§5: "The system should NOT continuously
poll REST endpoints for values that can be streamed").

Honesty rules carried over:
- the stream runs ONLY when the configured market-data provider is
  "alpaca" (checked every cycle, so runtime provider switches take effect
  without a restart); any other provider -> state "disabled";
- a broken/quiet stream degrades to the REST path — SAME provider, so
  provenance never mixes (§33 forbids cross-provider fallback, not
  transport fallback within one provider);
- the cache serves only what the socket delivered; freshness is the
  consumer's check (stale = absent).

Subscriptions: watchlist tickers + the overview indices (SPY/QQQ),
re-diffed every SUBSCRIPTION_REFRESH_SECONDS so watchlist changes follow.
VIX is excluded — Alpaca serves no index feed (honest absence).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from libs.market_data.alpaca_stream import (
    ALPACA_STOCK_STREAM_URL,
    DEFAULT_STOCK_FEED,
    QuoteCache,
    auth_message,
    parse_stream_payload,
    subscribe_message,
    unsubscribe_message,
)

logger = logging.getLogger(__name__)

#: The one process-wide live-quote cache; consumers import this.
CACHE = QuoteCache()

#: A streamed value older than this is not "current" for display overrides.
STREAM_FRESH_SECONDS = 30.0

#: How often the supervisor re-reads settings / the connection re-diffs its
#: subscription set against the watchlist.
SUPERVISOR_INTERVAL_SECONDS = 15.0
SUBSCRIPTION_REFRESH_SECONDS = 60.0

#: Reconnect backoff bounds (doubles between attempts).
BACKOFF_MIN_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0

#: Symbols always streamed alongside the watchlist (overview indices; VIX is
#: not servable on Alpaca and is never subscribed).
BASE_SYMBOLS = ("SPY", "QQQ")

#: Status surface for GET /api/market/stream/status — plain facts only.
STATUS: dict = {
    "state": "starting",  # starting | disabled | connecting | connected | error
    "feed": None,
    "connected_at": None,
    "last_message_at": None,
    "messages_applied": 0,
    "subscribed": [],
    "error": None,
}


async def _desired_symbols() -> list[str]:
    """Watchlist tickers + base indices, deduped and sorted."""
    from .db import SessionLocal, WatchlistItem

    async with SessionLocal() as session:
        rows = await session.execute(select(WatchlistItem.ticker))
        tickers = {t for (t,) in rows.all()}
    return sorted(tickers | set(BASE_SYMBOLS))


def _stream_wanted() -> tuple[bool, str, str, str]:
    """(wanted, key_id, secret, reason) from CURRENT settings."""
    from libs.common.config import get_settings

    settings = get_settings()
    if settings.market_data_provider != "alpaca":
        return False, "", "", (
            f"market data provider is {settings.market_data_provider!r} — "
            "streaming is Alpaca-only"
        )
    if not settings.alpaca_api_key_id or not settings.alpaca_api_secret_key:
        return False, "", "", "Alpaca credentials not configured"
    return True, settings.alpaca_api_key_id, settings.alpaca_api_secret_key, ""


async def _run_connection(key_id: str, secret: str) -> None:
    """One websocket session: auth, subscribe, read until failure/cancel."""
    import websockets

    url = ALPACA_STOCK_STREAM_URL.format(feed=DEFAULT_STOCK_FEED)
    STATUS.update({"state": "connecting", "feed": DEFAULT_STOCK_FEED, "error": None})
    subscribed: set[str] = set()

    async with websockets.connect(url, max_size=2**22) as ws:
        # connected banner -> auth -> authenticated ack.
        await ws.recv()
        await ws.send(auth_message(key_id, secret))
        ack = parse_stream_payload(await ws.recv())
        if not any(m.get("T") == "success" and m.get("msg") == "authenticated" for m in ack):
            raise RuntimeError(f"stream auth rejected: {ack[:1]}")
        STATUS.update({
            "state": "connected",
            "connected_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("alpaca stream connected (feed=%s)", DEFAULT_STOCK_FEED)

        last_refresh = 0.0
        loop = asyncio.get_running_loop()
        while True:
            # Periodic subscription diff (watchlist may have changed).
            now = loop.time()
            if now - last_refresh >= SUBSCRIPTION_REFRESH_SECONDS or not subscribed:
                last_refresh = now
                desired = set(await _desired_symbols())
                to_add = sorted(desired - subscribed)
                to_drop = sorted(subscribed - desired)
                if to_add:
                    await ws.send(subscribe_message(to_add))
                if to_drop:
                    await ws.send(unsubscribe_message(to_drop))
                if to_add or to_drop:
                    subscribed = desired
                    STATUS["subscribed"] = sorted(subscribed)
                    logger.info(
                        "alpaca stream subscription: +%d -%d (%d total)",
                        len(to_add), len(to_drop), len(subscribed),
                    )

            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue  # quiet market — loop for the next refresh window
            messages = parse_stream_payload(frame)
            for m in messages:
                if m.get("T") == "error":
                    raise RuntimeError(f"stream error message: {m}")
            CACHE.apply(messages)
            STATUS["messages_applied"] = CACHE.messages_applied
            STATUS["last_message_at"] = datetime.now(timezone.utc).isoformat()


async def market_stream_loop() -> None:
    """Supervisor: (re)connect while the provider is alpaca; idle otherwise.

    Never raises (lifespan cancellation contract): every failure is recorded
    on STATUS and retried with bounded backoff.
    """
    backoff = BACKOFF_MIN_SECONDS
    while True:
        try:
            wanted, key_id, secret, reason = _stream_wanted()
            if not wanted:
                if STATUS["state"] != "disabled":
                    CACHE.clear()
                    STATUS.update({
                        "state": "disabled", "error": reason,
                        "subscribed": [], "connected_at": None,
                    })
                await asyncio.sleep(SUPERVISOR_INTERVAL_SECONDS)
                continue
            await _run_connection(key_id, secret)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            STATUS.update({"state": "error", "error": f"{type(exc).__name__}: {exc}"})
            logger.warning(
                "alpaca stream disconnected (%s); reconnecting in %.0fs",
                exc, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX_SECONDS)
        else:  # pragma: no cover — _run_connection only exits by raising
            await asyncio.sleep(backoff)
        # A session that lived long enough to connect resets the backoff.
        if STATUS.get("state") == "connected":
            backoff = BACKOFF_MIN_SECONDS


def stream_status() -> dict:
    """Facts for the status endpoint — never more than what happened."""
    return {
        **STATUS,
        "messages_applied": CACHE.messages_applied,
        "cached_symbols": CACHE.symbols(),
        "fresh_window_seconds": STREAM_FRESH_SECONDS,
    }
