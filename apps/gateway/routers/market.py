"""Market overview API (development plan §22.1, §6.1).

Quotes come from the configured MarketDataProvider — Massive is the only
supported real source. With NO provider configured this endpoint answers 503
``MARKET_DATA_NOT_CONFIGURED`` and reports no numbers at all: an overview of
invented prices would be worse than no overview (§44 rule 18).
The market regime is COMPUTED by the Market Regime Engine
(libs.trading_core.signals.classify_regime, plan §6.1) from stored SPY daily
bars — no placeholder.

SPY/QQQ/VIX are system reference symbols: exempt from the watchlist-only
historical data rule (plan §4.2) because the Market Regime Engine requires
index data regardless of what the user watches (plan §6.1) — see ADR-005.
Their bars use the same lazy backfill path as watchlist analysis, so the first
overview request writes a SYSTEM-attributed DATA_BACKFILL audit event in the
same transaction as the inserted bars (rule 12); every later request is
read-only.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.config import get_settings
from libs.market_data import get_provider
from libs.trading_core.signals import classify_regime

from .. import market_stream
from ..db import get_session
from ..deps import require_market_data_provider
from .analysis import ensure_daily_bars

router = APIRouter(prefix="/api/market", tags=["market"])

# Headline dashboard indices: SPY/QQQ for direction, VIX for volatility.
# System reference symbols, exempt from the watchlist-only data rule (ADR-005).
INDEX_SYMBOLS = ["SPY", "QQQ", "VIX"]

# The regime engine reads the broad-market index (plan §6.1).
REGIME_SYMBOL = "SPY"

#: The §39 CROSS-ASSET REACTION set — the symbols a macro release is measured
#: against (ADR-005 reference symbols, ADR-009). SEPARATE from
#: :data:`INDEX_SYMBOLS` on purpose: that list is what the dashboard QUOTES
#: live (and includes VIX, an index with no tradable bars), while this one is
#: what the macro seam stores DAILY BARS for so a CPI print can be read across
#: equities, duration, gold, oil and the dollar. Merging them would put eight
#: extra REST quote calls on every dashboard poll and would ask the bar
#: backfill for VIX, which the equity providers do not serve.
#:
#: Every one of these is an ETF PROXY for the exposure named, never the
#: exposure itself — TLT is not "long rates", it is a fund holding long
#: Treasuries — and the pure layer labels them as proxies
#: (``libs.trading_core.events.macro.ASSET_ROLES`` / ``PROXY_ROLES``), which
#: is the same vocabulary this list is ordered by.
#:
#: DIA and VIXY were added (2026-08-23) because a macro release has no issuer
#: to attach to: a reader of a GDP print cannot look up "the stock", so the
#: index reaction IS the reader's instrument. DIA carries the Dow, and VIXY
#: stands in for the volatility index — VIX itself has no tradable bars, so
#: this is a proxy in the strict sense the roles below record, and one with a
#: known flaw worth stating: VIXY holds VIX FUTURES, so roll cost makes it
#: track VIX's DIRECTION faithfully but not its LEVEL. The UI badges it.
MACRO_REFERENCE_SYMBOLS = [
    "SPY",
    "DIA",
    "QQQ",
    "VIXY",
    "TLT",
    "IEF",
    "SHY",
    "GLD",
    "USO",
    "UUP",
]

#: REST snapshot quotes are the day-change BASELINE (they carry the previous
#: close); with the websocket streaming live prices, the baseline only needs
#: refreshing occasionally — this TTL bounds Alpaca REST calls to ~1/min
#: instead of one per UI poll (data_source.md §5). Keyed by provider so a
#: runtime switch never serves the old vendor's numbers.
REST_QUOTES_TTL_SECONDS = 60.0
_rest_quotes_cache: dict = {"at": None, "provider": None, "quotes": None}


def _rest_quotes_cached(provider_name: str):
    """Provider REST quotes, cached for REST_QUOTES_TTL_SECONDS.

    The stream (when live) supersedes prices anyway; this cache only slows
    the baseline churn. Errors propagate — an unanswerable provider is a
    real fault, never served from a stale cache silently.
    """
    now = datetime.now(timezone.utc)
    cached_at = _rest_quotes_cache["at"]
    if (
        _rest_quotes_cache["provider"] == provider_name
        and cached_at is not None
        and (now - cached_at).total_seconds() < REST_QUOTES_TTL_SECONDS
        and _rest_quotes_cache["quotes"] is not None
    ):
        return _rest_quotes_cache["quotes"]
    quotes = get_provider(provider_name).get_quotes(INDEX_SYMBOLS)
    _rest_quotes_cache.update({"at": now, "provider": provider_name, "quotes": quotes})
    return quotes


@router.get("/stream/status")
async def market_stream_status() -> dict:
    """The websocket stream's honest status (data_source.md §5/§35).

    Plain facts: connection state, feed, subscription set, message counts,
    and the freshness window consumers apply. "disabled" names the reason
    (non-Alpaca provider, missing credentials). Never 503s — the status of
    an optional transport is always reportable.
    """
    return market_stream.stream_status()


@router.get("/overview")
async def market_overview(session: AsyncSession = Depends(get_session)) -> dict:
    """Index quotes + the SPY-derived regime, or 503 when unconfigured.

    Everything in this payload is market data or computed from it, so with no
    provider configured there is nothing honest to return: the shared guard
    raises 503 ``MARKET_DATA_NOT_CONFIGURED`` before any number is produced.
    """
    require_market_data_provider()
    settings = get_settings()
    quotes = _rest_quotes_cached(settings.market_data_provider)

    # Real market regime from SPY daily bars (plan §6.1; ADR-005 exemption).
    bars = await ensure_daily_bars(session, REGIME_SYMBOL, settings.market_data_provider)
    regime = classify_regime(
        [b.close for b in bars],
        [b.high for b in bars],
        [b.low for b in bars],
    )

    # Streamed override (data_source.md §5): when the websocket has a FRESH
    # trade for a symbol, its price supersedes the (possibly cached) REST
    # snapshot. SAME provider, different transport — provenance never mixes;
    # per-index "transport" says which path served the number.
    indices = []
    for q in quotes:
        price, ts, transport = q.price, q.ts, "rest"
        change_pct = q.change_pct
        streamed = market_stream.CACHE.fresh(
            q.symbol, market_stream.STREAM_FRESH_SECONDS
        )
        if (
            streamed is not None
            and streamed.last_price is not None
            and streamed.trade_ts is not None
            and streamed.trade_ts >= q.ts
        ):
            # Re-base the day change on the SAME previous close the REST
            # quote used: prev_close = price / (1 + pct/100).
            denom = 1.0 + q.change_pct / 100.0
            prev_close = q.price / denom if denom > 0 else None
            price, ts, transport = streamed.last_price, streamed.trade_ts, "stream"
            if prev_close is not None and prev_close > 0:
                change_pct = (price / prev_close - 1.0) * 100.0
        indices.append(
            {
                "symbol": q.symbol,
                "price": price,
                "change_pct": change_pct,
                "ts": ts.isoformat(),
                "transport": transport,
            }
        )

    return {
        "provider": settings.market_data_provider,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "stale": False,
        "market_regime": regime.classification.value,
        "indices": indices,
    }


# ---------------------------------------------------------------------------
# Capability detection (guide §16): which parts of the data plan actually
# work, verified by probing the REAL API — never assumed from configuration.
# ---------------------------------------------------------------------------

# Probing costs real provider calls (three, for Massive), so the last verdict
# is served for this long before re-probing. ``?refresh=true`` bypasses it —
# e.g. right after the user upgrades their plan.
CAPABILITIES_TTL_SECONDS = 300.0

_capabilities_cache: dict = {"at": None, "provider": None, "payload": None}


@router.get("/capabilities")
async def market_capabilities(refresh: bool = False) -> dict:
    """Live provider entitlements (guide §16), probed — not assumed.

    ``{"provider", "as_of", "capabilities", "message"}`` where
    ``capabilities`` maps each capability to ``true`` (verified working),
    ``false`` (the plan does not include it) or an error string (the probe
    itself failed — which is a fault, NOT evidence of absence). ``null`` with
    a message when the configured provider cannot probe (the stub has no plan
    to detect). 503 when no provider is configured at all.

    The UI uses this to disable features gracefully instead of letting them
    fail deep in a workflow — e.g. options selection greys out when
    ``option_chain`` is false, with the reason visible.
    """
    require_market_data_provider()
    settings = get_settings()
    now = datetime.now(timezone.utc)

    cached_at = _capabilities_cache["at"]
    if (
        not refresh
        and cached_at is not None
        and _capabilities_cache["provider"] == settings.market_data_provider
        and (now - cached_at).total_seconds() < CAPABILITIES_TTL_SECONDS
    ):
        return _capabilities_cache["payload"]

    provider = get_provider(settings.market_data_provider)
    probe = getattr(provider, "probe_capabilities", None)
    if probe is None:
        payload = {
            "provider": settings.market_data_provider,
            "as_of": now.isoformat(),
            "capabilities": None,
            "message": (
                f"provider '{settings.market_data_provider}' does not support "
                "capability probing; nothing is claimed about its entitlements"
            ),
        }
    else:
        payload = {
            "provider": settings.market_data_provider,
            "as_of": now.isoformat(),
            "capabilities": probe(),
            "message": None,
        }

    _capabilities_cache.update(
        {"at": now, "provider": settings.market_data_provider, "payload": payload}
    )
    return payload
