"""Deterministic SYNTHETIC prediction-market provider — development/tests only.

OPT-IN ONLY, NEVER A FALLBACK (§44 rule 18): reachable only by explicitly
setting ``PREDICTION_MARKETS_PROVIDER=stub``. An unconfigured install reports
prediction markets honestly unavailable, and no code path substitutes the
stub when Polymarket cannot answer.

Determinism: every field derives from ``zlib.crc32`` of the query/market id
(the :mod:`libs.llm.stub` seam), and every timestamp derives from caller
inputs or the ``STUB_ANCHOR_DATE`` seam — frozen by the test harness
(conftest), today (UTC) in a live dev install — the exact discipline
``libs/market_data/stub.py::_default_series_end`` set, so tests are
byte-identical across runs while a dev install's markets stay current.

The synthetic shape exercises the honesty paths downstream code must handle:

- market index 1 reports ``volume=None``/``liquidity=None`` (the thin-market
  path the interpretation layer must weight down);
- ``get_price_history`` returns an EMPTY list for outcome names the market
  does not carry (honest absence, never an invented series);
- questions are prefixed ``[SYNTHETIC]`` so a stub row that ever reached a
  real table is immediately recognizable.
"""
import zlib
from datetime import date, datetime, time, timedelta, timezone

from .provider import (
    CAPABILITY_KEYS,
    MARKET_STATUS_ACTIVE,
    MarketOutcome,
    MarketSnapshot,
    PredictionMarketError,
    PredictionMarketInfo,
    PricePoint,
)


def _anchor() -> datetime:
    """Midnight UTC of ``STUB_ANCHOR_DATE`` when set (the frozen test seam) —
    else now (UTC). Resolved at CALL time, never import time, so a test that
    changes settings is honored (the market-data stub's seam discipline)."""
    from libs.common.config import get_settings

    anchor = get_settings().stub_anchor_date
    if anchor:
        try:
            return datetime.combine(
                date.fromisoformat(anchor), time(0, 0), tzinfo=timezone.utc
            )
        except ValueError:
            pass  # a malformed anchor must never break the provider
    return datetime.now(timezone.utc)


def _price_from(seed: int) -> float:
    """A deterministic contract price in [0.05, 0.95]."""
    return round(0.05 + (seed % 9001) / 10000.0, 4)


class StubPredictionMarketProvider:
    """Deterministic synthetic markets (see module docstring)."""

    name = "stub"

    def capabilities(self) -> dict[str, bool | str]:
        return {k: True for k in CAPABILITY_KEYS}

    def search_markets(
        self,
        query: str,
        *,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[PredictionMarketInfo]:
        if limit <= 0:
            return []
        seed = zlib.crc32(query.encode("utf-8"))
        markets = []
        for i in range(min(limit, 2)):
            markets.append(self.get_market(f"stub-market-{seed}-{i}"))
        return markets

    def get_market(self, market_id: str) -> PredictionMarketInfo:
        if not market_id.startswith("stub-market-"):
            raise PredictionMarketError(
                f"unknown stub market id {market_id!r} — the stub only serves "
                "ids it minted itself"
            )
        h = zlib.crc32(market_id.encode("utf-8"))
        yes = _price_from(h)
        thin = market_id.endswith("-1")  # index 1: the thin-market path
        return PredictionMarketInfo(
            provider=self.name,
            market_id=market_id,
            provider_event_id=f"stub-event-{h % 1000}",
            question=f"[SYNTHETIC] Will outcome {h % 100} occur by year end?",
            url=f"https://stub-markets.example/{market_id}",
            outcomes=(
                MarketOutcome(name="Yes", price=yes),
                MarketOutcome(name="No", price=round(1.0 - yes, 4)),
            ),
            resolution_criteria=(
                None if thin else "[SYNTHETIC] Resolves YES on stub criterion."
            ),
            end_date=_anchor() + timedelta(days=30 + h % 60),
            status=MARKET_STATUS_ACTIVE,
            volume=None if thin else float(h % 500_000),
            liquidity=None if thin else float(h % 100_000),
            raw={"stub": True},
        )

    def get_market_snapshot(self, market_id: str) -> MarketSnapshot:
        info = self.get_market(market_id)  # validates the id
        h = zlib.crc32(f"snapshot|{market_id}".encode("utf-8"))
        mid = _price_from(h)
        half_spread = 0.005 + (h % 20) / 1000.0
        bid = round(max(0.01, mid - half_spread), 4)
        ask = round(min(0.99, mid + half_spread), 4)
        thin = market_id.endswith("-1")
        return MarketSnapshot(
            provider=self.name,
            market_id=market_id,
            observed_at=_anchor(),
            outcome_prices={"Yes": mid, "No": round(1.0 - mid, 4)},
            best_bid=bid,
            best_ask=ask,
            midpoint=round((bid + ask) / 2.0, 4),
            spread=round(ask - bid, 4),
            last_trade_price=mid,
            volume=info.volume,
            liquidity=info.liquidity,
            open_interest=None,  # the stub never states OI — the absent path
        )

    def get_price_history(
        self,
        market_id: str,
        *,
        outcome: str,
        start: datetime,
        end: datetime,
    ) -> list[PricePoint]:
        info = self.get_market(market_id)  # validates the id
        if outcome not in {o.name for o in info.outcomes}:
            return []  # honest absence: no such outcome, no invented series
        if start >= end:
            return []
        seed = zlib.crc32(f"history|{market_id}|{outcome}".encode("utf-8"))
        points: list[PricePoint] = []
        count = 24
        step = (end - start) / count
        price = _price_from(seed)
        for i in range(count):
            wiggle = ((zlib.crc32(f"{seed}|{i}".encode()) % 41) - 20) / 1000.0
            price = min(0.95, max(0.05, round(price + wiggle, 4)))
            points.append(PricePoint(ts=start + step * (i + 1), price=price))
        return points
