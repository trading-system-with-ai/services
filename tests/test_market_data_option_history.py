"""Historical option capability across the provider layer (Phase I §36-§37).

The §36 implied-move pipeline asks two questions no existing provider method
answers: "which contracts were LISTED on `underlying` for expiry E, as of a
date BEFORE the event" and "what did that contract close at each session".
The chain snapshot cannot answer either — it is current-state only, by design
and by these providers' own explicit refusal.

These tests pin the properties the pipeline depends on, in order of importance:

1. TIME IS NOT GUESSED FORWARD: ``as_of`` goes ON THE WIRE for the contract
   reference. Filtering today's universe locally would let a strike listed in
   reaction to an event slip into that event's PRE-event straddle — a wrong
   number wearing the shape of a right one.
2. THE EXPIRY IS COMPLETE: ``next_url`` pagination is followed to exhaustion,
   so a caller never mistakes a first page for the whole strike ladder. A
   truncated ladder can silently omit the ATM strike, which is the only strike
   §36 actually needs.
3. IDENTITY IS NEVER GUESSED: the OCC ticker and the row's own
   expiry/right/strike must AGREE. A row where they contradict is skipped, not
   resolved by preference — one of the two is wrong, and pricing the wrong
   contract is worse than pricing nothing.
4. NO FABRICATED PREMIUMS (§44 rule 18): a bar missing any OHLC field is
   SKIPPED, never zero-filled — an invented 0.0 premium makes a straddle look
   free. An empty window is ``[]`` and HTTP 403 is
   :class:`CapabilityNotAvailable` naming the endpoint.
5. ALPACA REFUSES BOTH, explicitly and by name, rather than returning ``[]``
   (which would read as "no contracts existed") or blending its own option
   bars with another vendor's contract identities (data_source.md §33).
6. THE STUB IS DETERMINISTIC AND LABELLED SYNTHETIC, so SQLite tests of the
   pipeline reproduce exactly without a network.

Massive runs over ``httpx.MockTransport``; the network is never touched, and
the wire shapes are the ones probed live against the base plan.
"""
from datetime import date

import httpx
import pytest

from libs.market_data.alpaca import AlpacaMarketDataProvider
from libs.market_data.massive import (
    DEFAULT_MAX_RATE_LIMIT_RETRIES,
    MassiveProvider,
)
from libs.market_data.provider import (
    Bar,
    CapabilityNotAvailable,
    MarketDataError,
    OptionContractRef,
)
from libs.market_data.stub import StubProvider

EXPIRY = date(2026, 8, 21)
AS_OF = date(2026, 8, 10)
WINDOW_START = date(2026, 8, 10)
WINDOW_END = date(2026, 8, 14)


def massive_with(handler, **kwargs) -> MassiveProvider:
    return MassiveProvider(
        api_key="test-key", transport=httpx.MockTransport(handler), **kwargs
    )


def alpaca_with(handler=None) -> AlpacaMarketDataProvider:
    handler = handler or (lambda request: httpx.Response(500))
    return AlpacaMarketDataProvider(
        api_key_id="test-key-id",
        api_secret_key="test-secret",
        transport=httpx.MockTransport(handler),
    )


def contract_row(strike: float, right: str = "C") -> dict:
    """One /v3/reference/options/contracts row in the live wire shape."""
    return {
        "ticker": f"O:AAPL260821{right}{int(strike * 1000):08d}",
        "underlying_ticker": "AAPL",
        "contract_type": "call" if right == "C" else "put",
        "strike_price": strike,
        "expiration_date": EXPIRY.isoformat(),
        "shares_per_contract": 100,
    }


def agg_row(day: date, close: float, *, volume: float = 500.0) -> dict:
    """One /v2/aggs option daily row: ``t`` in unix MILLISECONDS, ET session."""
    import datetime as _dt
    from zoneinfo import ZoneInfo

    start = _dt.datetime.combine(
        day, _dt.time(9, 30), tzinfo=ZoneInfo("America/New_York")
    )
    return {
        "t": int(start.timestamp() * 1000),
        "o": close - 0.2,
        "h": close + 0.3,
        "l": close - 0.4,
        "c": close,
        "v": volume,
        "n": 42,
    }


# ----------------------------------------------------------------------
# 1. as_of goes on the wire (Massive contract reference)
# ----------------------------------------------------------------------

def test_massive_contracts_send_as_of_and_expiry_on_the_wire():
    """The point-in-time key is the SERVER's job, not a local filter.

    ``as_of`` and ``expiration_date`` must appear as query parameters. If they
    did not, the adapter would be answering with today's listed universe —
    which includes strikes opened AFTER the event — and calling it history.
    """
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"results": [contract_row(210.0)]})

    refs = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF, right="C"
    )

    assert seen["as_of"] == AS_OF.isoformat()
    assert seen["expiration_date"] == EXPIRY.isoformat()
    assert seen["underlying_ticker"] == "AAPL"
    assert seen["contract_type"] == "call"
    assert refs == [
        OptionContractRef(
            ticker="O:AAPL260821C00210000",
            underlying="AAPL",
            expiry=EXPIRY,
            right="C",
            strike=210.0,
        )
    ]


def test_massive_contracts_put_filter_sends_put():
    """`right="P"` and `right="put"` both reach the server as contract_type=put."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("contract_type"))
        return httpx.Response(200, json={"results": [contract_row(210.0, "P")]})

    provider = massive_with(handler)
    assert provider.list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF, right="P"
    )[0].right == "P"
    provider.list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF, right="put"
    )
    assert seen == ["put", "put"]


def test_massive_contracts_no_right_filter_omits_contract_type():
    """No `right` means BOTH legs — the server is not narrowed at all."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(
            200, json={"results": [contract_row(210.0, "C"), contract_row(210.0, "P")]}
        )

    refs = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert "contract_type" not in seen[0].params
    assert [r.right for r in refs] == ["C", "P"]


# ----------------------------------------------------------------------
# 2. Pagination to exhaustion
# ----------------------------------------------------------------------

def test_massive_contracts_follow_next_url_to_exhaustion():
    """The whole strike ladder, not the first page.

    A ladder truncated at the page boundary can omit the ATM strike — the one
    strike the §36 straddle is built from — so a partial answer here is worse
    than a loud failure.
    """
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        if "cursor" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "results": [contract_row(200.0), contract_row(205.0)],
                    "next_url": "https://api.massive.com/v3/reference/options/"
                    "contracts?cursor=page2",
                },
            )
        return httpx.Response(
            200, json={"results": [contract_row(210.0), contract_row(215.0)]}
        )

    refs = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert [r.strike for r in refs] == [200.0, 205.0, 210.0, 215.0]
    assert len(calls) == 2


def test_massive_contracts_dedupe_across_a_page_boundary():
    """A contract repeated on two pages is ONE contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "cursor" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "results": [contract_row(205.0), contract_row(210.0)],
                    "next_url": "https://api.massive.com/v3/reference/options/"
                    "contracts?cursor=page2",
                },
            )
        return httpx.Response(
            200, json={"results": [contract_row(210.0), contract_row(215.0)]}
        )

    refs = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert [r.strike for r in refs] == [205.0, 210.0, 215.0]


def test_massive_contracts_page_cap_warns_rather_than_looping(caplog):
    """A server that never stops sending a cursor stops US, loudly."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [contract_row(200.0)],
                "next_url": "https://api.massive.com/v3/reference/options/"
                "contracts?cursor=forever",
            },
        )

    with caplog.at_level("WARNING"):
        refs = massive_with(handler, max_contract_pages=2).list_option_contracts(
            "AAPL", expiration_date=EXPIRY, as_of=AS_OF
        )
    assert len(refs) == 1  # deduped; the cap stopped the loop
    assert any("truncated at 2 pages" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------
# 3. Identity is never guessed
# ----------------------------------------------------------------------

def test_massive_contracts_parse_identity_from_the_occ_ticker_alone():
    """A row with ONLY a ticker is still fully identified — from the OCC form."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"ticker": "O:AAPL250801C00210000"}]}
        )

    ref = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=date(2025, 8, 1), as_of=AS_OF
    )[0]
    assert (ref.expiry, ref.right, ref.strike) == (date(2025, 8, 1), "C", 210.0)


def test_massive_contracts_skip_rows_whose_ticker_contradicts_the_fields():
    """Ticker says CALL, the row says put — neither is trusted, the row is dropped.

    Resolving the conflict by preference would price a leg we cannot name with
    confidence. Skipping loses one contract; guessing loses the straddle.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        bad_right = contract_row(210.0, "C") | {"contract_type": "put"}
        bad_expiry = contract_row(205.0, "C") | {"expiration_date": "2026-09-18"}
        return httpx.Response(
            200, json={"results": [bad_right, bad_expiry, contract_row(215.0)]}
        )

    refs = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert [r.strike for r in refs] == [215.0]


def test_massive_contracts_skip_unidentifiable_rows():
    """No ticker, no strike, unknown right -> skipped, never patched to a default."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"strike_price": 210.0, "expiration_date": EXPIRY.isoformat()},
                    {"ticker": "NOT-AN-OCC-SYMBOL", "strike_price": 0.0},
                    {
                        "ticker": "NOT-AN-OCC-SYMBOL-2",
                        "contract_type": "warrant",
                        "strike_price": 210.0,
                        "expiration_date": EXPIRY.isoformat(),
                    },
                    contract_row(210.0),
                ]
            },
        )

    refs = massive_with(handler).list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert [r.ticker for r in refs] == ["O:AAPL260821C00210000"]


def test_massive_contracts_empty_expiry_is_an_honest_empty_list():
    """An expiry with no listed contracts is [], not an error and not a guess."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert (
        massive_with(handler).list_option_contracts(
            "AAPL", expiration_date=EXPIRY, as_of=AS_OF
        )
        == []
    )


def test_massive_contracts_403_is_capability_not_available():
    """A plan-gated endpoint is named, never silently emptied."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="not entitled")

    with pytest.raises(CapabilityNotAvailable) as exc:
        massive_with(handler).list_option_contracts(
            "AAPL", expiration_date=EXPIRY, as_of=AS_OF
        )
    assert "/v3/reference/options/contracts" in str(exc.value)


def test_massive_contracts_reject_a_blank_underlying():
    """A blank ticker is a caller bug, refused before any request is sent."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.url)
        return httpx.Response(200, json={"results": []})

    with pytest.raises(MarketDataError):
        massive_with(handler).list_option_contracts(
            "  ", expiration_date=EXPIRY, as_of=AS_OF
        )
    assert sent == []


# ----------------------------------------------------------------------
# 4. Option bars: real numbers or none
# ----------------------------------------------------------------------

def test_massive_option_bars_parse_the_probed_wire_shape():
    """``t`` is unix MILLISECONDS and its EASTERN date is the trading date.

    Reading ms as ns (or UTC as ET) would place a bar on the wrong session —
    and the pre/post-event split is exactly a one-session distinction.
    """
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "results": [
                    agg_row(date(2026, 8, 10), 6.10),
                    agg_row(date(2026, 8, 11), 5.85),
                ]
            },
        )

    bars = massive_with(handler).get_option_history_bars(
        "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
    )
    assert seen["path"] == (
        "/v2/aggs/ticker/O:AAPL260821C00210000/range/1/day/2026-08-10/2026-08-14"
    )
    assert seen["sort"] == "asc" and seen["adjusted"] == "true"
    assert [b.ts for b in bars] == [date(2026, 8, 10), date(2026, 8, 11)]
    assert [b.close for b in bars] == [6.10, 5.85]
    assert isinstance(bars[0], Bar) and bars[0].volume == 500.0


def test_massive_option_bars_follow_next_url_and_sort_ascending():
    """The whole window, oldest first, deduplicated across the page boundary."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "cursor" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "results": [
                        agg_row(date(2026, 8, 11), 5.85),
                        agg_row(date(2026, 8, 10), 6.10),
                    ],
                    "next_url": "https://api.massive.com/v2/aggs/ticker/X/range/1/"
                    "day/a/b?cursor=page2",
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    agg_row(date(2026, 8, 11), 5.85),
                    agg_row(date(2026, 8, 12), 5.40),
                ]
            },
        )

    bars = massive_with(handler).get_option_history_bars(
        "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
    )
    assert [b.ts.day for b in bars] == [10, 11, 12]


def test_massive_option_bars_skip_rows_missing_ohlc_rather_than_zero_filling():
    """A premium we do not have must NOT become 0.0 — that prices a leg free."""

    def handler(request: httpx.Request) -> httpx.Response:
        broken = agg_row(date(2026, 8, 11), 5.85)
        del broken["c"]
        return httpx.Response(
            200, json={"results": [agg_row(date(2026, 8, 10), 6.10), broken]}
        )

    bars = massive_with(handler).get_option_history_bars(
        "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
    )
    assert [(b.ts.day, b.close) for b in bars] == [(10, 6.10)]


def test_massive_option_bars_missing_volume_is_zero_but_prices_are_not():
    """Volume may honestly be absent; the straddle is not priced from volume."""

    def handler(request: httpx.Request) -> httpx.Response:
        row = agg_row(date(2026, 8, 10), 6.10)
        del row["v"]
        return httpx.Response(200, json={"results": [row]})

    bars = massive_with(handler).get_option_history_bars(
        "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
    )
    assert bars[0].volume == 0.0 and bars[0].close == 6.10


def test_massive_option_bars_untraded_contract_is_an_empty_list():
    """An illiquid contract that never traded returns [], honestly."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resultsCount": 0, "results": []})

    assert (
        massive_with(handler).get_option_history_bars(
            "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
        )
        == []
    )


def test_massive_option_bars_404_is_an_empty_list_not_a_fault():
    """A symbol the deployment does not know is an absence, not an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"status": "NOT_FOUND"})

    assert (
        massive_with(handler).get_option_history_bars(
            "O:NOPE260821C00210000", WINDOW_START, WINDOW_END
        )
        == []
    )


def test_massive_option_bars_survive_a_429_burst_then_return_the_bars(monkeypatch):
    """THE LIVE FAILURE, at the endpoint that suffered it.

    ``POST /options/history/backfill?last=8`` fires four of these per event
    with no gap; Massive answered the tail of that burst with HTTP 429, the
    single retry the adapter used to do was not enough, and the events came
    back with NO bars. N 429s followed by a 200 must now yield the bars.
    """
    monkeypatch.setattr("libs.market_data.massive.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 3:
            return httpx.Response(429, json={"message": "slow down"})
        return httpx.Response(
            200,
            json={
                "resultsCount": 1,
                "results": [agg_row(WINDOW_START, 5.5)],
            },
        )

    bars = massive_with(handler).get_option_history_bars(
        "O:AAPL251031P00272500", WINDOW_START, WINDOW_END
    )
    assert calls["n"] == 4  # three refusals, then the answer
    assert len(bars) == 1
    assert bars[0].close == 5.5


def test_massive_option_bars_429_beyond_the_budget_is_an_honest_error(monkeypatch):
    """One more 429 than the budget allows raises rather than returning [].

    Critically NOT an empty list: ``[]`` means "this contract never traded",
    and a rate limit reported as an untraded contract is exactly how a
    straddle silently loses a leg.
    """
    monkeypatch.setattr("libs.market_data.massive.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"message": "slow down"})

    with pytest.raises(MarketDataError, match="429"):
        massive_with(handler).get_option_history_bars(
            "O:AAPL251031P00272500", WINDOW_START, WINDOW_END
        )
    assert calls["n"] == 1 + DEFAULT_MAX_RATE_LIMIT_RETRIES


def test_massive_option_bars_403_is_capability_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="options aggregates not in plan")

    with pytest.raises(CapabilityNotAvailable) as exc:
        massive_with(handler).get_option_history_bars(
            "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
        )
    assert "/v2/aggs/ticker/" in str(exc.value)


def test_massive_option_bars_reject_a_reversed_window():
    """An empty window is start == end; end < start is a caller bug."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.url)
        return httpx.Response(200, json={"results": []})

    with pytest.raises(ValueError):
        massive_with(handler).get_option_history_bars(
            "O:AAPL260821C00210000", WINDOW_END, WINDOW_START
        )
    assert sent == []


def test_massive_option_bars_ticker_is_passed_through_verbatim():
    """The symbol from the reference call is re-sent unmodified.

    Re-deriving it from parts is how a strike padded to nine digits instead of
    eight becomes a silent empty result that looks like "never traded".
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"results": []})

    massive_with(handler).get_option_history_bars(
        "O:AAPL250801C000210000", WINDOW_START, WINDOW_END
    )
    assert "O:AAPL250801C000210000" in seen[0]


# ----------------------------------------------------------------------
# 5. Alpaca refuses BOTH halves, by name
# ----------------------------------------------------------------------

def test_alpaca_list_option_contracts_refuses_and_names_the_gap():
    """[] would claim "no contracts existed"; the refusal claims only the gap."""
    with pytest.raises(CapabilityNotAvailable) as exc:
        alpaca_with().list_option_contracts(
            "AAPL", expiration_date=EXPIRY, as_of=AS_OF
        )
    message = str(exc.value)
    assert "/v2/options/contracts" in message
    assert "as_of" in message
    assert "AAPL" in message


def test_alpaca_option_history_bars_refuses_without_blending_providers():
    """Alpaca HAS option bars; pairing them with another vendor's contract
    identities is what §33 forbids, so this seam refuses."""
    with pytest.raises(CapabilityNotAvailable) as exc:
        alpaca_with().get_option_history_bars(
            "O:AAPL260821C00210000", WINDOW_START, WINDOW_END
        )
    assert "§33" in str(exc.value)


def test_alpaca_backtest_option_bars_method_is_untouched():
    """The pre-existing ``get_option_daily_bars`` still exists with its own
    ``{date: (open, close)}`` contract — the new capability took a new NAME
    precisely so the backtest resolver could not break silently."""
    import inspect

    sig = inspect.signature(AlpacaMarketDataProvider.get_option_daily_bars)
    assert list(sig.parameters) == ["self", "option_ticker", "start", "end"]
    assert AlpacaMarketDataProvider.get_option_daily_bars is not (
        AlpacaMarketDataProvider.get_option_history_bars
    )


# ----------------------------------------------------------------------
# 6. Stub: deterministic, synthetic, labelled
# ----------------------------------------------------------------------

def test_stub_contracts_are_a_fixed_call_and_put_ladder():
    """90..110 by 5, calls AND puts — assertable by exact strike."""
    refs = StubProvider().list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert sorted({r.strike for r in refs}) == [90.0, 95.0, 100.0, 105.0, 110.0]
    assert {r.right for r in refs} == {"C", "P"}
    assert len(refs) == 10
    assert all(r.expiry == EXPIRY and r.underlying == "AAPL" for r in refs)


def test_stub_contracts_are_deterministic_across_instances():
    """Two fresh providers agree exactly — SQLite tests must reproduce."""
    a = StubProvider().list_option_contracts(
        "MSFT", expiration_date=EXPIRY, as_of=AS_OF
    )
    b = StubProvider().list_option_contracts(
        "MSFT", expiration_date=EXPIRY, as_of=AS_OF
    )
    assert a == b


def test_stub_contract_tickers_round_trip_through_the_bars_call():
    """The ticker the ladder emits is the ticker the bars call accepts."""
    stub = StubProvider()
    for ref in stub.list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF
    ):
        assert stub.get_option_history_bars(ref.ticker, WINDOW_START, WINDOW_END)


def test_stub_right_filter_selects_one_leg_and_rejects_nonsense():
    stub = StubProvider()
    calls = stub.list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF, right="call"
    )
    puts = stub.list_option_contracts(
        "AAPL", expiration_date=EXPIRY, as_of=AS_OF, right="P"
    )
    assert {r.right for r in calls} == {"C"} and len(calls) == 5
    assert {r.right for r in puts} == {"P"} and len(puts) == 5
    assert (
        stub.list_option_contracts(
            "AAPL", expiration_date=EXPIRY, as_of=AS_OF, right="warrant"
        )
        == []
    )


def test_stub_bars_are_deterministic_and_decay():
    """A pure function of (ticker, start, end): same series, every time."""
    stub = StubProvider()
    first = stub.get_option_history_bars(
        "O:AAPL260821C00100000", WINDOW_START, WINDOW_END
    )
    second = StubProvider().get_option_history_bars(
        "O:AAPL260821C00100000", WINDOW_START, WINDOW_END
    )
    assert first == second
    assert [b.close for b in first] == sorted(
        (b.close for b in first), reverse=True
    )  # decaying


def test_stub_bars_omit_weekends_rather_than_synthesizing_them():
    """Aug 10-14 2026 is Mon-Fri; the window through Sunday adds no bars."""
    bars = StubProvider().get_option_history_bars(
        "O:AAPL260821C00100000", date(2026, 8, 10), date(2026, 8, 16)
    )
    assert [b.ts.day for b in bars] == [10, 11, 12, 13, 14]
    assert all(b.ts.weekday() < 5 for b in bars)


def test_stub_call_and_put_price_differently():
    """A straddle from the stub pair is not two copies of one leg."""
    stub = StubProvider()
    call = stub.get_option_history_bars(
        "O:AAPL260821C00100000", WINDOW_START, WINDOW_END
    )[0]
    put = stub.get_option_history_bars(
        "O:AAPL260821P00100000", WINDOW_START, WINDOW_END
    )[0]
    assert call.close != put.close


def test_stub_unknown_ticker_is_an_empty_list_and_reversed_window_raises():
    stub = StubProvider()
    assert stub.get_option_history_bars("garbage", WINDOW_START, WINDOW_END) == []
    with pytest.raises(ValueError):
        stub.get_option_history_bars(
            "O:AAPL260821C00100000", WINDOW_END, WINDOW_START
        )


def test_stub_backtest_option_bars_method_is_untouched():
    """The stub's pre-existing ``get_option_daily_bars`` still returns the
    ``{date: (open, close)}`` mapping the options backtest resolver expects."""
    rows = StubProvider().get_option_daily_bars(
        "AAPL260821C00100000", WINDOW_START, WINDOW_END
    )
    assert isinstance(rows, dict)
    assert all(isinstance(v, tuple) and len(v) == 2 for v in rows.values())
