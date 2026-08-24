"""Tests for the Alpaca PAPER-ONLY broker adapter (plan §11, §44 rule 18).

Every test runs against a mocked httpx transport — the network is NEVER
touched. The suite is organised around the two things that must hold no matter
what: live trading is unreachable, and no fill is ever invented.
"""
import json

import httpx
import pytest

from libs.broker import (
    BrokerError,
    BrokerNotConfigured,
    BrokerOrder,
    BrokerRejected,
    get_broker,
)
from libs.broker.alpaca import (
    ALPACA_PAPER_BASE_URL,
    KEY_HEADER,
    PAPER_HOST,
    SECRET_HEADER,
    AlpacaPaperBroker,
)

CLIENT_ORDER_ID = "ord-2026-08-10-0001"

PAPER_ACCOUNT = {
    "id": "acct-1",
    "account_number": "PA3ABCDEF",
    "cash": "98000.50",
    "equity": "101250.75",
    "buying_power": "196001.00",
    "currency": "USD",
    "is_paper": True,
    "status": "ACTIVE",
}

LIVE_ACCOUNT = {**PAPER_ACCOUNT, "account_number": "934FEDCBA", "is_paper": False}


def _order_payload(**overrides) -> dict:
    payload = {
        "id": "b1f0c4de-0000-4000-8000-000000000001",
        "client_order_id": CLIENT_ORDER_ID,
        "symbol": "AAPL",
        "side": "buy",
        "status": "accepted",
        "qty": "10",
        "filled_qty": "0",
        "filled_avg_price": None,
        "submitted_at": "2026-08-10T14:30:00.123456Z",
        "type": "market",
        "time_in_force": "day",
    }
    payload.update(overrides)
    return payload


def _broker(handler, **kwargs) -> AlpacaPaperBroker:
    return AlpacaPaperBroker(
        api_key="test-key",
        api_secret="test-secret",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _routing_handler(order_response=None, account=PAPER_ACCOUNT, calls=None):
    """Handler routing /v2/account and /v2/orders, recording calls."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append((request.method, request.url.path))
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=account)
        if request.url.path == "/v2/orders":
            return order_response or httpx.Response(200, json=_order_payload())
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


# ---------------------------------------------------------------------------
# PAPER-ONLY GUARD, layer 1: the constructor refuses a non-paper host
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "base_url, host",
    [
        ("https://api.alpaca.markets", "api.alpaca.markets"),
        ("https://api.alpaca.markets/v2", "api.alpaca.markets"),
        ("https://broker-api.example.com", "broker-api.example.com"),
        ("http://localhost:9999", "localhost"),
    ],
)
def test_non_paper_base_url_is_refused_naming_the_host(base_url, host):
    # Live trading must be impossible to reach by configuration alone: the
    # real Alpaca live host is refused exactly like any other foreign host,
    # and the error names the host so the operator sees what they pointed at.
    with pytest.raises(ValueError, match=host):
        AlpacaPaperBroker(api_key="k", api_secret="s", base_url=base_url)


def test_paper_base_url_is_accepted():
    broker = AlpacaPaperBroker(api_key="k", api_secret="s")
    assert broker.base_url == ALPACA_PAPER_BASE_URL
    assert PAPER_HOST in broker.base_url


def test_non_paper_refusal_message_states_paper_only():
    with pytest.raises(ValueError, match="PAPER ONLY"):
        AlpacaPaperBroker(
            api_key="k", api_secret="s", base_url="https://api.alpaca.markets"
        )


# --- Adversarial: the live host cannot be smuggled past the guard ----------
# The check compares the PARSED host, never a substring, so none of these
# spellings of the live host may construct an adapter.

@pytest.mark.parametrize(
    "base_url",
    [
        "https://API.ALPACA.MARKETS",                        # case variant
        "https://api.alpaca.markets/",                       # trailing slash
        "https://api.alpaca.markets:443",                    # explicit port
        "https://paper-api.alpaca.markets@api.alpaca.markets",  # userinfo decoy
        "https://user:pw@api.alpaca.markets",                # credentialed
        "https://api.alpaca.markets?x=paper-api.alpaca.markets",  # query decoy
        "https://api.alpaca.markets#paper-api.alpaca.markets",    # fragment decoy
        "https://evil.com/paper-api.alpaca.markets",         # host in the PATH
        "https://paper-api.alpaca.markets.evil.com",         # lookalike suffix
        "api.alpaca.markets",                                # bare, no scheme
        "//api.alpaca.markets",                              # scheme-relative
    ],
)
def test_live_host_cannot_be_smuggled_in_by_any_spelling(base_url):
    with pytest.raises(ValueError):
        AlpacaPaperBroker(api_key="k", api_secret="s", base_url=base_url)


def test_plaintext_http_is_refused_so_credentials_never_travel_in_the_clear():
    # The host is correct here — only the SCHEME is wrong. Allowing it would
    # put the API key and secret on the wire in cleartext.
    with pytest.raises(ValueError, match="cleartext"):
        AlpacaPaperBroker(
            api_key="k", api_secret="s", base_url="http://paper-api.alpaca.markets"
        )


def test_custom_port_on_the_paper_host_is_refused():
    with pytest.raises(ValueError, match="port"):
        AlpacaPaperBroker(
            api_key="k", api_secret="s",
            base_url="https://paper-api.alpaca.markets:8080",
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/",       # trailing slash
        "https://PAPER-API.ALPACA.MARKETS",        # case variant of the PAPER host
        "https://paper-api.alpaca.markets:443",    # the default port, spelled out
    ],
)
def test_paper_host_spellings_are_accepted_and_normalised(base_url):
    # Every accepted spelling collapses to ONE canonical URL, so nothing
    # downstream (logging, comparison) has to know about spelling variants.
    broker = AlpacaPaperBroker(api_key="k", api_secret="s", base_url=base_url)
    assert broker.base_url == ALPACA_PAPER_BASE_URL


# ---------------------------------------------------------------------------
# PAPER-ONLY GUARD, layer 2: submit re-verifies is_paper with the broker
# ---------------------------------------------------------------------------

def test_submit_against_live_account_raises_and_posts_no_order():
    # The account check happens FIRST. A key pair that somehow points at a
    # live account must never reach POST /v2/orders.
    calls: list[tuple[str, str]] = []
    broker = _broker(_routing_handler(account=LIVE_ACCOUNT, calls=calls))

    with pytest.raises(BrokerError, match="NOT a paper account"):
        broker.submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)

    assert ("POST", "/v2/orders") not in calls, "no order may be sent to a live account"
    assert calls == [("GET", "/v2/account")]


def test_submit_checks_account_before_posting_order():
    calls: list[tuple[str, str]] = []
    broker = _broker(_routing_handler(calls=calls))

    broker.submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)

    assert calls == [("GET", "/v2/account"), ("POST", "/v2/orders")]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, secret", [("", "s"), ("k", ""), ("", ""), ("   ", "s"), ("k", "  ")])
def test_missing_credentials_raise_at_construction(key, secret):
    # The adapter can never fire keyless (mirrors the LLM adapters).
    with pytest.raises(BrokerError, match="non-empty API"):
        AlpacaPaperBroker(api_key=key, api_secret=secret)


def test_auth_headers_are_sent():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json=PAPER_ACCOUNT)

    _broker(handler).get_account()

    assert captured["headers"][KEY_HEADER] == "test-key"
    assert captured["headers"][SECRET_HEADER] == "test-secret"


def test_credentials_are_never_logged(caplog):
    # If a request is logged, both header values must be redacted.
    caplog.set_level("DEBUG", logger="libs.broker.alpaca")
    _broker(_routing_handler()).submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)

    logged = caplog.text
    assert "test-key" not in logged
    assert "test-secret" not in logged
    assert "redacted" in logged


# ---------------------------------------------------------------------------
# submit_order: payload + parsing
# ---------------------------------------------------------------------------

def test_submit_order_builds_documented_payload():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        captured["body"] = json.loads(request.content)
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json=_order_payload())

    _broker(handler).submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)

    assert captured["method"] == "POST"
    assert captured["path"] == "/v2/orders"
    assert captured["body"] == {
        "symbol": "AAPL",
        "qty": "10",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": CLIENT_ORDER_ID,
    }


def test_submit_order_parses_the_response():
    order = _broker(_routing_handler()).submit_order(
        CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10
    )

    assert isinstance(order, BrokerOrder)
    assert order.broker_order_id == "b1f0c4de-0000-4000-8000-000000000001"
    assert order.client_order_id == CLIENT_ORDER_ID
    assert order.symbol == "AAPL"
    assert order.side == "BUY_TO_OPEN"
    assert order.status == "ACCEPTED"
    assert order.raw_status == "accepted"
    assert order.requested_quantity == 10
    assert order.filled_quantity == 0
    assert order.filled_avg_price is None
    assert order.submitted_at.year == 2026
    assert order.submitted_at.tzinfo is not None


def test_sell_to_close_maps_to_alpaca_sell():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_order_payload(side="sell"))

    order = _broker(handler).submit_order(CLIENT_ORDER_ID, "AAPL", "SELL_TO_CLOSE", 10)

    assert captured["body"]["side"] == "sell"
    assert order.side == "SELL_TO_CLOSE"


def test_no_side_can_open_a_short():
    # §5: Sell-to-Open does not exist. Anything that is not one of the two
    # documented sides is refused before any HTTP call is made.
    broker = _broker(_routing_handler())
    for bad_side in ("SELL_TO_OPEN", "BUY_TO_CLOSE", "short", "sell", ""):
        with pytest.raises(ValueError, match="side"):
            broker.submit_order(CLIENT_ORDER_ID, "AAPL", bad_side, 10)


def test_non_positive_quantity_is_refused():
    broker = _broker(_routing_handler())
    for bad_qty in (0, -1):
        with pytest.raises(ValueError, match="quantity"):
            broker.submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", bad_qty)


# ---------------------------------------------------------------------------
# Status mapping — every documented Alpaca status, plus the unknown case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("new", "ACCEPTED"),
        ("accepted", "ACCEPTED"),
        ("pending_new", "ACCEPTED"),
        ("accepted_for_bidding", "ACCEPTED"),
        ("partially_filled", "PARTIALLY_FILLED"),
        ("filled", "FILLED"),
        ("rejected", "REJECTED"),
        ("canceled", "CANCELED"),
        # A cancel REQUEST is not a cancel: the order is still live at the
        # broker and can still fill, so it must stay non-terminal and in the
        # order-sync sweep's watch (the raw status is preserved alongside).
        ("pending_cancel", "ACCEPTED"),
        ("expired", "EXPIRED"),
    ],
)
def test_every_alpaca_status_maps_to_the_documented_enum(raw, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_order_payload(status=raw))

    order = _broker(handler).get_order(CLIENT_ORDER_ID)

    assert order is not None
    assert order.status == expected
    assert order.raw_status == raw  # the broker's own word is always preserved


@pytest.mark.parametrize("raw", ["held", "calculated", "stopped", "some_new_state"])
def test_unknown_status_maps_to_accepted_and_keeps_the_raw_string(raw, caplog):
    # NEVER guess FILLED: an unrecognised state means "keep watching", not
    # "an execution happened". Inventing a fill is exactly the failure mode
    # this platform refuses.
    caplog.set_level("WARNING", logger="libs.broker.alpaca")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_order_payload(status=raw))

    order = _broker(handler).get_order(CLIENT_ORDER_ID)

    assert order is not None
    assert order.status == "ACCEPTED"
    assert order.raw_status == raw
    assert raw in caplog.text  # the surprise is logged, not swallowed


def test_unknown_status_never_maps_to_filled():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_order_payload(status="mystery", filled_qty="0", filled_avg_price=None),
        )

    order = _broker(handler).get_order(CLIENT_ORDER_ID)
    assert order is not None
    assert order.status != "FILLED"
    assert order.filled_quantity == 0
    assert order.filled_avg_price is None


# ---------------------------------------------------------------------------
# Fill parsing
# ---------------------------------------------------------------------------

def test_partial_fill_parses_quantity_and_average_price():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_order_payload(
                status="partially_filled",
                qty="10",
                filled_qty="4",
                filled_avg_price="231.4525",
            ),
        )

    order = _broker(handler).get_order(CLIENT_ORDER_ID)

    assert order is not None
    assert order.status == "PARTIALLY_FILLED"
    assert order.requested_quantity == 10
    assert order.filled_quantity == 4
    assert order.filled_avg_price == pytest.approx(231.4525)


def test_full_fill_parses_quantity_and_average_price():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_order_payload(
                status="filled", qty="10", filled_qty="10", filled_avg_price="230.10"
            ),
        )

    order = _broker(handler).get_order(CLIENT_ORDER_ID)

    assert order is not None
    assert order.status == "FILLED"
    assert order.filled_quantity == 10
    assert order.filled_avg_price == pytest.approx(230.10)


def test_blank_filled_avg_price_parses_to_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_order_payload(filled_avg_price=""))

    order = _broker(handler).get_order(CLIENT_ORDER_ID)
    assert order is not None
    assert order.filled_avg_price is None


# ---------------------------------------------------------------------------
# get_order
# ---------------------------------------------------------------------------

def test_get_order_404_returns_none():
    # A definitive "no such order" — distinct from a fault, which raises.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": 40410000, "message": "order not found"})

    assert _broker(handler).get_order(CLIENT_ORDER_ID) is None


def test_get_order_found_returns_broker_order_and_queries_by_client_id():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=_order_payload())

    order = _broker(handler).get_order(CLIENT_ORDER_ID)

    assert captured["path"] == "/v2/orders:by_client_order_id"
    assert captured["query"] == {"client_order_id": CLIENT_ORDER_ID}
    assert isinstance(order, BrokerOrder)
    assert order.client_order_id == CLIENT_ORDER_ID


def test_get_order_rejected_is_a_broker_order_not_an_exception():
    # DOCUMENTED SPLIT: a rejection at SUBMIT time raises BrokerRejected; a
    # rejection discovered LATER by polling is just an order in a terminal
    # state, so the gateway can record it against the stored order row.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_order_payload(status="rejected"))

    order = _broker(handler).get_order(CLIENT_ORDER_ID)

    assert isinstance(order, BrokerOrder)
    assert order.status == "REJECTED"
    assert order.raw_status == "rejected"


# ---------------------------------------------------------------------------
# Rejection at submit time
# ---------------------------------------------------------------------------

def test_submit_rejected_with_4xx_raises_broker_rejected():
    # DOCUMENTED CHOICE: submit_order raises BrokerRejected rather than
    # returning a REJECTED order, so a caller can never mistake a refusal for
    # a working order it should poll.
    order_response = httpx.Response(
        403, json={"code": 40310000, "message": "insufficient buying power"}
    )
    broker = _broker(_routing_handler(order_response=order_response))

    with pytest.raises(BrokerRejected, match="insufficient buying power"):
        broker.submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)


def test_submit_rejected_with_200_and_rejected_status_raises_broker_rejected():
    # Same fact in a different envelope: Alpaca can answer 200 with a
    # "rejected" status. It must behave identically.
    order_response = httpx.Response(200, json=_order_payload(status="rejected"))
    broker = _broker(_routing_handler(order_response=order_response))

    with pytest.raises(BrokerRejected, match="rejected"):
        broker.submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)


def test_broker_rejected_is_a_broker_error():
    # A caller that only handles BrokerError still catches business rejections.
    assert issubclass(BrokerRejected, BrokerError)


# ---------------------------------------------------------------------------
# Transport and HTTP faults
# ---------------------------------------------------------------------------

def test_network_error_raises_broker_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BrokerError, match="request failed"):
        _broker(handler).get_account()


def test_network_error_on_submit_raises_broker_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BrokerError, match="request failed"):
        _broker(handler).submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)


def test_http_500_raises_broker_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    with pytest.raises(BrokerError, match="HTTP 500"):
        _broker(handler).get_account()


def test_http_500_on_submit_raises_broker_error_not_rejected():
    # A 5xx is a FAULT, not a business rejection: the order may or may not
    # exist, so the caller must reconcile rather than assume refusal.
    order_response = httpx.Response(500, text="internal server error")
    broker = _broker(_routing_handler(order_response=order_response))

    with pytest.raises(BrokerError, match="HTTP 500") as exc_info:
        broker.submit_order(CLIENT_ORDER_ID, "AAPL", "BUY_TO_OPEN", 10)
    assert not isinstance(exc_info.value, BrokerRejected)


def test_get_order_500_raises_rather_than_returning_none():
    # A fault must never be read as "the order does not exist".
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BrokerError, match="HTTP 500"):
        _broker(handler).get_order(CLIENT_ORDER_ID)


# ---------------------------------------------------------------------------
# Account and positions
# ---------------------------------------------------------------------------

def test_get_account_parses_snapshot():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/account"
        return httpx.Response(200, json=PAPER_ACCOUNT)

    account = _broker(handler).get_account()

    assert account.cash == pytest.approx(98000.50)
    assert account.equity == pytest.approx(101250.75)
    assert account.buying_power == pytest.approx(196001.00)
    assert account.currency == "USD"
    assert account.is_paper is True
    assert account.account_number == "PA3ABCDEF"


def test_list_positions_parses_rows():
    rows = [
        {
            "symbol": "AAPL",
            "qty": "10",
            "avg_entry_price": "228.40",
            "market_value": "2310.00",
        },
        {
            "symbol": "MSFT",
            "qty": "3",
            "avg_entry_price": "402.15",
            "market_value": None,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/positions"
        return httpx.Response(200, json=rows)

    positions = _broker(handler).list_positions()

    assert [p.symbol for p in positions] == ["AAPL", "MSFT"]
    assert positions[0].quantity == 10
    assert positions[0].avg_entry_price == pytest.approx(228.40)
    assert positions[0].market_value == pytest.approx(2310.00)
    assert positions[1].market_value is None


def test_list_positions_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    assert _broker(handler).list_positions() == []


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

def test_cancel_order_resolves_client_id_then_deletes():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v2/orders:by_client_order_id":
            return httpx.Response(200, json=_order_payload())
        return httpx.Response(204)

    _broker(handler).cancel_order(CLIENT_ORDER_ID)

    assert calls == [
        ("GET", "/v2/orders:by_client_order_id"),
        ("DELETE", "/v2/orders/b1f0c4de-0000-4000-8000-000000000001"),
    ]


def test_cancel_unknown_order_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "not found"})

    with pytest.raises(BrokerError, match="unknown order"):
        _broker(handler).cancel_order(CLIENT_ORDER_ID)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_registry_blank_name_raises_not_configured(name):
    # Blank name = the unconfigured state, and it names the missing setting.
    with pytest.raises(BrokerNotConfigured, match="BROKER_PROVIDER"):
        get_broker(name)


def test_registry_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="unknown broker"):
        get_broker("nope")


def test_not_configured_is_not_a_value_error():
    # An unknown NAME is an operator typo; absence of config is a different
    # fact, mapped by callers to 503 BROKER_NOT_CONFIGURED.
    assert not issubclass(BrokerNotConfigured, ValueError)


def test_registry_has_no_simulated_entry():
    # THE POINT OF THIS PACKAGE. A simulated fill served through the broker
    # interface would be an execution that never happened, reported as one
    # that did. There is no name that reaches the internal simulator.
    from libs.broker import _BROKERS

    for forbidden in ("simulated", "sim", "stub", "paper", "fake", "mock", "internal"):
        assert forbidden not in _BROKERS

    for forbidden in ("simulated", "stub", "fake"):
        with pytest.raises(ValueError, match="unknown broker"):
            get_broker(forbidden)


def test_registry_registers_alpaca_paper():
    from libs.broker import _BROKERS

    assert "alpaca_paper" in _BROKERS
    assert sorted(_BROKERS) == ["alpaca_paper"]


def test_registry_alpaca_paper_builds_from_settings():
    # The factory reads credentials from settings and constructs the adapter
    # lazily. Either outcome is correct and both are safe:
    #   - no credentials configured -> BrokerError (never fires keyless);
    #   - credentials configured     -> a paper-pinned AlpacaPaperBroker.
    # Asserting only "no credentials" would make this test depend on the
    # developer's local .env, so both branches are checked explicitly.
    try:
        broker = get_broker("alpaca_paper")
    except BrokerError as exc:
        assert "non-empty API" in str(exc)
    else:
        assert isinstance(broker, AlpacaPaperBroker)
        # However it was built, it is pinned to the paper host.
        assert broker.base_url == ALPACA_PAPER_BASE_URL


def test_registry_alpaca_paper_can_never_be_built_for_a_live_host():
    # Whatever the environment holds, the factory has no path to a live host:
    # the constructor is the only way in and it refuses anything but paper.
    try:
        broker = get_broker("alpaca_paper")
    except BrokerError:
        pytest.skip("no Alpaca credentials configured in this environment")
    assert PAPER_HOST in broker.base_url
    assert "api.alpaca.markets" != (broker.base_url.split("//")[-1].split("/")[0])


def test_registry_has_no_live_alpaca_entry():
    # There is no name in this registry that reaches live trading.
    from libs.broker import _BROKERS

    assert all("live" not in name for name in _BROKERS)


# ---------------------------------------------------------------------------
# MLEG (roadmap Phase 1): atomic defined-risk verticals. THE SHAPE GUARD IS
# THE §5 SAFETY BOUNDARY — sell-to-open can only exist covered inside the
# pair, validated BEFORE any network I/O.
# ---------------------------------------------------------------------------
from libs.broker import BrokerOrderLeg
from libs.broker.provider import BUY_TO_CLOSE, SELL_TO_OPEN

LONG_OCC = "AAPL260918C00335000"
SHORT_OCC = "AAPL260918C00350000"


def open_legs() -> list[BrokerOrderLeg]:
    return [
        BrokerOrderLeg(symbol=LONG_OCC, side="BUY_TO_OPEN"),
        BrokerOrderLeg(symbol=SHORT_OCC, side=SELL_TO_OPEN),
    ]


def _mleg_payload(**overrides) -> dict:
    payload = _order_payload(
        symbol=None,
        legs=[
            _order_payload(symbol=LONG_OCC, side="buy", filled_qty="2",
                           filled_avg_price="7.60", qty="2"),
            _order_payload(symbol=SHORT_OCC, side="sell", filled_qty="2",
                           filled_avg_price="3.10", qty="2"),
        ],
        status="filled",
        filled_qty="2",
    )
    payload.update(overrides)
    return payload


def test_mleg_submits_atomic_order_with_correct_intents():
    calls = []
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        import json as _json
        bodies.append(_json.loads(request.content))
        return httpx.Response(200, json=_mleg_payload())

    order = _broker(handler).submit_mleg_order(CLIENT_ORDER_ID, open_legs(), 2)
    body = bodies[0]
    assert body["order_class"] == "mleg"
    assert body["qty"] == "2"
    intents = {leg["symbol"]: leg["position_intent"] for leg in body["legs"]}
    assert intents == {LONG_OCC: "buy_to_open", SHORT_OCC: "sell_to_open"}
    sides = {leg["symbol"]: leg["side"] for leg in body["legs"]}
    assert sides == {LONG_OCC: "buy", SHORT_OCC: "sell"}
    # One BrokerOrder for the whole spread: joined symbol, NET fill.
    assert order.symbol == f"{LONG_OCC}/{SHORT_OCC}"
    assert order.filled_avg_price == pytest.approx(7.60 - 3.10)
    assert order.filled_quantity == 2


def test_mleg_net_fill_is_none_until_every_leg_fills():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        payload = _mleg_payload(status="partially_filled")
        payload["legs"][1]["filled_avg_price"] = None
        payload["legs"][1]["filled_qty"] = "0"
        return httpx.Response(200, json=payload)

    order = _broker(handler).submit_mleg_order(CLIENT_ORDER_ID, open_legs(), 2)
    assert order.filled_avg_price is None  # half-filled net = invented number


def _shape_guard_broker():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        raise AssertionError("shape guard must reject BEFORE any order POST")

    return _broker(handler)


def test_mleg_shape_guard_rejects_every_illegal_pair():
    b = _shape_guard_broker()
    # A lone sell-to-open (the §5 nightmare) has no vocabulary here.
    with pytest.raises(ValueError, match="exactly TWO legs"):
        b.submit_mleg_order(CLIENT_ORDER_ID, [BrokerOrderLeg(SHORT_OCC, SELL_TO_OPEN)], 1)
    # Two sell-to-opens: not a defined-risk pair.
    with pytest.raises(ValueError, match="mleg pair must be"):
        b.submit_mleg_order(
            CLIENT_ORDER_ID,
            [BrokerOrderLeg(LONG_OCC, SELL_TO_OPEN), BrokerOrderLeg(SHORT_OCC, SELL_TO_OPEN)],
            1,
        )
    # Call vertical with the short strike BELOW the long: uncovered.
    with pytest.raises(ValueError, match="short strike must be ABOVE"):
        b.submit_mleg_order(
            CLIENT_ORDER_ID,
            [BrokerOrderLeg(SHORT_OCC, "BUY_TO_OPEN"), BrokerOrderLeg(LONG_OCC, SELL_TO_OPEN)],
            1,
        )
    # Cross-expiry legs: not a vertical.
    with pytest.raises(ValueError, match="SAME underlying, expiry and"):
        b.submit_mleg_order(
            CLIENT_ORDER_ID,
            [
                BrokerOrderLeg(LONG_OCC, "BUY_TO_OPEN"),
                BrokerOrderLeg("AAPL261016C00350000", SELL_TO_OPEN),
            ],
            1,
        )
    # Non-1:1 ratio.
    with pytest.raises(ValueError, match="1:1"):
        b.submit_mleg_order(
            CLIENT_ORDER_ID,
            [
                BrokerOrderLeg(LONG_OCC, "BUY_TO_OPEN", ratio=2),
                BrokerOrderLeg(SHORT_OCC, SELL_TO_OPEN),
            ],
            1,
        )
    # Close pair is legal shape-wise (SELL_TO_CLOSE long + BUY_TO_CLOSE short)
    # but a LIVE (non-paper) account is still refused by layer 2:
    def live_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=LIVE_ACCOUNT)
        raise AssertionError("live account must never see an order")

    with pytest.raises(BrokerError, match="NOT a paper account"):
        _broker(live_handler).submit_mleg_order(
            CLIENT_ORDER_ID,
            [
                BrokerOrderLeg(LONG_OCC, "SELL_TO_CLOSE"),
                BrokerOrderLeg(SHORT_OCC, BUY_TO_CLOSE),
            ],
            1,
        )


# ---------------------------------------------------------------------------
# Phase 2: collateral-attested short open + dedicated buyback (OCC-gated —
# short STOCK remains unconstructable through every path).
# ---------------------------------------------------------------------------


def test_short_open_requires_collateral_attestation_and_occ_symbol():
    b = _shape_guard_broker()
    with pytest.raises(ValueError, match="covered_by"):
        b.submit_short_open_order(CLIENT_ORDER_ID, SHORT_OCC, 1, covered_by="")
    with pytest.raises(ValueError, match="OCC OPTION symbol"):
        b.submit_short_open_order(CLIENT_ORDER_ID, "AAPL", 1, covered_by="pos-7")
    with pytest.raises(ValueError, match="OCC OPTION symbol"):
        b.submit_short_close_order(CLIENT_ORDER_ID, "AAPL", 1)


def test_short_open_and_buyback_wire_shape():
    import json as _json
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        bodies.append(_json.loads(request.content))
        return httpx.Response(
            200,
            json=_order_payload(symbol=SHORT_OCC, side=bodies[-1]["side"], status="accepted"),
        )

    broker = _broker(handler)
    broker.submit_short_open_order(CLIENT_ORDER_ID, SHORT_OCC, 2, covered_by="position:41")
    assert bodies[-1]["side"] == "sell"
    assert bodies[-1]["position_intent"] == "sell_to_open"
    assert bodies[-1]["qty"] == "2"
    broker.submit_short_close_order(CLIENT_ORDER_ID + "x", SHORT_OCC, 2)
    assert bodies[-1]["side"] == "buy"
    assert bodies[-1]["position_intent"] == "buy_to_close"


def test_short_open_refused_on_live_account():
    def live_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=LIVE_ACCOUNT)
        raise AssertionError("live account must never see an order")

    with pytest.raises(BrokerError, match="NOT a paper account"):
        _broker(live_handler).submit_short_open_order(
            CLIENT_ORDER_ID, SHORT_OCC, 1, covered_by="position:41"
        )


# ---------------------------------------------------------------------------
# Phase 3: margin-attested STOCK short + cover (stock-gated — the exact
# mirror: a naked short OPTION remains unconstructable through every path).
# ---------------------------------------------------------------------------


def test_stock_short_requires_margin_attestation_and_stock_symbol():
    b = _shape_guard_broker()
    with pytest.raises(ValueError, match="margin_attested_by"):
        b.submit_stock_short_order(CLIENT_ORDER_ID, "AAPL", 1, margin_attested_by="")
    # The mirror gate: an OCC OPTION symbol here would be a NAKED short
    # option — refused with the charter named, forever.
    with pytest.raises(ValueError, match="naked short options"):
        b.submit_stock_short_order(
            CLIENT_ORDER_ID, SHORT_OCC, 1, margin_attested_by="gate-chain:x"
        )
    with pytest.raises(ValueError, match="not a stock symbol"):
        b.submit_stock_cover_order(CLIENT_ORDER_ID, SHORT_OCC, 1)
    with pytest.raises(ValueError, match="not a stock symbol"):
        b.submit_stock_short_order(
            CLIENT_ORDER_ID, "aapl!", 1, margin_attested_by="gate-chain:x"
        )
    with pytest.raises(ValueError, match="quantity"):
        b.submit_stock_short_order(
            CLIENT_ORDER_ID, "AAPL", 0, margin_attested_by="gate-chain:x"
        )


def test_stock_short_and_cover_wire_shape():
    import json as _json
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=PAPER_ACCOUNT)
        bodies.append(_json.loads(request.content))
        return httpx.Response(
            200,
            json=_order_payload(symbol="AAPL", side=bodies[-1]["side"], status="accepted"),
        )

    broker = _broker(handler)
    broker.submit_stock_short_order(
        CLIENT_ORDER_ID, "AAPL", 30, margin_attested_by="gate-chain:oid-1"
    )
    assert bodies[-1]["side"] == "sell"
    assert bodies[-1]["position_intent"] == "sell_to_open"
    assert bodies[-1]["qty"] == "30"
    assert bodies[-1]["symbol"] == "AAPL"
    broker.submit_stock_cover_order(CLIENT_ORDER_ID + "x", "AAPL", 30)
    assert bodies[-1]["side"] == "buy"
    assert bodies[-1]["position_intent"] == "buy_to_close"


def test_stock_short_refused_on_live_account():
    def live_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json=LIVE_ACCOUNT)
        raise AssertionError("live account must never see an order")

    with pytest.raises(BrokerError, match="NOT a paper account"):
        _broker(live_handler).submit_stock_short_order(
            CLIENT_ORDER_ID, "AAPL", 1, margin_attested_by="gate-chain:x"
        )
