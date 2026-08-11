"""Broker provider interface (development plan §11 — real execution).

Consumers depend only on this Protocol. The concrete adapter (Alpaca's paper
Trading API v2) is selected by configuration via
:func:`libs.broker.get_broker`, never imported directly by callers — this keeps
the broker swappable without touching consumer code, exactly as
``libs.market_data`` and ``libs.llm`` do for their providers.

NO BROKER, NO FILL (§44 rule 18, applied to execution): there is no default
broker and there is deliberately NO ``"simulated"`` entry in the registry.
When the broker is unconfigured the registry raises
:class:`BrokerNotConfigured` and every consumer surfaces that as an honest 503
— an unconfigured install must NEVER be handed an internally simulated fill
dressed up as a broker fill. A simulated fill presented as a broker fill is
invented data, the exact failure the honesty rule exists to prevent. The
internal paper-fill simulator remains its own separately-labelled feature; it
is not, and can never become, a broker.

PAPER ONLY, RIGHT NOW: the only registered broker is ``"alpaca_paper"``, whose
adapter hard-refuses any non-paper host at construction and re-verifies
``is_paper`` against the live account before every order submission. Live
trading is not reachable by configuration alone — there is no setting, no
provider name, and no base URL that turns this package into a live trader.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class BrokerError(RuntimeError):
    """A broker call failed: transport, HTTP status, auth, or configuration.

    This is the *fault* case — we do not know that the broker made a business
    decision, only that the conversation with it failed. Callers must never
    treat it as "the order did not exist"; an order may or may not have been
    accepted, so recovery goes through :meth:`BrokerProvider.get_order` with
    the client_order_id, not through a blind retry.
    """


class BrokerRejected(BrokerError):
    """The broker refused an order on its own terms (a business rejection).

    Distinct from a plain :class:`BrokerError`: the conversation succeeded and
    the broker's answer was "no" (insufficient buying power, untradeable
    symbol, market closed for the order type, PDT restriction, ...). Raised by
    :meth:`BrokerProvider.submit_order` when the broker rejects the submission
    outright. A rejection discovered *later*, by polling an already-accepted
    order, is NOT an exception — it surfaces as a
    :class:`BrokerOrder` with ``status == "REJECTED"``.
    """


# The message every unconfigured-broker path reports verbatim, so the API
# error, the logs and the tests all name the SAME missing configuration.
BROKER_NOT_CONFIGURED_MESSAGE = (
    "broker is not configured — set BROKER_PROVIDER and the corresponding "
    "credentials"
)


class BrokerNotConfigured(RuntimeError):
    """No broker is configured (``BROKER_PROVIDER`` unset).

    Mirrors :class:`libs.market_data.ProviderNotConfigured`: an unknown broker
    name stays a ``ValueError`` (an operator typo), while this is the absence
    of any configuration at all — the state a fresh install starts in. Callers
    map it to HTTP 503 ``BROKER_NOT_CONFIGURED`` and place NOTHING (§44 rule
    18), never falling back to the internal fill simulator.

    Deliberately NOT a subclass of :class:`BrokerError`: a caller catching
    broker *faults* around a submit path must not silently swallow "you never
    configured a broker" as though it were a transient failure to retry.
    """

    def __init__(self, message: str = BROKER_NOT_CONFIGURED_MESSAGE) -> None:
        super().__init__(message)


# The two order sides this platform can emit (§5). Sell-to-Open does not
# exist: a sell is only ever the close of an existing long.
BUY_TO_OPEN = "BUY_TO_OPEN"
SELL_TO_CLOSE = "SELL_TO_CLOSE"
ORDER_SIDES = (BUY_TO_OPEN, SELL_TO_CLOSE)

# The normalised order lifecycle every adapter maps its broker's vocabulary
# onto. Consumers switch on these and never on a broker-specific string.
ORDER_STATUSES = (
    "ACCEPTED",          # live at the broker, no fill yet
    "PARTIALLY_FILLED",  # some quantity filled, still working
    "FILLED",            # fully filled, terminal
    "REJECTED",          # the broker refused it, terminal
    "CANCELED",          # canceled by us or by the broker, terminal
    "EXPIRED",           # time_in_force elapsed unfilled, terminal
)


@dataclass(frozen=True)
class BrokerOrder:
    """One order as the broker currently sees it.

    ``client_order_id`` is OUR idempotency key — the adapter sends it with the
    submission and looks orders up by it, so a submission whose response was
    lost can be reconciled without risking a duplicate.

    ``status`` is always one of :data:`ORDER_STATUSES`. ``raw_status`` keeps
    the broker's own string verbatim alongside it, so an unrecognised
    lifecycle state stays visible in logs and audit rather than being
    flattened away by the mapping.
    """

    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str  # BUY_TO_OPEN | SELL_TO_CLOSE (§5)
    status: str  # one of ORDER_STATUSES
    requested_quantity: int
    filled_quantity: int
    filled_avg_price: float | None
    submitted_at: datetime
    raw_status: str  # the broker's own status string, preserved verbatim


@dataclass(frozen=True)
class BrokerAccount:
    """The broker account snapshot.

    ``is_paper`` is a SAFETY field, not a display field: it is read back from
    the broker itself and checked before every submission, so a misconfigured
    key pointing at a funded live account is refused rather than traded.
    """

    cash: float
    equity: float
    buying_power: float
    currency: str
    is_paper: bool
    account_number: str


@dataclass(frozen=True)
class BrokerPosition:
    """One open position as the broker reports it.

    ``quantity`` is signed as the broker reports it, but this platform only
    ever opens longs (§5), so a negative quantity here means the account holds
    a short we did not create — a reconciliation problem worth surfacing, not
    something to net away silently.
    """

    symbol: str
    quantity: int
    avg_entry_price: float
    market_value: float | None


class BrokerProvider(Protocol):
    """Structural interface every broker adapter must satisfy."""

    def submit_order(
        self, client_order_id: str, symbol: str, side: str, quantity: int
    ) -> BrokerOrder:
        """Submit a market order and return it as the broker accepted it.

        `side` is ``BUY_TO_OPEN`` or ``SELL_TO_CLOSE`` (§5) — there is no
        Sell-to-Open, so no call here can open a short.

        Raises :class:`BrokerRejected` when the broker refuses the order
        outright, and :class:`BrokerError` on any transport/HTTP fault or when
        the safety preconditions (paper account) are not met.
        """
        ...

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        """Return the order with our `client_order_id`, or None if unknown.

        None means "the broker has no such order" — a definitive answer, used
        to reconcile a submission whose response we lost. A transport failure
        raises :class:`BrokerError` instead, because it does NOT mean absence.
        """
        ...

    def list_positions(self) -> list[BrokerPosition]:
        """Return all open positions in the account."""
        ...

    def get_account(self) -> BrokerAccount:
        """Return the account snapshot, including the ``is_paper`` flag."""
        ...

    def cancel_order(self, client_order_id: str) -> None:
        """Cancel the order with our `client_order_id`.

        Idempotent-ish by intent: cancelling an order that is already terminal
        raises :class:`BrokerError` only if the broker itself faults; callers
        should re-read with :meth:`get_order` to learn the settled state.
        """
        ...
