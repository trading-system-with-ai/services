"""Broker abstraction (development plan §11 — real execution).

Brokers are selected by name (``Settings.broker_provider``), mirroring the
libs.market_data and libs.llm patterns, so the execution venue is swappable by
configuration alone without touching consumer code.

THERE IS NO DEFAULT BROKER (§44 rule 18). The setting defaults to ``""`` and
:func:`get_broker` raises :class:`BrokerNotConfigured` for an empty/whitespace
name, so an unconfigured install places NOTHING. An unknown (non-empty) name is
still a ``ValueError`` — an operator typo, not an absent configuration.

THERE IS DELIBERATELY NO ``"simulated"`` ENTRY IN THIS REGISTRY.

That absence is the point, and it is load-bearing. The market data and LLM
packages each keep an opt-in ``"stub"`` because a synthetic *quote* clearly
labelled as such is a development convenience. A synthetic *fill* is not the
same kind of thing: it would be an execution that never happened, reported
through the same interface as one that did — invented data of the worst kind,
in the one place where the platform claims to have touched a real market. The
internal paper-fill simulator stays where it is, behind its own separately
labelled feature, and it is not reachable through this registry. When no
broker is configured, callers surface 503 ``BROKER_NOT_CONFIGURED`` and place
no order at all.

PAPER ONLY, RIGHT NOW: ``"alpaca_paper"`` is the only registered broker. Its
adapter refuses any non-paper host at construction and re-verifies the
account's paper flag with the broker before every submission. There is no
name, no URL and no setting in this package that reaches live trading.
"""
from typing import Callable

from .provider import (
    BrokerOrderLeg,  # noqa: F401
    BROKER_NOT_CONFIGURED_MESSAGE,
    BUY_TO_OPEN,
    BUY_TO_CLOSE,  # noqa: F401
    MLEG_LEG_SIDES,  # noqa: F401
    ORDER_SIDES,
    SELL_TO_OPEN,  # noqa: F401
    ORDER_STATUSES,
    SELL_TO_CLOSE,
    BrokerAccount,
    BrokerError,
    BrokerNotConfigured,
    BrokerOrder,
    BrokerPosition,
    BrokerProvider,
    BrokerRejected,
)


def _make_alpaca_paper() -> BrokerProvider:
    # Imported lazily so that importing libs.broker never requires httpx or
    # credentials. Construction raises BrokerError when either credential is
    # empty, guaranteeing the adapter is NEVER used keyless — and it pins
    # itself to the Alpaca paper host, so this factory cannot produce a live
    # trader no matter how the environment is set.
    #
    # ``alpaca_paper_base_url`` is passed through deliberately: it is NOT a
    # live seam. The constructor refuses any host but PAPER_HOST with a
    # ValueError naming the host, so pointing the setting at the live endpoint
    # is a hard startup failure here — which is exactly the intent (an
    # operator who tries must fail loudly, not trade real money).
    from libs.common.config import get_settings

    from .alpaca import ALPACA_PAPER_BASE_URL, AlpacaPaperBroker

    settings = get_settings()
    return AlpacaPaperBroker(
        api_key=settings.alpaca_api_key_id,
        api_secret=settings.alpaca_api_secret_key,
        base_url=settings.alpaca_paper_base_url or ALPACA_PAPER_BASE_URL,
    )


_BROKERS: dict[str, Callable[[], BrokerProvider]] = {
    # Alpaca paper trading. The ONLY entry, and paper-only by construction.
    # No "simulated" entry — see the module docstring.
    "alpaca_paper": _make_alpaca_paper,
}


def get_broker(name: str) -> BrokerProvider:
    """Instantiate the broker registered under `name` ("alpaca_paper").

    Raises :class:`BrokerNotConfigured` when `name` is empty or whitespace —
    the unconfigured state, in which the platform places no orders — and
    ``ValueError`` for an unknown non-empty name.
    """
    if not name or not name.strip():
        raise BrokerNotConfigured()
    try:
        factory = _BROKERS[name]
    except KeyError:
        known = sorted(_BROKERS)
        raise ValueError(f"unknown broker {name!r}; known: {known}") from None
    return factory()
