"""Event calendar provider abstraction (event spec §75; audit §6, ADR-008).

A SEPARATE registry from :mod:`libs.market_data`, cloning its contract
exactly — ``get_provider(name)``, :class:`ProviderNotConfigured` on an empty
name, ``ValueError`` on an unknown one, **no default and no cross-provider
fallback** — because calendar sources answer a different question and fail in
different ways:

============================  ==============================================
``alpaca_calendar``           exchange sessions (``/v2/calendar``, entitled)
``massive_calendar``          exchange holidays (entitled) + a Benzinga
                              earnings PROBE that is 403 today
``sec_edgar``                 authoritative PAST earnings from 8-K Item 2.02
                              + the cadence ESTIMATE for the next one
``fed``                       FOMC meeting/decision/presser/minutes and Fed
                              speeches, from free primary sources
``bls``                       CPI / PPI / Employment Situation / JOLTS
                              release schedule (free primary source)
``bea``                       GDP and PCE release schedule (free primary
                              source; ACTUALS need a free BEA_API_KEY)
``stub``                      SYNTHETIC events — tests only, opt-in only
============================  ==============================================

NO CROSS-PROVIDER FALLBACK. The audit's §6 earnings chain is ORDERED DATA,
not silent substitution: when the subscribed calendar says 403, the platform
does not quietly serve an estimate as if it were confirmed — it stores an
``EventStatus.ESTIMATED`` row that the UI is required to label. Which source
wins when two describe the same event is decided by
``EventSourceKind`` precedence in ``libs.trading_core.events``, never by
whichever adapter happened to run last.

Failure isolation is per-provider (audit §6): the gateway seam calls each
provider inside its own try, so a 403 on Benzinga, a Fed page-layout change
and an SEC rate-limit each cost only their own rows.
"""
import logging
from typing import Callable

from .macro_data import (  # noqa: F401 — the macro-values surface
    MacroDataError,
    MacroDataProvider,
    MacroObservation,
)
from .provider import (  # noqa: F401 — the package's public surface
    CAPABILITY_KEYS,
    EVENT_CALENDAR_NOT_CONFIGURED_MESSAGE,
    US_EVENT_TIMEZONE,
    CalendarProviderError,
    CapabilityNotAvailable,
    EventCalendarProvider,
    EventCandidate,
    MarketDataError,
    MarketDay,
    ProviderNotConfigured,
    blank_capabilities,
)

logger = logging.getLogger(__name__)


def _government_user_agent(settings) -> str:
    """The contact User-Agent every government source is sent (SEC's fair
    access policy; courtesy to BLS/BEA/Treasury). ``sec_user_agent`` is the
    single place an operator sets their contact address, so BLS and BEA reuse
    it rather than growing a second, separately-forgettable setting."""
    return (getattr(settings, "sec_user_agent", "") or "").strip() or (
        "trading-system-with-ai/0.1 (catalyst research; set SEC_USER_AGENT)"
    )


def _make_alpaca_calendar() -> EventCalendarProvider:
    # Imported lazily so importing libs.event_calendar never requires httpx
    # or a key (mirrors libs/market_data/__init__.py). Construction raises
    # when the Alpaca keys are blank — the adapter never fires keyless.
    from libs.common.config import get_settings

    from .alpaca_calendar import AlpacaCalendarProvider

    settings = get_settings()
    return AlpacaCalendarProvider(
        api_key_id=settings.alpaca_api_key_id,
        api_secret_key=settings.alpaca_api_secret_key,
    )


def _make_massive_calendar() -> EventCalendarProvider:
    from libs.common.config import get_settings

    from .massive_calendar import MassiveCalendarProvider

    settings = get_settings()
    return MassiveCalendarProvider(api_key=settings.massive_api_key)


def _make_sec_edgar() -> EventCalendarProvider:
    from libs.common.config import get_settings

    from .sec_edgar import SecEdgarProvider

    settings = get_settings()
    # sec_user_agent is required by SEC and has a Settings default that names
    # the env var to override; getattr keeps this registry importable if the
    # settings field has not landed yet.
    user_agent = getattr(settings, "sec_user_agent", "") or (
        "trading-system-with-ai/0.1 (catalyst research; set SEC_USER_AGENT)"
    )
    return SecEdgarProvider(user_agent=user_agent)


def _make_fed() -> EventCalendarProvider:
    from .fed import FedProvider

    return FedProvider()


def _make_bls() -> EventCalendarProvider:
    from libs.common.config import get_settings

    from .bls import BlsCalendarProvider

    return BlsCalendarProvider(user_agent=_government_user_agent(get_settings()))


def _make_bea() -> EventCalendarProvider:
    from libs.common.config import get_settings

    from .bea import BeaCalendarProvider

    return BeaCalendarProvider(user_agent=_government_user_agent(get_settings()))


def _make_stub() -> EventCalendarProvider:
    from .stub import StubEventCalendarProvider

    return StubEventCalendarProvider()


_PROVIDERS: dict[str, Callable[[], EventCalendarProvider]] = {
    # Exchange sessions — fixes "holidays are not modeled" (audit §5.2).
    "alpaca_calendar": _make_alpaca_calendar,
    # Exchange holidays (entitled) + the Benzinga earnings probe (403 today).
    "massive_calendar": _make_massive_calendar,
    # Free, authoritative, point-in-time-safe past earnings (audit §6 step 2).
    "sec_edgar": _make_sec_edgar,
    # Free primary source for every Fed event type (spec §9).
    "fed": _make_fed,
    # Free primary sources for the macro release calendar (spec §8, §38-§41).
    "bls": _make_bls,
    "bea": _make_bea,
    # Opt-in only (development + tests): SYNTHETIC, non-real events.
    "stub": _make_stub,
}

#: Providers needing no credentials — free primary sources. They are always
#: configured, which is why the platform still shows a real calendar on an
#: install with no vendor keys at all.
KEYLESS_PROVIDERS: tuple[str, ...] = ("sec_edgar", "fed", "bls", "bea")


def get_provider(name: str) -> EventCalendarProvider:
    """Instantiate the provider registered under `name`.

    Raises :class:`ProviderNotConfigured` when `name` is empty or whitespace —
    the unconfigured state — and ``ValueError`` for an unknown non-empty name
    (an operator typo, not an absent configuration).
    """
    if not name or not name.strip():
        raise ProviderNotConfigured(EVENT_CALENDAR_NOT_CONFIGURED_MESSAGE)
    try:
        factory = _PROVIDERS[name.strip()]
    except KeyError:
        known = sorted(_PROVIDERS)
        raise ValueError(
            f"unknown event calendar provider {name!r}; known: {known}"
        ) from None
    return factory()


def configured_provider_names(settings) -> list[str]:
    """Names of the providers whose prerequisites are satisfied.

    The keyless primary sources (SEC EDGAR, the Fed) are ALWAYS included:
    they need no subscription, so an install with no vendor keys still gets a
    real event calendar rather than an empty page. Vendor adapters appear
    only with their credentials, and the stub only when
    ``settings.event_calendar_providers`` names it explicitly — never by
    default, so a misconfiguration can never serve synthetic events.
    """
    requested = [
        n.strip()
        for n in str(getattr(settings, "event_calendar_providers", "") or "").split(",")
        if n.strip()
    ]
    if requested:
        names = []
        for name in requested:
            if name in _PROVIDERS:
                names.append(name)
            else:
                logger.warning(
                    "ignoring unknown event calendar provider %r (known: %s)",
                    name, sorted(_PROVIDERS),
                )
        return names

    names = list(KEYLESS_PROVIDERS)
    if (getattr(settings, "alpaca_api_key_id", "") or "").strip() and (
        getattr(settings, "alpaca_api_secret_key", "") or ""
    ).strip():
        names.append("alpaca_calendar")
    if (getattr(settings, "massive_api_key", "") or "").strip():
        names.append("massive_calendar")
    return names


def configured_providers(settings) -> list[EventCalendarProvider]:
    """Instantiate every configured provider, skipping ones that refuse.

    A provider whose construction raises (blank key that survived the check,
    a missing User-Agent) is LOGGED AND SKIPPED rather than taking the whole
    calendar down — the same failure isolation the ingestion tick applies to
    fetches (audit §6, spec §8).
    """
    providers: list[EventCalendarProvider] = []
    for name in configured_provider_names(settings):
        try:
            providers.append(get_provider(name))
        except (MarketDataError, ProviderNotConfigured, ValueError) as exc:
            logger.warning(
                "event calendar provider %r is configured but unusable: %s",
                name, exc,
            )
    return providers


def macro_data_provider(settings=None) -> "MacroDataProvider":
    """The source of macro time-series VALUES (§38-§41).

    Separate from :func:`get_provider`, which serves calendar DATES: the two
    answer different questions and fail differently. BLS v1 is keyless and
    therefore always available, so this returns the BLS client; BEA actuals
    live behind :func:`bea_macro_data_provider` and stay honestly unavailable
    without ``BEA_API_KEY`` rather than being substituted for.
    """
    from .macro_data import make_bls_macro_data_provider

    return make_bls_macro_data_provider(settings)


def bea_macro_data_provider(settings=None) -> "MacroDataProvider":
    """The BEA statistics client — unconfigured without ``bea_api_key``.

    Every call raises ``CapabilityNotAvailable`` until a key is set, which is
    the audit §6 "proven absence" verdict, never an estimate.
    """
    from .macro_data import make_bea_macro_data_provider

    return make_bea_macro_data_provider(settings)


def fed_documents_provider(settings=None):
    """The Fed DOCUMENTS client — statements, minutes, speeches (§42-§45).

    Deliberately NOT registered in ``_PROVIDERS``: that registry maps names to
    ``EventCalendarProvider``s, which answer "what is scheduled when", and a
    documents client has no ``fetch_events``. Handing it back from
    ``get_provider("fed")`` would satisfy the name and then fail at the call
    site, so the two surfaces stay separate and are reached by different
    functions.

    Keyless like the rest of the Fed surface, but never anonymous: the
    operator's contact User-Agent (``settings.sec_user_agent``, the single
    place a contact address is configured) is required, because
    federalreserve.gov answers 403 to requests without one.
    """
    from libs.common.config import get_settings

    from .fed_docs import FedDocumentsProvider

    return FedDocumentsProvider(
        user_agent=_government_user_agent(settings or get_settings())
    )
