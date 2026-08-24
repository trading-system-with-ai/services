"""Alpaca PAPER-ONLY broker adapter (development plan §11 — real execution).

Speaks the Alpaca Trading API v2 over raw httpx — deliberately NOT the
``alpaca-py`` SDK, because every other provider in this codebase
(``libs/llm/anthropic.py``, ``libs/llm/openai.py``) is a raw-httpx adapter and
a second HTTP style would be one more thing to keep in sync. Credentials come
from configuration (``settings.alpaca_api_key`` / ``settings.alpaca_api_secret``)
— never hardcoded here.

PAPER ONLY. THIS IS THE WHOLE POINT OF THE CLASS.

Live trading must be impossible to reach by configuration alone, so the guard
is applied twice, at two different layers:

1. **Construction** PARSES the base URL and rejects it unless every component
   checks out: the host must equal the paper host (:data:`PAPER_HOST`)
   exactly, the scheme must be ``https`` (credentials never cross the network
   in cleartext) and no non-443 port may be given. The comparison is on the
   PARSED host, never a substring match, so decoration cannot smuggle the live
   host through — ``https://paper-api.alpaca.markets@api.alpaca.markets``
   parses to host ``api.alpaca.markets`` and is refused, as is
   ``https://API.ALPACA.MARKETS`` and ``https://paper-api.alpaca.markets.evil.com``.
   The accepted URL is then stored NORMALISED. There is no flag, no environment
   variable and no settings field that disables this check — going live would
   require editing this module, which is exactly the friction we want.
2. **Every submission** re-reads the account and refuses unless the broker
   itself reports ``is_paper``. The URL check alone would be trusting our own
   configuration; this check trusts the broker. A key pair that somehow points
   at a funded live account is refused before any order is POSTed.

Failure policy:
  - Missing/blank key or secret -> :class:`BrokerError` at construction time,
    so the adapter can never fire keyless (mirrors the LLM adapters).
  - Network / HTTP-level failures -> :class:`BrokerError`.
  - An order the broker refuses at submission -> :class:`BrokerRejected`.
  - An unrecognised Alpaca status -> mapped to ``ACCEPTED`` (the safe,
    non-terminal reading) with the raw string preserved and a WARNING logged.
    We NEVER guess ``FILLED``: inventing a fill is exactly the class of
    invented data this platform refuses to produce.

Credentials are never logged. The request logger redacts both header values.
"""
import logging
import re
from datetime import date, datetime, timezone
from urllib.parse import urlparse

import httpx

from .provider import (
    BrokerOrderLeg,
    SELL_TO_OPEN,
    BUY_TO_CLOSE,
    BUY_TO_OPEN,
    SELL_TO_CLOSE,
    BrokerAccount,
    BrokerError,
    BrokerOrder,
    BrokerPosition,
    BrokerRejected,
)

logger = logging.getLogger(__name__)

# The ONLY host this adapter will talk to. Not a default — an invariant.
PAPER_HOST = "paper-api.alpaca.markets"
ALPACA_PAPER_BASE_URL = f"https://{PAPER_HOST}"

DEFAULT_TIMEOUT_SECONDS = 15.0

KEY_HEADER = "APCA-API-KEY-ID"
SECRET_HEADER = "APCA-API-SECRET-KEY"
_REDACTED = "***redacted***"

# Our side vocabulary (§5) -> Alpaca's. BUY_TO_OPEN opens a long,
# SELL_TO_CLOSE closes one. There is deliberately NO entry that can open a
# short: "sell_to_open" is not in this table and Alpaca's "sell" is reachable
# only via SELL_TO_CLOSE.
_SIDE_TO_ALPACA = {
    BUY_TO_OPEN: "buy",
    SELL_TO_CLOSE: "sell",
}

# Bare OCC option symbol: UNDERLYING + yymmdd + C/P + strike*1000 (8 digits).
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

# Alpaca order status -> our normalised lifecycle. Anything absent from this
# table maps to ACCEPTED with a WARNING (see _map_status).
_STATUS_MAP = {
    "new": "ACCEPTED",
    "accepted": "ACCEPTED",
    "pending_new": "ACCEPTED",
    "accepted_for_bidding": "ACCEPTED",
    "partially_filled": "PARTIALLY_FILLED",
    "filled": "FILLED",
    "rejected": "REJECTED",
    "canceled": "CANCELED",
    # A cancel REQUEST is not a cancel: Alpaca's pending_cancel order is
    # still live and can still fill before the cancel lands. Mapping it
    # terminal would drop it out of the order-sync sweep's watch and any
    # last-moment fill would become invisible — so it stays non-terminal
    # (the raw status is preserved verbatim alongside).
    "pending_cancel": "ACCEPTED",
    "expired": "EXPIRED",
}


def _map_status(raw: str) -> str:
    """Map an Alpaca status string onto our lifecycle enum.

    An unrecognised status becomes ``ACCEPTED`` — the safe non-terminal
    reading, meaning "live at the broker, keep watching it" — never
    ``FILLED``. Guessing a fill would fabricate an execution that may not have
    happened; guessing "still working" only costs another poll.
    """
    mapped = _STATUS_MAP.get(raw)
    if mapped is None:
        logger.warning(
            "Unrecognised Alpaca order status %r; treating as ACCEPTED and "
            "preserving the raw status (never guessing FILLED)",
            raw,
        )
        return "ACCEPTED"
    return mapped


def _to_float(value: object) -> float | None:
    """Parse an Alpaca numeric string to float; None when absent/unparseable."""
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(value: object, default: int = 0) -> int:
    """Parse an Alpaca quantity to int; `default` when absent/unparseable.

    Alpaca sends quantities as decimal strings ("5", "5.0"), so this goes
    through float first and truncates — this platform only trades whole
    shares/contracts.
    """
    parsed = _to_float(value)
    return default if parsed is None else int(parsed)


def occ_option_symbol(
    underlying: str, expiry: date, strike: float, right: str
) -> str:
    """Build the COMPACT OCC option symbol Alpaca uses everywhere.

    Layout: ``ROOT`` (NO padding — Alpaca's compact form, matching what its
    data API chain snapshots and GET /v2/positions report), ``YYMMDD``
    expiry, ``C``/``P``, then the strike in thousandths zero-padded to 8
    digits — e.g. a 2026-09-18 NVDA 150 call is ``NVDA260918C00150000``.

    2026-08-17 FIX: this helper used to left-pad the root to 6 chars (the
    canonical exchange OCC layout). Alpaca does NOT: its positions and
    chains report compact symbols, so padded local keys could never
    string-match broker rows in §18 reconciliation, and the Phase-1/2 OCC
    regex gates rejected our own symbols. One format everywhere: compact.

    Options ride the SAME ``POST /v2/orders`` endpoint as equities; only the
    symbol format and the contract-vs-share unit differ, which is why this is a
    symbol helper rather than a second code path.

    The strike is rounded to the nearest tenth of a cent before scaling: a
    float like 150.0000000001 must not become ``00150000`` off-by-one and
    silently address a strike that does not exist.
    """
    root = underlying.strip().upper()
    if not root or len(root) > 6:
        raise ValueError(f"invalid option underlying {underlying!r} (1-6 chars)")
    if right not in ("C", "P"):
        raise ValueError(f"option right must be 'C' or 'P', got {right!r}")
    if strike <= 0:
        raise ValueError(f"option strike must be > 0, got {strike!r}")
    thousandths = int(round(round(strike, 4) * 1000))
    return f"{root}{expiry.strftime('%y%m%d')}{right}{thousandths:08d}"


def _parse_timestamp(value: object) -> datetime:
    """Parse an Alpaca RFC-3339 timestamp; fall back to now(UTC) if unusable.

    A missing timestamp must not sink an otherwise-good order response — the
    order exists at the broker either way, and losing it over a formatting
    detail would be worse than an approximate submitted_at.
    """
    if isinstance(value, str) and value:
        text = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            logger.warning("Unparseable Alpaca timestamp %r; using current time", value)
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class AlpacaPaperBroker:
    """BrokerProvider backed by the Alpaca PAPER Trading API v2.

    Paper-only by construction: see the module docstring for the two-layer
    guard. This class has no live mode.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = ALPACA_PAPER_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`transport` is injectable so tests can mock the network (httpx.MockTransport).

        Raises :class:`BrokerError` when either credential is missing/blank,
        and ``ValueError`` naming the host when `base_url` is not the Alpaca
        paper host — a live URL is an operator error we refuse loudly, not a
        mode we support.
        """
        if not api_key or not api_key.strip():
            raise BrokerError(
                "AlpacaPaperBroker requires a non-empty API key "
                "(settings.alpaca_api_key); the broker is never used keyless"
            )
        if not api_secret or not api_secret.strip():
            raise BrokerError(
                "AlpacaPaperBroker requires a non-empty API secret "
                "(settings.alpaca_api_secret); the broker is never used keyless"
            )

        # PAPER-ONLY GUARD, layer 1. The URL is PARSED and its components are
        # compared exactly — never substring-matched — so no amount of
        # decoration can smuggle the live host past this check. urlparse
        # already lowercases the hostname and strips userinfo, which is what
        # makes "https://paper-api.alpaca.markets@api.alpaca.markets" (host:
        # api.alpaca.markets) and "https://API.ALPACA.MARKETS" both fail here.
        parsed = urlparse(base_url)
        host = parsed.hostname or base_url
        if host != PAPER_HOST:
            raise ValueError(
                f"AlpacaPaperBroker refuses non-paper host {host!r}: this "
                f"adapter is PAPER ONLY and talks to {PAPER_HOST!r} exclusively. "
                "Live trading is not reachable by configuration."
            )
        # HTTPS ONLY. The host check alone would still permit
        # "http://paper-api.alpaca.markets", which sends the API key and
        # secret across the network in CLEARTEXT. Credentials are the one
        # thing this adapter must never leak, so the scheme is an invariant
        # too, not a preference.
        if parsed.scheme != "https":
            raise ValueError(
                f"AlpacaPaperBroker refuses scheme {parsed.scheme!r}: the "
                "Alpaca API is HTTPS only and the API key/secret must never "
                "cross the network in cleartext."
            )
        # No custom port. The paper host serves HTTPS on 443; an explicit port
        # on the real hostname is not something a legitimate configuration
        # needs, and allowing it widens the target for no benefit.
        if parsed.port is not None and parsed.port != 443:
            raise ValueError(
                f"AlpacaPaperBroker refuses port {parsed.port}: {PAPER_HOST!r} "
                "is reached on the default HTTPS port only."
            )

        self.api_key = api_key
        self.api_secret = api_secret
        # Normalise: the guard compared parsed components, so store a canonical
        # URL rather than whatever spelling was supplied. This keeps the
        # request log, the error text and any future host comparison reading
        # the same string a case-variant spelling would otherwise fork.
        self.base_url = f"https://{PAPER_HOST}{parsed.path.rstrip('/')}"
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            KEY_HEADER: self.api_key,
            SECRET_HEADER: self.api_secret,
            "content-type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        allow_404: bool = False,
    ) -> httpx.Response:
        """Perform one Alpaca call, translating transport/HTTP faults.

        Returns the response for 2xx, and for 404 when `allow_404` (the caller
        distinguishes "no such order" from a fault). Everything else raises
        :class:`BrokerError`. Credentials never appear in the log line.
        """
        url = f"{self.base_url}{path}"
        logger.debug(
            "Alpaca %s %s (headers: %s=%s, %s=%s)",
            method, url, KEY_HEADER, _REDACTED, SECRET_HEADER, _REDACTED,
        )
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.request(
                    method, url, json=json_body, params=params, headers=self._headers()
                )
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca API request failed: {exc!r}") from exc

        if response.status_code == 404 and allow_404:
            return response
        if response.status_code >= 400:
            raise BrokerError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"{method} {path}: {response.text[:500]}"
            )
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict | list:
        try:
            return response.json()
        except ValueError as exc:
            raise BrokerError(
                f"Alpaca API returned a non-JSON body: {response.text[:200]!r}"
            ) from exc

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_order(payload: dict, fallback_side: str | None = None) -> BrokerOrder:
        """Build a BrokerOrder from an Alpaca order object.

        `fallback_side` is used only when Alpaca's payload omits the side (it
        does not in practice) — the side we asked for is then the best answer.
        """
        raw_status = str(payload.get("status", ""))
        alpaca_side = str(payload.get("side", "")).lower()
        if alpaca_side == "buy":
            side = BUY_TO_OPEN
        elif alpaca_side == "sell":
            side = SELL_TO_CLOSE
        else:
            side = fallback_side or alpaca_side.upper()

        return BrokerOrder(
            broker_order_id=str(payload.get("id", "")),
            client_order_id=str(payload.get("client_order_id", "")),
            symbol=str(payload.get("symbol", "")),
            side=side,
            status=_map_status(raw_status),
            requested_quantity=_to_int(payload.get("qty")),
            filled_quantity=_to_int(payload.get("filled_qty")),
            filled_avg_price=_to_float(payload.get("filled_avg_price")),
            submitted_at=_parse_timestamp(payload.get("submitted_at")),
            raw_status=raw_status,
        )

    # ------------------------------------------------------------------
    # BrokerProvider
    # ------------------------------------------------------------------

    def get_account(self) -> BrokerAccount:
        """Return the account snapshot (``GET /v2/account``).

        ``is_paper`` is taken from Alpaca's own boolean when present. If the
        field is absent we fall back to the host we are pinned to — which the
        constructor already guarantees is the paper host.
        """
        payload = self._json(self._request("GET", "/v2/account"))
        if not isinstance(payload, dict):
            raise BrokerError("Alpaca /v2/account returned an unexpected payload")

        raw_is_paper = payload.get("is_paper")
        is_paper = (
            bool(raw_is_paper)
            if raw_is_paper is not None
            else urlparse(self.base_url).hostname == PAPER_HOST
        )
        return BrokerAccount(
            cash=_to_float(payload.get("cash")) or 0.0,
            equity=_to_float(payload.get("equity")) or 0.0,
            buying_power=_to_float(payload.get("buying_power")) or 0.0,
            currency=str(payload.get("currency", "USD")),
            is_paper=is_paper,
            account_number=str(payload.get("account_number", "")),
        )

    def submit_order(
        self, client_order_id: str, symbol: str, side: str, quantity: int
    ) -> BrokerOrder:
        """Submit a day market order (``POST /v2/orders``).

        `side` is ``BUY_TO_OPEN`` -> Alpaca ``"buy"`` or ``SELL_TO_CLOSE`` ->
        Alpaca ``"sell"``. **No mapping here can produce a short**: there is no
        Sell-to-Open in this platform (§5), and a ``"sell"`` is only ever
        emitted to close an existing long — the gateway enforces that a
        SELL_TO_CLOSE never exceeds the held quantity before it ever calls
        this method. This adapter simply has no vocabulary for opening a short.

        SAFETY: the account is read FIRST and the order is POSTed only if the
        broker reports a paper account. A live account raises
        :class:`BrokerError` with no order ever sent.
        """
        if side not in _SIDE_TO_ALPACA:
            raise ValueError(
                f"unknown order side {side!r}; known: {sorted(_SIDE_TO_ALPACA)} "
                "(Sell-to-Open does not exist — §5)"
            )
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        # PAPER-ONLY GUARD, layer 2: trust the broker, not our own config.
        # This happens BEFORE the POST so a live account never sees an order.
        account = self.get_account()
        if not account.is_paper:
            raise BrokerError(
                "refusing to submit an order: Alpaca reports account "
                f"{account.account_number!r} is NOT a paper account. This "
                "adapter is paper-only and will not trade real money."
            )

        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": _SIDE_TO_ALPACA[side],
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }

        url = f"{self.base_url}/v2/orders"
        logger.debug(
            "Alpaca POST %s (headers: %s=%s, %s=%s) payload=%s",
            url, KEY_HEADER, _REDACTED, SECRET_HEADER, _REDACTED, payload,
        )
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca API request failed: {exc!r}") from exc

        # 4xx on submit is the broker refusing the order on its own terms —
        # a business rejection, not a transport fault.
        if 400 <= response.status_code < 500:
            raise BrokerRejected(
                f"Alpaca rejected order {client_order_id!r} for {symbol} "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            )
        if response.status_code >= 500:
            raise BrokerError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"POST /v2/orders: {response.text[:500]}"
            )

        body = self._json(response)
        if not isinstance(body, dict):
            raise BrokerError("Alpaca /v2/orders returned an unexpected payload")

        order = self._parse_order(body, fallback_side=side)
        # Alpaca can also answer 200 with status "rejected" — same fact,
        # different envelope, so it raises the same exception. A rejection
        # discovered later by get_order() stays a BrokerOrder(REJECTED).
        if order.status == "REJECTED":
            raise BrokerRejected(
                f"Alpaca rejected order {client_order_id!r} for {symbol} "
                f"(status {order.raw_status!r})"
            )
        return order

    def submit_short_open_order(
        self,
        client_order_id: str,
        option_symbol: str,
        quantity: int,
        covered_by: str,
    ) -> BrokerOrder:
        """Sell-to-open ONE collateralized short option leg (Phase 2:
        covered call / cash-secured put).

        SAFETY SHAPE: OPTION symbols only (the OCC regex is the gate) — a
        stock symbol here raises, so SHORT STOCK remains unconstructable
        through every path until the margin phase builds it deliberately.
        ``covered_by`` is the MANDATORY collateral attestation (the platform
        position id / cash reservation backing this short); it must be
        non-empty and is embedded in the client metadata trail. The REAL
        collateral enforcement lives in the gateway BEFORE this call — this
        parameter exists so a naked call site cannot even typecheck its way
        past review.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        if not covered_by or not str(covered_by).strip():
            raise ValueError(
                "covered_by attestation is required: a short open without "
                "named collateral is a naked short and does not exist in "
                "this platform (§5/§4)"
            )
        if _OCC_RE.match(option_symbol) is None:
            raise ValueError(
                f"{option_symbol!r} is not a bare OCC OPTION symbol — "
                "sell-to-open exists ONLY for collateralized option legs "
                "(no short stock, §5)"
            )

        account = self.get_account()
        if not account.is_paper:
            raise BrokerError(
                "refusing to submit a short-open order: Alpaca reports "
                f"account {account.account_number!r} is NOT a paper account."
            )

        payload = {
            "symbol": option_symbol,
            "qty": str(quantity),
            "side": "sell",
            "position_intent": "sell_to_open",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        url = f"{self.base_url}/v2/orders"
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca API request failed: {exc!r}") from exc
        if 400 <= response.status_code < 500:
            raise BrokerRejected(
                f"Alpaca rejected short-open {client_order_id!r} for "
                f"{option_symbol} (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )
        if response.status_code >= 500:
            raise BrokerError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"POST /v2/orders (short open): {response.text[:500]}"
            )
        body = self._json(response)
        if not isinstance(body, dict):
            raise BrokerError(
                "Alpaca /v2/orders (short open) returned an unexpected payload"
            )
        order = self._parse_order(body, fallback_side="SELL_TO_OPEN")
        if order.status == "REJECTED":
            raise BrokerRejected(
                f"Alpaca rejected short-open {client_order_id!r} "
                f"(status {order.raw_status!r})"
            )
        return order

    def submit_short_close_order(
        self, client_order_id: str, option_symbol: str, quantity: int
    ) -> BrokerOrder:
        """Buy BACK one short option leg (BUY_TO_CLOSE — Phase 2).

        OCC-gated like the short open; closing a short strictly REDUCES
        risk. Deliberately a separate method so the single-leg
        ``submit_order`` keeps its exact two-word vocabulary
        (BUY_TO_OPEN / SELL_TO_CLOSE) and the adversarial §5 tests stay
        meaningful.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        if _OCC_RE.match(option_symbol) is None:
            raise ValueError(
                f"{option_symbol!r} is not a bare OCC OPTION symbol — only "
                "short OPTION legs can be bought back here"
            )
        account = self.get_account()
        if not account.is_paper:
            raise BrokerError(
                "refusing to submit a buyback order: Alpaca reports account "
                f"{account.account_number!r} is NOT a paper account."
            )
        payload = {
            "symbol": option_symbol,
            "qty": str(quantity),
            "side": "buy",
            "position_intent": "buy_to_close",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        url = f"{self.base_url}/v2/orders"
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca API request failed: {exc!r}") from exc
        if 400 <= response.status_code < 500:
            raise BrokerRejected(
                f"Alpaca rejected buyback {client_order_id!r} for "
                f"{option_symbol} (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )
        if response.status_code >= 500:
            raise BrokerError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"POST /v2/orders (buyback): {response.text[:500]}"
            )
        body = self._json(response)
        if not isinstance(body, dict):
            raise BrokerError(
                "Alpaca /v2/orders (buyback) returned an unexpected payload"
            )
        order = self._parse_order(body, fallback_side="BUY_TO_CLOSE")
        if order.status == "REJECTED":
            raise BrokerRejected(
                f"Alpaca rejected buyback {client_order_id!r} "
                f"(status {order.raw_status!r})"
            )
        return order

    _STOCK_RE = re.compile(r"^[A-Z][A-Z.]{0,9}$")

    def _post_simple_order(
        self, payload: dict, label: str, fallback_side: str
    ) -> BrokerOrder:
        """Shared POST /v2/orders plumbing for the Phase 3 stock-short pair."""
        account = self.get_account()
        if not account.is_paper:
            raise BrokerError(
                f"refusing to submit a {label} order: Alpaca reports account "
                f"{account.account_number!r} is NOT a paper account."
            )
        url = f"{self.base_url}/v2/orders"
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca API request failed: {exc!r}") from exc
        if 400 <= response.status_code < 500:
            raise BrokerRejected(
                f"Alpaca rejected {label} {payload['client_order_id']!r} for "
                f"{payload['symbol']} (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )
        if response.status_code >= 500:
            raise BrokerError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"POST /v2/orders ({label}): {response.text[:500]}"
            )
        body = self._json(response)
        if not isinstance(body, dict):
            raise BrokerError(
                f"Alpaca /v2/orders ({label}) returned an unexpected payload"
            )
        order = self._parse_order(body, fallback_side=fallback_side)
        if order.status == "REJECTED":
            raise BrokerRejected(
                f"Alpaca rejected {label} {payload['client_order_id']!r} "
                f"(status {order.raw_status!r})"
            )
        return order

    def submit_stock_short_order(
        self,
        client_order_id: str,
        symbol: str,
        quantity: int,
        margin_attested_by: str,
    ) -> BrokerOrder:
        """Sell-to-open SHORT STOCK (roadmap Phase 3 — margin-backed).

        SAFETY SHAPE — the exact mirror of ``submit_short_open_order``:
        STOCK symbols only. An OCC option symbol here raises, so a naked
        short OPTION remains unconstructable through every path, forever
        (§4 charter + broker refusal). ``margin_attested_by`` is the
        MANDATORY attestation naming the §10 gate-chain audit that sized
        this short against margin buying power; the REAL enforcement lives
        in the gateway before this call — the parameter exists so a naked
        call site cannot typecheck its way past review. Alpaca enforces
        locate/HTB and maintenance margin on its side.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        if not margin_attested_by or not str(margin_attested_by).strip():
            raise ValueError(
                "margin_attested_by attestation is required: a stock short "
                "not sized against margin buying power does not exist in "
                "this platform (§5/Phase 3)"
            )
        if _OCC_RE.match(symbol) is not None:
            raise ValueError(
                f"{symbol!r} is an OCC OPTION symbol — naked short options "
                "do not exist in this platform (§4/§5) and never will"
            )
        if self._STOCK_RE.match(symbol) is None:
            raise ValueError(
                f"{symbol!r} is not a stock symbol — short-stock opens "
                "exist ONLY for equities (§5/Phase 3)"
            )
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": "sell",
            "position_intent": "sell_to_open",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        return self._post_simple_order(payload, "stock short", "SELL_TO_OPEN")

    def submit_stock_cover_order(
        self, client_order_id: str, symbol: str, quantity: int
    ) -> BrokerOrder:
        """Buy BACK short stock (buy-to-cover — Phase 3). Stock-gated like
        the short open; covering strictly REDUCES risk, so it carries no
        attestation and stays allowed under the kill switch (§18)."""
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        if _OCC_RE.match(symbol) is not None or self._STOCK_RE.match(symbol) is None:
            raise ValueError(
                f"{symbol!r} is not a stock symbol — only short STOCK can "
                "be covered here"
            )
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": "buy",
            "position_intent": "buy_to_close",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        return self._post_simple_order(payload, "stock cover", "BUY_TO_CLOSE")

    def submit_mleg_order(
        self,
        client_order_id: str,
        legs: list[BrokerOrderLeg],
        quantity: int,
    ) -> BrokerOrder:
        """Submit an ATOMIC two-leg option order (Alpaca ``order_class:
        "mleg"``) for a DEFINED-RISK vertical (roadmap Phase 1).

        THE SHAPE GUARD IS THE §5 SAFETY BOUNDARY: exactly two legs on the
        same underlying, same expiry, same right, different strikes; ratios
        1:1; the pair is either an OPEN (BUY_TO_OPEN long + SELL_TO_OPEN
        short — the short strike must be the FARTHER-OTM one so the long
        covers it) or a CLOSE (SELL_TO_CLOSE + BUY_TO_CLOSE). Anything else
        — a lone SELL_TO_OPEN, mismatched quantities, a credit-shaped pair,
        cross-expiry legs — raises before any network I/O. This is the ONLY
        path that can emit a sell-to-open, and it cannot emit one naked.

        Returns one BrokerOrder for the whole spread: ``symbol`` is
        "LONG_OCC/SHORT_OCC" and ``filled_avg_price`` the NET per-share
        debit/credit (buy legs minus sell legs) once every leg reports a
        fill — None until then.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        if len(legs) != 2:
            raise ValueError(
                f"mleg orders are exactly TWO legs (defined-risk vertical), "
                f"got {len(legs)}"
            )
        sides = {leg.side for leg in legs}
        if sides == {BUY_TO_OPEN, SELL_TO_OPEN}:
            opening = True
        elif sides == {SELL_TO_CLOSE, BUY_TO_CLOSE}:
            opening = False
        else:
            raise ValueError(
                "mleg pair must be {BUY_TO_OPEN, SELL_TO_OPEN} (open) or "
                f"{{SELL_TO_CLOSE, BUY_TO_CLOSE}} (close), got {sorted(sides)} "
                "— no other combination exists (§5)"
            )
        if any(leg.ratio != 1 for leg in legs):
            raise ValueError("defined-risk vertical requires 1:1 leg ratios")

        parsed = []
        for leg in legs:
            m = _OCC_RE.match(leg.symbol)
            if m is None:
                raise ValueError(f"leg symbol {leg.symbol!r} is not a bare OCC symbol")
            parsed.append(
                (leg, m.group(1), m.group(2), m.group(3), int(m.group(4)) / 1000.0)
            )
        (leg_a, und_a, exp_a, right_a, strike_a), (leg_b, und_b, exp_b, right_b, strike_b) = parsed
        if und_a != und_b or exp_a != exp_b or right_a != right_b:
            raise ValueError(
                "defined-risk vertical requires SAME underlying, expiry and "
                f"right, got {leg_a.symbol!r} vs {leg_b.symbol!r}"
            )
        if strike_a == strike_b:
            raise ValueError("vertical legs must have different strikes")
        # The long (bought-to-open / sold-to-close) leg must COVER the short:
        # for calls the short strike sits ABOVE the long, for puts BELOW.
        long_leg = next(
            (l, s) for (l, u, e, r, s) in parsed if l.side in (BUY_TO_OPEN, SELL_TO_CLOSE)
        )
        short_leg = next(
            (l, s) for (l, u, e, r, s) in parsed if l.side in (SELL_TO_OPEN, BUY_TO_CLOSE)
        )
        if right_a == "C" and not short_leg[1] > long_leg[1]:
            raise ValueError(
                "call vertical: the short strike must be ABOVE the long "
                f"strike (covered), got long {long_leg[1]:g} / short {short_leg[1]:g}"
            )
        if right_a == "P" and not short_leg[1] < long_leg[1]:
            raise ValueError(
                "put vertical: the short strike must be BELOW the long "
                f"strike (covered), got long {long_leg[1]:g} / short {short_leg[1]:g}"
            )

        # PAPER-ONLY GUARD, layer 2 (same as the single-leg path).
        account = self.get_account()
        if not account.is_paper:
            raise BrokerError(
                "refusing to submit an mleg order: Alpaca reports account "
                f"{account.account_number!r} is NOT a paper account."
            )

        intent = {
            BUY_TO_OPEN: ("buy", "buy_to_open"),
            SELL_TO_OPEN: ("sell", "sell_to_open"),
            SELL_TO_CLOSE: ("sell", "sell_to_close"),
            BUY_TO_CLOSE: ("buy", "buy_to_close"),
        }
        payload = {
            "order_class": "mleg",
            "qty": str(quantity),
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "ratio_qty": str(leg.ratio),
                    "side": intent[leg.side][0],
                    "position_intent": intent[leg.side][1],
                }
                for leg in legs
            ],
        }

        url = f"{self.base_url}/v2/orders"
        logger.debug(
            "Alpaca POST %s (mleg, headers redacted) payload=%s", url, payload
        )
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"Alpaca API request failed: {exc!r}") from exc

        if 400 <= response.status_code < 500:
            raise BrokerRejected(
                f"Alpaca rejected mleg order {client_order_id!r} "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            )
        if response.status_code >= 500:
            raise BrokerError(
                f"Alpaca API returned HTTP {response.status_code} for "
                f"POST /v2/orders (mleg): {response.text[:500]}"
            )
        body = self._json(response)
        if not isinstance(body, dict):
            raise BrokerError("Alpaca /v2/orders (mleg) returned an unexpected payload")
        order = self._parse_mleg_order(body, legs, fallback_side=legs[0].side)
        if order.status == "REJECTED":
            raise BrokerRejected(
                f"Alpaca rejected mleg order {client_order_id!r} "
                f"(status {order.raw_status!r})"
            )
        return order

    def _parse_mleg_order(
        self, payload: dict, legs: list[BrokerOrderLeg], fallback_side: str
    ) -> BrokerOrder:
        """BrokerOrder for a whole spread: symbol "LONG/SHORT"; the net
        per-share fill (buys minus sells) only when EVERY leg has filled —
        None until then (an half-filled net would be an invented number)."""
        base = self._parse_order(payload, fallback_side=fallback_side)
        leg_rows = payload.get("legs")
        net: float | None = None
        filled = 0
        if isinstance(leg_rows, list) and leg_rows:
            total = 0.0
            all_filled = True
            for row in leg_rows:
                if not isinstance(row, dict):
                    all_filled = False
                    break
                price = row.get("filled_avg_price")
                qty = row.get("filled_qty")
                try:
                    price_f = float(price) if price is not None else None
                    qty_i = int(float(qty)) if qty is not None else 0
                except (TypeError, ValueError):
                    price_f, qty_i = None, 0
                filled = max(filled, qty_i)
                if price_f is None or qty_i <= 0:
                    all_filled = False
                    continue
                total += price_f if str(row.get("side")) == "buy" else -price_f
            if all_filled:
                net = total
        symbol = "/".join(l.symbol for l in legs)
        return BrokerOrder(
            broker_order_id=base.broker_order_id,
            client_order_id=base.client_order_id,
            symbol=symbol,
            side=fallback_side,
            status=base.status,
            requested_quantity=base.requested_quantity,
            filled_quantity=filled or base.filled_quantity,
            filled_avg_price=net,
            submitted_at=base.submitted_at,
            raw_status=base.raw_status,
        )

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        """Look an order up by OUR id (``GET /v2/orders:by_client_order_id``).

        Returns None on 404 — a definitive "the broker has no such order",
        which is what makes this safe to use for reconciling a submission whose
        response was lost. Transport faults raise instead, because they do not
        mean absence.
        """
        response = self._request(
            "GET",
            "/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
            allow_404=True,
        )
        if response.status_code == 404:
            return None
        body = self._json(response)
        if not isinstance(body, dict):
            raise BrokerError(
                "Alpaca /v2/orders:by_client_order_id returned an unexpected payload"
            )
        return self._parse_order(body)

    def list_positions(self) -> list[BrokerPosition]:
        """Return all open positions (``GET /v2/positions``)."""
        body = self._json(self._request("GET", "/v2/positions"))
        if not isinstance(body, list):
            raise BrokerError("Alpaca /v2/positions returned an unexpected payload")

        positions: list[BrokerPosition] = []
        for row in body:
            if not isinstance(row, dict):
                logger.warning("Skipping malformed Alpaca position entry")
                continue
            positions.append(
                BrokerPosition(
                    symbol=str(row.get("symbol", "")),
                    quantity=_to_int(row.get("qty")),
                    avg_entry_price=_to_float(row.get("avg_entry_price")) or 0.0,
                    market_value=_to_float(row.get("market_value")),
                )
            )
        return positions

    def cancel_order(self, client_order_id: str) -> None:
        """Cancel the order carrying our `client_order_id`.

        Alpaca cancels by ITS order id, so this resolves the client id first.
        An unknown client id raises :class:`BrokerError` — asking to cancel
        something the broker never heard of is a real problem worth surfacing,
        not a silent no-op.
        """
        order = self.get_order(client_order_id)
        if order is None:
            raise BrokerError(
                f"cannot cancel unknown order {client_order_id!r}: "
                "the broker has no order with that client_order_id"
            )
        self._request("DELETE", f"/v2/orders/{order.broker_order_id}")
