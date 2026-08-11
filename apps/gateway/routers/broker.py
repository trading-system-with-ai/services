"""Broker status + reconciliation API (development plan §11, §18, §44 rule 18).

Two endpoints, both about the same question: does what we think we hold match
what the broker actually holds?

``GET /api/broker/status`` explains the execution venue — configured or not,
which provider, paper or nothing, and the live account snapshot when one is
reachable. It NEVER answers 503. It is the surface that EXPLAINS the
unconfigured state, so refusing to answer while unconfigured would be circular:
a user whose approve just 503'd comes here to find out why.

``GET /api/broker/reconcile`` compares the broker's positions and cash against
our local rows and reports every mismatch.

RECONCILIATION DOES NOT AUTO-CORRECT, AND THAT IS DELIBERATE (plan §18).

A mismatch means one of two ledgers is wrong and we do not know which. Local
rows drive risk sizing, heat, exits and every number the user sees; broker
positions are real money. "Fixing" either one automatically would be guessing
which reality to overwrite — and the wrong guess writes fiction into the very
records that are supposed to be the source of truth. Worse, the most likely
causes of a mismatch (a fill we never recorded, an exit that only moved local
rows, a manual trade in the broker UI) are exactly the conditions under which
continuing to trade compounds the error.

So a mismatch does two things and only two things: it writes a SYSTEM audit
event, and it PAUSES trading through the existing kill switch with an explicit
reason (plan §18 lists reconciliation mismatch as a documented kill-switch
trigger). A human then decides which ledger is right. Trading resumes only by
an explicit USER resume, exactly like any other pause.

NO SECRET MATERIAL. The account block reports the account NUMBER — an
identifier, not a credential, and the thing an operator needs to confirm they
are pointed at the account they think they are. API keys and secrets appear
nowhere in these responses, in any form, ever.
"""
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.broker import BrokerAccount, BrokerError, BrokerPosition
from libs.trading_core.models import ActorType, AuditAction

from .. import audit
from ..db import (
    Position,
    get_or_create_portfolio,
    get_or_create_system_state,
    get_session,
    utcnow,
)
from ..deps import (
    broker_mode,
    broker_unavailable_reason,
    resolve_broker,
    simulated_broker_mode,
)
from .orders import POSITION_OPEN, is_option_position

router = APIRouter(prefix="/api/broker", tags=["broker"])

logger = logging.getLogger(__name__)

# Cash is floating point on both sides and the broker rounds to cents, so an
# exact equality test would report a "mismatch" every time. A dollar is well
# inside any real divergence (a missed fill moves cash by far more) and well
# outside representation noise.
CASH_TOLERANCE_USD = 1.0

# Mismatch kinds, so a client can branch without parsing prose.
MISMATCH_QUANTITY = "QUANTITY_MISMATCH"
MISMATCH_MISSING_AT_BROKER = "MISSING_AT_BROKER"
MISMATCH_MISSING_LOCALLY = "MISSING_LOCALLY"
MISMATCH_CASH = "CASH_MISMATCH"

# The kill-switch reason written on a mismatch. Fixed prefix so an operator (or
# a test) can recognise a reconciliation pause at a glance.
PAUSE_REASON_PREFIX = "reconciliation mismatch"


def _account_payload(account: BrokerAccount) -> dict:
    """The account snapshot as reported to clients. Identifiers only."""
    return {
        "cash": account.cash,
        "equity": account.equity,
        "buying_power": account.buying_power,
        "currency": account.currency,
        "is_paper": account.is_paper,
        "account_number": account.account_number,
    }


@router.get("/status")
async def broker_status() -> dict:
    """The execution venue, explained (plan §44 rule 18).

    ``{"configured", "provider", "mode", "account", "error"}``:

    - ``configured`` — is there a usable execution venue at all;
    - ``provider``   — the configured name VERBATIM ("" when unset, never a
      cosmetic default that would let a UI claim an execution venue exists);
    - ``mode``       — "paper" when a real broker is configured (this platform
      has no other mode — the Alpaca adapter refuses any non-paper host), and
      ``null`` when unset or simulated;
    - ``account``    — the live broker snapshot, or ``null`` when there is no
      broker to ask or the call failed. An honest null, never a placeholder;
    - ``error``      — why ``account`` is null, when that needs explaining.

    NEVER 503, by design: this endpoint's whole job is to describe the
    unconfigured state, so it must be reachable in it. A broker call that fails
    is reported in ``error`` rather than raised — "we could not reach the
    broker" is information, not a server error.
    """
    provider = broker_mode()
    reason = broker_unavailable_reason()

    if not provider:
        return {
            "configured": False,
            "provider": "",
            "mode": None,
            "account": None,
            "error": reason,
        }

    if simulated_broker_mode():
        return {
            "configured": True,
            "provider": provider,
            # Not "paper": no broker account exists. Calling the internal
            # simulator a paper account is the exact conflation this platform
            # refuses — there is no account, so the honest answer is null.
            "mode": None,
            "account": None,
            "error": (
                "BROKER_PROVIDER=simulated: fills come from the INTERNAL "
                "simulator, not from any broker. There is no broker account "
                "to report. Development and backtest comparison only."
            ),
        }

    if reason is not None:
        # Configured but unusable (typo, missing credentials, refused host).
        return {
            "configured": False,
            "provider": provider,
            "mode": None,
            "account": None,
            "error": reason,
        }

    try:
        broker = resolve_broker()
        account = await asyncio.to_thread(broker.get_account)
    except BrokerError as exc:
        return {
            "configured": True,
            "provider": provider,
            "mode": "paper",
            "account": None,
            "error": f"the broker account could not be read: {exc}",
        }

    return {
        "configured": True,
        "provider": provider,
        # Paper is the only mode that exists: the adapter is pinned to the
        # paper host and re-verifies is_paper before every submission.
        "mode": "paper",
        "account": _account_payload(account),
        # A key pointing somewhere it should not be able to reach is worth
        # saying out loud even though submit_order would refuse it anyway.
        "error": (
            None
            if account.is_paper
            else (
                "the broker reports this is NOT a paper account — no order "
                "will be submitted; this adapter is paper-only"
            )
        ),
    }


def _local_open_quantities(positions: list[Position]) -> dict[str, int]:
    """Local OPEN stock quantities by ticker.

    Option positions are excluded: they were opened by the internal simulator
    (option execution is not wired at the broker) and have no counterpart in
    the broker's stock positions. Counting them would manufacture a mismatch
    on every sweep and pause trading for a difference that is expected.
    """
    out: dict[str, int] = {}
    for pos in positions:
        if is_option_position(pos):
            continue
        out[pos.ticker] = out.get(pos.ticker, 0) + pos.quantity
    return out


def _compare(
    broker_positions: list[BrokerPosition],
    local_quantities: dict[str, int],
    broker_cash: float,
    local_cash: float,
) -> list[dict]:
    """Every disagreement between the two ledgers, as ``mismatches`` rows.

    Each row is ``{"kind", "symbol", "broker", "local", "detail"}``. ``symbol``
    is null for the account-level cash row. Nothing here judges which side is
    right — that is a human's call (see the module docstring).
    """
    mismatches: list[dict] = []
    broker_quantities = {p.symbol: p.quantity for p in broker_positions}

    for symbol in sorted(set(broker_quantities) | set(local_quantities)):
        at_broker = broker_quantities.get(symbol)
        local = local_quantities.get(symbol)
        if at_broker == local:
            continue
        if at_broker is None:
            mismatches.append(
                {
                    "kind": MISMATCH_MISSING_AT_BROKER,
                    "symbol": symbol,
                    "broker": None,
                    "local": local,
                    "detail": (
                        f"we hold {local} share(s) of {symbol} locally but the "
                        "broker reports no position — an entry that never "
                        "filled, or an exit that only moved local rows"
                    ),
                }
            )
        elif local is None:
            mismatches.append(
                {
                    "kind": MISMATCH_MISSING_LOCALLY,
                    "symbol": symbol,
                    "broker": at_broker,
                    "local": None,
                    "detail": (
                        f"the broker holds {at_broker} share(s) of {symbol} "
                        "that we have no OPEN position for — a fill we never "
                        "recorded, or a trade placed outside this platform"
                    ),
                }
            )
        else:
            mismatches.append(
                {
                    "kind": MISMATCH_QUANTITY,
                    "symbol": symbol,
                    "broker": at_broker,
                    "local": local,
                    "detail": (
                        f"{symbol}: the broker holds {at_broker} share(s), we "
                        f"have {local} locally — a partial fill or a partial "
                        "exit is recorded on only one side"
                    ),
                }
            )

    if abs(broker_cash - local_cash) > CASH_TOLERANCE_USD:
        mismatches.append(
            {
                "kind": MISMATCH_CASH,
                "symbol": None,
                "broker": broker_cash,
                "local": local_cash,
                "detail": (
                    f"cash differs by ${abs(broker_cash - local_cash):,.2f} "
                    f"(broker ${broker_cash:,.2f} vs local ${local_cash:,.2f}, "
                    f"tolerance ${CASH_TOLERANCE_USD:,.2f})"
                ),
            }
        )
    return mismatches


@router.get("/reconcile")
async def reconcile(session: AsyncSession = Depends(get_session)) -> dict:
    """Compare broker positions + cash against local rows (plan §18).

    ``{"as_of", "configured", "broker", "local", "mismatches", "in_sync"}``.
    ``broker`` carries the account snapshot and its positions; ``local`` the
    portfolio cash and OPEN position quantities.

    ON MISMATCH: writes a SYSTEM audit event and PAUSES TRADING through the
    existing kill switch with an explicit reason. It does NOT auto-correct
    either ledger — see the module docstring for why halting is the deliberate
    choice. The pause and the audit event commit in one transaction (rule 12).

    Never places, cancels or amends an order. With no broker configured (or in
    simulated mode, where there IS no broker to compare against) it answers
    ``configured: false`` with null broker data, empty mismatches, and pauses
    NOTHING: an absent ledger is not a disagreement between ledgers.
    """
    as_of = datetime.now(timezone.utc).isoformat()
    reason = broker_unavailable_reason()
    portfolio = await get_or_create_portfolio(session)

    local_positions = list(
        (
            await session.execute(
                select(Position)
                .where(Position.status == POSITION_OPEN)
                .order_by(Position.ticker, Position.id)
            )
        )
        .scalars()
        .all()
    )
    local_quantities = _local_open_quantities(local_positions)
    local_block = {
        "cash": portfolio.cash,
        "positions": [
            {"symbol": symbol, "quantity": quantity}
            for symbol, quantity in sorted(local_quantities.items())
        ],
    }

    if reason is not None or simulated_broker_mode():
        # Nothing to reconcile against. Reported honestly rather than treated
        # as "everything matches" — an empty broker side is not agreement.
        return {
            "as_of": as_of,
            "configured": False,
            "broker": None,
            "local": local_block,
            "mismatches": [],
            "in_sync": False,
            "message": reason
            or (
                "BROKER_PROVIDER=simulated: there is no broker ledger to "
                "reconcile against (internal fills only)"
            ),
        }

    broker = resolve_broker()
    try:
        account = await asyncio.to_thread(broker.get_account)
        broker_positions = await asyncio.to_thread(broker.list_positions)
    except BrokerError as exc:
        # A fault is NOT a mismatch: we learned nothing about the broker's
        # ledger, so claiming a divergence — and pausing on it — would be
        # acting on an unknown. Report the failure and change nothing.
        logger.warning("reconciliation could not read the broker: %r", exc)
        return {
            "as_of": as_of,
            "configured": True,
            "broker": None,
            "local": local_block,
            "mismatches": [],
            "in_sync": False,
            "message": (
                f"the broker could not be read: {exc}. No comparison was "
                "possible — this is NOT a mismatch and nothing was paused."
            ),
        }

    mismatches = _compare(
        broker_positions, local_quantities, account.cash, portfolio.cash
    )
    broker_block = {
        "account": _account_payload(account),
        "positions": [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_entry_price": p.avg_entry_price,
                "market_value": p.market_value,
            }
            for p in broker_positions
        ],
    }

    result = {
        "as_of": as_of,
        "configured": True,
        "broker": broker_block,
        "local": local_block,
        "mismatches": mismatches,
        "in_sync": not mismatches,
    }

    if not mismatches:
        return result

    # --- MISMATCH: audit + pause (plan §18). No auto-correction. -----------
    summary = "; ".join(m["detail"] for m in mismatches)
    pause_reason = (
        f"{PAUSE_REASON_PREFIX}: {len(mismatches)} disagreement(s) between the "
        f"broker and local records — {summary}. Trading is paused until a "
        "human reconciles; the platform deliberately does NOT auto-correct "
        "either ledger (§18)."
    )
    state = await get_or_create_system_state(session)
    was_enabled = state.trading_enabled
    state.trading_enabled = False
    state.reason = pause_reason
    state.updated_by = "system:reconciliation"
    state.updated_at = utcnow()

    await audit.record(
        session,
        actor_type=ActorType.SYSTEM,
        action=AuditAction.KILL_SWITCH_TRIGGERED,
        entity_type="system_state",
        entity_id="global",
        details={
            "trigger": "BROKER_RECONCILIATION_MISMATCH",
            "provider": broker_mode(),
            "mismatches": mismatches,
            "broker_cash": account.cash,
            "local_cash": portfolio.cash,
            "trading_was_enabled": was_enabled,
            "auto_corrected": False,
            "policy": (
                "reconciliation mismatch is a documented kill-switch trigger "
                "(§18): halt and let a human decide which ledger is right — "
                "never overwrite one with the other"
            ),
        },
    )
    await session.commit()

    result["paused"] = True
    result["pause_reason"] = pause_reason
    return result
