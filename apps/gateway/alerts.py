"""Alert classification — the declarative audit-row -> alert mapping (§18/§29/§38).

The alerts feed is a severity-graded READ of the audit trail: nothing here is
a new event source. ``ALERT_RULES`` maps an :class:`AuditAction` value to a
rule ``(severity, title_builder(details, entity_id) -> str, predicate(details)
-> bool | None)``; any audit action absent from the table is NOT an alert.

Severity mapping (§18/§29/§38):

- CRITICAL — TRADING_PAUSED, KILL_SWITCH_TRIGGERED, ORDER_REJECTED: trading
  capability itself changed or an order failed.
- WARNING — RISK_DECISION (ONLY when the §10 chain genuinely rejected/vetoed
  — approving previews are routine, not alerts), EXIT_GENERATED (a §11
  mechanical exit fired), BACKTEST_FAILED.
- INFO — ORDER_FILLED, TRADING_RESUMED: normal operation worth surfacing.

RISK_DECISION detail shape (written by routers/orders.py — ONE shape for both
paths): ``{"decision": "APPROVE"|"APPROVE_WITH_RESIZE"|"REJECT"|"VETOED",
"veto_gate": <gate name or None>, "gates": {...}, "reason_codes": [...]}``.
An early gate veto records decision "VETOED" with the failing gate name; a
risk-engine REJECT records decision "REJECT" with ``veto_gate``
"RISK_APPROVAL" and the engine's reason codes. The predicate accepts EITHER
signal, so a fully passing preview (decision APPROVE/APPROVE_WITH_RESIZE,
veto_gate None) is never an alert.

ORDER_FILLED / ORDER_REJECTED audit details do not carry the ticker/side
(their entity_id is the order row id); the alerts router enriches the details
from the referenced Order row before classifying — the builders here degrade
gracefully when a piece is missing rather than crashing on a sparse row.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from libs.trading_core.models import AuditAction

from .db import AuditEvent

CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

# Entity types whose entity_id IS the ticker (see the writers: order previews
# audit with entity_id=ticker, backtests with entity_id=ticker).
_TICKER_ENTITY_TYPES = frozenset({"order_preview", "backtests"})


@dataclass(frozen=True)
class Alert:
    """One alert row — the exact GET /api/alerts item contract."""

    id: int  # the audit row id
    ts: str  # iso8601
    severity: str  # CRITICAL | WARNING | INFO
    title: str
    ticker: str  # "" when the event is not symbol-scoped
    action: str
    correlation_id: str


@dataclass(frozen=True)
class AlertRule:
    """severity + title builder + optional keep-predicate for one action."""

    severity: str
    title: Callable[[Mapping, str], str]
    # None -> every row with this action is an alert; else the row is an
    # alert only when predicate(details) is True.
    predicate: Callable[[Mapping], bool] | None = None


# --- title builders (details, entity_id) -> str ----------------------------


def _trading_paused_title(details: Mapping, entity_id: str) -> str:
    reason = details.get("reason")
    return f"Trading paused — {reason}" if reason else "Trading paused"


def _trading_resumed_title(details: Mapping, entity_id: str) -> str:
    return "Trading resumed"


def _kill_switch_title(details: Mapping, entity_id: str) -> str:
    reason = details.get("reason")
    return f"Kill switch triggered — {reason}" if reason else "Kill switch triggered"


def _risk_decision_title(details: Mapping, entity_id: str) -> str:
    """"Risk rejected NVDA — HEAT_LIMIT, CASH_FLOOR" for an engine REJECT;
    "Risk vetoed MSFT — TRADING_POOL_AUTHORIZATION" for an earlier gate veto
    (the failing gate is always named)."""
    if details.get("decision") == "REJECT":
        codes = ", ".join(details.get("reason_codes") or []) or "risk limits"
        return f"Risk rejected {entity_id} — {codes}"
    gate = details.get("veto_gate") or "gate veto"
    return f"Risk vetoed {entity_id} — {gate}"


def _risk_decision_is_alert(details: Mapping) -> bool:
    """True ONLY for a genuine rejection/veto (§10): an early gate veto
    (decision "VETOED", veto_gate set) or a risk-engine REJECT (veto_gate
    "RISK_APPROVAL"). Approving previews (APPROVE / APPROVE_WITH_RESIZE with
    veto_gate None) are routine and must NOT surface as alerts."""
    return (
        details.get("veto_gate") is not None
        or details.get("decision") in ("REJECT", "VETOED")
    )


def _exit_generated_title(details: Mapping, entity_id: str) -> str:
    rule = details.get("rule") or "exit rule"
    ticker = details.get("ticker")
    if ticker:
        return f"Mechanical exit {ticker} — {rule}"
    return f"Mechanical exit — {rule}"


def _order_filled_title(details: Mapping, entity_id: str) -> str:
    """"Order filled: BUY_TO_OPEN 12 NVDA @ 187.42" — side/quantity/ticker
    come from the router's Order-row enrichment (the audit details alone
    identify the order only by id); missing pieces are simply omitted."""
    bits = [
        str(piece)
        for piece in (details.get("side"), details.get("quantity"), details.get("ticker"))
        if piece not in (None, "")
    ]
    title = "Order filled"
    if bits:
        title += ": " + " ".join(bits)
    price = details.get("fill_price")
    if isinstance(price, (int, float)):
        title += f" @ {price:.2f}"
    return title


def _order_rejected_title(details: Mapping, entity_id: str) -> str:
    ticker = details.get("ticker")
    title = f"Order rejected {ticker}" if ticker else "Order rejected"
    reason = details.get("reason") or details.get("error")
    if reason:
        title += f" — {reason}"
    return title


def _backtest_failed_title(details: Mapping, entity_id: str) -> str:
    error = details.get("error") or "unknown error"
    return f"Backtest failed {entity_id} — {error}"


# --- the declarative table (§18/§29/§38) -----------------------------------

ALERT_RULES: dict[str, AlertRule] = {
    AuditAction.TRADING_PAUSED.value: AlertRule(CRITICAL, _trading_paused_title),
    AuditAction.KILL_SWITCH_TRIGGERED.value: AlertRule(CRITICAL, _kill_switch_title),
    AuditAction.ORDER_REJECTED.value: AlertRule(CRITICAL, _order_rejected_title),
    AuditAction.RISK_DECISION.value: AlertRule(
        WARNING, _risk_decision_title, predicate=_risk_decision_is_alert
    ),
    AuditAction.EXIT_GENERATED.value: AlertRule(WARNING, _exit_generated_title),
    AuditAction.BACKTEST_FAILED.value: AlertRule(WARNING, _backtest_failed_title),
    AuditAction.ORDER_FILLED.value: AlertRule(INFO, _order_filled_title),
    AuditAction.TRADING_RESUMED.value: AlertRule(INFO, _trading_resumed_title),
}

# The SQL IN filter for the feed query — everything else is not an alert.
ALERT_ACTIONS: tuple[str, ...] = tuple(ALERT_RULES)


def _alert_ticker(row: AuditEvent, details: Mapping) -> str:
    """The alert's symbol scope: an explicit details ticker wins; entity
    types whose entity_id IS the ticker fall back to it; else "" (honest
    empty for global events like pause/resume/kill switch)."""
    ticker = details.get("ticker")
    if isinstance(ticker, str) and ticker:
        return ticker
    if row.entity_type in _TICKER_ENTITY_TYPES:
        return row.entity_id
    return ""


def classify(row: AuditEvent, extra_details: Mapping | None = None) -> Alert | None:
    """Classify one audit row against ``ALERT_RULES`` -> :class:`Alert` | None.

    Returns None when the action has no rule OR the rule's predicate drops
    the row (e.g. an approving RISK_DECISION). ``extra_details`` is the
    router's enrichment (Order-row ticker/side/quantity for order events);
    the row's OWN details always win on key collisions — the audit record is
    the source of truth for what actually happened.
    """
    rule = ALERT_RULES.get(row.action)
    if rule is None:
        return None
    details: dict = dict(extra_details or {})
    details.update(row.details or {})
    if rule.predicate is not None and not rule.predicate(details):
        return None
    return Alert(
        id=row.id,
        ts=row.ts.isoformat(),
        severity=rule.severity,
        title=rule.title(details, row.entity_id),
        ticker=_alert_ticker(row, details),
        action=row.action,
        correlation_id=row.correlation_id or "",
    )
