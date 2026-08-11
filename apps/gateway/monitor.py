"""Automated position monitor (plan §26, §37).

A background asyncio task (started by the gateway lifespan when
``settings.position_monitor_interval_seconds`` > 0) that periodically runs
the SAME exit sweep as POST /api/positions/check-exits —
:func:`apps.gateway.routers.positions.run_exit_sweep`, never a
reimplementation (plan §21 spirit) — so §11 mechanical exits fire even when
nobody is clicking. Each sweep opens its own session, executes+commits under
the shared paper-execution lock (inside ``run_exit_sweep``), updates the
module-level :data:`STATE` served by GET /api/positions/monitor, increments
the telemetry counters and logs one structured line (§41).

RESILIENCE (documented behavior): the loop must survive transient failures —
a flaky DB connection or provider hiccup on one sweep must not kill the
monitor for the rest of the process lifetime. Every exception from a sweep is
logged (with traceback) and swallowed; the next tick runs normally.
``asyncio.CancelledError`` is ALWAYS re-raised so graceful shutdown (§26 —
lifespan cancels + awaits the task) is never swallowed.

NO MARKET DATA, NO SWEEP: when no market data provider is configured the
sweep is SKIPPED, not attempted. Every §11 exit rule compares against a
current price, so a sweep without market data could only act on invented
numbers — and acting means selling real positions. :func:`run_sweep_and_update`
therefore logs one WARNING per attempt and returns without touching a single
position, leaving the loop alive and quiet-ish (one line per tick, no
traceback, no crash loop) until a provider is configured.

NOTE for tests: httpx ASGITransport does not run the app lifespan, so the
loop never starts there — GET /api/positions/monitor then honestly reports
``enabled: false`` (§44 rule 18) and tests drive :func:`run_sweep_and_update`
or ``run_exit_sweep`` directly.
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from libs.common.config import get_settings
from libs.common.telemetry import REGISTRY
from libs.market_data import MARKET_DATA_NOT_CONFIGURED_MESSAGE

from .db import SessionLocal
from .deps import market_data_configured
from .routers.positions import run_exit_sweep

logger = logging.getLogger("position_monitor")

POSITION_MONITOR_SWEEPS_TOTAL = REGISTRY.counter(
    "position_monitor_sweeps_total",
    "Background exit sweeps completed by the position monitor (plan §26, §41).",
)
POSITION_MONITOR_EXITS_TOTAL = REGISTRY.counter(
    "position_monitor_exits_total",
    "Exits triggered (and executed) by background position-monitor sweeps "
    "(plan §26, §41).",
)


@dataclass
class MonitorState:
    """In-process state of the background monitor, served by
    GET /api/positions/monitor.

    ``enabled`` is True only while :func:`monitor_loop` is actually running
    (set on entry, cleared on exit) — configuration alone never claims a
    running task (§44 rule 18). ``last_result`` summarizes the newest sweep
    as ``{"checked": int, "exits_triggered": int}``.
    """

    enabled: bool = False
    interval_seconds: int = 0
    last_sweep_at: datetime | None = None
    sweeps_total: int = 0
    last_result: dict | None = None


# Module-level singleton: one gateway process runs at most one monitor loop.
STATE = MonitorState()


async def run_sweep_and_update() -> dict:
    """Run ONE monitor sweep and record it: session -> shared exit sweep
    (which commits, rule 12) -> STATE + telemetry + one structured log line.

    Split out from :func:`monitor_loop` so tests can drive a single sweep
    deterministically without a running background task.

    Skips entirely (no session, no evaluation, NO position change) with one
    WARNING line when no market data provider is configured — see the module
    docstring. The skip result keeps the sweep's shape with
    ``"skipped": "MARKET_DATA_NOT_CONFIGURED"`` added, and deliberately does
    NOT advance ``sweeps_total`` or the telemetry counter: nothing was swept.
    """
    if not market_data_configured():
        logger.warning(
            "position_monitor_skipped_no_market_data",
            extra={
                "extra_fields": {
                    "reason": MARKET_DATA_NOT_CONFIGURED_MESSAGE,
                    "sweeps_total": STATE.sweeps_total,
                }
            },
        )
        return {
            "checked": 0,
            "exits_triggered": [],
            "held": [],
            "skipped": "MARKET_DATA_NOT_CONFIGURED",
        }
    async with SessionLocal() as session:
        result = await run_exit_sweep(session)
    exits = len(result["exits_triggered"])
    STATE.last_sweep_at = datetime.now(timezone.utc)
    STATE.sweeps_total += 1
    STATE.last_result = {"checked": result["checked"], "exits_triggered": exits}
    POSITION_MONITOR_SWEEPS_TOTAL.inc()
    if exits:
        POSITION_MONITOR_EXITS_TOTAL.inc(exits)
    logger.info(
        "position_monitor_sweep",
        extra={
            "extra_fields": {
                "checked": result["checked"],
                "exits_triggered": exits,
                "held": len(result["held"]),
                "sweeps_total": STATE.sweeps_total,
            }
        },
    )
    return result


async def monitor_loop() -> None:
    """Sleep -> sweep forever (see module docstring for the resilience and
    cancellation contract). Started/cancelled by the gateway lifespan."""
    interval = get_settings().position_monitor_interval_seconds
    STATE.interval_seconds = interval
    STATE.enabled = True
    logger.info(
        "position_monitor_started",
        extra={"extra_fields": {"interval_seconds": interval}},
    )
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await run_sweep_and_update()
            except asyncio.CancelledError:
                # Graceful shutdown (§26) must never be swallowed.
                raise
            except Exception:
                # Transient failure (DB hiccup, provider error): log loudly,
                # keep the monitor alive for the next tick (module docstring).
                logger.exception("position_monitor_sweep_failed")
    finally:
        STATE.enabled = False
        logger.info("position_monitor_stopped")
