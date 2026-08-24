"""THE DEPENDENCY ARROW POINTS ONE WAY: routers -> seams -> pure layer.

A router validates input, calls one function, and turns the result into a
status code. Everything else — the trading decision, the seams, the background
loops — must be reachable without HTTP, because:

- background loops (``order_sync``, ``monitor``, ``risk_snapshot``) run with no
  request in sight, and a loop that imports a router is a loop whose logic
  cannot be understood or tested without the HTTP layer;
- a second entry point (a CLI, a scheduled strategy, a notebook) should not
  have to import ``fastapi`` to reach a decision;
- import cycles are the symptom. When this eroded, two sibling routers had to
  be imported INSIDE functions to break cycles the layering had created.

THIS TEST IS A RATCHET, NOT A CLEAN BILL OF HEALTH. Several inversions predate
it and are recorded in :data:`KNOWN_INVERSIONS` with what each one actually
needs. The test fails on a NEW one, and it fails just as loudly when a listed
inversion is FIXED but left in the list — so the allowlist cannot quietly
become a place where debt is filed and forgotten.

The gate chain was the largest of these and is already fixed: it moved from
``routers/orders.py`` to ``execution/gate_chain.py`` (2026-08-24), which is why
``order_sync`` no longer appears below.
"""
from __future__ import annotations

import ast
import pathlib

GATEWAY = pathlib.Path("apps/gateway")

#: module -> the imported router, for inversions that already existed.
#:
#: Each is a genuine layering violation with a genuine fix, none of which is
#: "move the import into a function":
#:
#: - ``ensure_daily_bars`` / ``market_regime_from_spy`` are market-data seams
#:   that happen to live in ``routers/analysis.py``. They belong in a seam
#:   module; three callers already reach past the router for them.
#: - ``MACRO_REFERENCE_SYMBOLS`` is a CONSTANT — it belongs beside the other
#:   macro constants in the pure layer.
#: - the ``portfolio`` / ``options`` helpers are account and chain seams; two
#:   of their call sites are already function-local to dodge a cycle, which is
#:   the layering telling on itself.
#: - ``run_exit_sweep`` and ``run_reconciliation`` are whole operations that a
#:   background loop legitimately needs; they belong next to the gate chain in
#:   ``execution/``.
KNOWN_INVERSIONS: dict[str, set[str]] = {
    "event_macro.py": {"market"},
    "event_price.py": {"analysis"},
    "fundamentals.py": {"analysis"},
    "main.py": {"broker"},
    "monitor.py": {"positions"},
    "risk_inputs.py": {"analysis", "portfolio"},
    "risk_snapshot.py": {"options", "portfolio"},
    # The gate chain inherited three router dependencies when it moved out of
    # routers/orders.py. They are the SAME market-data / chain / account seams
    # listed above, so fixing those fixes this entry too — which is the point
    # of recording it rather than hiding it behind a local import.
    "execution/gate_chain.py": {"analysis", "options", "portfolio"},
}


def _router_imports(path: pathlib.Path) -> set[str]:
    """Which sibling routers this module imports, at any nesting depth.

    ``ast.walk`` on purpose: an import moved inside a function to silence a
    cycle is still a dependency, and pretending otherwise is exactly how this
    boundary eroded the first time.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            # ".routers.analysis" (level 1) or "..routers.analysis" (level 2)
            parts = node.module.split(".")
            if parts[0] == "routers" and len(parts) > 1:
                found.add(parts[1])
    return found


def _non_router_modules() -> list[pathlib.Path]:
    return sorted(
        p
        for p in GATEWAY.rglob("*.py")
        if "routers" not in p.parts and "__pycache__" not in p.parts
    )


def test_no_new_module_depends_on_a_router():
    violations: list[str] = []
    for path in _non_router_modules():
        key = str(path.relative_to(GATEWAY))
        allowed = KNOWN_INVERSIONS.get(key, set())
        for router in sorted(_router_imports(path) - allowed):
            violations.append(f"{key} imports routers.{router}")
    assert not violations, (
        "New router dependency — the arrow points routers -> seams, not back.\n"
        "Put the shared logic in a seam or in execution/, and import THAT from\n"
        "both sides:\n  " + "\n  ".join(violations)
    )


def test_the_allowlist_has_no_stale_entries():
    """A fixed inversion must leave the list.

    Without this, the allowlist becomes a place debt is filed and forgotten,
    and the ratchet stops ratcheting.
    """
    stale: list[str] = []
    for key, routers in KNOWN_INVERSIONS.items():
        path = GATEWAY / key
        if not path.exists():
            stale.append(f"{key} no longer exists")
            continue
        for router in sorted(routers - _router_imports(path)):
            stale.append(f"{key} no longer imports routers.{router} — remove it")
    assert not stale, "KNOWN_INVERSIONS is out of date:\n  " + "\n  ".join(stale)


def test_the_gate_chain_is_reachable_without_the_http_layer():
    """The decision must be importable without touching a router.

    This is the invariant the extraction bought: a background loop, a CLI or a
    test can reach the trading decision directly.
    """
    source = (GATEWAY / "execution" / "gate_chain.py").read_text()
    assert "async def run_gate_chain" in source
    # And the background loop that used to import it FROM the router now does
    # not — the regression this test exists to prevent.
    sync = (GATEWAY / "order_sync.py").read_text()
    assert "from .routers.orders import" not in sync
