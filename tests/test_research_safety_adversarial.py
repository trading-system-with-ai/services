"""Structural isolation of the research capabilities from trade execution
(Catalyst research upgrade — ABSOLUTE SAFETY BOUNDARY; plan §10-§11).

Web search and prediction markets are RESEARCH ONLY: nothing they produce may
become a deterministic trade signal, reach §8 instrument selection, Tier-0
risk sizing, the §10 gate chain, trading-pool authorization or broker
submission. The other adversarial files prove boundaries BEHAVIOURALLY
(row-diffing, call-site whitelists); this file proves them STRUCTURALLY, at
the import graph — attack style, not happy path:

- Property A: no research module imports the execution stack.
- Property B: no execution module imports the research stack.
- Property C: the prediction-market package has NO trading surface — no
  function whose name says place/submit/sign/wallet/approve exists to guard.

The AST walk reads ``import``/``from ... import`` nodes, not text: a
docstring that merely MENTIONS an execution module is prose, not a
dependency (the test_risk_adversarial.py lesson, applied to imports).

This file grows with the upgrade (injection tests, DB-row diffs land with
the orchestrator in later loops); the import-graph properties are pinned
FIRST so every subsequent loop is born inside the boundary.
"""
from __future__ import annotations

import ast
import pathlib

SERVICES = pathlib.Path(__file__).resolve().parents[1]

#: Every module of the research stack, as it exists today. Directories are
#: walked recursively so a module added in a later loop is covered the moment
#: it exists — the guard must not depend on remembering to register files.
RESEARCH_ROOTS = (
    SERVICES / "libs" / "web_search",
    SERVICES / "libs" / "prediction_markets",
)

#: Pure research modules that will land in later loops (registered up front;
#: missing files are skipped, present files are checked).
RESEARCH_MODULES = (
    SERVICES / "libs" / "trading_core" / "events" / "web_research.py",
    SERVICES / "libs" / "trading_core" / "events" / "prediction_intel.py",
    SERVICES / "apps" / "gateway" / "event_research.py",
    SERVICES / "apps" / "gateway" / "event_prediction_markets.py",
)

#: Import prefixes that ARE the execution stack: instrument selection (§8),
#: risk sizing, directional/regime signals, broker submission, and the order/
#: trading-pool routers that host the gate chain and approval paths.
EXECUTION_PREFIXES = (
    "libs.broker",
    "libs.trading_core.risk",
    "libs.trading_core.strategies",
    "libs.trading_core.signals",
    "libs.trading_core.contracts",
    "apps.gateway.routers.orders",
    "apps.gateway.routers.trading_pool",
    "apps.gateway.routers.income",
    "apps.gateway.routers.plans",
    "apps.gateway.routers.watchlist",
    # The execution SEAMS, not just the routers: broker submission, order
    # reconciliation and the risk input/snapshot layer are where an import
    # would actually buy influence over a trade.
    "apps.gateway.broker_exec",
    "apps.gateway.order_sync",
    "apps.gateway.risk_inputs",
    "apps.gateway.risk_snapshot",
    "apps.gateway.risk_validation",
)

#: Execution-side modules that must never import the research stack back.
EXECUTION_FILES = (
    SERVICES / "apps" / "gateway" / "routers" / "orders.py",
    SERVICES / "apps" / "gateway" / "routers" / "trading_pool.py",
    SERVICES / "apps" / "gateway" / "routers" / "income.py",
    SERVICES / "apps" / "gateway" / "routers" / "plans.py",
    SERVICES / "apps" / "gateway" / "routers" / "watchlist.py",
    # The seams that actually place, reconcile and size trades. Omitting
    # these left the reverse direction (execution importing research)
    # unguarded exactly where it would matter most.
    SERVICES / "apps" / "gateway" / "broker_exec.py",
    SERVICES / "apps" / "gateway" / "order_sync.py",
    SERVICES / "apps" / "gateway" / "risk_inputs.py",
    SERVICES / "apps" / "gateway" / "risk_snapshot.py",
    SERVICES / "libs" / "trading_core" / "risk" / "engine.py",
    SERVICES / "libs" / "trading_core" / "strategies" / "instrument.py",
    SERVICES / "libs" / "trading_core" / "contracts" / "selector.py",
)

RESEARCH_PREFIXES = (
    "libs.web_search",
    "libs.prediction_markets",
    "libs.trading_core.events.web_research",
    "libs.trading_core.events.prediction_intel",
    "apps.gateway.event_research",
    "apps.gateway.event_prediction_markets",
)


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Every module name a file imports — AST nodes, never text search."""
    tree = ast.parse(path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # Relative imports (level > 0) stay inside their own package by
            # construction; absolute ones carry the full dotted path.
            if node.level == 0:
                modules.append(node.module)
    return modules


def _research_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root in RESEARCH_ROOTS:
        assert root.is_dir(), f"research package missing: {root}"
        files.extend(sorted(root.rglob("*.py")))
    files.extend(p for p in RESEARCH_MODULES if p.exists())
    return files


def test_property_a_research_modules_never_import_the_execution_stack():
    checked = 0
    for path in _research_files():
        for module in _imported_modules(path):
            for prefix in EXECUTION_PREFIXES:
                assert not (
                    module == prefix or module.startswith(prefix + ".")
                ), f"{path.relative_to(SERVICES)} imports execution module {module}"
        checked += 1
    assert checked >= 6, "the research packages should have been walked"


def test_property_b_execution_modules_never_import_the_research_stack():
    for path in EXECUTION_FILES:
        assert path.exists(), f"execution module moved? {path}"
        for module in _imported_modules(path):
            for prefix in RESEARCH_PREFIXES:
                assert not (
                    module == prefix or module.startswith(prefix + ".")
                ), f"{path.relative_to(SERVICES)} imports research module {module}"


def test_property_c_prediction_markets_package_has_no_trading_surface():
    """READ ONLY by construction: no function/method whose NAME says it
    trades exists anywhere in the package — there is nothing to guard,
    which is stronger than a guard. Same word-ban AST technique as
    test_events_news_intel.py's sentiment ban."""
    banned = (
        "place", "submit", "sign", "wallet", "approve", "execute",
        "buy", "sell", "trade", "cancel", "deposit", "withdraw",
    )
    root = SERVICES / "libs" / "prediction_markets"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name.lower()
                for word in banned:
                    assert word not in name, (
                        f"{path.relative_to(SERVICES)} defines {node.name!r} — "
                        "a trading-shaped name in a READ-ONLY package"
                    )
