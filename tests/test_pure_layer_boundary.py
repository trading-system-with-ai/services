"""THE PURE LAYER REACHES NOTHING — enforced across the whole package.

``libs/trading_core/`` performs no I/O. It takes values and returns values: no
HTTP client, no database session, no filesystem, no clock it was not handed as
an argument. Everything else in the architecture leans on that:

- ~4500 tests run in about two minutes because the layer holding the platform's
  arithmetic never waits on a socket;
- a point-in-time replay is reproducible because no function can quietly
  observe "now" or fetch a value that has since been revised;
- a seam is the ONLY place stored rows meet computation, which is what makes
  "reads never fetch" checkable rather than aspirational.

WHY THIS FILE REPLACED FIVE OTHERS. The rule used to be enforced by five
copies of ``test_module_imports_no_io_layer``, each with a single file path
hardcoded in it — macro.py, replay.py, prediction_intel.py, web_research.py,
fundamentals.py. Seventy-one modules live under ``libs/trading_core``; the
other sixty-six were protected by nothing but the habit of copying a test.
An audit found zero violations, so the discipline held — but a foundational
invariant should not depend on everyone remembering to duplicate a test file.

This walks the package instead. A new pure module is covered the moment it is
written, and nobody has to know this file exists.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PURE_ROOT = pathlib.Path("libs/trading_core")

#: Import prefixes a pure module may never reach for.
#:
#: ``apps`` is the gateway (database sessions, HTTP). ``libs.market_data`` /
#: ``libs.broker`` / ``libs.llm`` / ``libs.prediction_markets`` /
#: ``libs.web_search`` / ``libs.event_calendar`` are provider adapters, which
#: exist precisely to hold the network so this layer does not.
FORBIDDEN_PREFIXES = (
    "apps",
    "libs.market_data",
    "libs.broker",
    "libs.llm",
    "libs.prediction_markets",
    "libs.web_search",
    "libs.event_calendar",
)

#: Standard-library and third-party modules that ARE I/O. Importing one here
#: would be the same violation wearing stdlib clothes.
#:
#: Matched on the FULL dotted name, not the root package, because the split
#: matters: ``urllib.parse`` is string manipulation (URL normalisation, which
#: the web-research canonicaliser legitimately needs) while
#: ``urllib.request`` opens sockets. Banning the root would force a pure
#: module to hand-roll query-string parsing, which trades a real bug risk for
#: an imaginary one.
FORBIDDEN_IO_MODULES = (
    "httpx",
    "requests",
    "aiohttp",
    "socket",
    "sqlite3",
    "sqlalchemy",
    "subprocess",
    "urllib.request",
    "urllib.error",
    "http.client",
    "ftplib",
    "smtplib",
    "asyncio",
)

#: Third-party numerics are banned separately: the platform's arithmetic must
#: be readable and reproducible without a BLAS version entering the argument.
FORBIDDEN_NUMERICS = ("numpy", "pandas", "scipy")


def _pure_modules() -> list[pathlib.Path]:
    return sorted(
        p
        for p in PURE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _imported_names(path: pathlib.Path) -> list[str]:
    """Every module name this file imports, at any nesting depth.

    ``ast.walk`` rather than a top-level scan on purpose: an import moved
    inside a function to dodge a lint rule is still an import, and function-
    local imports are exactly how a boundary erodes quietly.
    """
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — stays inside the pure package
                continue
            if node.module:
                names.append(node.module)
    return names


def test_the_pure_package_is_not_empty():
    """A path typo here would make every test below vacuously pass."""
    modules = _pure_modules()
    assert len(modules) > 40, f"only found {len(modules)} pure modules — wrong root?"


@pytest.mark.parametrize(
    "path", _pure_modules(), ids=lambda p: str(p.relative_to(PURE_ROOT))
)
def test_module_reaches_no_io_layer(path: pathlib.Path):
    for name in _imported_names(path):
        assert not any(
            name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIXES
        ), f"{path} imports {name} — the pure layer may not reach a provider or the gateway"
        assert not any(
            name == m or name.startswith(m + ".") for m in FORBIDDEN_IO_MODULES
        ), f"{path} imports {name} — that is I/O, which belongs in a seam"


@pytest.mark.parametrize(
    "path", _pure_modules(), ids=lambda p: str(p.relative_to(PURE_ROOT))
)
def test_module_uses_no_third_party_numerics(path: pathlib.Path):
    for name in _imported_names(path):
        assert name.split(".")[0] not in FORBIDDEN_NUMERICS, (
            f"{path} imports {name} — the platform's arithmetic stays in stdlib "
            "so a result can be read and reproduced without a vendored BLAS"
        )
