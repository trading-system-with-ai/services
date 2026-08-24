"""Shared machinery for the §96 adversarial look-ahead suite.

Kept beside ``test_event_lookahead.py`` rather than inside it because the
recursive absence scan is the one assertion in this repository that is worth
unit-testing on its OWN — a sentinel scanner with a blind spot turns every
look-ahead test that uses it into a test that cannot fail, which is a worse
outcome than having no suite at all. ``test_event_lookahead.py`` therefore
opens by attacking this module (the ``test_the_scanner_*`` block) before it
attacks any endpoint.

The underscore prefix keeps pytest from collecting this file as a test module
while still letting the suite import it as ``tests._lookahead_util``.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator

from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# The recursive sentinel scan
# ---------------------------------------------------------------------------


def walk_strings(node: Any) -> Iterator[str]:
    """Every string-shaped value anywhere in a JSON payload, keys INCLUDED.

    Keys are yielded as well as values because several payloads in this
    platform key a mapping BY the artifact's identity — ``metrics_by_period``
    is keyed on the reference period, ``tenors`` on the tenor spelling — so a
    leaked future observation can show up as a KEY whose value is a plain
    number. A scan that walked values only would report those payloads clean.

    Numbers are rendered with ``repr`` so a sentinel planted as a distinctive
    float (``123.456789``) is findable in the same pass as one planted as a
    string. ``None``/bools are skipped: they carry no identity.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                yield key
            yield from walk_strings(value)
    elif isinstance(node, (list, tuple, set)):
        for item in node:
            yield from walk_strings(item)
    elif isinstance(node, str):
        yield node
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        yield repr(node)
    elif isinstance(node, (datetime, date)):
        yield node.isoformat()


def find_sentinel(payload: Any, sentinel: str) -> list[str]:
    """Every string in ``payload`` containing ``sentinel``, as evidence.

    Returns the offending strings rather than a bool so a failure message can
    show WHERE the future artifact surfaced — "absent" is a claim about the
    whole document and a bare ``False`` gives a reader nothing to debug.
    """
    needle = str(sentinel)
    return [text for text in walk_strings(payload) if needle in text]


def assert_absent(payload: Any, sentinel: str, *, where: str = "payload") -> None:
    """THE §96 assertion: no trace of ``sentinel`` anywhere in ``payload``.

    ``where`` names the endpoint so a failing run says which surface leaked
    without the reader cross-referencing line numbers.
    """
    hits = find_sentinel(payload, sentinel)
    assert not hits, (
        f"LOOK-AHEAD LEAK in {where}: the future sentinel {sentinel!r} "
        f"surfaced in {len(hits)} place(s): {hits[:5]}"
    )


def assert_present(payload: Any, sentinel: str, *, where: str = "payload") -> None:
    """The other half of every PAIR: the PAST twin must actually be visible.

    A gate that returned nothing at all would satisfy every ``assert_absent``
    in this suite while destroying the endpoint. Every look-ahead test here is
    therefore written as a pair — future absent AND past present — and this is
    the half that proves the endpoint still answers.
    """
    hits = find_sentinel(payload, sentinel)
    assert hits, (
        f"the PAST twin {sentinel!r} is missing from {where}: the gate is not "
        "point-in-time, it is simply returning nothing"
    )


# ---------------------------------------------------------------------------
# Instants
# ---------------------------------------------------------------------------


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def et(y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    """An ET wall-clock instant as the UTC equivalent the DB stores."""
    return datetime(y, m, d, hour, minute, tzinfo=EASTERN).astimezone(timezone.utc)


def iso(when: datetime) -> str:
    """An instant in the query-string form the UI sends."""
    return when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def weekdays(start: date, count: int) -> list[date]:
    """Consecutive WEEKDAYS from ``start`` — the bar dates ARE trading days,
    exactly as the pure library assumes (it never consults a calendar)."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days
