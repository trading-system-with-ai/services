"""LLM market-event selection (market-selection-v1).

The model here answers a genuinely semantic question — WHICH venue event a
catalyst is best read against — that the deterministic matcher was bad at:
it scored contracts individually, which is how brackets of two different
distributions ended up interleaved in one panel.

The boundary is the whole design, so it is what these tests pin:

  - the model may only NARROW a pool the provider supplied;
  - it may never introduce a market, a price or an id;
  - it may never take PART of a distribution;
  - every failure mode degrades to "the deterministic matcher decides".
"""
from __future__ import annotations

import pytest

from libs.llm.market_selection import (
    MARKET_SELECTION_VERSION,
    MarketEventOption,
    build_selection_prompt,
    parse_selection,
)

OPTIONS = [
    MarketEventOption(
        ref="e0",
        title="Will US GDP growth in Q3 2026 be less than 0.5%?",
        n_markets=7,
        end_date="2026-10-29T00:00:00+00:00",
        sample_questions=("Will US GDP growth in Q3 2026 be less than 0.5%?",),
    ),
    MarketEventOption(
        ref="e1",
        title="Will US GDP growth in 2026 be less than 0.5%?",
        n_markets=6,
        end_date="2027-01-29T00:00:00+00:00",
        sample_questions=("Will US GDP growth in 2026 be less than 0.5%?",),
    ),
]
REFS = [o.ref for o in OPTIONS]


def test_a_ref_the_caller_never_offered_is_dropped():
    """THE STRUCTURAL GUARANTEE. A model cannot introduce a market: an
    unrecognised ref resolves to nothing, so the worst a hallucination can do
    is make the pool smaller."""
    result = parse_selection(
        {"selections": [{"ref": "e99", "relation": "DIRECT", "reason": "x"}]},
        allowed_refs=REFS,
    )
    assert result.selections == ()


def test_an_unknown_relation_is_dropped_rather_than_coerced():
    """DIRECT/DERIVED/CONTEXT is a fixed vocabulary. Coercing an invented
    relation to the nearest one would let the model widen the taxonomy."""
    result = parse_selection(
        {"selections": [{"ref": "e0", "relation": "PROBABLY", "reason": "x"}]},
        allowed_refs=REFS,
    )
    assert result.selections == ()


def test_a_duplicate_ref_is_counted_once():
    result = parse_selection(
        {
            "selections": [
                {"ref": "e0", "relation": "DIRECT", "reason": "a"},
                {"ref": "e0", "relation": "CONTEXT", "reason": "b"},
            ]
        },
        allowed_refs=REFS,
    )
    assert [s.ref for s in result.selections] == ["e0"]


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        '{"selections": "nope"}',
        {"selections": [None, 3, "x"]},
        [],
        None,
        42,
    ],
)
def test_every_malformed_reply_degrades_to_selecting_nothing(raw):
    """A bad reply must cost only itself: the deterministic matcher behind
    this step is a complete answer, so nothing here may raise."""
    assert parse_selection(raw, allowed_refs=REFS).selections == ()


def test_selecting_nothing_is_a_valid_answer():
    """"None of these fit" is common and correct — preferred over a strained
    match, and it routes the caller back to the pure matcher."""
    result = parse_selection(
        {"selections": [], "note": "no distribution covers this release"},
        allowed_refs=REFS,
    )
    assert result.selections == ()
    assert "no distribution" in result.note


def test_the_selection_is_capped():
    """More than a couple of distributions on one catalyst is not a richer
    picture, it is an unreadable one."""
    result = parse_selection(
        {
            "selections": [
                {"ref": "e0", "relation": "DIRECT", "reason": "a"},
                {"ref": "e1", "relation": "CONTEXT", "reason": "b"},
            ]
        },
        allowed_refs=REFS,
        max_selected=1,
    )
    assert len(result.selections) == 1


def test_the_venue_listing_is_fenced_as_untrusted():
    """Event titles are third-party text. They ride inside an explicit fence
    so a title reading "ignore previous instructions" is visibly quoted
    material rather than an instruction."""
    system, user = build_selection_prompt(
        event_type="GDP",
        event_title="GDP (Advance Estimate), 3rd Quarter 2026",
        scheduled_at="2026-10-29T12:30:00+00:00",
        options=OPTIONS,
    )
    assert "<untrusted_prediction_markets>" in user
    assert "</untrusted_prediction_markets>" in user
    # The catalyst itself is platform DATA and rides outside the fence.
    assert user.index("GDP (Advance Estimate)") < user.index(
        "<untrusted_prediction_markets>"
    )


def test_the_prompt_forbids_naming_a_single_contract():
    """Selection is at EVENT granularity: there is no control that would let
    the model keep four brackets of seven, which is the failure this whole
    layer exists to prevent."""
    system, _ = build_selection_prompt(
        event_type="GDP",
        event_title="GDP",
        scheduled_at="2026-10-29T12:30:00+00:00",
        options=OPTIONS,
    )
    assert "whole events" in system.lower()
    assert "never name a single contract" in system.lower()


def test_the_prompt_says_selecting_nothing_is_correct():
    system, _ = build_selection_prompt(
        event_type="GDP",
        event_title="GDP",
        scheduled_at="2026-10-29T12:30:00+00:00",
        options=OPTIONS,
    )
    assert "correct and common answer" in system.lower()


def test_the_result_carries_its_contract_version():
    """A stored selection made under different rules is not comparable."""
    result = parse_selection({"selections": []}, allowed_refs=REFS)
    assert result.version == MARKET_SELECTION_VERSION
