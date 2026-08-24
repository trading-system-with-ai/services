"""Fed intelligence — statement diff, policy dimensions, two reaction windows
(event spec §9, §42-§45; audit §11.9; Phase H unit U2).

Pure stdlib, deterministic, **no I/O**. Like the rest of
``libs/trading_core/events/`` this module may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4): the gateway seam
(``apps/gateway/event_fed.py``) reads stored documents and bars out of the DB
and hands them here as plain values.

Four rules from the spec shape every function below, and they are the reason
this module looks the way it does:

1. **The source document is authoritative** (§44). The diff is computed by
   :mod:`difflib` over sentences of the VERBATIM statement text — never over
   a summary, never over an LLM's paraphrase. What comes out is a list of
   aligned sentence pairs with a status and a similarity ratio; the LLM's
   only job downstream is to explain a pair a human can already read.

2. **There is NO single hawkish/dovish score** (§43). The eight
   :data:`DIMENSIONS` are reported SEPARATELY, each with its own status and
   its own sentences. Collapsing "raised the inflation language, softened the
   growth language, three dissents" into one number destroys the only part
   that is actually actionable, so this module has no code path that produces
   one — no key named ``score``, ``hawkish`` or ``dovish`` appears in any
   payload it emits, and a test asserts that mechanically.

3. **The two reaction windows are separate measurements** (§45). The
   statement drops at 14:00 ET and the press conference starts at 14:30 ET;
   they are two different pieces of information and the market frequently
   reverses between them. :func:`fomc_reaction_windows` therefore reports
   ``statement`` (14:00-14:30) and ``press_conference`` (14:30-15:30) as
   two independent blocks and never sums or nets them. When no minute bars
   are stored the fallback is DAILY and says so via ``basis``, because a
   daily bar cannot separate the two windows at all.

4. **Fed funds futures pricing is UNAVAILABLE** (§43). No free primary
   source ships it, so ``market_pricing.status`` is the constant
   :data:`MARKET_PRICING_UNAVAILABLE` and the 2Y yield change travels beside
   it explicitly LABELLED a proxy — never as "the market's expectation".
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .taxonomy import UTC, require_utc

__all__ = [
    "CHANGED_SIMILARITY_FLOOR",
    "DIMENSIONS",
    "DIMENSION_KEYWORDS",
    "DIMENSION_COMMITTEE_DISPERSION",
    "DIMENSION_BALANCE_SHEET",
    "DIMENSION_EMPLOYMENT",
    "DIMENSION_FORWARD_GUIDANCE",
    "DIMENSION_GROWTH",
    "DIMENSION_INFLATION",
    "DIMENSION_POLICY_RATE",
    "DIMENSION_RISK_BALANCE",
    "DIRECTION_CUT",
    "DIRECTION_HIKE",
    "DIRECTION_HOLD",
    "FED_INTEL_MODEL_VERSION",
    "MARKET_PRICING_UNAVAILABLE",
    "NO_SINGLE_SCORE_NOTE",
    "PRESS_CONFERENCE_WINDOW_ET",
    "REACTION_BASIS_DAILY",
    "REACTION_BASIS_MINUTE",
    "SOURCE_AUTHORITATIVE_NOTE",
    "STATEMENT_WINDOW_ET",
    "STATUS_ADDED",
    "STATUS_CHANGED",
    "STATUS_NA",
    "STATUS_REMOVED",
    "STATUS_UNCHANGED",
    "DiffItem",
    "FedPacket",
    "Sentence",
    "StatementDiff",
    "build_fed_packet",
    "dimension_report",
    "dimensions_for",
    "fomc_reaction_daily",
    "fomc_reaction_windows",
    "normalize_sentence",
    "parse_target_range",
    "policy_rate_change",
    "split_sentences",
    "statement_diff",
    "vote_dispersion",
]

FED_INTEL_MODEL_VERSION = "fed-intel-v1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_ADDED = "ADDED"
STATUS_REMOVED = "REMOVED"
STATUS_CHANGED = "CHANGED"
STATUS_UNCHANGED = "UNCHANGED"
STATUS_NA = "NA"

#: A replaced sentence pair counts as CHANGED (rather than an unrelated
#: REMOVED + ADDED pair) when the normalized texts are at least this similar.
#: 0.6 is difflib's own "close match" convention and it is what keeps a
#: one-word edit ("moderate" → "solid") aligned as a single row while a
#: wholly new paragraph is reported as an addition.
CHANGED_SIMILARITY_FLOOR = 0.6

DIMENSION_POLICY_RATE = "POLICY_RATE"
DIMENSION_INFLATION = "INFLATION"
DIMENSION_EMPLOYMENT = "EMPLOYMENT"
DIMENSION_GROWTH = "GROWTH"
DIMENSION_BALANCE_SHEET = "BALANCE_SHEET"
DIMENSION_FORWARD_GUIDANCE = "FORWARD_GUIDANCE"
DIMENSION_RISK_BALANCE = "RISK_BALANCE"
DIMENSION_COMMITTEE_DISPERSION = "COMMITTEE_DISPERSION"

#: The §43 dimensions, in report order. They are reported SEPARATELY and are
#: never combined — see the module docstring.
DIMENSIONS: tuple[str, ...] = (
    DIMENSION_POLICY_RATE,
    DIMENSION_INFLATION,
    DIMENSION_EMPLOYMENT,
    DIMENSION_GROWTH,
    DIMENSION_BALANCE_SHEET,
    DIMENSION_FORWARD_GUIDANCE,
    DIMENSION_RISK_BALANCE,
    DIMENSION_COMMITTEE_DISPERSION,
)

#: Keyword taggers, matched case-insensitively against the NORMALIZED
#: sentence. Deliberately literal phrases from the Fed's own boilerplate
#: rather than a model: a keyword list is auditable and a reader can see
#: exactly why a sentence landed in a row. COMMITTEE_DISPERSION is derived
#: from the vote, not from prose, so its keyword set covers only the vote
#: sentence the statement itself carries.
DIMENSION_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    DIMENSION_POLICY_RATE: (
        "target range",
        "federal funds rate",
        "policy rate",
        "percentage point",
    ),
    DIMENSION_INFLATION: (
        "inflation",
        "prices",
        "price increases",
        "price stability",
    ),
    DIMENSION_EMPLOYMENT: (
        "unemployment",
        "job gains",
        "labor market",
        "employment",
        "workforce",
    ),
    DIMENSION_GROWTH: (
        "economic activity",
        "growth",
        "spending",
        "investment",
        "productivity",
    ),
    DIMENSION_BALANCE_SHEET: (
        "holdings",
        "balance sheet",
        "reserves",
        "treasury securities",
        "agency",
    ),
    DIMENSION_FORWARD_GUIDANCE: (
        "extent and timing",
        "additional adjustments",
        "future adjustments",
        "will carefully assess",
        "in determining",
        "incoming data",
    ),
    DIMENSION_RISK_BALANCE: (
        "risks",
        "uncertainty",
        "balance of risks",
        "attentive to",
    ),
    DIMENSION_COMMITTEE_DISPERSION: (
        "voting for",
        "voting against",
        "vote:",
        "dissent",
        "who preferred",
    ),
}

DIRECTION_CUT = "CUT"
DIRECTION_HIKE = "HIKE"
DIRECTION_HOLD = "HOLD"

REACTION_BASIS_MINUTE = "1m_bars"
REACTION_BASIS_DAILY = "daily"

#: ET wall-clock spans of the two §45 windows, as ``(start_hm, end_hm)``.
#: They are documented here so a payload consumer can label the columns
#: without re-deriving the Fed's schedule.
STATEMENT_WINDOW_ET = ((14, 0), (14, 30))
PRESS_CONFERENCE_WINDOW_ET = ((14, 30), (15, 30))

MARKET_PRICING_UNAVAILABLE = "UNAVAILABLE"

#: Disclaimer strings that travel INSIDE the payload (§44, §43). The UI shows
#: them verbatim; keeping them here means the API, the UI and the LLM prompt
#: cannot drift into three different claims about the same limitation.
SOURCE_AUTHORITATIVE_NOTE = (
    "The FOMC statement is the authoritative source; the diff below is "
    "computed verbatim from the released text."
)
NO_SINGLE_SCORE_NOTE = (
    "No single hawkish/dovish score by design — dimensions are reported "
    "separately."
)
MARKET_PRICING_NOTE = (
    "Fed funds futures pricing is unavailable from the free primary sources; "
    "the 2-year Treasury yield change is shown as a labelled proxy, not as "
    "the market-implied policy path."
)

_PROXY_KEY_2Y = "2y_yield_change_bp"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sentence:
    """One sentence of a statement, with its position kept.

    ``idx`` is the running index across the whole document and ``para_idx``
    the paragraph it came from; both survive into the diff so the UI can show
    the statement in ITS order rather than in difflib's.
    """

    idx: int
    para_idx: int
    text: str
    normalized: str

    def to_dict(self) -> dict[str, object]:
        return {
            "idx": self.idx,
            "para_idx": self.para_idx,
            "text": self.text,
            "dimensions": list(dimensions_for(self.normalized)),
        }


@dataclass(frozen=True)
class DiffItem:
    """One aligned sentence pair.

    ``similarity`` is difflib's ratio over the normalized texts and is
    ``1.0`` for UNCHANGED, ``None`` for a pure ADDED/REMOVED. ``dimensions``
    is the union of the tags on whichever side(s) exist — a sentence that
    changes from an employment claim into a growth claim is tagged with both,
    because both rows should surface it.
    """

    status: str
    previous_text: str | None = None
    current_text: str | None = None
    previous_idx: int | None = None
    current_idx: int | None = None
    para_idx: int | None = None
    dimensions: tuple[str, ...] = ()
    similarity: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "previous_text": self.previous_text,
            "current_text": self.current_text,
            "previous_idx": self.previous_idx,
            "current_idx": self.current_idx,
            "para_idx": self.para_idx,
            "dimensions": list(self.dimensions),
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class StatementDiff:
    """The §44 sentence-level diff of two statements."""

    items: tuple[DiffItem, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    model_version: str = FED_INTEL_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "counts", dict(self.counts))

    def to_dict(self) -> dict[str, object]:
        return {
            "items": [item.to_dict() for item in self.items],
            "counts": dict(self.counts),
            "model_version": self.model_version,
            "note": SOURCE_AUTHORITATIVE_NOTE,
        }


@dataclass(frozen=True)
class FedPacket:
    """The §42-§45 Fed packet — every section always present."""

    as_of: datetime
    event: Mapping[str, object] = field(default_factory=dict)
    previous_statement: Mapping[str, object] = field(default_factory=dict)
    statement_diff: Mapping[str, object] = field(default_factory=dict)
    dimensions: Mapping[str, object] = field(default_factory=dict)
    previous_minutes: Mapping[str, object] = field(default_factory=dict)
    subsequent_speeches: tuple[Mapping[str, object], ...] = ()
    data: Mapping[str, object] = field(default_factory=dict)
    market_pricing: Mapping[str, object] = field(default_factory=dict)
    previous_reaction: Mapping[str, object] = field(default_factory=dict)
    coverage: Mapping[str, object] = field(default_factory=dict)
    tiers: Mapping[str, str] = field(default_factory=dict)
    disclaimers: tuple[str, ...] = ()
    model_version: str = FED_INTEL_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", require_utc(self.as_of, name="as_of"))
        object.__setattr__(self, "subsequent_speeches", tuple(self.subsequent_speeches))
        object.__setattr__(self, "disclaimers", tuple(self.disclaimers))

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of": _iso(self.as_of),
            "event": dict(self.event),
            "previous_statement": dict(self.previous_statement),
            "statement_diff": dict(self.statement_diff),
            "dimensions": dict(self.dimensions),
            "previous_minutes": dict(self.previous_minutes),
            "subsequent_speeches": [dict(s) for s in self.subsequent_speeches],
            "data": dict(self.data),
            "market_pricing": dict(self.market_pricing),
            "previous_reaction": dict(self.previous_reaction),
            "coverage": dict(self.coverage),
            "tiers": dict(self.tiers),
            "disclaimers": list(self.disclaimers),
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

#: Abbreviations whose trailing period must NOT end a sentence. The Fed's
#: dissent paragraph is full of them ("Beth M. Hammack", "Lorie K. Logan"),
#: and splitting there would shatter the single most-read sentence of the
#: statement into four fragments that then diff against nothing.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "gov",
        "st",
        "jr",
        "sr",
        "inc",
        "no",
        "vs",
        "u.s",
        "e.g",
        "i.e",
    }
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s/\-]")

#: "3-1/2 to 3-3/4 percent" — the Fed writes the target range in eighths as
#: mixed fractions, never as a decimal, so the parser reads fractions.
_TARGET_RANGE = re.compile(
    r"target range for the federal funds rate at\s+"
    r"(?P<low>[0-9]+(?:-[0-9]+/[0-9]+)?)\s+to\s+"
    r"(?P<high>[0-9]+(?:-[0-9]+/[0-9]+)?)\s+percent",
    re.IGNORECASE,
)

#: "by a 9 – 3 vote" — the dash is an EN DASH surrounded by spaces on the
#: live page, but hyphens and em dashes show up in older releases.
_VOTE_LINE = re.compile(
    r"by a\s+(?P<for>\d+)\s*[‐-―\-]\s*(?P<against>\d+)\s+vote",
    re.IGNORECASE,
)


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return require_utc(value, name="timestamp").isoformat()
    return value.isoformat()


def normalize_sentence(text: str) -> str:
    """Lowercase, punctuation-light form used for alignment ONLY.

    The normalized form decides whether two sentences are "the same"; the
    VERBATIM text is what is reported (§44). Fractions keep their ``/`` and
    ``-`` so "3-1/2 to 3-3/4" does not normalize into the same string as
    "3-1/4 to 3-1/2" — a rate change must never look like an unchanged
    sentence.
    """
    lowered = _WHITESPACE.sub(" ", (text or "").replace("’", "'")).strip().lower()
    stripped = _PUNCT.sub("", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def _split_paragraph(text: str) -> list[str]:
    """Sentences of one paragraph, abbreviation-aware."""
    cleaned = _WHITESPACE.sub(" ", (text or "").replace("\xa0", " ")).strip()
    if not cleaned:
        return []
    pieces = _SENTENCE_END.split(cleaned)
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if out:
            tail = out[-1].rstrip()
            # Re-join when the previous fragment ended on an abbreviation or a
            # single capital initial ("Beth M." + "Hammack, ...").
            last_word = tail.rsplit(" ", 1)[-1].rstrip(".").lower()
            if last_word in _ABBREVIATIONS or (
                len(last_word) == 1 and last_word.isalpha()
            ):
                out[-1] = f"{tail} {piece}"
                continue
        out.append(piece)
    return out


def split_sentences(paragraphs: Sequence[str]) -> list[Sentence]:
    """Flatten paragraphs into positioned :class:`Sentence` values.

    Empty paragraphs (the Fed's page carries a stray ``&nbsp;`` paragraph)
    still consume a ``para_idx`` so the index matches the source document's
    numbering, but contribute no sentences.
    """
    out: list[Sentence] = []
    idx = 0
    for para_idx, para in enumerate(paragraphs or ()):
        for text in _split_paragraph(para):
            out.append(
                Sentence(
                    idx=idx,
                    para_idx=para_idx,
                    text=text,
                    normalized=normalize_sentence(text),
                )
            )
            idx += 1
    return out


def dimensions_for(text: str) -> tuple[str, ...]:
    """Dimension tags for one sentence, in :data:`DIMENSIONS` order."""
    hay = normalize_sentence(text)
    if not hay:
        return ()
    tags: list[str] = []
    for dimension in DIMENSIONS:
        for keyword in DIMENSION_KEYWORDS[dimension]:
            if normalize_sentence(keyword) in hay:
                tags.append(dimension)
                break
    return tuple(tags)


# ---------------------------------------------------------------------------
# §44 — the statement diff
# ---------------------------------------------------------------------------


def statement_diff(
    previous_paragraphs: Sequence[str],
    current_paragraphs: Sequence[str],
    *,
    changed_floor: float = CHANGED_SIMILARITY_FLOOR,
) -> StatementDiff:
    """Sentence-level diff of two statements (§44).

    Alignment is :class:`difflib.SequenceMatcher` over the NORMALIZED
    sentences; the reported text is always verbatim. A ``replace`` opcode is
    resolved pairwise: within the replaced block each previous sentence is
    matched to its best-scoring current sentence, and the pair becomes
    CHANGED when the ratio reaches ``changed_floor`` and a plain
    REMOVED + ADDED pair otherwise. Leftovers on either side become REMOVED /
    ADDED rows.

    Output order follows the CURRENT statement (with removed sentences kept
    in place at the point they disappeared), so the list reads top-to-bottom
    like the document it describes. The function is deterministic: identical
    inputs give byte-identical output.
    """
    prev = split_sentences(previous_paragraphs)
    cur = split_sentences(current_paragraphs)
    matcher = difflib.SequenceMatcher(
        None, [s.normalized for s in prev], [s.normalized for s in cur], autojunk=False
    )

    items: list[DiffItem] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                p = prev[i1 + offset]
                c = cur[j1 + offset]
                items.append(
                    DiffItem(
                        status=STATUS_UNCHANGED,
                        previous_text=p.text,
                        current_text=c.text,
                        previous_idx=p.idx,
                        current_idx=c.idx,
                        para_idx=c.para_idx,
                        dimensions=_merge_tags(p, c),
                        similarity=1.0,
                    )
                )
        elif tag == "delete":
            items.extend(_removed(prev[i] for i in range(i1, i2)))
        elif tag == "insert":
            items.extend(_added(cur[j] for j in range(j1, j2)))
        else:  # "replace"
            items.extend(
                _resolve_replace(prev[i1:i2], cur[j1:j2], changed_floor=changed_floor)
            )

    counts = {status: 0 for status in (
        STATUS_ADDED, STATUS_REMOVED, STATUS_CHANGED, STATUS_UNCHANGED
    )}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    counts["TOTAL"] = len(items)
    return StatementDiff(items=tuple(items), counts=counts)


def _merge_tags(*sentences: Sentence | None) -> tuple[str, ...]:
    seen: set[str] = set()
    for sentence in sentences:
        if sentence is None:
            continue
        seen.update(dimensions_for(sentence.text))
    return tuple(d for d in DIMENSIONS if d in seen)


def _removed(sentences: Iterable[Sentence]) -> list[DiffItem]:
    return [
        DiffItem(
            status=STATUS_REMOVED,
            previous_text=s.text,
            previous_idx=s.idx,
            para_idx=s.para_idx,
            dimensions=_merge_tags(s),
        )
        for s in sentences
    ]


def _added(sentences: Iterable[Sentence]) -> list[DiffItem]:
    return [
        DiffItem(
            status=STATUS_ADDED,
            current_text=s.text,
            current_idx=s.idx,
            para_idx=s.para_idx,
            dimensions=_merge_tags(s),
        )
        for s in sentences
    ]


def _resolve_replace(
    prev_block: Sequence[Sentence],
    cur_block: Sequence[Sentence],
    *,
    changed_floor: float,
) -> list[DiffItem]:
    """Pair up a replaced block into CHANGED rows plus leftovers.

    Greedy over the best ratio available, which is deterministic because ties
    break on the (previous, current) index pair. A quadratic scan is fine
    here: an FOMC statement is a dozen sentences, not a corpus.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(prev_block):
        for ci, c in enumerate(cur_block):
            ratio = difflib.SequenceMatcher(
                None, p.normalized, c.normalized, autojunk=False
            ).ratio()
            if ratio >= changed_floor:
                candidates.append((ratio, pi, ci))
    # Highest ratio first; ties resolve on position so the result never
    # depends on dict/set ordering.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    pair_by_prev: dict[int, tuple[int, float]] = {}
    used_cur: set[int] = set()
    for ratio, pi, ci in candidates:
        if pi in pair_by_prev or ci in used_cur:
            continue
        pair_by_prev[pi] = (ci, ratio)
        used_cur.add(ci)

    pair_by_cur = {ci: (pi, ratio) for pi, (ci, ratio) in pair_by_prev.items()}

    items: list[DiffItem] = []
    # Unmatched previous sentences are emitted first, in place, so a dropped
    # paragraph still appears above the text that replaced it.
    for pi, p in enumerate(prev_block):
        if pi not in pair_by_prev:
            items.extend(_removed([p]))
    for ci, c in enumerate(cur_block):
        if ci in pair_by_cur:
            pi, ratio = pair_by_cur[ci]
            p = prev_block[pi]
            items.append(
                DiffItem(
                    status=STATUS_CHANGED,
                    previous_text=p.text,
                    current_text=c.text,
                    previous_idx=p.idx,
                    current_idx=c.idx,
                    para_idx=c.para_idx,
                    dimensions=_merge_tags(p, c),
                    similarity=round(ratio, 6),
                )
            )
        else:
            items.extend(_added([c]))
    return items


# ---------------------------------------------------------------------------
# §43 — dimensions, target range, vote
# ---------------------------------------------------------------------------


def _fraction_to_float(token: str) -> float | None:
    """``3-3/4`` → ``3.75``; ``4`` → ``4.0``; anything else → ``None``."""
    token = (token or "").strip()
    if not token:
        return None
    whole, _, frac = token.partition("-")
    try:
        value = float(whole)
    except ValueError:
        return None
    if frac:
        num, _, den = frac.partition("/")
        try:
            num_f, den_f = float(num), float(den)
        except ValueError:
            return None
        if den_f == 0.0:
            return None
        value += num_f / den_f
    return value


def parse_target_range(text: str | None) -> dict[str, object] | None:
    """The target range as ``{low_pct, high_pct, text}``, or ``None``.

    Reads the Fed's own mixed-fraction wording. ``None`` means the phrase was
    not present — never a zero, never a guess.
    """
    if not text:
        return None
    match = _TARGET_RANGE.search(text)
    if match is None:
        return None
    low = _fraction_to_float(match.group("low"))
    high = _fraction_to_float(match.group("high"))
    if low is None or high is None:
        return None
    return {
        "low_pct": round(low, 6),
        "high_pct": round(high, 6),
        "text": _WHITESPACE.sub(" ", match.group(0)).strip(),
    }


def policy_rate_change(
    previous_target: Mapping[str, Any] | None,
    current_target: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Change in the target range midpoint, in basis points (§43).

    ``change_bp`` is ``None`` (with ``direction`` ``None``) whenever either
    side is missing — an unknown range is not a hold. A zero change IS a
    hold and says so.
    """
    prev_mid = _midpoint(previous_target)
    cur_mid = _midpoint(current_target)
    if prev_mid is None or cur_mid is None:
        return {
            "change_bp": None,
            "direction": None,
            "previous": dict(previous_target) if previous_target else None,
            "current": dict(current_target) if current_target else None,
            "reason": "target range unavailable on one or both statements",
        }
    change_bp = int(round((cur_mid - prev_mid) * 100.0))
    if change_bp > 0:
        direction = DIRECTION_HIKE
    elif change_bp < 0:
        direction = DIRECTION_CUT
    else:
        direction = DIRECTION_HOLD
    return {
        "change_bp": change_bp,
        "direction": direction,
        "previous": dict(previous_target),
        "current": dict(current_target),
        "reason": None,
    }


def _midpoint(target: Mapping[str, Any] | None) -> float | None:
    if not target:
        return None
    low = target.get("low_pct")
    high = target.get("high_pct")
    try:
        low_f = float(low)  # type: ignore[arg-type]
        high_f = float(high)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (low_f + high_f) / 2.0


def vote_dispersion(vote: Mapping[str, Any] | None) -> dict[str, object]:
    """The committee split (§43, COMMITTEE_DISPERSION).

    ``unanimous`` is ``True`` only when ``against`` is known AND zero: an
    unparsed vote is ``None``/``False``, never optimistically unanimous.
    """
    vote = dict(vote or {})
    for_votes = _int_or_none(vote.get("for"))
    against = _int_or_none(vote.get("against"))
    dissenters = [str(d) for d in (vote.get("dissenters") or []) if str(d).strip()]
    return {
        "for": for_votes,
        "against": against,
        "dissenters": dissenters,
        "unanimous": against == 0 if against is not None else None,
        "text": vote.get("text"),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def dimension_report(
    prev_statement: Mapping[str, Any] | None,
    cur_statement: Mapping[str, Any] | None,
    diff: StatementDiff | None,
) -> dict[str, dict[str, object]]:
    """Per-dimension view of the two statements (§43).

    EVERY dimension in :data:`DIMENSIONS` is always present, so the UI table
    has fixed rows and "we found nothing on the balance sheet" is visible
    rather than absent. Status is derived from the diff rows tagged with the
    dimension:

    * any CHANGED row, or both an ADDED and a REMOVED row → ``CHANGED``
    * only ADDED rows → ``ADDED``; only REMOVED rows → ``REMOVED``
    * only UNCHANGED rows → ``UNCHANGED``
    * no rows at all → ``NA``

    There is no aggregation across dimensions and no score. That is the
    point of §43.
    """
    prev_statement = dict(prev_statement or {})
    cur_statement = dict(cur_statement or {})
    prev_paras = list(prev_statement.get("paragraphs") or ())
    cur_paras = list(cur_statement.get("paragraphs") or ())
    prev_by_dim = _sentences_by_dimension(prev_paras)
    cur_by_dim = _sentences_by_dimension(cur_paras)
    items = list(diff.items) if diff is not None else []

    report: dict[str, dict[str, object]] = {}
    for dimension in DIMENSIONS:
        rows = [item for item in items if dimension in item.dimensions]
        statuses = {item.status for item in rows}
        if not rows:
            status = STATUS_NA
        elif STATUS_CHANGED in statuses or (
            STATUS_ADDED in statuses and STATUS_REMOVED in statuses
        ):
            status = STATUS_CHANGED
        elif STATUS_ADDED in statuses:
            status = STATUS_ADDED
        elif STATUS_REMOVED in statuses:
            status = STATUS_REMOVED
        else:
            status = STATUS_UNCHANGED
        report[dimension] = {
            "dimension": dimension,
            "status": status,
            "previous": list(prev_by_dim.get(dimension, ())),
            "current": list(cur_by_dim.get(dimension, ())),
            "changed_rows": [item.to_dict() for item in rows if item.status != STATUS_UNCHANGED],
            "notes": _dimension_notes(dimension, prev_statement, cur_statement),
        }

    # COMMITTEE_DISPERSION is a fact about the vote, not about prose: when the
    # vote parsed, it overrides whatever the keyword tagger found.
    prev_vote = vote_dispersion(prev_statement.get("vote"))
    cur_vote = vote_dispersion(cur_statement.get("vote"))
    row = report[DIMENSION_COMMITTEE_DISPERSION]
    row["previous_vote"] = prev_vote
    row["current_vote"] = cur_vote
    if prev_vote["against"] is not None and cur_vote["against"] is not None:
        row["status"] = (
            STATUS_UNCHANGED
            if prev_vote["against"] == cur_vote["against"]
            and prev_vote["for"] == cur_vote["for"]
            else STATUS_CHANGED
        )
    row["dissent_change"] = (
        None
        if prev_vote["against"] is None or cur_vote["against"] is None
        else int(cur_vote["against"]) - int(prev_vote["against"])
    )

    # POLICY_RATE likewise carries the parsed range, which is a fact.
    report[DIMENSION_POLICY_RATE]["policy_rate_change"] = policy_rate_change(
        prev_statement.get("target_range"), cur_statement.get("target_range")
    )
    return report


def _sentences_by_dimension(paragraphs: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {d: [] for d in DIMENSIONS}
    for sentence in split_sentences(paragraphs):
        for dimension in dimensions_for(sentence.text):
            out[dimension].append(sentence.text)
    return out


def _dimension_notes(
    dimension: str,
    prev_statement: Mapping[str, Any],
    cur_statement: Mapping[str, Any],
) -> str | None:
    if dimension == DIMENSION_POLICY_RATE:
        cur = cur_statement.get("target_range") or {}
        text = cur.get("text") if isinstance(cur, Mapping) else None
        return str(text) if text else None
    if dimension == DIMENSION_COMMITTEE_DISPERSION:
        vote = cur_statement.get("vote") or {}
        text = vote.get("text") if isinstance(vote, Mapping) else None
        return str(text) if text else None
    return None


# ---------------------------------------------------------------------------
# §45 — the two reaction windows
# ---------------------------------------------------------------------------


def _bar_ts(bar: Any) -> datetime | None:
    for attr in ("ts_utc", "ts", "timestamp"):
        value = getattr(bar, attr, None)
        if value is None and isinstance(bar, Mapping):
            value = bar.get(attr)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return None
            return value.astimezone(UTC)
    if isinstance(bar, Mapping):
        value = bar.get("ts_utc") or bar.get("ts")
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.astimezone(UTC)
    return None


def _bar_close(bar: Any) -> float | None:
    value = getattr(bar, "close", None)
    if value is None and isinstance(bar, Mapping):
        value = bar.get("close")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _window_move(
    bars: Sequence[Any], start: datetime, end: datetime
) -> dict[str, object]:
    """Close-to-close move across ``[start, end]`` from minute bars.

    ``pre_close`` is the last bar at or before ``start`` — the price standing
    when the document landed — and ``post_close`` the last bar at or before
    ``end``. Both ``None`` with a reason when the window has no bars; there
    is no zero standing in for "no data".
    """
    pre: tuple[datetime, float] | None = None
    post: tuple[datetime, float] | None = None
    for bar in bars:
        ts = _bar_ts(bar)
        close = _bar_close(bar)
        if ts is None or close is None:
            continue
        if ts <= start:
            pre = (ts, close)
        if start < ts <= end:
            post = (ts, close)
    if pre is None:
        return {
            "pre_close": None,
            "pre_ts": None,
            "post_close": None,
            "post_ts": None,
            "return_pct": None,
            "bars": 0,
            "reason": "no bar at or before the window start",
        }
    if post is None:
        return {
            "pre_close": pre[1],
            "pre_ts": _iso(pre[0]),
            "post_close": None,
            "post_ts": None,
            "return_pct": None,
            "bars": 0,
            "reason": "no bar inside the window",
        }
    if pre[1] <= 0.0:
        return {
            "pre_close": pre[1],
            "pre_ts": _iso(pre[0]),
            "post_close": post[1],
            "post_ts": _iso(post[0]),
            "return_pct": None,
            "bars": 1,
            "reason": "pre_close_not_positive",
        }
    n = sum(
        1
        for bar in bars
        if (ts := _bar_ts(bar)) is not None and start < ts <= end
    )
    return {
        "pre_close": pre[1],
        "pre_ts": _iso(pre[0]),
        "post_close": post[1],
        "post_ts": _iso(post[0]),
        "return_pct": round((post[1] / pre[1] - 1.0) * 100.0, 6),
        "bars": n,
        "reason": None,
    }


def fomc_reaction_windows(
    minute_bars_by_symbol: Mapping[str, Sequence[Any]],
    *,
    decision_at_utc: datetime,
    press_conf_at_utc: datetime,
    press_conf_end_utc: datetime,
) -> dict[str, object]:
    """The §45 pair of windows from 1-minute bars.

    ``statement`` runs ``decision_at_utc`` → ``press_conf_at_utc`` (14:00 →
    14:30 ET) and ``press_conference`` runs ``press_conf_at_utc`` →
    ``press_conf_end_utc`` (14:30 → 15:30 ET). They are reported side by side
    and are NEVER combined: the market reversing between the statement and
    the Chair's Q&A is the single most informative thing an FOMC replay can
    show, and one blended number erases it.

    ``return_pct`` is in PERCENT (0.4 is +0.4%) — unlike
    :class:`reaction.ReactionResult`, which reports fractions. The two never
    share a formatter, and the ``unit`` key in the payload says which is which.
    """
    start = require_utc(decision_at_utc, name="decision_at_utc")
    mid = require_utc(press_conf_at_utc, name="press_conf_at_utc")
    end = require_utc(press_conf_end_utc, name="press_conf_end_utc")
    if not (start <= mid <= end):
        raise ValueError(
            "windows must be ordered decision_at_utc <= press_conf_at_utc "
            "<= press_conf_end_utc"
        )

    statement: dict[str, object] = {}
    presser: dict[str, object] = {}
    for symbol in sorted(minute_bars_by_symbol):
        bars = list(minute_bars_by_symbol.get(symbol) or ())
        bars.sort(key=lambda b: (_bar_ts(b) or datetime.min.replace(tzinfo=UTC)))
        statement[symbol] = _window_move(bars, start, mid)
        presser[symbol] = _window_move(bars, mid, end)

    return {
        "basis": REACTION_BASIS_MINUTE,
        "unit": "percent",
        "statement": statement,
        "press_conference": presser,
        "windows": {
            "statement": {
                "start": _iso(start),
                "end": _iso(mid),
                "label_et": "14:00-14:30 ET",
            },
            "press_conference": {
                "start": _iso(mid),
                "end": _iso(end),
                "label_et": "14:30-15:30 ET",
            },
        },
        "separated": True,
        "model_version": FED_INTEL_MODEL_VERSION,
    }


def fomc_reaction_daily(
    daily_bars_by_symbol: Mapping[str, Sequence[Any]],
    *,
    decision_at_utc: datetime,
    session: Any = None,
    horizons: Sequence[int] = (1, 5),
    yields: Sequence[Any] = (),
) -> dict[str, object]:
    """Daily fallback when no minute bars are stored (§45).

    Delegates to :func:`macro.multi_asset_reaction` so an FOMC reaction and a
    CPI reaction are the SAME arithmetic, and falls back to
    :func:`reaction.event_reaction` per symbol only if that helper is
    unavailable. Either way ``basis`` is :data:`REACTION_BASIS_DAILY` and
    ``separated`` is ``False``: a daily bar spans both windows, so the payload
    states plainly that the statement and the presser could NOT be told apart
    — it does not quietly report the blend as if it were the statement.
    """
    from .reaction import DailyBar, event_reaction  # local: keeps imports flat
    from .taxonomy import eastern_date

    moment = require_utc(decision_at_utc, name="decision_at_utc")
    horizons = tuple(sorted({int(k) for k in horizons if int(k) >= 1}))

    if session is None:
        from libs.trading_core.models.enums import EventSession

        session = EventSession.DURING_MARKET

    assets: dict[str, object] = {}
    unavailable: dict[str, str] = {}
    yield_block: dict[str, object] = {}

    try:
        from .macro import multi_asset_reaction  # noqa: PLC0415
    except ImportError:  # pragma: no cover - macro ships in Phase G
        multi_asset_reaction = None  # type: ignore[assignment]

    if multi_asset_reaction is not None and daily_bars_by_symbol:
        roles = {symbol: "fomc" for symbol in daily_bars_by_symbol}
        table = multi_asset_reaction(
            {k: list(v or ()) for k, v in daily_bars_by_symbol.items()},
            list(yields),
            event_at_utc=moment,
            session=session,
            horizons=horizons,
            asset_roles=roles,
        )
        for symbol, item in table.assets.items():
            assets[symbol] = {
                "basis": item.basis,
                "pre_event_close": item.pre_event_close,
                "pre_event_date": _iso(item.pre_event_date),
                "react_date": _iso(item.react_date),
                # multi_asset_reaction reports FRACTIONS; this payload is in
                # percent, so the conversion happens once, here, and the
                # ``unit`` key below names the result.
                "returns_pct": {
                    str(k): (None if v is None else round(v * 100.0, 6))
                    for k, v in sorted(item.returns.items())
                },
                "reasons": dict(item.reasons),
            }
        unavailable = dict(table.unavailable)
        yield_block = {
            tenor: {
                "before": change.before,
                "after": change.after,
                "change_bp": change.change_bp,
                "reason": change.reason,
            }
            for tenor, change in table.yields.items()
        }
    else:
        event_day = eastern_date(moment)
        for symbol in sorted(daily_bars_by_symbol):
            bars = [b for b in (daily_bars_by_symbol.get(symbol) or ()) if isinstance(b, DailyBar)]
            if not bars:
                unavailable[symbol] = "no stored daily bars"
                continue
            result = event_reaction(bars, event_day, session, horizons=horizons)
            if not result.bars_available:
                unavailable[symbol] = result.reasons.get("bars", "reaction_unavailable")
                continue
            assets[symbol] = {
                "basis": result.basis,
                "pre_event_close": result.pre_event_close,
                "pre_event_date": _iso(result.pre_event_date),
                "react_date": _iso(result.react_date),
                "returns_pct": {
                    str(k): (None if v is None else round(v * 100.0, 6))
                    for k, v in sorted(result.returns.items())
                },
                "reasons": dict(result.reasons),
            }

    return {
        "basis": REACTION_BASIS_DAILY,
        "unit": "percent",
        "label": "daily (no intraday bars)",
        "separated": False,
        "separation_reason": (
            "daily bars cannot separate the 14:00 statement window from the "
            "14:30 press conference window"
        ),
        "assets": assets,
        "unavailable": unavailable,
        "yields": yield_block,
        "event_at_utc": _iso(moment),
        "horizons": list(horizons),
        "model_version": FED_INTEL_MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# §42 — the packet
# ---------------------------------------------------------------------------


def build_fed_packet(
    *,
    current_event: Mapping[str, Any] | None = None,
    previous_decision: Mapping[str, Any] | None = None,
    prev_statement: Mapping[str, Any] | None = None,
    prev_prev_statement: Mapping[str, Any] | None = None,
    prev_minutes: Mapping[str, Any] | None = None,
    speeches_since: Sequence[Mapping[str, Any]] = (),
    macro_prints: Mapping[str, Any] | None = None,
    reactions: Mapping[str, Any] | None = None,
    as_of: datetime,
    key_paragraph_sentences: int = 8,
) -> dict[str, object]:
    """Assemble the §42-§45 Fed packet as a JSON-ready dict.

    Every argument is already AS-OF GATED by the caller (the gateway filters
    ``released_at <= as_of``); this layer re-checks the released instants it
    can see and drops anything later, so a caller bug cannot leak a document
    from the future into an analysis.

    The diff compares ``prev_statement`` (the last statement released before
    this event) against ``prev_prev_statement`` — i.e. what the Committee
    changed at its LAST meeting. That is the question a trader has going into
    the next one; diffing the current event's statement is impossible because
    it has not been written yet.
    """
    moment = require_utc(as_of, name="as_of")
    prev_statement = _gate(prev_statement, moment)
    prev_prev_statement = _gate(prev_prev_statement, moment)
    prev_minutes = _gate(prev_minutes, moment)

    prev_paras = list((prev_statement or {}).get("paragraphs") or ())
    prev_prev_paras = list((prev_prev_statement or {}).get("paragraphs") or ())
    diff = statement_diff(prev_prev_paras, prev_paras)
    dimensions = dimension_report(prev_prev_statement, prev_statement, diff)

    speeches = []
    for item in speeches_since or ():
        gated = _gate(dict(item), moment)
        if gated is None:
            continue
        speeches.append(
            {
                "speaker": gated.get("speaker"),
                "title": gated.get("title"),
                "at": _as_iso(gated.get("released_at") or gated.get("at")),
                "url": gated.get("url"),
            }
        )
    speeches.sort(key=lambda s: (s.get("at") or "", s.get("url") or ""))

    reactions = dict(reactions or {})
    reaction_basis = reactions.get("basis")
    previous_reaction: dict[str, object] = {
        "basis": reaction_basis,
        "available": bool(reactions),
    }
    if reaction_basis == REACTION_BASIS_MINUTE:
        previous_reaction.update(
            {
                "statement": reactions.get("statement", {}),
                "press_conference": reactions.get("press_conference", {}),
                "windows": reactions.get("windows", {}),
                "separated": True,
                "unit": reactions.get("unit", "percent"),
            }
        )
    elif reaction_basis == REACTION_BASIS_DAILY:
        previous_reaction.update(
            {
                "statement": {},
                "press_conference": {},
                "daily": reactions,
                "separated": False,
                "label": reactions.get("label", "daily (no intraday bars)"),
                "unit": reactions.get("unit", "percent"),
            }
        )
    else:
        previous_reaction.update(
            {
                "statement": {},
                "press_conference": {},
                "separated": False,
                "reason": "no stored bars around the previous decision",
            }
        )

    macro_prints = dict(macro_prints or {})
    data_block = {
        "inflation": macro_prints.get("inflation"),
        "labor": macro_prints.get("labor"),
        "growth": macro_prints.get("growth"),
        "available": any(
            macro_prints.get(k) for k in ("inflation", "labor", "growth")
        ),
    }

    proxy_bp = _proxy_2y_bp(reactions)
    market_pricing = {
        "status": MARKET_PRICING_UNAVAILABLE,
        "reason": MARKET_PRICING_NOTE,
        "proxy": None if proxy_bp is None else {_PROXY_KEY_2Y: proxy_bp, "label": "2Y yield change (proxy)"},
    }

    prev_statement_block = {
        "available": prev_statement is not None,
        "url": (prev_statement or {}).get("url"),
        "title": (prev_statement or {}).get("title"),
        "released_at": _as_iso((prev_statement or {}).get("released_at")),
        "meeting_date": _as_iso((prev_statement or {}).get("meeting_date")),
        "vote": vote_dispersion((prev_statement or {}).get("vote")),
        "target_range": (prev_statement or {}).get("target_range"),
        "paragraphs": prev_paras,
    }
    compared_block = {
        "available": prev_prev_statement is not None,
        "url": (prev_prev_statement or {}).get("url"),
        "released_at": _as_iso((prev_prev_statement or {}).get("released_at")),
        "meeting_date": _as_iso((prev_prev_statement or {}).get("meeting_date")),
        "vote": vote_dispersion((prev_prev_statement or {}).get("vote")),
        "target_range": (prev_prev_statement or {}).get("target_range"),
    }
    prev_statement_block["compared_to"] = compared_block

    minutes_block = {
        "available": prev_minutes is not None,
        "url": (prev_minutes or {}).get("url"),
        "released_at": _as_iso((prev_minutes or {}).get("released_at")),
        "meeting_date": _as_iso((prev_minutes or {}).get("meeting_date")),
        "key_paragraphs": _key_minute_sentences(
            list((prev_minutes or {}).get("paragraphs") or ()),
            limit=key_paragraph_sentences,
        ),
    }

    coverage = {
        "previous_statement": prev_statement is not None,
        "compared_statement": prev_prev_statement is not None,
        "statement_diff": bool(diff.items),
        "previous_minutes": prev_minutes is not None,
        "subsequent_speeches": len(speeches),
        "macro_prints": data_block["available"],
        "previous_reaction": bool(reaction_basis),
        "market_pricing": False,
    }
    tiers = {
        "previous_statement": "DATA",
        "statement_diff": "QUANT",
        "dimensions": "QUANT",
        "previous_minutes": "DATA",
        "subsequent_speeches": "DATA",
        "data": "DATA",
        "market_pricing": "DATA",
        "previous_reaction": "QUANT",
    }

    packet = FedPacket(
        as_of=moment,
        event=dict(current_event or {}),
        previous_statement=prev_statement_block,
        statement_diff=diff.to_dict(),
        dimensions=dimensions,
        previous_minutes=minutes_block,
        subsequent_speeches=tuple(speeches),
        data=data_block,
        market_pricing=market_pricing,
        previous_reaction=previous_reaction,
        coverage=coverage,
        tiers=tiers,
        disclaimers=(
            SOURCE_AUTHORITATIVE_NOTE,
            NO_SINGLE_SCORE_NOTE,
            MARKET_PRICING_NOTE,
        ),
    )
    out = packet.to_dict()
    out["previous_decision"] = dict(previous_decision or {})
    return out


def _gate(
    doc: Mapping[str, Any] | None, as_of: datetime
) -> dict[str, Any] | None:
    """Drop a document whose ``released_at`` is after ``as_of`` (§14/§96)."""
    if not doc:
        return None
    released = doc.get("released_at")
    if isinstance(released, datetime):
        if released.tzinfo is None:
            return None
        if released.astimezone(UTC) > as_of:
            return None
    return dict(doc)


def _as_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo else None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value
    return None


def _proxy_2y_bp(reactions: Mapping[str, Any]) -> float | None:
    """The 2Y change in bp out of a daily reaction block, if present."""
    yields = reactions.get("yields")
    if not isinstance(yields, Mapping):
        return None
    for tenor, change in yields.items():
        if "2" not in str(tenor):
            continue
        if isinstance(change, Mapping):
            value = change.get("change_bp")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _key_minute_sentences(
    paragraphs: Sequence[str], *, limit: int
) -> list[dict[str, object]]:
    """First ``limit`` minutes sentences that carry a dimension tag.

    The minutes run to thousands of words; a packet that pasted all of them
    would blow the LLM budget on boilerplate. Keeping only the tagged
    sentences, in document order, is a mechanical selection a reader can
    audit — no summarisation happens here.
    """
    out: list[dict[str, object]] = []
    for sentence in split_sentences(paragraphs):
        tags = dimensions_for(sentence.text)
        if not tags:
            continue
        out.append(
            {
                "idx": sentence.idx,
                "para_idx": sentence.para_idx,
                "text": sentence.text,
                "dimensions": list(tags),
            }
        )
        if len(out) >= max(0, int(limit)):
            break
    return out
