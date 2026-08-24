"""The EventEvidenceBundle — the structured object the LLM reads (event spec
§46, §47, §49, §33, §35, §85, §91; audit §7, §11.6 Phase F unit U1).

Pure stdlib, deterministic, **no I/O**. Like every other module under
``libs/trading_core/events/`` this one may not import ``apps/``,
``libs.market_data`` or ``libs.event_calendar`` (audit §7.4 static guard); the
gateway seam (``apps/gateway/event_evidence.py``) composes the already-rendered
section payloads and hands them here to be framed, ordered, digested and
flattened.

Four ideas carry the module:

1. **The bundle is the LLM's ONLY source of numbers** (§47). The backend
   calculates; the LLM interprets. That contract is only enforceable if there
   is a machine-checkable list of what the backend actually said, which is
   what :func:`fact_index` is: every scalar in the bundle, keyed by its dotted
   path (``"price_analysis.pre_event.run_up_pct"``). The validator downstream
   accepts a number in the model's prose only when the model also names the
   path it came from and that path holds that value. A model that computes
   ``(1.42 - 1.31) / 1.31`` produces a number no path holds, and it is caught
   — not because the arithmetic is wrong, but because the platform never said
   it.

2. **Every section carries its own tier** (§49). ``"tier": "DATA"`` is a
   provider's own figure (a filer's XBRL, a vendor's close, an article's
   headline); ``"QUANT"`` is this platform's arithmetic over it. The tier is
   stamped on the SECTION, not inferred by the UI from the key name, because
   the same payload mixes both (``price_analysis`` carries DATA bars and QUANT
   returns) and only the seam that assembled it knows which is which. Prior
   LLM opinions, when U3 attaches them, are ``"LLM_PRIOR"`` — never evidence
   (§70).

3. **Consensus is absent at every instant, and says so in the same words
   every time** (§33, §98; audit §7.3). :data:`CONSENSUS_STATUS` is the one
   spelling. An omitted consensus block would read as "not applicable"; a
   computed EPS surprise would be a fabrication; and a block whose wording
   varies by section makes the UI guess whether two absences are the same
   absence. They are.

4. **Digest over CANONICAL JSON of the CONTENT, not over a dict** (§71, and
   the U3 cache key). :func:`bundle_digest` sorts keys and fixes separators,
   so the same evidence hashes identically across processes and Python
   versions. That hash is what lets a regenerate request answer "the evidence
   has not changed since the analysis you already have" instead of paying for
   a second inference on identical input — which it can only do if the hash
   covers the EVIDENCE and nothing else. :func:`digest_view` therefore prunes
   the clock readings (``as_of``, ``generated_at``, ``fetched_at``, the news
   window's END) from the hashed copy: two POSTs a minute apart over identical
   filings, bars and articles are the same question, and hashing the wall
   clock made every press a cache miss that spent a model call re-deriving an
   answer already on disk. The stored and served bundle keeps every one of
   those fields — they are pruned from the COPY that is hashed, never from
   the document.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not label a §35 regime.
:func:`compute_expectations_gap_inputs` returns the fundamental-momentum score
and the expectation proxies (run-up since the last print, material positive
and negative development counts) and STOPS — the four regimes in §35 are
interpretation, the spec says so in its own last line ("analytical
interpretation, not a guaranteed trading signal"), and a deterministic label
here would be an opinion wearing a number's clothes that the LLM would then
be unable to disagree with. It also never invents a section: a seam that
failed hands over ``available: false`` with a reason and the bundle carries
that verbatim into ``coverage``.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from libs.trading_core.events.taxonomy import require_utc

__all__ = [
    "BUNDLE_MODEL_VERSION",
    "CONSENSUS_STATUS",
    "EvidenceBundle",
    "MACRO_CONTEXT_PLACEHOLDER",
    "NEWS_CLUSTER_LIMIT",
    "OPTIONS_PLACEHOLDER",
    "PEER_CONTEXT_PLACEHOLDER",
    "SECTION_ORDER",
    "TIER_DATA",
    "TIER_LLM",
    "TIER_LLM_PRIOR",
    "TIER_QUANT",
    "bundle_digest",
    "bundle_to_json",
    "canonical_json",
    "compute_expectations_gap_inputs",
    "consensus_block",
    "digest_view",
    "fact_index",
    "json_safe",
]

#: The arithmetic this module's shape belongs to — stored alongside a bundle
#: so an analysis generated last month says which assembly produced its
#: evidence, exactly as ``FUNDAMENTALS_MODEL_VERSION`` does for the metrics.
#: v2 (Catalyst research upgrade): adds the ``web_research`` and
#: ``prediction_markets`` sections and populates ``options_analysis`` from
#: the live Phase I subsystem — a material contract change, so the version
#: is BUMPED rather than v1 silently mutating (an analysis cached under v1
#: was generated over different evidence and must not be served as v2's).
BUNDLE_MODEL_VERSION = "f1-evidence-v2"

#: The §49 tiers. DATA is somebody else's fact, QUANT is this platform's
#: arithmetic, LLM is a model's words, LLM_PRIOR is a model's EARLIER words
#: re-shown as context and never as evidence (§70).
TIER_DATA = "DATA"
TIER_QUANT = "QUANT"
TIER_LLM = "LLM"
TIER_LLM_PRIOR = "LLM_PRIOR"

#: §33/§98 — the single spelling of the consensus absence. The status string
#: is matched literally by the UI and by the "no fabricated consensus" grep in
#: the Phase F definition of done, so it is a constant rather than a sentence
#: three modules each write their own way.
CONSENSUS_STATUS = "CONSENSUS_DATA_UNAVAILABLE"

#: §36 — options intelligence is Phase I. The key is PRESENT and says
#: NOT_AVAILABLE_YET rather than being omitted: an absent key reads as "this
#: event has no options angle", which is false for every optionable name, and
#: the LLM would fill the silence.
OPTIONS_PLACEHOLDER: dict[str, Any] = {
    "status": "NOT_AVAILABLE_YET",
    "reason": "options intelligence not yet available (Phase I)",
    "tier": TIER_DATA,
}

#: §39 — macro proxies are Phase G.
MACRO_CONTEXT_PLACEHOLDER: dict[str, Any] = {
    "status": "NOT_AVAILABLE_YET",
    "reason": "macro regime context not yet available (Phase G)",
    "tier": TIER_DATA,
}

#: §37 — peer/sector comparison is Phase G/J. Mirrors the wording
#: ``fundamentals.PEER_CONTEXT_REASON`` already uses inside the valuation
#: block, so the two absences read as the same absence.
PEER_CONTEXT_PLACEHOLDER: dict[str, Any] = {
    "status": "NOT_AVAILABLE_YET",
    "reason": "peer/sector multiples not implemented (Phase G/J)",
    "tier": TIER_DATA,
}

#: Catalyst research upgrade — the honest defaults for the two research
#: sections when their gateway seam did not fill them. NEVER_RUN is a
#: runtime state ("no refresh has stored rows yet"), distinct from the
#: NOT_AVAILABLE_YET build-phase placeholders above, and distinct from the
#: seam's own NO_RELEVANT_PREDICTION_MARKET / provider-failure answers.
WEB_RESEARCH_PLACEHOLDER: dict[str, Any] = {
    "available": False,
    "reason": "NEVER_RUN",
    "tier": TIER_DATA,
}
PREDICTION_MARKETS_PLACEHOLDER: dict[str, Any] = {
    "available": False,
    "reason": "NEVER_RUN",
    "tier": TIER_DATA,
}

#: The §46 section order, top level. Fixed here rather than left to insertion
#: order so the JSON the model reads — and the digest taken over it — is the
#: same shape whichever seam happened to resolve first.
SECTION_ORDER: tuple[str, ...] = (
    "bundle_version",
    "event",
    "as_of",
    "previous_event",
    "previous_event_results",
    "previous_market_reaction",
    "fundamentals",
    "price_analysis",
    "options_analysis",
    "news",
    # Catalyst research upgrade (v2): web research rides beside news (both
    # are retrieved third-party text), prediction markets beside consensus
    # (both are market-expectation layers).
    "web_research",
    "consensus",
    "prediction_markets",
    "expectations_gap_inputs",
    "macro_context",
    "peer_context",
    "prior_analyses",
    "source_metadata",
    "coverage",
)

#: How many news clusters ride in the bundle. Twelve is the §46 cut: enough
#: that a quarter's material developments are all present (a heavily covered
#: mega-cap rarely exceeds eight material stories in one window) and few
#: enough that the untrusted-text block stays a bounded fraction of the
#: prompt. The COUNTS are computed over everything, so this truncation never
#: changes the §26 headline the model reads.
NEWS_CLUSTER_LIMIT = 12


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """``value`` as something :func:`json.dumps` accepts, recursively.

    The seams hand over payloads that are ALMOST JSON already — they were
    written to be returned by FastAPI, which serialises ``datetime``, ``date``
    and ``tuple`` on the way out. The bundle is not returned by FastAPI; it is
    hashed, stored in a JSON column and embedded in a prompt, so those types
    must be resolved HERE or the digest depends on which serialiser ran.
    ``expectations_gap_inputs`` returning ``metrics_considered`` as a tuple is
    the concrete case this exists for.

    NaN and infinity become ``None``: ``json.dumps`` emits them as bare
    ``NaN``/``Infinity`` tokens, which are not JSON, and a model reading
    ``"score": NaN`` has been handed a number that means nothing. The pure
    layers already refuse to produce them, so this is a backstop rather than a
    routine path — but a backstop that silently corrupts a digest is worse
    than none.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return require_utc(value, name="bundle datetime").isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [json_safe(item) for item in items]
    return str(value)


def _ordered(payload: Mapping[str, Any], order: Sequence[str]) -> dict[str, Any]:
    """``payload`` with ``order``'s keys first, then everything else sorted.

    Deterministic without being lossy: a key a seam adds tomorrow still
    travels (sorted into the tail) instead of being dropped by an
    order-as-whitelist rendering, which would silently shrink the evidence the
    moment an upstream section grew a field.
    """
    rendered: dict[str, Any] = {}
    for key in order:
        if key in payload:
            rendered[key] = payload[key]
    for key in sorted(payload):
        if key not in rendered:
            rendered[key] = payload[key]
    return rendered


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceBundle:
    """The §46 EventEvidenceBundle as a value object.

    Every section is a plain mapping the seam already rendered, not a nested
    dataclass: the seams own those shapes (they are the same dicts the
    ``/price-context``, ``/fundamentals`` and ``/news`` endpoints return, and
    the UI already reads them), and re-modelling them here would create a
    second definition free to drift from the endpoints'. What this class adds
    is the §46 FRAME — which sections exist, in which order, with which tier,
    and what the coverage story is when one of them could not be built.

    ``coverage`` is not optional and not derived: a section that failed is
    ``{"available": false, "reason": ...}`` here, and the model is told to
    read it. A bundle whose fundamentals block is missing because a vendor
    403'd looks, to a reader with no coverage map, exactly like a company
    that reports nothing.
    """

    event: Mapping[str, Any]
    as_of: datetime
    previous_event: Mapping[str, Any] | None = None
    previous_event_results: Mapping[str, Any] | None = None
    previous_market_reaction: Mapping[str, Any] | None = None
    fundamentals: Mapping[str, Any] | None = None
    price_analysis: Mapping[str, Any] | None = None
    options_analysis: Mapping[str, Any] = field(
        default_factory=lambda: dict(OPTIONS_PLACEHOLDER)
    )
    news: Mapping[str, Any] | None = None
    web_research: Mapping[str, Any] = field(
        default_factory=lambda: dict(WEB_RESEARCH_PLACEHOLDER)
    )
    consensus: Mapping[str, Any] | None = None
    prediction_markets: Mapping[str, Any] = field(
        default_factory=lambda: dict(PREDICTION_MARKETS_PLACEHOLDER)
    )
    expectations_gap_inputs: Mapping[str, Any] | None = None
    macro_context: Mapping[str, Any] = field(
        default_factory=lambda: dict(MACRO_CONTEXT_PLACEHOLDER)
    )
    peer_context: Mapping[str, Any] = field(
        default_factory=lambda: dict(PEER_CONTEXT_PLACEHOLDER)
    )
    prior_analyses: Sequence[Mapping[str, Any]] = ()
    source_metadata: Sequence[Mapping[str, Any]] = ()
    coverage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    model_version: str = BUNDLE_MODEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "as_of", require_utc(self.as_of, name="as_of")
        )
        object.__setattr__(self, "prior_analyses", tuple(self.prior_analyses))
        object.__setattr__(self, "source_metadata", tuple(self.source_metadata))
        object.__setattr__(self, "coverage", dict(self.coverage))

    def to_dict(self) -> dict:
        """The bundle as the JSON the model reads — see :func:`bundle_to_json`."""
        return bundle_to_json(self)


def consensus_block(reason: str | None = None) -> dict[str, Any]:
    """The §33 consensus block — ALWAYS unavailable, always worded the same.

    ``reason`` lets the seam pass the concrete vendor sentence the
    fundamentals payload already carries ("Massive Benzinga estimates not in
    subscription (403)") without letting it replace the machine-readable
    ``status``, which is what the UI and the Phase F grep key on.
    """
    return {
        "status": CONSENSUS_STATUS,
        "available": False,
        "reason": reason
        or "no consensus/estimate provider in subscription",
        "eps_consensus": None,
        "revenue_consensus": None,
        "guidance_expectations": None,
        "tier": TIER_DATA,
    }


def bundle_to_json(bundle: EvidenceBundle) -> dict:
    """The bundle as a JSON-safe dict with a deterministic key order (§46).

    ``json_safe`` runs over the WHOLE structure rather than over the sections
    the seam suspects of holding a ``datetime``: the payloads come from four
    different modules, each free to add a field, and a per-section allowlist
    here would be a digest that changes meaning the first time one of them
    does. Sections the caller left ``None`` are rendered as an explicit
    ``{"status": "NOT_AVAILABLE", ...}`` rather than dropped, for the same
    reason :data:`OPTIONS_PLACEHOLDER` is present: a missing key is read as
    "not applicable", and the model fills silence with plausibility.
    """
    def _section(value: Mapping[str, Any] | None, name: str) -> Any:
        if value is None:
            return {
                "status": "NOT_AVAILABLE",
                "available": False,
                "reason": (
                    bundle.coverage.get(name, {}).get("reason")
                    or f"{name} was not available at as_of"
                ),
            }
        return value

    payload: dict[str, Any] = {
        "bundle_version": bundle.model_version,
        "event": bundle.event,
        "as_of": bundle.as_of,
        "previous_event": _section(bundle.previous_event, "previous_event"),
        "previous_event_results": _section(
            bundle.previous_event_results, "previous_event_results"
        ),
        "previous_market_reaction": _section(
            bundle.previous_market_reaction, "previous_market_reaction"
        ),
        "fundamentals": _section(bundle.fundamentals, "fundamentals"),
        "price_analysis": _section(bundle.price_analysis, "price_analysis"),
        "options_analysis": bundle.options_analysis,
        "news": _section(bundle.news, "news"),
        "web_research": bundle.web_research,
        "consensus": bundle.consensus or consensus_block(),
        "prediction_markets": bundle.prediction_markets,
        "expectations_gap_inputs": _section(
            bundle.expectations_gap_inputs, "expectations_gap_inputs"
        ),
        "macro_context": bundle.macro_context,
        "peer_context": bundle.peer_context,
        "prior_analyses": list(bundle.prior_analyses),
        "source_metadata": list(bundle.source_metadata),
        "coverage": bundle.coverage,
    }
    return _ordered(json_safe(payload), SECTION_ORDER)


def canonical_json(bundle_json: Mapping[str, Any]) -> str:
    """``bundle_json`` as the exact bytes the digest is taken over.

    ``sort_keys`` on purpose: the digest must not depend on the insertion
    order :func:`bundle_to_json` produced, because a section reordered in a
    later release would otherwise invalidate every cached analysis without a
    single number having changed. ``ensure_ascii`` is left at its default so a
    Chinese headline (the live config is ``llm_output_language=zh``) hashes
    identically regardless of the terminal encoding that printed it.
    """
    return json.dumps(
        json_safe(bundle_json), sort_keys=True, separators=(",", ":"), default=str
    )


#: Keys whose value is a READING OF THE CLOCK rather than a piece of evidence.
#: Dropped at any depth by :func:`digest_view` before the cache key is taken.
#:
#: Every one of these answers "when was this assembled/fetched?", never "what
#: is true?". Two bundles differing only here were built from the same filings,
#: the same bars and the same articles — the platform said exactly the same
#: thing, one minute apart — and hashing the wall clock into the cache key made
#: EVERY press of the Generate button a miss, spending a model call to
#: re-derive an answer already on disk. That is the live failure this list
#: fixes: two POSTs a minute apart produced two different digests with
#: byte-identical evidence.
_VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "as_of",
        "generated_at",
        "fetched_at",
        "last_fetch_at",
        "window_end_utc",
        "computed_at",
        # Catalyst research upgrade (v2): the research sections' fetch/
        # observation clocks. A re-observation with IDENTICAL prices/evidence
        # must stay cache-valid; the prices and accepted-evidence content
        # themselves remain IN the digest, so any material change still
        # invalidates (plan Phase 10). The stored research window's
        # start/end are NOT volatile: they are frozen at run time and only
        # change when a new refresh lands — which should miss the cache.
        "retrieved_at",
        "observed_at",
        "first_seen_at",
        "last_seen_at",
        # The prediction-market HISTORY series' observation bookkeeping is
        # the same clock species: a flat market observed once more advances
        # history_end and observation_count with byte-identical prices, and
        # matched_at is the instant matching RAN (a re-match with identical
        # decisions must stay cache-valid — the decisions themselves, the
        # candidate count and every price remain in the digest).
        #
        # ``history_start`` is DELIBERATELY NOT HERE. It looks like a clock
        # and is not one: re-observing a market advances history_end and the
        # count but never the start, so keeping it in the digest costs no
        # spurious invalidation. And it is the only field that distinguishes
        # a two-observation series from a two-hundred-observation one when
        # the price has not moved — without it, a market acquiring real
        # depth leaves a stale analysis cached, and depth is precisely what
        # the reader is told to weigh a thin market's price against.
        "history_end",
        "observation_count",
        "matched_at",
    }
)


def digest_view(bundle_json: Mapping[str, Any]) -> dict[str, Any]:
    """``bundle_json`` with the clock readings pruned — what the digest covers.

    THE CACHE KEY IS THE EVIDENCE, AND A TIMESTAMP IS NOT EVIDENCE. The stored
    and served bundle keeps every one of these fields (``as_of`` is part of the
    §46 shape, the news window's end is what makes the counts readable, and a
    reader must be able to see when a vendor was last polled). They are removed
    only from the COPY that is hashed, because the question the digest answers
    is "has the evidence changed since the analysis I already have?" — and a
    minute passing is not a change in the evidence.

    Pruned, at any depth: :data:`_VOLATILE_KEYS`, plus ``news.window.end``
    specifically. The news window's END is the same clock read as ``as_of``
    (the window closes at the request instant); its START is a real parameter
    of what was searched — a 30-day window and a 90-day window are different
    evidence — so ``start`` stays.

    What is deliberately NOT pruned: ``coverage``, every measured number, and
    the reasons a section is missing. An analysis written when fundamentals
    were unavailable is NOT the same analysis as one written after the filing
    landed, and it must not be served from cache in its place.
    """
    def prune(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, Mapping):
            out: dict[str, Any] = {}
            for key, value in node.items():
                name = str(key)
                if name in _VOLATILE_KEYS:
                    continue
                # news.window.end only — the window START is a real search
                # parameter and stays in the hash.
                if name == "end" and path[-2:] == ("news", "window"):
                    continue
                out[name] = prune(value, path + (name,))
            return out
        if isinstance(node, (list, tuple)):
            return [prune(item, path + ("*",)) for item in node]
        return node

    return prune(json_safe(bundle_json), ())


def bundle_digest(bundle_json: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of :func:`digest_view` — the cache key (§71).

    Two bundles with the same digest are the same evidence, so regenerating an
    analysis over one is provably a repeat of the same inference. The digest is
    taken over the PRUNED view rather than the whole document: see
    :func:`digest_view` for which keys are dropped and why. ``coverage`` and
    every measured value still count — an analysis written when fundamentals
    were missing is NOT the same analysis as one written after the filing
    landed, even if every other number matches.
    """
    return hashlib.sha256(
        canonical_json(digest_view(bundle_json)).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# The fact index — what the model is allowed to quote
# ---------------------------------------------------------------------------

#: Keys whose values are prose or provenance rather than facts to be quoted.
#: They are still IN the bundle (the model must read the reason a number is
#: missing) but a "number" quoted from ``reasons.sma200`` is a sentence, and
#: allowing it would let a model cite an explanation as though it were a
#: measurement.
_NON_FACT_KEYS: frozenset[str] = frozenset(
    {"reasons", "notes", "unavailable", "untrusted_text_policy"}
)


def fact_index(
    bundle_json: Mapping[str, Any], *, include_strings: bool = True
) -> dict[str, Any]:
    """Every quotable scalar in the bundle, keyed by dotted path (§47).

    ``{"price_analysis": {"pre_event": {"run_up_pct": 0.17}}}`` becomes
    ``{"price_analysis.pre_event.run_up_pct": 0.17}``; list elements are
    indexed (``"news.clusters.0.materiality"``). THIS IS THE WHOLE ENFORCEMENT
    MECHANISM for §47: the downstream validator accepts a number in the
    model's output only if the model also names a path in this map holding
    that value. A model that multiplies two bundle numbers together produces
    something no path holds — which is exactly the failure §47 exists to
    prevent, caught mechanically rather than by a reviewer noticing.

    ``None`` values are INCLUDED, deliberately. "This field is null" is a fact
    the model may need to state ("margin unavailable — the filer does not
    report it"), and a map that omitted nulls would make every honest absence
    unquotable, pushing the model towards asserting the absence without a
    citation.

    ``include_strings=False`` narrows the map to numbers only, which is what a
    strict numeric check wants; the default keeps strings so an evidence
    reference like a publisher name or a fiscal label can also be cited.
    """
    flat: dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in _NON_FACT_KEYS:
                    continue
                walk(value, f"{path}.{key}" if path else str(key))
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}.{index}")
            return
        if isinstance(node, bool):
            # Booleans are flags, not quotable quantities: "available: true"
            # is not a number a narrative cites, and admitting them would let
            # ``1`` match ``True`` under Python's numeric tower.
            return
        if node is None or isinstance(node, (int, float)):
            flat[path] = node
            return
        if include_strings and isinstance(node, str):
            flat[path] = node

    walk(bundle_json, "")
    flat.pop("", None)
    return flat


# ---------------------------------------------------------------------------
# §35 expectations-gap INPUTS — inputs only, never a regime
# ---------------------------------------------------------------------------

#: How a fundamental-momentum count becomes a −1..+1 score. The count is
#: already the honest primitive (``expectations_gap_inputs`` in
#: ``fundamentals.py`` returns improved/weakened/compared); the score is the
#: same information on a scale a prompt can talk about without the model
#: having to divide — which it is forbidden to do (§47).
def _momentum_score(improved: int, weakened: int, compared: int) -> float | None:
    """``(improved - weakened) / compared`` pinned to [-1, 1], or ``None``.

    ``None`` when nothing was comparable, never ``0.0``: a zero score reads as
    "fundamentals flat", which is a finding, and "we could not compare the two
    snapshots" is not that finding (§44 rule 18).
    """
    if compared <= 0:
        return None
    score = (improved - weakened) / compared
    return max(-1.0, min(1.0, score))


def compute_expectations_gap_inputs(
    fundamentals_change: Mapping[str, Any] | None,
    price_context: Mapping[str, Any] | None,
    news: Mapping[str, Any] | None,
    *,
    consensus_available: bool = False,
) -> dict[str, Any]:
    """The deterministic §35 inputs — WITHOUT labelling a regime.

    §35 asks for "Fundamental Momentum vs Market Expectations", and the four
    regimes it lists (positive asymmetry, beat priced, negative asymmetry, bad
    news priced) are its own words for what an ANALYST concludes from that
    pair. This function supplies the pair and stops:

    - ``fundamental_momentum``: the −1..+1 score over the §29 directional
      metrics, with the counts that produced it, taken from
      ``fundamentals.expectations_gap_inputs`` output as-is.
    - ``expectation_proxies``: what the market has already done — the §32
      run-up since the previous print (the single best proxy the platform
      has for "expectations are elevated"), the relative move against SPY,
      and the §26 material positive/negative development counts.
    - ``consensus``: unavailable, stated (§33), because the DIRECT measure of
      expectations is the one thing the subscription does not carry, and a
      run-up proxy silently standing in for it would be the most flattering
      possible substitution.

    Deciding a regime here would hand the model a conclusion to agree with
    rather than evidence to weigh — and a wrong regime label computed by a
    formula is far harder to argue with than a wrong sentence written by a
    model that had to show its work. The regime enum lives in the LLM's output
    schema (U2), where it is a judgement, labelled as one, with the model's
    own reasoning attached.

    Every argument is optional and ``None`` is expected: a macro event has no
    fundamentals, a fresh install has no news, and this returns the shape with
    honest nulls and reasons rather than raising.
    """
    reasons: dict[str, str] = {}

    # --- fundamental momentum (QUANT, from §29 deltas) --------------------
    momentum: dict[str, Any]
    if not fundamentals_change:
        momentum = {
            "score": None,
            "label": None,
            "improved": None,
            "weakened": None,
            "compared": None,
            "metrics_considered": [],
        }
        reasons["fundamental_momentum"] = (
            "no fundamentals change table was available at as_of"
        )
    else:
        improved = int(fundamentals_change.get("improved") or 0)
        weakened = int(fundamentals_change.get("weakened") or 0)
        compared = int(fundamentals_change.get("compared") or 0)
        score = _momentum_score(improved, weakened, compared)
        momentum = {
            "score": score,
            # The library's own label travels through UNCHANGED — it is a
            # count-backed statement about the metrics, not a regime.
            "label": fundamentals_change.get("label"),
            "improved": improved,
            "weakened": weakened,
            "unchanged": int(fundamentals_change.get("unchanged") or 0),
            "unavailable": int(fundamentals_change.get("unavailable") or 0),
            "compared": compared,
            "metrics_considered": list(
                fundamentals_change.get("metrics_considered") or ()
            ),
        }
        if score is None:
            momentum_reason = fundamentals_change.get("reason")
            reasons["fundamental_momentum"] = (
                momentum_reason
                or "no directional metric was comparable across the two snapshots"
            )

    # --- expectation proxies (QUANT, from §32 positioning + §26 news) -----
    pre_event = (price_context or {}).get("pre_event") or {}
    run_up = pre_event.get("run_up_pct")
    if run_up is None:
        # The price seam's OWN reason when the whole block is unavailable
        # ("no_ticker" for a macro event), otherwise the narrower statement
        # that the block exists but this one measurement could not be taken.
        block_reason = (price_context or {}).get("reason")
        reasons["run_up_since_previous_event"] = (
            str(block_reason)
            if (price_context or {}).get("available") is False and block_reason
            else (
                "run-up since the previous event could not be measured from "
                "stored bars"
            )
        )

    counts = (news or {}).get("counts") or {}
    positive, negative = _material_direction_counts(news)
    proxies: dict[str, Any] = {
        "run_up_since_previous_event": run_up,
        "run_up_anchor_basis": pre_event.get("anchor_basis"),
        "relative_return_vs_benchmark": pre_event.get("relative_return"),
        "distance_from_52w_high_pct": pre_event.get("distance_from_52w_high_pct"),
        "realized_vol_20d": pre_event.get("realized_vol_20d"),
        "material_developments": counts.get("material"),
        "material_positive_developments": positive,
        "material_negative_developments": negative,
    }
    if not news or not news.get("available"):
        news_reason = (news or {}).get("reason")
        reasons["news_developments"] = (
            str(news_reason)
            if news_reason
            else "no news window was available at as_of"
        )

    return {
        "tier": TIER_QUANT,
        "model_version": BUNDLE_MODEL_VERSION,
        "fundamental_momentum": momentum,
        "expectation_proxies": proxies,
        "consensus": (
            {"available": True}
            if consensus_available
            else {
                "status": CONSENSUS_STATUS,
                "available": False,
                "reason": (
                    "no consensus/estimate provider in subscription — the "
                    "expectation side is proxied by price positioning and "
                    "news counts, which is weaker evidence and must be "
                    "described as such"
                ),
            }
        ),
        "interpretation": (
            "INPUTS ONLY — this platform does not label a §35 regime "
            "deterministically. Weigh fundamental_momentum against "
            "expectation_proxies and say which regime you think holds, with "
            "your reasoning."
        ),
        "reasons": reasons,
    }


#: §24 materiality categories whose developments cut in an identifiable
#: direction. Everything else (M&A, regulation, litigation) is material and
#: DIRECTIONLESS on its face — a probe is bad for the target and good for the
#: rival, and the lexicon cannot tell which side it is reading. Counting those
#: as negative would manufacture a bearish tilt out of coverage volume.
_POSITIVE_HINTS: frozenset[str] = frozenset(
    {"beat", "raise", "raises", "raised", "upgrade", "upgrades", "record",
     "surge", "wins", "win", "approval", "approved", "expands", "expansion",
     "strong", "growth", "outperform", "buyback", "dividend"}
)
_NEGATIVE_HINTS: frozenset[str] = frozenset(
    {"miss", "misses", "missed", "cut", "cuts", "downgrade", "downgrades",
     "warns", "warning", "probe", "lawsuit", "recall", "delay", "delays",
     "weak", "slump", "plunge", "layoffs", "resigns", "investigation",
     "halts", "halt", "loss", "losses"}
)


def _material_direction_counts(
    news: Mapping[str, Any] | None
) -> tuple[int | None, int | None]:
    """``(positive, negative)`` material development counts, or ``(None, None)``.

    A deliberately CRUDE keyword tally over the sanitised canonical headlines
    of MATERIAL clusters only, and labelled as a proxy everywhere it appears.
    It exists because §35 needs some expectation signal on the news side and
    the alternative — letting the LLM count them — puts a number in the
    model's output that no bundle path holds, which §47 forbids. A headline
    matching neither list counts towards neither: silence is not neutrality
    asserted, it is simply not counted, and the ``material_developments``
    total above it is what a reader compares against to see how much the
    direction split leaves out.
    """
    if not news or not news.get("available"):
        return None, None
    clusters = news.get("clusters") or []
    positive = 0
    negative = 0
    for cluster in clusters:
        if not isinstance(cluster, Mapping) or not cluster.get("material"):
            continue
        article = cluster.get("canonical_article") or {}
        text = str(article.get("safe_title") or "").lower()
        words = {word.strip(".,:;!?()[]\"'") for word in text.split()}
        if words & _POSITIVE_HINTS:
            positive += 1
        if words & _NEGATIVE_HINTS:
            negative += 1
    return positive, negative
