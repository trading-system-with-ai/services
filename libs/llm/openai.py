"""OpenAI-backed LLM recommendation provider (development plan §4.1).

Calls the OpenAI Responses API (``POST /v1/responses``) over raw httpx with
structured outputs (``text.format`` with a strict JSON schema), so the model is
constrained to emit the exact plan §4.1 recommendation schema. The model id and
API key come from configuration (``settings.llm_model`` / ``settings.llm_api_key``)
— never hardcoded here.

Failure policy (identical to the Anthropic provider — the two are
interchangeable by configuration alone):
  - Missing API key -> ProviderError at construction time. The registry in
    ``libs/llm/__init__`` only builds this class from settings, so it is NEVER
    used when ``llm_api_key`` is empty (an unconfigured install produces no
    recommendations at all).
  - Network / HTTP-level failures -> ProviderError (clear, actionable).
  - Malformed MODEL OUTPUT never raises: bad entries are logged and skipped, a
    refusal or unparseable body yields an empty list. LLM output is an
    information feature; a bad generation must degrade to "no candidates",
    never crash a request path.

SAFETY (plan §4.1, §44 rule 5, §46): drafts returned here are information
only — they carry zero execution authority and never touch the Watchlist,
Trading Pool, or orders.
"""
import json
import logging
import time
from datetime import datetime

import httpx

from .event_analysis import (
    EVENT_ANALYSIS_SCHEMA,
    EVENT_ANALYSIS_SCHEMA_NAME,
    PROMPT_VERSION,
    SYSTEM_PROMPT as EVENT_SYSTEM_PROMPT,
    EventAnalysisResult,
    build_user_message,
)
from .provider import (
    GroundingArticle,
    ProviderError,
    RecommendationDraft,
    language_instruction,
)
from .market_selection import (
    MarketSelectionResult,
    build_selection_prompt,
    parse_selection,
)
from .retry import post_json_with_retry

#: A selection reply is a handful of refs and one-sentence reasons — small by
#: construction. Bounding it keeps a runaway generation cheap.
SELECTION_MAX_OUTPUT_TOKENS = 800

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Read timeout for :meth:`OpenAIRecommendationProvider.analyze_event` ONLY.
#: The event-analysis prompt carries the whole §46 evidence bundle and asks
#: for a long structured note; a live gpt-5.6-sol run took 51s, and at the
#: 60s discovery timeout the NEXT one hit ``httpx.ReadTimeout`` and was
#: stored as a FAILED analysis having already paid for the inference. The
#: discovery calls keep the shorter budget: a hung recommendations refresh
#: should not hold a request open for four minutes.
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 240.0

#: Output-token ceiling for :meth:`analyze_event` ONLY. The §48 schema has
#: eighteen fields including three scenarios and a numbers_quoted list the
#: model is now required to fill with at least three citations; 4096 truncates
#: that note mid-JSON, which arrives as an unparseable body and a FAILED row.
DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS = 8000

# JSON schema for the structured output. OpenAI strict mode requires every
# property to be listed in "required" and "additionalProperties": false on every
# object; it does not support numeric min/max, so range enforcement happens in
# RecommendationDraft.__post_init__ (shared with every other provider).
_EVIDENCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "published_at": {"type": "string"},
        "snippet": {"type": "string"},
    },
    "required": ["source", "published_at", "snippet"],
    "additionalProperties": False,
}

_RECOMMENDATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "company": {"type": ["string", "null"]},
        "sentiment": {"type": "number"},
        "impact": {"type": "number"},
        "novelty": {"type": "number"},
        "source_reliability": {"type": "number"},
        "horizon": {"type": "string"},
        "catalyst_type": {"type": "string"},
        "reason_codes": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
    },
    "required": [
        "ticker", "company", "sentiment", "impact", "novelty",
        "source_reliability", "horizon", "catalyst_type", "reason_codes",
        "summary", "evidence",
    ],
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "recommendations": {"type": "array", "items": _RECOMMENDATION_SCHEMA},
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are the candidate-discovery engine of a systematic options trading "
    "platform (plan §4.1). Propose liquid, optionable U.S. large-cap equities "
    "that currently have a fresh, identifiable catalyst (earnings, guidance, "
    "product, regulatory, clinical, macro).\n"
    "Rules:\n"
    "- sentiment is in [-1, 1]; impact, novelty and source_reliability are in [0, 1].\n"
    "- Every evidence item must cite material published STRICTLY BEFORE the "
    "provided as-of time (news timestamp integrity, plan §20.3), with an "
    "ISO-8601 published_at.\n"
    "- Never propose any ticker in the provided exclusion list.\n"
    "- Your output is information for a human curator. It is NOT a trade "
    "signal and will never place orders."
)



_ENRICH_SYSTEM_PROMPT = (
    "You are the news-enrichment engine of a systematic options trading "
    "platform (plan §4.1 / Phase 8). You are given a numbered list of REAL "
    "news articles. They are your ONLY source of information for this task.\n"
    "Rules:\n"
    "- Propose candidates ONLY where the given articles contain a fresh, "
    "identifiable catalyst. If no article supports a candidate, return fewer "
    "or zero recommendations — never invent one.\n"
    "- Every evidence item's \"source\" field MUST be the article_url of one "
    "of the given articles, copied EXACTLY, with that article's published_at. "
    "The snippet must paraphrase THAT article only.\n"
    "- A recommendation's ticker must appear in the tickers list of at least "
    "one article it cites.\n"
    "- sentiment in [-1, 1]; impact, novelty, source_reliability in [0, 1].\n"
    "- Never propose any ticker in the provided exclusion list.\n"
    "- Your output is information for a human curator. It is NOT a trade "
    "signal and will never place orders."
)


def _format_articles(articles: list[GroundingArticle]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        tickers = ",".join(a.tickers) or "(none listed)"
        lines.append(
            f"[{i}] article_url: {a.url}\n"
            f"    title: {a.title}\n"
            f"    publisher: {a.publisher or '(unknown)'}\n"
            f"    published_at: {a.published_at}\n"
            f"    tickers: {tickers}\n"
            f"    description: {a.description or '(none)'}"
        )
    return "\n".join(lines)


def _usage_from(data: object) -> dict | None:
    """Token usage from a Responses body, or None when the API omitted it.

    Honest None: a missing usage block means "not reported", and recording
    zeros there would make a real call look free in the cost ledger.
    """
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    out: dict = {}
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = int(value)
    return out or None


def _has_refusal(data: object) -> bool:
    """True when the Responses body carries a refusal content part."""
    if not isinstance(data, dict):
        return False
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "refusal":
                return True
    return False


class OpenAIRecommendationProvider:
    """RecommendationProvider backed by the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = OPENAI_API_URL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        output_language: str = "en",
        analysis_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        analysis_max_output_tokens: int = DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS,
    ) -> None:
        """`transport` is injectable so tests can mock the network (httpx.MockTransport)."""
        if not api_key:
            raise ProviderError(
                "OpenAIRecommendationProvider requires a non-empty API key "
                "(settings.llm_api_key). Leave LLM_PROVIDER empty when no key "
                "is configured — the platform then produces no recommendations."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        # analyze_event ONLY (see the module constants): the analysis call is
        # a different shape of request from the discovery calls and gets its
        # own budget rather than stretching theirs.
        self.analysis_timeout_seconds = analysis_timeout_seconds
        self.analysis_max_output_tokens = max(
            analysis_max_output_tokens, max_output_tokens
        )
        self._transport = transport
        # Narrative-fields-only language addendum (see provider.py); "" for en.
        self._language_instruction = language_instruction(output_language)

    def generate(
        self,
        exclude_tickers: set[str],
        as_of: datetime,
        limit: int = 5,
    ) -> list[RecommendationDraft]:
        """Ask the model for up to `limit` drafts; parse+validate defensively."""
        payload = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "instructions": _SYSTEM_PROMPT + self._language_instruction,
            "input": (
                f"As-of time: {as_of.isoformat()}\n"
                f"Excluded tickers (never propose these): "
                f"{sorted(exclude_tickers) if exclude_tickers else 'none'}\n"
                f"Propose at most {limit} recommendations."
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "recommendations",
                    "strict": True,
                    "schema": _OUTPUT_SCHEMA,
                }
            },
        }
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI API request failed: {exc!r}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"OpenAI API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()

        text = self._extract_text(data)
        if not text:
            logger.warning("OpenAI response contained no output text; returning no drafts")
            return []

        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            logger.warning("OpenAI response text was not valid JSON; returning no drafts")
            return []

        raw_entries = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(raw_entries, list):
            logger.warning("OpenAI response missing 'recommendations' list; returning no drafts")
            return []

        drafts: list[RecommendationDraft] = []
        for entry in raw_entries:
            draft = self._parse_entry(entry)
            if draft is None:
                continue
            # Defence in depth: never surface an excluded ticker even if the
            # model ignores the instruction.
            if draft.ticker in exclude_tickers:
                logger.warning("Dropping draft for excluded ticker %s", draft.ticker)
                continue
            drafts.append(draft)
            if len(drafts) >= limit:
                break
        return drafts


    def enrich(
        self,
        articles: list[GroundingArticle],
        exclude_tickers: set[str],
        as_of: datetime,
        limit: int = 5,
    ) -> list[RecommendationDraft]:
        """Drafts grounded EXCLUSIVELY in `articles` (Phase 8).

        Same wire contract as :meth:`generate`, but the input is the real
        article batch and the instructions forbid outside knowledge. The
        ROUTER re-validates grounding (evidence urls ⊆ batch urls, ticker ∈
        cited articles' tickers) and drops violations — this method's own
        filtering is defence in depth, not the safety boundary.
        """
        if not articles:
            return []
        payload = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "instructions": _ENRICH_SYSTEM_PROMPT + self._language_instruction,
            "input": (
                f"As-of time: {as_of.isoformat()}\n"
                f"Excluded tickers (never propose these): "
                f"{sorted(exclude_tickers) if exclude_tickers else 'none'}\n"
                f"Propose at most {limit} recommendations.\n\n"
                f"ARTICLES (your only information source):\n"
                f"{_format_articles(articles)}"
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "recommendations",
                    "strict": True,
                    "schema": _OUTPUT_SCHEMA,
                }
            },
        }
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI API request failed: {exc!r}") from exc
        if response.status_code != 200:
            raise ProviderError(
                f"OpenAI API returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        text = self._extract_text(response.json())
        if not text:
            logger.warning("OpenAI enrich response had no output text")
            return []
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            logger.warning("OpenAI enrich response was not valid JSON")
            return []
        raw_entries = (
            parsed.get("recommendations") if isinstance(parsed, dict) else None
        )
        if not isinstance(raw_entries, list):
            return []
        known_urls = {a.url for a in articles}
        drafts: list[RecommendationDraft] = []
        for entry in raw_entries:
            draft = self._parse_entry(entry)
            if draft is None:
                continue
            if draft.ticker in exclude_tickers:
                logger.warning("Dropping enrich draft for excluded ticker %s", draft.ticker)
                continue
            if not draft.evidence or not all(
                e.get("source") in known_urls for e in draft.evidence
            ):
                logger.warning(
                    "Dropping enrich draft for %s: evidence not grounded in "
                    "the provided articles",
                    draft.ticker,
                )
                continue
            drafts.append(draft)
            if len(drafts) >= limit:
                break
        return drafts

    def select_prediction_market_events(
        self,
        *,
        event_type: str,
        event_title: str,
        scheduled_at: str,
        options,
        as_of: datetime,
    ) -> MarketSelectionResult:
        """Which VENUE EVENTS to read for one catalyst (market-selection-v1).

        A NARROWING call, not an admitting one. The reply can only name refs
        the caller minted, and the caller re-resolves every one against the
        pool it supplied — so nothing this model returns can introduce a
        market, a price or an id into the pipeline.

        Unlike ``analyze_event`` this degrades rather than raises on a bad
        reply: the deterministic matcher behind it is a complete answer, so a
        malformed selection costs only itself.
        """
        system, user = build_selection_prompt(
            event_type=event_type,
            event_title=event_title,
            scheduled_at=scheduled_at,
            options=options,
        )
        payload = {
            "model": self.model,
            "max_output_tokens": SELECTION_MAX_OUTPUT_TOKENS,
            "instructions": system,
            "input": f"As-of time: {as_of.isoformat()}\n\n{user}",
        }
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        response = post_json_with_retry(
            self.base_url,
            payload=payload,
            headers=headers,
            timeout_seconds=self.analysis_timeout_seconds,
            transport=self._transport,
            provider_name="OpenAI",
        )
        if response.status_code != 200:
            raise ProviderError(
                f"OpenAI API returned HTTP {response.status_code} for market selection"
            )
        data = response.json()
        if _has_refusal(data):
            raise ProviderError("refused")
        text = self._extract_text(data) or ""
        # parse_selection tolerates a raw string and degrades to "selected
        # nothing" on anything unparseable — a bad reply must not sink a
        # backfill the deterministic matcher can complete alone.
        return parse_selection(text, allowed_refs=[o.ref for o in options])

    def analyze_event(
        self,
        bundle_json: dict,
        *,
        as_of: datetime,
    ) -> EventAnalysisResult:
        """Pre-event research note for one catalyst (§46-§52).

        Same wire mechanics as :meth:`generate` — Responses API, strict
        json_schema — with the §48 analysis schema. Unlike the discovery
        calls there is no graceful "zero results" degradation here: the caller
        asked about ONE event, so a refusal, an empty body or unparseable JSON
        raises :class:`ProviderError` and the gateway records a FAILED
        analysis alongside the bundle it did have.

        The returned result is UNVALIDATED: ``violations`` is empty and the
        caller must run ``event_analysis.validate_analysis`` against the
        bundle's fact index before serving it (§47).
        """
        payload = {
            "model": self.model,
            "max_output_tokens": self.analysis_max_output_tokens,
            "instructions": EVENT_SYSTEM_PROMPT + self._language_instruction,
            "input": (
                f"As-of time: {as_of.isoformat()}\n\n"
                + build_user_message(bundle_json)
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": EVENT_ANALYSIS_SCHEMA_NAME,
                    "strict": True,
                    "schema": EVENT_ANALYSIS_SCHEMA,
                }
            },
        }
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        started = time.monotonic()
        # Phase 19.2: ONE bounded retry on transport failure / 429 / 5xx —
        # analysis requests carry the whole bundle, so a transient hiccup
        # should not discard the assembly. Discovery calls stay fail-fast.
        response = post_json_with_retry(
            self.base_url,
            payload=payload,
            headers=headers,
            timeout_seconds=self.analysis_timeout_seconds,
            transport=self._transport,
            provider_name="OpenAI",
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        if response.status_code != 200:
            raise ProviderError(
                f"OpenAI API returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        if _has_refusal(data):
            raise ProviderError("refused")
        text = self._extract_text(data)
        if not text:
            raise ProviderError("OpenAI event analysis response had no output text")
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                f"OpenAI event analysis response was not valid JSON: {exc!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ProviderError("OpenAI event analysis response was not a JSON object")

        return EventAnalysisResult(
            analysis=parsed,
            model=self.model,
            provider="openai",
            prompt_version=PROMPT_VERSION,
            usage=_usage_from(data),
            latency_ms=latency_ms,
            violations=[],
        )

    @staticmethod
    def _extract_text(data: object) -> str | None:
        """Pull the assistant's text out of a Responses API body.

        Prefers the ``output_text`` convenience field when the API supplies it,
        otherwise walks ``output[].content[]`` for the first ``output_text``
        part. A refusal part yields no text, so the caller degrades to "no
        candidates" rather than raising.
        """
        if not isinstance(data, dict):
            return None

        convenience = data.get("output_text")
        if isinstance(convenience, str) and convenience.strip():
            return convenience
        # Some SDK-shaped bodies expose output_text as a list of strings.
        if isinstance(convenience, list):
            joined = "".join(part for part in convenience if isinstance(part, str))
            if joined.strip():
                return joined

        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "refusal":
                    logger.warning(
                        "OpenAI recommendation request refused; returning no drafts"
                    )
                    return None
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
        return None

    @staticmethod
    def _parse_entry(entry: object) -> RecommendationDraft | None:
        """Validate one raw model entry; malformed entries are logged and dropped."""
        if not isinstance(entry, dict):
            logger.warning("Dropping malformed recommendation entry (not an object)")
            return None
        try:
            company = entry.get("company")
            return RecommendationDraft(
                ticker=str(entry["ticker"]),
                company=str(company) if company is not None else None,
                sentiment=float(entry["sentiment"]),
                impact=float(entry["impact"]),
                novelty=float(entry["novelty"]),
                source_reliability=float(entry["source_reliability"]),
                horizon=str(entry["horizon"]),
                catalyst_type=str(entry["catalyst_type"]),
                reason_codes=[str(code) for code in entry["reason_codes"]],
                summary=str(entry["summary"]),
                evidence=[dict(item) for item in entry["evidence"]],
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Dropping malformed recommendation entry: %s", exc)
            return None
