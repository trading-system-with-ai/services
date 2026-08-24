"""Anthropic-backed LLM recommendation provider (development plan §4.1).

Calls the Anthropic Messages API (``POST /v1/messages``) over raw httpx with
structured outputs (``output_config.format`` with a JSON schema), so the model
is constrained to emit the exact plan §4.1 recommendation schema. The model id
and API key come from configuration (``settings.llm_model`` /
``settings.llm_api_key``) — never hardcoded here.

Failure policy:
  - Missing API key -> ProviderError at construction time. The registry in
    ``libs/llm/__init__`` only builds this class from settings, so this class
    is NEVER used when ``llm_api_key`` is empty (the safe default provider is
    the stub).
  - Network / HTTP-level failures -> ProviderError (clear, actionable).
  - Malformed MODEL OUTPUT never raises: bad entries are logged and skipped,
    a refusal or unparseable body yields an empty list. LLM output is an
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
    PROMPT_VERSION,
    SYSTEM_PROMPT as EVENT_SYSTEM_PROMPT,
    EventAnalysisResult,
    build_user_message,
)
from .provider import ProviderError, RecommendationDraft, language_instruction
from .market_selection import (
    MarketSelectionResult,
    build_selection_prompt,
    parse_selection,
)
from .retry import post_json_with_retry

#: A selection reply is a handful of refs and one-sentence reasons.
SELECTION_MAX_TOKENS = 800

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Read timeout for :meth:`AnthropicRecommendationProvider.analyze_event`
#: ONLY — same reasoning as the OpenAI adapter's constant of the same name:
#: the analysis prompt carries the whole §46 bundle and asks for a long
#: structured note, and a request that times out at 60s is a FAILED row that
#: already paid for the inference. Discovery calls keep the shorter budget.
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 240.0

#: Output-token ceiling for :meth:`analyze_event` ONLY. 4096 truncates the
#: §48 note mid-JSON — an unparseable body and a FAILED row.
DEFAULT_ANALYSIS_MAX_TOKENS = 8000

# JSON schema for the structured output ("recommendations" wrapper around the
# plan §4.1 draft schema). Structured outputs do not support numeric min/max
# constraints, so range enforcement happens in RecommendationDraft.__post_init__.
_EVIDENCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "published_at": {"type": "string", "format": "date-time"},
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


def _usage_from(data: object) -> dict | None:
    """Token usage from a Messages body, or None when the API omitted it.

    Honest None: a missing usage block means "not reported"; zeros there would
    make a real call look free in the cost ledger.
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


class AnthropicRecommendationProvider:
    """RecommendationProvider backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = ANTHROPIC_API_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        output_language: str = "en",
        analysis_timeout_seconds: float = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        analysis_max_tokens: int = DEFAULT_ANALYSIS_MAX_TOKENS,
    ) -> None:
        """`transport` is injectable so tests can mock the network (httpx.MockTransport)."""
        if not api_key:
            raise ProviderError(
                "AnthropicRecommendationProvider requires a non-empty API key "
                "(settings.llm_api_key). Use the 'stub' provider when no key is configured."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        # analyze_event ONLY (see the module constants).
        self.analysis_timeout_seconds = analysis_timeout_seconds
        self.analysis_max_tokens = max(analysis_max_tokens, max_tokens)
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
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT + self._language_instruction,
            "output_config": {"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"As-of time: {as_of.isoformat()}\n"
                        f"Excluded tickers (never propose these): "
                        f"{sorted(exclude_tickers) if exclude_tickers else 'none'}\n"
                        f"Propose at most {limit} recommendations."
                    ),
                }
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic API request failed: {exc!r}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Anthropic API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()

        # Safety classifiers can decline with a normal 200 + stop_reason
        # "refusal" — that is model output, not a transport failure: log, skip.
        if data.get("stop_reason") == "refusal":
            logger.warning("Anthropic recommendation request refused; returning no drafts")
            return []

        text = next(
            (b.get("text") for b in data.get("content", []) if b.get("type") == "text"),
            None,
        )
        if not text:
            logger.warning("Anthropic response contained no text block; returning no drafts")
            return []

        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            logger.warning("Anthropic response text was not valid JSON; returning no drafts")
            return []

        raw_entries = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(raw_entries, list):
            logger.warning("Anthropic response missing 'recommendations' list; returning no drafts")
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

        A NARROWING call. The reply may only name refs the caller minted, and
        the caller re-resolves each against the pool it supplied — nothing
        returned here can introduce a market, a price or an id.

        Degrades rather than raises on a bad reply: the deterministic matcher
        behind it is a complete answer on its own.
        """
        system, user = build_selection_prompt(
            event_type=event_type,
            event_title=event_title,
            scheduled_at=scheduled_at,
            options=options,
        )
        payload = {
            "model": self.model,
            "max_tokens": SELECTION_MAX_TOKENS,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": f"As-of time: {as_of.isoformat()}\n\n{user}",
                }
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        response = post_json_with_retry(
            self.base_url,
            payload=payload,
            headers=headers,
            timeout_seconds=self.analysis_timeout_seconds,
            transport=self._transport,
            provider_name="Anthropic",
        )
        if response.status_code != 200:
            raise ProviderError(
                f"Anthropic API returned HTTP {response.status_code} for market selection"
            )
        data = response.json()
        if data.get("stop_reason") == "refusal":
            raise ProviderError("refused")
        text = next(
            (
                b.get("text")
                for b in data.get("content", []) or []
                if isinstance(b, dict) and b.get("type") == "text"
            ),
            "",
        )
        return parse_selection(text or "", allowed_refs=[o.ref for o in options])

    def analyze_event(
        self,
        bundle_json: dict,
        *,
        as_of: datetime,
    ) -> EventAnalysisResult:
        """Pre-event research note for one catalyst (§46-§52).

        Messages API with ``output_config.format`` json_schema, mirroring
        :meth:`generate`'s wire contract with the §48 analysis schema. A
        refusal, an empty body or unparseable JSON raises
        :class:`ProviderError` — for a single-event note there is no
        "fewer results" degradation, and the gateway records FAILED while
        still serving the bundle.

        The returned result is UNVALIDATED (``violations`` empty); the caller
        runs ``event_analysis.validate_analysis`` against the bundle's fact
        index (§47).
        """
        payload = {
            "model": self.model,
            "max_tokens": self.analysis_max_tokens,
            "system": EVENT_SYSTEM_PROMPT + self._language_instruction,
            "output_config": {
                "format": {"type": "json_schema", "schema": EVENT_ANALYSIS_SCHEMA}
            },
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"As-of time: {as_of.isoformat()}\n\n"
                        + build_user_message(bundle_json)
                    ),
                }
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
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
            provider_name="Anthropic",
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        if response.status_code != 200:
            raise ProviderError(
                f"Anthropic API returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        if data.get("stop_reason") == "refusal":
            raise ProviderError("refused")

        text = next(
            (
                b.get("text")
                for b in data.get("content", []) or []
                if isinstance(b, dict) and b.get("type") == "text"
            ),
            None,
        )
        if not text:
            raise ProviderError("Anthropic event analysis response had no text block")
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                f"Anthropic event analysis response was not valid JSON: {exc!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ProviderError(
                "Anthropic event analysis response was not a JSON object"
            )

        return EventAnalysisResult(
            analysis=parsed,
            model=self.model,
            provider="anthropic",
            prompt_version=PROMPT_VERSION,
            usage=_usage_from(data),
            latency_ms=latency_ms,
            violations=[],
        )

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
