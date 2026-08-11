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
from datetime import datetime

import httpx

from .provider import ProviderError, RecommendationDraft

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60.0

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
        self._transport = transport

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
            "instructions": _SYSTEM_PROMPT,
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
