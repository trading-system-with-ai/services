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

from datetime import datetime

import httpx

from .provider import ProviderError, RecommendationDraft

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60.0

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
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
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
