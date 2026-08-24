"""Tests for the LLM recommendation provider abstraction (plan §4.1, §20.3).

Covers: stub determinism, exclusion, evidence timestamp integrity, draft
range validation, the provider registry, and the Anthropic provider against a
mocked httpx transport (no network is ever touched).
"""
import json
from datetime import datetime, timezone

import httpx
import pytest

from libs.llm import (
    ProviderError,
    RecommendationDraft,
    StubRecommendationProvider,
    get_recommendation_provider,
)
from libs.llm.anthropic import ANTHROPIC_VERSION, AnthropicRecommendationProvider

AS_OF = datetime(2026, 8, 10, 14, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------

def test_stub_same_as_of_yields_identical_drafts():
    provider = StubRecommendationProvider()
    first = provider.generate(set(), AS_OF, limit=5)
    second = provider.generate(set(), AS_OF, limit=5)
    assert first == second
    assert len(first) == 5


def test_stub_same_day_different_time_yields_identical_drafts():
    # Determinism is per calendar day (seeded by as_of.date()).
    provider = StubRecommendationProvider()
    morning = provider.generate(set(), AS_OF.replace(hour=9, minute=5), limit=5)
    evening = provider.generate(set(), AS_OF.replace(hour=21, minute=45), limit=5)
    assert morning == evening


def test_stub_different_day_rotates_drafts():
    provider = StubRecommendationProvider()
    today = provider.generate(set(), AS_OF, limit=5)
    other_day = provider.generate(set(), datetime(2026, 8, 11, 14, 30, tzinfo=timezone.utc), limit=5)
    assert today != other_day


def test_stub_honors_exclusions():
    provider = StubRecommendationProvider()
    baseline = provider.generate(set(), AS_OF, limit=5)
    excluded = {d.ticker for d in baseline[:3]}
    drafts = provider.generate(excluded, AS_OF, limit=5)
    assert not excluded.intersection({d.ticker for d in drafts})
    # Exclusion re-fills the slots from the rest of the universe.
    assert len(drafts) == 5


def test_stub_evidence_strictly_before_as_of():
    # Plan §20.3: every citation must predate the as-of time strictly.
    provider = StubRecommendationProvider()
    for as_of in (AS_OF, AS_OF.replace(hour=0, minute=0)):  # midnight is the tight case
        for draft in provider.generate(set(), as_of, limit=5):
            assert draft.evidence, "every draft carries evidence"
            for item in draft.evidence:
                assert set(item) == {"source", "published_at", "snippet"}
                assert datetime.fromisoformat(item["published_at"]) < as_of


def test_stub_scores_within_ranges_and_limit():
    provider = StubRecommendationProvider()
    drafts = provider.generate(set(), AS_OF, limit=3)
    assert len(drafts) == 3
    for d in drafts:
        assert -1.0 <= d.sentiment <= 1.0
        assert 0.0 <= d.impact <= 1.0
        assert 0.0 <= d.novelty <= 1.0
        assert 0.0 <= d.source_reliability <= 1.0
        assert d.company and d.horizon and d.catalyst_type and d.summary
        assert d.reason_codes


# ---------------------------------------------------------------------------
# RecommendationDraft validation
# ---------------------------------------------------------------------------

def _draft_kwargs(**overrides):
    kwargs = dict(
        ticker="AAPL",
        company="Apple Inc.",
        sentiment=0.5,
        impact=0.5,
        novelty=0.5,
        source_reliability=0.5,
        horizon="1-2 weeks",
        catalyst_type="earnings_surprise",
        reason_codes=["EARNINGS_BEAT"],
        summary="s",
        evidence=[],
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"sentiment": 1.5},
        {"sentiment": -1.0001},
        {"impact": -0.1},
        {"impact": 1.1},
        {"novelty": 2.0},
        {"source_reliability": -0.5},
        {"ticker": ""},
    ],
)
def test_draft_range_validation_raises(overrides):
    with pytest.raises(ValueError):
        RecommendationDraft(**_draft_kwargs(**overrides))


def test_draft_boundary_values_accepted():
    RecommendationDraft(**_draft_kwargs(sentiment=-1.0, impact=0.0, novelty=1.0, source_reliability=1.0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_no_default_provider():
    # THERE IS NO DEFAULT (§44 rule 18). The Settings field defaults to "" —
    # unset — so an unconfigured install can never be served
    # template-generated drafts that read like real research.
    from libs.common.config import Settings

    assert Settings.model_fields["llm_provider"].default == ""


def test_registry_stub_is_opt_in_and_still_available():
    # The stub stays in the registry as an explicitly opt-in development/test
    # provider: asking for it by name still yields the stub implementation.
    provider = get_recommendation_provider("stub")
    assert isinstance(provider, StubRecommendationProvider)


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_registry_blank_name_raises_not_configured(name):
    # Blank name = the unconfigured state, and it names the missing setting.
    from libs.llm import LLMProviderNotConfigured

    with pytest.raises(LLMProviderNotConfigured, match="LLM_PROVIDER"):
        get_recommendation_provider(name)


def test_not_configured_is_a_provider_error_not_a_value_error():
    # LLMProviderNotConfigured subclasses ProviderError (callers that already
    # handle provider failure keep working) and is NOT a ValueError — an
    # unknown NAME is an operator typo, absence of config is a different fact.
    from libs.llm import LLMProviderNotConfigured

    assert issubclass(LLMProviderNotConfigured, ProviderError)
    assert not issubclass(LLMProviderNotConfigured, ValueError)


def test_registry_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_recommendation_provider("nope")


# ---------------------------------------------------------------------------
# Anthropic provider (mocked httpx transport — never touches the network)
# ---------------------------------------------------------------------------

VALID_ENTRY = {
    "ticker": "AAPL",
    "company": "Apple Inc.",
    "sentiment": 0.6,
    "impact": 0.7,
    "novelty": 0.5,
    "source_reliability": 0.8,
    "horizon": "1-2 weeks",
    "catalyst_type": "earnings_surprise",
    "reason_codes": ["EARNINGS_BEAT"],
    "summary": "Cloud momentum sets up a beat.",
    "evidence": [
        {
            "source": "Reuters",
            "published_at": "2026-08-09T12:00:00+00:00",
            "snippet": "Checks point to upside.",
        }
    ],
}


def _api_response(recommendations, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "stop_reason": stop_reason,
        "content": [
            {"type": "text", "text": json.dumps({"recommendations": recommendations})}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _provider_with(handler):
    return AnthropicRecommendationProvider(
        api_key="test-key",
        model="claude-sonnet-5",
        transport=httpx.MockTransport(handler),
    )


def test_anthropic_parses_valid_response_and_sends_expected_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_api_response([VALID_ENTRY]))

    drafts = _provider_with(handler).generate({"TSLA"}, AS_OF, limit=5)

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.ticker == "AAPL"
    assert draft.sentiment == pytest.approx(0.6)
    assert draft.evidence[0]["source"] == "Reuters"

    # Request shape per the Anthropic Messages API.
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert captured["body"]["model"] == "claude-sonnet-5"
    assert captured["body"]["output_config"]["format"]["type"] == "json_schema"
    assert "TSLA" in captured["body"]["messages"][0]["content"]


def test_anthropic_drops_malformed_entries_without_raising():
    malformed = [
        {**VALID_ENTRY, "ticker": "MSFT", "sentiment": 3.0},  # out of range
        {k: v for k, v in VALID_ENTRY.items() if k != "ticker"},  # missing field
        "not-an-object",
    ]

    def handler(request):
        return httpx.Response(200, json=_api_response([VALID_ENTRY] + malformed))

    drafts = _provider_with(handler).generate(set(), AS_OF)
    assert [d.ticker for d in drafts] == ["AAPL"]


def test_anthropic_drops_excluded_tickers_defensively():
    def handler(request):
        return httpx.Response(200, json=_api_response([VALID_ENTRY]))

    drafts = _provider_with(handler).generate({"AAPL"}, AS_OF)
    assert drafts == []


def test_anthropic_refusal_returns_empty_list():
    def handler(request):
        body = _api_response([], stop_reason="refusal")
        body["content"] = []
        return httpx.Response(200, json=body)

    assert _provider_with(handler).generate(set(), AS_OF) == []


def test_anthropic_network_error_raises_provider_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderError, match="request failed"):
        _provider_with(handler).generate(set(), AS_OF)


def test_anthropic_http_error_status_raises_provider_error():
    def handler(request):
        return httpx.Response(500, json={"type": "error", "error": {"type": "api_error"}})

    with pytest.raises(ProviderError, match="HTTP 500"):
        _provider_with(handler).generate(set(), AS_OF)


def test_anthropic_missing_api_key_raises():
    # The registry only constructs this provider from settings, so an empty
    # llm_api_key can never reach the network (plan §4.1 safe default).
    with pytest.raises(ProviderError, match="API key"):
        AnthropicRecommendationProvider(api_key="", model="claude-sonnet-5")


# ---------------------------------------------------------------------------
# OpenAI provider (mocked httpx transport — never touches the network)
#
# Mirrors the Anthropic suite: the two providers are interchangeable by
# configuration alone, so they must fail and degrade identically.
# ---------------------------------------------------------------------------

OPENAI_MODEL = "gpt-5.6-sol"


def _openai_response(recommendations):
    """A Responses API body carrying the structured-output JSON."""
    return {
        "id": "resp_test",
        "object": "response",
        "model": OPENAI_MODEL,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"recommendations": recommendations}),
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _openai_provider_with(handler):
    from libs.llm.openai import OpenAIRecommendationProvider

    return OpenAIRecommendationProvider(
        api_key="test-key",
        model=OPENAI_MODEL,
        transport=httpx.MockTransport(handler),
    )


def test_openai_parses_valid_response_and_sends_expected_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_response([VALID_ENTRY]))

    drafts = _openai_provider_with(handler).generate({"TSLA"}, AS_OF, limit=5)

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.ticker == "AAPL"
    assert draft.sentiment == pytest.approx(0.6)
    assert draft.evidence[0]["source"] == "Reuters"

    # Request shape per the OpenAI Responses API.
    assert captured["url"].endswith("/v1/responses")
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == OPENAI_MODEL
    fmt = captured["body"]["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert "TSLA" in captured["body"]["input"]


def test_openai_reads_output_text_convenience_field():
    def handler(request):
        body = _openai_response([VALID_ENTRY])
        body["output_text"] = json.dumps({"recommendations": [VALID_ENTRY]})
        return httpx.Response(200, json=body)

    assert [d.ticker for d in _openai_provider_with(handler).generate(set(), AS_OF)] == ["AAPL"]


def test_openai_drops_malformed_entries_without_raising():
    malformed = [
        {**VALID_ENTRY, "ticker": "MSFT", "sentiment": 3.0},  # out of range
        {k: v for k, v in VALID_ENTRY.items() if k != "ticker"},  # missing field
        "not-an-object",
    ]

    def handler(request):
        return httpx.Response(200, json=_openai_response([VALID_ENTRY] + malformed))

    drafts = _openai_provider_with(handler).generate(set(), AS_OF)
    assert [d.ticker for d in drafts] == ["AAPL"]


def test_openai_drops_excluded_tickers_defensively():
    def handler(request):
        return httpx.Response(200, json=_openai_response([VALID_ENTRY]))

    assert _openai_provider_with(handler).generate({"AAPL"}, AS_OF) == []


def test_openai_refusal_returns_empty_list():
    def handler(request):
        body = _openai_response([])
        body["output"] = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I can't help with that."}],
            }
        ]
        return httpx.Response(200, json=body)

    assert _openai_provider_with(handler).generate(set(), AS_OF) == []


def test_openai_unparseable_text_returns_empty_list():
    def handler(request):
        body = _openai_response([])
        body["output"][0]["content"][0]["text"] = "not json at all"
        return httpx.Response(200, json=body)

    assert _openai_provider_with(handler).generate(set(), AS_OF) == []


def test_openai_network_error_raises_provider_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderError, match="request failed"):
        _openai_provider_with(handler).generate(set(), AS_OF)


def test_openai_http_error_status_raises_provider_error():
    def handler(request):
        return httpx.Response(500, json={"error": {"message": "server error"}})

    with pytest.raises(ProviderError, match="HTTP 500"):
        _openai_provider_with(handler).generate(set(), AS_OF)


def test_openai_missing_api_key_raises():
    # The registry only constructs this provider from settings, so an empty
    # llm_api_key can never reach the network (plan §4.1 safe default).
    from libs.llm.openai import OpenAIRecommendationProvider

    with pytest.raises(ProviderError, match="API key"):
        OpenAIRecommendationProvider(api_key="", model=OPENAI_MODEL)


def test_registry_knows_openai_and_still_has_no_default(monkeypatch):
    """"openai" resolves; naming nothing is still the unconfigured state.

    The key is pinned to "" here rather than relying on the ambient
    environment: a developer machine with a real LLM_API_KEY in .env would
    otherwise construct the provider successfully and silently invert what
    this test claims to prove.
    """
    from libs.common.config import get_settings
    from libs.llm import LLMProviderNotConfigured

    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()
    try:
        # Registered by name — and keyless construction is refused, so the
        # OpenAI provider can never reach the network without a key.
        with pytest.raises(ProviderError, match="API key"):
            get_recommendation_provider("openai")
        # Naming nothing is the unconfigured state, not a fallback.
        with pytest.raises(LLMProviderNotConfigured):
            get_recommendation_provider("")
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# The analysis call gets its OWN timeout (Settings.llm_analysis_timeout_seconds)
#
# THE LIVE FAILURE THIS PINS. A gpt-5.6-sol event analysis took 51 seconds to
# answer; the next one hit httpx.ReadTimeout at the shared 60s budget and was
# stored as a FAILED analysis having already paid for the inference. The
# analysis prompt carries the whole evidence bundle and asks for a long
# structured note, so it is simply a different shape of request from the
# discovery calls — and raising THEIR timeout instead would let a hung
# recommendations refresh hold a request open for four minutes.
# ---------------------------------------------------------------------------


def _record_client_timeouts(monkeypatch) -> list:
    """Record the ``timeout=`` every ``httpx.Client`` in libs.llm is built with.

    Patched at the constructor rather than sniffed off the transport because
    the timeout is a property of the CLIENT — a transport-level assertion
    would keep passing if the adapter stopped passing a timeout at all, which
    is the exact regression this guards.
    """
    seen: list = []
    real_client = httpx.Client

    class RecordingClient(real_client):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("timeout"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", RecordingClient)
    return seen


def _analysis_body() -> dict:
    """A schema-complete analysis body — the adapters parse before returning,
    so a stub that only satisfied the timeout assertion would raise first."""
    scenario = {
        "conditions": "c", "guidance_conditions": "g",
        "why_market_reacts": "w", "evidence_refs": [],
    }
    return {
        "executive_summary": "s", "what_happened_last_time": "s",
        "what_changed_since": "s", "fundamental_developments": "s",
        "price_and_positioning": "s", "market_expectations": "s",
        "key_positive_catalysts": [], "key_negative_catalysts": [],
        "what_matters_most": "s",
        "scenarios": {"upside": scenario, "base": scenario, "downside": scenario},
        "surprise_threshold": {"narrative": "n", "confidence": "NOT_MEANINGFUL"},
        "key_unknowns": [], "invalidation": "i",
        "expectations_gap_regime": "INSUFFICIENT_DATA", "confidence": "LOW",
        "evidence_refs": [], "numbers_quoted": [],
    }


_TIMEOUT_BUNDLE = {"event": {"ticker": "AAPL"}, "fundamentals": {"revenue": 1.0}}


def test_openai_analyze_event_uses_the_analysis_timeout_not_the_shared_one(
    monkeypatch,
):
    from libs.llm.openai import OpenAIRecommendationProvider

    seen = _record_client_timeouts(monkeypatch)
    provider = OpenAIRecommendationProvider(
        api_key="k",
        model=OPENAI_MODEL,
        timeout_seconds=60.0,
        analysis_timeout_seconds=240.0,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text",
                                 "text": json.dumps(_analysis_body())}
                            ],
                        }
                    ]
                },
            )
        ),
    )
    provider.analyze_event(_TIMEOUT_BUNDLE, as_of=AS_OF)
    assert seen == [240.0]

    # ...and the DISCOVERY call is untouched: it keeps the shorter budget, so
    # a hung refresh does not sit for four minutes.
    seen.clear()
    provider.generate(set(), AS_OF, limit=1)
    assert seen == [60.0]


def test_anthropic_analyze_event_uses_the_analysis_timeout_not_the_shared_one(
    monkeypatch,
):
    seen = _record_client_timeouts(monkeypatch)
    provider = AnthropicRecommendationProvider(
        api_key="k",
        model="claude-opus-5",
        timeout_seconds=60.0,
        analysis_timeout_seconds=240.0,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "content": [{"type": "text",
                                 "text": json.dumps(_analysis_body())}]
                },
            )
        ),
    )
    provider.analyze_event(_TIMEOUT_BUNDLE, as_of=AS_OF)
    assert seen == [240.0]

    seen.clear()
    provider.generate(set(), AS_OF, limit=1)
    assert seen == [60.0]


def test_the_configured_analysis_timeout_reaches_both_factories(monkeypatch):
    """The setting is only worth having if it travels. An adapter default of
    240s with a factory that never passes the setting would leave an operator
    who raised LLM_ANALYSIS_TIMEOUT_SECONDS with no effect and no error."""
    from libs.common.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_ANALYSIS_TIMEOUT_SECONDS", "321")
    get_settings.cache_clear()
    try:
        assert get_settings().llm_analysis_timeout_seconds == 321.0
        monkeypatch.setenv("LLM_MODEL", OPENAI_MODEL)
        get_settings.cache_clear()
        assert get_recommendation_provider("openai").analysis_timeout_seconds == 321.0
        assert get_recommendation_provider("anthropic").analysis_timeout_seconds == 321.0
    finally:
        get_settings.cache_clear()


def test_the_analysis_call_asks_for_more_output_tokens_than_discovery(monkeypatch):
    """The §48 note has eighteen fields, three scenarios and a numbers_quoted
    list the prompt now requires at least three entries in. 4096 truncates it
    mid-JSON, which arrives as an unparseable body and a FAILED row."""
    from libs.llm.openai import OpenAIRecommendationProvider

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured[request.url.path] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "message",
                     "content": [{"type": "output_text",
                                  "text": json.dumps(_analysis_body())}]}
                ]
            },
        )

    provider = OpenAIRecommendationProvider(
        api_key="k", model=OPENAI_MODEL, transport=httpx.MockTransport(handler)
    )
    provider.analyze_event(_TIMEOUT_BUNDLE, as_of=AS_OF)
    assert captured["/v1/responses"]["max_output_tokens"] >= 6000


# ---------------------------------------------------------------------------
# Output language (Settings.llm_output_language)
# Narrative fields switch language; machine-read fields stay English. The
# instruction rides the system prompt on BOTH providers and BOTH methods, and
# "en"/unknown leaves the prompts byte-identical to before the feature.
# ---------------------------------------------------------------------------


def test_language_instruction_zh_and_default():
    from libs.llm.provider import language_instruction

    zh = language_instruction("zh")
    assert "简体中文" in zh
    # Machine-read fields are explicitly pinned to English.
    for field in ("horizon", "catalyst_type", "reason_codes"):
        assert field in zh
    assert language_instruction("en") == ""
    assert language_instruction("") == ""
    assert language_instruction("fr") == ""  # unknown = no addendum, never a guess


def test_openai_zh_output_language_reaches_both_wire_prompts():
    from libs.llm.openai import OpenAIRecommendationProvider
    from libs.llm.provider import GroundingArticle

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("instructions", []).append(
            json.loads(request.content)["instructions"]
        )
        return httpx.Response(200, json=_openai_response([VALID_ENTRY]))

    provider = OpenAIRecommendationProvider(
        api_key="test-key",
        model=OPENAI_MODEL,
        transport=httpx.MockTransport(handler),
        output_language="zh",
    )
    provider.generate(set(), AS_OF, limit=5)
    provider.enrich(
        [
            GroundingArticle(
                url="https://example.com/a1",
                title="Apple earnings beat",
                publisher="Reuters",
                published_at="2026-08-09T12:00:00+00:00",
                tickers=("AAPL",),
                description="Beat on cloud momentum.",
            )
        ],
        set(),
        AS_OF,
        limit=5,
    )
    assert len(captured["instructions"]) == 2
    for instructions in captured["instructions"]:
        assert "简体中文" in instructions


def test_openai_default_language_leaves_prompts_english():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_response([VALID_ENTRY]))

    _openai_provider_with(handler).generate(set(), AS_OF, limit=5)
    assert "简体中文" not in captured["body"]["instructions"]


def test_anthropic_zh_output_language_reaches_system_prompt():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_api_response([VALID_ENTRY]))

    provider = AnthropicRecommendationProvider(
        api_key="test-key",
        model="claude-sonnet-5",
        transport=httpx.MockTransport(handler),
        output_language="zh",
    )
    provider.generate(set(), AS_OF, limit=5)
    assert "简体中文" in captured["body"]["system"]
