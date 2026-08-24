"""Event analysis packages — persistence, API and audit (Phase F, U3; event
spec §16, §46-§52, §69-§71, §99; audit §7.2, §9.3, §11.6).

WHAT THESE TESTS ARE ABOUT, and what they deliberately are not. The narrative
the model writes and the evidence bundle it reads belong to other units and
have their own suites. THIS file pins the lifecycle in between: which call
spends a provider call and which does not, what lands on disk when it
succeeds, what lands on disk when it fails, and whether the reader can tell
the two apart afterwards.

THE PROVIDER IS FAKED, THE VALIDATOR IS NOT. Every test below injects a
provider whose behaviour it chose and, in most cases, a small hand-written
bundle in place of U1's composed one. That is not a shortcut around the other
units — it is the only way to plant the failures that matter. A real provider
does not refuse on command, and a real evidence bundle does not contain a
number the model then misquotes; both have to be staged for the FAILED and
INVALID paths to be reachable at all, and those two paths are the ones a user
is most likely to hit and least likely to have been tested for. What is NOT
faked is the enforcement itself: ``libs.llm.event_analysis.validate_analysis``
and ``libs.trading_core.events.evidence.fact_index`` are the real ones, so an
INVALID verdict here is the verdict production would reach.

The guarantees, in the order they appear:

1. **The evidence route never needs a model.** Asserted against the
   ``unconfigured_client`` — no LLM at all — because the bundle is measured
   and filed facts, and charging a reader for prose to see a filed EPS number
   would be absurd.
2. **Only the POST spends.** The two GETs are asserted against a provider
   factory that EXPLODES if called, so a read that reached a model would raise
   rather than quietly succeeding, and this cannot rot into a no-op.
3. **The cache key is the evidence.** A second POST on an unchanged bundle
   returns ``cached: true``, writes no second row and no second audit record;
   ``force=true`` writes a new row and keeps the old one; a CHANGED bundle
   re-runs on its own without force, because different evidence is a
   different question.
4. **Failure is a stored status, not a 5xx** (§44 rule 18). A raising provider
   yields HTTP 200, ``status: "FAILED"``, a NULL analysis, an honest error
   string and the bundle intact.
5. **A misquote is stored, not hidden** (§99). A model that quotes a number
   the fact index does not contain yields ``status: "INVALID"`` with the text
   AND the violations, because deleting the misquote destroys the evidence
   that it happened.
6. **No LLM configured is the one 503.** And the GETs still work, because
   nothing they serve came from a model.
7. **Prior analyses are gated on ``as_of``, are OK-only and are summaries**
   (§69, §70): a later opinion cannot leak backwards into an earlier run, an
   already-disproven one is not repeated, and no number travels with them.

Uses the shared ``client`` / ``unconfigured_client`` fixtures (conftest.py).
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from apps.gateway import event_analysis as seam
from apps.gateway.db import AuditEvent, EventAnalysisRow, EventRow, SessionLocal
from libs.common.config import get_settings
from libs.llm import ProviderError
from libs.llm.event_analysis import PROMPT_VERSION
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

#: A fixed anchor rather than ``now()``: every as-of assertion below is a
#: statement about ordering between instants, and a drifting clock would make
#: the "prior analysis written before this one" tests rot overnight.
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _iso(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


@pytest.fixture
async def fresh_db():
    """A clean schema for the tests that call the seam DIRECTLY.

    Most tests here get their schema from the shared ``client`` fixture, which
    drops and recreates it. The seam-level tests below take no HTTP client —
    they are about a function's own contract, not a route's — so they would
    otherwise inherit whatever rows the previous test left behind and collide
    on ``events.event_key``.
    """
    from apps.gateway.db import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


# ---------------------------------------------------------------------------
# The staged bundle and provider
# ---------------------------------------------------------------------------

#: A hand-written stand-in for U1's composed bundle. Small, but shaped like
#: the real thing where the shape is load-bearing: every section carries its
#: §49 tier, the consensus block carries the §33 unavailability string
#: verbatim (nothing may fabricate an estimate), and there are exactly two
#: numbers for the validator tests to quote correctly or incorrectly.
FAKE_BUNDLE = {
    "event": {"tier": "DATA", "event_key": "EARNINGS:AAPL:2026-08-27"},
    "fundamentals": {"tier": "DATA", "revenue": 123.5, "available": True},
    "price_analysis": {"tier": "QUANT", "reaction": {"1d": {"return_pct": 4.25}}},
    "consensus": {
        "tier": "DATA",
        "status": "CONSENSUS_DATA_UNAVAILABLE",
        "reason": "no consensus/estimate provider in subscription",
    },
    "options_analysis": {"tier": "DATA", "status": "NOT_AVAILABLE_YET"},
}


def _scenario(text: str) -> dict:
    return {
        "conditions": text,
        "guidance_conditions": text,
        "why_market_reacts": text,
        "evidence_refs": ["fundamentals.revenue"],
    }


def _analysis(**overrides) -> dict:
    """A schema-complete analysis whose every numeral is backed by a real
    bundle path.

    Built as a helper rather than a constant because the REAL validator is in
    play: it checks the required key set, the enums, and every numeral that
    appears in the prose. A test that wants to plant one specific violation
    has to be valid in every other respect, or it would pass for the wrong
    reason — "INVALID" would prove nothing about the misquote it meant to
    plant. The prose here is deliberately number-free apart from the two
    figures that ARE quoted.
    """
    base = {
        "executive_summary": "Revenue of 123.5 frames the print.",
        "what_happened_last_time": "The shares moved 4.25% on the day after.",
        "what_changed_since": "Coverage has been thin since.",
        "fundamental_developments": "Revenue of 123.5 is the only filed figure.",
        "price_and_positioning": "Positioning is not measurable from the bundle.",
        "market_expectations": "No consensus provider is in the subscription.",
        "prediction_market_expectations": None,
        "key_positive_catalysts": ["Guidance could be raised."],
        "key_negative_catalysts": ["Guidance could be cut."],
        "what_matters_most": "The guidance language, not the printed quarter.",
        "scenarios": {
            "upside": _scenario("Guidance is raised."),
            "base": _scenario("Guidance is held."),
            "downside": _scenario("Guidance is cut."),
        },
        "surprise_threshold": {
            "narrative": "No consensus exists, so no surprise can be measured.",
            "confidence": "NOT_MEANINGFUL",
        },
        "key_unknowns": ["Whether the guide is reiterated."],
        "evidence_conflicts": [],
        "web_research_highlights": [],
        "invalidation": "A pre-announcement before the date would invalidate this.",
        "expectations_gap_regime": "INSUFFICIENT_DATA",
        "confidence": "LOW",
        "evidence_refs": ["fundamentals.revenue"],
        "numbers_quoted": [
            {"path": "fundamentals.revenue", "value": 123.5},
            {"path": "price_analysis.reaction.1d.return_pct", "value": 4.25},
        ],
    }
    base.update(overrides)
    return base


#: A well-formed analysis: every number it uses is listed in
#: ``numbers_quoted`` with a path that exists and a matching value.
GOOD_ANALYSIS = _analysis()

#: The same analysis with one invented figure — a path the bundle does not
#: contain, and the bundle in fact says consensus is UNAVAILABLE. This is the
#: §47 violation the platform exists to catch, and it is the single most
#: important negative case in this file.
BAD_ANALYSIS = _analysis(
    market_expectations="Consensus EPS was 2.11, so this is a clear beat.",
    numbers_quoted=[
        {"path": "fundamentals.revenue", "value": 123.5},
        {"path": "price_analysis.reaction.1d.return_pct", "value": 4.25},
        {"path": "consensus.eps_estimate", "value": 2.11},
    ],
)


class FakeResult:
    """Stand-in for U2's ``EventAnalysisResult`` — the attributes the seam
    reads, and nothing else. Duck-typed deliberately: pinning the seam to the
    real dataclass here would make this file fail for a reason that has
    nothing to do with persistence."""

    def __init__(
        self,
        analysis: dict,
        *,
        provider: str = "stub",
        model: str = "stub-model",
        # Tracks the REAL contract version: the seam's cache key uses the
        # module's PROMPT_VERSION, and a fake pinned to an old literal would
        # store rows the cache lookup can never hit again.
        prompt_version: str = PROMPT_VERSION,
        usage: dict | None = None,
        latency_ms: int | None = 42,
        violations: list[str] | None = None,
    ) -> None:
        self.analysis = analysis
        self.provider = provider
        self.model = model
        self.prompt_version = prompt_version
        self.usage = (
            usage if usage is not None else {"input_tokens": 10, "output_tokens": 5}
        )
        self.latency_ms = latency_ms
        self.violations = violations or []


class FakeProvider:
    """A provider whose ``analyze_event`` does exactly what a test needs.

    ``calls`` is what the "only the POST spends" assertions read: a GET that
    reached a model would leave a mark here, and there would be no way to
    explain it away.
    """

    def __init__(self, result=None, *, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []

    def analyze_event(self, bundle_json, *, as_of):
        self.calls.append({"bundle": bundle_json, "as_of": as_of})
        if self.raises is not None:
            raise self.raises
        return self.result


def _install(
    monkeypatch,
    *,
    provider: FakeProvider | None = None,
    bundle: dict | None = None,
):
    """Stage the bundle and the provider. The VALIDATOR stays real.

    The bundle builder is patched at :func:`apps.gateway.event_analysis.build_bundle`
    — this seam's OWN boundary — rather than inside U1, so these tests pin
    THIS unit's behaviour and do not re-test the composition. The digest is
    computed from the staged bundle with the real hash so the cache tests
    exercise the real key. The §69 prior-analysis tests are the exception:
    they need the real ``build_bundle`` and stage only what is beneath it.
    """
    from libs.trading_core.events.evidence import bundle_digest

    staged = FAKE_BUNDLE if bundle is None else bundle

    async def fake_build_bundle(session, event_row, *, as_of, settings=None, **kw):
        # The REAL digest function, not a local sha256 of the dict: the cache
        # tests below are about which pairs of bundles share a key, and a
        # test-local hash would keep passing after the production one changed
        # its mind about what counts as evidence. The staged bundle is stamped
        # with the caller's as_of exactly as U1 stamps its own, so a later
        # instant over unchanged evidence is reachable here.
        payload = {**staged, "as_of": as_of.isoformat()}
        return payload, bundle_digest(payload)

    monkeypatch.setattr(seam, "build_bundle", fake_build_bundle)

    if provider is not None:
        import libs.llm as llm_pkg

        monkeypatch.setattr(
            llm_pkg, "get_recommendation_provider", lambda name: provider
        )
    return provider


def _explode_provider(monkeypatch):
    """A provider factory that raises if anything asks for a model.

    Used by the read-path tests: "this route does not call the LLM" is only a
    real assertion if calling one would be LOUD.
    """
    import libs.llm as llm_pkg

    def boom(name):  # pragma: no cover - reaching it IS the failure
        raise AssertionError(f"a read path asked for an LLM provider ({name!r})")

    monkeypatch.setattr(llm_pkg, "get_recommendation_provider", boom)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


async def _add_event(
    *,
    key: str = "EARNINGS:AAPL:2026-08-27",
    ticker: str | None = "AAPL",
    when: datetime = NOW + timedelta(days=9),
    status: EventStatus = EventStatus.CONFIRMED,
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=EventType.EARNINGS.value,
            title="Apple Q3 earnings",
            ticker=ticker,
            scheduled_at=when,
            session=EventSession.AFTER_MARKET.value,
            status=status.value,
            source=EventSourceKind.STRUCTURED_PROVIDER.value,
            source_name="test",
        )
        s.add(row)
        await s.commit()
        return row.id


async def _rows() -> list[EventAnalysisRow]:
    async with SessionLocal() as s:
        return list(
            (
                await s.execute(
                    select(EventAnalysisRow).order_by(EventAnalysisRow.id)
                )
            )
            .scalars()
            .all()
        )


async def _audit_rows() -> list[AuditEvent]:
    async with SessionLocal() as s:
        return list(
            (
                await s.execute(
                    select(AuditEvent)
                    .where(AuditEvent.action == "EVENT_ANALYSIS_GENERATED")
                    .order_by(AuditEvent.id)
                )
            )
            .scalars()
            .all()
        )


# ---------------------------------------------------------------------------
# 1. Evidence: no model needed, no model called
# ---------------------------------------------------------------------------


async def test_evidence_route_returns_tiers_and_digest_without_an_llm(
    unconfigured_client, monkeypatch
):
    """The bundle is measured and filed facts — it does not need a model.

    Run against the client with NO LLM configured at all, which is the whole
    point: a reader who wants the filed revenue and the previous reaction
    should be able to see them on an install that has never been given an API
    key. If this route ever grew a ``require_llm_provider`` guard, this test
    would 503 and say so.
    """
    event_id = await _add_event()
    _install(monkeypatch)
    _explode_provider(monkeypatch)

    r = await unconfigured_client.get(
        f"/api/events/{event_id}/evidence", params={"as_of": _iso(NOW)}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_id"] == event_id
    assert body["as_of"].startswith("2026-08-18T12:00")
    assert len(body["bundle_digest"]) == 64  # sha256 hex
    # §49: the tiers are labelled ON the sections, not inferred by the client.
    assert body["bundle"]["fundamentals"]["tier"] == "DATA"
    assert body["bundle"]["price_analysis"]["tier"] == "QUANT"
    # §33/§98: consensus is never fabricated — the string is the contract.
    assert body["bundle"]["consensus"]["status"] == "CONSENSUS_DATA_UNAVAILABLE"


async def test_evidence_badges_an_estimated_event_date(client, monkeypatch):
    """An ESTIMATED date may be analysed — but the payload must SAY it is
    derived (§7, §11). A cadence guess rendered as a confirmed fact is the
    exact failure the status column exists to prevent, and the badge is what
    carries it into every downstream surface."""
    event_id = await _add_event(
        key="EARNINGS:AAPL:2026-11-01", status=EventStatus.ESTIMATED
    )
    _install(monkeypatch)

    r = await client.get(f"/api/events/{event_id}/evidence")
    assert r.status_code == 200
    badge = r.json()["event_status_badge"]
    assert badge["is_estimated"] is True
    assert badge["status"] == "ESTIMATED"
    assert "DERIVED" in (badge["note"] or "")


async def test_evidence_rejects_a_future_as_of(client, monkeypatch):
    """422, never a silent clamp to now: clamping answers a DIFFERENT question
    than the one asked and the caller has no way to notice."""
    event_id = await _add_event()
    _install(monkeypatch)

    future = datetime.now(timezone.utc) + timedelta(days=3)
    r = await client.get(
        f"/api/events/{event_id}/evidence", params={"as_of": _iso(future)}
    )
    assert r.status_code == 422
    assert "future" in r.text


async def test_evidence_404s_only_for_a_missing_event(client, monkeypatch):
    _install(monkeypatch)
    r = await client.get("/api/events/999999/evidence")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 2. GET analysis: reads only
# ---------------------------------------------------------------------------


async def test_get_analysis_404s_with_a_machine_readable_code(client, monkeypatch):
    """"Nobody has run one yet" is a 404 with ``ANALYSIS_NOT_FOUND``, not a
    200 with ``available: false``.

    The distinction is not pedantry: an empty news window is a degradation of
    the platform's DATA and belongs at 200 with a reason, while a missing
    analysis is a resource that does not exist and that a POST creates. The
    UI turns this specific code into the "Generate analysis" button, so the
    code is load-bearing and pinned here.
    """
    event_id = await _add_event()
    _explode_provider(monkeypatch)

    r = await client.get(f"/api/events/{event_id}/analysis")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "ANALYSIS_NOT_FOUND"
    assert f"/api/events/{event_id}/analysis" in detail["message"]


async def test_get_analysis_never_calls_the_llm(client, monkeypatch):
    """Asserted against a factory that raises: a read that reached a model
    would explode rather than quietly succeed, so this cannot rot into a
    no-op the day someone adds a "refresh if stale" convenience."""
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)
    assert (await client.post(f"/api/events/{event_id}/analysis")).status_code == 200
    assert len(provider.calls) == 1

    _explode_provider(monkeypatch)
    r = await client.get(f"/api/events/{event_id}/analysis")
    assert r.status_code == 200
    assert r.json()["status"] == "OK"
    assert r.json()["cached"] is True


# ---------------------------------------------------------------------------
# 3. POST: the happy path, the cache and force
# ---------------------------------------------------------------------------


async def test_post_stores_the_bundle_with_the_analysis_and_audits_it(
    client, monkeypatch
):
    """The row carries the EXACT evidence the model saw, plus the audit trail.

    Storing the bundle is what makes the §47 claim checkable later: "every
    number is quoted from the evidence" is unverifiable against a bundle
    re-derived from tomorrow's filings. The audit row is rule 12 — a
    state-changing decision that leaves no trace did not happen.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)

    r = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "OK"
    assert body["cached"] is False
    assert body["violations"] == []
    assert body["analysis"]["executive_summary"] == GOOD_ANALYSIS["executive_summary"]
    assert body["bundle"]["fundamentals"]["revenue"] == 123.5
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert body["latency_ms"] == 42
    assert body["prompt_version"] == PROMPT_VERSION
    # §49: the LLM tier and the DATA/QUANT tier are separate KEYS, so a client
    # cannot render a model sentence with the authority of a filed number.
    assert set(body["tiers"]) == {"bundle", "analysis", "prior_analyses"}

    rows = await _rows()
    assert len(rows) == 1
    # The whole staged bundle is on disk, stamped with the instant it answered
    # (the stamp is part of the DOCUMENT; it is pruned only from the hash).
    assert {k: v for k, v in rows[0].bundle.items() if k != "as_of"} == FAKE_BUNDLE
    assert rows[0].bundle["as_of"].startswith("2026-08-18T12:00")
    assert rows[0].status == "OK"
    assert rows[0].violations == []

    audits = await _audit_rows()
    assert len(audits) == 1
    assert audits[0].entity_type == "event"
    assert audits[0].entity_id == str(event_id)
    assert audits[0].details["status"] == "OK"
    assert audits[0].details["violations_count"] == 0
    assert audits[0].details["digest"] == rows[0].bundle_digest


async def test_second_post_on_unchanged_evidence_is_cached(client, monkeypatch):
    """The cache key is the EVIDENCE, not the clock (§72).

    A second press must not spend a second call, must not write a second row
    and must not write a second audit record — an audit trail of "the user
    pressed a button that did nothing" is noise that buries the entries that
    matter.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)

    first = await client.post(f"/api/events/{event_id}/analysis")
    second = await client.post(f"/api/events/{event_id}/analysis")
    assert first.status_code == second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["id"] == first.json()["id"]
    assert len(provider.calls) == 1
    assert len(await _rows()) == 1
    assert len(await _audit_rows()) == 1


async def test_a_later_as_of_over_unchanged_evidence_is_still_a_cache_hit(
    client, monkeypatch
):
    """THE LIVE DEFECT. A minute passing is not new evidence.

    Two POSTs a minute apart used to produce two different digests over
    byte-identical filings, bars and articles, because the bundle stamps the
    request instant and the hash covered it. The second press therefore missed
    the cache and spent a model call re-deriving an answer already on disk —
    and, at 51s per call against a 60s timeout, the retry it forced was the
    thing that timed out and got stored as FAILED.

    The bundle here is the SAME staged evidence at two instants, so nothing
    but the clock differs, and the second press must not reach the provider.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)

    first = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    later = await client.post(
        f"/api/events/{event_id}/analysis",
        params={"as_of": _iso(NOW + timedelta(minutes=1))},
    )
    assert first.status_code == later.status_code == 200, later.text
    assert first.json()["cached"] is False
    assert later.json()["cached"] is True
    assert later.json()["id"] == first.json()["id"]
    assert later.json()["bundle_digest"] == first.json()["bundle_digest"]
    assert len(provider.calls) == 1
    assert len(await _rows()) == 1
    assert len(await _audit_rows()) == 1


async def test_changed_evidence_at_a_later_as_of_still_reruns(client, monkeypatch):
    """The other half: pruning the clock must not blind the cache to a number.

    A digest that ignored the instant AND the evidence would serve yesterday's
    analysis forever. One revenue figure moving is a different question, and
    it re-runs even though the clock moved too.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)
    first = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    assert first.json()["cached"] is False

    moved = {**FAKE_BUNDLE,
             "fundamentals": {**FAKE_BUNDLE["fundamentals"], "revenue": 124.5}}
    _install(monkeypatch, provider=provider, bundle=moved)
    second = await client.post(
        f"/api/events/{event_id}/analysis",
        params={"as_of": _iso(NOW + timedelta(minutes=1))},
    )
    assert second.json()["cached"] is False
    assert second.json()["bundle_digest"] != first.json()["bundle_digest"]
    assert len(provider.calls) == 2


async def test_the_default_as_of_is_truncated_so_the_common_path_caches(
    client, monkeypatch
):
    """Two presses of the button, one model call — with NO ``as_of`` passed.

    This is the case the UI actually hits, and it is the one a staged-bundle
    test cannot catch on its own. The bundle carries its own ``as_of``, so the
    instant is part of what the digest covers — correct, because a bundle
    answering a different moment IS a different document. But a default of
    ``now()`` at microsecond resolution means no two presses ever share a
    digest: the cache would miss every single time and every press would spend
    a model call while reporting the duplicate as fresh. The route therefore
    truncates the DEFAULT to the minute.

    An explicitly passed ``as_of`` is NOT truncated, and the second half of
    this test pins that: the as-of contract is that the caller's instant is
    honoured exactly, and rounding someone's stated question to a coarser grain
    would answer a different one than they asked.
    """
    event_id = await _add_event()
    # NOTHING is staged here — the real bundle builder, the real stub provider
    # and the real validator all run. That is the point: this is the ONE test
    # in the file that exercises the whole path end to end, and it is where an
    # integration defect between the units would surface. A staged bundle
    # cannot catch a cache miss caused by the as_of the ROUTE chose, because a
    # staged bundle does not carry one. The stub provider is counted through a
    # wrapper rather than replaced, so "one model call" stays a real assertion.
    import libs.llm as llm_pkg

    calls: list[dict] = []
    real_factory = llm_pkg.get_recommendation_provider

    def counting_factory(name):
        inner = real_factory(name)

        class Counted:
            def analyze_event(self, bundle_json, *, as_of):
                calls.append({"as_of": as_of})
                return inner.analyze_event(bundle_json, as_of=as_of)

        return Counted()

    monkeypatch.setattr(llm_pkg, "get_recommendation_provider", counting_factory)

    first = await client.post(f"/api/events/{event_id}/analysis")
    second = await client.post(f"/api/events/{event_id}/analysis")
    assert first.status_code == second.status_code == 200, first.text
    assert first.json()["bundle_digest"] == second.json()["bundle_digest"]
    assert second.json()["cached"] is True
    assert len(calls) == 1
    assert first.json()["as_of"].endswith(":00+00:00")
    # End to end, with the real validator: the stub quotes only real bundle
    # facts, so the package is OK — which is also what makes it cacheable.
    assert first.json()["status"] == "OK"
    assert first.json()["violations"] == []

    precise = "2026-08-18T12:00:00.123456Z"
    third = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": precise}
    )
    assert third.json()["as_of"] == "2026-08-18T12:00:00.123456+00:00"


async def test_force_regenerates_and_supersedes_without_deleting(client, monkeypatch):
    """``force`` INSERTS and DEMOTES; it never deletes.

    The forced answer lands on the same cache key as the one it replaces —
    same event, same evidence, same prompt version, same model — and the
    partial index holds at most one OK row there. The old row is therefore
    moved to ``SUPERSEDED`` rather than removed: its text, its bundle and its
    violations stay on disk and stay listed, so a regression introduced by a
    new model version is readable BESIDE the answer that replaced it. Deleting
    it would be the one outcome that makes such a regression undiagnosable,
    which is the whole reason this status exists.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)

    first = await client.post(f"/api/events/{event_id}/analysis")
    forced = await client.post(
        f"/api/events/{event_id}/analysis", params={"force": "true"}
    )
    assert forced.status_code == 200
    assert forced.json()["cached"] is False
    assert forced.json()["id"] != first.json()["id"]
    assert len(provider.calls) == 2

    rows = await _rows()
    assert len(rows) == 2
    assert rows[0].id == first.json()["id"] and rows[0].status == "SUPERSEDED"
    assert rows[1].id == forced.json()["id"] and rows[1].status == "OK"
    # Superseded is not erasure: the previous text survives intact.
    assert rows[0].analysis == rows[1].analysis
    assert rows[0].bundle_digest == rows[1].bundle_digest
    assert len(await _audit_rows()) == 2
    assert (await _audit_rows())[1].details["superseded_id"] == first.json()["id"]

    # And the cache is single-valued again: an unforced press returns the NEW
    # answer, never the demoted one.
    again = await client.post(f"/api/events/{event_id}/analysis")
    assert again.json()["cached"] is True
    assert again.json()["id"] == forced.json()["id"]


async def test_a_forced_rerun_that_fails_does_not_demote_the_good_answer(
    client, monkeypatch
):
    """Only a GOOD new answer supersedes.

    If a forced retry 403s, the analysis already on file is still the best
    thing the platform has and must keep serving the cache. Demoting it would
    turn a transient provider outage into the permanent loss of a valid
    answer — and the user would see the "Generate" call to action for an event
    that HAS been analysed.
    """
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    good = await client.post(f"/api/events/{event_id}/analysis")

    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("503 upstream")))
    forced = await client.post(
        f"/api/events/{event_id}/analysis", params={"force": "true"}
    )
    assert forced.json()["status"] == "FAILED"

    rows = await _rows()
    assert {r.id: r.status for r in rows}[good.json()["id"]] == "OK"

    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    again = await client.post(f"/api/events/{event_id}/analysis")
    assert again.json()["cached"] is True
    assert again.json()["id"] == good.json()["id"]


async def test_changed_evidence_reruns_without_force(client, monkeypatch):
    """Different evidence is a different question, so no ``force`` is needed.

    This is the other half of the cache contract and the one that would rot
    unnoticed: a cache keyed on ``event_id`` alone would serve last week's
    answer for this week's filings and look perfectly correct in every test
    that never changed the bundle.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=provider)
    first = await client.post(f"/api/events/{event_id}/analysis")

    moved = {**FAKE_BUNDLE, "fundamentals": {"tier": "DATA", "revenue": 200.0}}
    _install(monkeypatch, provider=provider, bundle=moved)
    second = await client.post(f"/api/events/{event_id}/analysis")

    assert second.json()["cached"] is False
    assert second.json()["bundle_digest"] != first.json()["bundle_digest"]
    assert len(provider.calls) == 2


# ---------------------------------------------------------------------------
# 4. Failure is a stored status
# ---------------------------------------------------------------------------


async def test_provider_failure_is_a_stored_status_not_a_5xx(client, monkeypatch):
    """A 403, a timeout or a refusal answers HTTP 200 with ``FAILED``.

    Two things are being defended. First, the caller asked a question the
    platform can still PARTLY answer — the evidence bundle is assembled and
    is right there — so throwing the whole response away would discard real
    work. Second, ``analysis`` stays NULL: a placeholder narrative in that
    field would be indistinguishable from a real one to every downstream
    reader, which is exactly the fabrication §44 rule 18 forbids.
    """
    event_id = await _add_event()
    provider = FakeProvider(raises=ProviderError("upstream 403 SUBSCRIPTION_DENIED"))
    _install(monkeypatch, provider=provider)

    r = await client.post(f"/api/events/{event_id}/analysis")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "FAILED"
    assert body["analysis"] is None
    assert body["latency_ms"] is None  # 0 would read as "answered instantly"
    assert "SUBSCRIPTION_DENIED" in body["error"]
    assert body["bundle"]["fundamentals"]["revenue"] == 123.5

    rows = await _rows()
    assert len(rows) == 1 and rows[0].status == "FAILED"
    assert (await _audit_rows())[0].details["status"] == "FAILED"


async def test_an_adapter_bug_is_also_a_failed_row(client, monkeypatch):
    """Not only ``ProviderError``. A KeyError inside an adapter is still the
    model failing to answer as far as the user is concerned, and letting it
    become a 500 would lose the evidence bundle AND the trail."""
    event_id = await _add_event()
    provider = FakeProvider(raises=KeyError("output_text"))
    _install(monkeypatch, provider=provider)

    r = await client.post(f"/api/events/{event_id}/analysis")
    assert r.status_code == 200
    assert r.json()["status"] == "FAILED"
    assert "KeyError" in r.json()["error"]


async def test_a_failed_row_does_not_satisfy_a_later_request(client, monkeypatch):
    """Retry means retry. Only ``OK`` rows serve the cache — a user pressing
    the button after a 403 is asking the platform to TRY AGAIN, and returning
    the stored failure with ``cached: true`` would strand them."""
    event_id = await _add_event()
    failing = FakeProvider(raises=ProviderError("timeout"))
    _install(monkeypatch, provider=failing)
    assert (await client.post(f"/api/events/{event_id}/analysis")).json()["status"] == "FAILED"

    working = FakeProvider(FakeResult(GOOD_ANALYSIS))
    _install(monkeypatch, provider=working)
    second = await client.post(f"/api/events/{event_id}/analysis")
    assert second.json()["status"] == "OK"
    assert second.json()["cached"] is False
    assert len(working.calls) == 1
    assert {r.status for r in await _rows()} == {"FAILED", "OK"}


async def test_get_serves_the_last_good_analysis_and_flags_the_newer_failure(
    client, monkeypatch
):
    """A failed retry must not cost the reader the analysis already on disk.

    THE LIVE DEFECT'S SECOND HALF. The GET took the NEWEST row, so one timeout
    replaced a complete piece of research with an error banner — the platform
    still HAD the note and told the reader it did not. The good row is now the
    primary payload and the failure rides along as ``last_attempt``, which is
    the honest pair: the analysis is shown, and it is shown as possibly stale.
    """
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    good = (await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )).json()
    assert good["status"] == "OK"

    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("ReadTimeout")),
             bundle={**FAKE_BUNDLE, "fundamentals": {"tier": "DATA", "revenue": 999.0}})
    failed = (await client.post(
        f"/api/events/{event_id}/analysis",
        params={"as_of": _iso(NOW + timedelta(hours=1))},
    )).json()
    assert failed["status"] == "FAILED"

    _explode_provider(monkeypatch)
    body = (await client.get(f"/api/events/{event_id}/analysis")).json()
    assert body["status"] == "OK"
    assert body["id"] == good["id"]
    assert body["analysis"]["executive_summary"] == GOOD_ANALYSIS["executive_summary"]
    # The failure is NOT hidden — it is named, with the reason and the instant.
    attempt = body["last_attempt"]
    assert attempt["status"] == "FAILED"
    assert "ReadTimeout" in attempt["error"]
    assert attempt["id"] == failed["id"]
    assert attempt["created_at"]
    # provider/model are the CONFIGURED ones on a failure (nothing answered),
    # so they are present as keys and may be empty on an install that has not
    # named a model — the reader still learns which seam was tried.
    assert set(attempt) == {
        "id", "status", "error", "created_at", "as_of", "provider", "model"
    }
    # No stale bundle confusion: the payload's bundle is the GOOD row's.
    assert body["bundle"]["fundamentals"]["revenue"] == 123.5


async def test_get_omits_last_attempt_when_the_newest_row_is_the_good_one(
    client, monkeypatch
):
    """The key is ABSENT, not null: nothing failed, so there is nothing to
    tell the reader, and a null would be one more branch for the UI."""
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    await client.post(f"/api/events/{event_id}/analysis")

    _explode_provider(monkeypatch)
    body = (await client.get(f"/api/events/{event_id}/analysis")).json()
    assert body["status"] == "OK"
    assert "last_attempt" not in body
    assert "last_good" not in body


async def test_get_returns_the_failure_itself_when_no_good_analysis_exists(
    client, monkeypatch
):
    """With nothing good on file the failure IS the whole story — unchanged
    behaviour, and still a 200 carrying the bundle the run did assemble."""
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("upstream 403")))
    await client.post(f"/api/events/{event_id}/analysis")

    _explode_provider(monkeypatch)
    r = await client.get(f"/api/events/{event_id}/analysis")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "FAILED"
    assert body["analysis"] is None
    assert "403" in body["error"]
    assert "last_attempt" not in body  # it is not "alongside" anything
    assert body["bundle"]["fundamentals"]["revenue"] == 123.5


async def test_get_still_404s_when_no_row_of_any_status_exists(client, monkeypatch):
    """Unchanged: "nobody has run one" is a resource that does not exist."""
    event_id = await _add_event()
    _explode_provider(monkeypatch)
    r = await client.get(f"/api/events/{event_id}/analysis")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "ANALYSIS_NOT_FOUND"


async def test_a_failed_post_points_at_the_last_good_analysis(client, monkeypatch):
    """The failure response carries the fallback, so the UI need not guess.

    A pointer, not the package: inlining the good analysis into the response
    to a request that FAILED would let a client render an older answer as the
    answer to the question just asked. The id is one GET away.
    """
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    good = (await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )).json()

    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("ReadTimeout")),
             bundle={**FAKE_BUNDLE, "fundamentals": {"tier": "DATA", "revenue": 999.0}})
    failed = (await client.post(
        f"/api/events/{event_id}/analysis",
        params={"as_of": _iso(NOW + timedelta(hours=1))},
    )).json()

    assert failed["status"] == "FAILED"
    assert failed["analysis"] is None  # the fallback is NOT inlined
    assert failed["last_good"] == {
        "id": good["id"],
        "created_at": good["created_at"],
        "as_of": good["as_of"],
        "status": "OK",
    }


async def test_a_superseded_row_is_not_resurrected_as_the_last_good_one(
    client, monkeypatch
):
    """A demoted answer must not outlive the answer that replaced it.

    ``force`` demotes the old OK row to SUPERSEDED precisely because an
    operator judged the new answer better on the same evidence. If a LATER run
    then fails, the honest report is "the newest attempt failed, and here is
    the answer that was current" — the forced one — never the row the operator
    already rejected. Resurrecting it would silently undo the supersede, and
    the reader would have no way to tell which of the three rows they are
    looking at.
    """
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    first = (await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )).json()
    forced = (await client.post(
        f"/api/events/{event_id}/analysis",
        params={"as_of": _iso(NOW), "force": "true"},
    )).json()
    assert {r.status for r in await _rows()} == {"SUPERSEDED", "OK"}

    # Now a third run, on CHANGED evidence, fails.
    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("ReadTimeout")),
             bundle={**FAKE_BUNDLE, "fundamentals": {"tier": "DATA", "revenue": 999.0}})
    failed = (await client.post(
        f"/api/events/{event_id}/analysis",
        params={"as_of": _iso(NOW + timedelta(hours=1))},
    )).json()
    assert failed["status"] == "FAILED"
    # The POST's pointer names the row that was CURRENT, not the demoted one.
    assert failed["last_good"]["id"] == forced["id"] != first["id"]

    _explode_provider(monkeypatch)
    body = (await client.get(f"/api/events/{event_id}/analysis")).json()
    assert body["status"] == "OK"
    assert body["id"] == forced["id"]
    assert body["id"] != first["id"]  # the superseded row stays superseded
    assert body["last_attempt"]["id"] == failed["id"]


async def test_a_failed_post_with_nothing_good_on_file_omits_last_good(
    client, monkeypatch
):
    """No fallback exists, so none is claimed."""
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("boom")))
    body = (await client.post(f"/api/events/{event_id}/analysis")).json()
    assert body["status"] == "FAILED"
    assert "last_good" not in body


# ---------------------------------------------------------------------------
# 5. A misquote is stored, not hidden (§47, §99)
# ---------------------------------------------------------------------------


async def test_invented_number_is_stored_as_invalid_with_its_violations(
    client, monkeypatch
):
    """THE test this whole unit exists for.

    The model quotes ``consensus.eps_estimate = 2.11``. There is no such path
    in the evidence — the bundle says consensus is UNAVAILABLE — so the figure
    was invented, which §47 forbids absolutely. The response is 200 with
    ``INVALID``, the violation named, AND THE TEXT STILL PRESENT: deleting the
    misquote would destroy the evidence that the model produced it, and the
    §99 transparency rule is that the reader sees what was actually said,
    badged for what it is.
    """
    event_id = await _add_event()
    provider = FakeProvider(FakeResult(BAD_ANALYSIS))
    _install(monkeypatch, provider=provider)

    r = await client.post(f"/api/events/{event_id}/analysis")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "INVALID"
    assert body["analysis"] is not None
    assert body["analysis"]["executive_summary"] == BAD_ANALYSIS["executive_summary"]
    assert any("consensus.eps_estimate" in v for v in body["violations"])

    rows = await _rows()
    assert rows[0].status == "INVALID" and rows[0].analysis is not None
    # The count on the audit row is the real validator's, and it is >= 1
    # rather than exactly 1 on purpose: the same invented 2.11 trips both the
    # unknown-path check AND the prose-numeral check, and pinning an exact
    # number here would make this test fail the day the validator legitimately
    # grows a rule. What matters is that the count is non-zero and matches the
    # list the caller was shown.
    audit_details = (await _audit_rows())[0].details
    assert audit_details["violations_count"] == len(body["violations"]) >= 1


async def test_a_wrong_value_for_a_real_path_is_also_a_violation(client, monkeypatch):
    """Quoting a path that EXISTS with a number that does not match is the
    subtler misquote and the more dangerous one: it looks sourced. The check
    is on the value, not merely on the path's existence."""
    event_id = await _add_event()
    wrong = _analysis(
        numbers_quoted=[
            {"path": "fundamentals.revenue", "value": 999.0},
            {"path": "price_analysis.reaction.1d.return_pct", "value": 4.25},
        ]
    )
    _install(monkeypatch, provider=FakeProvider(FakeResult(wrong)))

    body = (await client.post(f"/api/events/{event_id}/analysis")).json()
    assert body["status"] == "INVALID"
    assert any("fundamentals.revenue" in v for v in body["violations"])


async def test_provider_reported_violations_are_kept_alongside_the_platforms(
    client, monkeypatch
):
    """Two independent validators, one list. A provider that already knows its
    output was degraded (a truncated response, an unfilled field) must not
    have that finding dropped just because the platform's own number check
    happens to pass."""
    event_id = await _add_event()
    result = FakeResult(GOOD_ANALYSIS, violations=["provider: response was truncated"])
    _install(monkeypatch, provider=FakeProvider(result))

    body = (await client.post(f"/api/events/{event_id}/analysis")).json()
    assert body["status"] == "INVALID"
    assert "provider: response was truncated" in body["violations"]


# ---------------------------------------------------------------------------
# 6. No LLM configured
# ---------------------------------------------------------------------------


async def test_post_503s_when_no_llm_is_configured(unconfigured_client, monkeypatch):
    """The ONE non-200 outcome on the POST besides 404/422.

    An install with no model configured cannot produce an analysis at all, and
    the honest answer is to say which setting is missing — not an empty
    package that a reader could mistake for "the model had nothing to say".
    """
    event_id = await _add_event()
    _install(monkeypatch)

    r = await unconfigured_client.post(f"/api/events/{event_id}/analysis")
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "LLM_NOT_CONFIGURED"
    assert await _rows() == []


async def test_the_reads_still_work_with_no_llm_configured(
    unconfigured_client, monkeypatch
):
    """Nothing the GETs serve came from a model, so nothing they serve should
    depend on one being configured."""
    event_id = await _add_event()
    _install(monkeypatch)

    assert (
        await unconfigured_client.get(f"/api/events/{event_id}/evidence")
    ).status_code == 200
    assert (
        await unconfigured_client.get(f"/api/events/{event_id}/analyses")
    ).json()["count"] == 0
    assert (
        await unconfigured_client.get(f"/api/events/{event_id}/analysis")
    ).status_code == 404


# ---------------------------------------------------------------------------
# 7. History and event memory (§69, §70)
# ---------------------------------------------------------------------------


async def test_analyses_history_lists_failures_too_newest_first(client, monkeypatch):
    """A list that showed only successes would let a model that fails four
    times in five look perfectly reliable. The trail of what was TRIED is part
    of what makes the analysis auditable (§99)."""
    event_id = await _add_event()
    _install(monkeypatch, provider=FakeProvider(raises=ProviderError("timeout")))
    await client.post(f"/api/events/{event_id}/analysis")
    _install(monkeypatch, provider=FakeProvider(FakeResult(GOOD_ANALYSIS)))
    await client.post(f"/api/events/{event_id}/analysis")

    body = (await client.get(f"/api/events/{event_id}/analyses")).json()
    assert body["count"] == 2
    assert [a["status"] for a in body["analyses"]] == ["OK", "FAILED"]
    # Summaries, not packages: no bundle travels in a list view.
    assert "bundle" not in body["analyses"][0]
    assert body["analyses"][0]["executive_summary"] == GOOD_ANALYSIS["executive_summary"]
    assert body["analyses"][1]["error"] == "timeout"


async def test_prior_analyses_are_gated_on_as_of_and_are_ok_only(fresh_db):
    """§69/§70 event memory: the gate is real and the filter is real.

    Three plants. A prior OK analysis at an EARLIER as-of is remembered. A
    LATER one is not — an opinion formed after the instant being asked about
    knows things this run must not, and feeding it back would be a look-ahead
    leak laundered through the model's own prose (§96). An INVALID one is not
    remembered either: it contains at least one number the platform has
    already proven wrong, and repeating it into a fresh prompt would propagate
    it. Its violations stay visible on its own package.
    """
    older_id = await _add_event(key="EARNINGS:AAPL:2026-05-01", when=NOW - timedelta(days=100))
    newer_id = await _add_event(key="EARNINGS:AAPL:2026-08-27")

    async with SessionLocal() as s:
        s.add_all(
            [
                EventAnalysisRow(
                    event_id=older_id,
                    as_of=NOW - timedelta(days=100),
                    kind="PRE_EVENT",
                    bundle=FAKE_BUNDLE,
                    bundle_digest="a" * 64,
                    analysis={
                        "executive_summary": "the older read",
                        "expectations_gap_regime": "BEAT_PRICED",
                        "confidence": "MODERATE",
                        "numbers_quoted": [{"path": "fundamentals.revenue", "value": 123.5}],
                    },
                    status="OK",
                    violations=[],
                ),
                EventAnalysisRow(
                    event_id=older_id,
                    as_of=NOW + timedelta(days=1),
                    kind="PRE_EVENT",
                    bundle=FAKE_BUNDLE,
                    bundle_digest="b" * 64,
                    analysis={"executive_summary": "from the future"},
                    status="OK",
                    violations=[],
                ),
                EventAnalysisRow(
                    event_id=older_id,
                    as_of=NOW - timedelta(days=50),
                    kind="PRE_EVENT",
                    bundle=FAKE_BUNDLE,
                    bundle_digest="c" * 64,
                    analysis={"executive_summary": "quoted an invented number"},
                    status="INVALID",
                    violations=["numbers_quoted path not in evidence: x"],
                ),
            ]
        )
        await s.commit()

    async with SessionLocal() as s:
        event = await s.get(EventRow, newer_id)
        priors = await seam.prior_analyses_for_ticker(
            s, "AAPL", before_as_of=NOW, exclude_event_id=event.id
        )

    assert [p["executive_summary"] for p in priors] == ["the older read"]
    assert priors[0]["expectations_gap_regime"] == "BEAT_PRICED"
    assert priors[0]["event_key"] == "EARNINGS:AAPL:2026-05-01"
    # §70: summaries only — no numbers travel with a prior opinion, because a
    # past run's figures are ITS quotations of a DIFFERENT bundle.
    assert "numbers_quoted" not in priors[0]
    assert "bundle" not in priors[0]


async def test_prior_analyses_are_empty_for_a_macro_event(fresh_db):
    """A CPI release has no issuer, so there is no per-ticker memory to read
    and none is invented."""
    async with SessionLocal() as s:
        assert await seam.prior_analyses_for_ticker(s, None, before_as_of=NOW) == []
        assert await seam.prior_analyses_for_ticker(s, "   ", before_as_of=NOW) == []


async def test_the_bundle_carries_prior_analyses_as_an_llm_prior_tier(
    client, monkeypatch
):
    """The §70 tier label is what stops a past opinion from being laundered
    into evidence: it is named LLM_PRIOR, it carries the note saying so, and
    it lives beside the DATA/QUANT sections rather than inside them."""
    event_id = await _add_event()

    async def fake_evidence(session, event_row, *, as_of, settings, **kw):
        return dict(FAKE_BUNDLE)

    import sys

    monkeypatch.setitem(
        sys.modules,
        "apps.gateway.event_evidence",
        type("M", (), {"build_evidence_bundle": staticmethod(fake_evidence)}),
    )
    monkeypatch.setitem(
        sys.modules,
        "libs.trading_core.events.evidence",
        type("M", (), {"bundle_digest": staticmethod(lambda b: "d" * 64)}),
    )

    async with SessionLocal() as s:
        event = await s.get(EventRow, event_id)
        bundle, digest = await seam.build_bundle(s, event, as_of=NOW)

    assert digest == "d" * 64
    prior = bundle["prior_analyses"]
    assert prior["tier"] == "LLM_PRIOR"
    assert "OPINIONS, not evidence" in prior["note"]
    assert prior["items"] == []


# ---------------------------------------------------------------------------
# 8. ORM / migration mirror sanity
# ---------------------------------------------------------------------------


def test_the_row_keeps_failure_columns_nullable():
    """``analysis``/``usage``/``latency_ms``/``error`` must stay NULLABLE, and
    ``violations`` NOT NULL.

    A failed call has no output, no token counts and no meaningful duration;
    columns that forced a value there would make the ORM write a 0 or a "" and
    a later reader could not tell a real instant answer from a failure. And an
    empty violations list ("checked, nothing wrong") is a different claim from
    NULL ("never checked") — every row this platform writes HAS been checked.
    """
    cols = EventAnalysisRow.__table__.columns
    for nullable in ("analysis", "usage", "latency_ms", "error", "provider", "model"):
        assert cols[nullable].nullable is True, nullable
    for required in ("bundle", "bundle_digest", "status", "violations", "as_of"):
        assert cols[required].nullable is False, required


# ---------------------------------------------------------------------------
# 9. END-TO-END WIRING — every unit real, nothing faked (verifier pass)
# ---------------------------------------------------------------------------
#
# Everything above this line stages either the bundle or the provider, for the
# reasons the module docstring gives: the FAILED and INVALID paths cannot be
# reached with real components, and staging is the only way to pin them.
#
# The cost of that choice is that no test above ever runs U1's composed bundle
# through U3's seam into U2's real provider and back through the real
# validator. Each unit is proven against its own contract, and a name that
# drifted between two of those contracts would leave every existing test green
# — the mock would answer to the old name on one side and the real code to the
# new one on the other. These tests close that gap. They deliberately assert
# almost nothing about CONTENT (the stub's prose is not analysis and its
# wording is not a contract); what they assert is that the seams MEET: a real
# bundle is accepted by a real provider, its output survives the real
# validator with zero violations, and the row that lands on disk is readable
# back through the real route.


async def _seeded_event(when: datetime | None = None) -> int:
    """A real event row for an end-to-end run, with no evidence staged.

    The bundle U1 composes for it is mostly unavailability — no bars, no
    filed statements, no news are seeded — and that is the point: a sparse
    bundle is the state a freshly-added event is actually in, it is the
    hardest case for a "quote only what you can see" contract (there is
    almost nothing to quote), and it is where a provider is most tempted to
    invent. The composition itself is U1's suite's business.
    """
    return await _add_event(
        key="EARNINGS:E2E:2026-08-27",
        ticker="E2E",
        when=when or (NOW + timedelta(days=9)),
    )


async def test_e2e_real_bundle_real_stub_real_validator_lands_ok(client):
    """The whole Phase F path with NOTHING mocked: U1 composes, U3 persists,
    U2's stub answers, U2's validator checks, U3 serves.

    ``LLM_PROVIDER=stub`` comes from the ``client`` fixture, so the provider
    is resolved the way production resolves it rather than injected. A zero
    ``violations`` list here is a real verdict from the real validator over a
    real fact index: it means the stub quoted only paths U1 actually emitted,
    which is the one cross-unit invariant no single unit's suite can prove
    about itself.
    """
    event_id = await _seeded_event()

    resp = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The seams met: a real provider ran, and the real validator cleared it.
    assert body["status"] == "OK", body.get("violations") or body.get("error")
    assert body["violations"] == []
    assert body["provider"] == "stub"
    assert body["cached"] is False

    # U1's bundle came through U3 intact, tiers and all (§49).
    bundle = body["bundle"]
    assert bundle["event"]["event_key"] == "EARNINGS:E2E:2026-08-27"
    assert {s.get("tier") for s in bundle.values() if isinstance(s, dict)} & {
        "DATA",
        "QUANT",
    }
    # §33: nothing anywhere fabricated a consensus.
    assert bundle["consensus"]["status"] == "CONSENSUS_DATA_UNAVAILABLE"

    # U2's schema arrived whole — the keys U4 renders are present.
    analysis = body["analysis"]
    for key in (
        "executive_summary",
        "scenarios",
        "surprise_threshold",
        "expectations_gap_regime",
        "confidence",
        "numbers_quoted",
        "evidence_refs",
    ):
        assert key in analysis, key
    assert set(analysis["scenarios"]) == {"upside", "base", "downside"}

    # It landed on disk, and it was audited.
    rows = await _rows()
    assert len(rows) == 1 and rows[0].status == "OK"
    assert len(await _audit_rows()) == 1


async def test_e2e_every_quoted_number_exists_in_the_composed_bundle(client):
    """§47, checked against U1's OWN fact index rather than a staged one.

    This is the assertion the whole tier separation exists to make possible:
    take the numbers the model printed, take the facts the platform measured,
    and show the first set is a subset of the second. Run over the real
    composed bundle it also proves the two units agree on PATH SPELLING — a
    fact index that dotted its paths differently from the bundle the provider
    read would surface here as a quoted path that does not resolve.
    """
    from libs.trading_core.events.evidence import fact_index

    event_id = await _seeded_event()
    resp = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    body = resp.json()
    assert body["status"] == "OK"

    facts = fact_index(body["bundle"])
    for quoted in body["analysis"]["numbers_quoted"]:
        assert quoted["path"] in facts, quoted["path"]
        expected = facts[quoted["path"]]
        actual = quoted["value"]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert abs(float(actual) - float(expected)) <= 1e-6, quoted
        else:
            assert actual == expected, quoted


async def test_e2e_evidence_route_serves_the_same_bundle_the_post_hashed(client):
    """GET /evidence and POST /analysis must be looking at the same evidence.

    If they diverged, the reader would be shown one set of facts while the
    model was reasoning over another — the failure mode that makes an
    "evidence-backed" narrative worthless, and one that no unit test can see
    because each unit builds its own bundle. Same event, same as-of, so the
    digests must be equal.
    """
    event_id = await _seeded_event()

    ev = await client.get(
        f"/api/events/{event_id}/evidence", params={"as_of": _iso(NOW)}
    )
    assert ev.status_code == 200, ev.text
    an = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    assert an.status_code == 200, an.text

    assert ev.json()["bundle_digest"] == an.json()["bundle_digest"]
    # U4 reads the evidence fallback into the SAME payload shape.
    assert isinstance(ev.json()["bundle"], dict)


async def test_e2e_second_post_is_cached_and_force_spends_again(client):
    """The cache key survives a real round trip.

    The staged-bundle cache tests above hash a dict a test wrote; this one
    hashes a bundle U1 composed twice from the database. It is the only place
    that would catch a bundle carrying a non-deterministic field (a wall clock
    read, a set iterated) — that would change the digest between two identical
    requests and silently spend a model call on every press.
    """
    event_id = await _seeded_event()
    params = {"as_of": _iso(NOW)}

    first = (await client.post(f"/api/events/{event_id}/analysis", params=params)).json()
    assert first["cached"] is False

    second = (await client.post(f"/api/events/{event_id}/analysis", params=params)).json()
    assert second["cached"] is True
    assert second["bundle_digest"] == first["bundle_digest"]
    assert len(await _rows()) == 1

    forced = (
        await client.post(
            f"/api/events/{event_id}/analysis", params={**params, "force": "true"}
        )
    ).json()
    assert forced["cached"] is False
    assert len(await _rows()) == 2

    # And the read route serves the newest one back.
    latest = (await client.get(f"/api/events/{event_id}/analysis")).json()
    assert latest["status"] == "OK"
    history = (await client.get(f"/api/events/{event_id}/analyses")).json()
    assert history["count"] == 2


async def test_e2e_a_provider_failure_still_serves_the_composed_bundle(client, monkeypatch):
    """§44 rule 18 over the REAL bundle: the model failing costs the reader
    the prose, never the filed facts.

    The provider is the only thing replaced here — a real one cannot be made
    to fail on demand — so the bundle attached to the FAILED row is U1's
    composed one, and this proves the failure path carries it through
    unaltered rather than storing a stub.
    """
    import libs.llm as llm_pkg

    event_id = await _seeded_event()
    monkeypatch.setattr(
        llm_pkg,
        "get_recommendation_provider",
        lambda name: FakeProvider(raises=ProviderError("upstream 503")),
    )

    resp = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(NOW)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "FAILED"
    assert body["analysis"] is None
    assert "upstream 503" in (body["error"] or "")
    # The evidence half survived: U4 renders it under the failure banner.
    assert body["bundle"]["event"]["event_key"] == "EARNINGS:E2E:2026-08-27"
    assert body["bundle"]["consensus"]["status"] == "CONSENSUS_DATA_UNAVAILABLE"

    rows = await _rows()
    assert len(rows) == 1 and rows[0].status == "FAILED"


# ---------------------------------------------------------------------------
# Phase G — a MACRO event runs the whole analysis path (event spec §38-§41,
# §46; contract U3)
# ---------------------------------------------------------------------------


async def test_e2e_a_cpi_event_analyses_with_macro_context_and_no_issuer_sections(
    client,
):
    """POST /analysis on a CPI print — the real bundle, the real stub, the real
    validator, over an event that has NO TICKER.

    This is the cross-unit case Phase G created and no single unit's suite can
    prove on its own: the bundle composer must SKIP the three issuer seams
    (there is no company whose price, filings or news these would be) and FILL
    ``macro_context`` with the §38 packet instead, and the whole thing must
    still be a bundle the provider accepts and the validator passes.

    What is asserted is the SHAPE, not the prose: the stub's words are not
    analysis and its wording is not a contract. What IS a contract is that the
    three absent sections say WHY they are absent (a missing key would read as
    "this event has no news", which is false in a different direction), that
    the macro block is present and real rather than the pre-Phase-G
    placeholder, and that the §33 consensus string survives into a payload the
    model was handed — because that string is the only thing standing between
    a macro narrative and an invented expectation.
    """
    from apps.gateway.db import MacroObservationRow

    release_at = datetime(2026, 8, 12, 12, 30, tzinfo=timezone.utc)  # 08:30 ET
    async with SessionLocal() as s:
        s.add(
            EventRow(
                event_key="CPI:2026-08-12",
                event_type=EventType.CPI.value,
                title="CPI — July 2026",
                ticker=None,
                scheduled_at=release_at,
                session=EventSession.BEFORE_MARKET.value,
                status=EventStatus.CONFIRMED.value,
                source=EventSourceKind.GOVERNMENT_AGENCY.value,
                source_name="bls",
                agency="Bureau of Labor Statistics",
                release_period="2026-07",
                importance=95,
            )
        )
        # Two stored levels so the packet has a real previous release to quote
        # rather than only coverage notes: 324.000 -> 324.648 is +0.20%.
        s.add(
            MacroObservationRow(
                series_id="CUSR0000SA0",
                period="2026-06",
                value=324.000,
                release_at=datetime(2026, 7, 15, 12, 30, tzinfo=timezone.utc),
                release_basis="SCHEDULED",
                provider="bls",
            )
        )
        s.add(
            MacroObservationRow(
                series_id="CUSR0000SA0",
                period="2026-07",
                value=324.648,
                release_at=release_at,
                release_basis="SCHEDULED",
                provider="bls",
            )
        )
        await s.commit()
        event_id = (
            await s.execute(
                select(EventRow).where(EventRow.event_key == "CPI:2026-08-12")
            )
        ).scalar_one().id

    as_of = release_at + timedelta(days=1)
    resp = await client.post(
        f"/api/events/{event_id}/analysis", params={"as_of": _iso(as_of)}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in {"OK", "INVALID"}, body.get("error")

    bundle = body["bundle"]
    # The three ticker-dependent sections are absent AND explain themselves.
    for section in ("price_analysis", "fundamentals", "news"):
        assert bundle["coverage"][section]["available"] is False
        assert "no ticker" in bundle["coverage"][section]["reason"]

    # macro_context is the real §38 packet, not the Phase-F placeholder.
    macro = bundle["macro_context"]
    assert macro["kind"] == "macro_event_packet"
    assert macro.get("status") != "NOT_AVAILABLE_YET"
    assert macro["packet"]["previous_release"]["period"] == "2026-07"
    assert macro["packet"]["previous_release"]["actual"]["headline"][
        "value"
    ] == pytest.approx(0.20, abs=0.005)

    # §33 — no consensus, no surprise, in the payload the model was handed.
    assert macro["consensus_status"] == "CONSENSUS DATA UNAVAILABLE"
    assert bundle["consensus"]["status"] == "CONSENSUS_DATA_UNAVAILABLE"


async def test_the_macro_prompt_carries_the_component_question_end_to_end(client):
    """§41 — the extra question reaches the prompt for a macro bundle only.

    Asserted over the message the prompt builder actually produces from a
    composed bundle, rather than over a hand-written dict, so a change to the
    bundle's ``event.event_type`` spelling breaks this loudly instead of
    silently dropping the question.
    """
    from apps.gateway import event_evidence
    from libs.llm.event_analysis import build_user_message

    async with SessionLocal() as s:
        s.add(
            EventRow(
                event_key="CPI:2026-09-11",
                event_type=EventType.CPI.value,
                title="CPI — August 2026",
                ticker=None,
                scheduled_at=datetime(2026, 9, 11, 12, 30, tzinfo=timezone.utc),
                session=EventSession.BEFORE_MARKET.value,
                status=EventStatus.CONFIRMED.value,
                source=EventSourceKind.GOVERNMENT_AGENCY.value,
                source_name="bls",
                release_period="2026-08",
            )
        )
        await s.commit()
        row = (
            await s.execute(
                select(EventRow).where(EventRow.event_key == "CPI:2026-09-11")
            )
        ).scalar_one()
        bundle = await event_evidence.build_evidence_bundle(
            s, row, as_of=NOW, settings=get_settings()
        )

    message = build_user_message(bundle)
    assert "COMPONENT" in message
    assert "LLM ANALYSIS" in message
    assert "CONSENSUS" in message.upper()
