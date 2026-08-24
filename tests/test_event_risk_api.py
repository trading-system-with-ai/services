"""Phase K event risk on the wire (§62-§67) — SHADOW.

Four surfaces, one guarantee.

The surfaces:

- ``GET /api/events/{id}/risk`` — the §63 snapshot plus the §66 options panel;
- the ``RISK_DECISION`` audit detail's ``shadow.event`` block;
- a trade plan's ``event_risk`` key, computed FRESH on every read;
- the seam itself (``apps/gateway/event_risk.py``), whose reads are pinned
  directly where an endpoint would hide them.

The guarantee (§65), which the load-bearing tests here exist to protect: **none
of it changes a Tier 0 decision.** ``test_a_planted_event_leaves_the_approval_
byte_identical`` runs the same preview twice — once with no event in the
registry and once with an EXTREME one planted — and asserts the whole ``risk``
block and every gate come back byte-identical. ``test_a_raising_seam_leaves_the
_order_path_intact`` monkeypatches the seam to explode and asserts the preview
still succeeds with ``shadow.event = {"error": ...}``.

Two honesty properties are pinned as hard as the guarantee:

- **UNKNOWN is not LOW** (§63). An event with no stored straddle and no prior
  print classifies ``UNKNOWN`` with a ``reason``, never ``LOW``.
- **Every historical statistic carries its ``n``** (§64). Asserted on every
  payload that carries a ``historical`` mapping, including the empty one.

Fixture arithmetic is stated, not derived: prior prints are stored with
``actual_move_pct`` values this file writes literally (``[-9, 11, -8, 10]``),
so a median of 9.5 is a number a reader can check by hand rather than one the
fixture computed the same way the code under test does.
"""
from datetime import datetime, timedelta, timezone

from apps.gateway.execution import gate_chain

import pytest

from apps.gateway import event_risk as seam
from apps.gateway.db import (
    EventOptionMetricRow,
    EventRow,
    Position,
    SessionLocal,
)
from apps.gateway.routers import orders as orders_router
from libs.trading_core.events.implied_move import (
    BASIS_HISTORICAL,
    BASIS_LIVE,
    STATUS_NO_DATA,
    STATUS_OK,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)
from libs.trading_core.risk.event_risk import (
    EVENT_RISK_MODEL_VERSION,
    STATE_EXTREME,
    STATE_LADDER,
    STATE_UNKNOWN,
)

from .test_order_preview import BULL_TICKER, authorize, preview

#: The 15 keys ``classify_event_risk`` always returns — the U1/U2/U3 contract.
#: Asserted as a SUPERSET check on the payload (the seam adds its own
#: provenance keys), because a missing one breaks the Risk tab silently.
SNAPSHOT_KEYS = {
    "event_type",
    "time_to_event_days",
    "historical",
    "implied",
    "expected_move_pct",
    "expected_move_basis",
    "position_exposure_usd",
    "exposure_share",
    "option_greeks",
    "event_risk_state",
    "sensitivity",
    "drivers",
    "caveats",
    "reason",
    "model_version",
}

#: §64: the four statistics and the sample size behind them, in ONE mapping.
HISTORICAL_KEYS = {"median_abs", "p75_abs", "p90_abs", "max_abs", "n"}

#: The prior prints' REALIZED moves, written literally so every statistic in
#: this file is hand-checkable. Absolutes sorted: [8, 9, 10, 11] -> median at
#: nearest rank ceil(0.5*4)=2 -> 9; p75 rank 3 -> 10; p90 rank 4 -> 11; max 11.
PRIOR_MOVES = [-9.0, 11.0, -8.0, 10.0]
PRIOR_MEDIAN_ABS = 9.0
PRIOR_P75_ABS = 10.0
PRIOR_MAX_ABS = 11.0


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


async def _add_event(
    *,
    key: str,
    ticker: str | None,
    when: datetime,
    event_type: EventType = EventType.EARNINGS,
    status: EventStatus = EventStatus.CONFIRMED,
) -> int:
    async with SessionLocal() as s:
        row = EventRow(
            event_key=key,
            event_type=event_type.value,
            title=f"{ticker or 'macro'} {event_type.value}",
            ticker=ticker,
            scheduled_at=when,
            event_timezone="America/New_York",
            session=EventSession.AFTER_MARKET.value,
            status=status.value,
            source=EventSourceKind.COMPANY_IR_SEC.value,
            source_name="sec_edgar",
            revision_history=[],
        )
        s.add(row)
        await s.commit()
        return row.id


def _as_stored_fraction(pct: float | None) -> float | None:
    """A PERCENT number written by a test, stored the way PRODUCTION stores it.

    THE UNIT SEAM THIS FIXTURE EXISTS TO RESPECT. Every move column on
    ``event_option_metrics`` is written by the event-options layer as a
    FRACTION of spot — ``implied_move.pct`` is ``points / spot`` and
    ``actual_move_pct`` is ``post.close / pre_close - 1.0`` — so an 8.8% print
    is persisted as ``0.088``, never as ``8.8``.

    The arguments above stay PERCENT numbers because that is what a reader can
    check by hand against the assertions ("median of 9.5"), but they are
    converted HERE so the rows this fixture plants are byte-identical in
    convention to the rows a real backfill writes. A fixture that stored
    ``8.8`` would be testing a database state that cannot occur, and would
    have hidden a 100x understatement of every event's risk.
    """
    return None if pct is None else pct / 100.0


async def _add_metrics(
    event_id: int,
    *,
    as_of: datetime,
    basis: str = BASIS_HISTORICAL,
    implied_move_pct: float | None = None,
    actual_move_pct: float | None = None,
    iv_before: float | None = None,
    iv_crush_pct: float | None = None,
    status: str = STATUS_OK,
) -> None:
    """Plant one metrics row. The ``*_pct`` arguments are PERCENT numbers
    (``8.8`` = 8.8%) and are stored as the FRACTIONS production stores;
    ``iv_before`` is already a fraction (``0.62`` = 62% IV) and is stored
    as given, matching ``implied_vol``'s own output."""
    async with SessionLocal() as s:
        s.add(
            EventOptionMetricRow(
                event_id=event_id,
                as_of=as_of,
                basis=basis,
                implied_move_pct=_as_stored_fraction(implied_move_pct),
                actual_move_pct=_as_stored_fraction(actual_move_pct),
                iv_before=iv_before,
                iv_crush_pct=_as_stored_fraction(iv_crush_pct),
                status=status,
                notes={},
            )
        )
        await s.commit()


async def _add_position(ticker: str, *, quantity: int, avg_price: float) -> None:
    async with SessionLocal() as s:
        s.add(
            Position(
                ticker=ticker,
                instrument="LONG_STOCK",
                quantity=quantity,
                avg_price=avg_price,
                max_loss=quantity * avg_price * 0.1,
                status="OPEN",
                opened_at=datetime.now(timezone.utc),
            )
        )
        await s.commit()


async def _plant_upcoming_earnings(
    ticker: str = BULL_TICKER,
    *,
    days_away: float = 1.3,
    implied_move_pct: float | None = 8.8,
    prior_moves: list[float] | None = None,
    status: EventStatus = EventStatus.CONFIRMED,
) -> dict:
    """One upcoming print with four PAST comparable prints behind it.

    ``days_away=1.3`` and ``implied_move_pct=8.8`` are the §65 panel's worked
    example: base HIGH (8.8 >= 8), imminent (1.3 <= 3) -> one level up ->
    EXTREME. Stated here so the expected state in the assertions is a
    prediction from the documented table, not a transcription of output.
    """
    now = datetime.now(timezone.utc)
    when = now + timedelta(days=days_away)
    event_id = await _add_event(
        key=f"EARNINGS:{ticker}:{when.date().isoformat()}",
        ticker=ticker,
        when=when,
        status=status,
    )
    if implied_move_pct is not None:
        await _add_metrics(
            event_id,
            as_of=now,
            basis=BASIS_LIVE,
            implied_move_pct=implied_move_pct,
            iv_before=0.62,
        )
    prior_ids: list[int] = []
    for index, move in enumerate(
        PRIOR_MOVES if prior_moves is None else prior_moves
    ):
        past = when - timedelta(days=91 * (index + 1))
        pid = await _add_event(
            key=f"EARNINGS:{ticker}:{past.date().isoformat()}",
            ticker=ticker,
            when=past,
        )
        await _add_metrics(
            pid,
            as_of=past,
            implied_move_pct=6.0,
            actual_move_pct=move,
            iv_before=0.70,
            iv_crush_pct=-30.0 - index,
        )
        prior_ids.append(pid)
    return {"event_id": event_id, "prior_ids": prior_ids, "when": when, "now": now}


async def _latest_risk_decision(client, ticker: str) -> dict:
    r = await client.get("/api/audit", params={"entity_id": ticker})
    events = [e for e in r.json() if e["action"] == "RISK_DECISION"]
    assert events, "no RISK_DECISION event recorded"
    return events[0]


def _assert_n_everywhere(historical: dict) -> None:
    """§64: the four statistics and ``n`` travel in ONE mapping, always.

    With ``n == 0`` every statistic is ``None`` — never ``0.0``, which would
    read as "this name does not move" rather than "nobody measured it".
    """
    assert set(historical) == HISTORICAL_KEYS
    assert isinstance(historical["n"], int)
    if historical["n"] == 0:
        for key in ("median_abs", "p75_abs", "p90_abs", "max_abs"):
            assert historical[key] is None, key


# ---------------------------------------------------------------------------
# (a) GET /api/events/{id}/risk
# ---------------------------------------------------------------------------


async def test_event_risk_endpoint_serves_the_snapshot_and_options(client):
    """The full §63 + §66 payload for a planted print with real history."""
    planted = await _plant_upcoming_earnings()
    r = await client.get(f"/api/events/{planted['event_id']}/risk")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["available"] is True
    assert body["reason"] is None
    assert body["ticker"] == BULL_TICKER
    # §65: the enforcement mode is a literal in the payload, not a comment.
    assert body["enforcement"] == "SHADOW"
    assert body["model_version"] == EVENT_RISK_MODEL_VERSION
    assert body["nav_basis"] == "COST"

    snap = body["snapshot"]
    assert SNAPSHOT_KEYS <= set(snap)
    # The §65 worked example: base HIGH from an 8.8% implied move, one level
    # up for imminence (1.3 days) -> EXTREME.
    assert snap["event_risk_state"] == STATE_EXTREME
    assert snap["expected_move_pct"] == pytest.approx(8.8)
    assert snap["expected_move_basis"] == "IMPLIED"
    assert snap["implied"]["pct"] == pytest.approx(8.8)
    assert snap["implied"]["basis"] == BASIS_LIVE
    assert snap["reason"] is None  # non-null ONLY in UNKNOWN
    assert snap["drivers"], "an EXTREME state must say why"
    assert snap["model_version"] == EVENT_RISK_MODEL_VERSION

    # §64 — the statistics are the hand-checked ones, with n beside them.
    hist = snap["historical"]
    _assert_n_everywhere(hist)
    assert hist["n"] == len(PRIOR_MOVES)
    assert hist["median_abs"] == pytest.approx(PRIOR_MEDIAN_ABS)
    assert hist["p75_abs"] == pytest.approx(PRIOR_P75_ABS)
    assert hist["max_abs"] == pytest.approx(PRIOR_MAX_ABS)

    opts = body["options"]
    # PERCENT numbers, the one convention this payload speaks end to end: a
    # stored IV of 0.62 is 62% and a stored implied move of 0.088 is 8.8%.
    # Pinned here because serving the raw stored FRACTIONS beside a snapshot
    # that speaks percent is exactly the 100x confusion this surface must not
    # ship — it renders an EXTREME print as "implied move 0.09%".
    assert opts["event_iv"] == pytest.approx(62.0)
    assert opts["implied_move_pct"] == pytest.approx(8.8)
    # The snapshot and the options block agree about the SAME measurement.
    assert opts["implied_move_pct"] == pytest.approx(snap["implied"]["pct"])
    assert opts["implied_basis"] == BASIS_LIVE
    assert opts["is_live_basis"] is True
    # §44 rule 18: the crush this platform cannot know is NAMED, not invented.
    assert opts["expected_iv_crush"] == "NO_DATA"
    assert opts["expected_iv_crush_note"]
    # The realized crushes ARE measured — with their n (§64).
    _assert_n_everywhere(opts["historical_iv_crush"])
    assert opts["historical_iv_crush"]["n"] == len(PRIOR_MOVES)
    _assert_n_everywhere(opts["historical_implied_move"])
    # §66: the sentence that explains how being directionally right loses.
    assert "realized move" in opts["explainer"]
    assert "implied move" in opts["explainer"]


async def test_event_with_no_metrics_is_unknown_not_low(client):
    """§63's sharpest honesty rule: UNKNOWN is not LOW.

    An event with no stored straddle and no prior print has not been measured.
    Reporting LOW would be a fabricated permission — the one of the five
    values that reads as "go ahead" — so the state is UNKNOWN with a reason
    and the coverage block names the backfill that would fill the gap.
    """
    now = datetime.now(timezone.utc)
    event_id = await _add_event(
        key="EARNINGS:BARE:2099-01-01",
        ticker="BARE",
        when=now + timedelta(days=2),
    )
    body = (await client.get(f"/api/events/{event_id}/risk")).json()
    snap = body["snapshot"]

    assert snap["event_risk_state"] == STATE_UNKNOWN
    assert snap["event_risk_state"] not in STATE_LADDER  # it is NOT a rung
    assert snap["reason"], "UNKNOWN must carry the reason it was not measured"
    assert snap["expected_move_pct"] is None
    assert snap["expected_move_basis"] == "NONE"
    # §64 holds even here: n is present and 0, statistics are null not zero.
    _assert_n_everywhere(snap["historical"])
    assert snap["historical"]["n"] == 0
    assert body["snapshot"]["coverage"]["reason"] is not None
    assert "backfill" in body["snapshot"]["coverage"]["reason"]


async def test_event_without_a_ticker_is_available_false_not_404(client):
    """A CPI release has no issuer whose position this would be. The row
    exists, so 404 would be a lie; ``available: false`` with a reason is the
    honest answer."""
    now = datetime.now(timezone.utc)
    event_id = await _add_event(
        key="CPI:2099-01",
        ticker=None,
        when=now + timedelta(days=2),
        event_type=EventType.CPI,
    )
    r = await client.get(f"/api/events/{event_id}/risk")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"]
    assert body["snapshot"] is None
    assert body["options"] is None
    assert body["enforcement"] == "SHADOW"


async def test_missing_event_is_a_404(client):
    r = await client.get("/api/events/999999/risk")
    assert r.status_code == 404


async def test_market_wide_fomc_flag_rides_along_separately(client):
    """§62: an FOMC decision moves EVERY position, so it is reported beside
    the ticker's own state and never folded into it.

    Pinned by planting an imminent FOMC and asserting the EARNINGS state is
    the SAME as it was without one — a market-wide event that silently bumped
    a single name's state would make "EXTREME" mean two different things.
    """
    planted = await _plant_upcoming_earnings()
    before = (await client.get(f"/api/events/{planted['event_id']}/risk")).json()
    assert before["market_wide"] is None

    await _add_event(
        key="FOMC_DECISION:2099-01-01",
        ticker=None,
        when=datetime.now(timezone.utc) + timedelta(days=2),
        event_type=EventType.FOMC_DECISION,
    )
    after = (await client.get(f"/api/events/{planted['event_id']}/risk")).json()
    assert after["market_wide"] is not None
    assert after["market_wide"]["event_type"] == EventType.FOMC_DECISION.value
    assert after["market_wide"]["days_away"] == pytest.approx(2.0, abs=0.05)
    # The ticker's OWN state is untouched by the market-wide flag.
    assert (
        after["snapshot"]["event_risk_state"]
        == before["snapshot"]["event_risk_state"]
    )


async def test_estimated_date_adds_a_caveat_and_never_lowers_the_state(client):
    """§7/§11: a derived date is labelled, never presented as a fact — and
    never treated as less risky, because an uncertain date is not a smaller
    gap."""
    confirmed = await _plant_upcoming_earnings(ticker="CONF")
    estimated = await _plant_upcoming_earnings(
        ticker="ESTM", status=EventStatus.ESTIMATED
    )
    a = (await client.get(f"/api/events/{confirmed['event_id']}/risk")).json()
    b = (await client.get(f"/api/events/{estimated['event_id']}/risk")).json()

    assert a["snapshot"]["event_risk_state"] == b["snapshot"]["event_risk_state"]
    assert a["snapshot"]["is_estimated"] is False
    assert b["snapshot"]["is_estimated"] is True
    assert any("ESTIMATED" in c for c in b["snapshot"]["caveats"])


async def test_exposure_share_bumps_the_state_and_is_never_a_fabricated_zero(client):
    """A held position raises the state; NO held position leaves the share an
    honest ``None`` with its caveat, never ``0.0``."""
    planted = await _plant_upcoming_earnings(
        ticker="EXPO", days_away=30.0, implied_move_pct=4.5
    )
    # No position: MODERATE base (4.5 >= 4), no imminence, no exposure bump.
    bare = (await client.get(f"/api/events/{planted['event_id']}/risk")).json()
    assert bare["snapshot"]["exposure_share"] is None
    assert bare["snapshot"]["position_exposure_usd"] is None
    assert any("exposure" in c.lower() for c in bare["snapshot"]["caveats"])
    base_state = bare["snapshot"]["event_risk_state"]

    # A position worth a big slice of a cost-basis NAV bumps it (>=25% is two
    # levels, >=10% one) — the assertion is only that it MOVED UP, so the
    # exact rung stays the pure library's business.
    await _add_position("EXPO", quantity=400, avg_price=250.0)
    heavy = (await client.get(f"/api/events/{planted['event_id']}/risk")).json()
    assert heavy["snapshot"]["exposure_share"] is not None
    assert heavy["snapshot"]["position_exposure_usd"] == pytest.approx(100_000.0)
    assert STATE_LADDER.index(heavy["snapshot"]["event_risk_state"]) > STATE_LADDER.index(
        base_state
    )


async def test_no_data_metrics_never_enter_the_sample(client):
    """A NO_DATA row is the server RETRACTING its own computation; a number
    sitting beside it is withdrawn and must not become a historical move."""
    now = datetime.now(timezone.utc)
    when = now + timedelta(days=5)
    event_id = await _add_event(
        key="EARNINGS:RETR:2099-01-01", ticker="RETR", when=when
    )
    past = when - timedelta(days=91)
    pid = await _add_event(
        key="EARNINGS:RETR:2098-10-01", ticker="RETR", when=past
    )
    # A retracted row carrying a number: it must be IGNORED, not sampled.
    await _add_metrics(
        pid, as_of=past, actual_move_pct=99.0, status=STATUS_NO_DATA
    )
    body = (await client.get(f"/api/events/{event_id}/risk")).json()
    assert body["snapshot"]["historical"]["n"] == 0
    assert body["snapshot"]["event_risk_state"] == STATE_UNKNOWN


# ---------------------------------------------------------------------------
# (b) shadow.event in the RISK_DECISION audit row — and the guarantee
# ---------------------------------------------------------------------------


async def test_shadow_event_appears_in_the_risk_decision(client):
    """The §65 block lands in the audit row beside ``shadow.statistical``,
    carrying its enforcement mode and a hypothetical-only verdict."""
    await _plant_upcoming_earnings()
    await authorize(client, BULL_TICKER)
    await preview(client, BULL_TICKER)

    event = await _latest_risk_decision(client, BULL_TICKER)
    shadow = event["details"]["shadow"]
    assert "event" in shadow
    block = shadow["event"]
    assert block["enforcement"] == "SHADOW"
    assert block["model_version"] == EVENT_RISK_MODEL_VERSION
    assert block["snapshot"]["event_risk_state"] == STATE_EXTREME
    _assert_n_everywhere(block["snapshot"]["historical"])
    # The verdict vocabulary is SUBJUNCTIVE by contract — nothing happened.
    assert set(block["verdict"]) == {"would_warn", "would_cap_qty"}
    assert block["verdict"]["would_warn"] is True
    assert "SHADOW" in block["note"]
    # Every cap that IS emitted is a hypothetical row with the real numbers.
    for cap in block["caps"]:
        assert cap["code"] == "EVENT_EXPOSURE_CAP"
        assert cap["layer"] == "CONCENTRATION"
        assert cap["sentence"]
        assert "historical_n" in cap["measured"]


async def test_a_planted_event_leaves_the_approval_byte_identical(client):
    """THE LOAD-BEARING TEST (§65). The same preview, run with and without an
    EXTREME event in the registry, must produce a byte-identical ``risk``
    block and byte-identical gates.

    This is the assertion that would fail the instant anyone wired the event
    cap into ``assess(extra_caps=...)``. The cap is computed either way — the
    audit row below proves it exists — and it still changes nothing.

    ``shadow_statistical`` is compared with its ``as_of`` stamp dropped: that
    key is a wall-clock instant which differs between any two runs by
    construction, and including it would make this test fail for a reason
    that has nothing to do with the guarantee it defends. EVERY OTHER KEY of
    the shadow block is compared, including the whole hypothetical caps list
    and the hypothetical verdict — which is precisely where a promoted event
    cap would show up.
    """
    def _decision_surface(body: dict) -> dict:
        risk = dict(body["risk"])
        shadow = dict(risk.get("shadow_statistical") or {})
        shadow.pop("as_of", None)
        risk["shadow_statistical"] = shadow
        return risk

    await authorize(client, BULL_TICKER)
    before = await preview(client, BULL_TICKER)

    await _plant_upcoming_earnings()
    after = await preview(client, BULL_TICKER)

    assert _decision_surface(after) == _decision_surface(before)
    assert after["gates"] == before["gates"]
    assert after["why_not_trade"] == before["why_not_trade"]
    # The numbers that ARE the decision, spelled out so a future refactor of
    # the helper above cannot quietly weaken the comparison.
    for key in ("decision", "approved_quantity", "reason_codes", "trade_risk_usd"):
        assert after["risk"][key] == before["risk"][key], key

    # ...and the event layer really did run and really did produce a state.
    event = await _latest_risk_decision(client, BULL_TICKER)
    block = event["details"]["shadow"]["event"]
    assert block["snapshot"]["event_risk_state"] == STATE_EXTREME


async def test_an_extreme_event_cap_binds_only_the_hypothetical_verdict(client):
    """An emitted EVENT cap joins the ONE shadow verdict the statistical layer
    computes — the same list the STRESS cap joins — and the Tier 0 approved
    quantity above it is untouched.

    A cap only exists when the limit is NOT satisfied, so this test asserts
    the WIRING (a cap, if present, appears in ``shadow.statistical.caps.rows``
    with its own layer) rather than forcing a cap into existence with a
    fixture that would pin the pure library's thresholds a second time.
    """
    await _plant_upcoming_earnings()
    await _add_position(BULL_TICKER, quantity=500, avg_price=200.0)
    await authorize(client, BULL_TICKER)
    body = await preview(client, BULL_TICKER)

    event = await _latest_risk_decision(client, BULL_TICKER)
    details = event["details"]
    event_caps = details["shadow"]["event"]["caps"]
    stat = details["shadow"]["statistical"]
    if event_caps and "caps" in stat:
        codes = {row["code"] for row in stat["caps"]["rows"]}
        assert "EVENT_EXPOSURE_CAP" in codes
        # It is in the SHADOW caps list — and the decision above still is not.
        assert details["approved_quantity"] == body["risk"]["approved_quantity"]
    # Whatever happened above: the engine recorded no event reason code.
    assert not any("EVENT" in code for code in details["reason_codes"])


async def test_a_raising_seam_leaves_the_order_path_intact(client, monkeypatch):
    """Failure isolation (§65): the event seam is research context, and
    research context may never cost a user their order path.

    The seam is monkeypatched to raise. The preview must still succeed, the
    decision must be unchanged, and ``shadow.event`` must carry the error
    rather than vanishing — a silently absent block would look like "no event"
    and hide a broken layer.
    """
    await _plant_upcoming_earnings()
    await authorize(client, BULL_TICKER)
    healthy = await preview(client, BULL_TICKER)

    async def _boom(*a, **k):
        raise RuntimeError("event seam exploded")

    monkeypatch.setattr(gate_chain.event_risk, "shadow_event_block", _boom)
    broken = await preview(client, BULL_TICKER)

    for key in ("decision", "approved_quantity", "reason_codes", "trade_risk_usd"):
        assert broken["risk"][key] == healthy["risk"][key], key
    assert broken["gates"] == healthy["gates"]
    event = await _latest_risk_decision(client, BULL_TICKER)
    block = event["details"]["shadow"]["event"]
    assert "error" in block
    assert "RuntimeError" in block["error"]


async def test_shadow_event_is_none_when_the_ticker_has_no_event(client):
    """No planted event: the block is present and says so, rather than being
    absent (which a reader could not distinguish from a missing feature)."""
    await authorize(client, BULL_TICKER)
    await preview(client, BULL_TICKER)
    event = await _latest_risk_decision(client, BULL_TICKER)
    block = event["details"]["shadow"]["event"]
    assert block["snapshot"] is None
    assert block["reason"] == seam.NO_EVENT_REASON
    assert block["caps"] == []
    assert block["verdict"] == {"would_warn": False, "would_cap_qty": None}
    assert block["enforcement"] == "SHADOW"


# ---------------------------------------------------------------------------
# (c) The trade plan's event_risk — fresh on read
# ---------------------------------------------------------------------------


async def test_plan_event_risk_is_computed_fresh_on_read(client):
    """§65: a stored plan can never carry a stale countdown.

    A plan is generated with NO event in the registry (so ``event_risk`` is
    null), the event is planted afterwards, and the SAME stored plan is read
    again — the panel now renders. The stored ``preview`` blob is untouched
    throughout, which is what "fresh on read" means.
    """
    await authorize(client, BULL_TICKER)
    r = await client.post("/api/plans/generate", json={"ticker": BULL_TICKER})
    assert r.status_code == 201, r.text
    plan_id = r.json()["id"]
    assert r.json()["event_risk"] is None

    stored_preview = (await client.get(f"/api/plans/{plan_id}")).json()["preview"]

    await _plant_upcoming_earnings()
    detail = (await client.get(f"/api/plans/{plan_id}")).json()
    block = detail["event_risk"]
    assert block is not None
    assert block["enforcement"] == "SHADOW"
    assert block["snapshot"]["event_risk_state"] == STATE_EXTREME
    assert block["computed_at"]
    _assert_n_everywhere(block["snapshot"]["historical"])
    # A research plan owns no position, so the share is an honest null.
    assert block["snapshot"]["exposure_share"] is None
    # The stored preview did NOT change — nothing was written on a read.
    assert detail["preview"] == stored_preview


async def test_plan_list_carries_event_risk_too(client):
    """The list view and the detail view must not disagree about a plan."""
    await _plant_upcoming_earnings()
    await authorize(client, BULL_TICKER)
    r = await client.post("/api/plans/generate", json={"ticker": BULL_TICKER})
    assert r.status_code == 201
    rows = (await client.get("/api/plans")).json()
    assert rows
    assert rows[0]["event_risk"] is not None
    assert rows[0]["event_risk"]["snapshot"]["event_risk_state"] == STATE_EXTREME


async def test_a_raising_plan_seam_never_breaks_the_plan_read(client, monkeypatch):
    """A plan read is a core surface; the event panel is additive research
    context. Its failure is its own absence with a reason, never a 500."""
    await authorize(client, BULL_TICKER)
    r = await client.post("/api/plans/generate", json={"ticker": BULL_TICKER})
    plan_id = r.json()["id"]

    from apps.gateway.routers import plans as plans_router

    async def _boom(*a, **k):
        raise RuntimeError("plan event seam exploded")

    monkeypatch.setattr(plans_router.event_risk, "plan_event_risk", _boom)
    detail = await client.get(f"/api/plans/{plan_id}")
    assert detail.status_code == 200
    assert "RuntimeError" in detail.json()["event_risk"]["error"]


# ---------------------------------------------------------------------------
# (d) The seam itself
# ---------------------------------------------------------------------------


async def test_upcoming_event_for_picks_the_nearest_future_print(client):
    """Nearest, not most important, and STRICTLY future — an event at ``now``
    has already happened for a trade being placed now."""
    now = datetime.now(timezone.utc)
    await _add_event(
        key="EARNINGS:NEAR:past", ticker="NEAR", when=now - timedelta(days=1)
    )
    far = await _add_event(
        key="EARNINGS:NEAR:far", ticker="NEAR", when=now + timedelta(days=10)
    )
    near = await _add_event(
        key="EARNINGS:NEAR:near", ticker="NEAR", when=now + timedelta(days=2)
    )
    async with SessionLocal() as s:
        row = await seam.upcoming_event_for(s, "NEAR", now=now)
        assert row is not None and row.id == near
        # Outside the horizon: nothing, rather than the far one.
        tight = await seam.upcoming_event_for(s, "NEAR", now=now, horizon_days=1)
        assert tight is None
        # Widening it finds the far one once the near one is gone.
        later = await seam.upcoming_event_for(
            s, "NEAR", now=now + timedelta(days=3)
        )
        assert later is not None and later.id == far


async def test_canceled_events_are_not_risks(client):
    """A canceled print is not a catalyst — it is a print that will not
    happen."""
    now = datetime.now(timezone.utc)
    await _add_event(
        key="EARNINGS:GONE:x",
        ticker="GONE",
        when=now + timedelta(days=2),
        status=EventStatus.CANCELED,
    )
    async with SessionLocal() as s:
        assert await seam.upcoming_event_for(s, "GONE", now=now) is None


async def test_position_exposure_is_none_not_zero_when_nothing_is_held(client):
    """The distinction the classifier depends on: "no exposure measured" is
    not "zero exposure", and a fabricated 0.0 would make every unfunded
    snapshot look small."""
    async with SessionLocal() as s:
        assert await seam.position_exposure_for(s, "NOTHELD") is None
    await _add_position("HELD", quantity=10, avg_price=50.0)
    async with SessionLocal() as s:
        assert await seam.position_exposure_for(s, "HELD") == pytest.approx(500.0)


def test_greeks_from_rows_skips_not_ok_rows_and_returns_none_when_empty():
    """``data_ok: false`` rows carry documented ZEROS so one bad position
    cannot corrupt the portfolio TOTAL. Borrowing them here would turn "the
    contract fell off today's chain" into "this position has no convexity" —
    the opposite claim — so they are skipped, and an all-skipped ticker is
    ``None`` rather than a dict of zeros."""
    rows = [
        {"ticker": "AAA", "gamma": 0.0, "vega_usd": 0.0, "theta_usd_per_day": 0.0,
         "data_ok": False},
        {"ticker": "BBB", "gamma": 3.0, "vega_usd": 90.0, "theta_usd_per_day": -12.0,
         "data_ok": True},
        {"ticker": "BBB", "gamma": 1.0, "vega_usd": 10.0, "theta_usd_per_day": -3.0,
         "data_ok": True},
    ]
    assert seam.greeks_from_rows(rows, "AAA") is None
    assert seam.greeks_from_rows(rows, "ZZZ") is None
    assert seam.greeks_from_rows(None, "BBB") is None
    net = seam.greeks_from_rows(rows, "BBB")
    assert net == {"gamma": 4.0, "vega": 100.0, "theta": -15.0}


async def test_greeks_wired_into_the_snapshot_drive_sensitivity_not_state(client):
    """§66: convexity is a SECOND axis. A big vega raises ``sensitivity`` and
    leaves ``event_risk_state`` exactly where it was — folding greeks into the
    state would make "HIGH" mean two different things depending on whether the
    holder owns stock or options."""
    # 10 days out: inside the default horizon but OUTSIDE the 3-day imminence
    # window, so the state below is driven purely by the expected move and is
    # not sitting at the EXTREME ceiling where a bump would be invisible.
    planted = await _plant_upcoming_earnings(ticker="GRK", days_away=10.0)
    async with SessionLocal() as s:
        row = await seam.upcoming_event_for(s, "GRK", now=datetime.now(timezone.utc))
        flat = await seam.snapshot_for_event(
            s, row, now=datetime.now(timezone.utc)
        )
        juiced = await seam.snapshot_for_event(
            s,
            row,
            now=datetime.now(timezone.utc),
            option_greeks={"gamma": 40.0, "vega": 500.0, "theta": -80.0},
        )
    assert flat["option_greeks"] is None  # never a dict of zeros
    assert flat["sensitivity"] == "LOW"
    assert juiced["option_greeks"] == {"gamma": 40.0, "vega": 500.0, "theta": -80.0}
    assert juiced["sensitivity"] == "HIGH"
    assert juiced["event_risk_state"] == flat["event_risk_state"]
    assert planted["event_id"] == row.id


async def test_the_snapshot_is_deterministic(client):
    """§63 forbids an LLM assigning the state, and determinism is the property
    that makes the prohibition checkable: two calls over the same stored rows
    at the same instant produce byte-identical payloads."""
    planted = await _plant_upcoming_earnings(ticker="DET")
    stamp = datetime.now(timezone.utc)
    async with SessionLocal() as s:
        row = await seam.upcoming_event_for(s, "DET", now=stamp)
        first = await seam.snapshot_for_event(s, row, now=stamp, nav=100_000.0)
        second = await seam.snapshot_for_event(s, row, now=stamp, nav=100_000.0)
    assert first == second
    assert planted["event_id"] == row.id


def test_no_llm_or_network_in_the_seam():
    """§63/§27, asserted statically over the seam's own source.

    Tokenized rather than substring-matched so the prose above — which
    legitimately discusses LLMs and fetching in order to forbid them — cannot
    fail its own rule. Only executable tokens are scanned.
    """
    import io
    import tokenize
    from pathlib import Path

    source = Path(seam.__file__).read_text()
    banned = {
        "llm", "openai", "anthropic", "prompt", "completion",
        "httpx", "requests", "aiohttp", "urllib", "fetch",
    }
    found: set[str] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            continue
        if tok.type == tokenize.NAME and tok.string.lower() in banned:
            found.add(tok.string)
    assert not found, f"the event-risk seam must not reference {sorted(found)}"


async def test_market_wide_flag_ignores_a_distant_meeting(client):
    """A meeting three weeks out is not a flag on today's trade — a permanent
    low-grade warning is a warning nobody reads."""
    now = datetime.now(timezone.utc)
    await _add_event(
        key="FOMC_DECISION:far",
        ticker=None,
        when=now + timedelta(days=21),
        event_type=EventType.FOMC_DECISION,
    )
    async with SessionLocal() as s:
        assert await seam.market_wide_flag(s, now=now) is None
        near = now + timedelta(days=19)
        assert await seam.market_wide_flag(s, now=near) is not None


# ---------------------------------------------------------------------------
# The unit seam — a stored FRACTION is not a percent number
# ---------------------------------------------------------------------------


async def test_stored_fractions_become_percent_numbers_not_a_100x_understatement(
    client,
):
    """The regression this surface can least afford, pinned end to end.

    ``event_option_metrics`` stores every move column as a FRACTION of spot
    (``implied_move.pct`` is ``points / spot``; ``actual_move_pct`` is
    ``post.close / pre_close - 1.0``), while the Phase K classifier documents
    the OPPOSITE convention — every move it takes is a PERCENT number. Handing
    one to the other unconverted understates every event by 100x, and it does
    so SILENTLY: the number still looks like a number.

    This test writes the rows by hand at the storage layer, bypassing
    ``_add_metrics``' own conversion, so it pins the SEAM rather than the
    fixture. A real 8.8% print stored as ``0.088`` must classify EXTREME (8.8
    clears the 8.0 HIGH threshold, and 1.3 days is inside the imminence
    window that bumps it one level) — never MODERATE off a "0.09% move".
    """
    now = datetime.now(timezone.utc)
    when = now + timedelta(days=1.3)
    event_id = await _add_event(key="EARNINGS:FRAC:x", ticker="FRAC", when=when)
    async with SessionLocal() as s:
        # Written RAW, exactly as the event-options backfill persists them.
        s.add(
            EventOptionMetricRow(
                event_id=event_id,
                as_of=now,
                basis=BASIS_LIVE,
                implied_move_pct=0.088,  # 8.8%, stored as a fraction
                iv_before=0.62,  # 62% IV, already a fraction
                status=STATUS_OK,
                notes={},
            )
        )
        await s.commit()

    async with SessionLocal() as s:
        snap = await seam.snapshot_for(s, "FRAC", now=now)

    assert snap["implied"]["pct"] == pytest.approx(8.8), (
        "a stored 0.088 must reach the classifier as 8.8, not 0.088"
    )
    assert snap["expected_move_pct"] == pytest.approx(8.8)
    assert snap["event_risk_state"] == STATE_EXTREME, (
        "an 8.8% move 1.3 days out is EXTREME; reading the fraction as a "
        "percent lands it under the 4% MODERATE floor and hides the print"
    )
    # And the §66 block agrees with the snapshot about the same measurement,
    # so no consumer has to know which storage column a key came from.
    async with SessionLocal() as s:
        row = await seam.upcoming_event_for(s, "FRAC", now=now)
        opts = await seam.options_risk_block(s, row, now=now)
    assert opts["implied_move_pct"] == pytest.approx(8.8)
    assert opts["event_iv"] == pytest.approx(62.0)
