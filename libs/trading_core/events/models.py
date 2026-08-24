"""Typed event domain objects and the reconciliation rules (event spec §6,
§7, §15, §78).

Pure stdlib, deterministic, no I/O — the persistence and provider layers
(`apps/gateway/event_calendar.py`, `libs/event_calendar/`) call *into* this
module, never the other way round (audit §7.4 static guard: nothing here may
import `libs.market_data` or `libs.event_calendar`).

Three rules live here, and they are the whole reason event ingestion can run
unattended:

1. **Identity** — :func:`same_event` decides whether an incoming candidate
   is the event we already have. Equal ``event_key`` is the fast path; the
   ESTIMATED-drift windows (EARNINGS ±21d, FOMC_MINUTES ±7d) absorb the case
   where a cadence estimate and the confirmed date land on different days.
2. **Authority** — :func:`source_rank` (§78). A lower-authority source never
   overwrites a higher-authority one, and an LLM never writes a date at all.
3. **Merge** — :func:`merge` applies (1) and (2) to produce the new row plus
   a single ``change`` word that the gateway turns into an audit action. A
   CONFIRMED date is never silently downgraded to ESTIMATED, a moved
   confirmed date becomes REVISED with a revision-history entry, and
   CANCELED only ever arrives explicitly (§7).

Everything is frozen: :class:`Event` is a value, and a merge returns a new
one rather than mutating the row the caller is holding.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from libs.trading_core.events.taxonomy import (
    DEFAULT_EVENT_TIMEZONE,
    MACRO_EVENT_TYPES,
    require_utc,
)
from libs.trading_core.models.enums import (
    EventSession,
    EventSourceKind,
    EventStatus,
    EventType,
)

__all__ = [
    "EARNINGS_DRIFT_WINDOW",
    "Event",
    "EventCandidate",
    "MINUTES_DRIFT_WINDOW",
    "SOURCE_RANK",
    "merge",
    "previous_comparable",
    "same_event",
    "source_rank",
]

#: §7: an ESTIMATED earnings date derived from filing cadence can miss the
#: confirmed date by weeks; treat any same-ticker earnings within this window
#: as the same quarter's release rather than a second card. Quarterly cadence
#: is ~91 days, so 21 days can never swallow a neighbouring quarter.
EARNINGS_DRIFT_WINDOW = timedelta(days=21)
#: FOMC minutes are released ~3 weeks after the decision; an estimate drifts
#: by days, not weeks, and consecutive meetings are ~6 weeks apart.
MINUTES_DRIFT_WINDOW = timedelta(days=7)

#: §78 source priority. Lower = more authoritative. LLM is deliberately far
#: away from everything else: it must never win a date comparison.
SOURCE_RANK: Mapping[EventSourceKind, int] = {
    EventSourceKind.USER: 0,
    EventSourceKind.COMPANY_IR_SEC: 1,
    EventSourceKind.GOVERNMENT_AGENCY: 2,
    EventSourceKind.FEDERAL_RESERVE: 2,
    EventSourceKind.STRUCTURED_PROVIDER: 3,
    EventSourceKind.DERIVED: 4,
    EventSourceKind.NEWS: 5,
    EventSourceKind.LLM: 99,
}

#: Highest (numerically largest) rank still allowed to promote an ESTIMATED
#: event to CONFIRMED — structured providers and better, never DERIVED/NEWS/LLM.
_CONFIRM_MAX_RANK = SOURCE_RANK[EventSourceKind.STRUCTURED_PROVIDER]


def source_rank(kind: EventSourceKind) -> int:
    """Authority rank of a source kind (§78); lower wins."""
    return SOURCE_RANK[kind]


@dataclass(frozen=True)
class Event:
    """A persisted, typed catalyst (spec §6).

    ``scheduled_at`` is always tz-aware UTC (§10); ``event_timezone`` is the
    event's own zone, retained so the UI can show the local wall clock
    without re-deriving it. ``status`` carries the §7 date-knowledge state
    into every downstream payload — an ESTIMATED date must never be rendered
    or alerted on as a fact.

    ``revision_history`` is an append-only list of
    ``{scheduled_at, status, source_name, at}`` dicts recording the value
    that was *replaced*, so an audit can reconstruct what the platform
    believed at any earlier point (§84).
    """

    event_key: str
    event_type: EventType
    title: str
    scheduled_at: datetime
    status: EventStatus
    source: EventSourceKind
    source_name: str
    event_id: int | None = None
    ticker: str | None = None
    company_id: str | None = None
    event_timezone: str = DEFAULT_EVENT_TIMEZONE
    session: EventSession = EventSession.UNKNOWN
    source_url: str | None = None
    source_event_id: str | None = None
    last_verified_at: datetime | None = None
    previous_event_id: int | None = None
    comparison_reason: str | None = None
    importance: int | None = None
    series_id: str | None = None
    agency: str | None = None
    release_period: str | None = None
    fiscal_quarter: int | None = None
    fiscal_year: int | None = None
    speaker: str | None = None
    topic: str | None = None
    revision_history: tuple[Mapping[str, object], ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduled_at", require_utc(self.scheduled_at, name="scheduled_at"))
        for name in ("last_verified_at", "created_at", "updated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_utc(value, name=name))
        if self.ticker is not None:
            object.__setattr__(self, "ticker", self.ticker.strip().upper() or None)
        object.__setattr__(self, "revision_history", tuple(self.revision_history))

    @property
    def is_estimated(self) -> bool:
        """§7/§11: an estimate is never alerted on nor stated as a fact."""
        return self.status is EventStatus.ESTIMATED


@dataclass(frozen=True)
class EventCandidate:
    """What a calendar provider emits, before it has a DB identity.

    Same shape as :class:`Event` minus the persisted identity fields, plus
    ``raw`` for the provider's own untouched payload (kept for debugging and
    for later phases that mine provider-specific fields — never read by the
    merge rules).
    """

    event_key: str
    event_type: EventType
    title: str
    scheduled_at: datetime
    status: EventStatus
    source: EventSourceKind
    source_name: str
    ticker: str | None = None
    company_id: str | None = None
    event_timezone: str = DEFAULT_EVENT_TIMEZONE
    session: EventSession = EventSession.UNKNOWN
    source_url: str | None = None
    source_event_id: str | None = None
    last_verified_at: datetime | None = None
    importance: int | None = None
    series_id: str | None = None
    agency: str | None = None
    release_period: str | None = None
    fiscal_quarter: int | None = None
    fiscal_year: int | None = None
    speaker: str | None = None
    topic: str | None = None
    raw: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduled_at", require_utc(self.scheduled_at, name="scheduled_at"))
        if self.last_verified_at is not None:
            object.__setattr__(
                self, "last_verified_at", require_utc(self.last_verified_at, name="last_verified_at")
            )
        if self.ticker is not None:
            object.__setattr__(self, "ticker", self.ticker.strip().upper() or None)

    @property
    def is_estimated(self) -> bool:
        return self.status is EventStatus.ESTIMATED

    def to_event(self, *, now: datetime, event_id: int | None = None) -> Event:
        """Materialise a brand-new :class:`Event` from this candidate.

        Used for the ``created`` path only; an existing row goes through
        :func:`merge` so that authority and revision history are respected.
        """
        stamp = require_utc(now, name="now")
        return Event(
            event_id=event_id,
            event_key=self.event_key,
            event_type=self.event_type,
            title=self.title,
            ticker=self.ticker,
            company_id=self.company_id,
            scheduled_at=self.scheduled_at,
            event_timezone=self.event_timezone,
            session=self.session,
            status=self.status,
            source=self.source,
            source_name=self.source_name,
            source_url=self.source_url,
            source_event_id=self.source_event_id,
            last_verified_at=self.last_verified_at or stamp,
            importance=self.importance,
            series_id=self.series_id,
            agency=self.agency,
            release_period=self.release_period,
            fiscal_quarter=self.fiscal_quarter,
            fiscal_year=self.fiscal_year,
            speaker=self.speaker,
            topic=self.topic,
            revision_history=(),
            created_at=stamp,
            updated_at=stamp,
        )


# ---------------------------------------------------------------------------
# Identity (§5, §7)
# ---------------------------------------------------------------------------


def same_event(existing: Event, incoming: EventCandidate | Event) -> bool:
    """Is ``incoming`` the same real-world event as ``existing``?

    Equal ``event_key`` is decisive. Otherwise only two drift windows apply,
    both of which exist to absorb an ESTIMATED date being replaced by the
    confirmed one without leaving a duplicate card behind:

    - both EARNINGS, same ticker, ``|Δscheduled_at| <= 21 days``
    - both FOMC_MINUTES, ``|Δscheduled_at| <= 7 days``

    Types never cross, and a matching ticker alone is never enough.
    """
    if existing.event_key == incoming.event_key:
        return True
    if existing.event_type is not incoming.event_type:
        return False
    delta = abs(existing.scheduled_at - incoming.scheduled_at)
    if existing.event_type is EventType.EARNINGS:
        if not existing.ticker or existing.ticker != incoming.ticker:
            return False
        return delta <= EARNINGS_DRIFT_WINDOW
    if existing.event_type is EventType.FOMC_MINUTES:
        return delta <= MINUTES_DRIFT_WINDOW
    return False


# ---------------------------------------------------------------------------
# Merge (§7, §78)
# ---------------------------------------------------------------------------

#: Metadata that any source may fill in when the existing row has nothing —
#: enriching a null never rewrites a fact, so it is not authority-gated.
_METADATA_FIELDS = (
    "company_id",
    "source_url",
    "source_event_id",
    "series_id",
    "agency",
    "release_period",
    "fiscal_quarter",
    "fiscal_year",
    "speaker",
    "topic",
)


def _revision_entry(event: Event, at: datetime) -> dict[str, object]:
    """Snapshot of the value being replaced (§84 auditability)."""
    return {
        "scheduled_at": event.scheduled_at.isoformat(),
        "status": event.status.value,
        "source_name": event.source_name,
        "at": at.isoformat(),
    }


def merge(
    existing: Event,
    incoming: EventCandidate,
    now: datetime,
) -> tuple[Event, str | None]:
    """Reconcile a stored event with a freshly fetched candidate.

    Returns ``(event, change)`` where ``change`` is one of ``None``,
    ``"rescheduled"``, ``"confirmed"``, ``"revised"``, ``"reverified"``,
    ``"canceled"`` or ``"metadata"``. (``"created"`` is the caller's word for
    a row this function never saw.) The gateway maps the date-moving changes
    onto ``EVENT_UPDATED`` audit rows.

    Authority rules (§78):

    - The incoming candidate may rewrite ``scheduled_at``/``session``/
      ``status``/source fields only if ``rank(incoming) <= rank(existing)``,
      or as the one documented exception: an ESTIMATED row is promoted to
      CONFIRMED by any source of structured-provider authority or better,
      even if that source ranks below whoever produced the estimate.
    - LLM output is barred from the date path *absolutely* (§78), not merely
      relatively: an LLM candidate cannot rewrite a date even when the stored
      row was itself written by an LLM (equal rank would otherwise satisfy
      ``rank(incoming) <= rank(existing)``). An extracted date is a claim to
      be verified against a structured source, never a fact to be written.
    - CONFIRMED is never downgraded to ESTIMATED — a cadence estimate
      arriving after a confirmation is ignored for the date.
    - A CONFIRMED (or REVISED) event whose date is moved by an authorised
      source becomes REVISED and appends the replaced value to
      ``revision_history``.
    - CANCELED arrives only when the candidate explicitly says so; it is
      never inferred from an event disappearing from a provider's feed.
    - A CANCELED row is terminal for automated sources: only a USER (rank 0)
      can bring it back.
    - Re-verification by any source always refreshes ``last_verified_at``,
      which is what lets the UI say "confirmed, last checked 2h ago" without
      pretending the date changed.
    - When a date move is accepted the row adopts the candidate's
      ``event_key``. The key embeds the ET date, so keeping the old one would
      make the next tick re-create the event under the new key — exactly the
      duplicate the drift window in :func:`same_event` exists to prevent. The
      caller must therefore persist ``event_key`` alongside the other columns.
    """
    stamp = require_utc(now, name="now")
    rank_in = source_rank(incoming.source)
    rank_ex = source_rank(existing.source)

    updates: dict[str, object] = {}
    change: str | None = None

    # Metadata enrichment is always allowed INTO A NULL, whatever the rank —
    # filling a blank source_url is not overwriting a fact.
    for name in _METADATA_FIELDS:
        if getattr(existing, name) is None and getattr(incoming, name) is not None:
            updates[name] = getattr(incoming, name)
    # UNKNOWN is the null of sessions: learning that a release is BMO enriches
    # a blank, it does not overwrite a claim, so it is not authority-gated.
    if existing.session is EventSession.UNKNOWN and incoming.session is not EventSession.UNKNOWN:
        updates["session"] = incoming.session
    if updates:
        change = "metadata"

    existing_confirmed = existing.status in (EventStatus.CONFIRMED, EventStatus.REVISED)
    promotes_estimate = (
        existing.status is EventStatus.ESTIMATED
        and incoming.status is EventStatus.CONFIRMED
        and rank_in <= _CONFIRM_MAX_RANK
    )
    # LLM never writes dates (§78) — an absolute floor, not a comparison.
    # Without this, two LLM-sourced rows (rank 99 == 99) would satisfy the
    # relative test and let extracted text move a stored date.
    llm_barred = rank_in >= SOURCE_RANK[EventSourceKind.LLM]
    may_write = (rank_in <= rank_ex or promotes_estimate) and not llm_barred
    if existing.status is EventStatus.CANCELED and incoming.source is not EventSourceKind.USER:
        # Terminal until a human says otherwise (§7): a provider that merely
        # still lists a withdrawn event must not resurrect it.
        may_write = False

    if may_write:
        if incoming.status is EventStatus.CANCELED:
            if existing.status is not EventStatus.CANCELED:
                updates["status"] = EventStatus.CANCELED
                updates["revision_history"] = (
                    *existing.revision_history,
                    _revision_entry(existing, stamp),
                )
                change = "canceled"
        elif existing.status is EventStatus.CANCELED:
            # Only a USER reaches this branch (see may_write above): the
            # cancellation is being withdrawn, so the row returns to the
            # candidate's own status and the cancellation is kept in history.
            updates["status"] = incoming.status
            updates["scheduled_at"] = incoming.scheduled_at
            updates["revision_history"] = (
                *existing.revision_history,
                _revision_entry(existing, stamp),
            )
            change = "confirmed" if incoming.status is EventStatus.CONFIRMED else "rescheduled"
        else:
            date_moved = incoming.scheduled_at != existing.scheduled_at
            # A date is only rewritten by a source that is authoritative for
            # dates. An ESTIMATED candidate never moves a CONFIRMED date (§7:
            # no downgrade), even when its source outranks the confirmer.
            may_move_date = date_moved and not (
                existing_confirmed and incoming.status is EventStatus.ESTIMATED
            )
            if may_move_date:
                updates["scheduled_at"] = incoming.scheduled_at
                updates["revision_history"] = (
                    *existing.revision_history,
                    _revision_entry(existing, stamp),
                )
                if existing_confirmed:
                    updates["status"] = EventStatus.REVISED
                    change = "revised"
                elif incoming.status is EventStatus.CONFIRMED:
                    updates["status"] = EventStatus.CONFIRMED
                    change = "confirmed"
                else:
                    change = "rescheduled"
            elif (
                incoming.status is EventStatus.CONFIRMED
                and existing.status is EventStatus.ESTIMATED
            ):
                updates["status"] = EventStatus.CONFIRMED
                change = "confirmed"

            if change in ("rescheduled", "confirmed", "revised"):
                # Re-key on the new date. The natural key embeds the ET date
                # (taxonomy.event_key), so a row that kept its old key after a
                # date move would be re-created under the new key on the very
                # next tick — the duplicate the drift window exists to prevent.
                if incoming.event_key:
                    updates["event_key"] = incoming.event_key
                # The winning source becomes the row's source of record.
                updates["source"] = incoming.source
                updates["source_name"] = incoming.source_name
                if incoming.title:
                    updates["title"] = incoming.title
                if incoming.event_timezone:
                    updates["event_timezone"] = incoming.event_timezone
                if incoming.source_url is not None:
                    updates["source_url"] = incoming.source_url
                if incoming.source_event_id is not None:
                    updates["source_event_id"] = incoming.source_event_id

            # An authorised source may also CORRECT a session we already hold
            # (BMO -> AMC is a real, tradeable change), not merely fill a blank.
            if (
                incoming.session is not EventSession.UNKNOWN
                and incoming.session is not existing.session
                and "session" not in updates
            ):
                updates["session"] = incoming.session
                if change is None:
                    change = "metadata"

    # Re-verification: the same source repeating what we already hold IS
    # information (the date is still good), so the timestamp always moves —
    # that is what lets the UI say "confirmed, last checked 2h ago".
    updates["last_verified_at"] = incoming.last_verified_at or stamp
    updates["updated_at"] = stamp
    if change is None:
        change = "reverified"

    return replace(existing, **updates), change


# ---------------------------------------------------------------------------
# Previous comparable event (§15)
# ---------------------------------------------------------------------------

_PRIOR_EARNINGS_REASON = "prior quarterly earnings"
_PRIOR_MACRO_REASON = "prior release of the same series"
_PRIOR_DECISION_REASON = "prior FOMC decision"
_PRIOR_SPEECH_REASON = "prior speech by the same speaker (low confidence)"
_PRIOR_MINUTES_REASON = "prior FOMC minutes"
_PRIOR_GENERIC_REASON = "prior event of the same type"


def previous_comparable(
    event: Event,
    candidates: Iterable[Event],
) -> tuple[Event | None, str | None]:
    """Pick the previous comparable event and say *why* (§15).

    Never compares unlike events: the type must match exactly, and the
    per-type rules narrow further —

    - ``EARNINGS``  : same ticker, latest CONFIRMED/REVISED earlier event
    - macro         : same type (and ``series_id`` when both carry one), the
      most recent earlier release period
    - ``FOMC_DECISION`` / ``FOMC_MINUTES`` : the previous one
    - ``FED_SPEECH``: the same speaker's previous speech, explicitly flagged
      low confidence — two speeches by one governor are only loosely
      comparable
    - ``MARKET_HOLIDAY`` : no comparable predecessor; returns ``(None, None)``

    Ties on ``scheduled_at`` are broken by the larger ``event_id`` so the
    result is deterministic. Returns ``(None, None)`` when nothing qualifies
    — an honest absence, never a nearest-neighbour guess.
    """
    if event.event_type is EventType.MARKET_HOLIDAY:
        return None, None

    pool = [
        other
        for other in candidates
        if other.event_type is event.event_type
        and other.scheduled_at < event.scheduled_at
        and not (other.event_id is not None and other.event_id == event.event_id)
        and other.event_key != event.event_key
        and other.status is not EventStatus.CANCELED
    ]
    if not pool:
        return None, None

    if event.event_type is EventType.EARNINGS:
        if not event.ticker:
            return None, None
        pool = [
            other
            for other in pool
            if other.ticker == event.ticker
            and other.status in (EventStatus.CONFIRMED, EventStatus.REVISED)
        ]
        reason = _PRIOR_EARNINGS_REASON
    elif event.event_type in MACRO_EVENT_TYPES:
        if event.series_id:
            same_series = [o for o in pool if o.series_id == event.series_id]
            if same_series:
                pool = same_series
        pool = [o for o in pool if o.release_period != event.release_period]
        reason = _PRIOR_MACRO_REASON
    elif event.event_type is EventType.FOMC_DECISION:
        reason = _PRIOR_DECISION_REASON
    elif event.event_type is EventType.FOMC_MINUTES:
        reason = _PRIOR_MINUTES_REASON
    elif event.event_type is EventType.FED_SPEECH:
        if not event.speaker:
            return None, None
        pool = [
            o
            for o in pool
            if o.speaker and o.speaker.strip().lower() == event.speaker.strip().lower()
        ]
        reason = _PRIOR_SPEECH_REASON
    else:
        if event.ticker:
            pool = [o for o in pool if o.ticker == event.ticker]
        reason = _PRIOR_GENERIC_REASON

    if not pool:
        return None, None
    best = max(pool, key=lambda o: (o.scheduled_at, o.event_id or 0))
    return best, reason
