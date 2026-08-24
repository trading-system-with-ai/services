"""libs/event_calendar/fed_docs — Fed statements, minutes, speeches (§42-§45).

Every test runs the REAL adapter over ``httpx.MockTransport`` against fixtures
downloaded live from federalreserve.gov with the contact User-Agent; the
network is never touched.

Pinned, in order of importance:

1. THE SOURCE DOCUMENT IS AUTHORITATIVE (§44). Parsing is checked against
   markup the Fed actually served: the July 2026 statement's ``9 – 3`` vote
   with its THREE dissenters (an en dash, and names carrying middle initials
   — "Beth M. Hammack" truncated to "Beth M" would be stored and then diffed
   as the committee's dissent), the ``3-1/2 to 3-3/4 percent`` target range as
   3.50/3.75, and the June statement's unanimous ``12 – 0``.
2. NO FABRICATION. Boilerplate is stripped, statement text is not; an absent
   target range is ``None``, never last meeting's; a 404 is
   :class:`DocumentNotFound`, never an empty document.
3. POINT-IN-TIME (§14/§96). A statement released after ``as_of`` is withheld,
   and the release instant is the RSS ``pubDate`` when the feed knows it —
   never fetch time.
4. THE CONTACT USER-AGENT. federalreserve.gov 403s anonymous requests; the
   header is asserted on the wire, and construction without one is refused.
5. NO HAWK/DOVE SCORE (§43). This module reports what the document says and
   never scores it — asserted structurally over the parsed shapes.
"""
import pathlib
from datetime import date, datetime, timezone

import httpx
import pytest

from libs.event_calendar.fed_docs import (
    DOC_TYPE_MINUTES,
    DOC_TYPE_SPEECH,
    DOC_TYPE_STATEMENT,
    RSS_KIND_MINUTES,
    RSS_KIND_OTHER,
    RSS_KIND_STATEMENT,
    DocumentNotFound,
    FedDocsError,
    FedDocumentsProvider,
    classify_press_monetary,
    minutes_meeting_dates,
    article_fragment,
    minutes_sections,
    parse_article,
    parse_fed_fraction,
    parse_press_monetary_rss,
    parse_target_range,
    parse_vote,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "events"

UTC = timezone.utc

STATEMENT_JUL = "fomc_statement_2026-07-29.html"
STATEMENT_JUN = "fomc_statement_2026-06-17.html"
MINUTES_JUN = "fomc_minutes_2026-06-17.html"
RSS_PRESS = "fed_press_monetary.xml"

DECISION_JUL = date(2026, 7, 29)
DECISION_JUN = date(2026, 6, 17)

TEST_UA = "trading-system-with-ai/0.1 (tests@example.com)"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def _paragraphs(name: str) -> list[str]:
    return parse_article(_fixture(name))["paragraphs"]


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _handler(routes: dict[str, object] | None = None, *, seen: list | None = None):
    """Serve fixtures by URL path; anything unrouted is a 404.

    404-by-default is deliberate: a test that fetches a document it did not
    stub must FAIL as "not published", which is the same answer the live site
    gives, rather than silently reusing another fixture.
    """
    table = {
        "/newsevents/pressreleases/monetary20260729a.htm": _fixture(STATEMENT_JUL),
        "/newsevents/pressreleases/monetary20260617a.htm": _fixture(STATEMENT_JUN),
        "/monetarypolicy/fomcminutes20260617.htm": _fixture(MINUTES_JUN),
        "/feeds/press_monetary.xml": _fixture(RSS_PRESS),
    }
    table.update(routes or {})

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = table.get(request.url.path)
        if body is None:
            return httpx.Response(404, text="Page Not Found")
        if isinstance(body, httpx.Response):
            return body
        if isinstance(body, int):
            return httpx.Response(body, text="upstream said no")
        return httpx.Response(200, text=body)

    return handle


def _provider(routes=None, *, seen=None, user_agent: str = TEST_UA):
    return FedDocumentsProvider(
        user_agent=user_agent,
        transport=httpx.MockTransport(_handler(routes, seen=seen)),
    )


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def test_statement_url_is_the_decision_day_a_release():
    """The statement is the day's FIRST release — the ``a`` suffix."""
    provider = _provider()
    assert provider.statement_url(DECISION_JUL) == (
        "https://www.federalreserve.gov/newsevents/pressreleases/"
        "monetary20260729a.htm"
    )


def test_minutes_url_is_keyed_by_the_meeting_end_date_not_the_release():
    """June 16-17 minutes live at ...20260617.htm though released July 8.

    Keying by the release date would 404 on every meeting.
    """
    assert _provider().minutes_url(DECISION_JUN) == (
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260617.htm"
    )


def test_provider_refuses_to_construct_without_a_contact_user_agent():
    """An anonymous client would be 403ed; refuse at construction instead."""
    with pytest.raises(ValueError, match="contact User-Agent"):
        FedDocumentsProvider(user_agent="   ")


# ---------------------------------------------------------------------------
# Statement parsing — the live July 2026 markup
# ---------------------------------------------------------------------------


def test_statement_body_is_the_verbatim_paragraphs_without_navigation():
    """Text is the document's own paragraphs; the share menu is not text."""
    article = parse_article(_fixture(STATEMENT_JUL))
    paragraphs = article["paragraphs"]

    assert article["title"] == "Federal Reserve issues FOMC statement"
    assert article["article_time"] == "July 29, 2026"
    assert article["release_time"].startswith("For release at 2:00 p.m.")

    assert paragraphs[0] == (
        "The Federal Open Market Committee approved the following statement "
        "for release by a 9 – 3 vote:"
    )
    assert paragraphs[1].startswith(
        "The Committee decided to maintain the target range for the federal "
        "funds rate at 3-1/2 to 3-3/4 percent"
    )
    joined = "\n".join(paragraphs)
    # Navigation and footer boilerplate never enter the diffable text.
    assert "Share" not in joined
    assert "For media inquiries" not in joined
    assert "Implementation Note issued" not in joined
    assert "Last Update" not in joined
    # The nbsp spacer paragraph is dropped, not stored as an empty sentence.
    assert all(p.strip() for p in paragraphs)


def test_statement_paragraph_count_is_the_documents_own_body():
    """Five substantive paragraphs on the live July page; June has four.

    Pinned exactly: a layout change that swallowed the dissent paragraph would
    otherwise pass every other assertion here.
    """
    assert len(_paragraphs(STATEMENT_JUL)) == 5
    assert len(_paragraphs(STATEMENT_JUN)) == 4


def test_vote_parses_the_en_dash_count_and_all_three_dissenters():
    """"by a 9 – 3 vote" + three names carrying middle initials.

    The middle initial is the trap: a period-terminated name regex yields
    ["Beth M"], which would be persisted and rendered as the dissent.
    """
    vote = parse_vote(_paragraphs(STATEMENT_JUL))

    assert vote["for"] == 9
    assert vote["against"] == 3
    assert vote["dissenters"] == [
        "Beth M. Hammack",
        "Neel Kashkari",
        "Lorie K. Logan",
    ]
    assert vote["unanimous"] is False
    # The source sentences travel with the parse (§44).
    assert "9 – 3 vote" in vote["text"]
    assert "Voting against the monetary policy action" in vote["text"]


def test_vote_on_a_unanimous_statement_has_no_dissenters():
    vote = parse_vote(_paragraphs(STATEMENT_JUN))

    assert (vote["for"], vote["against"]) == (12, 0)
    assert vote["dissenters"] == []
    assert vote["unanimous"] is True


def test_vote_is_empty_rather_than_guessed_when_the_statement_is_silent():
    """No vote sentence -> None/[], never a committee-size assumption."""
    vote = parse_vote(["The Committee decided to maintain the target range."])

    assert vote["for"] is None
    assert vote["against"] is None
    assert vote["dissenters"] == []
    assert vote["text"] == ""


def test_target_range_reads_the_feds_mixed_fractions_as_percent_floats():
    """"3-1/2 to 3-3/4 percent" -> 3.50/3.75, with the sentence attached."""
    target = parse_target_range(_paragraphs(STATEMENT_JUL))

    assert target == {
        "low_pct": 3.5,
        "high_pct": 3.75,
        "text": target["text"],
    }
    assert "3-1/2 to 3-3/4 percent" in target["text"]


def test_target_range_is_none_when_the_statement_does_not_state_one():
    """Absent is ABSENT: inheriting last meeting's range would report a hold
    the document never claimed."""
    assert parse_target_range(["Economic activity is expanding at a solid pace."]) is None


@pytest.mark.parametrize(
    "token,expected",
    [
        ("3-1/2", 3.5),
        ("3-3/4", 3.75),
        ("4", 4.0),
        ("5-1/4", 5.25),
        ("0", 0.0),
        ("3.75", 3.75),
        ("", None),
        ("not-a-rate", None),
    ],
)
def test_fed_fractions_parse_or_return_none(token, expected):
    assert parse_fed_fraction(token) == expected


def test_site_chrome_around_the_article_is_not_treated_as_statement_text():
    """The FULL page wraps the article in a shell whose banner, .gov
    explainers and footer address are ``<p>`` elements in ``col-sm-8``
    columns. Four chrome paragraphs ahead of the statement would be diffed as
    paragraphs the Committee ADDED — the fixture is trimmed to the article, so
    this reconstructs the shell to pin that the trim is not load-bearing.
    """
    full_page = (
        "<html><body>"
        '<div class="col-xs-12 col-sm-8">'
        "<p>An official website of the United States Government</p>"
        "<p>Official websites use .gov</p>"
        "</div>"
        + _fixture(STATEMENT_JUL)
        + '<div class="col-xs-12 col-sm-8">'
        "<p>Board of Governors of the Federal Reserve System</p>"
        "<p>20th Street and Constitution Avenue N.W., Washington, DC 20551</p>"
        "</div></body></html>"
    )

    article = parse_article(full_page)

    assert article["title"] == "Federal Reserve issues FOMC statement"
    assert len(article["paragraphs"]) == 5
    assert article["paragraphs"][0].startswith("The Federal Open Market Committee")
    joined = " ".join(article["paragraphs"])
    assert "official website" not in joined.lower()
    assert "Constitution Avenue" not in joined


def test_article_parser_falls_back_to_every_paragraph_on_a_layout_change():
    """A renamed container degrades to noisier text, never to an empty doc.

    An empty document would diff as "the entire statement was removed", which
    is the single worst wrong answer this module could produce.
    """
    mangled = _fixture(STATEMENT_JUL).replace("col-sm-8", "col-sm-NEW").replace(
        'id="article"', 'id="renamed"'
    )
    paragraphs = parse_article(mangled)["paragraphs"]

    assert len(paragraphs) >= 4
    assert any("9 – 3 vote" in p for p in paragraphs)


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------


def test_minutes_sections_come_from_the_document_in_order():
    """The eight bold section headings the June minutes actually carry.

    Derived from the page, never a hardcoded expected list in the parser: a
    section the Fed adds shows up instead of being silently dropped.
    """
    assert minutes_sections(_fixture(MINUTES_JUN)) == [
        "Developments in Financial Markets and Open Market Operations",
        "Staff Review of the Economic Situation",
        "Staff Review of the Financial Situation",
        "Staff Economic Outlook",
        "Participants' Views on Current Conditions and the Economic Outlook",
        "Committee Policy Actions",
        "Notation Vote",
        "Attendance",
    ]


def test_minutes_sections_ignore_the_sites_navigation_menus():
    """The Fed's sidebar links are ``<p><strong>`` too. Unscoped, the FULL
    page yields forty-eight "sections" — nav entries like "Reporting Forms"
    and "Legal Developments" — so the scan is narrowed to the article
    container. The fixture is trimmed, so the shell is reconstructed here to
    pin that the trim is not what makes this work.
    """
    shell = (
        "<html><body>"
        '<div class="nav"><p><strong>Reporting Forms</strong></p>'
        "<p><strong>Legal Developments</strong></p></div>"
        + _fixture(MINUTES_JUN)
        + '<div class="footer"><p><strong>Accessibility</strong></p></div>'
        "</body></html>"
    )

    assert minutes_sections(shell) == minutes_sections(_fixture(MINUTES_JUN))
    assert "Reporting Forms" not in minutes_sections(shell)


def test_article_fragment_counts_nested_divs():
    """An inner ``</div>`` must not truncate the slice — the statement's
    columns are divs inside the article."""
    fragment = article_fragment(
        '<html><div id="article"><div class="a"><p>kept</p></div>'
        "<p>also kept</p></div><p>dropped</p></html>"
    )

    assert "kept" in fragment and "also kept" in fragment
    assert "dropped" not in fragment


def test_article_fragment_returns_the_whole_document_when_there_is_no_article():
    assert article_fragment("<p>bare</p>") == "<p>bare</p>"


def test_minutes_signature_block_is_not_mistaken_for_a_section():
    """The centred ``<strong>Joshua Gallin</strong>`` is a signature."""
    assert "Joshua Gallin" not in minutes_sections(_fixture(MINUTES_JUN))


def test_minutes_body_is_long_and_starts_at_the_meeting_date_line():
    paragraphs = _paragraphs(MINUTES_JUN)

    assert len(paragraphs) > 80
    assert paragraphs[0].startswith("June 16–17, 2026 A joint meeting")
    assert any(p.startswith("Committee Policy Actions") for p in paragraphs)


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------


def test_rss_classifies_statements_fomc_minutes_and_everything_else():
    items = parse_press_monetary_rss(_fixture(RSS_PRESS))

    assert len(items) == 15
    # Newest first.
    assert items[0].published_at >= items[-1].published_at

    kinds = {item.title: item.kind for item in items}
    assert kinds["Federal Reserve issues FOMC statement"] == RSS_KIND_STATEMENT
    assert (
        kinds["Minutes of the Federal Open Market Committee, July 28–29, 2026"]
        == RSS_KIND_MINUTES
    )


def test_discount_rate_minutes_are_not_fomc_minutes():
    """A different committee's minutes ride the same feed (§44 provenance).

    Classifying them MINUTES would attach the Board's discount-rate discussion
    to an FOMC meeting as if the Committee had written it.
    """
    assert (
        classify_press_monetary(
            "Minutes of the Board's discount rate meetings on June 8 and June 17, 2026"
        )
        == RSS_KIND_OTHER
    )
    assert (
        classify_press_monetary("Minutes of the Federal Open Market Committee, June 16-17, 2026")
        == RSS_KIND_MINUTES
    )
    assert classify_press_monetary("Federal Reserve issues FOMC statement") == (
        RSS_KIND_STATEMENT
    )


def test_rss_pubdates_are_utc_instants():
    """"Wed, 29 Jul 2026 18:00:00 GMT" — the 14:00 EDT statement release."""
    items = parse_press_monetary_rss(_fixture(RSS_PRESS))
    statement = next(
        i for i in items if i.url.endswith("monetary20260729a.htm")
    )

    assert statement.published_at == datetime(2026, 7, 29, 18, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "title,expected",
    [
        (
            "Minutes of the Federal Open Market Committee, July 28–29, 2026",
            (date(2026, 7, 28), date(2026, 7, 29)),
        ),
        (
            "Minutes of the Federal Open Market Committee, June 16-17, 2026",
            (date(2026, 6, 16), date(2026, 6, 17)),
        ),
        ("Federal Reserve issues FOMC statement", None),
    ],
)
def test_minutes_titles_name_the_meeting_they_belong_to(title, expected):
    assert minutes_meeting_dates(title) == expected


def test_rss_items_without_a_pubdate_are_dropped():
    """No timestamp -> the item cannot answer "was this visible at as_of"."""
    xml = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><title>Federal Reserve issues FOMC statement</title>
      <link>https://example.invalid/a.htm</link></item>
    </channel></rss>"""

    assert parse_press_monetary_rss(xml) == []


def test_unparseable_rss_yields_no_items_rather_than_raising():
    assert parse_press_monetary_rss("<not xml") == []


# ---------------------------------------------------------------------------
# Provider — fetching
# ---------------------------------------------------------------------------


def test_fetch_statement_returns_the_parsed_document():
    provider = _provider()
    rss = provider.list_press_monetary()

    statement = provider.fetch_statement(DECISION_JUL, rss_items=rss)

    assert statement is not None
    assert statement.doc_type == DOC_TYPE_STATEMENT
    assert statement.meeting_date == DECISION_JUL
    assert statement.url.endswith("monetary20260729a.htm")
    assert statement.title == "Federal Reserve issues FOMC statement"
    assert statement.vote["for"] == 9
    assert statement.target_range["high_pct"] == 3.75
    assert statement.raw_html_len > 1000
    # text is exactly the paragraphs joined — the two can never drift.
    assert statement.text == "\n\n".join(statement.paragraphs)


def test_statement_released_at_prefers_the_rss_publication_instant():
    """The Fed's own timestamp beats our 14:00 ET convention."""
    provider = _provider()
    statement = provider.fetch_statement(
        DECISION_JUL, rss_items=provider.list_press_monetary()
    )

    assert statement.released_at == datetime(2026, 7, 29, 18, 0, tzinfo=UTC)


def test_statement_released_at_falls_back_to_the_14_00_et_convention():
    """No RSS item -> 14:00 ET, DST-correct (July = 18:00Z), matching the
    FOMC_DECISION event ``libs.event_calendar.fed`` already emits."""
    statement = _provider().fetch_statement(DECISION_JUL, rss_items=[])

    assert statement.released_at == datetime(2026, 7, 29, 18, 0, tzinfo=UTC)


def test_statement_is_withheld_from_an_as_of_before_its_release():
    """§14: a replay standing before 14:00 ET cannot read the statement."""
    provider = _provider()
    rss = provider.list_press_monetary()

    before = provider.fetch_statement(
        DECISION_JUL,
        as_of=datetime(2026, 7, 29, 17, 59, tzinfo=UTC),
        rss_items=rss,
    )
    after = provider.fetch_statement(
        DECISION_JUL,
        as_of=datetime(2026, 7, 29, 18, 0, tzinfo=UTC),
        rss_items=rss,
    )

    assert before is None
    assert after is not None


def test_as_of_gating_happens_before_the_fetch():
    """A withheld document costs no request: the release instant is known
    from the feed, so there is nothing to learn by fetching the page."""
    seen: list[httpx.Request] = []
    provider = _provider(seen=seen)
    rss = provider.list_press_monetary()
    seen.clear()

    assert (
        provider.fetch_statement(
            DECISION_JUL,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            rss_items=rss,
        )
        is None
    )
    assert seen == []


def test_list_press_monetary_drops_items_published_after_as_of():
    provider = _provider()

    items = provider.list_press_monetary(
        as_of=datetime(2026, 7, 1, tzinfo=UTC)
    )

    assert items
    assert all(item.published_at <= datetime(2026, 7, 1, tzinfo=UTC) for item in items)
    assert not any(i.url.endswith("monetary20260729a.htm") for i in items)


def test_fetch_minutes_takes_its_release_instant_from_the_rss():
    """June 16-17 minutes were RELEASED July 8 — three weeks later.

    Dating them by the meeting would make an as-of replay show minutes that
    did not exist for another twenty-one days.
    """
    provider = _provider()
    minutes = provider.fetch_minutes(
        DECISION_JUN, rss_items=provider.list_press_monetary()
    )

    assert minutes.doc_type == DOC_TYPE_MINUTES
    assert minutes.meeting_date == DECISION_JUN
    assert minutes.released_at == datetime(2026, 7, 8, 18, 0, tzinfo=UTC)
    assert len(minutes.paragraphs) > 80


def test_minutes_released_at_is_none_when_the_feed_does_not_reach_back():
    """An honest unknown, never the meeting date used as a release date."""
    minutes = _provider().fetch_minutes(DECISION_JUN, rss_items=[])

    assert minutes.released_at is None


def test_minutes_are_withheld_from_an_as_of_before_their_release():
    provider = _provider()
    rss = provider.list_press_monetary()

    assert (
        provider.fetch_minutes(
            DECISION_JUN, as_of=datetime(2026, 7, 1, tzinfo=UTC), rss_items=rss
        )
        is None
    )
    assert (
        provider.fetch_minutes(
            DECISION_JUN, as_of=datetime(2026, 8, 1, tzinfo=UTC), rss_items=rss
        )
        is not None
    )


def test_fetch_speech_reads_the_speaker_from_the_url_slug():
    """FED_SPEECH events already carry the URL; the slug names the speaker,
    the same derivation ``libs.event_calendar.fed`` makes from the RSS."""
    path = "/newsevents/speech/cook20260805a.htm"
    page = (
        '<div id="article"><div class="heading col-xs-12 col-sm-8 col-md-8">'
        '<p class="article__time">August 5, 2026</p>'
        '<h3 class="title">Outlook for the U.S. Economy</h3></div>'
        '<div class="col-xs-12 col-sm-8 col-md-8">'
        "<p>Thank you for the invitation to speak today.</p>"
        "<p>The labor market has cooled but remains solid.</p>"
        "</div></div>"
    )
    provider = _provider({path: page})

    speech = provider.fetch_speech("https://www.federalreserve.gov" + path)

    assert speech.doc_type == DOC_TYPE_SPEECH
    assert speech.speaker == "Cook"
    assert speech.title == "Outlook for the U.S. Economy"
    assert speech.paragraphs[0].startswith("Thank you for the invitation")
    # Noon ET anchor on a page that carries a date but no clock time.
    assert speech.released_at == datetime(2026, 8, 5, 16, 0, tzinfo=UTC)


def test_fetch_speech_prefers_a_byline_on_the_page():
    path = "/newsevents/speech/powell20260805a.htm"
    page = (
        '<div id="article"><div class="col-xs-12 col-sm-8 col-md-8">'
        "<p>Chair Jerome H. Powell</p>"
        "<p>At the Economic Club of New York</p>"
        "<p>Inflation has continued to ease over the past year.</p>"
        "</div></div>"
    )

    speech = _provider({path: page}).fetch_speech(
        "https://www.federalreserve.gov" + path
    )

    assert speech.speaker == "Chair Jerome H. Powell"


# ---------------------------------------------------------------------------
# Transport behaviour
# ---------------------------------------------------------------------------


def test_every_request_carries_the_contact_user_agent():
    """federalreserve.gov 403s anonymous scrapers; the header is the
    difference between a working adapter and a dead one."""
    seen: list[httpx.Request] = []
    provider = _provider(seen=seen)

    provider.list_press_monetary()
    provider.fetch_statement(DECISION_JUL, rss_items=[])
    provider.fetch_minutes(DECISION_JUN, rss_items=[])

    assert len(seen) == 3
    assert {r.headers["User-Agent"] for r in seen} == {TEST_UA}


def test_a_missing_document_is_document_not_found():
    """404 is the EXPECTED answer for a statement that has not dropped."""
    provider = _provider()

    with pytest.raises(DocumentNotFound, match="404"):
        provider.fetch_statement(date(2026, 12, 9), rss_items=[])


def test_document_not_found_is_a_fed_docs_error():
    """One except clause covers both — callers need not enumerate."""
    assert issubclass(DocumentNotFound, FedDocsError)


def test_a_server_fault_raises_rather_than_returning_an_empty_document():
    provider = _provider(
        {"/newsevents/pressreleases/monetary20260729a.htm": 503}
    )

    with pytest.raises(FedDocsError, match="503"):
        provider.fetch_statement(DECISION_JUL, rss_items=[])


def test_a_403_is_the_proven_absence_verdict_not_a_fault():
    from libs.market_data.provider import CapabilityNotAvailable

    provider = _provider({"/feeds/press_monetary.xml": 403})

    with pytest.raises(CapabilityNotAvailable, match="403"):
        provider.list_press_monetary()


def test_a_transport_fault_becomes_fed_docs_error():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns", request=request)

    provider = FedDocumentsProvider(
        user_agent=TEST_UA, transport=httpx.MockTransport(explode)
    )

    with pytest.raises(FedDocsError, match="ConnectError"):
        provider.list_press_monetary()


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_reports_true_when_the_feed_parses():
    from libs.event_calendar.provider import CAPABILITY_KEYS

    report = _provider().capabilities()

    assert set(report) == set(CAPABILITY_KEYS)
    assert report["fed_events"] is True


def test_capabilities_reports_false_on_403_and_a_string_on_a_fault():
    assert _provider({"/feeds/press_monetary.xml": 403}).capabilities()[
        "fed_events"
    ] is False

    verdict = _provider({"/feeds/press_monetary.xml": 500}).capabilities()[
        "fed_events"
    ]
    assert isinstance(verdict, str) and "500" in verdict


def test_capabilities_reports_a_string_when_200_parses_to_nothing():
    """A layout change is an availability problem even though HTTP said OK."""
    verdict = _provider(
        {"/feeds/press_monetary.xml": "<rss version='2.0'><channel/></rss>"}
    ).capabilities()["fed_events"]

    assert isinstance(verdict, str) and "format changed" in verdict


def test_capabilities_never_raises():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns", request=request)

    provider = FedDocumentsProvider(
        user_agent=TEST_UA, transport=httpx.MockTransport(explode)
    )

    assert isinstance(provider.capabilities()["fed_events"], str)


# ---------------------------------------------------------------------------
# §43 — no aggregate score anywhere
# ---------------------------------------------------------------------------


def test_no_hawkish_dovish_score_is_produced_by_this_module():
    """§43 forbids collapsing a statement to one number. The documents layer
    reports what the Fed wrote; scoring is not a key it can grow later."""
    provider = _provider()
    statement = provider.fetch_statement(
        DECISION_JUL, rss_items=provider.list_press_monetary()
    )
    payload = statement.to_dict()

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    banned = {"score", "hawkish", "dovish", "hawk_dove", "tone_score"}
    assert not banned & {k.lower() for k in keys(payload)}


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def test_the_package_factory_builds_a_documents_provider_with_the_contact_ua():
    from libs.event_calendar import fed_documents_provider

    class _Settings:
        sec_user_agent = "trading-system-with-ai/0.1 (ops@example.com)"

    provider = fed_documents_provider(_Settings())

    assert provider.name == "fed_docs"
    assert provider.user_agent == "trading-system-with-ai/0.1 (ops@example.com)"


def test_the_documents_provider_is_not_in_the_calendar_registry():
    """It has no ``fetch_events``; serving it from ``get_provider`` would
    satisfy the name and then fail at the call site."""
    from libs.event_calendar import get_provider

    with pytest.raises(ValueError, match="unknown event calendar provider"):
        get_provider("fed_docs")
