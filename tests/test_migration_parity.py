"""Migration ↔ code parity pins (risk-engine audit §8 items 1–2).

The raw SQL migrations own the production Postgres schema; the SQLAlchemy ORM
is a hand-maintained mirror and the sqlite test harness (``create_all``)
carries no CHECK constraints — so a vocabulary drift between code and a DB
CHECK is invisible to every other test. These string-level pins are the
cheapest possible tripwire:

- the ``orders.side`` CHECK in migration 017 is EXACTLY
  :data:`libs.broker.provider.MLEG_LEG_SIDES` (every side the gateway can
  write: long opens/closes plus the collateralized / margin-backed shorts);
- every ``migrations/NNN_*.sql`` file is mounted into the compose db service
  (a fresh volume would otherwise silently miss ALTERs that ``create_all``
  cannot replay).
"""
from __future__ import annotations

import re
from pathlib import Path

from libs.broker.provider import (
    BUY_TO_CLOSE,
    BUY_TO_OPEN,
    MLEG_LEG_SIDES,
    SELL_TO_CLOSE,
    SELL_TO_OPEN,
)

SERVICES = Path(__file__).resolve().parents[1]
MIGRATIONS = SERVICES / "migrations"


def _check_list(sql: str, column: str) -> set[str]:
    """Extract the quoted values of ``CHECK (<column> IN (...))`` from SQL."""
    m = re.search(rf"CHECK\s*\(\s*{column}\s+IN\s*\(([^)]*)\)\s*\)", sql)
    assert m is not None, f"no CHECK ({column} IN (...)) found"
    return set(re.findall(r"'([A-Z_]+)'", m.group(1)))


def test_migration_017_side_vocabulary_equals_mleg_leg_sides():
    sql = (MIGRATIONS / "017_orders_side_vocabulary.sql").read_text()
    assert "DROP CONSTRAINT IF EXISTS orders_side_check" in sql
    assert _check_list(sql, "side") == set(MLEG_LEG_SIDES)
    # The four sides the gateway actually writes (orders.py / income.py).
    assert set(MLEG_LEG_SIDES) == {
        BUY_TO_OPEN, SELL_TO_OPEN, SELL_TO_CLOSE, BUY_TO_CLOSE
    }


def test_migration_005_side_vocabulary_is_a_strict_subset_of_017():
    """005's original two-side CHECK is what 017 relaxes — pinned so the
    history stays legible (never edit 005 in place)."""
    old = _check_list((MIGRATIONS / "005_orders.sql").read_text(), "side")
    assert old == {BUY_TO_OPEN, SELL_TO_CLOSE}
    assert old < set(MLEG_LEG_SIDES)


def test_every_migration_is_mounted_in_docker_compose():
    compose = (SERVICES / "docker-compose.yml").read_text()
    files = sorted(p.name for p in MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql"))
    assert files, "no migrations found"
    missing = [
        f
        for f in files
        if f"./migrations/{f}:/docker-entrypoint-initdb.d/{f}:ro" not in compose
    ]
    assert missing == [], f"migrations not mounted in docker-compose.yml: {missing}"


def test_migration_numbers_are_contiguous():
    numbers = sorted(
        int(p.name[:3]) for p in MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")
    )
    assert numbers == list(range(1, len(numbers) + 1)), numbers


# ---------------------------------------------------------------------------
# ORM <-> migration column mirror (mechanical) for tables whose migration is
# a single CREATE TABLE (no later ALTERs): the column NAMES and ORDER in the
# SQL must equal the ORM's. Audit §8 item 15 called this mirror unverified.
# ---------------------------------------------------------------------------

_SINGLE_CREATE_TABLES = {
    # stock_bars_1m is the ONE registered table whose migration declares a
    # TABLE-LEVEL composite primary key (``PRIMARY KEY (ticker, ts)``) rather
    # than marking a single column. It is registered here because Phase C
    # finally gave it an ORM mirror (StockBar1mRow) after twenty migrations of
    # sitting unmapped — precisely the drift this pin exists to catch.
    "002_system_state_and_bars.sql": ["stock_bars_1m"],
    "007_stock_bars_daily.sql": ["stock_bars_daily"],
    "018_risk_snapshots.sql": [
        "risk_snapshots",
        "risk_metrics",
        "risk_contributions",
        "atm_iv_daily",
    ],
    "019_stress_runs.sql": ["stress_runs"],
    "020_risk_model_backtests.sql": ["risk_model_backtests"],
    "021_events.sql": [
        "events",
        "market_calendar",
        "event_ingest_state",
    ],
    "022_fundamental_statements.sql": ["fundamental_statements"],
    "024_event_analyses.sql": ["event_analyses"],
    # Phase I registers BOTH of its tables from one file, in file order —
    # `option_daily_bars` declares a TABLE-LEVEL composite primary key
    # (``PRIMARY KEY (option_ticker, bar_date)``) exactly as stock_bars_1m
    # does, so it exercises the same parse branch on a table whose ORM mirror
    # shipped in the same commit as its migration.
    "025_event_options.sql": ["option_daily_bars", "event_option_metrics"],
    # Phase G. `macro_observations` declares a TABLE-LEVEL composite primary
    # key (``PRIMARY KEY (series_id, period)``); `treasury_yields` declares a
    # table-level SINGLE-column one (``PRIMARY KEY (curve_date)``), which is
    # the one shape no earlier registered table exercises.
    "026_macro_data.sql": ["macro_observations", "treasury_yields"],
    # Phase H. `fed_documents` is the first registered table declaring BOTH a
    # table-level PRIMARY KEY and a table-level UNIQUE — two constraint lines
    # the column parser must skip, on a table whose `url` column carries the
    # uniqueness the ORM marks inline. It is also the first BIGSERIAL surrogate
    # key in the registered set.
    "027_fed_documents.sql": ["fed_documents"],
    # Catalyst research upgrade: web-search runs + evidence, both from one
    # file in file order (the 025 precedent) — ORM mirrors shipped in the
    # same commit as the migration.
    "030_web_research.sql": ["event_search_runs", "search_evidence"],
    # Prediction markets: four tables from one file, in file order.
    # `prediction_market_history` declares a TABLE-LEVEL composite primary
    # key (``PRIMARY KEY (market_id, outcome, ts)``), the stock_bars_1m
    # parse branch.
    "031_prediction_markets.sql": [
        "prediction_markets",
        "prediction_market_snapshots",
        "prediction_market_history",
        "event_prediction_markets",
    ],
}


#: Tables whose column list is a CREATE TABLE plus LATER ``ALTER TABLE ... ADD
#: COLUMN`` migrations. ``_SINGLE_CREATE_TABLES`` above cannot describe them —
#: it reads one CREATE and stops — so the mirror for these is the CREATE's
#: columns FOLLOWED BY every added column, in migration order, which is
#: exactly the order the ORM must append them in.
#:
#: ``news_articles`` is the first entry because Phase D (migration 023) adds
#: the five evidence columns to migration 012's mirror. Registering it here
#: rather than leaving it unpinned is the point: an ALTER-extended table is
#: MORE prone to ORM drift than a fresh CREATE, because the person adding the
#: column edits two files that are twenty migrations apart.
_CREATE_PLUS_ALTER_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "news_articles": ("012_news_articles.sql", ("023_news_evidence.sql",)),
    # Auto-strategy explainability (2026-08-20): 029 appends journal+advice.
    "portfolio_backtests": (
        "028_portfolio_backtests.sql",
        ("029_portfolio_journal_advice.sql",),
    ),
}


#: ``ALTER TABLE <table> ADD COLUMN [IF NOT EXISTS] <name> ...`` — the added
#: column's name. Case-insensitive because the migrations are not required to
#: shout, and ``IF NOT EXISTS`` is optional because the older migrations
#: (005, 006) predate the convention.
_ADD_COLUMN = re.compile(
    r"^\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
    re.I | re.M,
)


def _added_columns(sql: str, table: str) -> list[str]:
    """Column names an ALTER migration appends to ``table``, in file order."""
    return [
        name
        for target, name in _ADD_COLUMN.findall(sql)
        if target.lower() == table.lower()
    ]


#: SQL that declares a CONSTRAINT on the TABLE rather than a column. Anchored
#: at the start of the line so a column whose NAME merely begins with one of
#: these words is still read as a column.
_TABLE_CONSTRAINT = re.compile(
    r"^(CONSTRAINT|PRIMARY\s+KEY|UNIQUE\s*\(|FOREIGN\s+KEY|CHECK\s*\()",
    re.I,
)


def _sql_columns(sql: str, table: str) -> list[str]:
    m = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", sql, re.S
    )
    assert m is not None, f"CREATE TABLE {table} not found"
    cols: list[str] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        # TABLE-LEVEL constraints are not columns. Without this filter a
        # composite ``PRIMARY KEY (ticker, ts)`` line (stock_bars_1m) would be
        # read as a column literally named "PRIMARY" and the mirror check
        # would fail for a table whose ORM is in fact correct.
        if _TABLE_CONSTRAINT.match(line):
            continue
        cols.append(line.split()[0])
    return cols


def test_orm_columns_mirror_single_create_migrations():
    from apps.gateway.db import Base

    for filename, tables in _SINGLE_CREATE_TABLES.items():
        sql = (MIGRATIONS / filename).read_text()
        for table in tables:
            orm_cols = [c.name for c in Base.metadata.tables[table].columns]
            assert _sql_columns(sql, table) == orm_cols, (filename, table)


def test_orm_columns_mirror_create_plus_alter_migrations():
    """The ALTER-extended tables' ORM order == CREATE columns + ALTERs.

    Same mechanical mirror as above, one migration list longer. This is the
    check that catches the Phase D failure mode: someone adds
    ``materiality`` to migration 023 and forgets ``NewsArticleRow``, or adds
    it to the ORM in the wrong position — either way the live Postgres column
    order and the ORM's stop agreeing, and nothing else in the suite notices
    because the sqlite harness builds its schema from the ORM alone.
    """
    from apps.gateway.db import Base

    for table, (create_file, alter_files) in _CREATE_PLUS_ALTER_TABLES.items():
        expected = _sql_columns((MIGRATIONS / create_file).read_text(), table)
        for filename in alter_files:
            expected += _added_columns((MIGRATIONS / filename).read_text(), table)
        orm_cols = [c.name for c in Base.metadata.tables[table].columns]
        assert expected == orm_cols, (table, create_file, alter_files)


def test_migration_023_is_additive_only():
    """023 may only ADD to ``news_articles`` — never drop, rename or retype.

    ``news_articles`` is a LIVE table holding real fetched articles, and it is
    the grounding store: an LLM recommendation citing an article that is not
    in it is rejected. A migration that dropped or retyped a column here would
    destroy real evidence on the next fresh-volume apply, and (audit §13)
    there is no runner to review it first. So the pin is on the SQL text: the
    only DDL verbs allowed in this file are ADD COLUMN and CREATE INDEX.
    """
    sql = (MIGRATIONS / "023_news_evidence.sql").read_text()
    statements = [
        line.strip()
        for line in sql.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    forbidden = [
        s
        for s in statements
        if re.search(r"\b(DROP|RENAME|ALTER\s+COLUMN|TRUNCATE|DELETE)\b", s, re.I)
    ]
    assert forbidden == [], forbidden
    assert "CREATE TABLE" not in sql.upper().replace("-- ", "")
    # Every ALTER carries IF NOT EXISTS: the file is applied by hand onto a
    # database that may already have some of the columns.
    alters = [s for s in statements if s.upper().startswith("ALTER TABLE")]
    assert len(alters) == 5
    assert all("IF NOT EXISTS" in s.upper() for s in alters)


def test_migration_023_adds_the_five_evidence_columns_in_order():
    """The Phase D column list, spelled out, in the order the ORM appends it."""
    sql = (MIGRATIONS / "023_news_evidence.sql").read_text()
    assert _added_columns(sql, "news_articles") == [
        "cluster_id",
        "materiality",
        "materiality_score",
        "source_quality",
        "relevance",
    ]


def test_migration_023_persists_no_as_of_dependent_field():
    """No ``novelty``/``decay``/``score`` column — the §96 rule, pinned in SQL.

    Those three are functions of the as-of instant and of which OTHER articles
    shared the window. Persisting one would freeze a single request's viewpoint
    onto the article row, and a later read at a different ``as_of`` would
    inherit it — the look-ahead leak audit §7.1 forbids, invisible to every
    test that only checks the payload. The four columns 023 DOES add are
    properties of the article itself.
    """
    added = _added_columns((MIGRATIONS / "023_news_evidence.sql").read_text(), "news_articles")
    for banned in ("novelty", "decay", "score", "evidence_score", "rank"):
        assert banned not in added


def test_migration_023_creates_the_tickers_gin_index():
    """The containment half of the window query has an index (Postgres)."""
    sql = (MIGRATIONS / "023_news_evidence.sql").read_text()
    assert (
        "CREATE INDEX IF NOT EXISTS idx_news_articles_tickers_gin" in sql
        and "USING GIN (tickers)" in sql
    )


def test_migration_024_stores_the_bundle_beside_the_analysis():
    """``bundle`` is NOT NULL and ``analysis`` is nullable — the §47 pin.

    The model may not compute a number: every figure in ``analysis`` must be
    QUOTED from ``bundle``, and the ``numbers_quoted`` validator checks each
    one against the bundle's fact index. That check only means anything
    against the document the model actually saw, so the bundle is a SNAPSHOT
    stored beside the text, never re-derived at read time from today's
    filings, prices and articles. A row without its evidence is not an
    analysis, it is an assertion — hence NOT NULL.

    ``analysis`` is the mirror image: a FAILED provider call has no output,
    and storing a placeholder narrative there would be exactly the fabricated
    content §44 rule 18 forbids. Same for ``latency_ms`` (0 would read as
    "answered instantly with nothing") and ``usage``.
    """
    sql = (MIGRATIONS / "024_event_analyses.sql").read_text()
    body = re.search(
        r"CREATE TABLE IF NOT EXISTS event_analyses \((.*?)\n\);", sql, re.S
    )
    assert body is not None
    lines = {
        line.strip().split()[0]: line.strip()
        for line in body.group(1).splitlines()
        if line.strip() and not line.strip().startswith("--")
    }
    assert "NOT NULL" in lines["bundle"]
    assert "NOT NULL" in lines["bundle_digest"]
    for nullable in ("analysis", "usage", "latency_ms", "provider", "model", "error"):
        assert "NOT NULL" not in lines[nullable], nullable
    # "checked and clean" ([]) is a different fact from "never checked" (NULL).
    assert "NOT NULL" in lines["violations"] and "DEFAULT '[]'" in lines["violations"]


def test_migration_024_cache_index_is_unique_versioned_and_PARTIAL():
    """The dedupe is a DATABASE fact, it is versioned, and it is scoped to OK.

    Two properties in one pin, because they trade off against each other and
    a future edit is likely to break one while "fixing" the other.

    UNIQUE and versioned: re-pressing "Analyse" on unchanged evidence must
    return the stored row rather than spend another provider call, and two
    concurrent handlers must not both write one — ADR-007 has no distributed
    lock, so the index is the whole mechanism. ``model`` and
    ``prompt_version`` are IN the key because the same evidence read by a
    different model, or under revised instructions, is a DIFFERENT answer that
    must coexist with the old one rather than collide with it.

    PARTIAL (``WHERE status = 'OK'``): a plain four-column UNIQUE would ALSO
    forbid a second attempt after a failure — the FAILED row carries the same
    event, digest, prompt version and model, so the user's retry, which is the
    one action that status invites, could not be stored. The same applies to
    re-running after an INVALID answer and to an explicit ``force``. The
    predicate keeps the property that matters (at most one cached GOOD answer)
    and lets the attempt trail accumulate.
    """
    sql = (MIGRATIONS / "024_event_analyses.sql").read_text()
    m = re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS uq_event_analyses_cache\s*"
        r"ON event_analyses \(([^)]*)\)\s*WHERE\s+status\s*=\s*'OK'",
        sql,
        re.S,
    )
    assert m is not None, "cache index must be UNIQUE and partial on status='OK'"
    assert [c.strip() for c in m.group(1).split(",")] == [
        "event_id",
        "bundle_digest",
        "prompt_version",
        "model",
    ]
    # A TABLE-level UNIQUE constraint would be total and would re-break the
    # retry path, so its absence is part of the pin.
    assert "CONSTRAINT uq_event_analyses_cache UNIQUE" not in sql


def test_orm_mirrors_the_partial_cache_index():
    """The ORM's index carries the same predicate, on sqlite too.

    The test harness builds its schema from the ORM alone (``create_all``), so
    an ORM index without ``sqlite_where`` would be TOTAL in every test while
    production Postgres stayed partial — and the retry path would fail only in
    production, which is the exact drift this file exists to catch.
    """
    from apps.gateway.db import Base

    index = next(
        i
        for i in Base.metadata.tables["event_analyses"].indexes
        if i.name == "uq_event_analyses_cache"
    )
    assert index.unique is True
    assert [c.name for c in index.columns] == [
        "event_id",
        "bundle_digest",
        "prompt_version",
        "model",
    ]
    for dialect in ("postgresql", "sqlite"):
        assert str(index.dialect_options[dialect]["where"]) == "status = 'OK'"


def test_migration_024_as_of_is_a_column_distinct_from_created_at():
    """A historical re-run is CREATED today and is AS-OF then (§96).

    Collapsing the two would make every stored analysis look like it was
    answering a question about the moment it was written, and the look-ahead
    audit would have no column to check.
    """
    sql = (MIGRATIONS / "024_event_analyses.sql").read_text()
    cols = _sql_columns(sql, "event_analyses")
    assert "as_of" in cols and "created_at" in cols
