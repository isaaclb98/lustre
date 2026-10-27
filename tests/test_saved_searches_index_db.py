"""IndexDB unit tests for the saved_searches table.

These exercise the four IndexDB methods (create / list / get /
delete) directly against an in-memory SQLite database. The
in-memory database wiring reuses the same threading.Lock and
executescript pattern the production `_init_schema` uses, so the
tests cover the same SQL surface the routes do.

Saved searches store `positives` and `negatives` as JSON-encoded
strings on disk; the deserialise helper always returns Python
lists to callers, so every test asserts on lists directly.
"""
from __future__ import annotations

import sqlite3

import pytest

from search.index_db import IndexDB


@pytest.fixture
def index_db():
    """IndexDB pointed at :memory:, bypassing the qdrant refresh path.

    The Qdrant collection is what populates the photo data from
    Qdrant; the saved-searches tests don't touch that path, so we
    short-circuit it by leaving `qdrant_client=None` (the constructor
    tolerates None — only the refresh path uses it). This gives us
    a clean, isolated schema for testing the saved-searches methods.
    """
    db = IndexDB(":memory:", qdrant_client=None)
    yield db
    db.close()


def test_create_saved_search_returns_full_row(index_db):
    row = index_db.create_saved_search(
        "red-dress-no-manikin",
        ["red dress", "studio lighting"],
        ["manikin", "mannequin"],
    )
    assert row["name"] == "red-dress-no-manikin"
    assert row["positives"] == ["red dress", "studio lighting"]
    assert row["negatives"] == ["manikin", "mannequin"]
    assert isinstance(row["id"], int)
    assert row["created_at"]


def test_create_saved_search_duplicate_name_raises(index_db):
    index_db.create_saved_search("dup-name", ["a"], [])
    with pytest.raises(ValueError, match="already exists"):
        index_db.create_saved_search("dup-name", ["b"], [])


def test_create_saved_search_strips_name(index_db):
    row = index_db.create_saved_search("  padded-name  ", ["x"], [])
    assert row["name"] == "padded-name"


def test_list_saved_searches_orders_by_created_at_desc(index_db):
    # Insert three rows; newer rows must come first in the list
    # response. SQLite's TEXT-comparison ordering of ISO timestamps
    # is monotonic, so this is reliable without explicit
    # fractional-second handling.
    a = index_db.create_saved_search("first", ["a"], [])
    b = index_db.create_saved_search("second", ["b"], [])
    c = index_db.create_saved_search("third", ["c"], [])
    rows, total = index_db.list_saved_searches(limit=10, offset=0)
    assert total == 3
    assert [r["name"] for r in rows] == ["third", "second", "first"]
    assert [r["id"] for r in rows] == [c["id"], b["id"], a["id"]]


def test_list_saved_searches_paginates(index_db):
    for i in range(5):
        index_db.create_saved_search(f"s-{i}", [], [f"n{i}"])
    rows, total = index_db.list_saved_searches(limit=2, offset=0)
    assert total == 5
    assert len(rows) == 2
    rows2, _ = index_db.list_saved_searches(limit=2, offset=2)
    assert len(rows2) == 2
    # No overlap between pages.
    ids_page1 = {r["id"] for r in rows}
    ids_page2 = {r["id"] for r in rows2}
    assert ids_page1.isdisjoint(ids_page2)


def test_get_saved_search_round_trip(index_db):
    created = index_db.create_saved_search(
        "  padded  ", ["p1", "p2"], ["n1"]
    )
    fetched = index_db.get_saved_search(created["id"])
    assert fetched is not None
    assert fetched["name"] == "padded"  # trimmed on insert
    assert fetched["positives"] == ["p1", "p2"]
    assert fetched["negatives"] == ["n1"]


def test_get_saved_search_missing_returns_none(index_db):
    assert index_db.get_saved_search(99999) is None


def test_delete_saved_search_returns_true_on_hit(index_db):
    created = index_db.create_saved_search("to-go", ["x"], [])
    assert index_db.delete_saved_search(created["id"]) is True
    assert index_db.get_saved_search(created["id"]) is None


def test_delete_saved_search_returns_false_on_miss(index_db):
    assert index_db.delete_saved_search(99999) is False


def test_corrupt_json_in_db_deserialises_to_empty_list(tmp_path):
    """A single corrupt row must not break the whole /api/saved-searches
    response. The deserialise helper should treat unparseable JSON
    as an empty list rather than raise, so the rest of the rows
    still return cleanly.
    """
    db_path = str(tmp_path / "saved.db")
    # Bypass IndexDB so we can write malformed JSON directly.
    raw = sqlite3.connect(db_path)
    raw.executescript("""
        CREATE TABLE saved_searches (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE,
          positives TEXT NOT NULL DEFAULT '[]',
          negatives TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL
        );
    """)
    raw.execute(
        "INSERT INTO saved_searches (name, positives, negatives, created_at) VALUES (?, ?, ?, ?)",
        ("broken", "{not json", "[]", "2026-06-27T00:00:00Z"),
    )
    raw.execute(
        "INSERT INTO saved_searches (name, positives, negatives, created_at) VALUES (?, ?, ?, ?)",
        ("good", '["hello"]', "[]", "2026-06-27T00:00:01Z"),
    )
    raw.commit()
    raw.close()

    db = IndexDB(db_path, qdrant_client=None)
    try:
        rows, total = db.list_saved_searches(limit=10, offset=0)
        assert total == 2
        by_name = {r["name"]: r for r in rows}
        assert by_name["broken"]["positives"] == []  # corrupt → empty
        assert by_name["good"]["positives"] == ["hello"]
    finally:
        db.close()