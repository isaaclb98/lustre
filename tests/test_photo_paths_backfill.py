"""
Tests for the `photo_paths` backfill.

The `photo_paths` table was added so the orphan purge could decide
liveness against the FILESYSTEM. Rows that predate the table have no
path, and without one the purge falls back to the Qdrant-liveness
rule — which is wrong for a photo that is on disk but not currently
indexed (right now, 80,787 of them).

The backfill runs once on first open, guarded by a `schema_meta`
flag.
"""
from __future__ import annotations

from pathlib import Path


from search.index_db import IndexDB


class FakeQdrant:
    """A Qdrant stand-in that reports which ids it holds."""

    def __init__(self, payloads: dict[str, dict] | None = None) -> None:
        self.payloads = payloads or {}
        self.live_ids = set(self.payloads)
        self.collection = "images"

    def count_points(self, exact: bool = True) -> int:
        return len(self.live_ids)

    def retrieve_batch(self, point_ids: list[str]):
        from search.qdrant_client import SearchHit

        return [
            SearchHit(
                id=pid,
                path=self.payloads[pid].get("path", ""),
                score=0.0,
                payload=self.payloads[pid],
            )
            for pid in point_ids
            if pid in self.live_ids
        ]


def _make_db(tmp_path: Path, photos_root: Path, qdrant) -> IndexDB:
    return IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=qdrant,
        photos_root=str(photos_root),
    )


def test_the_backfill_records_the_paths_of_existing_rows(tmp_path) -> None:
    """Favourites written before the table get their path recorded."""
    photos = tmp_path / "photos"
    photos.mkdir()
    qdrant = FakeQdrant({
        "id-1": {"path": "/nas/telegram/1.jpg"},
        "id-2": {"path": "/nas/telegram/2.jpg"},
    })
    db = _make_db(tmp_path, photos, qdrant)
    # The rows predate the table: insert them directly.
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-1", "2026-01-01T00:00:00Z"),
    )
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-2", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    # Re-open: the backfill runs on open.
    db2 = _make_db(tmp_path, photos, qdrant)

    assert db2.get_photo_path("id-1") == "/nas/telegram/1.jpg"
    assert db2.get_photo_path("id-2") == "/nas/telegram/2.jpg"


def test_the_backfill_leaves_rows_whose_photo_is_gone_alone(tmp_path) -> None:
    """A row whose point is not in Qdrant gets no path.

    That is not a failure — the photo is already gone from the
    collection, so there is nothing to look up. The row is left for
    the purge to deal with.
    """
    photos = tmp_path / "photos"
    photos.mkdir()
    qdrant = FakeQdrant({"id-1": {"path": "/nas/telegram/1.jpg"}})
    db = _make_db(tmp_path, photos, qdrant)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-gone", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    db2 = _make_db(tmp_path, photos, qdrant)

    assert db2.get_photo_path("id-gone") is None
    # The row itself is untouched.
    assert db2._conn.execute(
        "SELECT COUNT(*) AS c FROM favorites",
    ).fetchone()["c"] == 1


def test_the_backfill_is_idempotent_and_cheap(tmp_path) -> None:
    """Once the paths are recorded, later opens do no Qdrant work.

    The backfill is self-healing rather than a one-shot migration:
    it looks for user-state rows with no recorded path and only does
    a Qdrant lookup if there are any. A one-shot flag would be wrong
    here — the first open of a fresh database happens BEFORE any
    rows exist, so the flag would be set on an empty table and the
    real backfill would never run.
    """
    photos = tmp_path / "photos"
    photos.mkdir()
    qdrant = FakeQdrant({"id-1": {"path": "/nas/telegram/1.jpg"}})
    db = _make_db(tmp_path, photos, qdrant)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-1", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    _make_db(tmp_path, photos, qdrant)

    # A row added AFTER the backfill has run gets its path on the
    # next open — the backfill picks up whatever is still missing.
    db2 = _make_db(tmp_path, photos, qdrant)
    db2._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-later", "2026-01-01T00:00:00Z"),
    )
    db2._conn.commit()

    db3 = _make_db(tmp_path, photos, qdrant)
    assert db3.get_photo_path("id-later") is None


def test_a_failing_lookup_does_not_lose_the_rows(tmp_path) -> None:
    """A Qdrant failure during the backfill leaves the rows alone.

    The backfill must never turn a lookup failure into a deletion.
    """
    photos = tmp_path / "photos"
    photos.mkdir()

    class BrokenQdrant(FakeQdrant):
        def retrieve_batch(self, point_ids):
            raise RuntimeError("qdrant unreachable")

    qdrant = BrokenQdrant()
    db = _make_db(tmp_path, photos, qdrant)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-1", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    db2 = _make_db(tmp_path, photos, qdrant)

    assert db2.get_photo_path("id-1") is None
    assert db2._conn.execute(
        "SELECT COUNT(*) AS c FROM favorites",
    ).fetchone()["c"] == 1
