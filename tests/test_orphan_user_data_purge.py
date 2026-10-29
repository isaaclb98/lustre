"""
Tests for the SQLite orphan purge.

The purge removes user-state rows (favourites, dislikes, album
memberships, album covers) whose photo is GONE FROM DISK.

The rule it used to follow — "delete the row if Qdrant no longer
holds the point" — was wrong, and dangerously so. A photo can be
absent from Qdrant while its file is perfectly alive on the NAS: it
was never indexed, or the collection was restored from a snapshot
older than its last index run. Right now 80,787 photos are in
exactly that state. Deleting their favourites would destroy real
user state for photos that are about to come back — and because the
point id is a uuid5 of the path, they WOULD come back as the same
id, re-attaching to a favourite that is no longer there.

So the rule is: a candidate row is an orphan only when its FILE is
gone from disk. The path comes from the `photo_paths` table,
recorded when the photo was favourited.

These tests pin both halves: the file-based rule, and the fallback
for rows that predate the `photo_paths` table.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from search.index_db import IndexDB


class FakeQdrant:
    """A Qdrant stand-in that reports which ids it holds."""

    def __init__(self, live_ids: set[str] | None = None) -> None:
        self.live_ids = live_ids if live_ids is not None else set()
        self.payloads: dict[str, dict] = {}
        self.collection = "images"
        self.collection_size = 0

    def count_points(self, exact: bool = True) -> int:
        # The collection size is independent of which ids it holds:
        # a collection with points that are NOT the ones we ask about
        # is still a non-empty collection, and that is what the
        # empty-collection guard keys on.
        return max(len(self.live_ids), self.collection_size)

    def retrieve_batch(self, point_ids: list[str]):
        from search.qdrant_client import SearchHit

        return [
            SearchHit(
                id=pid,
                path=self.payloads.get(pid, {}).get("path", ""),
                score=0.0,
                payload=self.payloads.get(pid),
            )
            for pid in point_ids
            if pid in self.live_ids
        ]

    def retrieve(self, point_id: str):
        from search.qdrant_client import SearchHit

        if point_id not in self.live_ids:
            return None
        return SearchHit(
            id=point_id,
            path=self.payloads.get(point_id, {}).get("path", ""),
            score=0.0,
            payload=self.payloads.get(point_id),
        )


@pytest.fixture
def photos_root(tmp_path) -> Path:
    """The mount the photos live on. Files here are 'on disk'."""
    root = tmp_path / "photos"
    root.mkdir()
    return root


@pytest.fixture
def db(tmp_path, photos_root) -> IndexDB:
    """An IndexDB whose photos root is a tmp dir."""
    return IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=FakeQdrant(),
        photos_root=str(photos_root),
    )


def _count(db: IndexDB, table: str, column: str = "id") -> int:
    """Count the rows in a user-state table.

    Not `list_favorites()` — that hydrates the metadata from Qdrant
    and omits rows whose photo Qdrant no longer holds. These tests
    are about whether the ROW survives, so they count the rows
    directly.
    """
    return db._conn.execute(
        f"SELECT COUNT(*) AS c FROM {table} WHERE {column} != ''",  # noqa: S608
    ).fetchone()["c"]


def _make_photo(photos_root: Path, name: str) -> str:
    """Create a photo file on disk and return its stored path."""
    p = photos_root / name
    p.write_bytes(b"x")
    return str(p)


def test_a_row_survives_while_the_file_is_on_disk(db, photos_root) -> None:
    """The core rule: the file is there, so the favourite stays.

    This is the case that the old rule got wrong. The photo is not
    in Qdrant (it has not been indexed yet, or the collection was
    restored from an older snapshot) but its file is alive. The
    favourite is valid and must survive.
    """
    path = _make_photo(photos_root, "alive.jpg")
    db.record_photo_path("id-alive", path)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-alive", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 0
    assert _count(db, "favorites") == 1


def test_a_row_is_deleted_when_the_file_is_gone(db, photos_root) -> None:
    """The file is gone, so the favourite is an orphan and goes."""
    path = _make_photo(photos_root, "doomed.jpg")
    db.record_photo_path("id-doomed", path)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-doomed", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()
    # The photo is deleted from disk.
    (photos_root / "doomed.jpg").unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 1
    assert _count(db, "favorites") == 0


def test_a_row_with_no_recorded_path_falls_back_to_qdrant(db) -> None:
    """Rows written before `photo_paths` existed use the Qdrant rule.

    The path is the only signal for the file-based rule, and an old
    row has none. Qdrant's liveness is then the only available
    evidence, so it is used — the old behaviour, applied to the
    small residue it applies to.
    """
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-legacy", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    # The collection is not empty — the empty-collection guard
    # does not apply, so the fallback decides.
    db.qdrant_client.collection_size = 5
    # Not in Qdrant -> orphan.
    purged = db.purge_orphaned_user_data()
    assert purged["favorites"] == 1

    # Now the same row, but Qdrant holds it -> survives.
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-legacy", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()
    db.qdrant_client.live_ids = {"id-legacy"}
    db.qdrant_client.collection_size = 5

    purged = db.purge_orphaned_user_data()
    assert purged["favorites"] == 0


def test_album_memberships_are_purged_by_file_liveness(db, photos_root) -> None:
    """An album member whose file is gone is removed too.

    Memberships do NOT require the photo to be a favourite — a photo
    can be in an album without being favourited — so liveness is
    checked against the file alone.
    """
    path = _make_photo(photos_root, "member.jpg")
    db.record_photo_path("id-member", path)
    db._conn.execute(
        "INSERT INTO albums (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (1, "Album", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    )
    db._conn.execute(
        "INSERT INTO album_memberships (album_id, favorite_id, added_at) "
        "VALUES (?, ?, ?)",
        (1, "id-member", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()
    (photos_root / "member.jpg").unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["album_memberships"] == 1


def test_a_dead_album_cover_is_reset(db, photos_root) -> None:
    """An album cover pointing at a dead photo is cleared, not deleted.

    The album itself is user-curated and stays; only the dangling
    cover reference is cleared, so the UI falls back to the first
    member.
    """
    path = _make_photo(photos_root, "cover.jpg")
    db.record_photo_path("id-cover", path)
    db._conn.execute(
        "INSERT INTO albums (id, name, created_at, updated_at, cover_favorite_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (1, "Album", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "id-cover"),
    )
    db._conn.commit()
    (photos_root / "cover.jpg").unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["album_covers_reset"] == 1
    row = db._conn.execute(
        "SELECT cover_favorite_id FROM albums WHERE id = 1",
    ).fetchone()
    assert row["cover_favorite_id"] == ""


def test_dislikes_are_purged_by_file_liveness(db, photos_root) -> None:
    """Dislikes follow the same rule as favourites."""
    path = _make_photo(photos_root, "disliked.jpg")
    db.record_photo_path("id-disliked", path)
    db._conn.execute(
        "INSERT INTO dislikes (id, disliked_at, source) VALUES (?, ?, ?)",
        ("id-disliked", "2026-01-01T00:00:00Z", "manual"),
    )
    db._conn.commit()
    (photos_root / "disliked.jpg").unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["dislikes"] == 1


def test_an_unresolvable_path_counts_as_alive(db, photos_root) -> None:
    """A path that cannot be checked is NOT treated as gone.

    Treating "cannot check" as "gone" would delete the user's
    favourites on any host that does not mount the photos — the same
    mistake the Qdrant-liveness rule was introduced to avoid, with a
    different trigger.
    """
    db.record_photo_path("id-unknown", "/some/other/mount/photo.jpg")
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-unknown", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 0


def test_no_photos_root_means_nothing_is_purged(db) -> None:
    """With no photos root configured, the file check cannot run.

    A host that does not mount the photos (a test, a laptop) must
    not purge anything on the strength of a check it could not
    perform.
    """
    db.photos_root = ""
    db.record_photo_path("id-any", "/nas/telegram/1.jpg")
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-any", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 0


def test_the_purge_keeps_feedback_events_and_saved_searches(
    db, photos_root,
) -> None:
    """The history log and the saved prompts are not state.

    `feedback_events` is an append-only history log — nothing feeds
    it back into recommend — and `saved_searches` holds prompt text
    with no point ids. Neither is touched by the purge.
    """
    path = _make_photo(photos_root, "gone.jpg")
    db.record_photo_path("id-gone", path)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-gone", "2026-01-01T00:00:00Z"),
    )
    db._conn.execute(
        "INSERT INTO feedback_events (photo_id, kind, at, source) "
        "VALUES (?, ?, ?, ?)",
        ("id-gone", "like", "2026-01-01T00:00:00Z", "manual"),
    )
    db._conn.execute(
        "INSERT INTO saved_searches (name, positives, negatives, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("Mine", "[]", "[]", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()
    (photos_root / "gone.jpg").unlink()

    db.purge_orphaned_user_data()

    assert db._conn.execute(
        "SELECT COUNT(*) AS c FROM feedback_events",
    ).fetchone()["c"] == 1
    assert db._conn.execute(
        "SELECT COUNT(*) AS c FROM saved_searches",
    ).fetchone()["c"] == 1


def test_an_empty_collection_does_not_purge_everything(db, photos_root) -> None:
    """The NAS-dropout guard still holds.

    A collection that reports ZERO points is far more likely to mean
    "the collection was just recreated" than "every photo was
    deleted". Purging on that signal would wipe the user's entire
    taste history irrecoverably.
    """
    path = _make_photo(photos_root, "still-here.jpg")
    db.record_photo_path("id-here", path)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-here", "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()
    # Qdrant holds nothing — the collection looks empty.
    db.qdrant_client.live_ids = set()

    purged = db.purge_orphaned_user_data()

    # The file is still on disk, so the row survives regardless.
    assert purged["favorites"] == 0
    assert _count(db, "favorites") == 1


def test_delete_user_state_for_ids_cleans_the_photo_paths(db, photos_root) -> None:
    """The confirmed purge removes the path rows too.

    The photo is gone, so its path record is dead weight. Leaving it
    behind would grow the table forever.
    """
    db.record_photo_path("id-1", "/nas/telegram/1.jpg")
    db.record_photo_path("id-2", "/nas/telegram/2.jpg")
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("id-1", "2026-01-01T00:00:00Z"),
    )
    db._conn.execute(
        "INSERT INTO dislikes (id, disliked_at, source) VALUES (?, ?, ?)",
        ("id-2", "2026-01-01T00:00:00Z", "manual"),
    )
    db._conn.commit()

    purged = db.delete_user_state_for_ids(["id-1"])

    assert purged["favorites"] == 1
    assert purged["dislikes"] == 0
    assert purged["photo_paths"] == 1
    assert db.get_photo_path("id-1") is None
    assert db.get_photo_path("id-2") is not None
