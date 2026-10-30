"""
tests/test_multi_root_orphan_purge.py — the purge across several mounts.

A production library is not always one directory. A source can be
mounted as a SIBLING of the photos root (a second NAS, a second
share), and its photos then live under a different top-level path in
the container.

The purge's liveness check resolves a stored payload path onto a
local file. It could only do that under ONE root — the photos mount
— so a photo on a sibling mount was unresolvable, and "unresolvable"
means ALIVE (the rule that stops the purge deleting favourites for
photos it simply cannot see). The consequence was that the sibling
mount's favourites were never garbage collected: the photo could be
deleted from disk and its favourite row would stay forever.

These tests pin the multi-root behaviour:

- a photo under a second root IS garbage collected when its file goes
- a photo under an UNKNOWN root (a namespace we cannot see) is kept
- an unreachable root (a mount that is down) keeps its rows — a mount
  that is down is not evidence that its files were deleted
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
def roots(tmp_path) -> tuple[Path, Path]:
    """Two separate mounts: the photos root and a sibling source."""
    main = tmp_path / "photos"
    sibling = tmp_path / "sibling"
    main.mkdir()
    sibling.mkdir()
    return main, sibling


@pytest.fixture
def db(tmp_path, roots) -> IndexDB:
    main, sibling = roots
    return IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=FakeQdrant(),
        photos_root=str(main),
        photos_roots=(str(main), str(sibling)),
    )


def _favourite(db: IndexDB, point_id: str, path: str) -> None:
    db.record_photo_path(point_id, path)
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        (point_id, "2026-01-01T00:00:00Z"),
    )
    db._conn.commit()


def _count(db: IndexDB, table: str) -> int:
    return db._conn.execute(
        f"SELECT COUNT(*) AS c FROM {table} WHERE id != ''",  # noqa: S608
    ).fetchone()["c"]


def test_a_favourite_on_a_sibling_mount_is_garbage_collected(db, roots) -> None:
    """The file under the second root is gone, so the favourite goes.

    This is the case the single-root purge could not handle. The
    photo lives under the sibling mount, the file is deleted, and the
    favourite row is still there. A purge that only knows the photos
    root cannot resolve the path, so it treats the row as alive and
    keeps it forever — the favourite of a photo that no longer exists
    sits in the user's list indefinitely.
    """
    _, sibling = roots
    photo = sibling / "gone.jpg"
    photo.write_bytes(b"x")
    _favourite(db, "id-sibling", str(photo))

    # The photo is deleted from disk.
    photo.unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 1
    assert _count(db, "favorites") == 0


def test_a_favourite_on_a_sibling_mount_survives_while_the_file_is_there(
    db, roots
) -> None:
    """The file under the second root is alive, so the favourite stays."""
    _, sibling = roots
    photo = sibling / "alive.jpg"
    photo.write_bytes(b"x")
    _favourite(db, "id-alive", str(photo))

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 0
    assert _count(db, "favorites") == 1


def test_a_path_outside_every_root_is_kept(db) -> None:
    """A path in a namespace we cannot see is ALIVE, not gone.

    The stored path might come from another machine's mount. It does
    not resolve here, so we cannot check it — and "cannot check" must
    never be treated as "deleted", because that would delete the
    user's favourites on any host that does not mount the photos.
    """
    _favourite(db, "id-foreign", "/some/other/mount/photo.jpg")

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 0
    assert _count(db, "favorites") == 1


def test_an_unreachable_root_keeps_its_rows(tmp_path, roots) -> None:
    """A mount that is DOWN is not evidence that its files were deleted.

    The sibling root is removed from disk entirely — the mount is
    gone, so every file under it looks missing. Treating that as
    "the photos were deleted" would wipe the favourites of an entire
    library because a NAS share was briefly unreachable. An
    unreachable root means we cannot check, so the rows stay.
    """
    main, sibling = roots
    photo = sibling / "unreachable.jpg"
    photo.write_bytes(b"x")
    db = IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=FakeQdrant(),
        photos_root=str(main),
        photos_roots=(str(main), str(sibling)),
    )
    _favourite(db, "id-unreachable", str(photo))

    # The mount goes away: the file inside it is unlinked, and then
    # the directory that held it is removed, so the root no longer
    # exists at all.
    photo.unlink()
    sibling.rmdir()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 0
    assert _count(db, "favorites") == 1


def test_the_photos_root_still_works_with_a_second_root(tmp_path, roots) -> None:
    """Adding a second root does not break the first one's purge."""
    main, sibling = roots
    photo = main / "doomed.jpg"
    photo.write_bytes(b"x")
    db = IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=FakeQdrant(),
        photos_root=str(main),
        photos_roots=(str(main), str(sibling)),
    )
    _favourite(db, "id-doomed", str(photo))
    photo.unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 1
    assert _count(db, "favorites") == 0


def test_a_single_root_still_purges(tmp_path) -> None:
    """A single-root deployment behaves exactly as before.

    `photos_roots` is optional. When it is not given, the photos root
    is the only root, which is the pre-existing behaviour.
    """
    main = tmp_path / "photos"
    main.mkdir()
    photo = main / "doomed.jpg"
    photo.write_bytes(b"x")
    db = IndexDB(
        db_path=str(tmp_path / "index.db"),
        qdrant_client=FakeQdrant(),
        photos_root=str(main),
    )
    _favourite(db, "id-doomed", str(photo))
    photo.unlink()

    purged = db.purge_orphaned_user_data()

    assert purged["favorites"] == 1
    assert _count(db, "favorites") == 0
