from __future__ import annotations

import time

import pytest

from search.index_db import ImageNotInCacheError, IndexDB


class FakeQdrant:
    def __init__(self, batches):
        self.batches = batches
        self.refreshes = 0

    def scroll_all(self, batch_size=1000):
        self.refreshes += 1
        yield from self.batches

    def retrieve_batch(self, point_ids):
        """The points this fake holds — the authoritative liveness set.

        The orphan purge checks liveness against Qdrant, not against
        the derived `images` mirror, so the fake has to answer the
        same question the real client does: which of these ids do
        you still hold?
        """
        from search.qdrant_client import SearchHit

        wanted = {str(i) for i in point_ids}
        live = []
        for batch in self.batches:
            for point in batch:
                if str(point["id"]) in wanted:
                    live.append(
                        SearchHit(
                            id=str(point["id"]),
                            path=point["payload"].get("path", ""),
                            score=0.0,
                            payload=point["payload"],
                        )
                    )
        return live


def _point(pid: str, path: str) -> dict:
    return {
        "id": pid,
        "payload": {
            "id": pid,
            "path": path,
            "shard": "",
            "mtime": 100,
            "size": 200,
            "indexed_at": "2026-01-01T00:00:00+00:00",
        },
    }


@pytest.fixture
def index_db(tmp_path):
    qdrant = FakeQdrant([[_point("a", "/photos/a.jpg"), _point("b", "/photos/b.jpg")]])
    db = IndexDB(str(tmp_path / "images.db"), qdrant, refresh_interval_seconds=3600)
    try:
        yield db, qdrant
    finally:
        db.close()


def test_init_from_qdrant_populates_table(index_db):
    db, _ = index_db
    assert db.init_from_qdrant() == 2
    assert db.get_by_id("a")["path"] == "/photos/a.jpg"
    assert db.get_by_id("b")["size"] == 200


def test_pick_random_returns_requested_ids(index_db):
    db, _ = index_db
    db.init_from_qdrant()
    ids = db.pick_random(2)
    assert len(ids) == 2
    assert set(ids) == {"a", "b"}


def test_pick_unseen_excludes_seen(index_db):
    db, _ = index_db
    db.init_from_qdrant()
    assert db.pick_unseen(10, {"a"}) == ["b"]


def test_favorite_round_trip(index_db):
    db, _ = index_db
    db.init_from_qdrant()
    db.mark_favorite("a")
    assert db.count_favorites() == 1
    favorites = db.list_favorites()
    assert favorites[0]["id"] == "a"
    assert favorites[0]["favorited_at"]
    db.unmark_favorite("a")
    assert db.count_favorites() == 0


def test_mark_favorite_requires_cached_photo(index_db):
    db, _ = index_db
    db.init_from_qdrant()
    with pytest.raises(ImageNotInCacheError):
        db.mark_favorite("missing")


def test_maybe_refresh_respects_interval(index_db):
    db, qdrant = index_db
    db.init_from_qdrant()
    refreshes = qdrant.refreshes
    assert db.maybe_refresh() is False
    assert qdrant.refreshes == refreshes
    db._last_refresh = time.time() - 4000
    assert db.maybe_refresh() is True
    assert qdrant.refreshes == refreshes + 1


def test_get_by_id_returns_none_for_missing(index_db):
    db, _ = index_db
    db.init_from_qdrant()
    assert db.get_by_id("missing") is None


def test_init_from_qdrant_skips_scroll_when_cache_populated(tmp_path):
    """A populated SQLite cache must survive across IndexDB instances
    without triggering a Qdrant scroll."""
    qdrant_a = FakeQdrant([[_point("a", "/photos/a.jpg"), _point("b", "/photos/b.jpg")]])
    db_path = str(tmp_path / "images.db")
    db_a = IndexDB(db_path, qdrant_a, refresh_interval_seconds=3600)
    assert db_a.init_from_qdrant() == 2
    db_a.close()

    # New Qdrant collection grew a third point; the cached DB doesn't know yet.
    qdrant_b = FakeQdrant(
        [
            [
                _point("a", "/photos/a.jpg"),
                _point("b", "/photos/b.jpg"),
                _point("c", "/photos/c.jpg"),
            ]
        ]
    )
    db_b = IndexDB(db_path, qdrant_b, refresh_interval_seconds=3600)
    try:
        refreshes_before = qdrant_b.refreshes
        # Default call must NOT scroll Qdrant — it just restores from SQLite.
        assert db_b.init_from_qdrant() == 2
        assert qdrant_b.refreshes == refreshes_before
        assert db_b.count_images() == 2
        assert db_b.get_by_id("c") is None
        # `force=True` bypasses the skip and rebuilds from current Qdrant state.
        assert db_b.init_from_qdrant(force=True) == 3
        assert qdrant_b.refreshes == refreshes_before + 1
        assert db_b.get_by_id("c") is not None
    finally:
        db_b.close()


def test_init_from_qdrant_skip_resets_refresh_clock(index_db):
    db, qdrant = index_db
    assert db.init_from_qdrant() == 2
    refreshes = qdrant.refreshes
    db._last_refresh = 0.0  # simulate state right after process restart
    # Skip path must bump the refresh clock so maybe_refresh() doesn't
    # immediately re-scroll right after startup.
    db.init_from_qdrant()
    assert db._last_refresh > 0
    assert qdrant.refreshes == refreshes  # still no scroll


def test_random_picker_delegates_to_index_db():
    import asyncio
    import threading

    from search.random import RandomPicker

    class FakeIndexDB:
        def __init__(self):
            self.called_with = None

        def pick_random(self, n):
            self.called_with = n
            return ["x", "y"]

    db = FakeIndexDB()
    # The sync Playwright session keeps an event loop running in the
    # main thread, which makes asyncio.run() and
    # loop.run_until_complete() unusable there. A private loop on its
    # own thread is immune to whatever the main thread is doing.
    bg_loop = asyncio.new_event_loop()
    bg_thread = threading.Thread(target=bg_loop.run_forever, daemon=True)
    bg_thread.start()
    try:
        future = asyncio.run_coroutine_threadsafe(
            RandomPicker(db).pick(2), bg_loop
        )
        picked = future.result(timeout=5)
    finally:
        bg_loop.call_soon_threadsafe(bg_loop.stop)
        bg_thread.join()
    assert picked == ["x", "y"]
    assert db.called_with == 2


# ---------- pick_random_rows (random shuffle page) ----------

def test_pick_random_rows_returns_full_metadata(tmp_path):
    """pick_random_rows returns dicts with id/path/width/height/etc —
    everything the random page needs to render without going back
    to Qdrant.
    """
    qdrant = FakeQdrant(
        [
            [
                {"id": "a", "payload": {"id": "a", "path": "/photos/a.jpg", "collection": "kpop", "indexed_at": "2026-01-01"}},
                {"id": "b", "payload": {"id": "b", "path": "/photos/b.jpg", "collection": "portrait", "indexed_at": "2026-01-01"}},
            ]
        ]
    )
    db_path = str(tmp_path / "images.db")
    db = IndexDB(db_path, qdrant, refresh_interval_seconds=3600)
    try:
        db.init_from_qdrant()
        rows = db.pick_random_rows(2)
        assert len(rows) == 2
        ids = {r["id"] for r in rows}
        assert ids == {"a", "b"}
        # Spot-check the fields the random page reads.
        for r in rows:
            assert r["path"].startswith("/photos/")
            assert "collection" in r
            assert "width" in r
            assert "height" in r
    finally:
        db.close()


def test_pick_random_rows_filters_by_collection(tmp_path):
    """When collections=[...] is passed, only matching rows are sampled."""
    qdrant = FakeQdrant(
        [
            [
                {"id": "a", "payload": {"id": "a", "path": "/a.jpg", "collection": "kpop"}},
                {"id": "b", "payload": {"id": "b", "path": "/b.jpg", "collection": "portrait"}},
                {"id": "c", "payload": {"id": "c", "path": "/c.jpg", "collection": "kpop"}},
            ]
        ]
    )
    db_path = str(tmp_path / "images.db")
    db = IndexDB(db_path, qdrant, refresh_interval_seconds=3600)
    try:
        db.init_from_qdrant()
        rows = db.pick_random_rows(10, collections=["kpop"])
        assert {r["id"] for r in rows} == {"a", "c"}
        # Multi-collection filter: union of the requested libraries.
        rows_multi = db.pick_random_rows(10, collections=["kpop", "portrait"])
        assert {r["id"] for r in rows_multi} == {"a", "b", "c"}
    finally:
        db.close()


def test_pick_random_rows_zero_or_negative_returns_empty(tmp_path):
    """n=0 or n<0 is a no-op, not a crash."""
    qdrant = FakeQdrant([])
    db_path = str(tmp_path / "images.db")
    db = IndexDB(db_path, qdrant, refresh_interval_seconds=3600)
    try:
        assert db.pick_random_rows(0) == []
        assert db.pick_random_rows(-5) == []
    finally:
        db.close()


def test_collection_column_migrated_for_existing_db(tmp_path):
    """A pre-random-shuffle DB without the `collection` column gets
    it added in-place on IndexDB open, same pattern as width/height.
    """
    import sqlite3

    db_path = str(tmp_path / "legacy.db")
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE images (
          id            TEXT PRIMARY KEY,
          path          TEXT NOT NULL,
          shard         TEXT DEFAULT '',
          mtime         INTEGER,
          size          INTEGER,
          indexed_at    TEXT,
          is_favorite   INTEGER DEFAULT 0,
          favorited_at  TEXT
        );
        INSERT INTO images (id, path) VALUES ('legacy1', '/photos/legacy.jpg');
        """
    )
    legacy.commit()
    legacy.close()

    qdrant = FakeQdrant([])
    db = IndexDB(db_path, qdrant, refresh_interval_seconds=3600)
    try:
        # Legacy row is still there, collection defaulted to "".
        cols = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(images)").fetchall()
        }
        assert "collection" in cols
        assert db.get_by_id("legacy1")["collection"] == ""
    finally:
        db.close()


# ---------- favorites persistence across cache rebuild ----------

def test_favorites_survive_init_from_qdrant_force_rebuild(tmp_path):
    """A mark_favorite call must persist across a full
    init_from_qdrant(force=True) rebuild. This is the whole point of
    the persistence refactor: user state lives in a separate table
    that the rebuild never touches.
    """
    from PIL import Image

    real = tmp_path / "real.jpg"
    Image.new("RGB", (640, 480), (0, 0, 0)).save(real)

    qdrant = FakeQdrant(
        [
            [
                {"id": "real", "payload": {"id": "real", "path": str(real), "indexed_at": "2026-01-01"}},
            ]
        ]
    )
    db_path = str(tmp_path / "images.db")
    db = IndexDB(db_path, qdrant, refresh_interval_seconds=3600)
    try:
        db.init_from_qdrant()
        db.mark_favorite("real")
        # Favourite recorded.
        row = db.get_by_id("real")
        assert bool(row["is_favorite"])
        assert row["favorited_at"]

        # Force a full rebuild from Qdrant — same data this time.
        db.init_from_qdrant(force=True)
        row = db.get_by_id("real")
        assert bool(row["is_favorite"])
        assert row["favorited_at"]
        assert db.count_favorites() == 1
    finally:
        db.close()


def test_favorite_survives_photo_removal_from_qdrant(tmp_path):
    """If a photo's id disappears from Qdrant AND the rebuild comes
    back empty, the favourite survives as an orphan (the NAS-dropout
    guard: an empty scroll means 'Qdrant unreachable', not 'photo
    deleted', so we don't purge). It re-attaches automatically if the
    same id reappears.

    Contrast with test_purge_removes_orphan_favorites_and_dislikes:
    when the rebuild is NON-empty, orphans ARE purged — a non-empty
    scroll is trustworthy evidence the photo is really gone.
    """
    from PIL import Image

    real = tmp_path / "real.jpg"
    Image.new("RGB", (640, 480), (0, 0, 0)).save(real)

    qdrant_present = FakeQdrant(
        [
            [
                {"id": "real", "payload": {"id": "real", "path": str(real), "indexed_at": "2026-01-01"}},
            ]
        ]
    )
    db_path = str(tmp_path / "images.db")
    db = IndexDB(db_path, qdrant_present, refresh_interval_seconds=3600)
    try:
        db.init_from_qdrant()
        db.mark_favorite("real")
        assert db.count_favorites() == 1

        # Qdrant loses the point (operator deleted the source photo,
        # ran heal, etc). The cache rebuilds without it.
        db.qdrant_client = FakeQdrant([])
        db.init_from_qdrant(force=True)

        # No row in `images` for the deleted id...
        assert db.get_by_id("real") is None
        # ...but the favourite row in `favorites` survives.
        assert db.count_favorites() == 0  # 0 because JOIN excludes orphans
        with db._lock:
            row = db._conn.execute(
                "SELECT id FROM favorites WHERE id = ?", ("real",)
            ).fetchone()
        assert row is not None and row["id"] == "real"

        # Re-indexing the same id (same uuid5) re-attaches it.
        db.qdrant_client = qdrant_present
        db.init_from_qdrant(force=True)
        assert db.count_favorites() == 1
        assert bool(db.get_by_id("real")["is_favorite"])
    finally:
        db.close()
def test_favorite_id_set_returns_only_listed_favourites(index_db):
    """favorite_id_set(ids) returns the subset of ids that are favourited.

    Single IN-clause query (Phase C1): 1 SQLite round trip instead
    of N individual get_by_id calls.
    """
    db, _ = index_db
    db.init_from_qdrant()
    # Mark 2 ids as favourites (a and b are the seeded points).
    db.mark_favorite("a")
    db.mark_favorite("b")
    # Ask for the 2 known ids + a missing one; expect a + b back.
    out = db.favorite_id_set(["a", "b", "missing"])
    assert out == {"a", "b"}


def test_favorite_id_set_empty_input_returns_empty_set(tmp_path):
    from unittest.mock import MagicMock

    from search.index_db import IndexDB

    qdrant = MagicMock()
    db = IndexDB(str(tmp_path / "images.db"), qdrant, refresh_interval_seconds=3600)
    assert db.favorite_id_set([]) == set()


# ---------------------------------------------------------------------------
# purge_orphaned_user_data — the core fix for the prune aftermath:
# favourites/dislikes/album rows whose photo left Qdrant must be
# deleted on the next forced cache rebuild, not kept forever as
# orphans (they poison qdrant.recommend and render as dead tiles).
# ---------------------------------------------------------------------------


def _point_full(pid: str, path: str) -> dict:
    return {"id": pid, "payload": {"id": pid, "path": path, "indexed_at": "2026-01-01"}}


def _make_db(tmp_path, initial_points):
    qdrant = FakeQdrant([[_point_full(p, f"/photos/{p}.jpg") for p in initial_points]])
    db = IndexDB(str(tmp_path / "images.db"), qdrant, refresh_interval_seconds=3600)
    db.init_from_qdrant()
    return db


def test_purge_removes_orphan_favorites_and_dislikes(tmp_path):
    db = _make_db(tmp_path, ["a", "b"])
    try:
        db.mark_favorite("a")
        db.mark_favorite("b")
        db.mark_dislike("a")
        db.mark_dislike("b")

        # 'b' is pruned from Qdrant (file deleted from disk).
        db.qdrant_client = FakeQdrant([[_point_full("a", "/photos/a.jpg")]])
        db.init_from_qdrant(force=True)

        # Orphan rows for 'b' are gone from the raw tables...
        with db._lock:
            fav_rows = db._conn.execute("SELECT id FROM favorites").fetchall()
            dis_rows = db._conn.execute("SELECT id FROM dislikes").fetchall()
        assert {r["id"] for r in fav_rows} == {"a"}
        assert {r["id"] for r in dis_rows} == {"a"}
        # ...and the live favourite is untouched.
        assert db.count_favorites() == 1
    finally:
        db.close()


def test_purge_removes_orphan_album_memberships_and_resets_cover(tmp_path):
    db = _make_db(tmp_path, ["a", "b"])
    try:
        album_id = db.create_album("Test album")
        db.add_album_member(album_id, "a")
        db.add_album_member(album_id, "b")
        db.set_album_cover(album_id, "b")

        # 'b' is pruned.
        db.qdrant_client = FakeQdrant([[_point_full("a", "/photos/a.jpg")]])
        db.init_from_qdrant(force=True)

        members = db.list_album_member_ids(album_id)
        assert members == ["a"]
        album = db.get_album(album_id)
        # Cover pointed at the pruned photo → reset to '' so the UI
        # falls back to first_member_id.
        assert album["cover_favorite_id"] == ""
    finally:
        db.close()


def test_purge_keeps_live_rows_and_feedback_history(tmp_path):
    db = _make_db(tmp_path, ["a", "b"])
    try:
        db.mark_favorite("a")
        db.mark_dislike("b")
        db.record_feedback("a", "like", "manual")
        album_id = db.create_album("Keepers")
        db.add_album_member(album_id, "a")
        db.set_album_cover(album_id, "a")

        # Rebuild with the same points — nothing is an orphan.
        db.qdrant_client = FakeQdrant([
            [_point_full("a", "/photos/a.jpg"), _point_full("b", "/photos/b.jpg")],
        ])
        counts = db.init_from_qdrant(force=True)
        assert counts == 2

        assert db.count_favorites() == 1
        with db._lock:
            dis = db._conn.execute("SELECT COUNT(*) AS n FROM dislikes").fetchone()["n"]
            fb = db._conn.execute("SELECT COUNT(*) AS n FROM feedback_events").fetchone()["n"]
        assert dis == 1
        # feedback_events is a history log — never purged.
        assert fb == 1
        assert db.list_album_member_ids(album_id) == ["a"]
        assert db.get_album(album_id)["cover_favorite_id"] == "a"
    finally:
        db.close()


def test_purge_skipped_when_rebuild_yields_zero_rows(tmp_path):
    """NAS-dropout guard: an empty rebuild is far more likely to mean
    'Qdrant unreachable / collection mid-rebuild' than 'every photo
    was deleted'. Purging on an empty rebuild would wipe the entire
    taste history irrecoverably, so orphans survive until a
    non-empty rebuild confirms they're really gone.
    """
    db = _make_db(tmp_path, ["a"])
    try:
        db.mark_favorite("a")
        album_id = db.create_album("Survivor")
        db.add_album_member(album_id, "a")

        db.qdrant_client = FakeQdrant([])  # empty scroll
        db.init_from_qdrant(force=True)

        with db._lock:
            fav_rows = db._conn.execute("SELECT id FROM favorites").fetchall()
            mem_rows = db._conn.execute(
                "SELECT favorite_id FROM album_memberships"
            ).fetchall()
        assert [r["id"] for r in fav_rows] == ["a"]
        assert [r["favorite_id"] for r in mem_rows] == ["a"]
    finally:
        db.close()


def test_purge_returns_per_table_counts(tmp_path):
    db = _make_db(tmp_path, ["a", "b"])
    try:
        db.mark_favorite("b")
        db.mark_dislike("b")
        album_id = db.create_album("Counts")
        db.add_album_member(album_id, "b")
        db.set_album_cover(album_id, "b")

        db.qdrant_client = FakeQdrant([[_point_full("a", "/photos/a.jpg")]])
        db.init_from_qdrant(force=True)  # triggers the purge

        counts = db.purge_orphaned_user_data()  # idempotent second run
        assert counts == {
            "favorites": 0,
            "dislikes": 0,
            "album_memberships": 0,
            "album_covers_reset": 0,
        }
    finally:
        db.close()


def test_purge_uses_qdrant_not_the_mirror_for_liveness(tmp_path):
    """A stale/empty `images` mirror must NOT delete user state.

    This is the safety property the purge inversion exists for. The
    `images` table is a DERIVED mirror — it can be empty, stale, or
    half-built. If it were used to decide which favourites are
    orphans, a stale mirror would make every favourite look
    orphaned and the purge would delete the user's authoritative
    state.

    So: the mirror is emptied, but Qdrant still holds the photos.
    The purge must keep every row.
    """
    db = _make_db(tmp_path, ["a", "b"])
    try:
        db.mark_favorite("a")
        db.mark_favorite("b")
        db.mark_dislike("a")
        album_id = db.create_album("Test album")
        db.add_album_member(album_id, "a")
        db.set_album_cover(album_id, "a")

        # The mirror is wiped — but Qdrant still holds a and b.
        with db._lock:
            db._conn.execute("DELETE FROM images")
            db._conn.commit()

        db.purge_orphaned_user_data()

        # Every user row survived, because Qdrant still holds the
        # photos they point at.
        with db._lock:
            favs = db._conn.execute("SELECT id FROM favorites").fetchall()
            dis = db._conn.execute("SELECT id FROM dislikes").fetchall()
        assert {r["id"] for r in favs} == {"a", "b"}
        assert {r["id"] for r in dis} == {"a"}
        assert db.list_album_member_ids(album_id) == ["a"]
        assert db.get_album(album_id)["cover_favorite_id"] == "a"
    finally:
        db.close()


def test_purge_keeps_everything_when_the_liveness_check_fails(tmp_path):
    """A failed liveness check must never become a mass deletion.

    If Qdrant cannot be reached, the answer to "does this photo
    still exist?" is unknown — and the only safe response to
    "unknown" is to keep the user's state. Deleting it would turn a
    transient outage into permanent data loss.
    """
    db = _make_db(tmp_path, ["a", "b"])
    try:
        db.mark_favorite("a")
        db.mark_favorite("b")

        class _BrokenQdrant:
            def retrieve_batch(self, point_ids):
                raise RuntimeError("qdrant is down")

        db.qdrant_client = _BrokenQdrant()
        db.purge_orphaned_user_data()

        with db._lock:
            favs = db._conn.execute("SELECT id FROM favorites").fetchall()
        assert {r["id"] for r in favs} == {"a", "b"}
    finally:
        db.close()
