"""
search/index_db.py

SQLite-backed search-side index cache and user-state store.

Qdrant remains the source of truth for indexed photos and embeddings.
This database keeps a rebuildable cache of photo metadata plus the
non-rebuildable per-photo user state owned by the search side.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from search.random_session import CursorDeck, random_start_uuid

if TYPE_CHECKING:
    from search.qdrant_client import QdrantSearch

logger = logging.getLogger(__name__)

# How long a corpus-size count is reused. The settings page polls the
# index status every second; without this the poll asks Qdrant to
# count every point in the collection on every request.
_COUNT_CACHE_TTL_S = 10.0

DEFAULT_INDEX_DB_PATH = "./data/images.db"


class ImageNotInCacheError(KeyError):
    """Raised when user state is requested for a photo missing from the cache."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexDB:
    """
    SQLite cache for Qdrant photo metadata and per-photo user state.

    The class is intentionally synchronous. FastAPI routes should call
    it through `asyncio.to_thread(...)` so SQLite and Qdrant scroll work
    do not block the event loop.
    """

    def __init__(
        self,
        db_path: str,
        qdrant_client: QdrantSearch,
        refresh_interval_seconds: int = 21600,
        photos_root: str = "",
        path_prefix: str = "",
        photos_roots: tuple[str, ...] = (),
    ):
        self.db_path = db_path
        self.qdrant_client = qdrant_client
        self.refresh_interval_seconds = refresh_interval_seconds
        # Where the photo files live ON THIS MACHINE, and the prefix
        # the indexer used on ITS machine. Together they map a stored
        # payload path to a local file, which is what the orphan purge
        # needs to answer "is this photo still on disk?".
        self.photos_root = photos_root
        # Every root a stored path may resolve under. `photos_root` is
        # the primary one (it is also the base for relative paths and
        # for the prefix rewrite); the extra roots are the other photo
        # MOUNTS — a source mounted as a sibling of the photos root
        # (/gravure next to /nas) keeps its files under a different
        # top-level path, and without its root listed here the purge
        # could never confirm that one of its photos is gone, so the
        # favourite would never be garbage collected.
        self.photos_roots = (
            tuple(photos_roots)
            if photos_roots
            else ((photos_root,) if photos_root else ())
        )
        self.path_prefix = path_prefix
        self._lock = threading.RLock()
        # The periodic maintenance clock starts at "now": there is
        # no derived cache to build at startup, and the corpus in
        # Qdrant is always current.
        self._last_refresh = time.time()
        # The corpus's point ids, cached per collection filter. A
        # full Qdrant id scroll is ~25s on the 2.2M corpus, so the
        # scroll is done once and the result reused; only the
        # shuffle is per session.
        self._count_cache: int | None = None
        self._count_cache_at: float = 0.0
        # Serialises cache BUILDS (the scroll), so a request and the
        # background warmer do not both pay for it.
        # Refresh lock — non-reentrant guard against the periodic
        # background task and the manual /api/cache/refresh endpoint
        # racing. Acquired before the maintenance runs, released
        # after. NOT the same as `_lock` (which guards DB operations);
        # this one spans the whole Qdrant scroll + rebuild and
        # prevents two scrolls from running at once.
        self._refresh_lock = threading.Lock()

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # Round-35: enable WAL mode so the indexer subprocess can write
        # to images.db concurrently (WAL allows multiple readers + one
        # writer; DELETE mode would block the search app on every
        # indexer INSERT). The change is persistent — the journal mode
        # is stored in the database file header and survives reconnects.
        # Skip for :memory: because it has no journal.
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL is the WAL sweet spot — fsyncs on commit but not on
            # read, which is fine because reads come from the WAL itself
            # and the indexer writes in transaction-sized batches.
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._backfill_photo_paths()

    def _backfill_photo_paths(self) -> None:
        """Fill in `photo_paths` for the rows that predate the table.

        The `photo_paths` table was added so the orphan purge could
        decide liveness against the FILESYSTEM rather than against
        Qdrant. Rows written before it existed have no path, and
        without one the purge has to fall back to the Qdrant-liveness
        rule — which is wrong for a photo that is on disk but not
        currently indexed (right now, 80,787 of them).

        So on open, the existing user-state rows get their paths
        resolved from Qdrant and recorded.

        A row whose point is no longer in Qdrant gets NO path. That
        is not a failure — it means the photo is already gone from
        the collection, so there is nothing to look up, and the row
        is left for the purge to deal with.

        This is self-healing rather than a one-shot migration: it
        looks for user-state rows with no recorded path, and only
        does a Qdrant lookup if there are any. Once the paths are
        recorded there is nothing left to look up, so the steady
        state costs a few small SELECTs and no network calls. A
        one-shot flag would be wrong here — the first open of a
        fresh database happens BEFORE any rows exist, so the flag
        would be set on an empty table and the real backfill would
        never run.
        """
        with self._lock:
            candidate_ids: set[str] = set()
            for table, column in (
                ("favorites", "id"),
                ("dislikes", "id"),
                ("album_memberships", "favorite_id"),
                ("albums", "cover_favorite_id"),
            ):
                rows = self._conn.execute(
                    f"SELECT DISTINCT {column} AS v FROM {table} "  # noqa: S608
                    f"WHERE {column} IS NOT NULL AND {column} != ''"
                ).fetchall()
                candidate_ids.update(r["v"] for r in rows)

            # Only the rows that still need a path.
            pending = [
                pid for pid in sorted(candidate_ids)
                if self._conn.execute(
                    "SELECT 1 FROM photo_paths WHERE id = ? LIMIT 1",
                    (pid,),
                ).fetchone() is None
            ]

        if not pending:
            return

        # Resolve the paths from Qdrant. `retrieve_batch` is one
        # round-trip per chunk and returns the points the collection
        # still holds.
        resolved: dict[str, str] = {}
        chunk_size = 500
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            try:
                hits = self.qdrant_client.retrieve_batch(chunk)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "photo_paths backfill: lookup failed for %d ids (%s) "
                    "— leaving them unresolved", len(chunk), e,
                )
                continue
            for hit in hits:
                path = (hit.payload or {}).get("path") or hit.path
                if path:
                    resolved[hit.id] = path

        if resolved:
            with self._lock:
                self._conn.executemany(
                    """
                    INSERT INTO photo_paths (id, path, recorded_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      path        = excluded.path,
                      recorded_at = excluded.recorded_at
                    """,
                    [(pid, path, _utc_now())
                     for pid, path in resolved.items()],
                )
                self._conn.commit()
            logger.info(
                "photo_paths backfill: recorded %d path(s) for %d "
                "existing user-state row(s)",
                len(resolved), len(pending),
            )
        unresolved = len(pending) - len(resolved)
        if unresolved:
            logger.info(
                "photo_paths backfill: %d row(s) have no path — their "
                "photo is not in the collection", unresolved,
            )

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                -- Persistent user state. This database holds ONLY
                -- authoritative user state — favourites, dislikes,
                -- albums, memberships, saved searches, feedback.
                --
                -- The photo corpus is NOT mirrored here. Qdrant owns
                -- it: the photos, their metadata, and the native
                -- text index the filename filter queries. A derived
                -- mirror was a second copy of the same facts that
                -- could disagree with the collection — and, worse, it
                -- could be used to decide that a favourite's photo
                -- no longer exists, which would delete the user's
                -- state over a cache that was merely stale.
                --
                -- Liveness is therefore always decided against
                -- Qdrant — see purge_orphaned_user_data().
                --
                -- ...with ONE exception, and it is the important one.
                -- "Is this photo still around?" has two possible
                -- meanings:
                --
                --   1. Is it in the collection right now?
                --   2. Does the file still exist on disk?
                --
                -- These differ, and conflating them loses user data.
                -- A photo can be absent from Qdrant while its file is
                -- perfectly alive on the NAS — because it was never
                -- indexed, or because the collection was restored from
                -- a snapshot older than the photo's last index run.
                -- In that state the favourite is still VALID; the
                -- photo is coming back, and because the point id is
                -- uuid5(path) it comes back as the SAME id, so the
                -- favourite re-attaches on its own.
                --
                -- Deleting the favourite on "absent from Qdrant"
                -- would destroy it for a photo that is about to
                -- return. So the path is recorded alongside the id,
                -- and liveness is decided against the FILESYSTEM.
                --
                -- This table is the id -> path map. It is written when
                -- a photo is favourited / disliked / added to an
                -- album, at the moment the path is known, and it is
                -- the only place a path can be recovered from: the
                -- point id is a one-way uuid5 hash, so it cannot be
                -- reversed.
                CREATE TABLE IF NOT EXISTS photo_paths (
                  id            TEXT PRIMARY KEY,
                  path          TEXT NOT NULL,
                  recorded_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS favorites (
                  id            TEXT PRIMARY KEY,
                  favorited_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_favorites_favorited_at
                  ON favorites(favorited_at DESC);

                -- Persistent "do not recommend" feedback. Independent
                -- of the `favorites` table so king can ✕ photos that
                -- they never liked. Mirrors favorites structurally so
                -- the for-you ranking service reads them identically.
                CREATE TABLE IF NOT EXISTS dislikes (
                  id            TEXT PRIMARY KEY,
                  disliked_at   TEXT NOT NULL,
                  source        TEXT NOT NULL DEFAULT 'manual'
                );

                CREATE INDEX IF NOT EXISTS idx_dislikes_disliked_at
                  ON dislikes(disliked_at DESC);

                -- Append-only feedback event log. Records every like
                -- and dislike (including cross-page toggles of an
                -- existing favourite) with the page that produced it.
                -- Used by the for-you ranking to (1) compute "trained
                -- N seconds ago" copy and (2) infer dwell-time style
                -- signals in a later phase. Source values:
                --   'grid'     — photo-card ♥ / ✕ on a result grid
                --   'lightbox' — fullscreen viewer
                --   'for_you'  — /for-you feed itself
                --   'detail'   — /photo/{id} page
                CREATE TABLE IF NOT EXISTS feedback_events (
                  id            INTEGER PRIMARY KEY AUTOINCREMENT,
                  photo_id      TEXT NOT NULL,
                  kind          TEXT NOT NULL CHECK (kind IN ('like','dislike')),
                  at            TEXT NOT NULL,
                  source        TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_events_at
                  ON feedback_events(at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_events_photo
                  ON feedback_events(photo_id);

                -- User-created albums. A photo can be in zero, one,
                -- or many albums — album membership is independent
                -- of favourites status, so a photo can be in album
                -- X without being in the favourites table (default
                -- album) and vice versa. `name` is unique so the
                -- user can't create two albums with the same label.
                CREATE TABLE IF NOT EXISTS albums (
                  id                INTEGER PRIMARY KEY AUTOINCREMENT,
                  name              TEXT NOT NULL UNIQUE,
                  description       TEXT DEFAULT '',
                  cover_favorite_id TEXT DEFAULT '',
                  created_at        TEXT NOT NULL,
                  updated_at        TEXT NOT NULL
                );

                -- Many-to-many membership: same photo can be in
                -- multiple albums. ON DELETE CASCADE keeps the
                -- membership table clean when an album is deleted
                -- (FK integrity is local-only — SQLite enforces it
                -- even without explicit PRAGMA foreign_keys=ON
                -- because we always go through IndexDB methods).
                CREATE TABLE IF NOT EXISTS album_memberships (
                  album_id     INTEGER NOT NULL
                                REFERENCES albums(id) ON DELETE CASCADE,
                  favorite_id  TEXT NOT NULL,
                  added_at     TEXT NOT NULL,
                  PRIMARY KEY (album_id, favorite_id)
                );

                CREATE INDEX IF NOT EXISTS idx_album_memberships_favorite
                  ON album_memberships(favorite_id);

                -- Named prompt presets. A saved search captures only
                -- the prompt shape (positive / negative text lists) so
                -- the user can recall "that +red-dress −manikin" combo
                -- and re-run it without re-typing. Centroid, view,
                -- favourites-filter and limits are intentionally NOT
                -- stored — those are session state, not part of the
                -- preset. `name` is UNIQUE so the UI can address a
                -- saved search by name without ambiguity.
                CREATE TABLE IF NOT EXISTS saved_searches (
                  id          INTEGER PRIMARY KEY AUTOINCREMENT,
                  name        TEXT NOT NULL UNIQUE,
                  positives   TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
                  negatives   TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
                  created_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_saved_searches_created_at
                  ON saved_searches(created_at DESC);

                -- One-shot migration flags. Keyed by an
                -- identifier (e.g. 'fts_v1'); value is the
                -- timestamp the migration completed. Lets
                -- additive migrations run exactly once per DB
                -- instead of every open. Idempotent because
                -- callers use INSERT OR IGNORE.
                CREATE TABLE IF NOT EXISTS schema_meta (
                  key   TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                """
            )
            self._conn.commit()


    def _ensure_column(self, table: str, column: str, type_sql: str) -> None:
        """Add a column to `table` if it doesn't already exist.

        Used by `_init_schema` for additive migrations. Idempotent:
        re-opening a DB that already has the column is a no-op. SQLite
        doesn't support `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on
        the Python version floor this project targets, so we read
        `PRAGMA table_info` first.
        """
        with self._lock:
            cols = {
                row["name"]
                for row in self._conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            if column not in cols:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {type_sql}"
                )
                self._conn.commit()

    def _payloads_by_id(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Photo metadata for `point_ids`, straight from Qdrant.

        The photo's metadata — path, shard, collection, mtime, size,
        indexed_at, width, height, blurhash — belongs to the photo,
        so Qdrant is the source of truth for it. The `images` table
        was a derived mirror of exactly this, and keeping both meant
        two copies of the same facts that could disagree.

        Returns a map of id → metadata for the ids Qdrant holds.
        Ids it does not hold are simply absent — the caller decides
        what an absent id means (an orphan, usually).
        """
        if not point_ids:
            return {}
        try:
            hits = self.qdrant_client.retrieve_batch(list(point_ids))
        except Exception as e:  # noqa: BLE001
            logger.warning("payload fetch failed for %d ids: %s", len(point_ids), e)
            return {}
        out: dict[str, dict[str, Any]] = {}
        for hit in hits:
            payload = hit.payload or {}
            # Keyed by the Qdrant POINT id, because that is what the
            # caller asked for: the random deck is built from point
            # ids and joined back on them. The payload's own `id`
            # field is carried inside the row, so the API can surface
            # the app-level id.
            out[hit.id] = {
                "id": payload.get("id", hit.id),
                "path": payload.get("path", hit.path or ""),
                "shard": payload.get("shard", ""),
                "collection": payload.get("collection", ""),
                "mtime": payload.get("mtime", 0),
                "size": payload.get("size", 0),
                "indexed_at": payload.get("indexed_at", ""),
                "width": payload.get("width", 0),
                "height": payload.get("height", 0),
                "blurhash": payload.get("blurhash", ""),
            }
        return out

    def _live_point_ids(self, point_ids: list[str]) -> set[str]:
        """Which of `point_ids` does Qdrant still hold?

        This is the authoritative liveness check — the collection is
        the source of truth for whether a photo exists. The `images`
        table is a derived mirror and must never be used for this:
        a stale or empty mirror would make every favourite look
        orphaned, and the purge would delete the user's authoritative
        state.

        Qdrant's `retrieve` is a single round-trip that returns the
        points it holds and omits the ones it does not, so one call
        answers the whole batch. The ids are chunked so a very large
        candidate set cannot exceed the request-size limit.

        If the check itself fails (Qdrant unreachable), the SAFE
        answer is "everything is live" — keep all user state. A
        failed liveness check must never become a mass deletion.
        """
        if not point_ids:
            return set()
        chunk_size = 500
        live: set[str] = set()
        for start in range(0, len(point_ids), chunk_size):
            chunk = point_ids[start:start + chunk_size]
            try:
                hits = self.qdrant_client.retrieve_batch(chunk)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "liveness check failed for %d ids (%s) — keeping all "
                    "user state (a failed check must never delete anything)",
                    len(chunk), e,
                )
                return set(point_ids)
            live.update(hit.id for hit in hits)
        return live

    def _file_still_exists(self, stored_path: str) -> bool:
        """True if the photo's file is still on disk.

        The stored path is the one the INDEXER recorded, which may be
        from a different machine's mount (Z:\\images\\... on Windows).
        This maps it onto this machine's mount and checks for the
        file.

        Three outcomes, and they are not the same thing:

        - The path maps onto our mount and the file is there -> alive.
        - The path maps onto our mount and the file is NOT there ->
          the photo is genuinely gone. This is the orphan case.
        - The path does NOT map onto our mount -> we cannot check.
          Treated as ALIVE.

        That last case matters. A payload path from a machine whose
        mount we do not have (or a legacy path outside the photos
        root) is not evidence that the photo was deleted — it is
        evidence that we are looking in the wrong place. Treating
        "cannot check" as "gone" would delete the user's favourites
        on any host that does not mount the photos: the exact
        mistake the Qdrant-liveness rule was introduced to avoid,
        just with a different trigger.
        """
        if not stored_path:
            return True
        if not self.photos_root:
            return True

        local = self._map_to_local(stored_path)
        if local is None:
            return True
        try:
            return local.exists()
        except OSError:
            # An unreadable path is not a deleted photo.
            logger.warning("purge: could not stat %r; treating as alive", local)
            return True

    def _map_to_local(self, stored_path: str) -> Path | None:
        """Map a stored payload path onto this machine's photos mount.

        Returns the local path, or None when the stored path cannot
        be mapped — meaning we have no way to check whether the file
        is there.

        The mapping mirrors `resolve_local`, but WITHOUT its
        existence check: `resolve_local` returns None both for "the
        file is gone" and for "the path does not map", and the two
        need different answers here.
        """
        p = Path(stored_path)
        prefix = self.path_prefix

        if prefix and str(p).startswith(prefix):
            rel = str(p)[len(prefix):].lstrip("\\/").replace("\\", "/")
            return Path(self.photos_root) / rel

        if p.is_absolute():
            # An absolute path is only checkable if it lands inside
            # one of the photo roots we have mounted. A path outside
            # them belongs to a namespace we cannot see.
            root = self._root_for(p)
            if root is None:
                return None
            # The root itself has to be reachable. A mount that is
            # down (or a root that was renamed away) makes every file
            # under it look deleted, and "the file is gone" is a
            # destructive conclusion — it takes the user's favourites
            # with it. An unreachable root means we cannot check, and
            # "cannot check" means alive.
            if not self._root_is_reachable(root):
                logger.warning(
                    "purge: photo root %r is not reachable; treating "
                    "paths under it as alive (cannot check)",
                    root,
                )
                return None
            return p

        # Relative: it is relative to the photos root by definition.
        return Path(self.photos_root) / p

    def _root_for(self, p: Path) -> str | None:
        """The photo root an absolute path falls under, if any.

        The LONGEST matching root wins, so a path under a source dir
        is attributed to that source dir rather than to the mount it
        happens to sit inside. The source dir is the narrower
        statement of where the file lives, and it is the one that
        goes away if that source is unmounted.
        """
        best: str | None = None
        for root in self.photos_roots:
            try:
                p.relative_to(root)
            except ValueError:
                continue
            if best is None or len(root) > len(best):
                best = root
        return best

    @staticmethod
    def _root_is_reachable(root: str) -> bool:
        """True if the root is present and is a directory.

        `Path.is_dir()` swallows OSError and answers False, which is
        the answer we want: a root we cannot stat is a root we cannot
        check under.
        """
        try:
            return Path(root).is_dir()
        except OSError:
            return False

    def find_orphaned_user_ids(self) -> list[str]:
        """The user-state ids whose photo is gone from DISK.

        Analysis only — nothing is deleted. This is the decision
        half of the purge; `purge_orphaned_user_data` applies it.
        `count_orphaned_user_data` reports on it without applying
        it, so the UI can show what a purge would remove before the
        operator asks for one.

        Orphans accumulate when the indexer prunes Qdrant points
        (files deleted from disk): the favourite/dislike/album rows
        were deliberately kept so re-indexing the same path (same
        uuid5 id) would re-attach them. That re-attach window has a
        cost though — orphan ids poison qdrant.recommend() (a single
        missing id 404s the WHOLE call; see filter_live_ids in
        search/for_you.py) and orphan album members render as dead
        thumbnails.

        Called from POST /api/cache/refresh.

        **Liveness is decided by the FILESYSTEM, and by nothing
        else.**

        The obvious rule — "delete the row if Qdrant no longer holds
        the point" — is wrong, and dangerously so. A photo can be
        absent from Qdrant while its file is perfectly alive on the
        NAS: it was never indexed, or the collection was restored from
        a snapshot older than its last index run. Right now 80,787
        photos are in exactly that state. Deleting their favourites
        would destroy real user state for photos that are about to
        come back — and because the point id is uuid5(path), they
        WOULD come back as the same id, re-attaching to a favourite
        that is no longer there.

        So the rule is: a candidate row is an orphan only when its
        FILE is gone from disk. The path comes from `photo_paths`,
        recorded when the photo was favourited / disliked / added to
        an album. The point id is a one-way hash and cannot be
        reversed, so this table is the only way back to the path.

        For a candidate with NO recorded path (written before this
        table existed), the Qdrant check remains the only signal
        available, so it is used as a fallback. That is the old
        behaviour, applied to the small residue it applies to.

        Purges:
          - favorites / dislikes rows whose file is gone from disk
          - album_memberships whose photo's file is gone (a
            membership of a deleted photo renders as a dead
            thumbnail; note memberships do NOT require the photo
            to be a favourite — add_album_member allows any id —
            so liveness is checked against the file alone)
          - albums.cover_favorite_id refs pointing at photos whose
            file is gone (reset to '' so the UI falls back to the
            first member)

        Keeps: feedback_events (append-only history log, not state —
        nothing feeds it back into recommend) and saved_searches
        (prompt text only, no point ids).

        Returns per-table deletion counts for logging/API response.
        """
        # Every id the user state refers to. This is a small set —
        # thousands of rows, not millions — so each one can be
        # checked individually.
        with self._lock:
            candidate_ids: set[str] = set()
            for table, column in (
                ("favorites", "id"),
                ("dislikes", "id"),
                ("album_memberships", "favorite_id"),
                ("albums", "cover_favorite_id"),
            ):
                rows = self._conn.execute(
                    f"SELECT DISTINCT {column} AS v FROM {table} "  # noqa: S608
                    f"WHERE {column} IS NOT NULL AND {column} != ''"
                ).fetchall()
                candidate_ids.update(r["v"] for r in rows)

            if not candidate_ids:
                return []

            recorded_paths = {
                r["id"]: r["path"]
                for r in self._conn.execute("SELECT id, path FROM photo_paths").fetchall()
            }

        # Which of those ids are genuinely gone?
        #
        # The rule: a candidate is an orphan when its FILE is gone
        # from disk. The path comes from photo_paths (recorded when
        # the photo was favourited). For candidates with no recorded
        # path — written before photo_paths existed — fall back to the
        # Qdrant liveness check, which is the only signal available.
        #
        # The filesystem check needs the payload path resolved to a
        # LOCAL path: the indexer may have stored a path from another
        # machine's mount (Z:\images\... on Windows), and the search
        # app sees it at a different mount point. `resolve_local`
        # does that mapping.

        # ids whose file is confirmed gone from disk
        disk_gone: set[str] = set()
        # ids with no recorded path -> judged by Qdrant liveness
        no_path: set[str] = set()
        for pid in candidate_ids:
            path = recorded_paths.get(pid)
            if not path:
                no_path.add(pid)
                continue
            if not self._file_still_exists(path):
                disk_gone.add(pid)

        # The Qdrant fallback only applies to the rows with no
        # recorded path. It needs a non-empty collection to be
        # meaningful: a collection reporting ZERO points is far more
        # likely to mean "the collection was just recreated" or "we
        # are looking at the wrong collection" than "every photo was
        # deleted", and on that signal EVERY pathless row would look
        # dead. The file-based rule does not need this guard — it
        # consults the filesystem, not the collection, so an empty
        # collection tells it nothing.
        orphan_set = set(disk_gone)
        if no_path:
            try:
                total_points = int(self.qdrant_client.count_points())
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "purge: count_points failed, keeping pathless user data: %s", e,
                )
                total_points = -1
            if total_points == 0:
                logger.info(
                    "purge: collection %r is empty — not evidence that a "
                    "pathless photo is gone; keeping those rows",
                    getattr(self.qdrant_client, "collection", "?"),
                )
            elif total_points > 0:
                live_ids = self._live_point_ids(sorted(no_path))
                orphan_set.update(pid for pid in no_path if pid not in live_ids)

        # SQLite binds parameters from a sequence, not a set.
        return sorted(orphan_set)

    def count_orphaned_user_data(self) -> dict[str, int]:
        """How many user-state rows a purge would remove.

        Reports what `purge_orphaned_user_data` would delete without
        deleting any of it, so the operator can see the scope before
        asking for the purge.

        Returns the same per-table keys the purge reports, so the
        UI can show "12 favourites, 3 album entries" without the
        operator having to run it to find out.
        """
        orphans = self.find_orphaned_user_ids()
        if not orphans:
            return {
                "favorites": 0,
                "dislikes": 0,
                "album_memberships": 0,
                "album_covers_reset": 0,
            }
        with self._lock:
            placeholders = ",".join("?" * len(orphans))
            counts = {
                "favorites": self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM favorites "  # noqa: S608
                    f"WHERE id IN ({placeholders})",
                    orphans,
                ).fetchone()["c"],
                "dislikes": self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM dislikes "  # noqa: S608
                    f"WHERE id IN ({placeholders})",
                    orphans,
                ).fetchone()["c"],
                "album_memberships": self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM album_memberships "  # noqa: S608
                    f"WHERE favorite_id IN ({placeholders})",
                    orphans,
                ).fetchone()["c"],
                "album_covers_reset": self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM albums "  # noqa: S608
                    f"WHERE cover_favorite_id IN ({placeholders})",
                    orphans,
                ).fetchone()["c"],
            }
        return counts

    def purge_orphaned_user_data(self) -> dict[str, int]:
        """Delete user-state rows whose photo is gone from DISK.

        The delete half of `find_orphaned_user_ids`. See that method
        for why liveness is decided against the filesystem rather
        than against Qdrant.

        Purges:
          - favorites / dislikes rows whose file is gone from disk
          - album_memberships whose photo's file is gone (a
            membership of a deleted photo renders as a dead
            thumbnail; note memberships do NOT require the photo
            to be a favourite — add_album_member allows any id —
            so liveness is checked against the file alone)
          - albums.cover_favorite_id refs pointing at photos whose
            file is gone (reset to '' so the UI falls back to the
            first member)

        Keeps: feedback_events (append-only history log, not state —
        nothing feeds it back into recommend) and saved_searches
        (prompt text only, no point ids).

        Returns per-table deletion counts for logging/API response.
        """
        orphans = self.find_orphaned_user_ids()
        if not orphans:
            return {
                "favorites": 0,
                "dislikes": 0,
                "album_memberships": 0,
                "album_covers_reset": 0,
            }
        with self._lock:
            placeholders = ",".join("?" * len(orphans))
            cur_memberships = self._conn.execute(
                f"""
                DELETE FROM album_memberships
                WHERE favorite_id IN ({placeholders})
                """,  # noqa: S608
                orphans,
            )
            cur_favs = self._conn.execute(
                f"DELETE FROM favorites WHERE id IN ({placeholders})",  # noqa: S608
                orphans,
            )
            cur_dis = self._conn.execute(
                f"DELETE FROM dislikes WHERE id IN ({placeholders})",  # noqa: S608
                orphans,
            )
            cur_covers = self._conn.execute(
                f"""
                UPDATE albums SET cover_favorite_id = ''
                WHERE cover_favorite_id IN ({placeholders})
                """,  # noqa: S608
                orphans,
            )
            self._conn.commit()
            purged = {
                "favorites": cur_favs.rowcount,
                "dislikes": cur_dis.rowcount,
                "album_memberships": cur_memberships.rowcount,
                "album_covers_reset": cur_covers.rowcount,
            }
        if any(purged.values()):
            logger.info("purged orphaned user data: %s", purged)
        return purged

    def delete_user_state_for_ids(self, point_ids: list[str]) -> dict[str, int]:
        """Delete the user-state rows referring to the given point ids.

        This is the SQLite half of the confirmed orphan purge. The
        Qdrant half removes the embeddings; this removes the
        favourites, dislikes, album memberships and album covers that
        pointed at them.

        Called AFTER the Qdrant delete, and only for the ids the
        operator confirmed. The `photo_paths` rows go too — the
        photo is gone, so its path record is dead weight and would
        otherwise grow forever.

        Returns per-table deletion counts.
        """
        if not point_ids:
            return {
                "favorites": 0,
                "dislikes": 0,
                "album_memberships": 0,
                "album_covers_reset": 0,
                "photo_paths": 0,
            }
        ids = sorted(set(point_ids))
        with self._lock:
            placeholders = ",".join("?" * len(ids))
            cur_memberships = self._conn.execute(
                f"""
                DELETE FROM album_memberships
                WHERE favorite_id IN ({placeholders})
                """,  # noqa: S608
                ids,
            )
            cur_favs = self._conn.execute(
                f"DELETE FROM favorites WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            )
            cur_dis = self._conn.execute(
                f"DELETE FROM dislikes WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            )
            cur_covers = self._conn.execute(
                f"""
                UPDATE albums SET cover_favorite_id = ''
                WHERE cover_favorite_id IN ({placeholders})
                """,  # noqa: S608
                ids,
            )
            cur_paths = self._conn.execute(
                f"DELETE FROM photo_paths WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            )
            self._conn.commit()
            purged = {
                "favorites": cur_favs.rowcount,
                "dislikes": cur_dis.rowcount,
                "album_memberships": cur_memberships.rowcount,
                "album_covers_reset": cur_covers.rowcount,
                "photo_paths": cur_paths.rowcount,
            }
        if any(purged.values()):
            logger.info("purged user state for %d confirmed orphan(s): %s",
                        len(ids), purged)
        return purged

    def new_cursor_deck(
        self, collections: tuple[str, ...] | list[str] = ()
    ) -> Any:
        """A fresh /random deck: a cursor into the corpus's id space.

        The deck is read on demand, one page at a time. Qdrant's
        REST scroll takes its `offset` as a point-id cursor and
        accepts ANY uuid — including one that does not exist —
        starting from the first point whose id sorts after it. So a
        session can begin anywhere in the id space with a bare
        `uuid4()`, and read forward from there.

        This replaces a full id scroll: 25s over 2.2M points, 200 MB
        of cached uuid strings, and a scroll's worth of page-cache
        traffic evicting Qdrant's hot collection from the cache.
        """
        clean = tuple(c for c in (collections or ()) if c)

        def fetch(cursor: Any, limit: int) -> tuple[list[str], Any]:
            return self.qdrant_client.scroll_point_ids(
                cursor, limit, list(clean) or None
            )

        # How many ids the walk will produce. The facet counts the
        # points per collection in 0.009s; an exact count with the
        # same filter takes 0.540s.
        total = self.qdrant_client.collection_counts(list(clean) or None)
        return CursorDeck(fetch, start=random_start_uuid(), total=total)

    def rows_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Look up full row dicts for a list of point ids.

        Returns one row per id that exists in the collection. Order
        of the returned rows is arbitrary — callers that need a
        specific order (e.g. /random session-cursor) should reorder
        themselves. Missing ids are silently dropped.

        The photo metadata comes from Qdrant, which owns it; the
        favourite/dislike flags come from SQLite, which owns them.
        That split is the point: neither store decides whether the
        other one's data is real.
        """
        if not ids:
            return []
        ids = [str(i) for i in ids if i]
        if not ids:
            return []
        payload_by_id = self._payloads_by_id(ids)
        if not payload_by_id:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT f.id AS fav_id, f.favorited_at AS favorited_at,
                       d.id AS dis_id
                FROM favorites f
                LEFT JOIN dislikes d ON d.id = f.id
                WHERE f.id IN ({placeholders})
                """,  # noqa: S608
                ids,
            ).fetchall()
            favs = {r["fav_id"]: r["favorited_at"] for r in rows if r["fav_id"]}
            dis_rows = self._conn.execute(
                f"SELECT id FROM dislikes WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            ).fetchall()
        disliked = {r["id"] for r in dis_rows}

        out: list[dict[str, Any]] = []
        for point_id, payload in payload_by_id.items():
            app_id = payload.get("id") or point_id
            out.append({
                **payload,
                "id": app_id,
                "point_id": point_id,
                "is_favorite": app_id in favs,
                "favorited_at": favs.get(app_id),
                "is_disliked": app_id in disliked,
            })
        return out

    def favorite_id_set(self, point_ids: list[str]) -> set[str]:
        """Subset of `point_ids` that are currently in the favourites table.

        Used by /api/search to mark `is_favorite` on each result
        without paying the O(N favourites) cost of list_favorite_ids()
        when the caller only cares about a 20-tile page. Single
        IN-clause query — 1 SQLite round trip instead of N.
        """
        if not point_ids:
            return set()
        placeholders = ",".join("?" for _ in point_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM favorites WHERE id IN ({placeholders})",  # noqa: S608
                list(point_ids),
            ).fetchall()
        return {row["id"] for row in rows}

    def is_indexed(self, point_id: str) -> bool:
        return self.get_by_id(point_id) is not None

    def count_images(self) -> int:
        """How many photos the corpus holds.

        Qdrant owns the corpus, so it answers. The `images` mirror
        used to answer this, which meant the number could disagree
        with the collection the search actually runs against.

        The count is cached for a few seconds. This used to be a
        local SQLite read that the settings page could poll as often
        as it liked; now it is a live Qdrant count, and an exact one
        walks every point — 0.277s on the 2.2M-point collection. The
        settings page polls the index status every second, so
        without this cache it asks Qdrant to count 2.2M points once
        a second, exactly while the indexer is busiest.
        """
        now = time.monotonic()
        with self._lock:
            if (
                self._count_cache is not None
                and now - self._count_cache_at < _COUNT_CACHE_TTL_S
            ):
                return self._count_cache
        try:
            count = int(self.qdrant_client.count_points())
        except Exception as e:  # noqa: BLE001
            logger.warning("count_points failed: %s", e)
            return 0
        with self._lock:
            self._count_cache = count
            self._count_cache_at = now
        return count

    def record_photo_path(self, point_id: str, path: str) -> None:
        """Record the on-disk path for a point id.

        The path is the payload's `path` value — the location the
        INDEXER saw, which may be a different machine's mount (the
        search app resolves it through `resolve_local`).

        This is recorded the moment a photo is favourited, disliked or
        added to an album — the only moment the path is available
        without a Qdrant round trip. It is what lets the orphan purge
        tell "this photo is gone from disk" (delete the row) apart
        from "this photo is merely not indexed right now" (keep the
        row, it will re-attach).

        Idempotent. A re-record overwrites the path, which is correct
        if the photo was re-indexed from a different location.
        """
        if not point_id or not path:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO photo_paths (id, path, recorded_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  path        = excluded.path,
                  recorded_at = excluded.recorded_at
                """,
                (point_id, path, _utc_now()),
            )
            self._conn.commit()

    def get_photo_path(self, point_id: str) -> str | None:
        """The recorded on-disk path for a point id, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT path FROM photo_paths WHERE id = ? LIMIT 1",
                (point_id,),
            ).fetchone()
        return row["path"] if row else None

    def list_photo_paths(self) -> dict[str, str]:
        """Every recorded id -> path pair."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path FROM photo_paths"
            ).fetchall()
        return {r["id"]: r["path"] for r in rows}

    def mark_favorite(self, point_id: str) -> None:
        # Photo must exist in the cache before it can be favourited;
        # otherwise the UI would have no /photo/{id} page to link to
        # from /favorites. Favourites for orphan ids (photos removed
        # from Qdrant) are kept in the table but unreachable via UI.
        with self._lock:
            if not self._exists_locked(point_id):
                raise ImageNotInCacheError(point_id)
            self._conn.execute(
                """
                INSERT INTO favorites (id, favorited_at)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET favorited_at = excluded.favorited_at
                """,
                (point_id, _utc_now()),
            )
            self._conn.commit()
        # Record the path OUTSIDE the lock: it needs a Qdrant lookup,
        # and holding the DB lock across a network call would serialise
        # every writer behind it.
        self._record_path_from_qdrant(point_id)

    def _record_path_from_qdrant(self, point_id: str) -> None:
        """Look the point's path up in Qdrant and record it.

        Best-effort: a Qdrant failure leaves the path unrecorded,
        which makes the orphan purge fall back to the Qdrant-liveness
        rule for that id. Never raises.
        """
        try:
            hit = self.qdrant_client.retrieve(point_id)
        except Exception:  # noqa: BLE001 — path recording is best-effort
            return
        if hit is None:
            return
        path = (hit.payload or {}).get("path") or hit.path
        if path:
            self.record_photo_path(point_id, path)

    def unmark_favorite(self, point_id: str) -> None:
        # DELETE is idempotent and never raises on missing row, which
        # matches the previous behaviour where unmarking an unliked
        # photo was a no-op.
        with self._lock:
            self._conn.execute(
                "DELETE FROM favorites WHERE id = ?",
                (point_id,),
            )
            self._conn.commit()

    def list_favorite_ids(self) -> list[str]:
        """Return every favourite id in the cache, with no JOIN.

        Used by the dynamic favourites-centroid compute and the
        for-you recommend seed: every id the user has favourited,
        including ones whose photo is no longer in the `images`
        table (orphans — kept between purge runs).

        Callers that feed these ids to Qdrant must handle orphans
        themselves: `retrieve` silently drops unknown ids, but
        `recommend` 404s the WHOLE call if any positive/negative
        id is missing. The for-you route filters through
        `search.for_you.filter_live_ids` for exactly this reason.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM favorites ORDER BY favorited_at DESC"
            ).fetchall()
        return [str(r["id"]) for r in rows]

    def list_favorites(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """The favourites, newest first, with each photo's metadata.

        The favourite ROWS (id + when it was favourited) are
        authoritative user state and live in SQLite. The photo
        METADATA (path, shard, collection, size, dimensions,
        blurhash) belongs to the photo, so it comes from Qdrant,
        which owns it.

        A favourite whose photo Qdrant no longer holds is an orphan
        and is omitted — the same shape the old INNER JOIN produced,
        but decided by the collection rather than by a derived
        mirror. An orphan is not deleted here; the purge handles
        that, and it also checks Qdrant.
        """
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT f.id AS id, f.favorited_at AS favorited_at
                FROM favorites f
                ORDER BY f.favorited_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        if not rows:
            return []

        payload_by_id = self._payloads_by_id([r["id"] for r in rows])
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = payload_by_id.get(row["id"])
            if payload is None:
                # The photo is gone from the collection — an orphan.
                continue
            out.append({**payload, "id": row["id"], "favorited_at": row["favorited_at"]})
        return out

    def list_dislikes(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """The dislikes, newest first, with each photo's metadata.

        The dislike ROWS are authoritative user state and live in
        SQLite. The photo METADATA comes from Qdrant, which owns it.
        A dislike whose photo Qdrant no longer holds is an orphan and
        is omitted from the list — but the row is kept, so a
        re-indexed photo re-attaches the dislike.
        """
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT d.id AS id, d.disliked_at AS disliked_at, d.source AS source
                FROM dislikes d
                ORDER BY d.disliked_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        if not rows:
            return []

        payload_by_id = self._payloads_by_id([r["id"] for r in rows])
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = payload_by_id.get(row["id"])
            if payload is None:
                continue
            out.append({
                **payload,
                "id": row["id"],
                "disliked_at": row["disliked_at"],
                "source": row["source"],
            })
        return out

    def count_dislikes(self) -> int:
        """How many dislikes point at photos the collection still holds.

        A dislike whose photo was pruned is an orphan: it is not
        shown, and it is not counted. The liveness check is against
        Qdrant.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT id FROM dislikes WHERE id != ''"
            ).fetchall()
        if not rows:
            return 0
        live = self._live_point_ids([r["id"] for r in rows])
        return len(live)

    def count_favorites(self) -> int:
        """How many favourites point at photos the collection still holds.

        A favourite whose photo was pruned is an orphan: it is not
        shown, and it is not counted. The liveness check is against
        Qdrant — never against a derived mirror. The row itself
        stays in the table, so a re-indexed photo re-attaches it.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT id FROM favorites WHERE id != ''"
            ).fetchall()
        if not rows:
            return 0
        live = self._live_point_ids([r["id"] for r in rows])
        return len(live)

    # ----------------------------------------------------------------
    # Dislikes (persistent "do-not-recommend" feedback)
    # ----------------------------------------------------------------
    def mark_dislike(self, point_id: str, source: str = "manual") -> None:
        """Record a ✕ on a photo.

        Idempotent — re-pressing ✕ on an already-disliked photo just
        refreshes the timestamp, which matters because time decay on
        the for-you feed weights recent feedback more.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dislikes (id, disliked_at, source)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  disliked_at = excluded.disliked_at,
                  source      = excluded.source
                """,
                (point_id, _utc_now(), source),
            )
            self._conn.commit()
        # Record the path so the orphan purge can tell "gone from disk"
        # from "merely not indexed right now". See photo_paths.
        self._record_path_from_qdrant(point_id)

    def unmark_dislike(self, point_id: str) -> None:
        """Remove a ✕. Used by undo and by the /for-you Reset button."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM dislikes WHERE id = ?",
                (point_id,),
            )
            self._conn.commit()

    def is_disliked(self, point_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM dislikes WHERE id = ? LIMIT 1",
                (point_id,),
            ).fetchone()
        return row is not None

    def list_dislike_ids(self) -> list[str]:
        """All dislike ids, including orphans. Same shape as
        `list_favorite_ids` — consumed symmetrically by the for-you
        ranking service."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM dislikes ORDER BY disliked_at DESC"
            ).fetchall()
        return [row["id"] for row in rows]

    def dislike_id_set(self, ids: list[str]) -> set[str]:
        """Subset of `ids` that are currently in the dislikes table.

        Used by /api/search to mark is_disliked on each result without
        paying the O(N dislikes) cost of list_dislike_ids() when the
        caller only cares about a 20-tile page.
        """
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM dislikes WHERE id IN ({placeholders})",  # noqa: S608
                list(ids),
            ).fetchall()
        return {row["id"] for row in rows}


    # ----------------------------------------------------------------
    # Feedback event log (append-only; power for-you observability)
    # ----------------------------------------------------------------
    def record_feedback(
        self, photo_id: str, kind: str, source: str
    ) -> None:
        """Insert one row. `kind` ∈ {'like', 'dislike'}.

        Cross-page actions (e.g. lightbox ♥ also flips the per-page
        state) each emit one event so the timestamp reflects the most
        recent feedback on the photo.
        """
        assert kind in ("like", "dislike"), kind  # noqa: S101
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO feedback_events (photo_id, kind, at, source)
                VALUES (?, ?, ?, ?)
                """,
                (photo_id, kind, _utc_now(), source),
            )
            self._conn.commit()

    def most_recent_feedback(self) -> str | None:
        """ISO timestamp of the latest feedback event, or None. Used by
        the for-you header chip ("trained 4 min ago")."""
        with self._lock:
            row = self._conn.execute(
                "SELECT at FROM feedback_events ORDER BY at DESC LIMIT 1"
            ).fetchone()
        return row["at"] if row else None

    def feedback_counts(self) -> tuple[int, int]:
        """Return (n_likes, n_dislikes) across all feedback events.
        Not the same as len(favorites) — a favourite removed later
        leaves its `like` event in the log."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  SUM(CASE WHEN kind = 'like'    THEN 1 ELSE 0 END) AS likes,
                  SUM(CASE WHEN kind = 'dislike' THEN 1 ELSE 0 END) AS dislikes
                FROM feedback_events
                """
            ).fetchone()
        return (
            int(row["likes"] or 0) if row else 0,
            int(row["dislikes"] or 0) if row else 0,
        )

    def reset_feedback(self) -> None:
        """Wipe all dislikes + feedback events (NOT favourites).
        Used by the /for-you Reset button. Keeps the favorites table
        intact so a reset still leaves a centroid to recommend from."""
        with self._lock:
            self._conn.execute("DELETE FROM dislikes")
            self._conn.execute("DELETE FROM feedback_events")
            self._conn.commit()

    def reset_side_store(self) -> None:
        """Wipe every user-mutable table for a "fresh install" reset.

        Used by the admin Index rebuild mode BEFORE spawning the
        indexer subprocess, so user data referencing soon-to-be-
        replaced point IDs (favourites, dislikes, albums, saved
        searches, feedback events) is cleared first.

        Does NOT touch schema_meta — the schema version persists
        across resets so we don't accidentally trigger a migration
        on the next startup.
        """
        with self._lock:
            self._conn.executescript(
                """
                DELETE FROM favorites;
                DELETE FROM dislikes;
                DELETE FROM feedback_events;
                DELETE FROM album_memberships;
                DELETE FROM albums;
                DELETE FROM saved_searches;
                """
            )
            self._conn.commit()
            logger.info("reset_side_store: wiped all user-mutable tables")

    # ---------------------- Albums ----------------------
    #
    # User-curated collections of favourites. Membership is
    # independent of favourites status (a photo can be in an album
    # without being favourited, and vice versa), so the existing
    # `favorites` table is unaffected. The favourites table IS the
    # default album at the UI layer — there's no row for it in
    # `albums` because that would duplicate the source of truth.
    #
    # Album centroids are registered separately with the
    # DynamicCentroidRegistry under the name `album:{id}`. The
    # album endpoints below return `album_id`; the centroid layer
    # keys on that id so renames don't break references.

    def create_album(
        self, name: str, description: str = "",
    ) -> int:
        """Create a new album. Returns the new album id.

        Raises ValueError on duplicate name (UNIQUE constraint).
        Names are trimmed; empty / whitespace-only names raise
        ValueError so we don't end up with a nameless album in the
        registry that the UI can't address.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("album name is required")
        now = _utc_now()
        with self._lock:
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO albums (name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, description or "", now, now),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"album name already exists: {name!r}") from e
        if cur.lastrowid is None:
            raise RuntimeError("sqlite INSERT did not return lastrowid")
        return int(cur.lastrowid)

    def rename_album(
        self, album_id: int, name: str, description: str | None = None,
    ) -> bool:
        """Rename an album and optionally update its description.

        Returns True if a row was updated, False if the album
        doesn't exist. Raises ValueError on duplicate name or
        empty name. The centroid key (`album:{id}`) is stable
        across renames so search/Discover references survive.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("album name is required")
        with self._lock:
            try:
                if description is None:
                    cur = self._conn.execute(
                        """
                        UPDATE albums SET name = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (name, _utc_now(), album_id),
                    )
                else:
                    cur = self._conn.execute(
                        """
                        UPDATE albums SET name = ?, description = ?,
                                          updated_at = ?
                        WHERE id = ?
                        """,
                        (name, description, _utc_now(), album_id),
                    )
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"album name already exists: {name!r}") from e
        return cur.rowcount > 0

    def delete_album(self, album_id: int) -> bool:
        """Delete an album. Memberships cascade via FK.

        Returns True if a row was deleted, False if the album
        didn't exist. The centroid registration in app.py must be
        torn down separately — call sites are responsible for
        unregistering `album:{album_id}` from the
        DynamicCentroidRegistry so the centroid doesn't linger.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM albums WHERE id = ?",
                (album_id,),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_album(self, album_id: int) -> dict[str, Any] | None:
        """Return one album row by id, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, name, description, cover_favorite_id,
                       created_at, updated_at
                FROM albums WHERE id = ?
                """,
                (album_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_albums(self) -> list[dict[str, Any]]:
        """Return every album with its member count + first member.

        Ordered by name asc so the UI list is stable. Member
        count is `album_memberships` rows, NOT joined against
        `images` — orphan memberships (favourites whose photo is
        gone from the cache) still count toward the album total
        because they're still part of the user's curation.

        `first_member_id` is the most RECENTLY added photo in the
        album (ORDER BY m.added_at DESC LIMIT 1) — drives the
        /albums index card thumbnail. Empty string when the album
        has no members yet.

        The field name is historical: it was originally the
        chronologically-first member (ORDER BY added_at ASC). The
        card cover now shows the newest addition instead, so the
        name reads as "the member whose photo leads the card".
        `added_at DESC, favorite_id DESC` gives a deterministic
        tie-break — two members added in the same microsecond
        would otherwise pick arbitrarily between them, and the
        cover would flicker between reloads.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT a.id, a.name, a.description, a.cover_favorite_id,
                       a.created_at, a.updated_at,
                       (SELECT COUNT(*) FROM album_memberships m
                        WHERE m.album_id = a.id) AS member_count,
                       COALESCE(
                         (SELECT m.favorite_id FROM album_memberships m
                          WHERE m.album_id = a.id
                          ORDER BY m.added_at DESC, m.favorite_id DESC LIMIT 1),
                         ''
                       ) AS first_member_id
                FROM albums a
                ORDER BY a.name ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def add_album_member(
        self, album_id: int, favorite_id: str,
    ) -> bool:
        """Add `favorite_id` to album `album_id`.

        Returns True if the membership exists after the call
        (whether newly inserted or already present), False if the
        album doesn't exist. Idempotent — re-adding a favourite
        that's already a member is a no-op that returns True.
        Photo existence in the cache is NOT checked here — albums
        are user curation, not a validation surface, so a future-
        favourited photo id can be added ahead of time.
        """
        if not favorite_id:
            raise ValueError("favorite_id is required")
        now = _utc_now()
        with self._lock:
            album_exists = self._conn.execute(
                "SELECT 1 FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
            if album_exists is None:
                return False
            self._conn.execute(
                """
                INSERT OR IGNORE INTO album_memberships
                  (album_id, favorite_id, added_at)
                VALUES (?, ?, ?)
                """,
                (album_id, favorite_id, now),
            )
            self._conn.commit()
        # Album members are user curation, so the photo may not exist
        # yet — the path lookup is best-effort and may find nothing.
        self._record_path_from_qdrant(favorite_id)
        return True

    def remove_album_member(
        self, album_id: int, favorite_id: str,
    ) -> bool:
        """Remove `favorite_id` from album `album_id`.

        Returns True if the album exists (whether or not the
        membership was actually present), False if the album
        doesn't exist. Idempotent — removing a non-member is a
        no-op that still returns True. Always safe to call.
        """
        with self._lock:
            album_exists = self._conn.execute(
                "SELECT 1 FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
            if album_exists is None:
                return False
            self._conn.execute(
                """
                DELETE FROM album_memberships
                WHERE album_id = ? AND favorite_id = ?
                """,
                (album_id, favorite_id),
            )
            self._conn.commit()
        return True

    def list_album_member_ids(self, album_id: int) -> list[str]:
        """Return every favourite id in `album_id`.

        No JOIN against `images` — orphan memberships (favourites
        whose photo is no longer in the cache) still appear, and
        the downstream Qdrant retrieve naturally drops any id it
        can't find. Same contract as `list_favorite_ids`.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT favorite_id FROM album_memberships
                WHERE album_id = ?
                ORDER BY added_at DESC
                """,
                (album_id,),
            ).fetchall()
        return [str(r["favorite_id"]) for r in rows]

    def list_album_members(
        self, album_id: int, limit: int = 200, offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return paginated members of `album_id` with photo metadata.

        The MEMBERSHIP is authoritative user state and lives in
        SQLite. The photo METADATA comes from Qdrant, which owns it.

        A membership whose photo Qdrant no longer holds is an orphan
        and is hidden from the UI — the same shape the old INNER
        JOIN produced, but decided by the collection rather than by
        a derived mirror. The orphan row still exists in
        `album_memberships` and still feeds the album centroid
        compute — only the UI hides it.
        """
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.favorite_id AS id, m.added_at AS added_at
                FROM album_memberships m
                WHERE m.album_id = ?
                ORDER BY m.added_at DESC
                LIMIT ? OFFSET ?
                """,
                (album_id, limit, offset),
            ).fetchall()
        if not rows:
            return []

        payload_by_id = self._payloads_by_id([r["id"] for r in rows])
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = payload_by_id.get(row["id"])
            if payload is None:
                # The photo is gone from the collection — an orphan.
                continue
            out.append({**payload, "id": row["id"], "added_at": row["added_at"]})
        return out

    def count_album_members(self, album_id: int) -> int:
        """How many of the album's members still exist in the collection.

        Liveness is decided by Qdrant, so a membership whose photo
        was pruned is neither shown nor counted. For the total
        membership count including orphans, use
        `len(list_album_member_ids(album_id))`.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT favorite_id AS id FROM album_memberships "
                "WHERE album_id = ?",
                (album_id,),
            ).fetchall()
        if not rows:
            return 0
        live = self._live_point_ids([r["id"] for r in rows])
        return len(live)

    def list_albums_for_favorite(
        self, favorite_id: str,
    ) -> list[dict[str, Any]]:
        """Return every album containing `favorite_id`.

        Used by the per-favourite "in album X" UI badge so the
        user can see which albums a photo belongs to and toggle
        membership from the photo detail page.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT a.id, a.name, a.description
                FROM albums a
                INNER JOIN album_memberships m ON m.album_id = a.id
                WHERE m.favorite_id = ?
                ORDER BY a.name ASC
                """,
                (favorite_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_album_cover(
        self, album_id: int, favorite_id: str | None,
    ) -> bool:
        """Set the cover photo for an album.

        Pass `favorite_id=""` or None to clear the cover. Returns
        True if the album exists (even if the cover was cleared),
        False if the album doesn't exist. The cover is a UI
        affordance — no validation that the favourite is actually
        a member, since albums allow non-favourite photos and the
        cover should still work for orphan selections.
        """
        with self._lock:
            album_exists = self._conn.execute(
                "SELECT 1 FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
            if album_exists is None:
                return False
            self._conn.execute(
                """
                UPDATE albums SET cover_favorite_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (favorite_id or "", _utc_now(), album_id),
            )
            self._conn.commit()
        if favorite_id:
            # The cover can point at a photo that is not a favourite, so
            # this id may have no photo_paths row yet.
            self._record_path_from_qdrant(favorite_id)
        return True

    # ---------------------- Saved searches ----------------------
    #
    # Named prompt presets. The user types a set of positive / negative
    # prompts, names them ("red-dress-no-manikin"), and can re-apply
    # them later from a dropdown. Only the prompt strings are stored —
    # view, centroid, favourites filter, and result limits are session
    # state, not part of the preset.
    #
    # `positives` and `negatives` are serialised JSON arrays of strings
    # in SQLite, matching the same pattern used elsewhere in this file
    # for list-shaped fields. Validation happens in the route layer
    # (length, empty-after-strip) — the IndexDB layer just persists
    # what's handed in.

    def create_saved_search(
        self, name: str, positives: list[str], negatives: list[str],
    ) -> dict[str, Any]:
        """Insert a new saved search. Returns the row as a dict.

        Raises ValueError on UNIQUE-name conflict (routes map this
        to HTTP 409). Empty name / non-list inputs are NOT validated
        here — the route layer owns input shape so the IndexDB can
        stay storage-only.
        """
        name = (name or "").strip()
        now = _utc_now()
        with self._lock:
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO saved_searches
                      (name, positives, negatives, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        name,
                        json.dumps(list(positives or []), ensure_ascii=False),
                        json.dumps(list(negatives or []), ensure_ascii=False),
                        now,
                    ),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"saved search name already exists: {name!r}") from e
        if cur.lastrowid is None:
            raise RuntimeError("sqlite INSERT did not return lastrowid")
        return self.get_saved_search(int(cur.lastrowid)) or {}

    def list_saved_searches(
        self, limit: int = 200, offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (rows, total). Ordered by created_at DESC — newest
        saved searches surface at the top of the UI dropdown. Limit
        and offset are clamped to non-negative ints so callers don't
        need to validate.
        """
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, name, positives, negatives, created_at
                FROM saved_searches
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            total = self._conn.execute(
                "SELECT COUNT(*) AS n FROM saved_searches"
            ).fetchone()
        rows_typed = (_deserialize_saved_search_row(r) for r in rows)
        deserialized = [r for r in rows_typed if r is not None]
        return (
            deserialized,
            int(total["n"] if total else 0),
        )

    def get_saved_search(self, saved_id: int) -> dict[str, Any] | None:
        """Return one saved search by id, or None if missing."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, name, positives, negatives, created_at
                FROM saved_searches
                WHERE id = ?
                """,
                (int(saved_id),),
            ).fetchone()
        return _deserialize_saved_search_row(row) if row else None

    def delete_saved_search(self, saved_id: int) -> bool:
        """Delete a saved search by id. Returns True if a row was
        removed, False if the id didn't exist."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM saved_searches WHERE id = ?",
                (int(saved_id),),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_by_id(self, point_id: str) -> dict[str, Any] | None:
        """The row for one photo: its metadata plus its user flags.

        The metadata comes from Qdrant, the favourite flag from
        SQLite. Returns None when Qdrant does not hold the photo —
        the photo is gone, so there is no row.
        """
        point_id = str(point_id)
        payloads = self._payloads_by_id([point_id])
        payload = payloads.get(point_id)
        if payload is None:
            return None
        with self._lock:
            fav = self._conn.execute(
                "SELECT favorited_at FROM favorites WHERE id = ?",
                (point_id,),
            ).fetchone()
        return {
            **payload,
            "id": point_id,
            "is_favorite": fav is not None,
            "favorited_at": fav["favorited_at"] if fav is not None else None,
        }

    def maybe_refresh(self) -> bool:
        """Run the periodic maintenance if the interval has elapsed.

        The maintenance is the orphan purge: user-state rows whose
        photo the collection no longer holds are dropped. There is no
        derived cache to rebuild — the corpus lives in Qdrant and is
        always current there.

        This is the single entry point for the periodic work, so the
        timestamp the status endpoint reports is the timestamp of the
        last real run rather than a separate bookkeeping value.

        Returns True when the purge ran, False when it was not due.
        """
        if time.time() - self._last_refresh <= self.refresh_interval_seconds:
            return False
        self.purge_orphaned_user_data()
        self._last_refresh = time.time()
        return True

    def try_acquire_refresh_lock(self, *, blocking: bool = False) -> bool:
        """Try to acquire the refresh lock.

        Returns True if acquired, False if already held by another
        caller (the periodic task, the manual endpoint, etc.). The
        caller MUST pair every successful acquire with
        `release_refresh_lock()` in a finally block. Non-blocking by
        default — the manual endpoint prefers to bail fast and let
        the periodic task finish than to queue a second scroll.
        """
        if blocking:
            self._refresh_lock.acquire()
            return True
        return self._refresh_lock.acquire(blocking=False)

    def release_refresh_lock(self) -> None:
        """Release the refresh lock. No-op if not held."""
        with suppress(RuntimeError):
            self._refresh_lock.release()

    def last_refresh_time(self) -> float:
        """Unix timestamp of the last successful maintenance run,
        or 0.0 if never refreshed.
        """
        with self._lock:
            return self._last_refresh

    def qdrant_point_count(self) -> int:
        """Return the current point count in the Qdrant collection.

        Used for drift detection vs the IndexDB cache. Returns -1 if
        Qdrant is unreachable (caller logs the error and continues).
        """
        try:
            info = self.qdrant_client.client.get_collection(self.qdrant_client.collection)
            return int(info.points_count)
        except Exception as e:  # noqa: BLE001
            logger.warning("qdrant_point_count failed: %s", e)
            return -1

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exists_locked(self, point_id: str) -> bool:
        """Does the collection still hold this photo?

        Qdrant is the source of truth for photo existence. The
        `images` mirror used to answer this, which meant a stale
        mirror could make a live photo look deleted (or a deleted
        one look live).
        """
        return bool(self._live_point_ids([str(point_id)]))

    def _row_from_point(self, point: dict[str, Any]) -> dict[str, Any]:
        payload = point.get("payload") or {}
        point_id = str(point.get("id") or payload.get("id") or "")
        path = str(payload.get("path") or "")
        if not point_id or not path:
            raise ValueError(f"Qdrant point missing id/path payload: {point!r}")
        return {
            "id": point_id,
            "path": path,
            "shard": str(payload.get("shard") or ""),
            "collection": str(payload.get("collection") or ""),
            "mtime": _optional_int(payload.get("mtime")),
            "size": _optional_int(payload.get("size")),
            "indexed_at": payload.get("indexed_at"),
            "blurhash": str(payload.get("blurhash") or ""),
        }

def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _deserialize_saved_search_row(row: sqlite3.Row | dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a saved_searches row into a JSON-friendly dict.

    `positives` and `negatives` are stored as JSON-encoded strings in
    SQLite (matching how the rest of this file serialises list
    columns); on read we always hand back Python lists so callers
    don't have to think about the on-disk shape. Corrupt JSON is
    treated as an empty list rather than raising — a single bad row
    shouldn't take down the whole /api/saved-searches response, and
    the user can delete the bad row via DELETE on its id.
    """
    if row is None:
        return None
    out = dict(row)
    for col in ("positives", "negatives"):
        raw = out.get(col)
        if isinstance(raw, list):
            continue  # already deserialised (tests / callers that pass dicts)
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else []
        except (TypeError, ValueError):
            parsed = []
        if not isinstance(parsed, list):
            parsed = []
        out[col] = [str(x) for x in parsed]
    return out
