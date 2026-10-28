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

import numpy as np

from search.random_session import StridedDeck, random_stride

if TYPE_CHECKING:
    from search.qdrant_client import QdrantSearch

logger = logging.getLogger(__name__)

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
        id_cache_ttl_seconds: int = 900,
    ):
        self.db_path = db_path
        self.qdrant_client = qdrant_client
        self.refresh_interval_seconds = refresh_interval_seconds
        self._lock = threading.RLock()
        # The periodic maintenance clock starts at "now": there is
        # no derived cache to build at startup, and the corpus in
        # Qdrant is always current.
        self._last_refresh = time.time()
        # The corpus's point ids, cached per collection filter. A
        # full Qdrant id scroll is ~25s on the 2.2M corpus, so the
        # scroll is done once and the result reused; only the
        # shuffle is per session.
        self._id_cache: dict[tuple[str, ...], tuple[float, list[str]]] = {}
        self._id_cache_ttl_s = float(id_cache_ttl_seconds)
        # Serialises cache BUILDS (the scroll), so a request and the
        # background warmer do not both pay for it.
        self._id_cache_lock = threading.Lock()
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

    def purge_orphaned_user_data(self) -> dict[str, int]:
        """Delete user-state rows whose photo is gone from Qdrant.

        Orphans accumulate when the indexer prunes Qdrant points
        (files deleted from disk): the favourite/dislike/album rows
        were deliberately kept so re-indexing the same path (same
        uuid5 id) would re-attach them. That re-attach window has a
        cost though — orphan ids poison qdrant.recommend() (a single
        missing id 404s the WHOLE call; see filter_live_ids in
        search/for_you.py) and orphan album members render as dead
        thumbnails.

        Called from POST /api/cache/refresh.

        **Liveness is decided by Qdrant, and by nothing else.**
        There used to be an `images` table in this database — a
        DERIVED mirror of the collection. It could be empty or stale
        (a half-finished rebuild, a mirror that was never populated,
        a mirror that drifted from the collection). If it were used
        to decide which user rows are orphans, a stale mirror would
        delete the user's favourites, dislikes and album memberships
        — the authoritative state — because the photos they point at
        are not in the mirror. The mirror said nothing about whether
        a photo existed; only Qdrant did. The mirror is gone now, and
        so is the possibility of that mistake.

        So the candidate ids are collected from the user-state
        tables, and each is checked against Qdrant directly. A
        candidate that Qdrant still holds is kept, no matter what
        the mirror says. A candidate Qdrant no longer holds is
        deleted.

        Purges:
          - favorites / dislikes rows whose id Qdrant no longer holds
          - album_memberships whose photo Qdrant no longer holds (a
            membership of a deleted photo renders as a dead
            thumbnail; note memberships do NOT require the photo
            to be a favourite — add_album_member allows any id —
            so liveness is checked against Qdrant only)
          - albums.cover_favorite_id refs pointing at photos Qdrant
            no longer holds (reset to '' so the UI falls back to
            the first member)

        Keeps: feedback_events (append-only history log, not state —
        nothing feeds it back into recommend) and saved_searches
        (prompt text only, no point ids).

        Returns per-table deletion counts for logging/API response.
        """
        # Every id the user state refers to. This is a small set —
        # thousands of rows, not millions — so each one is checked
        # against Qdrant directly.
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
                return {
                    "favorites": 0,
                    "dislikes": 0,
                    "album_memberships": 0,
                    "album_covers_reset": 0,
                }

        # NAS-dropout guard: a collection that reports ZERO points is
        # far more likely to mean "the collection was just recreated"
        # or "we are looking at the wrong collection" than "every
        # photo was deleted". Purging on that signal would wipe the
        # user's entire taste history irrecoverably. An empty
        # collection is therefore not evidence that anything is an
        # orphan — the orphans survive until the collection has
        # points again and a real liveness check can answer.
        try:
            total_points = int(self.qdrant_client.count_points())
        except Exception as e:  # noqa: BLE001
            logger.warning("purge: count_points failed, keeping all user data: %s", e)
            return {
                "favorites": 0,
                "dislikes": 0,
                "album_memberships": 0,
                "album_covers_reset": 0,
            }
        if total_points == 0:
            logger.info(
                "purge: collection %r is empty — not evidence that any "
                "photo is gone; keeping all user data",
                getattr(self.qdrant_client, "collection", "?"),
            )
            return {
                "favorites": 0,
                "dislikes": 0,
                "album_memberships": 0,
                "album_covers_reset": 0,
            }

        # Which of those ids does Qdrant still hold? This is the
        # authoritative liveness check.
        live_ids = self._live_point_ids(sorted(candidate_ids))

        with self._lock:
            orphans = [pid for pid in candidate_ids if pid not in live_ids]
            if not orphans:
                return {
                    "favorites": 0,
                    "dislikes": 0,
                    "album_memberships": 0,
                    "album_covers_reset": 0,
                }
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

    def shuffled_id_deck(
        self, collections: tuple[str, ...] | list[str] = ()
    ) -> list[str]:
        """Materialize a full shuffled permutation of all point ids.

        This is what backs the /random session-cursor shape: one
        full shuffle per session, then sequential reads via
        `rows_by_ids`.

        The ids come from Qdrant, which owns the corpus. The
        collection filter is applied SERVER-SIDE against the
        `collection` keyword index, so the filtered case does not
        have to fetch any payload.

        **The id list is cached; the shuffle is per session.**

        A full id scroll over the 2.2M-point production corpus
        takes ~25s — Qdrant has to read every point record, and the
        REST API has no id-only scroll mode. That was cheap when
        the deck came from a local SQLite table (2.26s for the
        whole thing, scroll and shuffle together); over the network
        it is not, and a session that pays 25s to open is a session
        the user does not use.

        So the scroll result is cached per collection filter and
        reused, and only the SHUFFLE is done per session — which is
        what makes each session's order different, and what the
        /random UX actually promises ("a fresh shuffle per call").

        The shuffle is a numpy permutation over the cached list:
        ~0.4s for 2.2M ids, against ~1.7s for `random.shuffle`'s
        Python-level loop. The returned list shares the cached
        list's string objects, so a session costs a pointer array
        (~17 MB for 2.2M ids) rather than a second copy of every
        id (~180 MB).

        Returns ALL ids matching the collection filter (or all ids
        if `collections` is empty), in a random order. The caller
        uses this as the deck; offsets slice into it.
        """
        collections = tuple(c for c in (collections or ()) if c)
        ids = self._cached_point_ids(collections)
        if len(ids) < 2:
            return list(ids)
        # Qdrant returns ids in storage order, and the ids are uuid5
        # hashes of the path, so that order is already uncorrelated
        # with the photos. The deck therefore only needs to be a
        # permutation of it, not a full shuffle: a random start plus
        # a stride coprime with the length visits every index exactly
        # once and costs two integers.
        #
        # The alternative — a numpy permutation — cost a pointer
        # array per session: 18.5 MB for the 2.2M-point corpus, held
        # for the session's 30-minute TTL with no cap on the number
        # of live sessions.
        rng = np.random.default_rng()
        n = len(ids)
        start = int(rng.integers(0, n))
        stride = random_stride(n, rng)
        return StridedDeck(ids, start, stride)

    def _cached_point_ids(
        self, collections: tuple[str, ...]
    ) -> list[str]:
        """The corpus's point ids, cached per collection filter.

        The cache holds the id LIST (shared, immutable). Callers that
        want a random order make their own permuted copy — see
        `shuffled_id_deck`.
        """
        with self._lock:
            entry = self._id_cache.get(collections)
            if entry is not None:
                return entry[1]

        return self.refresh_id_cache(collections)

    def refresh_id_cache(
        self, collections: tuple[str, ...] = ()
    ) -> list[str]:
        """Rebuild the cached id list for a collection filter.

        Called on a miss, and by the background warmer. A stale
        entry is served rather than rebuilt on the request path —
        the ids do not change unless the corpus is re-indexed, and a
        request should never wait ~25s for a fresh scroll.
        """
        collections = tuple(c for c in (collections or ()) if c)
        with self._id_cache_lock:
            with self._lock:
                entry = self._id_cache.get(collections)
                if entry is not None and (time.time() - entry[0]) < self._id_cache_ttl_s:
                    return entry[1]
            ids = self._all_point_ids(collections)
            with self._lock:
                self._id_cache[collections] = (time.time(), ids)
        return ids

    def _all_point_ids(self, collections: tuple[str, ...] = ()) -> list[str]:
        """Every point id in the collection, optionally filtered by collection.

        Scrolls the collection with `with_payload=False` so only the
        ids cross the wire. The collection filter is a server-side
        `MatchAny` against the keyword index on `collection`, so the
        filtered case transfers no payload at all.
        """
        try:
            hits = self.qdrant_client.scroll_ids(collections)
        except Exception as e:  # noqa: BLE001
            logger.warning("id scroll failed (%s) — empty deck", e)
            return []
        return [hit.id for hit in hits]

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
        """
        try:
            return int(self.qdrant_client.count_points())
        except Exception as e:  # noqa: BLE001
            logger.warning("count_points failed: %s", e)
            return 0

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
