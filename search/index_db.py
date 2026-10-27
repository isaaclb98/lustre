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
    ):
        self.db_path = db_path
        self.qdrant_client = qdrant_client
        self.refresh_interval_seconds = refresh_interval_seconds
        self._lock = threading.RLock()
        self._last_refresh = 0.0
        # Refresh lock — non-reentrant guard against the periodic
        # background task and the manual /api/cache/refresh endpoint
        # racing. Acquired before `init_from_qdrant` runs, released
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
                CREATE TABLE IF NOT EXISTS images (
                  id            TEXT PRIMARY KEY,
                  path          TEXT NOT NULL,
                  shard         TEXT DEFAULT '',
                  collection    TEXT DEFAULT '',
                  mtime         INTEGER,
                  size          INTEGER,
                  indexed_at    TEXT,
                  width         INTEGER,
                  height         INTEGER,
                  blurhash       TEXT DEFAULT ''
                );

                -- Persistent user state. Independent of the images
                -- cache so a cache rebuild (init_from_qdrant) never
                -- drops a favourite for a photo that still exists.
                -- A favourite whose photo id was PRUNED from Qdrant
                -- (file deleted from disk) is purged at the end of
                -- the next forced cache rebuild — see
                -- purge_orphaned_user_data().
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
            # Lightweight migrations for additive columns. Idempotent —
            # _ensure_column checks PRAGMA table_info before issuing
            # ALTER TABLE ADD COLUMN, so re-opens are no-ops. Must run
            # BEFORE creating the indexes that reference these columns,
            # since a pre-migration DB doesn't have them yet.
            self._ensure_column("images", "width", "INTEGER")
            self._ensure_column("images", "height", "INTEGER")
            self._ensure_column("images", "collection", "TEXT DEFAULT ''")
            self._ensure_column("images", "blurhash", "TEXT DEFAULT ''")
            self._conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_images_path
                  ON images(path);
                CREATE INDEX IF NOT EXISTS idx_images_collection
                  ON images(collection) WHERE collection != '';
                """
            )
            self._conn.commit()

            # Filename/path FTS5 index. Backed by an FTS5 virtual
            # table that stores its own copy of `path` — no
            # `content='images'` linkage. The alternative
            # `content=` mode requires a `'rebuild'` step after
            # any bulk insert AND silently fails to populate the
            # inverted index on certain commit-timing
            # combinations, so we keep the path denormalised
            # inside the FTS index itself. The FTS index is
            # read-only from the application code's perspective
            # — the triggers below are the only path that
            # touches it. Sync triggers cover INSERT / UPDATE /
            # DELETE on `images`.
            #
            # Why not a `path_tokens` column on `images`? FTS5
            # handles the tokenisation once and serves MATCH
            # queries via the inverted index, which is O(log N)
            # instead of full-table scans. A denormalised tokens
            # column would either be hand-maintained (drift risk)
            # or built lazily on read (slow).
            #
            # The triggers populate `images_fts` automatically as
            # `init_from_qdrant` walks Qdrant and INSERTs rows
            # into `images`. Single code path; source of truth is
            # Qdrant. The migration method forces a one-time
            # refresh on legacy DBs (where `images` is already
            # populated but `images_fts` is empty).
            #
            # Patterns accepted by `path_token_ids()`:
            #   * `chaewon`           — token substring match
            #                            (FTS5 default).
            #   * `chaewon*`          — token prefix match.
            #   Anything else (notably `*chaewon` suffix or
            #   `*.jpg` glob) raises ValueError — FTS5 doesn't
            #   support suffix matching and fnmatch semantics are
            #   not native. Callers that need suffix should
            #   switch to a substring (drop the leading `*`).
            self._create_images_fts()

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

    FTS_TABLE = "images_fts"
    FTS_TOKENIZER = "unicode61 remove_diacritics 2"

    def _create_images_fts(self) -> None:
        """Create the FTS5 virtual table + sync triggers if missing.

        Triggers cover the three DML paths that hit `images`:
        `ai` (AFTER INSERT), `au` (AFTER UPDATE), `ad` (AFTER
        DELETE). The OLD/NEW pseudo-rows carry the rowid and path
        values; we project to the FTS table's `(rowid, path)` shape.

        Note: FTS5 has two ways to delete a row from the index — the
        `INSERT INTO fts(fts, rowid, ...) VALUES('delete', ...)`
        "delete command" and the SQL `DELETE FROM fts WHERE ...`.
        The delete command requires every indexed column to be
        re-supplied in the INSERT and is brittle (varies across
        SQLite minor versions — we hit SQLITE_LOGIC_ERROR on the
        3.45.1 build under the typical syntax). The plain SQL DELETE
        works reliably inside triggers, so we use that form here.

        CREATE TRIGGER has IF NOT EXISTS on SQLite >= 3.10; the
        floor we target ships 3.41+, so it's safe. CREATE VIRTUAL
        TABLE also supports IF NOT EXISTS in FTS5.

        Schema-failure recovery: if any of these statements raises
        we re-raise so the surrounding `_init_schema` doesn't
        silently leave the DB in a half-built state.
        """
        with self._lock:
            # FTS_TABLE and FTS_TOKENIZER are static class constants
            # inlined here as literal identifiers — no user input.
            ddl = """
                CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
                  path,
                  tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS images_ai
                AFTER INSERT ON images BEGIN
                  INSERT INTO images_fts(rowid, path)
                  VALUES (new.rowid, new.path);
                END;

                CREATE TRIGGER IF NOT EXISTS images_au
                AFTER UPDATE ON images BEGIN
                  DELETE FROM images_fts
                  WHERE rowid = old.rowid;
                  INSERT INTO images_fts(rowid, path)
                  VALUES (new.rowid, new.path);
                END;

                CREATE TRIGGER IF NOT EXISTS images_ad
                AFTER DELETE ON images BEGIN
                  DELETE FROM images_fts
                  WHERE rowid = old.rowid;
                END;
                """
            self._conn.executescript(ddl)
            self._conn.commit()

    def path_token_ids(self, pattern: str) -> list[str] | None:
        """Resolve a filename/path-substring pattern to image ids.

        Returns `None` when the pattern is empty / whitespace-only —
        the caller is expected to skip the filter in that case so we
        don't pay the FTS5 round-trip on every request. Otherwise
        returns the list of image ids whose `path` matches the
        pattern under the FTS5 token-substring / prefix rules:

          * `chaewon`  → token-substring match. FTS5 splits the path
            into tokens on `/`, `-`, `_`, `.` etc, and matches each
            token as a substring. So `/photos/kpop/chaewon/2024.jpg`
            tokenises to [photos, kpop, chaewon, 2024, jpg] and any
            single-token query (e.g. `chaewon`, `kpop`, `2024`)
            hits. The match is anchored to token boundaries — `won`
            does NOT match `chaewon` (substring but not token-aligned).
          * `chaewon*` → token-prefix match. `won*` would NOT match
            `chaewon` (token prefix, not substring prefix).

        Anything else (notably `*ewon` or `*.jpg`) raises a
        ValueError so the caller surfaces a 400 with a clear error.
        The fnmatch-style suffix / bracket / question-mark syntax is
        intentionally NOT supported — FTS5 has no native equivalent,
        and silently no-op'ing would confuse users into thinking the
        filter is broken. Callers can either drop the leading `*`
        (becomes a token match) or use a plain substring via the
        implicit token-substring semantics above.

        The matching itself is case-insensitive (FTS5 default with
        unicode61), and diacritics are folded (remove_diacritics=2
        tokenizer option).

        Performance: FTS5 with content='images' is fast — typically
        sub-10ms for tens of thousands of matching rows on the
        live 1.5M cache. The caller should still apply the
        cardinality guard (see app.py) before passing the result to
        Qdrant, since `HasId` with >50% of the collection is
        strictly worse than no filter at all.
        """
        if not pattern or not pattern.strip():
            return None
        # Validate: accept only alphanumeric runs and (optionally) a
        # single trailing `*`. Reject anything else so the caller can
        # surface a 400. Strip quotes / surrounding whitespace first
        # so casual typing doesn't trip the validator.
        candidate = pattern.strip().strip('"').strip("'")
        if not candidate:
            return None
        # Explicit suffix / glob rejection. FTS5 has no native
        # suffix-match and we deliberately don't try to emulate it
        # — silent zero-result matches would look like a broken
        # filter. The user should drop the leading `*` and use the
        # implicit token-substring semantics instead.
        if "*" in candidate[:-1]:
            # Trailing `*` is the only `*` we accept (prefix match).
            # Anything else — leading `*`, internal `*`, multiple
            # `*` — is a syntax error. We surface this BEFORE the
            # FTS5 layer so the message is about filename patterns,
            # not about FTS5 internals.
            raise ValueError(
                f"filename pattern only supports trailing '*' for "
                f"prefix match (got {pattern!r})"
            )
        prefix_only = candidate.endswith("*")
        body = candidate[:-1] if prefix_only else candidate
        # Whitespace or empty after stripping the optional `*` is a
        # no-op (treated like no pattern) — the user typed just `*`.
        if not body or body != body.strip() or any(ch.isspace() for ch in body):
            raise ValueError(
                f"filename pattern must be a single token "
                f"(got {pattern!r})"
            )
        # Disallow FTS5 operators inside the body. The unicode61
        # tokeniser already strips most punctuation, but quotes and
        # colons are FTS5 syntax we explicitly want to forbid.
        if any(ch in body for ch in ('"', "'", ":", "(", ")", "+", "-")):
            # `+` and `-` are FTS5 mandatory/exclusion operators;
            # we don't surface them as a user feature.
            raise ValueError(
                f"filename pattern must not contain FTS5 operators "
                f"(got {pattern!r})"
            )
        # Build the FTS5 query. Token-substring is the FTS5 default
        # (just `body`), so a bare alphanumeric token is enough.
        # Prefix mode appends `*` to express "starts with body".
        fts_query = f"{body}*" if prefix_only else body
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"""
                    SELECT images.id FROM {self.FTS_TABLE}
                    INNER JOIN images ON images.rowid = {self.FTS_TABLE}.rowid
                    WHERE {self.FTS_TABLE} MATCH ?
                    """,  # noqa: S608
                    (fts_query,),
                ).fetchall()
            except sqlite3.OperationalError as e:
                # FTS5 syntax errors fall through as OperationalError.
                # Surface as ValueError so the API layer can 400
                # uniformly.
                raise ValueError(f"invalid filename pattern {pattern!r}: {e}") from e
        return [str(r["id"]) for r in rows] if rows else []




    def init_from_qdrant(self, force: bool = False) -> int:
        """
        Rebuild the photo-metadata cache from Qdrant.

        The `images` table is disposable cache of Qdrant payload
        metadata. This method wipes it and repopulates from a fresh
        scroll. User-data tables (favorites, dislikes, albums) are
        NOT wiped — but when the rebuild produces a non-empty cache,
        `purge_orphaned_user_data()` runs afterwards and deletes
        user rows whose photo id is no longer in Qdrant (see that
        method for the full contract and the empty-rebuild guard).

        When `force=False` (the default) and the cache already contains
        rows, this is a no-op for the Qdrant scroll: the previous
        build is restored from SQLite and the periodic refresh task
        remains responsible for catching drift. Pass `force=True` to
        always rebuild (used by tests and manual recovery / the
        `POST /api/cache/refresh` endpoint).
        """
        if not force:
            with self._lock:
                existing = int(
                    self._conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
                )
            if existing > 0:
                logger.info(
                    "index cache already populated (%d rows); skipping Qdrant scroll",
                    existing,
                )
                self._last_refresh = time.time()
                return existing
        count = 0
        with self._lock:
            # Disposable cache: a force rebuild is a full wipe + repopulate.
            # User-data tables (favorites/dislikes/albums) survive the
            # wipe itself; orphans among them are purged afterwards by
            # purge_orphaned_user_data() (below), only when the rebuild
            # produced rows.
            self._conn.execute("DELETE FROM images")
            try:
                for batch in self.qdrant_client.scroll_all():
                    rows = [self._row_from_point(point) for point in batch]
                    if not rows:
                        continue
                    self._conn.executemany(
                        """
                        INSERT INTO images (id, path, shard, collection, mtime, size, indexed_at, blurhash)
                        VALUES (:id, :path, :shard, :collection, :mtime, :size, :indexed_at, :blurhash)
                        """,
                        rows,
                    )
                    count += len(rows)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                logger.exception("index cache rebuild failed")
                raise
            self._last_refresh = time.time()
        logger.info("index cache built from Qdrant: %d points", count)
        # Purge user-data rows whose photo id is no longer in the
        # (freshly rebuilt) cache. Only safe right here, right after
        # a full rebuild — at any other moment `images` may be stale
        # and the purge would delete rows for photos that still
        # exist. Skipped when the rebuild produced zero rows: an
        # empty Qdrant is far more likely to mean "collection
        # unreachable / mid-rebuild" than "user deleted 2M photos",
        # and wiping the entire taste history on a transient blip
        # is unrecoverable.
        if count > 0:
            self.purge_orphaned_user_data()
        return count

    def refresh_incremental(self) -> dict[str, int]:
        """
        Round-35: sync images.db to match Qdrant via a 3-way diff.

        Replaces the wipe+rebuild path that init_from_qdrant(force=True)
        used to do. Instead of nuking every row and re-INSERTing 2M,
        this:

          1. Cheap count check: if Qdrant's points_count equals our
             sqlite row count, return early. Steady-state cost is
             one round-trip to Qdrant (an `O(1)` `count()`), nothing
             else. This is the new periodic-refresh fast path.

          2. Drift detected: id-only scroll of Qdrant (cheap, no
             payload deserialization) + one SELECT id FROM images.
             Set-difference in memory:

               new_ids     = qdrant_ids - sqlite_ids  →  INSERT
               orphan_ids  = sqlite_ids - qdrant_ids  →  DELETE
               common_ids  = qdrant_ids ∩ sqlite_ids  →  no-op
                             (the indexer's write-through keeps
                              common rows current; a payload
                              mutation without an id change is
                              rare and self-corrects on the next
                              rebuild pass)

          3. INSERTs: retrieve payloads for new_ids and feed them
             through upsert_records (the existing INSERT OR REPLACE).
             Bounded by drift size, not total collection size.

          4. DELETEs: chunk into 999-id batches (SQLite's default
             SQLITE_MAX_VARIABLE_NUMBER) and DELETE WHERE id IN (...).

        Returns:
            {
              "qdrant_count": int,
              "sqlite_count": int,   # before the sync
              "inserted":     int,
              "deleted":      int,
              "skipped":      bool,  # true when fast path took it
            }

        Concurrency:
          - The Qdrant id-scroll is a point-in-time snapshot.
          - Any indexer writes that land mid-scroll are missing
            from the diff but get written via write-through anyway.
          - The orphan-DELETE step can theoretically delete a row
            the indexer just wrote via write-through — but only
            if that exact id was deleted from Qdrant in the same
            window. In practice that means an external admin
            deleted the point at the same time the indexer was
            re-upserting it, which is a user-driven ambiguity and
            the LAST write wins either way.
          - No DELETE FROM images; only the orphan ids get removed.

        User data:
          - purge_orphaned_user_data() runs at the end (same
            contract as init_from_qdrant): when the orphan set is
            non-empty, favorite/dislike rows for those ids are
            cleared.
        """
        # Step 1: cheap count check.
        try:
            qdrant_count = int(
                self.qdrant_client.client.count(
                    collection_name=self.qdrant_client.collection,
                ).count
            )
        except Exception as e:
            logger.warning("refresh_incremental: Qdrant count() failed: %s", e)
            raise

        with self._lock:
            sqlite_count = int(
                self._conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
            )

        if qdrant_count == sqlite_count:
            # Steady-state fast path. Skip the whole diff. ~1ms.
            self._last_refresh = time.time()
            logger.debug(
                "refresh_incremental: in sync (qdrant=%d, sqlite=%d); fast path",
                qdrant_count,
                sqlite_count,
            )
            return {
                "qdrant_count": qdrant_count,
                "sqlite_count": sqlite_count,
                "inserted": 0,
                "deleted": 0,
                "skipped": True,
            }

        logger.info(
            "refresh_incremental: drift detected (qdrant=%d, sqlite=%d); diffing",
            qdrant_count,
            sqlite_count,
        )

        # Step 2a: id-only scroll of Qdrant.
        qdrant_ids: set[str] = set()
        try:
            for id_batch in self.qdrant_client.scroll_ids_only():
                qdrant_ids.update(id_batch)
        except Exception:
            logger.exception("refresh_incremental: scroll_ids_only failed")
            raise

        # Step 2b: SELECT all ids from images.
        with self._lock:
            sqlite_ids = {
                row["id"]
                for row in self._conn.execute("SELECT id FROM images").fetchall()
            }

        # Step 2c: set-difference.
        new_ids = qdrant_ids - sqlite_ids
        orphan_ids = sqlite_ids - qdrant_ids

        # Step 3: INSERT new ids.
        inserted = 0
        if new_ids:
            logger.info(
                "refresh_incremental: %d new ids to insert", len(new_ids)
            )
            try:
                # retrieve() with a long list of ids can fail with
                # a payload-size error on huge diffs. Chunk into
                # 256-id retrievals to stay under the wire limit.
                retrieved_points = []
                for i in range(0, len(new_ids), 256):
                    chunk = list(new_ids)[i : i + 256]
                    retrieved_points.extend(
                        self.qdrant_client.client.retrieve(
                            collection_name=self.qdrant_client.collection,
                            ids=chunk,
                            with_payload=True,
                            with_vectors=False,
                        )
                    )
                inserted = self.upsert_records(retrieved_points)
            except Exception:
                logger.exception(
                    "refresh_incremental: retrieve+upsert failed for %d new ids",
                    len(new_ids),
                )
                raise

        # Step 4: DELETE orphan ids.
        deleted = 0
        if orphan_ids:
            logger.info(
                "refresh_incremental: %d orphan ids to delete", len(orphan_ids)
            )
            with self._lock:
                try:
                    # Chunk into 999-id batches to respect SQLite's
                    # default SQLITE_MAX_VARIABLE_NUMBER.
                    for i in range(0, len(orphan_ids), 999):
                        chunk = list(orphan_ids)[i : i + 999]
                        placeholders = ",".join("?" for _ in chunk)
                        cur = self._conn.execute(
                            f"DELETE FROM images WHERE id IN ({placeholders})",  # noqa: S608
                            chunk,
                        )
                        deleted += cur.rowcount
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise

        self._last_refresh = time.time()

        # Step 5: purge user-data orphans (same contract as
        # init_from_qdrant — only after we've actually changed
        # the images cache).
        if orphan_ids:
            self.purge_orphaned_user_data()

        logger.info(
            "refresh_incremental: done inserted=%d deleted=%d (qdrant=%d, sqlite was %d)",
            inserted,
            deleted,
            qdrant_count,
            sqlite_count,
        )
        return {
            "qdrant_count": qdrant_count,
            "sqlite_count": sqlite_count,
            "inserted": inserted,
            "deleted": deleted,
            "skipped": False,
        }

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

        Called at the end of every forced `images` cache rebuild
        (init_from_qdrant(force=True)) — i.e. on POST /api/cache/
        refresh and the periodic 6h refresh — when `images` is
        guaranteed to mirror Qdrant. NOT called on the startup
        warm-up path, which may short-circuit against a stale cache.

        **Liveness is decided by Qdrant, never by the `images` cache.**
        The `images` table is a DERIVED mirror: it can be empty or
        stale (a half-finished rebuild, a mirror that was never
        populated, a mirror that drifted from the collection). If it
        were used to decide which user rows are orphans, a stale
        mirror would delete the user's favourites, dislikes and album
        memberships — the authoritative state — because the photos
        they point at are not in the mirror. The mirror says nothing
        about whether a photo exists; only Qdrant does.

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

    def pick_random(self, n: int) -> list[str]:
        if n <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM images ORDER BY RANDOM() LIMIT ?",
                (int(n),),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def shuffled_id_deck(
        self, collections: tuple[str, ...] | list[str] = ()
    ) -> list[str]:
        """Materialize a full shuffled permutation of all point ids.

        This is what backs the /random session-cursor shape: one
        full shuffle per session, then sequential reads via
        `rows_by_ids`. At 182 photos this is microseconds; at 2M
        it's a one-time ~5s query, then O(1) per request.

        Returns ALL ids matching the collection filter (or all ids
        if `collections` is empty), in a random order. The caller
        uses this as the deck; offsets slice into it.
        """
        collections = tuple(c for c in (collections or ()) if c)
        with self._lock:
            if collections:
                placeholders = ",".join("?" for _ in collections)
                rows = self._conn.execute(
                    f"""
                    SELECT id FROM images
                    WHERE collection IN ({placeholders})
                    ORDER BY RANDOM()
                    """,  # noqa: S608
                    collections,
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id FROM images ORDER BY RANDOM()"
                ).fetchall()
        return [str(row["id"]) for row in rows]

    def rows_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Look up full row dicts for a list of point ids.

        Returns one row per id that exists in the cache. Order of
        the returned rows is arbitrary — callers that need a
        specific order (e.g. /random session-cursor) should reorder
        themselves. Missing ids are silently dropped.
        """
        if not ids:
            return []
        ids = [str(i) for i in ids if i]
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT i.id, i.path, i.shard, i.collection, i.mtime,
                       i.size, i.indexed_at, i.width, i.height,
                       i.blurhash,
                       (f.id IS NOT NULL) AS is_favorite,
                       f.favorited_at,
                       (d.id IS NOT NULL) AS is_disliked
                FROM images i
                LEFT JOIN favorites f ON i.id = f.id
                LEFT JOIN dislikes d ON i.id = d.id
                WHERE i.id IN ({placeholders})
                """,  # noqa: S608
                ids,
            ).fetchall()
        return [dict(row) for row in rows]

    def pick_unseen(self, n: int, exclude: set[str]) -> list[str]:
        if n <= 0:
            return []
        exclude = set(exclude or set())
        with self._lock:
            if not exclude:
                rows = self._conn.execute(
                    "SELECT id FROM images ORDER BY RANDOM() LIMIT ?",
                    (int(n),),
                ).fetchall()
            else:
                placeholders = ",".join("?" for _ in exclude)
                params: list[Any] = [*exclude, int(n)]
                rows = self._conn.execute(
                    f"""
                    SELECT id FROM images
                    WHERE id NOT IN ({placeholders})
                    ORDER BY RANDOM()
                    LIMIT ?
                    """,  # noqa: S608
                    params,
                ).fetchall()
        return [str(row["id"]) for row in rows]

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
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()
        return int(row["n"] if row else 0)

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
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        # INNER JOIN — a favourite whose photo id is no longer in the
        # cache (orphan) is excluded. We could expose orphans via a
        # separate query if the operator wants to clean them up, but
        # for the UI the JOIN is the right shape.
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT i.id, i.path, i.shard, i.collection, i.mtime,
                       i.size, i.indexed_at, i.width, i.height, i.blurhash,
                                              f.favorited_at
                FROM images i
                INNER JOIN favorites f ON i.id = f.id
                ORDER BY f.favorited_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_favorites(self) -> int:
        # Count only favourites whose photo is still in the cache.
        # Orphan favourites don't appear in /favorites but stay in the
        # table for potential re-attachment.
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM favorites f
                INNER JOIN images i ON i.id = f.id
                """
            ).fetchone()
        return int(row["n"] if row else 0)

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

    def list_dislikes(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """Same JOIN shape as `list_favorites` so the dislike gallery
        page can reuse the result-grid partial without changes."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT i.id, i.path, i.shard, i.collection, i.mtime,
                       i.size, i.indexed_at, i.width, i.height, i.blurhash,
                       d.disliked_at, d.source
                FROM images i
                INNER JOIN dislikes d ON i.id = d.id
                ORDER BY d.disliked_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_dislikes(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM dislikes d "
                "INNER JOIN images i ON i.id = d.id"
            ).fetchone()
        return int(row["n"] if row else 0)

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
        searches, feedback events) is cleared first. The images
        cache is also wiped; init_from_qdrant will repopulate it
        after the indexer finishes.

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
                DELETE FROM images;
                DELETE FROM images_fts;
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

        INNER JOIN against `images` so orphan memberships are
        hidden from the UI (same shape as `list_favorites`). The
        orphan rows still exist in `album_memberships` and still
        feed the album centroid compute — only the UI hides them.
        """
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT i.id, i.path, i.shard, i.collection, i.mtime,
                       i.size, i.indexed_at, i.width, i.height, i.blurhash,
                       m.added_at
                FROM album_memberships m
                INNER JOIN images i ON i.id = m.favorite_id
                WHERE m.album_id = ?
                ORDER BY m.added_at DESC
                LIMIT ? OFFSET ?
                """,
                (album_id, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_album_members(self, album_id: int) -> int:
        """Count members of `album_id` whose photo is in the cache.

        Matches the JOIN semantics of `list_album_members` (orphan
        memberships hidden). For the total membership count
        including orphans, use `len(list_album_member_ids(album_id))`.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM album_memberships m
                INNER JOIN images i ON i.id = m.favorite_id
                WHERE m.album_id = ?
                """,
                (album_id,),
            ).fetchone()
        return int(row["n"] if row else 0)

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
        # LEFT JOIN: photo metadata always returned, favourite info
        # filled in when present. (f.id IS NOT NULL) becomes the
        # boolean `is_favorite` field.
        with self._lock:
            row = self._conn.execute(
                """
                SELECT i.id, i.path, i.shard, i.collection, i.mtime,
                       i.size, i.indexed_at, i.width, i.height, i.blurhash,
                       (f.id IS NOT NULL) AS is_favorite,
                       f.favorited_at
                FROM images i
                LEFT JOIN favorites f ON i.id = f.id
                WHERE i.id = ?
                """,
                (point_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def maybe_refresh(self) -> bool:
        if time.time() - self._last_refresh <= self.refresh_interval_seconds:
            return False
        self.init_from_qdrant(force=True)
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
        """Unix timestamp of the last successful refresh (init_from_qdrant),
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
        row = self._conn.execute(
            "SELECT 1 FROM images WHERE id = ? LIMIT 1",
            (point_id,),
        ).fetchone()
        return row is not None

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

    def upsert_records(self, points: Any) -> int:
        """Round‑21: insert or replace rows in the local SQLite cache.

        Used by the SyncManager after moving points from
        `images_pending` → `images` in qdrant. Without this, the
        frontend endpoints (`/api/random`, `/api/photo/{id}/raw`,
        etc.) only see the rows from the previous full rebuild —
        newly indexed photos never reach the random deck or the
        thumbnail URL resolver.

        Upsert is keyed on `id`. Existing rows are replaced; new rows
        are inserted. `indexed_at` defaults to the current timestamp
        if missing from the payload.

        Returns the number of rows written. Best‑effort: if any
        row fails, the batch is rolled back and the count is 0.
        """
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for p in points:
            try:
                payload = p.payload if hasattr(p, "payload") else (p.get("payload") or {})
                point_id = str(
                    getattr(p, "id", None) if hasattr(p, "id")
                    else (p.get("id") or payload.get("id") or "")
                )
                path = str(payload.get("path") or "")
                if not point_id or not path:
                    continue
                rows.append({
                    "id": point_id,
                    "path": path,
                    "shard": str(payload.get("shard") or ""),
                    "collection": str(payload.get("collection") or ""),
                    "mtime": _optional_int(payload.get("mtime")),
                    "size": _optional_int(payload.get("size")),
                    "indexed_at": payload.get("indexed_at") or now,
                    "blurhash": str(payload.get("blurhash") or ""),
                    # Round‑30: photo-page dimensions. None when the
                    # source image couldn't be decoded.
                    "width": _optional_int(payload.get("width")),
                    "height": _optional_int(payload.get("height")),
                })
            except (KeyError, TypeError, ValueError):
                # Payload schema drift or malformed point from an
                # older index version. Skip this point rather than
                # crash the whole hydrate; the rest of the batch is
                # still useful.
                continue
        if not rows:
            return 0
        with self._lock:
            try:
                self._conn.executemany(
                    """
                    INSERT INTO images (id, path, shard, collection, mtime, size, indexed_at, blurhash, width, height)
                    VALUES (:id, :path, :shard, :collection, :mtime, :size, :indexed_at, :blurhash, :width, :height)
                    ON CONFLICT(id) DO UPDATE SET
                        path=excluded.path,
                        shard=excluded.shard,
                        collection=excluded.collection,
                        mtime=excluded.mtime,
                        size=excluded.size,
                        indexed_at=excluded.indexed_at,
                        blurhash=excluded.blurhash,
                        width=excluded.width,
                        height=excluded.height
                    """,
                    rows,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                logger.exception("IndexDB.upsert_records failed")
                return 0
        return len(rows)

    def pick_random_rows(
        self, n: int, collections: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Sample N rows from the cache, optionally filtered by collection.

        Returns a list of row dicts (id, path, shard, collection, mtime,
        size, indexed_at, is_favorite, favorited_at, width, height) —
        everything the random page needs to render without going back
        to Qdrant.

        Implementation: random rowids in [MIN(rowid), MAX(rowid)] plus
        WHERE rowid IN (...) + ORDER BY id + LIMIT n. Each rowid
        lookup is an O(log N) index seek, so the whole sample is O(n)
        instead of `ORDER BY RANDOM()`'s O(N log N) full-table sort.
        Some random rowids won't match the collection filter (over-
        fetch handles that), so the returned set is approximately
        uniform but biased toward collections with denser rowids.
        """
        if n <= 0:
            return []
        collections = [c for c in (collections or []) if c]
        select_cols = (
            "i.id, i.path, i.shard, i.collection, i.mtime, i.size, "
            "i.indexed_at, i.width, i.height, i.blurhash, "
            "(f.id IS NOT NULL) AS is_favorite, f.favorited_at, "
            "(d.id IS NOT NULL) AS is_disliked"
        )
        join_sql = (
            "FROM images i "
            "LEFT JOIN favorites f ON i.id = f.id "
            "LEFT JOIN dislikes d ON i.id = d.id"
        )
        with self._lock:
            # Get the rowid range for the filtered set. With no filter
            # this is the whole table.
            if collections:
                placeholders = ",".join("?" for _ in collections)
                bounds = self._conn.execute(
                    f"SELECT MIN(rowid), MAX(rowid) FROM images "  # noqa: S608
                    f"WHERE collection IN ({placeholders})",
                    collections,
                ).fetchone()
            else:
                bounds = self._conn.execute(
                    "SELECT MIN(rowid), MAX(rowid) FROM images"
                ).fetchone()
            min_rid, max_rid = bounds[0], bounds[1]
            if min_rid is None or max_rid is None:
                return []
            # Pick n random rowids in [min_rid, max_rid]. Some won't
            # exist (gaps from deletes) or won't match the collection
            # filter, so over-fetch significantly to amortise that.
            # With a 1.85M-row prod collection that has rowid gaps,
            # `max(n * 3, n + 8)` under-fetches ~5% of the time
            # (returns 19 of 20). Bumped to `n * 10` with a +50 floor
            # to handle small n, plus a 3-attempt retry that picks
            # another `n * 5` rowids each pass. Total worst-case
            # budget is `n * 25` picks — still O(n log N) for the
            # rowid lookups, still much cheaper than ORDER BY RANDOM().
            import random
            results: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            attempts = 0
            while len(results) < n and attempts < 3:
                attempts += 1
                target_count = max(n * 10, n + 50) if attempts == 1 else n * 5
                picked = {
                    random.randint(min_rid, max_rid) for _ in range(target_count)  # noqa: S311
                }
                if not picked:
                    continue
                rid_placeholders = ",".join("?" for _ in picked)
                where_parts = [f"i.rowid IN ({rid_placeholders})"]
                params: list[Any] = [*picked]
                if collections:
                    coll_placeholders = ",".join("?" for _ in collections)
                    where_parts.append(f"i.collection IN ({coll_placeholders})")
                    params.extend(collections)
                # Over-fetch at SQL level by 10x to give the caller
                # buffer for _random_rows_to_results' lazy-liveness
                # filter. With prod's lazy-liveness cache mis-categorising
                # some alive files as dead (NAS mounts sometimes report
                # non-existent for reachable paths), the effective dead
                # rate on first-call is much higher than the underlying
                # true rate — empirical testing on the prod collection
                # showed ~5-10% of fetched rows being dropped post-filter
                # even with 3x over-fetch. 10x is the buffer that
                # actually saturates at 20/20 for the 5-column grid.
                #
                # NO ORDER BY: the random rowid selection already gives
                # us randomness. ORDER BY i.id would sort the sample by
                # ID, which means the lexicographically smallest IDs
                # (which are tiny in number and thus always in any
                # large random sample) always end up at the front of
                # the result set. The frontend dedupes against what's
                # on screen, so a stable prefix of IDs means the same
                # photos keep getting filtered out, and the scroll
                # stops after a few batches instead of exploring the
                # full library.
                sql = f"""
                    SELECT {select_cols}
                    {join_sql}
                    WHERE {' AND '.join(where_parts)}
                    LIMIT ?
                """
                params.append(int(n) * 10)
                rows = self._conn.execute(sql, params).fetchall()
                for row in rows:
                    row_dict = dict(row)
                    row_id = row_dict["id"]
                    if row_id in seen_ids:
                        continue
                    seen_ids.add(row_id)
                    results.append(row_dict)
                    if len(results) >= n:
                        break
        return results


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
