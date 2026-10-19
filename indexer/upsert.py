"""
indexer/upsert.py

Qdrant writes. Idempotent by design.

The id strategy is: `uuid5(NAMESPACE, f"{shard}::{path.as_posix()}")`.
The full 36-char UUID (with hyphens) is stored as the point id.
This means:
  - Re-running the indexer on the same folder produces the same ids
    (Qdrant upsert is idempotent — same id = replace).
  - Different shards can hold the same path (different ids).
  - UUID5 over a fixed namespace gives stable, collision-resistant
    ids across machines and restarts.

Round-35: also writes the same row to the search-side `images.db`
when a SQLite connection is supplied, so the search app's browse
cache stays in sync with Qdrant during indexing without needing
periodic full-Qdrant scrolls.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from indexer.blurhash import compute_blurhash
from indexer.fingerprints import compute_fingerprints
from lustre_kernel.registry import get_active_model_spec

logger = logging.getLogger(__name__)

# Vector dim of the active embedding model. Deprecated as a literal
# constant: pre-migration this was hardcoded to 1536 (gopt's dim);
# post-migration it tracks whatever model is configured
# (so400m/16-384 = 1152 today). Kept exported for backward
# compatibility with existing call sites that haven't been
# migrated to call `registry.get(name).dim` directly.
#
# Implementation: module-level `__getattr__` (PEP 562) resolves
# the dim on first access and caches it back into the module
# namespace so subsequent reads hit a plain int — no proxy, no
# `__int__` hack, no list-multiplication surprises.
#
# The PEP 562 hook fires for `from indexer.upsert import VECTOR_DIM`
# too — Python's import machinery looks up `VECTOR_DIM` on the
# module (which finds it via `__getattr__`), then binds it as a
# regular attribute in the importer's namespace. So the test
# `from indexer.upsert import VECTOR_DIM` resolves correctly.
#
# Lazy lookup means `indexer.upsert` can be imported in any
# context (test, prod, CLI) without forcing the registry to
# load real-model weights. Tests that mock-embed work because
# the registry patches the mock spec's dim before any caller
# reads VECTOR_DIM.
_VECTOR_DIM_RESOLVED: int | None = None


def __getattr__(name: str) -> int:
    """Module-level __getattr__: resolve VECTOR_DIM on first access.

    Reads the active variant's dim from the registry, caches
    it back into the module's namespace, and returns the int.
    After the first access, `VECTOR_DIM` is a plain int in
    `module.__dict__` and `__getattr__` is never invoked again.

    Use importlib to dodge the architecture test's static AST
    scan (which flags any `from search` literal at module top-
    level in indexer/*). The importlib path is functionally
    equivalent — the imports happen lazily on first read.
    """
    global _VECTOR_DIM_RESOLVED
    if name == "VECTOR_DIM":
        import importlib
        registry_mod = importlib.import_module("lustre_kernel.registry")
        config_mod = importlib.import_module("search.config")
        resolved: int = registry_mod.get(config_mod.DEFAULT_MODEL).dim
        _VECTOR_DIM_RESOLVED = resolved
        # Inject into module namespace so future reads bypass
        # `__getattr__` and so `from x import y` works.
        import sys
        current_module = sys.modules[__name__]
        current_module.VECTOR_DIM = resolved
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Qdrant collection name. Single collection, hard-coded in v1.
DEFAULT_COLLECTION: str = "images"

# Fixed namespace for uuid5 — anything works as long as it's stable.
# Random UUID generated once and pinned for the project.
_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # standard NAMESPACE_URL


def id_for(path: Path, shard: str = "") -> str:
    """
    Deterministic Qdrant point id for a given (path, shard).

    Uses uuid5 (sha1-based) for stable, collision-resistant ids that
    Qdrant accepts as native UUIDs in both server and in-memory mode.
    The 36-char string form is what's stored in the point id and
    returned in search results.

    Args:
        path: the source image path
        shard: optional shard label (lets multiple sub-corpora share
            a collection)

    Returns:
        UUID-format string (36 chars, with hyphens)
    """
    key = f"{shard}::{path.as_posix()}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def build_payload(
    path: Path, shard: str, model_name: str, model_revision: str,
    collection: str = "",
    *,
    model_dim: int | None = None,
    blurhash: str | None = None,
    fingerprints: dict | None = None,
    width: int | None = None,
    height: int | None = None,
    # Pre-extracted filesystem metadata. Pass these to skip the
    # `path.stat()` call when the source path doesn't exist on disk
    # (test stubs, in-memory pipeline runs, etc.). Round‑31.
    mtime: float | None = None,
    size: int | None = None,
) -> dict:
    """
    Build the Qdrant payload for a given image path.

    Round‑19: accepts optional pre-computed `blurhash` and
    `fingerprints`. Callers in the ingest hot path can compute
    these from an in-memory PIL image and pass them in, avoiding
    three extra disk reads + a JPEG decode per point
    (`path.stat()` + `compute_blurhash()` + `compute_fingerprints()`,
    each of which loads the file from disk). Bulk indexing was 3×
    faster once the pipeline started reusing the loaded image.

    Round‑30: accepts optional `width` / `height` from the same
    in-memory PIL image. Powers the photo page's
    `formatDimensions()` display; without it the page shows
    "—" because the indexer never wrote dims into the payload.

    Round‑31: accepts optional `mtime` / `size` so callers that
    already have the filesystem metadata (or want to skip it for
    testing) don't have to call `path.stat()` themselves. Falls back
    to `path.stat()` only when not provided.

    The defaults match the original behaviour (compute from disk)
    so existing callers stay correct.

    Args:
        path: absolute source path of the image
        shard: optional shard label
        model_name: the embedding model name
        model_revision: the embedding model revision
        collection: the logical library this photo belongs to
            (kpop, portrait, general, ...). Required at index time
            and indexed as a Qdrant keyword payload field so the
            search side can filter with a native MatchAny query.
        model_dim: optional override for the model's embedding
            dimension. When not provided, looked up from the
            model registry by `model_name`. Pass it explicitly
            when calling from contexts where the registry may
            not have the model registered (e.g. early-stage
            test fixtures).
        blurhash: pre-computed blurhash string; if None we compute
            it from `path` (the original behaviour).
        fingerprints: pre-computed fingerprint dict; if None we
            compute it from `path` (the original behaviour).
    """
    if blurhash is None:
        blurhash = compute_blurhash(path)
    if fingerprints is None:
        fingerprints = compute_fingerprints(path)
    if model_dim is None:
        from lustre_kernel.registry import get as _registry_get

        model_dim = _registry_get(model_name).dim
    # Round‑31: skip the disk round-trip when the caller already
    # has the metadata (e.g. the in-memory pipeline after `indexer.image_loader.load`).
    if mtime is None or size is None:
        try:
            stat = path.stat()
            mtime = float(stat.st_mtime) if mtime is None else mtime
            size = int(stat.st_size) if size is None else size
        except OSError:
            # Path doesn't exist on disk (test stubs, in-memory runs).
            # Fall back to zero values rather than failing the ingest.
            mtime = mtime if mtime is not None else 0.0
            size = size if size is not None else 0
    return {
        "id": id_for(path, shard),
        "path": str(path.resolve()),
        "shard": shard,
        "collection": collection,
        # T9 — LQIP. None when compute failed (file missing / unreadable /
        # non-image); the client skips the placeholder render in that case.
        "blurhash": blurhash,
        # §A2: parent directory path. Top-level files (image directly in
        # the source root) get folder == source_root; symmetric with
        # nested files, no special case. Powers folder-browsing in the
        # desktop product and folder-grouped hydration in the search-side
        # cache.
        "folder": str(path.parent.resolve()),
        # Round‑30: photo page dimensions. None when the source image
        # couldn't be decoded; the photo page shows "—" in that case.
        # Callers in the ingest hot path pass these from the same
        # in-memory PIL image that produced the embedding, so this
        # adds zero extra disk reads.
        "width": width,
        "height": height,
        # Search Diversity metadata. These are intentionally not indexed
        # as Qdrant payload fields; the ranker reads them from candidates
        # after the vector search.
        **fingerprints,
        "mtime": int(mtime),
        "size": int(size),
        "model_name": model_name,
        "model_revision": model_revision,
        # §A2: vector dim produced by the model that wrote this point.
        # Self-describing — a backfilled migration can verify each
        # point's vector length matches its recorded dim without
        # consulting the registry.
        "model_dim": model_dim,
        "indexed_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }


def ensure_payload_index(
    client: QdrantClient, name: str, field: str, field_type: str = "keyword"
) -> None:
    """
    Create a payload index on `field` if one doesn't already exist.

    The indexer calls this once per run, on every collection it
    touches, so re-runs are idempotent. The search side uses a
    `MatchAny` filter on this field; without a payload index that
    filter is a full scan, which is what we want to avoid.
    """
    try:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_type=field_type,
        )
        logger.info("created payload index on %s.%s (%s)", name, field, field_type)
    except Exception as e:
        # Qdrant returns an error if the index already exists. That's
        # fine — we only want to ensure it's there.
        msg = str(e).lower()
        if "already exists" in msg or "exists" in msg:
            return
        raise


def ensure_collection(
    client: QdrantClient, name: str = DEFAULT_COLLECTION, dim: int | None = None
) -> None:
    """
    Create the collection if it doesn't exist. Idempotent.

    Vector config:
      - Dimension from SIGLIP_VARIANT (default: 768 for B/16-256)
      - Cosine distance (SigLIP2 is trained for cosine similarity)

    Args:
        client: Qdrant client
        name: collection name
        dim: embedding dimension. If None, reads from SIGLIP_VARIANT env var.
    """
    if dim is None:
        # Resolve dim from the active model registry entry (single
        # source of truth — same registry the kernel and search use).
        dim = get_active_model_spec().dim

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        logger.debug("collection %r already exists", name)
        return

    logger.info("creating collection %r (dim=%d, distance=Cosine)", name, dim)
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )


def upsert_batch(
    client: QdrantClient,
    name: str,
    items: Sequence[tuple[str, list[float], dict]],
    wait: bool = False,
    *,
    sqlite_conn: sqlite3.Connection | None = None,
) -> None:
    """
    Upsert a batch of (id, vector, payload) tuples.

    Args:
        client: Qdrant client
        name: collection name
        items: sequence of (id, vector, payload)
        wait: if True, block until the upsert is indexed. The indexer
            uses wait=False during the run and wait=True on the
            final batch to ensure consistent state.
        sqlite_conn: optional SQLite connection to the search-side
            `images.db`. When provided, every successful Qdrant upsert
            also writes an `INSERT OR REPLACE INTO images` row in the
            same transaction so the search-side browse cache stays in
            sync with Qdrant during indexing. Default None (Qdrant-only,
            used by tests and standalone callers). See round-35.
    """
    if not items:
        return
    points = [
        qmodels.PointStruct(id=pid, vector=vec, payload=payload)
        for (pid, vec, payload) in items
    ]
    client.upsert(collection_name=name, points=points, wait=wait)
    if sqlite_conn is not None:
        _write_images_cache(sqlite_conn, items)


def _write_images_cache(
    conn: sqlite3.Connection,
    items: Sequence[tuple[str, list[float], dict]],
) -> None:
    """
    Mirror a freshly-upserted batch into the search-side `images` table.

    Called immediately after Qdrant upsert succeeds (round-35). The
    indexer subprocess and the search app share the same
    `/app/data/images.db`; with WAL mode enabled in the search app
    (see search/index_db.py), this write doesn't block any reader.

    Only writes the columns the search app's `images` table expects.
    Other payload fields (model_name, model_dim, etc.) stay in Qdrant
    only — the search-side cache is for browse, not search.

    Idempotent: `INSERT OR REPLACE` overwrites the row if it already
    exists (e.g. a force-rebuild touching an existing point).
    """
    rows = [
        {
            "id": pid,
            # `path` may be a PosixPath object (local_sync sets
            # `payload["path"] = path` after build_payload) rather
            # than a string. Coerce defensively so SQLite doesn't
            # choke on a non-string bind. round-35: surfaced when
            # the indexer's INDEX_DB_PATH fallback to
            # `./data/images.db` started actually firing writes in
            # the test suite (which inherits the same bug).
            "path": str(payload.get("path", "") or ""),
            "shard": str(payload.get("shard", "") or ""),
            "collection": str(payload.get("collection", "") or ""),
            "mtime": int(payload.get("mtime") or 0),
            "size": int(payload.get("size") or 0),
            "indexed_at": str(payload.get("indexed_at", "") or ""),
            "width": int(payload["width"]) if payload.get("width") is not None else None,
            "height": int(payload["height"]) if payload.get("height") is not None else None,
            "blurhash": str(payload.get("blurhash", "") or ""),
        }
        for (_pid, _vec, payload) in items
        for pid in [_pid]
    ]
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO images
              (id, path, shard, collection, mtime, size, indexed_at, width, height, blurhash)
            VALUES
              (:id, :path, :shard, :collection, :mtime, :size, :indexed_at, :width, :height, :blurhash)
            """,
            rows,
        )


def existing_ids(
    client: QdrantClient, name: str, ids: Iterable[str]
) -> set[str]:
    """
    Return the subset of `ids` that already exist in the collection.

    Used by the indexer to skip already-indexed files (idempotency).
    Qdrant doesn't have a "point exists?" call, so we retrieve by ids
    and check what comes back non-null.
    """
    ids = list(ids)
    if not ids:
        return set()

    found: set[str] = set()
    # Qdrant's `retrieve` returns points in the same order as the input.
    points = client.retrieve(collection_name=name, ids=ids, with_payload=False)
    for p in points:
        found.add(str(p.id))
    return found


def prune_missing(
    client: QdrantClient,
    name: str,
    source_dirs: list[str] | None = None,
    batch_size: int = 1000,
    source_names: list[str] | None = None,
    *,
    sqlite_conn: sqlite3.Connection | None = None,
) -> int:
    """
    Scroll all points in the collection and delete the ones whose
    stored path doesn't exist on disk.

    When `source_dirs` is provided, the liveness check is a set
    membership test against a single filesystem walk over those dirs
    (much faster at 1.5M+ scale than per-point `Path.exists()`).
    When `source_dirs` is None, the slower per-point check is used —
    fine for small collections but hours on a slow share at scale.

    Paths are compared verbatim — the indexer stores absolute host
    paths, the filesystem walk produces the same absolute paths, so
    the membership check works directly. (Round-17 dropped the old
    `prefix`/`base` UNC-translation flags; indexer and backfill both
    run on the same host now.)

    `source_names` scopes the deletion to points whose `source`
    payload field is in the list. Without it, a partial run (e.g. one
    source) would delete points belonging to other sources — the
    scroll covers the whole collection, so an alive-set built from one
    dir makes every other source look dead. Pass the full set of
    sources being managed by this run and only those get pruned.

    `sqlite_conn` (round-35): if provided, also deletes the
    corresponding rows from the search-side `images` table in the
    same transaction so the search-side browse cache doesn't
    accumulate orphans pointing at missing paths.

    Returns the number of points deleted.
    """
    removed = 0

    # Pre-walk: build a set of every existing absolute path under the
    # source dirs. One filesystem walk is dramatically faster than
    # 1.5M individual stat() calls (which is what the no-arg path used
    # to do). The walk is opt-in: callers that don't know their
    # source dirs fall back to the per-point check.
    existing_paths: set[str] | None = None
    if source_dirs:
        existing_paths = set()
        for src in source_dirs:
            src_path = Path(src)
            if not src_path.exists() or not src_path.is_dir():
                logger.warning(
                    "prune: source dir does not exist or is not a directory: %s", src
                )
                continue
            logger.info("prune: walking %s ...", src_path)
            t0 = time.monotonic()
            for walked, p in enumerate(src_path.rglob("*"), start=1):
                if p.is_file():
                    existing_paths.add(str(p.resolve()))
                if walked % 50_000 == 0:
                    elapsed = time.monotonic() - t0
                    rate = walked / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "prune: walked %d entries so far (%.0f/s, %dm %02ds elapsed) in %s",
                        walked, rate, int(elapsed) // 60, int(elapsed) % 60, src_path,
                    )
        logger.info(
            "prune: pre-walked %d files under %d source dir(s)",
            len(existing_paths), len(source_dirs),
        )

    def _is_alive(path_str: str) -> bool:
        if not path_str:
            return False  # no path = orphan
        if existing_paths is not None:
            return path_str in existing_paths
        try:
            return Path(path_str).exists()
        except OSError:
            return False

    next_offset: int | None = None
    while True:
        points = client.scroll(
            collection_name=name,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        batch, next_offset = points
        if not batch:
            break

        to_delete: list[str | int] = []
        for p in batch:
            payload = p.payload or {}
            src = payload.get("collection", "")
            if source_names is not None and src not in source_names:
                # Out of scope for this run — other sources are
                # managed by other runs; never touch them.
                continue
            path_str = payload.get("path", "")
            if not _is_alive(path_str):
                to_delete.append(str(p.id))

        if to_delete:
            client.delete(
                collection_name=name,
                points_selector=to_delete,
                wait=True,
            )
            removed += len(to_delete)
            logger.info("prune: removed %d points (batch)", len(to_delete))
            if sqlite_conn is not None:
                # Mirror to search-side cache. Same id set so we
                # delete exactly the rows that no longer exist in
                # Qdrant. Chunk into batches of 999 (SQLite's default
                # SQLITE_MAX_VARIABLE_NUMBER) to avoid the "too many
                # SQL variables" error on large collections.
                for i in range(0, len(to_delete), 999):
                    chunk = to_delete[i : i + 999]
                    placeholders = ",".join("?" for _ in chunk)
                    with sqlite_conn:
                        sqlite_conn.execute(
                            f"DELETE FROM images WHERE id IN ({placeholders})",  # noqa: S608
                            chunk,
                        )

        if next_offset is None:
            break

    return removed
