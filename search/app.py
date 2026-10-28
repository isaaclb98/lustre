"""
search/app.py — FastAPI factory.

Routes, request/response models, and startup/shutdown wiring live here.
The app is a pure JSON API; the SvelteKit frontend consumes it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, cast

import zipstream  # streaming ZIP writer for /favorites/download.zip
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from lustre_kernel.qdrant_url import client_kwargs as _qdrant_client_kwargs
from search import config, text_encoder
from search.centroids import (
    CentroidStore,
    DynamicCentroidRegistry,
    DynamicCentroidSpec,
    blend_centroids,
)
from search.diversity import (
    DiversityResultCache,
    DiversityStats,
)
from search.image_resolver import guess_content_type, resolve_local, resolve_url
from search.index_db import DEFAULT_INDEX_DB_PATH, IndexDB
from search.models import (
    DiversityMetadata,
    PromptState,
    SearchResult,
)
from search.qdrant_client import QdrantSearch, SearchHit
from search.random import RandomPicker
from search.routers.for_you import invalidate_for_you_cache as _for_you_invalidate_deck
from search.search_snapshot import SearchSnapshotCache

logger = logging.getLogger(__name__)

# Module-level state for the dynamic centroid registry. Initialised
# by `create_app` (or reset by `reset_for_tests`). Tests reach in
# to inspect cache state and to force invalidation; the
# `_invalidate_likes_centroid` helper below reads this.
_dynamic_centroids = None  # type: DynamicCentroidRegistry | None

# ----- Lazy path-liveness cache (dual-store sync, defense in depth) -----
# Maps absolute path → (exists, expires_at_monotonic). Populated on
# first miss; subsequent checks within the TTL return the cached
# value without touching the filesystem. Bounds the per-request
# stat() cost of the lazy-validation check (`_is_path_alive` below).
# Reads beyond the TTL re-stat. Negative results are cached too —
# a freshly-deleted file shows up as dead within the TTL window.
_path_liveness_cache: dict[str, tuple[bool, float]] = {}
# Memory cap for the lazy liveness cache. At 1.5M points with a
# 60s TTL the steady-state cache stays well under this; the cap is
# a defence against pathological workloads (long-running container
# with millions of distinct paths). Eviction drops the oldest 10%
# of entries by insertion order, which is O(n) on the dict. Bumping
# the cap trades memory for fewer re-stats.
_PATH_LIVENESS_CACHE_MAX: int = 50000


def _is_path_alive(path: str) -> bool:
    """Lazy liveness check for a photo's on-disk path.

    Cached for `_cfg.path_liveness_ttl_seconds` (default 60s). On a
    miss, calls `Path(path).exists()` once and caches the result.
    Both alive and dead outcomes are cached — a deleted file stays
    cached as dead for the TTL window.

    Used by the read paths (/random, /albums/{id}/members, /photo/{id},
    /photo/{id}/raw) to silently skip dead photos without waiting
    for the periodic IndexDB refresh to catch up.

    The TTL is short enough that the cache reflects filesystem
    state within a minute; it's also long enough that a single page
    load (50+ photos) hits `Path.exists()` once, not 50+ times.
    """
    import time as _time
    from pathlib import Path as _Path

    if not path:
        return False
    now = _time.monotonic()
    cached = _path_liveness_cache.get(path)
    if cached is not None:
        exists, expires_at = cached
        if now < expires_at:
            return exists
    try:
        exists = _Path(path).exists()
    except OSError:
        # Permission error, dangling symlink, etc. Treat as dead
        # so the caller skips it. The periodic refresh + scheduled
        # prune will eventually clean up the IndexDB row.
        exists = False
    if _cfg is None:  # noqa: SIM108 — if/else is clearer than ternary here
        # Defensive: if config isn't loaded yet, fall back to a
        # conservative TTL. The real TTL gets applied once create_app
        # finishes wiring things up.
        ttl = 60.0
    else:
        ttl = _cfg.path_liveness_ttl_seconds
    _path_liveness_cache[path] = (exists, now + ttl)
    # Evict the oldest 10% if we've blown past the cap. Cheap;
    # re-stats on next read. Defends against pathological workloads
    # (long-running container with millions of distinct paths).
    if len(_path_liveness_cache) > _PATH_LIVENESS_CACHE_MAX:
        evict_count = _PATH_LIVENESS_CACHE_MAX // 10
        for _ in range(evict_count):
            try:
                _path_liveness_cache.pop(next(iter(_path_liveness_cache)))
            except StopIteration:
                break
    return exists


def _is_row_alive(row: dict[str, Any] | None) -> bool:
    """True if the row's `path` field is currently alive on disk.

    Defensive wrapper: a None row is dead. Otherwise delegates to
    `_is_path_alive` which caches the filesystem check.
    """
    if not row:
        return False
    return _is_path_alive(str(row.get("path") or ""))


class _ResizeError(Exception):
    """Raised when the Lanczos-resize path can't produce a cached file."""


def _photo_cache_dir() -> Path:
    """Return (and create) the disk cache dir for resized photos.

    Lives under `cfg.photo_cache_dir` if configured, else under
    `Path(_cfg.index_db_path).parent / "photo_cache"` so the cache
    sits on the same volume as the IndexDB. In the cluster the
    IndexDB is on `/app/data` (PVC), so the cache survives pod
    restarts and can grow to whatever the PVC allows.

    Falls back to `<cwd>/photo_cache` if `index_db_path` is
    `:memory:` (test mode) or otherwise has no parent directory.
    """
    cfg = _cfg or config.load()
    override = getattr(cfg, "photo_cache_dir", None)
    if override:
        base = Path(override)
    else:
        idx = cfg.index_db_path
        if idx and idx != ":memory:":
            parent = Path(idx).parent
            if str(parent) and str(parent) not in ("", "/"):
                base = parent / "photo_cache"
            else:
                base = Path.cwd() / "photo_cache"
        else:
            base = Path.cwd() / "photo_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resize_cached(point_id: str, width: int, source: Path) -> Path:
    """Return a Lanczos-resized JPEG for `(point_id, width)`.

    Cache layout: `<cache>/<point_id>/w<width>.jpg`. The cache file
    is keyed off the source's mtime+size, so re-encoding the
    source on disk produces a new cache entry on the next request
    (the old one is left behind — harmless, just stale).

    Lanczos is the highest-quality downsampling filter PIL ships
    with; it's a 3-lobed Lanczos variant that visually beats the
    browser's default scaling for photos that are downsized by 2x
    or more (which is most of them, given that a 12 MP source
    lands in a 1920px-wide lightbox).
    """
    from PIL import Image, ImageOps

    target = _photo_cache_dir() / point_id / f"w{width}.jpg"
    src_stat = source.stat()
    cache_marker = target.with_suffix(".src.json")
    # Cache hit check: file exists AND the source fingerprint
    # matches what we last resized from.
    if target.exists() and cache_marker.exists():
        try:
            meta = json.loads(cache_marker.read_text())
            if (
                meta.get("st_mtime_ns") == src_stat.st_mtime_ns
                and meta.get("st_size") == src_stat.st_size
            ):
                return target
        except (json.JSONDecodeError, OSError):
            pass  # stale marker — fall through to re-resize

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(source) as im:
            # EXIF orientation: many phone photos are stored
            # landscape with a rotation flag rather than the
            # rotated pixels. Without this fix-up, a portrait
            # shot would display sideways in the lightbox.
            im = cast(Any, ImageOps.exif_transpose(im))
            im.load()  # force decode so resize sees real pixels
            # Convert non-JPEG sources to RGB for JPEG output.
            if im.mode not in ("RGB", "L"):
                im = cast(Any, im.convert("RGB"))
            # Only downscale; never upscale. If the source is
            # smaller than the requested width, copy through.
            if im.width > width:
                im = im.resize(
                    (width, max(1, round(im.height * width / im.width))),
                    Image.Resampling.LANCZOS,
                )
            # quality=90 is a sensible default — visible quality
            # is indistinguishable from 95 in side-by-side tests,
            # but the file is ~30% smaller.
            im.save(target, format="JPEG", quality=90, optimize=True)
    except (OSError, ValueError) as exc:
        # Surface as _ResizeError so the caller falls back to the
        # original file. Common cause: a HEIC the runtime can't
        # decode, or a corrupt JPEG header.
        raise _ResizeError(str(exc)) from exc

    cache_marker.write_text(
        json.dumps(
            {"st_mtime_ns": src_stat.st_mtime_ns, "st_size": src_stat.st_size}
        )
    )
    return target


def _coerce_view(raw: str | None) -> str:
    """Return 'grid' or 'feed' from a query-param value; fallback to default."""
    if _cfg is None:
        return "grid"
    if raw and raw in _cfg.valid_views:
        return raw
    return _cfg.default_view


def _make_favourites_centroid_spec(
    qdrant: Any, index_db: Any,
) -> DynamicCentroidSpec:
    """Build the registration spec for the favourites dynamic centroid.

    Round‑29b: superseded by `_make_likes_centroid_spec` (the album
    was renamed from Favourites to Likes). This one is still
    registered as a back‑compat alias so old
    `?centroid=favourites` URLs keep working.
    """
    import numpy as np

    def compute() -> tuple[list[float], int, list[str]] | None:
        fav_ids = index_db.list_favorite_ids()
        if not fav_ids:
            return None
        pairs = qdrant.retrieve_batch_with_vectors(fav_ids)
        if not pairs:
            return None
        vectors = np.asarray(
            [v for _, v in pairs if v], dtype=np.float32,
        )
        if vectors.size == 0:
            return None
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return None
        centroid = (centroid / norm).tolist()
        seed_ids = [pid for pid, _ in pairs]
        return (centroid, len(seed_ids), seed_ids)

    return DynamicCentroidSpec(
        name="favourites",
        label="Favourites (legacy alias)",
        description=(
            "Mean of every liked photo's embedding, re-normalised "
            "to unit length. Back‑compat alias for the new 'likes' "
            "centroid."
        ),
        compute_fn=compute,
        source="favourites",
        empty_message=(
            "Like a few photos first — this centroid is built "
            "from your likes list."
        ),
    )


def _make_likes_centroid_spec(
    qdrant: Any, index_db: Any,
) -> DynamicCentroidSpec:
    """Build the registration spec for the likes dynamic centroid.

    Round‑29b: renamed from `favourites` to `likes` to match the
    user‑visible album label. Same compute shape — just a
    different registry key. The old `favourites` name remains
    registered as a back‑compat alias; see
    `_make_favourites_centroid_spec`.
    """
    import numpy as np

    def compute() -> tuple[list[float], int, list[str]] | None:
        fav_ids = index_db.list_favorite_ids()
        if not fav_ids:
            return None
        pairs = qdrant.retrieve_batch_with_vectors(fav_ids)
        if not pairs:
            return None
        vectors = np.asarray(
            [v for _, v in pairs if v], dtype=np.float32,
        )
        if vectors.size == 0:
            return None
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return None
        centroid = (centroid / norm).tolist()
        seed_ids = [pid for pid, _ in pairs]
        return (centroid, len(seed_ids), seed_ids)

    return DynamicCentroidSpec(
        name="likes",
        label="Likes",
        description=(
            "Mean of every liked photo's embedding, "
            "re-normalised to unit length. Updates as you like."
        ),
        compute_fn=compute,
        source="likes",
        empty_message=(
            "Like a few photos first — this centroid is built "
            "from your likes list."
        ),
    )


def _make_dislikes_centroid_spec(
    qdrant: Any, index_db: Any,
) -> DynamicCentroidSpec:
    """Build the registration spec for the dislikes dynamic centroid.

    Round‑29: mirror of `_make_likes_centroid_spec` so the
    Search button on the Dislikes album card has a real centroid
    to query. Without this, /api/centroids/dislikes/search 404s.

    The compute path is identical to favourites (mean of member
    vectors, L2-normalised) — only the source table differs.
    """
    import numpy as np

    def compute() -> tuple[list[float], int, list[str]] | None:
        dislike_ids = index_db.list_dislike_ids()
        if not dislike_ids:
            return None
        pairs = qdrant.retrieve_batch_with_vectors(dislike_ids)
        if not pairs:
            return None
        vectors = np.asarray(
            [v for _, v in pairs if v], dtype=np.float32,
        )
        if vectors.size == 0:
            return None
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return None
        centroid = (centroid / norm).tolist()
        seed_ids = [pid for pid, _ in pairs]
        return (centroid, len(seed_ids), seed_ids)

    return DynamicCentroidSpec(
        name="dislikes",
        label="Dislikes",
        description=(
            "Mean of every disliked photo's embedding, "
            "re-normalised to unit length. Updates as you dislike."
        ),
        compute_fn=compute,
        source="dislikes",
        empty_message=(
            "Dislike a few photos first — this centroid is built "
            "from your dislikes list."
        ),
    )


def _invalidate_likes_centroid() -> None:
    """Drop the cached likes centroid so the next read recomputes.

    Round‑29b: renamed from `favourites` to `likes` to match the
    user-visible album label. Old name is preserved as a back-compat
    alias (see the `_register_likes_centroid_under_old_name` call in
    create_app) so legacy `?centroid=favourites` URLs still work.

    Invalidates both names so callers that check the legacy alias
    (e.g. `cached_n_images('favourites')`) see the cache drop too.
    """
    if _dynamic_centroids is not None:
        _dynamic_centroids.invalidate("likes")
        _dynamic_centroids.invalidate("favourites")


def _invalidate_dislikes_centroid() -> None:
    """Drop the cached dislikes centroid so the next read recomputes.

    Round‑29: mirror of `_invalidate_likes_centroid`, wired
    into mark_dislike / unmark_dislike so the centroid stays in
    sync with the dislikes table.
    """
    if _dynamic_centroids is not None:
        _dynamic_centroids.invalidate("dislikes")


# ---------------------- Album centroid helpers ----------------------
#
# Each user album auto-registers as a dynamic centroid under the
# name `album:{id}`. The id (not the name) is the key so renames
# don't break existing search/Discover references that pass the
# centroid name to /api/search. The compute closure is essentially
# the same shape as favourites — mean of L2-normalised member
# vectors — so the existing DynamicCentroidSpec machinery handles
# caching, lazy compute, and invalidation for free.

def _album_centroid_name(album_id: int) -> str:
    """Centroid registry key for an album. Stable across renames."""
    return f"album:{album_id}"


def _make_album_centroid_spec(
    qdrant: Any, index_db: Any, album_id: int,
) -> DynamicCentroidSpec:
    """Build a DynamicCentroidSpec for an album.

    Reads the album name lazily inside `compute` so a rename
    surfaces in the next centroid read. The spec's `name` stays
    the id-keyed form (`album:{id}`) — the user-visible label is
    refreshed at every compute call.
    """
    import numpy as np

    def compute() -> tuple[list[float], int, list[str]] | None:
        # Snapshot ids first so a concurrent membership change
        # doesn't give us a torn view.
        ids = index_db.list_album_member_ids(album_id)
        if not ids:
            return None
        pairs = qdrant.retrieve_batch_with_vectors(ids)
        if not pairs:
            return None
        vectors = np.asarray(
            [v for _, v in pairs if v], dtype=np.float32,
        )
        if vectors.size == 0:
            return None
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return None
        centroid = (centroid / norm).tolist()
        # Same orphan-stripping pattern as favourites: seed ids
        # are the ids that actually returned vectors. See
        # `_make_favourites_centroid_spec` for the rationale.
        seed_ids = [pid for pid, _ in pairs]
        return (centroid, len(vectors), seed_ids)

    # The static `label` is set at registration time; if the user
    # renames the album, the label goes stale until re-registration.
    # We accept that trade-off because the registry stores one
    # spec per centroid name and the spec doesn't currently support
    # a "label_fn". Renames are infrequent; the stale label is a
    # cosmetic issue, not a correctness one (the centroid itself
    # is always computed fresh from current membership).
    album = index_db.get_album(album_id)
    label = f"Album: {album['name']}" if album else f"Album {album_id}"

    return DynamicCentroidSpec(
        name=_album_centroid_name(album_id),
        label=label,
        description=(
            f"Mean of every photo in album #{album_id}'s embedding, "
            "re-normalised to unit length. Updates as membership "
            "changes."
        ),
        compute_fn=compute,
        source=f"album:{album_id}",
        empty_message=(
            "Add photos to this album to build its centroid."
        ),
    )


def _register_album_centroid(album_id: int) -> None:
    """Register (or re-register) an album's centroid.

    Called from POST /api/albums (new album) and from
    PATCH /api/albums/{id} (rename) so the user-visible label
    refreshes. Idempotent — calling on an already-registered id
    replaces the spec.

    Uses the module-level `_qdrant` / `_index_db` globals set by
    `create_app` so the closure captures the live instances (same
    pattern as `_make_favourites_centroid_spec`). If called
    before create_app has run (e.g. in a test that exercises the
    helpers directly), the lookups return None and we silently
    no-op — the actual registration is wired through the HTTP
    endpoints which always run inside create_app's scope.
    """
    if _dynamic_centroids is None or _qdrant is None or _index_db is None:
        return
    _dynamic_centroids.register(_make_album_centroid_spec(
        _qdrant, _index_db, album_id,
    ))


def _unregister_album_centroid(album_id: int) -> None:
    """Remove an album's centroid from the registry entirely.

    Called from DELETE /api/albums/{id} so the deleted album
    stops appearing on `/centroids` and in `/api/centroids`
    immediately, without waiting for a process restart. Drops
    the spec, any cached value, and any dirty flag in one shot.
    """
    if _dynamic_centroids is None:
        return
    _dynamic_centroids.unregister(_album_centroid_name(album_id))


def _invalidate_album_centroid(album_id: int) -> None:
    """Drop the cached album centroid so the next read recomputes.

    Wired into add_album_member / remove_album_member. Same
    pattern as favourites: invalidate on every membership change
    so the next search reflects the new set.
    """
    if _dynamic_centroids is not None:
        _dynamic_centroids.invalidate(_album_centroid_name(album_id))


HERE = Path(__file__).parent
# Module-level sentinel for FastAPI `Query([])` default — using a literal
# list in a default arg would call `Query()` once at import time, which
# ruff B008 forbids. Use `None` as the default and resolve to a fresh
# list inside the handler.
_EMPTY_COLLECTIONS: list[str] = []

# ---------------------- Qdrant client wiring ----------------------

# Module-level references, set by create_app(). Tests can replace
# them with mocks before calling the test client.
_qdrant: QdrantSearch | None = None
_cfg: config.Config | None = None
_centroid_store: CentroidStore | None = None
_index_db: IndexDB | None = None


def get_qdrant() -> QdrantSearch:
    if _qdrant is None:
        raise RuntimeError("QdrantSearch not initialized — call create_app() first")
    return _qdrant


def get_cfg() -> config.Config:
    if _cfg is None:
        raise RuntimeError("Config not initialized — call create_app() first")
    return _cfg


def get_centroid_store() -> CentroidStore:
    if _centroid_store is None:
        raise RuntimeError("CentroidStore not initialized — call create_app() first")
    return _centroid_store


def get_index_db() -> IndexDB:
    if _index_db is None:
        raise RuntimeError("IndexDB not initialized — call create_app() first")
    return _index_db


def reset_for_tests() -> None:
    """Drop module state so the next create_app() rebuilds. Test-only."""
    global _qdrant, _cfg, _centroid_store, _dynamic_centroids, _index_db
    if _index_db is not None:
        with suppress(Exception):
            _index_db.close()
    _qdrant = None
    _cfg = None
    _centroid_store = None
    _dynamic_centroids = None  # DynamicCentroidRegistry, set in create_app
    _index_db = None
    text_encoder.reset_encoder_for_tests()


# ---------------------- App factory ----------------------


def create_app(
    cfg: config.Config | None = None,
    qdrant: QdrantSearch | None = None,
    index_db: IndexDB | None = None,
) -> FastAPI:
    """
    Build a FastAPI app with all routes wired.

    Args:
        cfg: pre-loaded config (defaults to config.load())
        qdrant: pre-built QdrantSearch (defaults to one built from cfg)
        index_db: pre-built IndexDB (defaults to one built from cfg)
    """
    global _qdrant, _cfg, _index_db
    _cfg = cfg or config.load()
    logging.basicConfig(level=_cfg.log_level)

    if qdrant is None:
        client = QdrantClient(**_qdrant_client_kwargs(
            url=_cfg.qdrant_url,
            api_key=_cfg.qdrant_api_key,
            timeout=_cfg.query_timeout_ms // 1000,  # int seconds
            prefer_grpc=_cfg.qdrant_prefer_grpc,
            grpc_port=_cfg.qdrant_grpc_port,
        ))
        qdrant = QdrantSearch(
            client=client,
            collection=_cfg.qdrant_collection,
            timeout_ms=_cfg.query_timeout_ms,
            recommend_timeout_ms=_cfg.recommend_timeout_ms,
        )
        # Ensure the payload indexes the search API needs exist on
        # the canonical collection. The indexer creates them too
        # on every run, but a fresh collection (dev, fresh prod,
        # after `--rebuild`) needs this bootstrap before any
        # `/api/collections` call lands.
        # Idempotent; safe to call on every startup.
        qdrant.ensure_payload_index("collection")
    _qdrant = qdrant

    # Build the centroid store from config and load it now so the
    # routes can serve from a populated store. `load()` is total —
    # a missing dir, a corrupt file, or a model/dim mismatch all
    # result in skipping that file (or an empty store), never a
    # crash. Centroids are read-only on the search side: a manual
    # `POST /api/centroids/reload` is the only way to refresh.
    global _centroid_store
    _centroid_store = CentroidStore(
        centroids_dir=Path(_cfg.centroids_dir) if _cfg.centroids_dir else None,
        expected_model=_cfg.centroid_expected_model,
        expected_feature_dim=_cfg.centroid_expected_feature_dim,
    )
    _centroid_store.load()

    if index_db is None:
        db_path = _cfg.index_db_path
        if _cfg.test_mode and db_path == DEFAULT_INDEX_DB_PATH:
            db_path = ":memory:"
        index_db = IndexDB(
            db_path=db_path,
            qdrant_client=qdrant,
            id_cache_ttl_seconds=getattr(
                cfg, "index_id_cache_ttl_seconds", 900
            ),
        )
    _index_db = index_db
    random_picker = RandomPicker(index_db)

    # Option B (Sept 2026): single canonical collection. The
    # previous SyncManager that moved points from `images_pending`
    # to `images` on a background interval is deleted — the indexer
    # now writes directly to the canonical collection, so there's
    # no staging area to drain.
    diversity_cache = DiversityResultCache(
        ttl_seconds=_cfg.diversity_cache_ttl_seconds,
        max_entries=_cfg.diversity_cache_max_entries,
    )
    # Frozen rankings that make plain /api/search offset paging stable.
    # Separate from diversity_cache on purpose: this holds an ordered
    # (id, score) list and no MMR stats, so plain search and diversity
    # don't recouple through a shared type.
    snapshot_cache = SearchSnapshotCache(
        ttl_seconds=_cfg.search_snapshot_ttl_seconds,
        max_entries=_cfg.search_snapshot_max_entries,
    )

    # ---------------- In-app indexer (admin Index button) ----------------
    # Spawned as a child subprocess via `indexer.local_sync --json-progress`.
    # Same module the old host-side CLI invoked, but now owned by the
    # search process so the Index button in the UI can drive it without
    # SSH/host-side venv. Subprocess isolation = a torch deadlock or OOM
    # in the indexer can never take down the search backend.
    #
    # Option B: the indexer writes directly to the canonical
    # collection (`_cfg.qdrant_collection`). No read/write split, no
    # staging area, no SyncManager. The change-detection code in
    # local_sync reads the same collection it writes to, so change
    # detection always sees the canonical corpus.
    from search.indexer_runner import IndexerRunner, default_indexer_command_factory
    indexer_cmd_factory = default_indexer_command_factory(
        python=sys.executable,
        sources=_cfg.indexer_sources,
        model=_cfg.model_name,
        device=_cfg.indexer_device,
        qdrant_url=_cfg.qdrant_url,
        qdrant_api_key=_cfg.qdrant_api_key,
        qdrant_collection=_cfg.qdrant_collection,
        batch_size=_cfg.indexer_batch_size,
    )
    indexer_runner = IndexerRunner(command_factory=indexer_cmd_factory)

    # ---------------- Dynamic centroids ----------------
    #
    # Runtime-computed centroids that complement the disk-loaded
    # `.pt` set. v1 ships one: `favourites`, the mean of every
    # favourited photo's embedding, re-normalised to unit length.
    # Future "themes" register the same way without touching
    # static loading or the search plumbing.
    #
    # IMPORTANT: registered AFTER index_db is initialised (above)
    # so the compute closure captures a live IndexDB, not None.
    global _dynamic_centroids
    _dynamic_centroids = DynamicCentroidRegistry()
    _dynamic_centroids.register(_make_favourites_centroid_spec(qdrant, index_db))
    # Round‑29b: also register the centroid under the new name
    # `likes` (which matches the user‑visible album label). Same
    # compute closure; just a different registry key. Keeps old
    # saved searches / shared links that pass `?centroid=favourites`
    # working without forcing everyone to migrate.
    _dynamic_centroids.register(_make_likes_centroid_spec(qdrant, index_db))
    # Round‑29: register the dislikes centroid so the Search
    # button on the Dislikes album card has a real centroid to
    # query. New dislikes / undislikes invalidate it via
    # `_invalidate_dislikes_centroid` (wired into the dislikes
    # router below).
    _dynamic_centroids.register(_make_dislikes_centroid_spec(qdrant, index_db))
    # Register every existing album as a centroid. New albums
    # register themselves via POST /api/albums → _register_album_centroid;
    # renames via PATCH /api/albums/{id}; deletes via
    # _unregister_album_centroid. This loop only runs on startup
    # to re-attach any albums that survived a process restart.
    # SQLite is fast here (a handful of rows in practice); we
    # don't bother with asyncio.to_thread for what amounts to a
    # sub-millisecond query.
    for existing in index_db.list_albums():
        _dynamic_centroids.register(
            _make_album_centroid_spec(qdrant, index_db, existing["id"])
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Tier 2.3 (defensive) — startup work as background tasks.
        #
        # The previous version ran two heavy startup operations inline:
        #
        #   1. text_encoder.get_encoder()    (~3 GB / ~30 s on cold cache)
        #   2. the native path text indexes   (a text-index build over
        #                                     the collection)
        #
        # Both are wrapped in to_thread, but they still BLOCK the lifespan
        # completion signal. With gunicorn --workers 1 and a startupProbe
        # budget of 5 min (30 × 10s), a startup step that takes
        # longer than 5 min makes the pod fail to become Ready, ArgoCD
        # kills it, the user sees 502 on every endpoint. That was the
        # prod failure in commit 542a553.
        #
        # Fix: kick both off as background tasks and let lifespan return
        # immediately. The /healthz probe returns 200 while these are
        # still running; /api/search will lazy-load the encoder on the
        # first text query, and the search results will be a no-op
        # (with a clear warning logged) until the startup step finishes.
        async def _bg_text_encoder_warmup() -> None:
            try:
                # get_encoder is a module-level lazy singleton — first
                # call does the actual ~30 s load. Off-thread because
                # it's sync and would otherwise monopolise the loop.
                await asyncio.to_thread(
                    text_encoder.get_encoder,
                    test_mode=_cfg.test_mode,
                )
                logger.info("text encoder warmed in background")
            except Exception as e:  # noqa: BLE001
                logger.warning("text encoder background warm-up failed: %s", e)

        async def _bg_warm_id_cache() -> None:
            """Warm the /random session deck's id cache.

            The deck is the corpus's point ids, shuffled per session.
            Fetching them is a full Qdrant id scroll — ~25s on the
            2.2M corpus, because Qdrant has to read every point
            record and the REST API has no id-only scroll mode.

            That is paid ONCE here rather than on the user's first
            click, and refreshed on a timer so the cache does not go
            stale against a re-indexed corpus. The scroll runs in a
            thread so the event loop is never blocked.
            """
            ttl = max(60.0, float(getattr(_cfg, "index_id_cache_ttl_seconds", 900)))
            while True:
                try:
                    count = await asyncio.to_thread(
                        index_db.refresh_id_cache, ()
                    )
                    logger.info(
                        "random deck id cache ready: %d ids", len(count)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.warning("random deck id cache warm-up failed: %s", e)
                await asyncio.sleep(ttl)

        async def _bg_ensure_text_indexes() -> None:
            """Create the native text indexes the filename filter needs.

            A collection restored from a snapshot, or one created by an
            older indexer, has no text index on `path` — and a filename
            query against an unindexed field falls back to a full
            payload scan. Creating the indexes is idempotent, so this
            runs on every startup.

            It runs in the background for the same reason the other
            startup work does: building a text index over millions of
            paths takes long enough that doing it inline would blow
            the readiness probe budget.
            """
            try:
                await asyncio.to_thread(qdrant.ensure_path_text_index)
                logger.info("native path text indexes ready")
            except Exception as e:  # noqa: BLE001
                logger.warning("native path text index setup failed: %s", e)

        refresh_task: asyncio.Task[None] | None = None
        # Log a warning immediately if Qdrant is unreachable so the
        # issue is visible in pod logs even before the startup step
        # completes. healthz() is sync but cheap (<10 ms typical) so
        # keeping it inline is fine.
        if not qdrant.healthz():
            logger.warning(
                "Qdrant unreachable at startup (%s) — search will fail until it recovers",
                _cfg.qdrant_url,
            )

        # Fire-and-forget; tasks survive past this `await` because they
        # hold references via the closure. asyncio.create_task schedules
        # them on the running loop without blocking lifespan return.
        encoder_warmup_task = asyncio.create_task(_bg_text_encoder_warmup())
        text_index_task = asyncio.create_task(_bg_ensure_text_indexes())
        id_cache_task: asyncio.Task[None] | None = None
        if not getattr(cfg, "test_mode", False):
            id_cache_task = asyncio.create_task(_bg_warm_id_cache())

        # Option B: no SyncManager to start. The indexer writes
        # directly to the canonical collection, so the search side
        # always reads the latest state without needing a background
        # drain loop.

        # Periodic user-data purge. Drops the user-state rows whose
        # photo the indexer has pruned, without the operator having
        # to hit POST /api/cache/refresh manually. The manual endpoint
        # stays as a force-now override. The task is cooperative: the
        # purge is wrapped in `asyncio.to_thread` so the event loop
        # isn't blocked.
        interval = _cfg.index_db_refresh_interval_seconds
        if interval > 0:
            async def _periodic_refresh_loop() -> None:
                while True:
                    try:
                        await asyncio.sleep(interval)
                        # Take the refresh lock so a manual
                        # POST /api/cache/refresh fired during a
                        # periodic rebuild bails instead of running
                        # a second Qdrant scroll in parallel. The lock
                        # is non-blocking; if the manual path holds
                        # it, we just skip this tick (next tick retries).
                        if not await asyncio.to_thread(index_db.try_acquire_refresh_lock):
                            logger.debug(
                                "periodic refresh: lock held by another path, skipping this tick"
                            )
                            continue
                        try:
                            t0 = time.time()
                            # Skip the periodic purge while the
                            # indexer is actively writing. A purge
                            # mid-index would see photos that are
                            # mid-upsert as missing and drop the
                            # user's rows that point at them. After
                            # the indexer job finishes, the next tick
                            # catches any external deletes.
                            from search.indexer_runner import IndexerState
                            try:
                                idx_status = indexer_runner.status()
                                if idx_status.state is IndexerState.RUNNING:
                                    logger.debug(
                                        "periodic refresh: indexer is "
                                        "running (job_id=%s); skipping "
                                        "this tick — the indexer is "
                                        "still writing",
                                        idx_status.job_id,
                                    )
                                    continue
                            except Exception as e:  # noqa: BLE001
                                # Defensive: never let a status probe
                                # failure block the refresh loop. Log
                                # and fall through to the rebuild.
                                logger.debug(
                                    "periodic refresh: indexer status "
                                    "probe failed (%s); proceeding with "
                                    "rebuild", e,
                                )

                            # There is no derived cache to rebuild —
                            # the corpus lives in Qdrant and is
                            # always current there. The one thing
                            # that still needs a periodic trigger is
                            # the orphan purge: user-state rows whose
                            # photo the indexer has pruned.
                            #
                            # `maybe_refresh` is the single entry
                            # point for that work, so the timestamp
                            # the status endpoint reports is the
                            # timestamp of the last real run.
                            ran = await asyncio.to_thread(
                                index_db.maybe_refresh
                            )
                            dt_ms = int((time.time() - t0) * 1000)
                            if ran:
                                logger.info(
                                    "periodic user-data purge ran in %d ms",
                                    dt_ms,
                                )
                            else:
                                logger.debug(
                                    "periodic user-data purge: "
                                    "not due yet"
                                )
                        finally:
                            await asyncio.to_thread(index_db.release_refresh_lock)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        logger.warning("periodic IndexDB refresh failed: %s", e)
            refresh_task = asyncio.create_task(_periodic_refresh_loop())
        try:
            yield
        finally:
            # Option B: no SyncManager to stop.
            if refresh_task is not None:
                refresh_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:  # noqa: BLE001
                    logger.warning("refresh task shutdown: %s", e)
            # Cancel and wait for the background startup tasks. If
            # a background startup step is mid-write, give it a
            # short grace period then cancel — we'd rather lose
            # unflushed rows than block shutdown.
            for _bg_task in (
                locals().get("encoder_warmup_task"),
                locals().get("index_init_task"),
                locals().get("id_cache_task"),
            ):
                if _bg_task is None:
                    continue
                if not _bg_task.done():
                    try:
                        await asyncio.wait_for(_bg_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        _bg_task.cancel()
                        try:  # noqa: SIM105
                            await _bg_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                            pass
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:  # noqa: BLE001
                        logger.warning("background startup task shutdown: %s", e)
            await asyncio.to_thread(index_db.close)

    app = FastAPI(
        title="lustre",
        version="0.1.0",
        lifespan=lifespan,
        # Don't redirect /photo/{id} to /photo/{id}/ (trailing slash).
        redirect_slashes=False,
    )

    app.state.index_db = index_db
    app.state.random_picker = random_picker
    app.state.diversity_cache = diversity_cache
    # Round-35: tests need to mutate indexer_runner.state to
    # exercise the "skip refresh while indexing" guards in
    # _periodic_refresh_loop and POST /api/cache/refresh.
    app.state.indexer_runner = indexer_runner

    # Router includes (§B2) follow below. Auth was removed; deploy
    # behind a reverse proxy that handles access control (caddy auth,
    # oauth2-proxy, tailscale, etc.) if you need it.

    # ---------------------- Router includes (§B2) ----------------------
    # Each resource group is one APIRouter in search/routers/. The
    # inline @app.* route handlers below are still present (the
    # extraction is incremental — this commit wires the smallest
    # self-contained one first). Inline duplicates will be removed
    # in follow-up commits as each router is verified.
    from search.routers.admin_index import build_admin_index_router
    from search.routers.albums import build_albums_router
    from search.routers.centroids import build_centroids_reload_router
    from search.routers.centroids_list import build_centroids_list_router
    from search.routers.centroids_search import build_centroids_search_router
    from search.routers.collections import build_collections_router
    from search.routers.dislikes import build_dislikes_router
    from search.routers.favorites import build_favorites_router
    from search.routers.for_you import build_for_you_router
    from search.routers.random import build_random_router
    from search.routers.saved_searches import build_saved_searches_router
    from search.routers.search import build_search_router
    from search.routers.similar import build_similar_router
    from search.routers.system import build_system_router
    from search.routers.thumbnails import build_thumbnails_router
    app.include_router(build_collections_router(qdrant=qdrant))
    app.include_router(build_saved_searches_router(index_db=index_db))
    app.include_router(build_similar_router(
        qdrant=qdrant,
        cfg=_cfg,
        index_db=index_db,
    ))
    app.include_router(build_random_router(index_db=index_db, cfg=_cfg))
    app.include_router(build_for_you_router(
        index_db=index_db,
        qdrant=qdrant,
        cfg=_cfg,
    ))
    app.include_router(build_for_you_router(
        index_db=index_db,
        qdrant=qdrant,
        cfg=_cfg,
    ))
    app.include_router(build_favorites_router(
        index_db=index_db,
        cfg=_cfg,
        invalidate_likes_centroid=_invalidate_likes_centroid,
        invalidate_for_you=_for_you_invalidate_deck,
    ))
    app.include_router(build_dislikes_router(
        index_db=index_db,
        cfg=_cfg,
        invalidate_likes_centroid=_invalidate_likes_centroid,
        invalidate_dislikes_centroid=_invalidate_dislikes_centroid,  # round‑29
        invalidate_for_you=_for_you_invalidate_deck,
    ))
    app.include_router(build_albums_router(
        index_db=index_db,
        cfg=_cfg,
        register_album_centroid=_register_album_centroid,
        unregister_album_centroid=_unregister_album_centroid,
        invalidate_album_centroid=_invalidate_album_centroid,
    ))
    app.include_router(build_centroids_reload_router(
        centroid_store=_centroid_store,
    ))
    app.include_router(build_centroids_list_router(
        centroid_store=_centroid_store,
        dynamic_centroids=_dynamic_centroids,
    ))
    app.include_router(build_centroids_search_router(
        qdrant=qdrant,
        cfg=_cfg,
        index_db=index_db,
        centroid_store=_centroid_store,
        dynamic_centroids=_dynamic_centroids,
        diversity_cache=diversity_cache,
    ))
    app.include_router(build_system_router(
        qdrant=qdrant,
        cfg=_cfg,
        index_db=index_db,
        path_liveness_cache=_path_liveness_cache,
        path_liveness_cache_max=_PATH_LIVENESS_CACHE_MAX,
    ))
    app.include_router(build_thumbnails_router())
    app.include_router(build_admin_index_router(
        indexer_runner=indexer_runner,
        index_db=index_db,
    ))

    # Option B (Sept 2026): the previous /api/sync/* endpoints
    # (status, pause, resume) are removed. There is no SyncManager
    # to surface — the indexer writes directly to the canonical
    # collection, so reads always see the latest state.

    def _parse_collections(request: Request) -> list[str]:
        """
        Read all `?collection=` query params from the request, in
        stable order. The multi-value shape is what powers the
        chip-style filter UI on the frontend.
        """
        # getlist() preserves order and skips duplicates the way
        # the URL is written; we don't dedupe here because the
        # user might paste a duplicate and the search behavior
        # is the same.
        return [c for c in request.query_params.getlist("collection") if c]

    def _parse_prompts(request: Request, name: str) -> list[str]:
        """
        Read all multi-value prompt params, stripped and in URL order.
        Empty values are ignored so `?positives=` behaves like no chip.
        """
        return [p.strip() for p in request.query_params.getlist(name) if p.strip()]

    def _parse_filename(request: Request) -> str:
        """
        Read the optional `?filename=` query param, stripped.

        Single-value: the existing UI is a single text input. If the
        user ever sends multiple `?filename=` values, the first
        non-empty one wins (consistent with how single-valued
        `?centroid=` works in `_parse_centroid`). Empty /
        whitespace-only returns "" so callers can do the standard
        "if not raw: skip" check.

        The pattern itself is validated later — the helper at the
        top of `create_app` translates the raw string to image ids
        via the native path filter and surfaces 400 on an invalid
        syntax.
        """
        for raw in request.query_params.getlist("filename"):
            value = raw.strip()
            if value:
                return value
        return ""

    def _parse_centroid(request: Request) -> str | None:
        """
        Read the active centroid from `?centroid=...` (back-compat).

        Returns the FIRST non-empty `?centroid=` value. For new code
        prefer `_parse_centroids` which returns the full list —
        this helper exists so single-centroid call sites and
        response shapes (`centroid: str | None`) keep working
        unchanged. The full multi-centroid contract lives in
        `_parse_centroids` + `centroids` response field.
        """
        for raw in request.query_params.getlist("centroid"):
            value = raw.strip()
            if value:
                return value
        return None

    def _resolve_query_vector(
        centroid_names: list[str] | None,
        prompt_state: PromptState,
        weights: list[float] | None = None,
        filename_pattern: str = "",
        centroid_specs: list[DynamicCentroidSpec] | None = None,
        collections: list[str] | None = None,
    ) -> tuple[list[float], str | None, str | None]:
        """
        Resolve the query vector for a search request.

        Centroid-anchored search is mutually exclusive with text
        search. When `centroid_names` is non-empty, the text prompts
        must be empty (the caller enforces this via `PromptState`).
        Multiple centroid names blend into one vector via
        `blend_centroids` (weighted mean, re-normalised) — see
        `?centroid=&centroid=&weights=` URL contract.

        Filename-only mode (a non-empty `filename_pattern` with no
        prompts and no centroids) returns a zero vector. The
        filename filter is applied separately via `HasId`, so the
        vector's role is reduced to "rank the allowed ids" — and
        cosine similarity to a zero vector is constant for every
        point, so Qdrant falls back to its id-based tie-breaker
        (stable, deterministic ordering by id). The user gets a
        browseable list of the matching files without the "must
        have a positive prompt" error.

        Returns one of three outcomes:

          (vector, None, None)            — success
          ([], "centroid_not_found", ...) — no vector, error tag
          ([], "empty", ...)              — no vector, error tag

        The third tuple slot is a short human-readable detail
        string for the API response and the search page header.
        """
        if centroid_names:
            entries: list[tuple[list[float], str]] = []
            missing: list[str] = []
            for name in centroid_names:
                vec = _fetch_centroid_vector(name)
                if vec is None:
                    missing.append(name)
                    continue
                entries.append((vec, name))
            if missing:
                # All-or-nothing: if any centroid is missing/empty,
                # surface the first missing name. Partial blends are
                # surprising (the user asked for A+B and got just A)
                # so we refuse rather than silently drop one.
                return [], "centroid_not_found", (
                    f"centroid {missing[0]!r} not loaded"
                )
            expected_dim = (
                _cfg.centroid_expected_feature_dim if _cfg else None
            )
            try:
                blended = blend_centroids(
                    entries, weights, expected_dim=expected_dim,
                )
            except ValueError as e:
                return [], "centroid_not_found", str(e)
            return blended, None, None
        if not prompt_state.positives:
            if filename_pattern.strip():
                # Filename-only mode: bypass the prompt requirement
                # and return a zero vector. The filename filter
                # (HasId) and `collections` filter (MatchAny)
                # restrict the candidate set; cosine similarity to
                # the zero vector is constant, so Qdrant falls
                # through to its id-based tie-breaker. The user
                # sees a deterministic, browseable list of the
                # matching files.
                dim = _cfg.centroid_expected_feature_dim if _cfg else None
                if not dim:
                    # Last-resort default for the SigLIP2 model the
                    # project uses. Tests run without a real config
                    # sometimes — this keeps the page rendering.
                    dim = 768
                return [0.0] * dim, None, None
            if collections:
                # Collections-only mode: same trick as filename-only.
                # No prompt → zero vector, the `collections` MatchAny
                # filter restricts candidates, Qdrant's id tie-breaker
                # gives a stable browse order. Useful for "show me
                # everything in <library>" without a semantic prompt.
                dim = _cfg.centroid_expected_feature_dim if _cfg else None
                if not dim:
                    dim = 768
                return [0.0] * dim, None, None
            return [], "empty", "at least one positive prompt is required"
        vec = text_encoder.embed_query_multi(
            tuple(prompt_state.positives),
            tuple(prompt_state.negatives),
        )
        return vec, None, None

    def _fetch_centroid_vector(name: str) -> list[float] | None:
        """Return the vector for a single centroid by name, or None
        if it's unknown / empty. Used by `_resolve_query_vector`
        when blending N centroids. Kept separate so the per-centroid
        fallback (static → dynamic) is consistent with the
        single-centroid path and tested in one place."""
        if _centroid_store is not None:
            spec = _centroid_store.get(name)
            if spec is not None:
                return spec.vector
        if _dynamic_centroids is not None:
            dyn = _dynamic_centroids.get_vector(name)
            if dyn is not None:
                return dyn[0]
            # Dynamic spec exists but has no data yet — distinct
            # from "unknown name". The caller (multi-centroid path)
            # surfaces this as the "not loaded" error rather than
            # silently dropping the entry.
            if _dynamic_centroids.get_spec(name) is not None:
                return None
        return None

    # _normalize_prompt_state lives in search/_indexed_helpers.py (§B2 step 37).

    # _search_query_string lives in search/_indexed_helpers.py (§B2 step 36).

    def _favorite_id_set_sync(point_ids: list[str]) -> set[str]:
        favorites: set[str] = set()
        for pid in point_ids:
            row = index_db.get_by_id(pid)
            if row and int(row.get("is_favorite") or 0) == 1:
                favorites.add(pid)
        return favorites

    async def _favorite_id_set(point_ids: list[str]) -> set[str]:
        return await asyncio.to_thread(_favorite_id_set_sync, point_ids)

    async def _results_from_hits(
        hits: list[SearchHit],
        favorite_ids: set[str] | None = None,
        dislike_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        # Parallel-fetch favorite + dislike sets when either is missing.
        # When both are pre-resolved (the hot /api/search path), skip
        # the DB hit entirely. (Round-6 — dislike state now returned
        # in results so per-tile .is-neg class works on every page.)
        if favorite_ids is None or dislike_ids is None:
            ids = [h.id for h in hits]
            fav_set, dis_set = await asyncio.gather(
                _favorite_id_set(ids),
                asyncio.to_thread(index_db.dislike_id_set, ids),
            )
            if favorite_ids is None:
                favorite_ids = fav_set
            if dislike_ids is None:
                dislike_ids = dis_set
        return [
            SearchResult(
                id=h.id,
                path=h.path,
                score=h.score,
                score_str=f"{h.score:.3f}",
                url=resolve_url(h.id, _cfg.web_ui_url),
                is_favorite=h.id in favorite_ids,
                is_disliked=h.id in dislike_ids,
                # LQIP from the Qdrant payload (set at index time, T9).
                # None when the point was indexed before blurhash landed.
                blurhash=(h.payload or {}).get("blurhash"),
                # Dimensions for the photo-card caption row (Phase E).
                width=(h.payload or {}).get("width"),
                height=(h.payload or {}).get("height"),
            )
            for h in hits
        ]
    async def _favorite_ids_for_filter() -> set[str]:
        rows = await asyncio.to_thread(index_db.list_favorites, _cfg.max_results_total, 0)
        return {str(row["id"]) for row in rows}

    # /api/search is wired via search/routers/search.py (§B2 step 40).
    # Placed here (not at the top with the other routers) because it
    # needs the two closure-bound helpers above to be defined first.
    app.include_router(build_search_router(
        qdrant=qdrant,
        cfg=_cfg,
        index_db=index_db,
        diversity_cache=diversity_cache,
        snapshot_cache=snapshot_cache,
        resolve_query_vector=_resolve_query_vector,
        favorite_ids_for_filter=_favorite_ids_for_filter,
    ))

    def _diversity_metadata(stats: DiversityStats) -> DiversityMetadata:
        return DiversityMetadata(
            requested=stats.requested,
            applied=stats.applied,
            mode=stats.mode,
            strength=stats.strength,
            candidate_count=stats.candidate_count,
            result_count=stats.result_count,
            duplicate_images_collapsed=stats.duplicate_images_collapsed,
            semantic_groups_covered=stats.semantic_groups_covered,
            depth=stats.depth,
            pool_depth=stats.pool_depth,
        )

    @app.get("/api/photo/{point_id}")
    async def photo_metadata(point_id: str) -> JSONResponse:
        """Fetch metadata for a single photo by ID.

        Used by the frontend's dedicated photo page to render the
        large photo + metadata sidebar. Returns everything the
        sidebar can show: identity, file info, indexing info,
        favourite status. Does NOT return the vector — that's a
        different concern, and `payload` carries enough for the UI.
        """
        try:
            hit = qdrant.retrieve(point_id)
        except (ConnectionError, OSError) as e:
            logger.warning("Qdrant unreachable for /api/photo/%s: %s", point_id, e)
            raise HTTPException(status_code=502, detail="Qdrant unreachable") from e
        if hit is None:
            raise HTTPException(status_code=404, detail="Photo not found")

        # Determine favourite + dislike status. The dedicated photo
        # page needs both so its Like / Dislike buttons can mirror
        # the Lightbox's pressed-state pattern (mutual exclusivity,
        # un-toggle on second click).
        fav_ids = await _favorite_id_set([point_id])
        is_fav = point_id in fav_ids
        is_disliked = await asyncio.to_thread(index_db.is_disliked, point_id)

        # Pull the Qdrant payload — every indexed field lives here.
        # Default everything to None so the response is always
        # well-typed (older points indexed before some fields were
        # added won't have them).
        p = hit.payload or {}
        return JSONResponse(content={
            "id": hit.id,
            "path": hit.path,
            "score": hit.score,
            "is_favorite": is_fav,
            "is_disliked": is_disliked,
            "url": f"/photo/{point_id}/raw",
            # Image dimensions
            "width": p.get("width"),
            "height": p.get("height"),
            "blurhash": p.get("blurhash"),
            # File info
            "size": p.get("size"),
            "mtime": p.get("mtime"),
            "folder": p.get("folder"),
            "shard": p.get("shard"),
            "collection": p.get("collection"),
            # Indexing info — which model wrote this point, when
            "model_name": p.get("model_name"),
            "model_revision": p.get("model_revision"),
            "model_dim": p.get("model_dim"),
            "indexed_at": p.get("indexed_at"),
            # Fingerprints (used by Diversity ranker; useful debug info)
            "content_sha256": p.get("content_sha256"),
            "dhash": p.get("dhash"),
        })

    @app.get("/photo/{point_id}/raw")
    async def photo_raw(
        request: Request,
        point_id: str,
        # Optional target width (pixels). When set, the server
        # returns a Lanczos-resized version of the original. Width
        # is preserved; height scales proportionally. Cached on
        # disk so repeat requests hit the cache. Browsers' default
        # image scaling (Lanczos or similar) is usually fine, but
        # serving a pre-resized image at the exact viewport width
        # avoids the small lossiness of in-browser scaling — and
        # also slashes bandwidth when the source is e.g. 12 MP
        # but the lightbox only needs 1920 px wide.
        w: int | None = Query(
            None,
            ge=64,
            le=8192,
            description="Optional target width in pixels. Server Lanczos-resizes if set.",
        ),
    ) -> Response:
        # Every blocking call in this handler is wrapped in
        # asyncio.to_thread — qdrant.retrieve, resolve_local,
        # _is_path_alive, and local.stat can each block for
        # seconds-to-minutes on a slow Qdrant or TrueNAS NFS. Without
        # these wrappers, a single slow request blocks the entire
        # uvicorn worker's event loop, and with WEB_CONCURRENCY=1
        # that hangs the whole service (see probes commit
        # gitops@9c075d0 — those are the band-aid for this same bug).
        try:
            hit = await asyncio.to_thread(qdrant.retrieve, point_id)
        except (ConnectionError, OSError) as e:
            logger.warning("Qdrant unreachable for /photo/%s/raw: %s", point_id, e)
            raise HTTPException(status_code=502, detail="Qdrant unreachable") from e
        if hit is None:
            raise HTTPException(status_code=404, detail="Photo not found")
        # Lazy liveness: resolve the payload path to its local mount
        # first, then check the resolved path. The payload path is
        # often a Windows UNC (or any source-side path) that is
        # never alive on the search server, so checking it directly
        # 404s every photo.
        local = await asyncio.to_thread(
            resolve_local, hit.path, _cfg.nas_images_base, _cfg.path_prefix
        )
        # _is_path_alive does an os.stat / os.path.exists walk that
        # can hang on a stuck TrueNAS NFS export.
        is_alive = await asyncio.to_thread(
            lambda p: local is not None and _is_path_alive(p),
            str(local) if local is not None else "",
        )
        if not is_alive or local is None:
            raise HTTPException(status_code=404, detail="File not found on disk")
        # After this point mypy can narrow `local: Path` (not `Path | None`).
        local_path: Path = local

        # ---- Resize branch ----
        # Only JPEG/PNG/WebP/HEIF are supported by PIL out of the box
        # (and the existing code only sets media types for those). If
        # the original is some exotic format (RAW, TIFF, etc.) or the
        # resize fails for any reason, fall back to the original file.
        if w is not None:
            try:
                cached = await asyncio.to_thread(_resize_cached, point_id, w, local_path)
            except _ResizeError as e:
                logger.warning("Resize failed for %s @ w=%d: %s", point_id, w, e)
            else:
                stat = await asyncio.to_thread(cached.stat)
                etag = hashlib.md5(  # noqa: S324
                    f"{stat.st_mtime_ns}-{stat.st_size}-w{w}".encode()
                ).hexdigest()
                if request.headers.get("if-none-match") == etag:
                    return Response(
                        status_code=304,
                        headers={
                            "ETag": etag,
                            "Cache-Control": "public, max-age=31536000, immutable",
                        },
                    )
                return FileResponse(
                    cached,
                    media_type="image/jpeg",
                    headers={
                        "ETag": etag,
                        # Cached file is immutable for the lifetime of
                        # the (point_id, width) pair. The cache is
                        # keyed off the source file's mtime+size
                        # inside _resize_cached, so a re-encoded
                        # source automatically invalidates.
                        "Cache-Control": "public, max-age=31536000, immutable",
                    },
                )

        stat = await asyncio.to_thread(local_path.stat)
        etag = hashlib.md5(f"{stat.st_mtime_ns}-{stat.st_size}".encode()).hexdigest()  # noqa: S324

        # Per-spec If-None-Match: respond 304 with no body if the
        # client's cached ETag matches. Saves bandwidth on every
        # cache hit (back/forward, grid re-render, etc.).
        # Per plan §5.C5: photos can change on disk (file replaced,
        # re-indexed), so the cache MUST revalidate every time. We pair
        # this with an ETag above so repeat fetches hit 304 — no body
        # transfer, just a cheap round trip to confirm the bytes are
        # still the same.
        cache_headers = {
            "ETag": etag,
            "Cache-Control": "public, max-age=0, must-revalidate",
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=cache_headers)

        filename = local_path.name
        return FileResponse(
            local_path,
            media_type=guess_content_type(local_path),
            filename=filename,
            content_disposition_type="inline",
            headers=cache_headers,
        )

    # /api/search is wired via search/routers/search.py (§B2 step 40).


    # ---------------------- Favourites ----------------------

    def _favorite_rows_to_results(rows: list[dict[str, Any]]) -> list[SearchResult]:
        def _maybe_int(v: Any) -> int | None:
            try:
                iv = int(v) if v is not None else None
            except (TypeError, ValueError):
                return None
            return iv if iv and iv > 0 else None
        return [
            SearchResult(
                id=str(row["id"]),
                path=str(row["path"]),
                score=0.0,
                score_str="",
                url=resolve_url(str(row["id"]), _cfg.web_ui_url),
                is_favorite=True,
                width=_maybe_int(row.get("width")),
                height=_maybe_int(row.get("height")),
            )
            for row in rows
        ]


    def _dislike_rows_to_results(rows: list[dict[str, Any]]) -> list[SearchResult]:
        """Same shape as `_favorite_rows_to_results`; is_favorite comes
        from a fresh favourites lookup so tiles show true heart state."""
        def _maybe_int(v: Any) -> int | None:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        fav_ids = set(index_db.list_favorite_ids())
        return [
            SearchResult(
                id=str(row["id"]),
                path=str(row["path"]),
                score=0.0,
                score_str="",
                url=resolve_url(str(row["id"]), _cfg.web_ui_url),
                is_favorite=str(row["id"]) in fav_ids,
                width=_maybe_int(row.get("width")),
                height=_maybe_int(row.get("height")),
            )
            for row in rows
        ]

    def _album_member_rows_to_results(rows: list[dict[str, Any]]) -> list[SearchResult]:
        """Same shape as `_favorite_rows_to_results` but does NOT
        hardcode is_favorite=True — album membership is independent
        of favourites status, so the favourite state on the tile
        comes from the row's actual `is_favorite` column (left JOIN
        in IndexDB.get_by_id; album rows from list_album_members
        don't include it so default False is fine for v1).
        """
        # Lazy liveness: drop dead rows from album tiles. Resolve the
        # payload path to its local mount first — the row's `path`
        # is the source-side path (often a Windows UNC) that is
        # never alive on the search server, so checking it directly
        # would drop every row.
        alive = []
        for r in rows:
            local = resolve_local(
                str(r.get("path") or ""),
                _cfg.nas_images_base,
                _cfg.path_prefix,
            )
            if local is not None and _is_path_alive(str(local)):
                alive.append(r)
        if len(alive) < len(rows):
            logger.debug(
                "album members: dropped %d dead row(s) from %d via lazy liveness",
                len(rows) - len(alive), len(rows),
            )
        rows = alive
        def _maybe_int(v: Any) -> int | None:
            try:
                iv = int(v) if v is not None else None
            except (TypeError, ValueError):
                return None
            return iv if iv and iv > 0 else None
        return [
            SearchResult(
                id=str(row["id"]),
                path=str(row["path"]),
                score=0.0,
                score_str="",
                url=resolve_url(str(row["id"]), _cfg.web_ui_url),
                is_favorite=bool(int(row.get("is_favorite") or 0)),
                width=_maybe_int(row.get("width")),
                height=_maybe_int(row.get("height")),
            )
            for row in rows
        ]

    def _random_rows_to_results(rows: list[dict[str, Any]]) -> list[SearchResult]:
        """Build SearchResult objects from SQLite random-sample rows.

        Mirrors `_favorite_rows_to_results` but reads is_favorite from
        the cache (some random samples may already be favourites) and
        uses the cache's width/height for future masonry support.

        No liveness filter here — the picker over-fetches enough that
        we always return exactly `len(rows)` results, even if some are
        photos whose NAS file has been deleted. The user sees a broken
        tile for at most one cache refresh (60 s); the periodic
        IndexDB refresh cleans truly-gone entries. The liveness filter
        in `_favorite_rows_to_results` (album view) stays on because
        that path can't over-fetch and broken tiles there are jarring.
        """
        def _maybe_int(v: Any) -> int | None:
            try:
                iv = int(v) if v is not None else None
            except (TypeError, ValueError):
                return None
            return iv if iv and iv > 0 else None
        out: list[SearchResult] = []
        for row in rows:
            is_fav = bool(int(row.get("is_favorite") or 0))
            _bh = row.get("blurhash") or None
            out.append(
                SearchResult(
                    id=str(row["id"]),
                    path=str(row["path"]),
                    score=0.0,
                    score_str="",
                    url=resolve_url(str(row["id"]), _cfg.web_ui_url),
                    is_favorite=is_fav,
                    is_disliked=bool(int(row.get("is_disliked") or 0)),
                    width=_maybe_int(row.get("width")),
                    height=_maybe_int(row.get("height")),
                    blurhash=_bh,
                )
            )
        return out

    # /api/favorites/{id} POST/DELETE is wired via search/routers/favorites.py (§B2 step 19).


    # /api/similar/{id} is wired via search/routers/similar.py (§B2 step 18).

    # /api/favorites (list) is wired via search/routers/favorites.py (§B2 step 19).


    # ---------------------- Favourites ZIP download ----------------------
    #
    # One-shot .zip of every favourite currently in the cache. Streamed
    # end-to-end so the response begins before every file has been
    # read into memory. Entry names are flattened to `{shard}__{basename}`
    # to avoid silent overwrites when two favourites share a basename
    # but live in different shards, and to avoid zip-path traversal
    # entirely. Files that can't be resolved on disk (deleted after
    # indexing, NAS unmounted) are skipped and recorded in a
    # `_missing.txt` manifest at the root of the archive.

    FAVORITES_ZIP_PAGE_SIZE = 500

    @app.api_route(
        "/favorites/download.zip", methods=["GET", "HEAD"],
        response_class=StreamingResponse,
    )
    async def favorites_download_zip() -> StreamingResponse:
        zs = zipstream.ZipStream(compress_type=zipstream.ZIP_STORED)

        # Favourites whose local file we couldn't resolve. Collected
        # here and emitted as `_missing.txt` at the root of the
        # archive so the user can see what was skipped.
        missing: list[tuple[str, str]] = []

        offset = 0
        while True:
            rows = await asyncio.to_thread(
                index_db.list_favorites,
                limit=FAVORITES_ZIP_PAGE_SIZE,
                offset=offset,
            )
            if not rows:
                break
            for row in rows:
                point_id = str(row["id"])
                payload_path = str(row.get("path") or "")
                shard = str(row.get("shard") or "").strip()
                local = resolve_local(
                    payload_path,
                    _cfg.nas_images_base,
                    _cfg.path_prefix,
                )
                if local is None:
                    missing.append((point_id, payload_path))
                    continue
                # Flatten with shard prefix when known, otherwise
                # bare basename. We strip path separators from the
                # shard so no entry can ever traverse (``local.name``
                # is already just the file name via Path.name).
                safe_shard = (
                    shard.replace("/", "_").replace("\\", "_").strip("_")
                )
                entry_name = (
                    f"{safe_shard}__{local.name}" if safe_shard
                    else local.name
                )
                try:
                    # add_path streams the file in chunks as the
                    # consumer drains the response — no
                    # full-file buffering in RAM.
                    zs.add_path(local, arcname=entry_name)
                except (FileNotFoundError, OSError) as e:
                    logger.warning(
                        "favourites zip: failed to add %s for point %s: %s",
                        local, point_id, e,
                    )
                    missing.append((point_id, str(local)))
            offset += FAVORITES_ZIP_PAGE_SIZE

        if missing:
            lines = [
                "favourites zip manifest",
                f"missing files: {len(missing)}",
                "",
            ]
            for point_id, payload_path in missing:
                lines.append(f"{point_id}\t{payload_path}")
            zs.add("\n".join(lines) + "\n", arcname="_missing.txt")

        stamp = time.strftime("%Y-%m-%d")
        filename = f"favorites-{stamp}.zip"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        }
        return StreamingResponse(
            zs,
            media_type="application/zip",
            headers=headers,
        )

    @app.api_route(
        "/albums/{album_id}/download.zip", methods=["GET", "HEAD"],
        response_class=StreamingResponse,
    )
    async def album_download_zip(album_id: int) -> StreamingResponse:
        """Stream an album's photos as a zip — same shape and rules
        as `/favorites/download.zip`. Membership comes from
        `index_db.list_album_members` (INNER JOIN against the cache,
        so orphan rows are hidden from the archive too). Files we
        can't resolve on disk are skipped and recorded in
        `_missing.txt`. Album id with no row → 404.
        """
        album = await asyncio.to_thread(index_db.get_album, album_id)
        if album is None:
            raise HTTPException(
                status_code=404, detail=f"Album {album_id} not found",
            )

        zs = zipstream.ZipStream(compress_type=zipstream.ZIP_STORED)
        missing: list[tuple[str, str]] = []

        offset = 0
        while True:
            rows = await asyncio.to_thread(
                index_db.list_album_members,
                album_id,
                limit=FAVORITES_ZIP_PAGE_SIZE,
                offset=offset,
            )
            if not rows:
                break
            for row in rows:
                point_id = str(row["id"])
                payload_path = str(row.get("path") or "")
                shard = str(row.get("shard") or "").strip()
                local = resolve_local(
                    payload_path,
                    _cfg.nas_images_base,
                    _cfg.path_prefix,
                )
                if local is None:
                    missing.append((point_id, payload_path))
                    continue
                safe_shard = (
                    shard.replace("/", "_").replace("\\", "_").strip("_")
                )
                entry_name = (
                    f"{safe_shard}__{local.name}" if safe_shard
                    else local.name
                )
                try:
                    zs.add_path(local, arcname=entry_name)
                except (FileNotFoundError, OSError) as e:
                    logger.warning(
                        "album zip: failed to add %s for point %s: %s",
                        local, point_id, e,
                    )
                    missing.append((point_id, str(local)))
            offset += FAVORITES_ZIP_PAGE_SIZE

        if missing:
            lines = [
                f"album {album_id} zip manifest",
                f"missing files: {len(missing)}",
                "",
            ]
            for point_id, payload_path in missing:
                lines.append(f"{point_id}\t{payload_path}")
            zs.add("\n".join(lines) + "\n", arcname="_missing.txt")

        stamp = time.strftime("%Y-%m-%d")
        # Slug the album name for the filename. Empty / unsafe names
        # fall back to the bare id (the prefix below adds "album-").
        # Cap length so the filename stays well under common
        # filesystem limits even after concatenation.
        raw = re.sub(r"[^A-Za-z0-9._-]+", "-", album["name"]).strip("-")
        slug = raw[:64] if raw else str(album_id)
        filename = f"album-{slug}-{stamp}.zip"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        }
        return StreamingResponse(
            zs,
            media_type="application/zip",
            headers=headers,
        )

    # ---------------------- Albums ----------------------
    #
    # User-curated collections of favourites. The favourites table
    # is the implicit default album (no row in `albums`); every
    # endpoint below is for user-created albums only. Album
    # membership is independent of favourites status: a photo can
    # be in an album without being favourited, and vice versa.
    #
    # Each album auto-registers as a dynamic centroid under the
    # name `album:{id}` so it can be used as a search primitive
    # via /api/search?centroid=album:42 (same shape as the
    # existing `favourites` centroid).

    # /api/albums/* is wired via search/routers/albums.py (§B2 step 20).

    # ---------------------- Cache refresh ----------------------
    #
    # Two paths to refresh the IndexDB:
    #   1. Periodic background task (lifespan) — runs every
    #      INDEX_DB_REFRESH_INTERVAL_SECONDS (default 6h).
    #   2. Manual override via this endpoint — force-now button.
    # The periodic path respects the skip-if-populated branch in
    # the orphan purge; the manual path runs it on demand.
    # A lock guards against the two paths racing — only one
    # refresh runs at a time; the other short-circuits with a
    # clear log message.

    @app.api_route("/api/cache/refresh", methods=["GET", "POST"], response_model=None)
    async def api_cache_refresh(
        request: Request,
        full: bool = Query(
            False,
            description=(
                                "Deprecated. There is no cache to refresh: "
                                "the photo corpus lives in Qdrant and the "
                                "SQLite side holds only authoritative user "
                                "state. The endpoint now runs the orphan "
                                "purge, which checks liveness against "
                                "Qdrant."
                            ),
        ),
    ) -> dict[str, Any] | JSONResponse:
        """Purge user-state rows whose photo Qdrant no longer holds.

        There is no derived cache to refresh any more. The photo
        corpus, its metadata, and the native text index all live in
        Qdrant; SQLite holds only the user's own state — favourites,
        dislikes, albums, memberships, saved searches, feedback.

        What this endpoint still does is the one thing that needed a
        trigger: dropping user-state rows that point at photos the
        indexer has pruned. The purge checks liveness against
        Qdrant, so a favourite is only dropped when the COLLECTION
        says the photo is gone — never because a local mirror was
        stale.

        The `full` parameter is accepted and ignored, so an old
        caller does not break.
        """
        purged = await asyncio.to_thread(index_db.purge_orphaned_user_data)
        logger.info("cache refresh: purged orphaned user data: %s", purged)
        return {
            "status": "ok",
            "purged": purged,
            "note": (
                "no cache to refresh — the corpus lives in Qdrant; this "
                "ran the orphan purge against Qdrant's liveness"
            ),
        }

    # ------------------------------------------------------------------
    # Static SPA fallback
    # ------------------------------------------------------------------
    # When the lustre image is built as a single container (see
    # docker/Dockerfile.search), the SvelteKit build output lands at
    # /app/static. Mount its hashed assets at /_app and fall back to
    # index.html for every other path so the SPA can hydrate and route
    # client-side via hash or pushState.
    static_dir = Path(os.environ.get("FRONTEND_DIR", "/app/static"))
    if static_dir.is_dir():
        app_static = static_dir / "_app"
        if app_static.is_dir():
            # SvelteKit adapter-static emits content-hashed filenames
            # under /_app/immutable/... — these never change once
            # built, so emit Cache-Control: immutable for the whole
            # /_app/ tree. Browsers stop revalidating on every visit.
            class CachedStatic(StaticFiles):
                """StaticFiles mount that emits Cache-Control: immutable.

                SvelteKit adapter-static writes content-hashed assets
                to /_app/immutable/{js,assets}/<filename>. Their content
                URL is stable forever, so a 1-year immutable cache
                header is correct. (Tier 1.3 — see docs/performance-
                improvements.md.) We attach the header to every file
                response, including /_app/immutable/.../LICENSES.txt.
                """

                def file_response(self, *args: Any, **kwargs: Any) -> Response:
                    resp = super().file_response(*args, **kwargs)
                    resp.headers["Cache-Control"] = (
                        "public, max-age=31536000, immutable"
                    )
                    return resp

            app.mount(
                "/_app", CachedStatic(directory=str(app_static)), name="spa-assets"
            )

        @app.get("/favicon.svg", include_in_schema=False)
        async def favicon() -> FileResponse:
            return FileResponse(static_dir / "favicon.svg")

        # SPA fallback — every non-API, non-asset path returns
        # index.html. API routes get proper 404 JSON instead of
        # being silently swallowed by the SPA catch-all.
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_shell(full_path: str) -> FileResponse:
            # Never serve SPA HTML for /api/* — those should 404
            # with a proper JSON detail so API clients can
            # distinguish "endpoint doesn't exist" from "got HTML".
            if full_path.startswith("api/"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Not found")
            # Strip the leading slash; resolve and guard against
            # directory traversal.
            safe = (static_dir / full_path).resolve()
            if safe.is_file() and safe.is_relative_to(static_dir.resolve()):
                return FileResponse(safe)
            return FileResponse(static_dir / "index.html")

    return app


# Build the module-level `app` so production servers (`gunicorn search.app:app`)
# can load it directly. The defaults are loaded from env once at import time;
# tests that need a custom config call `create_app(cfg=..., qdrant=...)`
# explicitly and ignore the module-level instance.
app: FastAPI = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("search.app:app", host="0.0.0.0", port=8000)  # noqa: S104
