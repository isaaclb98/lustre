"""
search/config.py — environment variable loading + validation.

Loaded once at process start, validated up-front, frozen for the
lifetime of the process. See .env.example for the full table.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse as _QdrResp

from search.diversity_config import Diversity, load_diversity_from_env

logger = logging.getLogger(__name__)

# Load .env from cwd (or any ancestor) on import. Real process env wins.
load_dotenv()

# SigLIP2 variant mapping: variant name -> (model name, vector dimension)
SIGLIP_VARIANTS = {
    "B/16-256": ("ViT-B-16-SigLIP2-256", 768),
    "L/16-256": ("ViT-L-16-SigLIP2-256", 1024),
    "gopt/16-384": ("ViT-gopt-16-SigLIP2-384", 1536),
    "so400m/16-384": ("ViT-so400m-patch16-384", 1152),
}

# Reverse mapping: dim -> variant name. The reconciler uses this to
# suggest a likely SIGLIP_VARIANT in the mismatch error message so
# the operator doesn't have to dig through git history to figure
# out which model produced the stored vectors.
SIGLIP_VARIANTS_BY_DIM: dict[int, str] = {
    dim: name for name, (_model, dim) in SIGLIP_VARIANTS.items()
}


class QdrantVariantMismatch(RuntimeError):
    """Raised by reconcile_variant_from_qdrant when stored Qdrant
    vectors don't match the env-configured SigLIP2 variant.

    Distinct from a generic exception so callers (and tests) can
    catch this specific condition. The message includes the three
    recovery paths the operator can take, with the exact curl
    commands for the destructive ones.
    """


DEFAULT_VARIANT = "B/16-256"  # smallest + fastest; end-user ghcr default.
# Override via SIGLIP_VARIANT (or MODEL_NAME) for higher-quality
# embeddings. Prod keeps so400m via .env — this default only
# applies to fresh deployments where no env override is set.

def get_siglip_variant() -> str:
    """Get the configured SigLIP2 variant from SIGLIP_VARIANT env var."""
    variant = os.environ.get("SIGLIP_VARIANT", DEFAULT_VARIANT)
    if variant not in SIGLIP_VARIANTS:
        raise ValueError(
            f"Invalid SIGLIP_VARIANT '{variant}'. "
            f"Must be one of: {', '.join(SIGLIP_VARIANTS.keys())}"
        )
    return variant

def get_model_name_for_variant(variant: str) -> str:
    """Get the model name for a given variant."""
    if variant not in SIGLIP_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return SIGLIP_VARIANTS[variant][0]

def get_vector_dim_for_variant(variant: str) -> int:
    """Get the vector dimension for a given variant."""
    if variant not in SIGLIP_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return SIGLIP_VARIANTS[variant][1]

def get_vector_dim() -> int:
    """Get the vector dimension for the currently configured variant."""
    return get_vector_dim_for_variant(get_siglip_variant())


# Thumbnail storage path (inside container)
THUMBNAIL_DIR = os.environ.get("THUMBNAIL_DIR", "/app/data/thumbnails")


# Variant reconciliation against Qdrant. Replaces the older file-based
# tracking (data/siglip_variant.json). Qdrant is the single source of
# truth — the dim of the live collection's vectors = the model that
# produced them. Per-point payload (`model_variant`) is informational
# and written by the indexer alongside `model_name`/`model_revision`.
def reconcile_variant_from_qdrant(
    env_variant: str,
    qdrant_url: str,
    qdrant_api_key: str | None,
    qdrant_collection: str,
) -> None:
    """Reconcile env variant against Qdrant on startup.

    Refuse-to-start reconciler. On dim or payload mismatch, raises
    `QdrantVariantMismatch` instead of dropping the collection. The
    search app fails to boot, the gunicorn worker dies, the docker
    container crash-loops, the operator sees the error in `docker logs`
    and fixes the env. The collection is preserved.

    The previous implementation called `client.delete_collection()`
    on mismatch. That was a destructive default reachable from any
    code path that imported `search.config` (most of the test
    suite), so a routine test run against a prod-shaped Qdrant URL
    could wipe a multi-GB production index in milliseconds. The
    destruction is now opt-in by the operator, not by the test fixture.

    Behavior by case:

      - Collection absent: no-op. Fresh install; the user will
        reindex via Settings → Index.

      - Collection empty: no-op. Fresh install via a different path.

      - Collection's vector dim ≠ env variant's expected dim:
        vectors are in the wrong embedding space. Raise
        `QdrantVariantMismatch` with a recovery message that tells
        the operator how to manually drop, rename, or fix the env.
        **No automatic delete.**

      - Sample point has `model_variant` payload that differs
        from env_variant: same. Raise, do not delete.
        Tolerates missing payload (legacy data) — dim check is
        the binding constraint for the four registered variants
        because their dims (768/1024/1152/1536) are unique.

      - Otherwise: app boots normally.

    To recover from a mismatch without losing data, the operator
    has three options (all surfaced in the raised error message):

      1. Fix the env: set SIGLIP_VARIANT to the variant that
         actually produced the stored vectors. App boots.

      2. Rename the bad collection, set QDRANT_COLLECTION to a
         fresh name, and reindex from scratch. The bad data is
         preserved on disk for postmortem.

      3. Manually drop the collection (curl -X DELETE), then
         reindex. This is a deliberate, audited action — not
         something the app does on its own.
    """
    env_dim_expected = get_vector_dim_for_variant(env_variant)

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10)
        info = client.get_collection(collection_name=qdrant_collection)
    except _QdrResp as e:
        # Collection doesn't exist — fresh install.
        logger.info("No Qdrant collection %r yet: %s. Fresh install.",
                    qdrant_collection, e)
        return
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not reach Qdrant for variant reconcile (%s): %s. "
            "Continuing; search endpoints will surface their own errors.",
            qdrant_url, e,
        )
        return

    if info.points_count == 0:
        logger.info("Qdrant collection %r is empty: nothing to reconcile.",
                    qdrant_collection)
        return

    # The collection's config.vectors_count / config.params.vectors
    # shape depends on Qdrant version. Easiest robust read:
    stored_dim = _qdrant_collection_dim(info)
    if stored_dim is None:
        logger.warning(
            "Could not read vector dim from Qdrant collection %r. "
            "Skipping reconcile; proceeding with env variant %s.",
            qdrant_collection, env_variant,
        )
        return

    if stored_dim != env_dim_expected:
        raise QdrantVariantMismatch(
            f"Qdrant collection {qdrant_collection!r} has stored dim={stored_dim} "
            f"but env SIGLIP_VARIANT={env_variant!r} expects dim={env_dim_expected}. "
            f"The vectors are in the wrong embedding space — similarity search "
            f"would return wrong results. Refusing to start.\n"
            f"\n"
            f"Pick one of these recovery paths:\n"
            f"  (1) Set SIGLIP_VARIANT to the variant that produced the stored "
            f"vectors (likely {SIGLIP_VARIANTS_BY_DIM.get(stored_dim, '?')!r}) and restart.\n"
            f"  (2) Reindex from scratch with the new variant. To preserve the "
            f"old data for postmortem, rename the bad collection first:\n"
            f"      curl -X PATCH http://{qdrant_url}/collections/{qdrant_collection} "
            f"-H 'Content-Type: application/json' "
            f"-d '{{\"name\": \"{qdrant_collection}_bad_<timestamp>\"}}'\n"
            f"      then set QDRANT_COLLECTION={qdrant_collection}_v2 and reindex.\n"
            f"  (3) Manually drop the collection (deletes ~hours of indexing):\n"
            f"      curl -X DELETE http://{qdrant_url}/collections/{qdrant_collection}\n"
            f"      then reindex via Settings → Index.\n"
            f"\n"
            f"This reconciler no longer drops collections automatically. That "
            f"behavior was too dangerous to be a startup default."
        )

    # Dim matches. Optional soft check: per-point payload.
    try:
        sample, _next = client.scroll(
            collection_name=qdrant_collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not scroll sample point for payload check: %s", e)
        return

    if not sample:
        return

    stored_variant = sample[0].payload.get("model_variant") if sample[0].payload else None
    if stored_variant is None:
        # Legacy data — no `model_variant` payload. Dim already matches,
        # so the binding constraint holds. Proceed.
        logger.info(
            "Qdrant dim matches env variant %s; sample point has no "
            "`model_variant` payload (legacy data). Proceeding.",
            env_variant,
        )
        return

    if stored_variant != env_variant:
        raise QdrantVariantMismatch(
            f"Qdrant collection {qdrant_collection!r} has dim={stored_dim} (matches "
            f"env) but per-point payload reports model_variant={stored_variant!r} "
            f"≠ env SIGLIP_VARIANT={env_variant!r}. The vectors may be a re-embed "
            f"of the wrong variant, or the payload was written by an older indexer. "
            f"Refusing to start to prevent returning wrong search results.\n"
            f"\n"
            f"Pick one of these recovery paths:\n"
            f"  (1) Set SIGLIP_VARIANT={stored_variant!r} and restart (if the "
            f"stored variant is the one you actually want).\n"
            f"  (2) Reindex from scratch with the new variant. To preserve the "
            f"old data for postmortem, rename the bad collection first:\n"
            f"      curl -X PATCH http://{qdrant_url}/collections/{qdrant_collection} "
            f"-H 'Content-Type: application/json' "
            f"-d '{{\"name\": \"{qdrant_collection}_bad_<timestamp>\"}}'\n"
            f"      then set QDRANT_COLLECTION={qdrant_collection}_v2 and reindex.\n"
            f"  (3) Manually drop the collection (deletes ~hours of indexing):\n"
            f"      curl -X DELETE http://{qdrant_url}/collections/{qdrant_collection}\n"
            f"      then reindex via Settings → Index.\n"
            f"\n"
            f"This reconciler no longer drops collections automatically. That "
            f"behavior was too dangerous to be a startup default."
        )

    logger.info("Variant reconciled against Qdrant: %s (dim=%d)",
                env_variant, stored_dim)


def _qdrant_collection_dim(info: Any) -> int | None:
    """Read the vector dim from a Qdrant CollectionInfo object.

    Tolerates Qdrant-client version differences: `vectors_count`,
    `config.params.vectors.size`, or `config.vectors_count`.
    Returns None if it can't figure it out.
    """
    try:
        vectors_cfg = info.config.params.vectors
    except AttributeError:
        return None

    # `vectors` is one of: an int (uniform dim), a VectorParams,
    # or a dict of named vectors. Try them in order.
    if isinstance(vectors_cfg, int):
        return vectors_cfg
    size = getattr(vectors_cfg, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors_cfg, dict):
        for v in vectors_cfg.values():
            s = getattr(v, "size", None)
            if isinstance(s, int):
                return s
        return None
    return None

# Backward compatibility: these are derived from the variant.
# DEFAULT_MODEL is what the env says at import time — it follows
# the active runtime variant, NOT the code default. Callers that
# want the code default should use `get_model_name_for_variant(
# DEFAULT_VARIANT)` explicitly. The constant exists for callers
# that import DEFAULT_MODEL at module load before any env is read.
DEFAULT_MODEL: str = get_model_name_for_variant(get_siglip_variant())
DEFAULT_COLLECTION: str = "images"
# Option B (Sept 2026): the indexer writes directly to the canonical
# collection. The round-14 staging-collection design (`images_pending`
# + SyncManager) was deleted because the
# `--qdrant-collection` ambiguity it introduced was the root cause
# of the change-detection silent re-embed bug. See commit 75aa9bb for
# the diagnosis; the follow-up commit deleted `search/sync.py`,
# `images_pending`, and `_sync_meta` entirely.
DEFAULT_RESULT_LIMIT: int = 28

# Mapping from open_clip arch tag → (centroid-file `model` string).
# The expected feature dim is read from the model registry at lookup
# time, not stored here — the registry is the single source of truth.
# The centroid's `model` field is a short lowercase tag written by
# `isaac-image-scoring`; the search side's MODEL_NAME is the open_clip
# arch tag. We map between them so the store can refuse to load a
# centroid that lives in a different embedding space than the indexed
# images.
#
# Add new entries here only when introducing a new model family.
# The unknown-model branch raises at config load time — fail fast
# rather than serve garbage cosine results.
_CENTROID_MODEL_COMPAT = {
    # Round-29: registered B/16-256 as the ghcr end-user default. Adding
    # the centroid-compat mapping here so the search container boots
    # when SIGLIP_VARIANT=B/16-256 — without this entry, config.load()
    # raises "no centroid-compat mapping" and the container fails the
    # healthcheck loop.
    "ViT-B-16-SigLIP2-256": "siglip2",
    "ViT-gopt-16-SigLIP2-384": "siglip2",
    "ViT-L-16-SigLIP2-256": "siglip2",
    "ViT-so400m-patch16-384": "siglip2",
}


def centroid_compat_for(model_name: str) -> tuple[str, int]:
    """
    Return (expected_model_tag, expected_feature_dim) for the
    given open_clip arch tag. Raises ValueError on unknown models
    so the search container fails to start with a clear error
    rather than silently loading mismatched centroids.

    `expected_feature_dim` is sourced from the model registry by
    `model_name`; the registry is the only place model dimensions
    are referenced.

    open_clip tags carry an "hf-hub:<vendor>/" prefix (e.g.
    ``hf-hub:timm/ViT-SO400M-16-SigLIP2-384``); the map is keyed by
    the bare arch tag. Normalize by splitting on ``/`` and taking
    the last segment so the deployment's MODEL_NAME matches
    regardless of how open_clip names the model.
    """
    bare = model_name.split("/")[-1] if "/" in model_name else model_name
    if bare not in _CENTROID_MODEL_COMPAT:
        raise ValueError(
            f"MODEL_NAME={model_name!r} has no centroid-compat mapping. "
            f"Add one in search/config.py _CENTROID_MODEL_COMPAT. "
            f"Known models: {sorted(_CENTROID_MODEL_COMPAT)}"
        )
    from lustre_kernel.registry import get as _registry_get
    return _CENTROID_MODEL_COMPAT[bare], _registry_get(bare).dim


@dataclass(frozen=True)
class Config:
    qdrant_url: str
    qdrant_collection: str
    qdrant_api_key: str | None
    model_name: str
    model_revision: str
    device: str
    top_k_default: int
    top_k_max: int
    query_timeout_ms: int
    nas_images_base: str
    path_prefix: str
    web_ui_url: str
    log_level: str
    # In test mode the real model is replaced with a deterministic mock.
    # Set SEARCH_TEST_MODE=1 from conftest to enable.
    test_mode: bool
    # qdrant_prefer_grpc: when True, the qdrant-client uses the gRPC
    # transport (port 6334 by default) instead of REST/JSON. ~3x faster
    # on bulk vector fetches at depth 5000 because protobuf avoids the
    # JSON parse cost and the per-message encoding overhead. Off by
    # default so existing deploys see no behaviour change; flip on via
    # `QDRANT_PREFER_GRPC=true` once the qdrant container has port 6334
    # exposed (docker-compose exposes it as of the gRPC port addition;
    # see the compose file's qdrant service ports).
    qdrant_prefer_grpc: bool = False
    # qdrant_grpc_port: gRPC port for `prefer_grpc=True`. Defaults to
    # `QDRANT_GRPC_PORT` env var if set, else the qdrant docker
    # image's standard gRPC port (6334). Ignored unless prefer_grpc=True.
    qdrant_grpc_port: int = 6334
    # Option B (Sept 2026): single canonical collection. The
    # previous split-collection design (separate read + write + a
    # SyncManager that moved points between them) was deleted; see
    # commit 75aa9bb and the follow-up removal of `search/sync.py`.
    # Search Diversity. These knobs apply only to ordinary /api/search and
    # the SSR search page.
    diversity_max_candidate_pool_size: int = 5000
    diversity_cache_ttl_seconds: int = 300
    diversity_cache_max_entries: int = 64
    diversity_duplicate_hamming_distance: int = 10
    diversity_relevance_drop: float = 0.10
    # Stable pagination for plain /api/search. Qdrant offset paging is
    # unsound on HNSW — each page is ranked at a different depth
    # (offset+limit) and neighbouring pages disagree near the boundary,
    # producing duplicate rows. The router instead freezes one ranking
    # per query and slices pages from it, growing it in bands.
    #
    # Measured on prod (2.05M points, Qdrant 1.19.0):
    #   single deep fetch: depth 1000 = 26ms, 5000 = 179ms, 10000 = 473ms
    #   band fetch w/ exclusion: 0 excl = 9ms, 1000 = 12ms, 10000 = 33ms
    # Bands keep page 1 at ~10ms regardless of how deep the user scrolls.
    search_initial_band_size: int = 256
    search_band_size: int = 500
    # Hard ceiling on one frozen ranking. 20k ids = 833 pages of 24,
    # far past real scroll depth, and bounds both snapshot memory
    # (~800KB ids+scores) and the exclusion request body (~36 bytes/id).
    # Past this `has_more` goes false rather than degrading further.
    search_max_snapshot_size: int = 20_000
    search_snapshot_ttl_seconds: int = 300
    search_snapshot_max_entries: int = 64
    # Background-prefetch the next band after serving a page. Hides the
    # ~0.5-1s cold band fetch behind the user's reading time. Disable
    # for deterministic call-count assertions in tests, or as an ops
    # kill-switch if background load is ever a problem.
    search_prefetch_next_band: bool = True
    # Surprise Me: fetch a deep pool (no vectors), shuffle, return a
    # small random slice. Pool size controls the diversity-relevance
    # trade-off (bigger = more diverse but slower).
    surprise_pool_size: int = 5000
    surprise_result_count: int = 28
    # Custom centroids: read-only dir of .pt files produced by
    # `isaac-image-scoring`. Optional — when unset or missing, the
    # centroid store is empty and the feature is effectively off.
    # Defaults to None so existing test fixtures (which construct
    # Config directly) keep working unchanged.
    centroids_dir: str | None = None
    # Single source of truth for the Diversity knob. Routers resolve
    # query params against this default via `resolve_diversity()`.
    diversity: Diversity = field(default_factory=load_diversity_from_env)
    # Per-request timeout for Qdrant's recommend() API (used by /api/for-you's
    # centroid blend and any future recommend-style endpoints). Recommend
    # is heavier than a plain search (Qdrant has to fetch positive/negative
    # point vectors, compute their mean, then run an HNSW search across
    # the whole collection), and the default 2s used for normal search is
    # too tight over HTTPS through a reverse proxy.
    # Raised 10s -> 40s: at the prod library size (1.96M points, 409 fav
    # ids), Qdrant's HNSW search latency has crept to 2-10s and occasionally
    # trips the 10s cap. The graceful fallback (qdrant_client.recommend)
    # turns a timeout into an empty result set, which the for-you cache
    # then sticks at for 5 minutes — effectively "For You is broken until
    # the cache TTL expires". 40s gives the heavy recommend path room to
    # succeed under load; on a healthy Qdrant the median is <2s, so the
    # user rarely notices the wider window.
    recommend_timeout_ms: int = 40000
    # Derived from MODEL_NAME: which `model` tag and dim centroids
    # must have to be loaded. Defaults match the production model
    # so the centroid-compat guard is meaningful out of the box;
    # production always sets these via config.load().
    centroid_expected_model: str = "siglip2"
    # Sourced from the model registry by `config.load()`. Default
    # pulls from the registry so tests that construct `Config()`
    # directly (without going through `config.load()`) still see a
    # real dim. Production always overrides via `centroid_compat_for()`.
    #
    # Reads from `DEFAULT_MODEL` rather than hardcoding
    # "ViT-gopt-16-SigLIP2-384" so the default follows whichever
    # variant is the current prod default (so400m-patch16-384 as
    # of the model-variant migration). `model_name` itself is also
    # set from `DEFAULT_MODEL` further down — both flow from the
    # same `get_siglip_variant()` lookup, so the registry call here
    # never disagrees with the production override.
    centroid_expected_feature_dim: int = field(
        default_factory=lambda: __import__(
            "lustre_kernel.registry", fromlist=["get"],
        ).get(DEFAULT_MODEL).dim,
    )
    index_db_path: str = "./data/images.db"
    # ----- Operational constants (formerly module-level in app.py) -----
    # All env-driven so an operator can tune the running service without
    # a code change. Defaults match the prior hardcoded values exactly.
    # Defensive cap on how far the client can paginate through a
    # single search. The frontend walks results 28 at a time and
    # honours `has_more`, so this only matters if the client
    # IGNORES has_more and keeps requesting further offsets. Set
    # high (1M) so a real user never hits it; the safety net is
    # for runaway scripts / old clients with no has_more logic.
    max_results_total: int = 1_000_000
    static_assets_version: int = 32
    max_prompt_chars: int = 512
    max_prompts_total: int = 16
    # `valid_views` and `default_view` are a closed enum; not env-driven.
    # Moved to Config for testability (tests can construct a Config with
    # custom values instead of monkey-patching module globals).
    valid_views: tuple[str, ...] = ("grid", "feed")
    default_view: str = "grid"
    # FTS filter cardinality guard (see app.py:_resolve_filename_filter).
    filename_cardinality_guard: float = 0.5
    # ----- Periodic maintenance -----
    # How often the search container runs the orphan purge in the
    # background, dropping the user-state rows whose photo the
    # indexer has pruned, without an operator hitting
    # POST /api/cache/refresh. Manual refresh still works as a
    # force-now override. Default 6h is a sweet spot: long enough to
    # not waste Qdrant round-trips, short enough that /random
    # and /albums are rarely more than 6h stale.
    index_db_refresh_interval_seconds: int = 21600
    # How often the /random session deck's id cache is rebuilt.
    # The deck is the corpus's point ids; fetching them is a full
    # Qdrant id scroll (~25s on the 2.2M corpus), so the scroll is
    # done on a timer in the background and the request path only
    # ever reads the cache. 15 min keeps the deck fresh against a
    # re-index without scrolling often enough to matter.
    # TTL for the lazy path-liveness cache (see `app.py:_is_path_alive`).
    # Bounds per-request `Path.exists()` cost while keeping the read
    # path fresh. 60s means a freshly-deleted file shows up as dead
    # within a minute; tune higher if you're on a slow NAS.
    path_liveness_ttl_seconds: int = 60
    # ----- In-app indexer (admin Index button) --------------------------
    # The search backend can spawn its own indexer subprocess via the
    # admin Index API. These knobs configure that subprocess; they're
    # independent of the search-side `model_name`/`device` so the
    # read path (search) and write path (indexer) can be tuned
    # separately (e.g. search on GPU, index on CPU).
    indexer_sources: tuple[str, ...] = ()
    indexer_device: str = "cpu"
    indexer_batch_size: int = 8
    # Every directory a stored photo path may resolve under, for the
    # orphan purge's liveness check. The primary one is PHOTOS_DIR
    # (the photos mount); the others are the indexer's source dirs —
    # a source mounted as a SIBLING of the photos root keeps its
    # files under a different top-level path, and a path outside
    # every root is "cannot check" (treated as alive), so without
    # them that source's favourites would never be garbage
    # collected. Derived from `indexer_sources`, so adding a source
    # gives it purge coverage with no second place to configure.
    photos_roots: tuple[str, ...] = ()
    # (Single-user auth removed. Front the service with a reverse-proxy
    # auth if access control is needed: caddy, oauth2-proxy, tailscale, etc.)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"env {name}={raw!r} is not a valid int") from e


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"env {name}={raw!r} is not a valid float") from e


def _bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Truthy: 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _str(name: str, default: str) -> str:
    """Read a string env var, falling back to default when unset/empty."""
    raw = os.environ.get(name, default)
    return raw if raw else default


def _opt_str(name: str) -> str | None:
    """Read an optional string env var. Returns None when unset or empty.

    Distinguishes "" (treated as None) so callers that use the
    "configured or None" idiom (`os.environ.get(name) or None`)
    see the same value. None and "" both round-trip to None.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    return raw


def _str_any_present(name: str) -> str | None:
    """Read an env var, returning None ONLY when fully absent.

    Unlike `_opt_str` this preserves the empty string when the
    variable is set to "" — needed by flags like SEARCH_TEST_MODE
    that are checked for "is this set?" rather than "is this
    truthy?".
    """
    return os.environ.get(name)


def load() -> Config:
    """
    Load config from environment. Validates required fields and
    invariants. Raises ValueError on invalid input.
    """
    # Round-35: PHOTOS_DIR is the canonical env var name going forward.
    # NAS_IMAGES_BASE is accepted as a deprecated alias so existing
    # deployments (notably the production install on the AMD host)
    # don't break on image upgrade. PHOTOS_DIR wins if both are set;
    # using the old name emits a one-line deprecation warning so the
    # operator can fix the .env at their leisure.
    #
    # The internal Config field is still called `nas_images_base` —
    # not renamed because ~30 tests build Config(...) directly with
    # `nas_images_base=` as a keyword arg, and renaming the field
    # would be a no-value churn. The user-facing name (env var +
    # error message + docs) is what matters to deployers.
    photos_dir = _str("PHOTOS_DIR", "").strip()
    legacy_nas_base = _str("NAS_IMAGES_BASE", "").strip()
    if not photos_dir and legacy_nas_base:
        photos_dir = legacy_nas_base
        warnings.warn(
            "NAS_IMAGES_BASE is deprecated; rename to PHOTOS_DIR in your .env",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "NAS_IMAGES_BASE is deprecated — rename to PHOTOS_DIR in .env"
        )
    if not photos_dir and not os.environ.get("SEARCH_TEST_MODE"):
        # In test mode the photos dir may be a fixture path, set by
        # conftest. Don't enforce the required check there.
        raise ValueError(
            "PHOTOS_DIR is required (NAS_IMAGES_BASE is a deprecated alias)"
        )

    # `INDEXER_SOURCES` is a comma-separated list of source dirs;
    # defaults to `NAS_IMAGES_PATH` so a single-source setup just
    # sets the latter.
    indexer_sources = tuple(
        s.strip()
        for s in _str(
            "INDEXER_SOURCES",
            _str("NAS_IMAGES_PATH", ""),
        ).split(",")
        if s.strip()
    )

    # The roots the orphan purge may resolve a stored photo path
    # under. `photos_dir` is the photos mount; every source that is
    # NOT already inside it is a separate mount (a source mounted as
    # a sibling of the photos root) and needs its own entry, or the
    # purge could never confirm that one of its photos is gone.
    #
    # Sources INSIDE the photos mount are not added: the mount
    # already covers them, and listing them would narrow the root a
    # path is attributed to. The mount is the durable statement of
    # "these files live here"; a source dir can be renamed or
    # retired without the mount going away.
    photos_roots: tuple[str, ...] = (photos_dir,) if photos_dir else ()
    for src in indexer_sources:
        if not photos_dir:
            photos_roots = photos_roots + (src,)
            continue
        try:
            Path(src).relative_to(photos_dir)
        except ValueError:
            photos_roots = photos_roots + (src,)

    # Validate SigLIP2 variant before loading the rest of the config
    variant = get_siglip_variant()
    index_db_path = _opt_str("INDEX_DB_PATH")
    if not index_db_path and os.environ.get("SEARCH_TEST_MODE"):
        index_db_path = ":memory:"
    if not index_db_path:
        index_db_path = "./data/images.db"

    # Determine data directory from index_db_path (kept for any future
    # path-aware logic; the Qdrant reconciler below does not use it).
    _data_dir = "./data" if index_db_path == ":memory:" else str(Path(index_db_path).parent)

    # Reconcile env variant against Qdrant on startup. Qdrant is the
    # single source of truth — its dim + per-point payload together
    # tell us whether the indexed vectors match the env variant. On
    # mismatch, raises QdrantVariantMismatch (which propagates up
    # through gunicorn startup and crashes the container — that's
    # the design). The operator gets the error in `docker logs` and
    # picks one of the three recovery paths the error message lists.
    # The collection is preserved; destruction is opt-in by the
    # operator, never automatic.
    reconcile_variant_from_qdrant(
        variant,
        qdrant_url=_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=_opt_str("QDRANT_API_KEY"),
        qdrant_collection=_str("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )

    top_k_default = _int("TOP_K_DEFAULT", DEFAULT_RESULT_LIMIT)
    top_k_max = _int("TOP_K_MAX", 200)
    if not (1 <= top_k_default <= top_k_max):
        raise ValueError(
            f"TOP_K_DEFAULT={top_k_default} must be in [1, TOP_K_MAX={top_k_max}]"
        )

    expected_model, expected_dim = centroid_compat_for(
        _str("MODEL_NAME", DEFAULT_MODEL)
    )

    cfg = Config(
        qdrant_url=_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=_str("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        qdrant_api_key=_opt_str("QDRANT_API_KEY"),
        model_name=_str("MODEL_NAME", DEFAULT_MODEL),
        model_revision=_str("MODEL_REVISION", ""),
        device=_str("DEVICE", "cpu"),
        top_k_default=top_k_default,
        top_k_max=top_k_max,
        # Per-request timeout for the search API (query_points). The
        # default of 60s is generous for the production corpus (2M
        # points): a 50k limit typically completes in 22-27s and
        # would otherwise be at the edge of an older 30s default,
        # which produced intermittent 504s on transient Qdrant load.
        # 60s leaves ~2x headroom without binding gunicorn (which
        # is configured with a 2-minute worker timeout).
        query_timeout_ms=_int("QUERY_TIMEOUT_MS", 60000),
        # Per-request timeout for the recommend() path. Recommend
        # has a separate budget because the HNSW walk over a
        # positive+negative vector mean takes a different code path
        # with a different cost profile. Bumped to match the search
        # timeout so the two paths have symmetric headroom.
        recommend_timeout_ms=_int("RECOMMEND_TIMEOUT_MS", 60000),
        nas_images_base=photos_dir,
        path_prefix=_str("PATH_PREFIX", ""),
        web_ui_url=_str("WEB_UI_URL", "http://localhost:8000"),
        log_level=_str("LOG_LEVEL", "INFO"),
        test_mode=bool(
            os.environ.get("SEARCH_TEST_MODE")
            or os.environ.get("SEARCH_NO_MODEL")
        ),
        qdrant_prefer_grpc=_bool("QDRANT_PREFER_GRPC", False),
        qdrant_grpc_port=_int("QDRANT_GRPC_PORT", 6334),
        diversity_max_candidate_pool_size=_int("DIVERSITY_MAX_CANDIDATE_POOL_SIZE", 5000),
        diversity_cache_ttl_seconds=_int("DIVERSITY_CACHE_TTL_SECONDS", 300),
        diversity_cache_max_entries=_int("DIVERSITY_CACHE_MAX_ENTRIES", 64),
        diversity_duplicate_hamming_distance=_int(
            "DIVERSITY_DUPLICATE_HAMMING_DISTANCE", 10
        ),
        diversity_relevance_drop=_float("DIVERSITY_RELEVANCE_DROP", 0.10),
        search_initial_band_size=_int("SEARCH_INITIAL_BAND_SIZE", 256),
        search_band_size=_int("SEARCH_BAND_SIZE", 500),
        search_max_snapshot_size=_int("SEARCH_MAX_SNAPSHOT_SIZE", 20_000),
        search_snapshot_ttl_seconds=_int("SEARCH_SNAPSHOT_TTL_SECONDS", 300),
        search_snapshot_max_entries=_int("SEARCH_SNAPSHOT_MAX_ENTRIES", 64),
        search_prefetch_next_band=_bool(
            "SEARCH_PREFETCH_NEXT_BAND", True
        ),
        centroids_dir=_opt_str("CENTROIDS_DIR"),
        centroid_expected_model=expected_model,
        centroid_expected_feature_dim=expected_dim,
        index_db_path=index_db_path,
        max_results_total=_int("MAX_RESULTS_TOTAL", 1_000_000),
        static_assets_version=_int("STATIC_ASSETS_VERSION", 32),
        max_prompt_chars=_int("MAX_PROMPT_CHARS", 512),
        max_prompts_total=_int("MAX_PROMPTS_TOTAL", 16),
        filename_cardinality_guard=_float("FILENAME_CARDINALITY_GUARD", 0.5),
        index_db_refresh_interval_seconds=_int("INDEX_DB_REFRESH_INTERVAL_SECONDS", 21600),
        path_liveness_ttl_seconds=_int("PATH_LIVENESS_TTL_SECONDS", 60),
        # In-app indexer (admin Index button). `INDEXER_SOURCES` is a
        # comma-separated list; defaults to `NAS_IMAGES_PATH` so a
        # single-source setup just sets the latter.
        indexer_sources=indexer_sources,
        indexer_device=_str("INDEXER_DEVICE", "cpu"),
        indexer_batch_size=_int("INDEXER_BATCH_SIZE", 8),
        # Roots the orphan purge may resolve a stored path under: the
        # photos mount plus every indexer source that is NOT already
        # inside it. See the `photos_roots` field comment for why.
        photos_roots=photos_roots,
        # Auth removed — no env vars to read here.
    )

    if cfg.diversity_max_candidate_pool_size < cfg.top_k_default:
        raise ValueError(
            "DIVERSITY_MAX_CANDIDATE_POOL_SIZE must be >= TOP_K_DEFAULT"
        )
    if cfg.diversity_cache_ttl_seconds < 0 or cfg.diversity_cache_max_entries < 1:
        raise ValueError("Diversity cache settings must be non-negative and non-empty")
    if not 0 <= cfg.diversity_duplicate_hamming_distance <= 64:
        raise ValueError(
            "DIVERSITY_DUPLICATE_HAMMING_DISTANCE must be between 0 and 64"
        )
    if not math.isfinite(cfg.diversity_relevance_drop) or cfg.diversity_relevance_drop < 0:
        raise ValueError("DIVERSITY_RELEVANCE_DROP must be finite and >= 0")

    # Stable-pagination snapshot settings. Bands may be smaller than a
    # page (the router loops until the page is satisfied), but they must
    # be positive, and the snapshot ceiling must be able to hold at
    # least one full page or `has_more` would be false on page 1.
    if cfg.search_initial_band_size < 1 or cfg.search_band_size < 1:
        raise ValueError(
            "SEARCH_INITIAL_BAND_SIZE and SEARCH_BAND_SIZE must be >= 1"
        )
    if cfg.search_max_snapshot_size < cfg.top_k_max:
        raise ValueError(
            "SEARCH_MAX_SNAPSHOT_SIZE must be >= TOP_K_MAX"
        )
    if cfg.search_snapshot_ttl_seconds < 0 or cfg.search_snapshot_max_entries < 1:
        raise ValueError(
            "SEARCH_SNAPSHOT_TTL_SECONDS must be >= 0 and "
            "SEARCH_SNAPSHOT_MAX_ENTRIES must be >= 1"
        )

    # Validate photos dir if set (test mode may set it later).
    if cfg.nas_images_base and not Path(cfg.nas_images_base).is_dir():
        raise ValueError(
            f"PHOTOS_DIR does not exist or is not a directory: {cfg.nas_images_base}"
        )

    return cfg
