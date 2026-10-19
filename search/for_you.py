"""
search/for_you.py — For You feed service.

Public API
----------
    build_for_you_pool(fav_ids, dis_ids, qdrant, top_pct=1.0, rng=None,
                       cache_key_extra="") -> list[str]
        Materialise the top `top_pct`% of library by relevance-to-taste,
        uniformly shuffled. Cached for `_FOR_YOU_TTL_SECONDS` keyed by
        (fav_ids, dis_ids, top_pct, cache_key_extra); invalidated by
        invalidate_for_you_cache().

    rank_for_you(..., limit=30, page=0, seed=None) -> tuple[list, int, bool]
        Slice `limit` ids from `page * limit` of the shuffled pool,
        fetch their payloads via qdrant.retrieve_batch, return
        (hits, total, has_more). Pass a fresh `seed` on each page
        reload for a fresh shuffle; reuse the same `seed` for
        paginated scroll calls within a single page-mount to walk
        forward through the same shuffle coherently.

    invalidate_for_you_cache() -> None
        Drop the cache. Called from the favorites/dislikes routes
        whenever the user changes their feedback so the next
        /api/for-you/feed request sees fresh ranking signal.

Design notes
------------
The pool is built once per (fav_ids, dis_ids, top_pct, seed) tuple
via qdrant.recommend(positive=fav, negative=dis, limit=pool_size) and
then uniform-random shuffled. The "top top_pct%" semantic is
approximate: it is whatever Qdrant returns for the top-`pool_size`
candidates by recommend() score. This is the fast path (1-3s on
800k library). A separate, slower precomputed-score path is a
future concern.

Cold start (no favorites yet) falls back to a zero-vector search
because Qdrant recommend() requires non-empty positives. The result
degrades to /random-equivalent behaviour for fresh users — see the
zero-vector search() branch below for the same shape.
"""
from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import random

    from search.qdrant_client import SearchHit

logger = logging.getLogger(__name__)


_FOR_YOU_TTL_SECONDS: float = 300.0
"""Default cache TTL for a single shuffled pool entry.

The cache is keyed by `(fav_ids, dis_ids, top_pct, seed)` — same key
reuses the same cached shuffle, different key builds a fresh one.
The TTL bounds the cache lifetime so an unused entry is eventually
reaped. Pool invalidation on like/dislike changes happens via
`invalidate_for_you_cache()` (called from the favorites and dislikes
routes), which clears all entries regardless of TTL."""

DEFAULT_FOR_YOU_TOP_PCT: float = 1.0
"""Default percentile of library to use as the random pool.
At an 800k library, 1% yields 8000 candidates — fast (~1-3s
recommend) and large enough that uniform shuffle gives real
variety per page refresh."""

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_for_you_cache: dict[
    tuple[tuple[str, ...], tuple[str, ...], float, str],
    tuple[float, list[str]],
] | None = None


def invalidate_for_you_cache() -> None:
    """Drop all cached shuffled pools. Call after every Like/Dislike
    write so the next /api/for-you/feed request sees fresh ranking
    signal. Wired from the favorites and dislikes routes.
    """
    global _for_you_cache
    _for_you_cache = None

# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def _library_size(index_db: Any) -> int:
    """Return the current Qdrant point count, or 0 if unreachable.

    Uses the cached index_db.qdrant_point_count() helper rather than
    the live Qdrant API so a transient Qdrant blip doesn't break the
    endpoint. -1 (Qdrant unreachable) is treated as 0 here — the
    caller will short-circuit on an empty pool.
    """
    try:
        n = int(index_db.qdrant_point_count())
    except Exception:  # noqa: BLE001
        logger.warning("for_you: library size lookup failed; treating as 0")
        return 0
    return max(n, 0)


def _zero_vector() -> list[float]:
    """Cold-start placeholder query vector. See for_you.py for the
    same shape; duplicated here to keep this module dependency-free
    on the for_you pipeline (so it can be exercised independently
    in tests).
    """
    import os

    from lustre_kernel.registry import get as _registry_get
    from search.for_you_compute import zero_vector

    model_name = os.environ.get("MODEL_NAME", "ViT-L-16-SigLIP2-256")
    return zero_vector(_registry_get(model_name).dim)


def _pool_size_for(library_size: int, top_pct: float) -> int:
    """How many candidates to fetch. Delegates to the same pure helper
    the explore-mode preview uses. Floor of 1 to avoid degenerate
    0-size pools; no ceiling (per Isaac: "no pool ceiling").
    """
    from search.for_you_compute import explore_pool_size

    if library_size <= 0:
        return 0
    return explore_pool_size(library_size, top_pct)


def build_for_you_pool(
    *,
    fav_ids: list[str],
    dis_ids: list[str],
    qdrant: Any,
    index_db: Any,
    top_pct: float = DEFAULT_FOR_YOU_TOP_PCT,
    rng: random.Random | None = None,
    cache_key_extra: str = "",
) -> list[str]:
    """Return the top `top_pct`% of library by relevance-to-taste,
    uniformly shuffled. Cached for `_FOR_YOU_TTL_SECONDS` keyed by
    (fav_ids, dis_ids, top_pct, cache_key_extra).

    `cache_key_extra` is an opaque string the caller controls.
    Pass a fresh random seed on each page reload so the user sees
    a different shuffle on every reload; reuse the same seed for
    paginated scroll calls within the same page-mount to keep the
    walk coherent across page=0, page=1, ...

    `rng` accepts a `random.Random` instance for deterministic
    shuffling in tests; production callers leave it None and get a
    default Random (which is fine because we always make a fresh
    RNG instance — no global state).
    """
    global _for_you_cache
    if _for_you_cache is None:
        _for_you_cache = {}

    cache_key = (
        tuple(fav_ids), tuple(dis_ids), float(top_pct),
        cache_key_extra or "",
    )
    now = _time.monotonic()
    cached = _for_you_cache.get(cache_key)
    if cached is not None and (now - cached[0]) < _FOR_YOU_TTL_SECONDS:
        return list(cached[1])

    library_size = _library_size(index_db)
    pool_size = _pool_size_for(library_size, top_pct)
    if pool_size == 0:
        _for_you_cache[cache_key] = (now, [])
        return []

    if fav_ids:
        hits = qdrant.recommend(
            positive=fav_ids,
            negative=dis_ids,
            limit=pool_size,
        )
    else:
        _hits, _ = qdrant.search(
            vector=_zero_vector(),
            limit=pool_size,
        )
        hits = _hits

    ids = [h.id for h in hits]
    rng_obj = rng if rng is not None else __import__("random").Random()
    rng_obj.shuffle(ids)
    _for_you_cache[cache_key] = (now, ids)
    return list(ids)


def filter_live_ids(ids: list[str], qdrant: Any) -> list[str]:
    """Drop ids that no longer exist in Qdrant.

    Orphan favourites/dislikes (photo deleted from disk and pruned
    from Qdrant, row kept in SQLite for re-attach semantics — see
    index_db.py schema notes) poison qdrant.recommend(): if ANY
    positive/negative id is missing, Qdrant 404s the WHOLE call and
    the graceful fallback caches an empty pool for the TTL — the
    user sees 'No recommendations yet' for 5 minutes per attempt.

    Unlike retrieve, recommend does not skip unknown ids, so filter
    through retrieve_batch (which does) before handing them over.
    One extra round-trip per pool build; the cache absorbs it.
    """
    if not ids:
        return []
    found = qdrant.retrieve_batch(list(ids))
    found_ids = {h.id for h in found}
    live = [i for i in ids if i in found_ids]
    dropped = len(ids) - len(live)
    if dropped:
        logger.info(
            "for-you: filtered %d orphan id(s) from recommend seed "
            "(%d -> %d live)", dropped, len(ids), len(live),
        )
    return live


def rank_for_you(
    *,
    fav_ids: list[str],
    dis_ids: list[str],
    qdrant: Any,
    index_db: Any,
    limit: int = 30,
    page: int = 0,
    top_pct: float = DEFAULT_FOR_YOU_TOP_PCT,
    rng: random.Random | None = None,
    seed: str | None = None,
) -> tuple[list[SearchHit], int, bool]:
    """Slice a page of `limit` items starting at `page * limit` from
    a shuffled pool. Returns (hits, pool_total, has_more).

    `seed` is an opaque string the caller controls. Pass a fresh
    random value on every page reload; reuse the same value for
    paginated scroll calls within the same page-mount. Different
    seeds = different cache entries = different shuffles; same seed
    = same cached shuffle = coherent walk across page=0, page=1, ...

    Without a seed, the cache falls back to a global key and the
    same shuffle is reused across all requests until the TTL
    expires. Tests should always pass a seed to keep isolation.

    `limit` is clamped to >= 1 (no upper bound — the natural cap is
    pool size). `page` is clamped to >= 0. `top_pct` is validated
    (0, 100].
    """
    if limit <= 0:
        limit = 1
    page = max(page, 0)

    pool = build_for_you_pool(
        fav_ids=fav_ids,
        dis_ids=dis_ids,
        qdrant=qdrant,
        index_db=index_db,
        top_pct=top_pct,
        rng=rng,
        cache_key_extra=seed or "",
    )
    pool_total = len(pool)
    if pool_total == 0:
        return [], 0, False

    start = page * limit
    if start >= pool_total:
        return [], pool_total, False
    end = start + limit
    page_ids = pool[start:end]
    has_more = end < pool_total

    hits = qdrant.retrieve_batch(page_ids) if page_ids else []
    # `retrieve_batch` returns hits in input order, but skips missing
    # ids. Re-align to the shuffled pool order so the page renders
    # in walk-through order.
    by_id = {h.id: h for h in hits}
    ordered = [by_id[i] for i in page_ids if i in by_id]
    return ordered, pool_total, has_more
