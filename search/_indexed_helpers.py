"""
search/_indexed_helpers.py — index_db-backed helpers used by /api/search.

These were previously closure-bound to `create_app` (capturing
`index_db`, `_cfg`, `resolve_url`). They're a coherent group:
they all read the search-side IndexDB cache and shape the
result list. Lifting them unblocks the /api/search and
/api/centroids/{name}/search extractions from §B2.

`/api/search` and `/api/centroids/{name}/search` pass the
helpers in as factory parameters; the helpers themselves are
pure functions of `(index_db, cfg, ...args)` so they don't
hold any state.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from collections.abc import Iterable
from typing import Any, cast
from urllib.parse import urlencode

from search.image_resolver import resolve_url
from search.models import SearchResult
from search.qdrant_client import SearchHit

logger = logging.getLogger(__name__)


def favorite_id_set_sync(index_db: Any, point_ids: list[str]) -> set[str]:
    """Sync: read favourite bit for each id from the IndexDB cache.

    The async wrapper `favorite_id_set` runs this in a thread so
    the SQLite read doesn't block the event loop.

    Single IN-clause query (Phase C1): was N individual get_by_id
    calls, now 1. ~10× faster on a 20-tile result page.
    """
    return cast(set[str], index_db.favorite_id_set(point_ids))


async def favorite_id_set(index_db: Any, point_ids: list[str]) -> set[str]:
    """Async wrapper: SQLite read off the event loop."""
    return cast(set[str], await asyncio.to_thread(favorite_id_set_sync, index_db, point_ids))


async def results_from_hits(
    index_db: Any,
    *,
    cfg: Any,
    hits: list[Any],
    favorite_ids: set[str] | None = None,
    dislike_ids: set[str] | None = None,
) -> list[SearchResult]:
    """Convert raw Qdrant hits into wire-shape SearchResult entries.

    `favorite_ids` and `dislike_ids` are pre-resolved when the
    caller already has them (avoids a second DB round-trip on the
    hot /api/search path); when omitted, they're resolved in
    parallel via the IndexDB.

    Each hit's `score_str` is formatted to 3 decimals so SSR +
    JS render identically. `blurhash`, `width`, `height` come
    from the Qdrant payload (set at index time).
    """
    if favorite_ids is None or dislike_ids is None:
        ids = [h.id for h in hits]
        fav_set, dis_set = await asyncio.gather(
            favorite_id_set(index_db, ids),
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
            url=resolve_url(h.id, cfg.web_ui_url),
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


async def resolve_filename_filter(
    qdrant: Any,
    *,
    cfg: Any,
    pattern: str,
) -> tuple[Any | None, str | None]:
    """
    Translate a raw `?filename=` pattern to a Qdrant filter condition.

    Returns one of three outcomes (always a 2-tuple):

      (None, None)            — no filter requested (pattern was
                                empty), OR the cardinality guard
                                decided to skip the filter because
                                the pattern covers > 50% of the
                                corpus. Caller passes `None` to
                                qdrant.search meaning "no filter".
      (<condition>, None)     — a Qdrant filter condition to AND
                                into the query. The condition is
                                evaluated server-side against the
                                native text index on the `path`
                                payload field.
      (None, "bad_request")   — pattern was syntactically invalid
                                (caller surfaces a 400).

    The condition is a native Qdrant text-index match rather than a
    materialised id list. The old implementation asked the SQLite
    FTS5 index for the matching ids and shipped them to Qdrant as a
    `HasId` filter — up to ~143k ids serialised into every request
    for a broad prefix pattern. The native index does the matching
    inside Qdrant, so nothing crosses the wire and the per-request
    cost no longer scales with how many photos the pattern matches.

    The matched sets are identical: measured against the live
    corpus, the native filter returns the same ids as FTS5 for
    every pattern tested (see `scripts/parity_harness.py`).

    The cardinality guard is the load-bearing piece: when a
    pattern like `2024` matches 60% of the collection, applying
    the filter to the rest actively hurts search quality (HNSW
    has to pick the top-K from a non-uniform distribution) without
    meaningfully narrowing anything. By dropping the filter above
    the guard threshold we get "full ranking + 0 wasted work",
    which is always at least as good as the filtered ranking.

    The guard's count comes from Qdrant's own filtered count, so
    the guard and the filter always agree on what the corpus
    contains. (The old implementation compared against the SQLite
    cache's row count, which can drift from the collection the
    filter actually runs against.)
    """
    from search._filename_filter import resolve_filename_filter as _resolve

    return await _resolve(qdrant, cfg=cfg, pattern=pattern)


def surprise_search(hits: list[SearchHit], k: int) -> list[SearchHit]:
    """Shuffle hits and return up to k. Non-deterministic.

    Used by /api/search's Surprise Me mode — the user gets a
    random sample from the top-N candidates rather than the
    strict ranking. Without this the "Surprise Me" UI is
    indistinguishable from the standard top-K view.
    """
    shuffled = list(hits)
    random.shuffle(shuffled)
    return shuffled[:k]


def search_query_string(
    q: str,
    positives: list[str],
    negatives: list[str],
    collections: list[str],
    view: str = "grid",
    centroid: str | None = None,
    favorites: bool = False,
    centroids: list[str] | None = None,
    weights: list[float] | None = None,
    diverse: bool = False,
    diversity_mode: str = "off",
    diversity_depth: str = "auto",
    filename: str = "",
) -> str:
    """
    Build a multi-value search-state query string for photo back links.

    `view` is included so the user lands back on the same view they
    came from. We omit it when it's the default ('grid') to keep
    canonical URLs clean. Centroid state round-trips via repeated
    `?centroid=` params (with `?weights=` when not all-equal) so the
    "view tile → back" path lands on the same centroid search the
    user came from, not a bare `/`.

    `centroids` is the canonical input — a list of centroid names
    in blend order. The legacy `centroid` param is kept for
    single-centroid call sites and emits one `?centroid=` param.
    When `centroids` is supplied (even length 1), it takes
    precedence — the function never emits the same centroid twice
    from both inputs.

    `filename` round-trips the path-substring filter so the
    "view tile → back" link returns to the same narrowed search
    the user came from. Empty / whitespace-only is omitted so
    canonical URLs stay clean. The filter is single-valued, so a
    plain `?filename=` param (no list).
    """
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    params.extend(("positives", p) for p in positives)
    params.extend(("negatives", n) for n in negatives)
    params.extend(("collection", c) for c in collections)
    if centroids is not None:
        params.extend(("centroid", c) for c in centroids)
    elif centroid:
        params.append(("centroid", centroid))
    if weights is not None and any(w != 1.0 for w in weights):
        params.append(("weights", ",".join(str(w) for w in weights)))
    if favorites:
        params.append(("favorites", "true"))
    if filename.strip():
        params.append(("filename", filename.strip()))
    if view and view != "grid":
        params.append(("view", view))
    if diversity_mode and diversity_mode != "off":
        params.append(("diversity", diversity_mode))
        if diversity_depth and diversity_depth != "auto":
            params.append(("diversity_depth", diversity_depth))
    elif diverse:
        # Legacy callers that only know the boolean retain the old
        # URL shape; current search pages emit the explicit mode.
        params.append(("diverse", "true"))
    return urlencode(params)


def normalize_prompt_state(
    cfg: Any,
    q: str,
    positives_raw: list[str],
    negatives_raw: list[str],
) -> Any:
    """
    Normalize q/positive/negative prompt inputs for search.

    Display text is preserved for response/template echo. Dedupe is
    case-insensitive per side, overlong prompts are dropped, and q is
    appended to positives if it is a usable non-duplicate prompt.
    """
    from search.models import PromptState  # local import to avoid circular dep at module load

    max_prompt_chars = cfg.max_prompt_chars
    max_prompts_total = cfg.max_prompts_total

    effective_q = (q or "").strip()
    positive_keys: set[str] = set()
    negative_keys: set[str] = set()
    positive_entries: list[tuple[str, bool]] = []
    negative_entries: list[tuple[str, bool]] = []

    def add_positive(text: str) -> None:
        prompt = text.strip()
        key = prompt.lower()
        if not prompt or len(prompt) > max_prompt_chars or key in positive_keys:
            return
        positive_keys.add(key)
        positive_entries.append((prompt, True))

    def add_negative(text: str) -> None:
        prompt = text.strip()
        key = prompt.lower()
        if not prompt or len(prompt) > max_prompt_chars or key in negative_keys:
            return
        negative_keys.add(key)
        negative_entries.append((prompt, True))

    for prompt in positives_raw:
        add_positive(prompt)
    if effective_q:
        prompt = effective_q
        key = prompt.lower()
        if len(prompt) <= max_prompt_chars and key not in positive_keys:
            positive_keys.add(key)
            positive_entries.append((prompt, False))
    for prompt in negatives_raw:
        add_negative(prompt)

    remaining = max_prompts_total
    capped_positive_entries = positive_entries[:remaining]
    remaining -= len(capped_positive_entries)
    capped_negative_entries = negative_entries[:remaining]
    positives = [prompt for prompt, _explicit in capped_positive_entries]
    negatives = [prompt for prompt, _explicit in capped_negative_entries]
    return PromptState(
        q=effective_q,
        positives=positives,
        negatives=negatives,
        positive_chips=[
            prompt for prompt, explicit in capped_positive_entries if explicit
        ],
        negative_chips=[
            prompt for prompt, explicit in capped_negative_entries if explicit
        ],
    )


def diversity_page(
    cfg: Any,
    qdrant: Any,
    diversity_cache: Any,
    *,
    vector: list[float],
    effective_limit: int,
    offset: int,
    collections: list[str],
    allowed_ids: list[str] | None,
    favorite_ids: set[str] | None,
    mode: str,
    strength: float,
    depth: str,
    pool_depth: int,
    path_filter: Any | None = None,
) -> tuple[list[SearchHit], bool, Any]:
    """Build or retrieve one complete, stable Diversity ordering.

    Wraps the closure that was previously in `create_app`. The
    diversity_cache is passed in so callers control cache lifecycle
    (tests use a fresh in-memory cache).
    """
    from search._result_helpers import diversity_metadata as _diversity_metadata
    from search.diversity import (
        DiversityStats,
        rank_diverse,
        relevance_drop_for_mode,
    )
    from search.diversity_compute import _collapse_duplicate_indices

    cache_key = diversity_cache_key(
        cfg,
        vector, mode, depth, pool_depth, collections, allowed_ids, favorite_ids,
        path_filter,
    )
    cached = diversity_cache.get(cache_key)
    if cached is not None:
        hits = list(cached.hits)
        page = hits[offset:offset + effective_limit]
        return page, len(hits) > offset + effective_limit, _diversity_metadata(cached.stats)

    search_allowed_ids = allowed_ids
    if favorite_ids is not None:
        favorite_list = sorted(favorite_ids)
        if search_allowed_ids is None:
            search_allowed_ids = favorite_list
        else:
            favorite_set = set(favorite_list)
            search_allowed_ids = [
                point_id for point_id in search_allowed_ids
                if point_id in favorite_set
            ]
        if not search_allowed_ids:
            stats = DiversityStats(
                requested=True, applied=True, mode=mode, strength=strength,
                depth=depth, pool_depth=0,
            )
            return [], False, _diversity_metadata(stats)

    # Payload-first reordering.
    #
    # The collapse that drives diversity (content_sha256 + dhash) needs only
    # the payload fields `dhash` and `content_sha256` — it does NOT need the
    # 1152-dim vector. Fetching vectors for the full pool before collapsing
    # wastes most of the wire transfer: at depth 5000, ~40% of candidates
    # collapse away, so we re-download 40% of vectors we immediately discard.
    #
    # Reordered pipeline:
    #   1. payload-only top-K from Qdrant (returns hits with payload, no vectors)
    #   2. dhash/content_sha256 collapse on the payload-only hits (no transfer)
    #   3. vectors for the surviving ids only (~60% of the original transfer)
    #   4. rank_diverse over the (hit, vector) pairs — same contract as before.
    #
    # This preserves the diversity ranking byte-for-byte (verified: 24/24 top
    # agreement with the original fetch-with-vectors path on prod Qdrant) while
    # cutting the fetch step from ~2300ms to ~150ms at depth 5000.
    pool_depth = _pool_depth_for(cfg, mode, pool_depth)
    hits, _ = qdrant.search(
        vector,
        limit=pool_depth,
        offset=0,
        collections=collections or None,
        allowed_ids=search_allowed_ids,
        path_filter=path_filter,
    )
    if not hits:
        stats = DiversityStats(
            requested=True, applied=True, mode=mode, strength=strength,
            depth=depth, pool_depth=pool_depth,
        )
        diversity_cache.put(cache_key, [], stats)
        return [], False, _diversity_metadata(stats)

    # Collapse runs against the payload-only hits — only dhash and
    # content_sha256 are read, both indexed payload fields. `query_scores`
    # is the relevance signal (lower scores mean "less similar to the
    # query") and breaks ties when multiple candidates collapse to one
    # representative.
    query_scores = [float(h.score) for h in hits]
    keep_indices = _collapse_duplicate_indices(
        hits,
        query_scores=query_scores,
        duplicate_hamming_distance=cfg.diversity_duplicate_hamming_distance,
    )
    survivor_ids = [hits[i].id for i in keep_indices]

    # retrieve_batch_with_vectors returns pairs in Qdrant-order, NOT the
    # order requested. Re-attach the original hit metadata via id.
    raw_pairs = qdrant.retrieve_batch_with_vectors(survivor_ids)
    by_id = {h.id: h for h in hits}
    ordered_hits: list[Any] = []
    ordered_vectors: list[list[float]] = []
    for pid, vec in raw_pairs:
        hit = by_id.get(pid)
        if hit is None:
            continue
        ordered_hits.append(hit)
        ordered_vectors.append(vec)

    # Fast path: stack the per-hit vectors into a single (N, D) float32
    # matrix and hand it to rank_diverse via the (hits, ndarray) shape.
    # This skips the `_as_float_list` Python list comprehension inside
    # rank_diverse that does `float(x)` for every element (~3.3M calls
    # at depth 5000). The fast path is byte-identical to the legacy path
    # when given the same inputs (verified by a regression test).
    import numpy as _np
    vectors_ndarray = _np.asarray(ordered_vectors, dtype=_np.float32)

    ranking = rank_diverse(
        (ordered_hits, vectors_ndarray),
        vector,
        mode=mode,
        strength=strength,
        duplicate_hamming_distance=cfg.diversity_duplicate_hamming_distance,
        relevance_drop=relevance_drop_for_mode(
            mode, cfg.diversity_relevance_drop,
        ),
        max_results=cfg.max_results_total,
        depth=depth,
        pool_depth=len(hits),
    )
    diversity_cache.put(cache_key, ranking.hits, ranking.stats)
    page = ranking.hits[offset:offset + effective_limit]
    return page, len(ranking.hits) > offset + effective_limit, _diversity_metadata(ranking.stats)


def _pool_depth_for(cfg: Any, mode: str, requested: int) -> int:
    """Resolve the candidate-pool depth used by the diverse re-ranker.

    The mode-specific overrides on cfg.diversity_pool_depths win
    over the user's `pool_depth` query value when the user didn't
    explicitly request one (pool_depth <= 0). Falls back to the
    raw `requested` for unknown modes.
    """
    overrides = cast(dict[str, int], getattr(cfg, "diversity_pool_depths", {}) or {})
    if requested > 0:
        return requested
    return overrides.get(mode, requested or 500)


def centroid_diversity_page(
    cfg: Any,
    qdrant: Any,
    diversity_cache: Any,
    *,
    vector: list[float],
    effective_limit: int,
    offset: int,
    collections: list[str],
    allowed_ids: list[str] | None,
    seed_ids: list[str],
    mode: str,
    strength: float,
    depth: str,
    pool_depth: int,
    path_filter: Any | None = None,
) -> tuple[list[SearchHit], bool, Any]:
    """Diversity ranking for centroid/album search — the Layer-2-aware
    analogue of `diversity_page`.

    Differences from the text-search path:

      * Seed exclusion is load-bearing here: results must not echo the
        album's own members. Layer 1 (`exclude_ids=seed_ids`) rides
        along with the Qdrant call; Layer 2 drops candidates that are
        *near*-dups of a seed vector (threshold calibrated from the
        seed set's own intra-cluster distances). Layer 2 needs the
        candidate vectors, so this path fetches with vectors up front
        instead of payload-first — at pool sizes <= 5000 the extra
        transfer is acceptable and the ranking stays cache-stable.
      * No favorites intersection (centroid routes don't take
        `?favorites=`); allowed_ids passes through as-is.

    Paging contract matches `diversity_page`: one complete stable
    ranking per cache key, sliced for `?offset`. Sample mode never
    reaches this function (the route rejects sample+diversity with
    400), so cache keys are always deterministic.
    """
    from search._result_helpers import diversity_metadata as _diversity_metadata
    from search.centroids import (
        calibrate_near_dup_threshold,
        filter_near_duplicates,
    )
    from search.diversity import (
        DiversityStats,
        rank_diverse,
        relevance_drop_for_mode,
    )

    cache_key = diversity_cache_key(
        cfg,
        vector, mode, depth, pool_depth, collections, allowed_ids, None,
    )
    cached = diversity_cache.get(cache_key) if diversity_cache is not None else None
    if cached is not None:
        hits = list(cached.hits)
        page = hits[offset:offset + effective_limit]
        return page, len(hits) > offset + effective_limit, _diversity_metadata(cached.stats)

    actual_pool_depth = _pool_depth_for(cfg, mode, pool_depth)
    pairs, _ = qdrant.search_with_vectors(
        vector,
        actual_pool_depth,
        0,
        collections or None,
        allowed_ids,
        seed_ids or None,
        path_filter,
    )

    # Layer 2: drop candidates that are near-duplicates of a seed
    # photo. Static centroids have no seed_ids and skip this.
    if pairs and seed_ids:
        seed_pairs = qdrant.retrieve_batch_with_vectors(seed_ids)
        seed_vecs = [v for _, v in seed_pairs]
        if seed_vecs:
            threshold = calibrate_near_dup_threshold(seed_vecs)
            keep_mask = filter_near_duplicates(
                [v for _, v in pairs], seed_vecs, threshold,
            )
            pairs = [
                p for p, keep in zip(pairs, keep_mask, strict=False) if keep
            ]

    if not pairs:
        stats = DiversityStats(
            requested=True, applied=True, mode=mode, strength=strength,
            candidate_count=0, result_count=0,
            depth=depth, pool_depth=actual_pool_depth,
        )
        if diversity_cache is not None:
            diversity_cache.put(cache_key, [], stats)
        return [], False, _diversity_metadata(stats)

    import numpy as _np
    ordered_hits = [h for h, _ in pairs]
    vectors_ndarray = _np.asarray([v for _, v in pairs], dtype=_np.float32)

    ranking = rank_diverse(
        (ordered_hits, vectors_ndarray),
        vector,
        mode=mode,
        strength=strength,
        duplicate_hamming_distance=cfg.diversity_duplicate_hamming_distance,
        relevance_drop=relevance_drop_for_mode(
            mode, cfg.diversity_relevance_drop,
        ),
        max_results=cfg.max_results_total,
        depth=depth,
        pool_depth=len(pairs),
    )
    if diversity_cache is not None:
        diversity_cache.put(cache_key, ranking.hits, ranking.stats)
    page = ranking.hits[offset:offset + effective_limit]
    return page, len(ranking.hits) > offset + effective_limit, _diversity_metadata(ranking.stats)


def _digest_values(values: Iterable[str] | None) -> str:
    """SHA-256 digest of a list/set/None of strings — used to hash
    request-shape inputs into the diversity cache key.

    Stable order (sorted) + UTF-8 replacement, so equivalent
    inputs in any order produce the same key. None maps to
    a fixed sentinel so the absence of a filter is still part
    of the cache key.
    """
    import hashlib
    digest = hashlib.sha256()
    if values is None:
        digest.update(b"<none>\0")
    for value in sorted(str(item) for item in (values or [])):
        digest.update(value.encode("utf-8", "replace"))
    return digest.hexdigest()


def _digest_filter(path_filter: Any | None) -> str:
    """SHA-256 digest of a Qdrant filter condition — used to hash a
    request-shape input into the diversity cache key.

    The filter is a Pydantic model, so its canonical JSON form is
    stable across requests that build an equivalent condition. None
    maps to the same sentinel as `_digest_values(None)`, so a request
    without a filename filter never shares a cache entry with one
    that has it.
    """
    import hashlib
    if path_filter is None:
        return _digest_values(None)
    try:
        canonical = path_filter.model_dump_json(exclude_none=True)
    except AttributeError:
        canonical = repr(path_filter)
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:32]


def diversity_cache_key(
    cfg: Any,
    vector: list[float],
    mode: str,
    depth: str,
    pool_depth: int,
    collections: list[str],
    allowed_ids: list[str] | None,
    favorite_ids: set[str] | None,
    path_filter: Any | None = None,
) -> str:
    """Build the cache key for one Diversity ordering request.

    The key includes the collection name, mode/depth knobs,
    a digest of the query vector, and digests of the filter
    inputs. Two requests that produce identical Diversity
    rankings must hash to the same key.
    """
    import hashlib
    vector_digest = hashlib.sha256(
        repr(tuple(round(float(value), 8) for value in vector)).encode("ascii")
    ).hexdigest()[:20]
    return "|".join((
        cfg.qdrant_collection,
        mode,
        depth,
        str(pool_depth),
        vector_digest,
        _digest_values(collections),
        _digest_values(allowed_ids),
        _digest_values(favorite_ids),
        _digest_filter(path_filter),
    ))


async def materialize_search_page(
    cfg: Any,
    qdrant: Any,
    snapshot_cache: Any,
    *,
    vector: list[float],
    offset: int,
    limit: int,
    collections: list[str],
    allowed_ids: list[str] | None,
    favorite_ids: set[str] | None,
    path_filter: Any | None = None,
) -> tuple[list[SearchHit], bool]:
    """Return one page of a *stable* ranking, plus `has_more`.

    Drop-in replacement for the plain-search call
    `qdrant.search(vec, limit, offset, ...)`, which is unsound on HNSW:
    each page is ranked at depth `offset+limit`, so neighbouring pages
    disagree near their boundary and return overlapping ids. Measured on
    prod (2.05M points) a walk to offset 480 yielded 78 duplicate ids.

    Instead: freeze one ranking per query and slice pages from it.

      - The snapshot holds `(id, score)` only. Payloads are hydrated per
        page via `retrieve_batch` (~2ms for 24 ids), which keeps
        favourite/dislike flags live rather than frozen for the TTL.
      - The snapshot grows in *bands*. Page 1 fetches a shallow initial
        band (~10ms); deeper pages append bands fetched with
        `exclude_ids` = everything frozen so far. A band is amortised
        over ~20 pages, so per-page cost stays ~10-35ms at any depth,
        versus 179ms/473ms for a single 5k/10k-deep fetch on page 1.
      - Growth stops at `search_max_snapshot_size`; past that
        `has_more` goes False rather than degrading further.

    Returns `(hits, has_more)` where `hits` are `SearchHit` with the
    snapshot's score restored (`retrieve_batch` returns score=0.0).
    """
    from search.search_snapshot import snapshot_key

    cap = cfg.search_max_snapshot_size

    # Never grow a snapshot deeper than the request needs, and never
    # past the cap. An out-of-range offset must not trigger a deep walk.
    if offset >= cap:
        return [], False
    target = min(offset + limit, cap)

    key = snapshot_key(
        collection=cfg.qdrant_collection,
        vector=vector,
        collections=collections,
        allowed_ids=allowed_ids,
        favorite_ids=favorite_ids,
        band_size=cfg.search_band_size,
        path_filter=path_filter,
    )
    snap = await asyncio.to_thread(snapshot_cache.get_or_create, key)

    def _grow_to(wanted: int) -> None:
        """Grow the snapshot until it holds `wanted` ids (or is exhausted).

        Runs under the snapshot's own lock so two concurrent page
        requests for the same query don't fetch the same band twice —
        a racer waits for the in-flight band instead of duplicating it.
        The cache's dict lock is separate and never held across a fetch.
        """
        with snap.lock:
            first_band = len(snap) == 0
            while len(snap) < wanted and not snap.exhausted and len(snap) < cap:
                band = (
                    cfg.search_initial_band_size if first_band
                    else cfg.search_band_size
                )
                first_band = False
                band = min(band, cap - len(snap))
                hits, _ = qdrant.search(
                    vector,
                    band,
                    0,  # always rank from the top of what remains
                    collections or None,
                    allowed_ids,
                    list(snap.ids) or None,
                    path_filter,
                )
                if not hits:
                    snap.exhausted = True
                    break
                # extend() marks exhausted on a short band, and dedupes
                # locally — so a silently-dropped exclusion filter can
                # never spin this loop (it adds 0, flags exhausted, exits).
                snap.extend(hits, band)

    await asyncio.to_thread(_grow_to, target)

    page_pairs = snap.page(offset, limit)
    snapshot_len = len(snap)
    has_more = (
        snapshot_len > offset + limit
        # Exactly filled the page and Qdrant may have more to give.
        or (not snap.exhausted and snapshot_len < cap)
    )
    if not page_pairs:
        return [], has_more

    page_ids = [pid for pid, _ in page_pairs]
    score_by_id = dict(page_pairs)
    hydrated = await asyncio.to_thread(qdrant.retrieve_batch, page_ids)
    by_id = {h.id: h for h in hydrated}

    # Rebuild in snapshot order. `retrieve_batch` drops ids it can't
    # find (photo pruned mid-session) — skipping them here is what keeps
    # a dead tile from rendering, and has_more is computed from the
    # snapshot rather than the hydrated length so one dropped id can't
    # make the client think the result set ended.
    hits_out: list[SearchHit] = []
    for pid in page_ids:
        hit = by_id.get(pid)
        if hit is None:
            continue
        hit.score = score_by_id[pid]
        hits_out.append(hit)

    _prefetch_next_band(
        cfg,
        qdrant,
        snap,
        vector=vector,
        collections=collections,
        allowed_ids=allowed_ids,
        path_filter=path_filter,
        offset=offset,
        limit=limit,
        cap=cap,
    )
    return hits_out, has_more


def _prefetch_next_band(
    cfg: Any,
    qdrant: Any,
    snap: Any,
    *,
    vector: list[float],
    collections: list[str],
    allowed_ids: list[str] | None,
    offset: int,
    limit: int,
    cap: int,
    path_filter: Any | None = None,
) -> None:
    """Kick off a background fetch of the next band, if one will be needed.

    A cold band fetch costs ~0.5-1s once the exclusion set passes
    ~1750 ids (measured on 2.05M points), which would otherwise be a
    visible hitch roughly every `search_band_size / page_size` pages.
    Serving that cost in the background hides it behind the time the
    user spends looking at the current page — the SPA already prefetches
    two viewports ahead, so the next request usually finds the band
    already frozen.

    Fire-and-forget on a daemon thread: this is pure optimisation, so a
    failure must never affect the response already returned. The snapshot
    it mutates lives in the cache beyond this request's lifetime, which
    is the whole point.
    """
    # Kill-switch. Defaults to off when the attribute is absent so a
    # MagicMock cfg (tests) doesn't accidentally trigger background
    # threads; the real Config sets it True. Tests disable it explicitly
    # for deterministic call-count assertions.
    if not getattr(cfg, "search_prefetch_next_band", False):
        return
    if snap.exhausted or len(snap) >= cap:
        return
    # Only prefetch when the served page consumed the end of what is
    # frozen; otherwise there is slack already available and the next
    # page needs no fetch at all.
    if len(snap) > offset + limit:
        return
    # At most one background band per snapshot. claim_prefetch() is
    # guarded by its own lock, never by snap.lock (which may be held
    # across a long fetch).
    if not snap.claim_prefetch():
        return

    wanted = min(len(snap) + cfg.search_band_size, cap)
    band_size = cfg.search_band_size

    def _run() -> None:
        try:
            with snap.lock:
                while len(snap) < wanted and not snap.exhausted and len(snap) < cap:
                    band = min(band_size, cap - len(snap))
                    hits, _ = qdrant.search(
                        vector,
                        band,
                        0,  # rank from the top of what remains
                        collections or None,
                        allowed_ids,
                        list(snap.ids) or None,
                        path_filter,
                    )
                    if not hits:
                        snap.exhausted = True
                        break
                    snap.extend(hits, band)
        except Exception:
            # Optimisation only — never propagate to a served response.
            logger.debug("background band prefetch failed", exc_info=True)
        finally:
            snap.release_prefetch()

    threading.Thread(
        target=_run, daemon=True, name="search-band-prefetch"
    ).start()
