"""
search/routers/centroids_search.py — /api/centroids/{name}/search (§B2 step 15).

GET /api/centroids/{name}/search:
    Search using a loaded centroid as the query vector. Mutually
    exclusive with text prompts (the URL shape carries no prompt
    params, so the only failure mode is an unknown centroid name).

The route hands the request straight to Qdrant for the vector
search; it doesn't touch the text encoder chain. Two non-trivial
behaviours:

1. **Two-layer near-duplicate exclusion for dynamic centroids.**
   When the centroid has `seed_ids` (favourites or album
   members), the results must NOT echo the inputs:
     - Layer 1 — exact-id `must_not` at Qdrant: cheap, kills
       exact-seed matches at the filter level so the over-fetch
       doesn't waste bandwidth.
     - Layer 2 — numpy post-pass on candidate vectors: for each
       candidate, compute its cosine distance to the NEAREST
       seed vector and drop hits tighter than a threshold
       calibrated from the seed set's OWN intra-cluster pairwise
       distances (the "how-close-do-two-versions-of-the-same-photo-get"
       scale for THIS centroid).
   Both layers no-op for static `.pt` centroids (no seed_ids).

2. **Over-fetch 3x for Layer 2 headroom.** We ask Qdrant for
   `effective_limit * 3` candidates so that after Layer 2 drops
   near-dups we still have enough to trim back to `effective_limit`.
   For extreme cases where > 2/3 of the top results are near-dups
   of the seeds we'd under-fill the page — that's accepted as a
   rare edge (the alternative — refetching with a larger limit
   until we have enough — introduces its own failure mode).

3. **Sample-centroid mode (round-34, `?mode=sample`).** When
   `mode=sample` is set on a dynamic centroid, the route picks
   a fresh random K-subset of the seed set and searches
   against the mean of THAT subset, instead of the full mean.
   Each request gets a fresh sample — there's no per-user
   cache, so refreshing the home page or re-issuing the query
   surfaces a different cluster centroid each time. This is
   how the "Surprise me" button on /albums avoids the stuck
   feel of the deterministic centroid.

   Static `.pt` centroids reject `mode=sample` with 400 — they
   have no source population to sample from (a `.pt` file is
   already a single precomputed vector). The mode param is
   silently ignored for the static case if the request would
   otherwise succeed; the route's 400 path is reserved for
   the explicit-ask case where a static centroid was named.

4. **Surprise Me on centroid search (round-91, `?surprise=true`).**
   When `surprise` is set, the route runs the same Gumbel-max
   deck flow as `/api/search?surprise=true` but with the
   centroid vector as the query. Same temperature / pool / seed
   / session contract; same `SurpriseDeckCache` shape; same
   `surprise_session` round-trip. Mutually exclusive with
   `mode=sample` (both randomize, pick one) and with
   `diversity` (MMR re-rank on a per-request randomized pool
   is meaningless). The seed ids of the centroid are passed
   to Qdrant as `exclude_ids` so the results don't echo back
   the album members / favourites. The `centroids=[name]`
   field of the cache key is what scopes each centroid's
   deck — different centroids produce different decks even
   with identical temperature/pool/seed.

Tests pin:
- 404 for unknown centroid names.
- 404 for "loaded but empty" dynamic centroids (the empty-state
  UI distinction).
- 400 for invalid limit/offset (manual validation, not 422).
- `exclude_ids` Layer 1 includes the seed ids when the centroid
  is dynamic.
- Static centroids skip Layer 1 / Layer 2 entirely.
- Round-91: surprise=true on a centroid returns a Surprise
  deck (session_id, session_total, surprise=true in the
  response). Same-seed=Same-deck, Different-seed=Different-deck.
- Round-91: surprise + mode=sample and surprise + diversity
  both 400 (mutex).
- Round-91: surprise_temperature out of [0, 1] 400, surprise_pool
  out of [200, 50000] 400, surprise_seed non-integer 400.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from search._indexed_helpers import (
    centroid_diversity_page,
    resolve_filename_filter,
    results_from_hits,
)
from search._logging import log_qdrant_error, log_qdrant_timeout
from search._result_helpers import (
    bad_request,
    coerce_view,
    internal_error,
    parse_collections,
    parse_filename,
    qdrant_timeout,
    qdrant_unreachable,
)
from search.centroids import (
    calibrate_near_dup_threshold,
    filter_near_duplicates,
)
from search.centroids_compute import (
    DEFAULT_CLUSTER_SAMPLE_N,
    DEFAULT_SAMPLE_K,
    cluster_then_sample_centroid,
)
from search.diversity import (
    resolve_depth,
    resolve_mode,
)
from search.models import DiversityMetadata, SearchResponse, SearchResult
from search.surprise_deck import (
    DEFAULT_POOL_SIZE,
    DEFAULT_TEMPERATURE,
    POOL_SIZE_MAX,
    POOL_SIZE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    SurpriseDeckCache,
    build_surprise_deck,
    compute_cache_key,
)

logger = logging.getLogger(__name__)


# Round‑34: ?mode= discriminator for the search route. Static
# `.pt` centroids have no source population to sample from, so
# they 400 on `mode=sample` rather than silently doing the same
# thing as `mode=centroid`. Dynamic centroids (likes, dislikes,
# album:{id}) accept both.
_VALID_MODES = frozenset({"centroid", "sample"})


def build_centroids_search_router(
    *,
    qdrant: Any,
    cfg: Any,
    index_db: Any,
    centroid_store: Any,
    dynamic_centroids: Any,
    diversity_cache: Any = None,
) -> APIRouter:
    """Build the /api/centroids/{name}/search router.

    Dependencies:
      - `qdrant`: QdrantSearch wrapper.
      - `cfg`:    search Config (limits, default view, web_ui_url).
      - `index_db`: search-side IndexDB cache (filename lookup).
      - `centroid_store`: static centroid store (may be None).
      - `dynamic_centroids`: dynamic centroid registry (may be None).
      - `diversity_cache`: DiversityResultCache shared with /api/search
        (may be None — then diversity ranking still works but results
        are not cached across requests/pagination).
    """
    router = APIRouter()
    # Process-local cache of pre-generated Surprise Me decks for
    # centroid searches. Keyed by the ETag hash of the params
    # (centroid name + temperature + pool + seed), so the same
    # centroid + knobs always hit the same deck and a different
    # seed always produces a new one. Mirrors the per-router
    # cache in build_search_router — process-local because
    # "give me a fresh deck on refresh" is the desired UX and
    # a shared cache would turn the cache into a sticky-session
    # failure mode on a load balancer.
    surprise_deck_cache = SurpriseDeckCache()

    @router.get("/api/centroids/{name}/search", response_model=SearchResponse)
    async def search_by_centroid(
        name: str,
        request: Request,
        limit: int = Query(cfg.top_k_default, description="max results"),
        offset: int = Query(0, description="offset into the full result set"),
        mode: str = Query(
            "centroid",
            description=(
                "Retrieval mode. 'centroid' (default) uses the full mean "
                "of the seed set. 'sample' picks a random K-subset of the "
                "seeds and uses the mean of THAT subset. Each request "
                "re-rolls, so refreshing surfaces a different cluster."
            ),
        ),
        sample_k: int = Query(
            DEFAULT_SAMPLE_K,
            description=(
                "Cluster count (k-means K) for sample mode. Defaults to "
                f"{DEFAULT_SAMPLE_K}. Only used when mode=sample. Round-75."
            ),
        ),
        sample_n: int = Query(
            DEFAULT_CLUSTER_SAMPLE_N,
            description=(
                "Number of cluster centroids to average per request. "
                f"Defaults to {DEFAULT_CLUSTER_SAMPLE_N}. Must satisfy 1 <= n <= sample_k. "
                "Only used when mode=sample. Round-75."
            ),
        ),
        diverse: bool = Query(False, description="apply MMR diversity re-ranking"),
        diversity: str | None = Query(
            None, description="Diversity strength: off, low, balanced, or high",
        ),
        diversity_depth: str | None = Query(
            None, description="Diversity candidate depth: auto, 500, 1000, 2000, 5000, or 10000",
        ),
        surprise: bool = Query(
            False,
            description=(
                "Surprise Me — Gumbel-max over the top N candidates from "
                "this centroid. Mutually exclusive with mode=sample and "
                "with diversity. Same cache+session contract as "
                "/api/search?surprise=true."
            ),
        ),
        surprise_temperature: float = Query(
            DEFAULT_TEMPERATURE,
            description=(
                "Gumbel noise scale for Surprise Me on centroid search. "
                "T=0 is deterministic top-K; higher = more variety. "
                "Ignored unless surprise=true."
            ),
        ),
        surprise_pool: int = Query(
            DEFAULT_POOL_SIZE,
            description=(
                "How many top candidates the Gumbel draw is taken from "
                "for centroid search. Larger = more variety, slower "
                "first page. Range 200..50000, default 5000. Ignored "
                "unless surprise=true."
            ),
        ),
        surprise_seed: int | None = Query(
            None,
            description=(
                "Reseed knob for Surprise Me on centroid search. Pass "
                "a fresh random int for a new deck every request; pass "
                "the same value across requests for a shareable "
                "deterministic URL. The route generates a random int "
                "when this is absent so each new search gives a "
                "different deck by default. Ignored unless surprise=true."
            ),
        ),
        surprise_session: str | None = Query(
            None,
            description=(
                "Opaque session id from a prior Surprise Me response on "
                "this centroid. Pass back with offset to walk the cached "
                "deck without recomputing. Ignored unless surprise=true."
            ),
        ),
    ) -> SearchResponse | JSONResponse:
        """Search using a loaded centroid as the query vector."""
        if centroid_store is None:
            raise HTTPException(
                status_code=503, detail="centroid store not initialized",
            )
        # Validate the mode param up-front so a typo doesn't silently
        # behave like the default. The centroid source check below
        # narrows the rejection further (static .pt rejects sample).
        if mode not in _VALID_MODES:
            return bad_request(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )
        if sample_k <= 0:
            return bad_request("sample_k must be > 0")
        if sample_n <= 0:
            return bad_request("sample_n must be > 0")
        if sample_n > sample_k:
            return bad_request(
                f"sample_n ({sample_n}) must be <= sample_k ({sample_k})"
            )
        # Diversity resolution — identical contract to /api/search
        # (same helpers, same 400 messages).
        try:
            diversity_mode, diversity_strength = resolve_mode(diversity, diverse)
        except ValueError as exc:
            return bad_request(str(exc))
        try:
            diversity_depth_mode, diversity_pool_depth = resolve_depth(
                diversity_depth, diversity_mode,
            )
        except ValueError as exc:
            return bad_request(str(exc))
        diversity_active = diversity_mode != "off"
        # Sample mode re-rolls a different query vector per request, so
        # a cached stable ranking is meaningless (and would silently
        # defeat the re-roll). Mutual exclusion mirrors the
        # surprise+diversity rule on /api/search.
        if diversity_active and mode == "sample":
            return bad_request(
                "Diversity cannot be combined with sample mode "
                "(Surprise Me). Choose one retrieval mode."
            )
        # Round-91: Surprise Me on centroid search. Same Gumbel-max
        # algorithm as /api/search?surprise=true but with the
        # centroid vector as the query. Mutex with mode=sample
        # (both randomize the result, pick one) and with diversity
        # (MMR rerank on a per-request randomized pool is
        # meaningless). Surprise is allowed for both static and
        # dynamic centroids — a static .pt has a fixed vector but
        # the pool around it is still meaningful.
        if surprise and mode == "sample":
            return bad_request(
                "Surprise Me cannot be combined with mode=sample. "
                "Choose one retrieval mode."
            )
        if surprise and diversity_active:
            return bad_request(
                "Diversity cannot be combined with Surprise Me. "
                "Choose one search mode."
            )
        # Surprise knob validation — lifted above the branch so the
        # ETag pre-compute (when added) and the branch see the same
        # parsed values. Mirrors the validation block in
        # /api/search. Centroid search has no `q` / `positives` /
        # `negatives`, so the cache-key input is just the centroid
        # name and the surprise knobs.
        surprise_temperature_parsed: float = DEFAULT_TEMPERATURE
        surprise_pool_parsed: int = DEFAULT_POOL_SIZE
        surprise_seed_parsed: int | None = None
        if surprise:
            try:
                surprise_temperature_parsed = float(surprise_temperature)
            except (TypeError, ValueError):
                return bad_request("surprise_temperature must be a number")
            if not (TEMPERATURE_MIN <= surprise_temperature_parsed <= TEMPERATURE_MAX):
                return bad_request(
                    f"surprise_temperature must be in "
                    f"[{TEMPERATURE_MIN}, {TEMPERATURE_MAX}], "
                    f"got {surprise_temperature_parsed}"
                )
            try:
                surprise_pool_parsed = int(surprise_pool)
            except (TypeError, ValueError):
                return bad_request("surprise_pool must be an integer")
            if not (POOL_SIZE_MIN <= surprise_pool_parsed <= POOL_SIZE_MAX):
                return bad_request(
                    f"surprise_pool must be in "
                    f"[{POOL_SIZE_MIN}, {POOL_SIZE_MAX}], "
                    f"got {surprise_pool_parsed}"
                )
            if surprise_seed is not None:
                try:
                    surprise_seed_parsed = int(surprise_seed)
                except (TypeError, ValueError):
                    return bad_request("surprise_seed must be an integer")
        # Look up static first; fall back to dynamic (registry does
        # lazy compute + cache). This keeps the route's contract
        # the same regardless of which backend the centroid came from.
        # `centroid_name` echoes the canonical form back to the client
        # (static centroids are stored lowercased, dynamic use the
        # registered name as-is).
        # `seed_ids` is the list of source point ids that fed the
        # centroid (favourite ids / album member ids). Empty for
        # static `.pt` centroids. Drives the two-layer near-dup
        # exclusion below: Layer 1 is the `exclude_ids` server-side
        # filter; Layer 2 is the numpy post-pass on candidate
        # vectors. Both no-op when `seed_ids` is empty (the static
        # case).
        vector: list[float] | None = None
        centroid_name = name
        seed_ids: list[str] = []
        static_spec = centroid_store.get(name)
        if static_spec is not None:
            # Sample mode requires a source population. A static
            # `.pt` centroid is a single precomputed vector — the
            # "subset" would be the same as the whole, and a 400
            # is clearer than silently behaving like centroid mode.
            if mode == "sample":
                return bad_request(
                    f"mode=sample is not available for static centroid "
                    f"{name!r}; only dynamic (album/likes/dislikes) "
                    f"centroids have a source population to sample from"
                )
            vector = static_spec.vector
            centroid_name = static_spec.name
        elif dynamic_centroids is not None:
            dyn = dynamic_centroids.get_vector(name)
            dyn_spec = dynamic_centroids.get_spec(name)
            if dyn is not None:
                full_vector, _n_images, full_seed_ids = dyn
                if dyn_spec is not None:
                    centroid_name = dyn_spec.name
                if mode == "sample":
                    # Round-34: re-pick a fresh K-subset and search
                    # against the mean of THAT subset, instead of the
                    # full mean. We pull the seed vectors from Qdrant
                    # (the registry stored ids but not vectors, since
                    # the existing centroid path doesn't need them
                    # after compute). No per-request cache: each call
                    # re-rolls, so refreshing the home page surfaces a
                    # different cluster each time.
                    if not full_seed_ids:
                        return bad_request(
                            f"centroid {name!r} has no seed ids; "
                            f"sample mode requires a non-empty source set"
                        )
                    seed_pairs = qdrant.retrieve_batch_with_vectors(
                        full_seed_ids,
                    )
                    picked_ids: list[str] = []
                    picked_vecs: list[list[float]] = []
                    for pid, vec in seed_pairs:
                        if not vec:
                            continue
                        picked_ids.append(pid)
                        picked_vecs.append(vec)
                    if not picked_vecs:
                        return bad_request(
                            f"could not retrieve seed vectors for "
                            f"centroid {name!r}"
                        )
                    # Round-75: cluster the seed set into K groups,
                    # then average N of those cluster centroids per
                    # request. Each cluster represents one visual
                    # mode of the album, so the sub-centroid blends
                    # those modes instead of relying on which photos
                    # happened to land in the random K-sample.
                    vector, _picked_count, picked_seed_ids = cluster_then_sample_centroid(
                        picked_ids, picked_vecs, k=sample_k, n=sample_n,
                    )
                    # Use the picked subset as the exclude list so
                    # the results don't echo back the sample itself.
                    # (The full centroid path excludes the FULL seed
                    # set; here the picked set IS what the centroid
                    # is "about", so excluding it is the analogous
                    # behaviour.)
                    seed_ids = picked_seed_ids
                else:
                    vector = full_vector
                    seed_ids = full_seed_ids
            else:
                # Distinguish "unknown name" from "known but empty":
                # the empty case surfaces a 404 too (treated as
                # "nothing to search against") so the UI can show its
                # own empty-state copy. Names not registered are
                # caught below.
                if dynamic_centroids.get_spec(name) is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"centroid {name!r} not loaded",
                    )
                raise HTTPException(
                    status_code=404,
                    detail=f"centroid {name!r} has no data yet",
                )
        else:
            raise HTTPException(
                status_code=404, detail=f"centroid {name!r} not loaded",
            )
        if vector is None:
            raise HTTPException(
                status_code=404, detail=f"centroid {name!r} not loaded",
            )
        # Manual validation (consistent with /api/search shape).
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return bad_request("limit must be an integer")
        if not (1 <= limit <= cfg.top_k_max):
            return bad_request(f"limit must be in [1, {cfg.top_k_max}]")
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            return bad_request("offset must be an integer")
        if offset < 0:
            return bad_request("offset must be >= 0")
        if offset >= cfg.max_results_total:
            return SearchResponse(
                query="",
                positives=[],
                negatives=[],
                view=coerce_view(cfg.default_view),
                centroid=centroid_name,
                results=[], took_ms=0, offset=offset, limit=0, has_more=False,
            )
        effective_limit = min(limit, cfg.max_results_total - offset)
        collections = parse_collections(request)
        filename_pattern = parse_filename(request)
        path_filter, fname_err = await resolve_filename_filter(
            qdrant, cfg=cfg, pattern=filename_pattern,
        )
        if fname_err == "bad_request":
            return bad_request(
                f"invalid filename pattern {filename_pattern!r}"
            )

        # Auto-roll a fresh seed when surprise is on but the client
        # didn't provide one. Same pattern as /api/search. Type
        # narrows to int so downstream callers see a real int.
        if surprise and surprise_seed_parsed is None:
            surprise_seed_parsed = secrets.randbelow(2**31)
        # Use 0 as a placeholder when not surprise — the session
        # branch below uses the session id directly and skips the
        # cache-key recompute when session is present.
        seed_for_cache_key: int = (
            surprise_seed_parsed if surprise_seed_parsed is not None else 0
        )
        # Pre-compute the Surprise Me cache key from the request
        # inputs. Same role as in /api/search: the key gates the
        # cache hit and doubles as the session id. For centroid
        # search the only "query" input is the centroid name —
        # there's no `q`, no positives/negatives, no favorites
        # flag. The `centroids=[centroid_name]` field on
        # compute_cache_key is what makes the deck
        # centroid-specific; different centroids get different
        # decks even with the same temperature/pool/seed.
        # `collections` is included so a different
        # `?collections=foo` filter produces a different deck
        # (the underlying pool is narrower).
        surprise_cache_key: str | None = (
            surprise_session
            if surprise and surprise_session
            else (
                compute_cache_key(
                    q="",
                    positives=[],
                    negatives=[],
                    centroids=[centroid_name],
                    weights=None,
                    collections=collections,
                    favorites=False,
                    filename="",
                    temperature=surprise_temperature_parsed,
                    pool_size=surprise_pool_parsed,
                    seed=seed_for_cache_key,
                )
                if surprise else None
            )
        )

        t0 = time.time()
        # `allowed_ids == []` short-circuit (see `/` and `/api/search`
        # for the rationale). `deck_ids` is declared up-front so every
        # branch (surprise / diversity / default) can populate it; the
        # default branch leaves it empty since it's only meaningful
        # for the surprise path.
        deck_ids: list[str] = []
        if diversity_active:
            # MMR diversity ranking over the centroid's candidate
            # pool, with the same two-layer seed exclusion (Layer 1
            # exclude_ids + Layer 2 near-dup post-pass) applied
            # inside. Cache-stable paging like /api/search.
            try:
                hits, has_more, diversity_meta = centroid_diversity_page(
                    cfg,
                    qdrant,
                    diversity_cache,
                    vector=vector,
                    effective_limit=effective_limit,
                    offset=offset,
                    collections=collections,
                    allowed_ids=None,
                    seed_ids=seed_ids or [],
                    mode=diversity_mode,
                    strength=diversity_strength,
                    depth=diversity_depth_mode,
                    pool_depth=diversity_pool_depth,
                    path_filter=path_filter,
                )
            except (ConnectionError, OSError) as e:
                log_qdrant_error(
                    logger, endpoint="/api/centroids/{name}/search",
                    op="diversity", error=e,
                )
                return qdrant_unreachable(str(e))
            except Exception as e:  # noqa: BLE001
                if "timeout" in type(e).__name__.lower() or "Timeout" in str(e):
                    log_qdrant_timeout(
                        logger, endpoint="/api/centroids/{name}/search",
                        op="diversity", timeout_ms=cfg.query_timeout_ms,
                    )
                    return qdrant_timeout(str(e))
                log_qdrant_error(
                    logger, endpoint="/api/centroids/{name}/search",
                    op="diversity", error=e,
                )
                return internal_error(str(e))
        elif surprise and surprise_cache_key is not None:
            # Round-91: Surprise Me on centroid search. Same
            # Gumbel-max algorithm as /api/search?surprise=true
            # but with the centroid vector as the query. Excludes
            # the centroid's seed ids at the Qdrant level so the
            # results don't echo the inputs (matches the
            # non-surprise centroid path's Layer 1 exclusion). No
            # Layer 2 near-dup post-pass — at high temperature
            # the Gumbel draw dilutes near-dups; at T=0 a few
            # near-dups may surface. Acceptable trade-off for
            # keeping the route simple.
            try:
                hits, has_more, deck_ids = await build_surprise_deck(
                    qdrant=qdrant,
                    vec=vector,
                    collections=collections,
                    surprise_allowed_ids=None,
                    path_filter=path_filter,
                    surprise_pool=surprise_pool_parsed,
                    surprise_temperature=surprise_temperature_parsed,
                    surprise_seed=(
                        surprise_seed_parsed if surprise_seed_parsed is not None else 0
                    ),
                    surprise_session=surprise_session,
                    surprise_cache_key=surprise_cache_key,
                    surprise_deck_cache=surprise_deck_cache,
                    effective_limit=effective_limit,
                    offset=offset,
                    exclude_ids=seed_ids or None,
                )
            except (ConnectionError, OSError) as e:
                log_qdrant_error(
                    logger, endpoint="/api/centroids/{name}/search",
                    op="surprise", error=e,
                )
                return qdrant_unreachable(str(e))
            except Exception as e:  # noqa: BLE001
                if "timeout" in type(e).__name__.lower() or "Timeout" in str(e):
                    log_qdrant_timeout(
                        logger, endpoint="/api/centroids/{name}/search",
                        op="surprise", timeout_ms=cfg.query_timeout_ms,
                    )
                    return qdrant_timeout(str(e))
                log_qdrant_error(
                    logger, endpoint="/api/centroids/{name}/search",
                    op="surprise", error=e,
                )
                return internal_error(str(e))
            diversity_meta = DiversityMetadata()
        else:
            diversity_meta = None
            # See module docstring for the over-fetch + two-layer
            # near-dup exclusion rationale.
            over_fetch_limit = min(
                effective_limit * 3, cfg.max_results_total - offset,
            )
            try:
                if seed_ids:
                    # Need vectors for the Layer 2 post-pass, so go
                    # through `search_with_vectors` (one extra
                    # `with_vectors=True` per hit, no second
                    # round-trip). The same `exclude_ids` Layer 1
                    # filter rides along.
                    pairs, _ = qdrant.search_with_vectors(
                        vector, limit=over_fetch_limit, offset=offset,
                        collections=collections or None,
                        allowed_ids=None,
                        exclude_ids=seed_ids,
                        path_filter=path_filter,
                    )
                    # Fetch the seed vectors themselves for the
                    # calibration + post-pass. Orphans (ids whose
                    # photo is gone from Qdrant) are silently
                    # dropped here — `retrieve_batch_with_vectors`
                    # omits missing ids from the response.
                    seed_pairs = qdrant.retrieve_batch_with_vectors(seed_ids)
                    seed_vecs: list[list[float]] = [
                        v for _, v in seed_pairs
                    ]
                    if seed_vecs:
                        threshold = calibrate_near_dup_threshold(seed_vecs)
                        cand_vecs = [vec for _, vec in pairs]
                        keep_mask = filter_near_duplicates(
                            cand_vecs, seed_vecs, threshold,
                        )
                        before_count = len(pairs)
                        kept_pairs = [
                            p for p, keep in zip(pairs, keep_mask, strict=False) if keep
                        ]
                        dropped = before_count - len(kept_pairs)
                        # Trim back to what the user asked for.
                        hits = [h for h, _ in kept_pairs[:effective_limit]]
                        # If anything was dropped, signal `has_more`
                        # so the user knows there might be more
                        # distinct results if they paginate. Also
                        # if we filled the limit and there were more
                        # candidates kept than what we showed.
                        if dropped > 0 or len(kept_pairs) > effective_limit:
                            has_more = True
                        else:
                            # Nothing dropped AND we filled the page
                            # — but Qdrant only gave us `before_count`
                            # candidates, not `over_fetch_limit`, so
                            # we can't use the over-fetched limit
                            # for the standard "hit the limit means
                            # more" heuristic. Use the user's
                            # `effective_limit`: if we filled it,
                            # there may be more; if we didn't, there
                            # isn't.
                            has_more = len(hits) >= effective_limit
                    else:
                        # Seed vectors weren't retrievable (all
                        # orphans) — skip Layer 2 and just trim.
                        # Don't apply exclude_ids post-hoc because
                        # Layer 1 already excluded them server-side.
                        hits = [h for h, _ in pairs[:effective_limit]]
                        has_more = (
                            len(pairs) > effective_limit
                            or len(pairs) >= effective_limit
                        )
                else:
                    # Static centroid (or empty dynamic): no
                    # near-dup exclusion. Original single-shot
                    # search path.
                    hits, has_more = qdrant.search(
                        vector, limit=effective_limit, offset=offset,
                        collections=collections or None,
                        allowed_ids=None,
                        path_filter=path_filter,
                    )
            except (ConnectionError, OSError) as e:
                log_qdrant_error(
                    logger, endpoint="/api/centroids/{name}/search",
                    op="search", error=e,
                )
                return qdrant_unreachable(str(e))
            except Exception as e:  # noqa: BLE001
                if "timeout" in type(e).__name__.lower() or "Timeout" in str(e):
                    log_qdrant_timeout(
                        logger, endpoint="/api/centroids/{name}/search",
                        op="search", timeout_ms=cfg.query_timeout_ms,
                    )
                    return qdrant_timeout(str(e))
                log_qdrant_error(
                    logger, endpoint="/api/centroids/{name}/search",
                    op="search", error=e,
                )
                return internal_error(str(e))
        took_ms = int((time.time() - t0) * 1000)
        # Resolve fav/dis for the SearchResult list via the indexed
        # helper. The dynamic-centroid path already pulled the seed
        # vectors from Qdrant; this is the parallel lookup against
        # the IndexDB cache.
        results: list[SearchResult] = await results_from_hits(
            index_db, cfg=cfg, hits=hits,
        )
        # Round-91: Surprise Me on centroid search echoes the
        # session id + total in the response so the client can
        # walk the deck page-by-page (same contract as
        # /api/search?surprise=true and /api/random).
        session_id_out = (
            surprise_cache_key
            if (surprise and surprise_cache_key and deck_ids)
            else None
        )
        session_total_out = (
            len(deck_ids) if (surprise and deck_ids) else None
        )
        return SearchResponse(
            query="",
            positives=[],
            negatives=[],
            diverse=diversity_active,
            diversity=diversity_meta if diversity_meta is not None
            else DiversityMetadata(depth="auto", pool_depth=0),
            surprise=surprise and bool(deck_ids),
            view=coerce_view(cfg.default_view),
            centroid=centroid_name,
            results=results,
            took_ms=took_ms,
            offset=offset,
            limit=limit,
            has_more=has_more,
            session_id=session_id_out,
            session_total=session_total_out,
        )

    return router
