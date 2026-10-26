"""
search/routers/search.py — /api/search (§B2 step 40).

GET /api/search:
    The main search endpoint. Composes the text encoder + centroid
    blend + diversity re-rank + filename filter into a single ranked
    result list.

This router is the largest of the B2 extractions (~260 lines).
Most helpers it needs were lifted to module level in steps 13-39.
The two remaining closure-bound helpers are passed in as factory
parameters:
- `resolve_query_vector`: encodes the prompt state (or blends
  centroids) into a single vector. Depends on `text_encoder`,
  `_dynamic_centroids`, `_centroid_store`, `_cfg`.
- `favorite_ids_for_filter`: async wrapper around
  `index_db.list_favorites(...)` capped to `_cfg.max_results_total`.

The router takes both factories as `Any`-typed callable slots so
the existing closure bindings work without further refactoring.
A follow-up commit can lift the two helpers to module level and
replace the factory slots with explicit arguments.

Cache semantics:
- 400 for bad input (manual validation, not 422)
- 200 + ETag for cacheable responses
- 304 for If-None-Match hits
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from search._indexed_helpers import (
    diversity_page,
    materialize_search_page,
    normalize_prompt_state,
    resolve_filename_filter,
    results_from_hits,
)
from search._logging import log_qdrant_error, log_qdrant_timeout
from search._result_helpers import (
    bad_request,
    coerce_view,
    internal_error,
    parse_centroids,
    parse_collections,
    parse_filename,
    parse_weights,
    qdrant_timeout,
    qdrant_unreachable,
)
from search.diversity import resolve_depth, resolve_mode
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


def build_search_router(
    *,
    qdrant: Any,
    cfg: Any,
    index_db: Any,
    diversity_cache: Any,
    snapshot_cache: Any,
    resolve_query_vector: Any,
    favorite_ids_for_filter: Any,
) -> APIRouter:
    """Build the /api/search router.

    Dependencies:
      - `qdrant`: QdrantSearch wrapper.
      - `cfg`:    search Config.
      - `index_db`: search-side IndexDB cache.
      - `diversity_cache`: DiversityResultCache instance (MMR path).
      - `snapshot_cache`: SearchSnapshotCache instance (plain path —
        holds the frozen rankings that make offset paging stable).
      - `resolve_query_vector`: closure-bound prompt/centroid
        vector resolver (passed through for now — §B2 step 41
        will lift it to module level).
      - `favorite_ids_for_filter`: closure-bound favorites-set
        resolver (passed through for now — §B2 step 42 will
        lift it to module level).
    """
    router = APIRouter()
    # Process-local cache of pre-generated Surprise Me decks. Keyed
    # by the ETag hash of the params, so the same query+temperature
    # +pool+seed always hits the same deck and a different seed
    # always produces a new one. Deliberately per-process — see
    # search/surprise_deck.py for the rationale.
    surprise_deck_cache = SurpriseDeckCache()

    @router.get("/api/search", response_model=SearchResponse)
    async def api_search(
        request: Request,
        q: str = Query("", description="text query"),
        limit: int = Query(cfg.top_k_default, description="max results"),
        offset: int = Query(0, description="offset into the full result set"),
        view: str = Query(cfg.default_view, description="result view: 'grid' or 'feed'"),
        favorites: bool = Query(False, description="restrict results to favourites"),
        diverse: bool = Query(False, description="apply MMR diversity re-ranking"),
        diversity: str | None = Query(
            None, description="Diversity strength: off, low, balanced, or high",
        ),
        diversity_depth: str | None = Query(
            None, description="Diversity candidate depth: auto, 500, 1000, 2000, 5000, or 10000",
        ),
        surprise: bool = Query(False, description="Surprise Me — random sample from deep pool"),
        surprise_temperature: float = Query(
            DEFAULT_TEMPERATURE,
            description="Gumbel noise scale for Surprise Me. T=0 is deterministic; higher = more variety.",
        ),
        surprise_pool: int = Query(
            DEFAULT_POOL_SIZE,
            description="How many top candidates the Gumbel draw is taken from. Larger = more variety, slower first page.",
        ),
        surprise_seed: int | None = Query(
            None,
            description="Reseed knob for Surprise Me. Omit (or pass a fresh random int) for a new deck every search; pass the same value across requests for a shareable deterministic URL. The route generates a random int when this is absent so each new search gives a different deck by default.",
        ),
        surprise_session: str | None = Query(
            None,
            description="Opaque session id from a prior Surprise Me response. Pass back with offset to walk the cached deck without recomputing.",
        ),
    ) -> JSONResponse:
        # Manual validation so we return 400 (not 422) for bad input.
        view = coerce_view(view)
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
        diverse = diversity_mode != "off"
        if surprise and diverse:
            return bad_request(
                "Diversity cannot be combined with Surprise Me. Choose one search mode."
            )
        prompt_state = normalize_prompt_state(
            cfg,
            q,
            [p.strip() for p in request.query_params.getlist("positives") if p.strip()],
            [p.strip() for p in request.query_params.getlist("negatives") if p.strip()],
        )
        active_centroids = parse_centroids(request)
        active_weights = parse_weights(request, len(active_centroids))
        active_centroid = active_centroids[0] if active_centroids else None
        filename_pattern = parse_filename(request)
        # Resolve the filename pattern into a native Qdrant filter
        # condition up front so any 400 lands before the long-running
        # Qdrant search. The condition is evaluated server-side
        # against the native text index on the `path` payload field,
        # so the matching photo ids never have to be materialised
        # and shipped back to Qdrant as a `HasId` filter.
        path_filter, fname_err = await resolve_filename_filter(
            qdrant, cfg=cfg, pattern=filename_pattern,
        )
        if fname_err == "bad_request":
            return bad_request(
                f"invalid filename pattern {filename_pattern!r}"
            )
        # Mutex: centroid search cannot coexist with text prompts.
        if active_centroids and (prompt_state.q or prompt_state.positives or prompt_state.negatives):
            return bad_request(
                "centroid search is exclusive — use ?centroid= or ?q=/?positives=, not both"
            )
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
            return JSONResponse(
                content=SearchResponse(
                    query="", positives=[], negatives=[],
                    diverse=diverse, surprise=surprise,
                    view=coerce_view(cfg.default_view),
                    centroid=active_centroid,
                    centroids=list(active_centroids),
                    weights=active_weights,
                    results=[], took_ms=0, offset=offset, limit=0, has_more=False,
                ).model_dump(),
                headers={"ETag": "deadbeef", "Cache-Control": "private, max-age=10"},
            )
        effective_limit = min(limit, cfg.max_results_total - offset)

        # Validate the surprise knobs up front (any branch — pure
        # surprise, surprise+favorites, or just favorites where
        # the request is rejected — needs the same bounds check).
        # We keep the *parsed* values in the locals so the ETag
        # pre-compute and the cache lookup agree.
        surprise_temperature_parsed: float = DEFAULT_TEMPERATURE
        surprise_pool_parsed: int = DEFAULT_POOL_SIZE
        # Default to None (random) so a fresh search gives a fresh
        # deck. The route generates an int from secrets when the
        # client didn't pass one. Existing callers that pass an
        # explicit seed still get the deterministic shareable URL
        # behaviour.
        surprise_seed_parsed: int | None = None
        if surprise:
            try:
                surprise_temperature_parsed = float(surprise_temperature)
            except (TypeError, ValueError):
                return bad_request("surprise_temperature must be a number")
            if not (TEMPERATURE_MIN <= surprise_temperature_parsed <= TEMPERATURE_MAX):
                return bad_request(
                    f"surprise_temperature must be in "
                    f"[{TEMPERATURE_MIN}, {TEMPERATURE_MAX}]; "
                    f"got {surprise_temperature_parsed}"
                )
            try:
                surprise_pool_parsed = int(surprise_pool)
            except (TypeError, ValueError):
                return bad_request("surprise_pool must be an integer")
            if not (POOL_SIZE_MIN <= surprise_pool_parsed <= POOL_SIZE_MAX):
                return bad_request(
                    f"surprise_pool must be in "
                    f"[{POOL_SIZE_MIN}, {POOL_SIZE_MAX}]; "
                    f"got {surprise_pool_parsed}"
                )
            # `surprise_seed` may be absent (None) or an integer.
            # The Qdrant corpus size caps the pool slider at the
            # upper end; query the count lazily and clamp the
            # requested pool to it.
            if surprise_seed is not None:
                try:
                    surprise_seed_parsed = int(surprise_seed)
                except (TypeError, ValueError):
                    return bad_request("surprise_seed must be an integer")
            # Cap pool to actual corpus size when oversized.
            try:
                corpus_size = await asyncio.to_thread(qdrant.count_points)
                surprise_pool_parsed = min(surprise_pool_parsed, corpus_size)
            except Exception as e:  # noqa: BLE001
                logger.debug("count_points failed; using requested pool as-is: %s", e)

        collections = parse_collections(request)

        fav_task: asyncio.Task[set[str]] | None = None
        if surprise and not prompt_state.positives and not active_centroids:
            # Surprise with no query: use zero vector so Qdrant
            # returns results from the whole collection.
            dim = cfg.centroid_expected_feature_dim
            vec = [0.0] * dim
        else:
            # Kick off vec resolution + (conditional) favorites fetch in
            # parallel — both touch different systems (SigLIP2 model vs
            # SQLite) so there's no contention. Saves ~30-100 ms on
            # queries where the user has favorites filtered.
            fav_task = (
                asyncio.create_task(favorite_ids_for_filter())
                if favorites else None
            )
            vec, vec_err, vec_detail = await asyncio.to_thread(
                resolve_query_vector,
                active_centroids, prompt_state, weights=active_weights,
                filename_pattern=filename_pattern,
                collections=collections,
            )
            if vec_err == "centroid_not_found":
                return bad_request(vec_detail or f"centroid {active_centroid!r} not loaded")
            if vec_err == "empty":
                return bad_request(vec_detail or "at least one positive prompt is required")

        t0 = time.time()
        # If the client didn't pass a seed, generate a fresh random
        # one for this request so each new search gives a new deck.
        # The seed flows into both the Gumbel RNG and the cache key
        # (which is the session id), so the same session reuses the
        # same deck but different sessions are different. Materialise
        # the seed here — before the cache key is computed — so the
        # type narrows to int and downstream callers see a real int.
        if surprise and surprise_seed_parsed is None:
            surprise_seed_parsed = secrets.randbelow(2**31)
        # At this point, if `surprise` is True, `surprise_seed_parsed`
        # is guaranteed non-None (either user-supplied or generated).
        # Materialise the narrowed-typed values so downstream callers
        # (build_surprise_deck) see `int` and `str` rather than
        # `int | None` / `str | None`.
        seed_for_cache_key: int = (
            surprise_seed_parsed if surprise_seed_parsed is not None else 0
        )
        # Pre-compute the Surprise Me cache key from the request
        # inputs. The cache key is what gates a cache hit and is
        # also the session id we echo back. Computing it once at
        # the top of the handler means the ETag is deterministic
        # from the inputs alone — no matter which branch actually
        # built the deck, the ETag is stable across the
        # `allowed_ids is None and not allowed_ids` short-circuit.
        #
        # When the client passes a session id, the session id IS
        # the cache key from the original request — we use it
        # directly so pagination of an existing session hits the
        # same cache entry even if the request is missing fields
        # the original had (e.g. an auto-random seed the client
        # doesn't know). Without this, every page request with a
        # session id would miss the cache and rebuild the deck
        # from scratch.
        surprise_cache_key = (
            surprise_session
            if surprise and surprise_session
            else (
                compute_cache_key(
                    q=q,
                    positives=list(prompt_state.positives),
                    negatives=list(prompt_state.negatives),
                    centroids=list(active_centroids),
                    weights=list(active_weights) if active_weights else None,
                    collections=collections,
                    favorites=favorites,
                    filename=filename_pattern,
                    temperature=surprise_temperature_parsed,
                    pool_size=surprise_pool_parsed,
                    seed=seed_for_cache_key,
                )
                if surprise else None
            )
        )
        # A filename pattern that matches nothing yields no hits from
        # Qdrant — the native filter simply returns an empty result
        # set, so there is no separate zero-match short-circuit here.
        # (The old id-list path needed one, because an empty
        # `HasIdCondition` is not the same as no filter at all.)
        deck_ids: list[str] = []  # populated by the surprise branch; empty otherwise
        try:
            if diverse:
                if fav_task is not None:
                    favorite_ids = await fav_task
                else:
                    # No favorites filter requested — pass None
                    # to diversity_page so it takes the request's
                    # search_allowed_ids verbatim (not an empty
                    # allowed_ids, which would short-circuit to
                    # zero hits).
                    favorite_ids = None
                hits, has_more, diversity_meta = diversity_page(
                    cfg, qdrant, diversity_cache,
                    vector=vec,
                    effective_limit=effective_limit,
                    offset=offset,
                    collections=collections,
                    allowed_ids=None,
                    favorite_ids=favorite_ids,
                    mode=diversity_mode,
                    strength=diversity_strength,
                    depth=diversity_depth_mode,
                    pool_depth=diversity_pool_depth,
                    path_filter=path_filter,
                )
            elif favorites:
                # The favorites branch owns surprise+favorites
                # composition too, because `favorites` is the
                # more specific filter (it restricts the Qdrant
                # pool to a whitelist). When `surprise` is also
                # set, run the Gumbel deck flow over the
                # favorites-restricted pool; otherwise the
                # plain page-through-the-pool flow.
                if fav_task is not None:
                    favorite_ids = await fav_task
                else:
                    favorite_ids = set()
                fav_allowed = list(favorite_ids)
                if surprise:
                    assert surprise_seed_parsed is not None, (  # noqa: S101
                        "surprise_seed_parsed must be set when surprise=True"
                    )
                    assert surprise_cache_key is not None, (  # noqa: S101
                        "surprise_cache_key must be set when surprise=True"
                    )
                    hits, has_more, deck_ids = await build_surprise_deck(
                        qdrant=qdrant,
                        vec=vec,
                        collections=collections,
                        surprise_allowed_ids=fav_allowed,
                        path_filter=path_filter,
                        surprise_pool=surprise_pool_parsed,
                        surprise_temperature=surprise_temperature_parsed,
                        surprise_seed=surprise_seed_parsed,
                        surprise_session=surprise_session,
                        surprise_cache_key=surprise_cache_key,
                        surprise_deck_cache=surprise_deck_cache,
                        effective_limit=effective_limit,
                        offset=offset,
                    )
                else:
                    hits, has_more = await asyncio.to_thread(
                        qdrant.search,
                        vec,
                        effective_limit,
                        offset,
                        collections or None,
                        fav_allowed,
                        None,
                        path_filter,
                    )
                diversity_meta = DiversityMetadata()
            elif surprise:
                # Pure surprise (no favorites filter).
                assert surprise_seed_parsed is not None, (  # noqa: S101
                    "surprise_seed_parsed must be set when surprise=True"
                )
                assert surprise_cache_key is not None, (  # noqa: S101
                    "surprise_cache_key must be set when surprise=True"
                )
                hits, has_more, deck_ids = await build_surprise_deck(
                    qdrant=qdrant,
                    vec=vec,
                    collections=collections,
                    surprise_allowed_ids=None,
                    path_filter=path_filter,
                    surprise_pool=surprise_pool_parsed,
                    surprise_temperature=surprise_temperature_parsed,
                    surprise_seed=surprise_seed_parsed,
                    surprise_session=surprise_session,
                    surprise_cache_key=surprise_cache_key,
                    surprise_deck_cache=surprise_deck_cache,
                    effective_limit=effective_limit,
                    offset=offset,
                )
                diversity_meta = DiversityMetadata()
            else:
                favorite_ids = set()
                # Stable pagination: rank once, slice pages from a
                # frozen snapshot. Passing `offset` straight to
                # Qdrant is unsound on HNSW — each page ranks at
                # depth offset+limit, so neighbouring pages overlap.
                hits, has_more = await materialize_search_page(
                    cfg,
                    qdrant,
                    snapshot_cache,
                    vector=vec,
                    offset=offset,
                    limit=effective_limit,
                    collections=collections,
                    allowed_ids=None,
                    favorite_ids=None,
                    path_filter=path_filter,
                )
                diversity_meta = DiversityMetadata()
        except (ConnectionError, OSError) as e:
            log_qdrant_error(logger, endpoint="/api/search", error=e, op="search")
            return qdrant_unreachable(str(e))
        except Exception as e:  # noqa: BLE001
            if "timeout" in type(e).__name__.lower() or "Timeout" in str(e):
                log_qdrant_timeout(
                    logger, endpoint="/api/search", op="search",
                    timeout_ms=cfg.query_timeout_ms,
                )
                return qdrant_timeout(str(e))
            log_qdrant_error(
                logger, endpoint="/api/search", error=e, op="search",
            )
            return internal_error(str(e))
        took_ms = int((time.time() - t0) * 1000)
        # Resolve fav/dis for the SearchResult list via the indexed
        # helper. The diversity path already pulled the favorite set;
        # this is the parallel lookup against the IndexDB cache.
        results: list[SearchResult] = await results_from_hits(
            index_db, cfg=cfg, hits=hits,
            favorite_ids=favorite_ids if favorites else None,
        )
        # Surprise Me: emit the session id (= cache_key) and deck
        # total so the client can walk the deck with offset+session.
        # Non-surprise requests leave both fields None (unchanged
        # from before this feature landed).
        surprise_session_out = (
            surprise_cache_key if (surprise and deck_ids) else None
        )
        surprise_total_out = (
            len(deck_ids) if (surprise and deck_ids) else None
        )
        body = SearchResponse(
            query=prompt_state.q,
            positives=prompt_state.positives,
            negatives=prompt_state.negatives,
            diverse=diverse,
            surprise=surprise,
            diversity=diversity_meta,
            view=view,
            centroid=active_centroid,
            centroids=list(active_centroids),
            weights=active_weights,
            results=results,
            took_ms=took_ms,
            offset=offset,
            limit=limit,
            has_more=has_more,
            session_id=surprise_session_out,
            session_total=surprise_total_out,
        )
        # ETag over every field that can change the response so
        # the browser's If-None-Match returns 304 only on truly
        # identical searches (Tier 2.4). Earlier the hash missed
        # filename/collections/diversity/weights — so a user
        # toggling a filter or switching the filename pattern
        # would get a 304 with the *wrong* response (the cached
        # one from the unfiltered query). Including every
        # variable that influences results_from_hits / diversity
        # rerank / filename filtering makes the cache honest.
        #
        # Canonical form: `\x1f` (ASCII unit separator) between
        # fields so a value containing `|` or `,` can't collide
        # with a different field's value. Without this, a prompt
        # of "a|b" would hash identically to two prompts "a"
        # and "b" — silent cache poisoning.
        etag_input = "\x1f".join(
            [
                q,
                ",".join(prompt_state.positives),
                ",".join(prompt_state.negatives),
                str(offset),
                str(limit),
                ",".join(active_centroids),
                ",".join(f"{w:.6f}" for w in (active_weights or [])),
                view,
                filename_pattern,
                ",".join(collections),
                diversity_mode,
                f"{diversity_strength:.6f}",
                diversity_depth_mode,
                str(diversity_pool_depth),
                # Surprise Me: include the cache_key so different
                # seeds/temperatures/pools collapse to different
                # ETags. Non-surprise requests have a fixed "off"
                # sentinel so the ETag is byte-for-byte identical
                # to before this feature landed.
                "off" if not surprise else (surprise_cache_key or "off"),
            ]
        ).encode("utf-8")
        etag = hashlib.sha256(etag_input).hexdigest()[:16]
        return JSONResponse(
            content=body.model_dump(),
            headers={
                "ETag": etag,
                "Cache-Control": "private, max-age=10",
            },
        )

    return router
