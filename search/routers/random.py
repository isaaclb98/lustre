"""
search/routers/random.py — /api/random (§B2 step 9).

GET /api/random?session=&offset=&limit=&collections=&view=:
    Walk through the library in a per-session shuffled order. The
    first call materializes a shuffled deck (random permutation of
    all point ids matching the collection filter) and returns the
    first `limit` photos. Subsequent calls pass the same `session`
    id and an `offset` to walk forward through the deck.

    Why session + offset (not "give me 20 random, dedupe client-side"):
    - Dedupe-on-client gets exponentially less productive as the
      on-screen set grows. The right shape is a server-side cursor
      that guarantees every photo is served exactly once per session.
    - One shuffle per session, O(N) where N = collection size.
      At 182 photos this is microseconds; at 2M it's a one-time
      ~5s query, then O(1) per request.

Two non-obvious things the route does:

1. **Session TTL** — sessions live 30 minutes by default
   (`random_session.RANDOM_SESSION_TTL_S`). After that the shuffled
   deck is discarded and the next call gets a fresh shuffle.
   Configurable for tests.

2. **No liveness filter** — the random route does NOT drop
   dead-NAS-file rows. The /random UX tolerates a broken tile for
   one cache refresh (60 s); filtering would require a second DB
   pass per request.

Tests pin:
- The endpoint returns the documented SearchResponse shape.
- Manual validation of `limit` and `offset` return 400 (not 422).
- First call materializes a session and returns `session_id`.
- Subsequent calls with the same session_id walk forward.
- Session expires after TTL.
- Two concurrent sessions don't see the same deck.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from search._result_helpers import bad_request as _bad_request
from search._result_helpers import coerce_view
from search.image_resolver import resolve_url
from search.models import ErrorResponse, SearchResponse, SearchResult
from search.random_session import RandomSessionStore

logger = logging.getLogger(__name__)


# Match the inline limit that was in `app.py`. Kept as a module
# constant so callers / tests / docs can reference one source of truth.
RANDOM_MAX_LIMIT = 200

# Default session lifetime is in search/random_session.py
# (RANDOM_SESSION_TTL_S). Imported from there.


def _random_rows_to_results(
    rows: list[dict[str, Any]], *, web_ui_url: str,
) -> list[SearchResult]:
    """Build SearchResult objects from SQLite random-sample rows.

    Mirrors `_favorite_rows_to_results` but reads is_favorite from
    the cache and uses the cache's width/height for future masonry
    support. No liveness filter — see module docstring.
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
                url=resolve_url(str(row["id"]), web_ui_url),
                is_favorite=is_fav,
                is_disliked=bool(int(row.get("is_disliked") or 0)),
                width=_maybe_int(row.get("width")),
                height=_maybe_int(row.get("height")),
                blurhash=_bh,
            )
        )
    return out



def build_random_router(
    index_db: Any,
    cfg: Any,
) -> APIRouter:
    """Build the random router with the live dependencies."""
    router = APIRouter()
    store = RandomSessionStore()

    @router.get("/api/random", response_model=SearchResponse)
    async def api_random(
        request: Request,  # kept for parity with /api/search
        limit: int = Query(cfg.top_k_default, description="max results"),
        offset: int = Query(0, description="position in the shuffled deck"),
        session: Annotated[
            str | None,
            Query(description="session id from a previous /api/random response"),
        ] = None,
        collections: Annotated[
            list[str] | None,
            Query(description="restrict to one or more collections; empty = whole set"),
        ] = None,
        view: str = Query(cfg.default_view, description="result view: 'grid' or 'feed'"),
    ) -> SearchResponse | JSONResponse:
        view = coerce_view(view)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return _bad_request("limit must be an integer")
        if not (1 <= limit <= RANDOM_MAX_LIMIT):
            return _bad_request(f"limit must be in [1, {RANDOM_MAX_LIMIT}]")
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            return _bad_request("offset must be an integer")
        if offset < 0:
            return _bad_request("offset must be >= 0")

        # Clean up the collection list (drop empties, dedupe while
        # preserving order so the response is stable for the client).
        seen: set[str] = set()
        clean_collections: list[str] = []
        for c in collections or []:
            c = (c or "").strip()
            if c and c not in seen:
                seen.add(c)
                clean_collections.append(c)

        try:
            # Return the existing live session, or make a new one. A
            # new session's deck is a CURSOR into the corpus's id
            # space, not a list: Qdrant's scroll takes a point-id
            # cursor and accepts any uuid, so the session can start
            # anywhere and read forward a page at a time. Nothing is
            # materialized up front, so the event loop is never
            # blocked and no full id scroll ever happens.
            existing = store.get(session)
            if existing is None:
                deck = await asyncio.to_thread(
                    index_db.new_cursor_deck, tuple(clean_collections)
                )
                session_id, sess = store.put_new(deck)
            else:
                session_id, sess = existing
        except Exception:
            logger.exception("random session materialize failed")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="internal_error",
                    detail="session materialize failed",
                    code="internal_error",
                ).model_dump(),
            )

        # The deck's length: how many ids the walk will produce. The
        # deck reads them from the collection as the session walks it.
        total = sess.deck.total
        # Clamp offset to the deck length. An offset past the end
        # means the caller has walked the whole session; return
        # empty results with has_more=False.
        effective_offset = min(max(offset, 0), total)
        # The deck reads the ids for this page on demand. The scroll
        # that backs it runs in a thread — it is a network call.
        batch_ids = await asyncio.to_thread(
            sess.deck.take, effective_offset, limit
        )

        # Materialize full SearchResults from the deck ids.
        rows: list[dict[str, Any]] = []
        if batch_ids:
            try:
                rows = index_db.rows_by_ids(batch_ids)
            except Exception:
                logger.exception("random deck lookup failed")
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(
                        error="internal_error",
                        detail="deck lookup failed",
                        code="internal_error",
                    ).model_dump(),
                )
            # The rows come back in arbitrary order from the lookup.
            # Re-order them to match the deck order — that's the
            # order the user is walking through.
            #
            # The deck holds Qdrant POINT ids; the row's `id` is the
            # app-level photo id (the payload's own `id` field). The
            # two are the same value in production, but the join
            # uses `point_id` so it holds either way.
            by_id = {str(r.get("point_id") or r["id"]): r for r in rows}
            rows = [by_id[i] for i in batch_ids if i in by_id]

        results = _random_rows_to_results(rows, web_ui_url=cfg.web_ui_url)
        has_more = (effective_offset + len(results)) < total

        return SearchResponse(
            query="",
            positives=[],
            negatives=[],
            view=view,
            centroid=None,
            centroids=[],
            weights=None,
            results=results,
            took_ms=0,
            offset=effective_offset,
            limit=limit,
            has_more=has_more,
            session_id=session_id,
            session_total=total,
        )

    # Expose the session store for tests via the router object.
    # `router._random_session_store` is an implementation detail.
    # setattr (vs. router._x = store) because APIRouter has no
    # such attribute and we don't want to monkey-patch the
    # starlette class. B010 is "setattr with constant"; this
    # is the legitimate form for adding attrs at runtime.
    setattr(router, "_random_session_store", store)  # noqa: B010

    return router