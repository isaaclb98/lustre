"""
search/routers/dislikes.py — /api/dislikes (§B2 step 4).

Dislike CRUD:

- POST /api/dislikes/{point_id}: mark a photo as disliked
- DELETE /api/dislikes/{point_id}: unmark
- GET /api/dislikes?limit=&offset=&as_results=:
    list dislikes. `as_results=true` returns a SearchResponse-compatible
    shape (same wire shape as `/api/favorites?as_results=true`) so the
    frontend grid renders dislikes with no special-case code.

The dislikes routing mirrors the favourites shape (persistent feedback
signal). Both endpoints mutate the user's preference signal, which
forces the `favourites dynamic centroid` to drop
their caches on the next request. The router takes both invalidators
as factory parameters; the app passes the live ones.

The `source` parameter on POST records the page the feedback event
came from (set by the client via a query param). It's a future-proofing
hook for Phase 2 analytics attribution — the value is stored as-is and
the schema allows any non-empty string.

Tests pin:

- The three endpoints' status codes and JSON response shapes.
- Manual validation of limit/offset returns 400 (not 422).
- Invalidations are called on every mark/unmark.
- `as_results=true` produces a SearchResponse-compatible shape with
  `is_favorite` reflecting the live favourites set (so the grid's
  heart icon stays correct).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from search._result_helpers import bad_request as _bad_request_json
from search.image_resolver import resolve_url
from search.models import SearchResponse, SearchResult


def _dislike_rows_to_results(
    rows: list[dict[str, Any]], *, web_ui_url: str, favorite_ids: set[str],
) -> list[SearchResult]:
    """Convert dislike DB rows to SearchResult entries.

    `is_favorite` comes from the live favourites set so the grid's
    heart icon stays correct even though the row came from the
    dislikes table.
    """

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
            url=resolve_url(str(row["id"]), web_ui_url),
            is_favorite=str(row["id"]) in favorite_ids,
            width=_maybe_int(row.get("width")),
            height=_maybe_int(row.get("height")),
        )
        for row in rows
    ]


def build_dislikes_router(
    *,
    index_db: Any,
    cfg: Any,
    invalidate_likes_centroid: Callable[[], None],
) -> APIRouter:
    """Build the dislikes router with the live dependencies."""
    router = APIRouter()

    @router.post("/api/dislikes/{point_id}", status_code=204)
    async def mark_dislike(point_id: str, source: str = "manual") -> None:
        await asyncio.to_thread(index_db.mark_dislike, point_id, source)
        await asyncio.to_thread(
            index_db.record_feedback, point_id, "dislike", source,
        )
        # Same invalidation shape as mark_favorite — every dislike moves
        # the user preference vector (just in the negative direction).
        invalidate_likes_centroid()

    @router.delete("/api/dislikes/{point_id}", status_code=204)
    async def unmark_dislike(point_id: str) -> None:
        await asyncio.to_thread(index_db.unmark_dislike, point_id)
        invalidate_likes_centroid()

    @router.get("/api/dislikes", response_model=None)
    async def list_dislikes(
        limit: int = Query(cfg.top_k_default, description="max dislikes"),
        offset: int = Query(0, description="offset into dislikes"),
        as_results: bool = Query(
            False, description="return SearchResponse-compatible shape",
        ),
    ) -> SearchResponse | dict[str, Any] | JSONResponse:
        # Manual validation so we return 400 (not 422) for bad input.
        # No upper cap on limit — Dislikes albums can grow arbitrarily
        # large and the caller (frontend) walks them via
        # ?limit=&offset= pagination, not "ask for everything once".
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return _bad_request_json("limit must be an integer")
        if limit < 1:
            return _bad_request_json("limit must be >= 1")
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            return _bad_request_json("offset must be an integer")
        if offset < 0:
            return _bad_request_json("offset must be >= 0")
        rows = await asyncio.to_thread(index_db.list_dislikes, limit, offset)
        total = await asyncio.to_thread(index_db.count_dislikes)
        if as_results:
            # Fresh lookup so `is_favorite` reflects the live favourites
            # set, not a stale cached state.
            fav_ids = await asyncio.to_thread(index_db.list_favorite_ids)
            return SearchResponse(
                query="",
                positives=[],
                negatives=[],
                view=cfg.default_view,
                centroid=None,
                results=_dislike_rows_to_results(
                    rows, web_ui_url=cfg.web_ui_url,
                    favorite_ids=set(fav_ids),
                ),
                took_ms=0,
                offset=offset,
                limit=limit,
                has_more=offset + len(rows) < total,
            )
        return {
            "items": rows,
            "count": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(rows) < total,
        }

    return router
