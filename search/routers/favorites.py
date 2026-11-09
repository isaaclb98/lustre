"""
search/routers/favorites.py — /api/favorites (§B2 step 3).

Favorites CRUD:

- POST /api/favorites/{point_id}: mark a photo as favourited
- DELETE /api/favorites/{point_id}: unmark
- GET /api/favorites?limit=&offset=&as_results=:
    list favourites. `as_results=true` returns a SearchResponse-compatible
    shape (use this when the frontend's image grid is the consumer).

The favourites routing is the canonical example of where invalidation
hooks live: any mark/unmark mutates the user's preference signal,
which forces the `favourites dynamic centroid` to
to drop their caches on the next request. The router takes both
invalidators as factory parameters; the app passes the live ones.

Tests pin:

- The three endpoints' status codes and JSON response shapes.
- Manual validation of limit/offset returns 400 (not 422).
- Invalidations are called on every mark/unmark.
- `as_results=true` produces a SearchResponse-compatible shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from search._result_helpers import bad_request as _bad_request_json
from search.image_resolver import resolve_url
from search.models import (
    FavoriteItem,
    FavoritesListResponse,
    FavoriteToggleResponse,
    SearchResponse,
    SearchResult,
)


def _favorite_rows_to_results(
    rows: list[dict[str, Any]], *, web_ui_url: str,
) -> list[SearchResult]:
    """Convert a list of favourite DB rows to SearchResult entries.

    Score is 0 because favourites aren't a vector search result — the
    UI just needs the same shape as a search hit to render in the
    grid component.
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
            is_favorite=True,
            width=_maybe_int(row.get("width")),
            height=_maybe_int(row.get("height")),
        )
        for row in rows
    ]


def build_favorites_router(
    *,
    index_db: Any,
    cfg: Any,
    invalidate_likes_centroid: Callable[[], None],
) -> APIRouter:
    """Build the favourites router with the live dependencies."""
    router = APIRouter()

    @router.post(
        "/api/favorites/{point_id}",
        response_model=FavoriteToggleResponse,
    )
    async def mark_favorite(point_id: str) -> FavoriteToggleResponse:
        try:
            await asyncio.to_thread(index_db.mark_favorite, point_id)
        except Exception as err:  # ImageNotInCacheError or similar
            raise HTTPException(
                status_code=404, detail="Photo not found in index cache",
            ) from err
        # Invalidate the favourites dynamic centroid so the next
        # search through it reflects the new favourite.
        invalidate_likes_centroid()
        row = await asyncio.to_thread(index_db.get_by_id, point_id)
        return FavoriteToggleResponse(
            id=point_id,
            favorited_at=str((row or {}).get("favorited_at") or ""),
        )

    @router.delete("/api/favorites/{point_id}", status_code=204)
    async def unmark_favorite(point_id: str) -> None:
        row = await asyncio.to_thread(index_db.get_by_id, point_id)
        if row is None or int(row.get("is_favorite") or 0) != 1:
            raise HTTPException(status_code=404, detail="Favourite not found")
        await asyncio.to_thread(index_db.unmark_favorite, point_id)
        # Same invalidation as mark_favorite — every unmark moves the
        # centroid, and we don't try to detect whether it moved enough
        # to matter. Cheap, simple, correct.
        invalidate_likes_centroid()

    @router.get("/api/favorites", response_model=None)
    async def api_favorites(
        limit: int = Query(
            cfg.top_k_default, description="max favourites",
        ),
        offset: int = Query(
            0, description="offset into favourites",
        ),
        as_results: bool = Query(
            False, description="return SearchResponse-compatible shape",
        ),
    ) -> SearchResponse | FavoritesListResponse | JSONResponse:
        # Manual validation so we return 400 (not 422) for bad input.
        # No upper cap on limit — Likes albums can grow arbitrarily
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
        rows = await asyncio.to_thread(index_db.list_favorites, limit, offset)
        total = await asyncio.to_thread(index_db.count_favorites)
        if as_results:
            return SearchResponse(
                query="",
                positives=[],
                negatives=[],
                view=cfg.default_view,
                centroid=None,
                results=_favorite_rows_to_results(
                    rows, web_ui_url=cfg.web_ui_url,
                ),
                took_ms=0,
                offset=offset,
                limit=limit,
                has_more=offset + len(rows) < total,
            )
        return FavoritesListResponse(
            favorites=[
                FavoriteItem(
                    id=str(row["id"]),
                    path=str(row["path"]),
                    favorited_at=str(row["favorited_at"] or ""),
                )
                for row in rows
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    return router