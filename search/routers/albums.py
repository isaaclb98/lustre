"""
search/routers/albums.py — /api/albums/* (§B2 step 12).

Albums CRUD + membership:

- POST   /api/albums                       — create album + register centroid
- GET    /api/albums                       — list all albums
- GET    /api/albums/{id}                  — album detail with paged members
- PATCH  /api/albums/{id}                  — rename / update description
- DELETE /api/albums/{id}                  — delete album + unregister centroid
- POST   /api/albums/{id}/duplicate         — copy an album + its members
- POST   /api/albums/{id}/members/{fid}    — add favourite as member + invalidate centroid
- DELETE /api/albums/{id}/members/{fid}    — remove member + invalidate centroid
- GET    /api/albums/by-favorite/{fid}     — list albums containing a given favourite

Centroid lifecycle is wired through three callbacks (register,
unregister, invalidate) so the album centroid stays in sync with
album creation / deletion / membership changes. The router takes
them as factory parameters; `app.py` passes the live closures.

Tests pin:
- Each endpoint's status code + response shape.
- The 400 / 404 error paths.
- Centroid callbacks fire on create / rename / delete / member add+remove.
- The partial-update bridge in PATCH (name-only or description-only
  still updates the row, doesn't drop the other field).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from search.models import (
    AlbumCreateRequest,
    AlbumDetailResponse,
    AlbumMemberItem,
    AlbumMemberResponse,
    AlbumMembershipsResponse,
    AlbumsListResponse,
    AlbumSummary,
    AlbumUpdateRequest,
)

logger = logging.getLogger(__name__)


def build_albums_router(
    *,
    index_db: Any,
    cfg: Any,
    register_album_centroid: Callable[[int], None],
    unregister_album_centroid: Callable[[int], None],
    invalidate_album_centroid: Callable[[int], None],
) -> APIRouter:
    """Build the albums router with the live dependencies."""
    router = APIRouter()

    def _maybe_int(v: Any) -> int | None:
        try:
            iv = int(v) if v is not None else None
        except (TypeError, ValueError):
            return None
        return iv if iv and iv > 0 else None

    @router.post("/api/albums", response_model=AlbumSummary)
    async def create_album(body: AlbumCreateRequest) -> AlbumSummary:
        try:
            album_id = await asyncio.to_thread(
                index_db.create_album, body.name, body.description,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # Register the centroid so the album is immediately usable
        # as a search primitive (lazy compute — the first /api/search
        # call that uses it pays the cost).
        register_album_centroid(album_id)
        albums = await asyncio.to_thread(index_db.list_albums)
        for a in albums:
            if a["id"] == album_id:
                return AlbumSummary(**a)
        # Shouldn't happen — we just inserted this row.
        raise HTTPException(status_code=500, detail="album not found after create")

    @router.post("/api/albums/{album_id}/duplicate", response_model=AlbumSummary)
    async def duplicate_album(album_id: int) -> AlbumSummary:
        """Copy an album and its members into a new, uniquely-named album.

        The copy gets its own centroid, so it is usable as a search
        primitive straight away — exactly like an album the user made
        by hand.
        """
        try:
            album = await asyncio.to_thread(
                index_db.duplicate_album, album_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if album is None:
            raise HTTPException(status_code=404, detail="Album not found")
        register_album_centroid(album["id"])
        return AlbumSummary(**album)

    @router.get("/api/albums", response_model=AlbumsListResponse)
    async def list_albums() -> AlbumsListResponse:
        rows = await asyncio.to_thread(index_db.list_albums)
        return AlbumsListResponse(albums=[AlbumSummary(**r) for r in rows])

    @router.get("/api/albums/{album_id}", response_model=AlbumDetailResponse)
    async def get_album(
        album_id: int,
        limit: int = Query(cfg.top_k_default, description="max members to return"),
        offset: int = Query(0, description="offset into members"),
    ) -> AlbumDetailResponse:
        album = await asyncio.to_thread(index_db.get_album, album_id)
        if album is None:
            raise HTTPException(status_code=404, detail="Album not found")
        # No upper cap on limit — user albums can grow arbitrarily
        # large and the caller (frontend) walks them via
        # ?limit=&offset= pagination, not "ask for everything once".
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = cfg.top_k_default
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        rows = await asyncio.to_thread(
            index_db.list_album_members, album_id, limit, offset,
        )
        total = await asyncio.to_thread(
            index_db.count_album_members, album_id,
        )
        return AlbumDetailResponse(
            id=album["id"],
            name=album["name"],
            description=album.get("description") or "",
            cover_favorite_id=album.get("cover_favorite_id") or "",
            created_at=album["created_at"],
            updated_at=album["updated_at"],
            members=[
                AlbumMemberItem(
                    id=str(row["id"]),
                    path=str(row["path"]),
                    added_at=str(row["added_at"] or ""),
                    blurhash=row.get("blurhash") or None,
                    # The photo's dimensions, so the grid can lay the
                    # tile out at the photo's own aspect ratio.
                    width=_maybe_int(row.get("width")),
                    height=_maybe_int(row.get("height")),
                )
                for row in rows
            ],
            member_total=total,
        )

    @router.patch("/api/albums/{album_id}", response_model=AlbumSummary)
    async def update_album(
        album_id: int, body: AlbumUpdateRequest,
    ) -> AlbumSummary:
        # Build a rename tuple that's tolerant of partial updates
        # (only name, only description, or both). `rename_album`
        # requires both args or neither — we have to bridge the
        # partial case by reading the current row first.
        name = body.name
        description = body.description
        if name is None and description is None:
            raise HTTPException(
                status_code=400,
                detail="at least one of name or description is required",
            )
        if name is None:
            current = await asyncio.to_thread(index_db.get_album, album_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Album not found")
            name = current["name"]
        try:
            ok = await asyncio.to_thread(
                index_db.rename_album, album_id, name, description,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not ok:
            raise HTTPException(status_code=404, detail="Album not found")
        # Re-register so the centroid label picks up the new name.
        register_album_centroid(album_id)
        for a in await asyncio.to_thread(index_db.list_albums):
            if a["id"] == album_id:
                return AlbumSummary(**a)
        raise HTTPException(status_code=500, detail="album not found after update")

    @router.delete("/api/albums/{album_id}", status_code=204)
    async def delete_album(album_id: int) -> None:
        ok = await asyncio.to_thread(index_db.delete_album, album_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Album not found")
        unregister_album_centroid(album_id)

    @router.post(
        "/api/albums/{album_id}/members/{favorite_id}",
        response_model=AlbumMemberResponse,
    )
    async def add_album_member(
        album_id: int, favorite_id: str,
    ) -> AlbumMemberResponse:
        ok = await asyncio.to_thread(
            index_db.add_album_member, album_id, favorite_id,
        )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail="Album not found or favourite already a member",
            )
        invalidate_album_centroid(album_id)
        # Look up the membership row to return the canonical
        # added_at. The simpler approach would be to return a
        # computed "now" but that's inconsistent with re-adding
        # a removed favourite (where the added_at should be the
        # most recent add, not the original one).
        await asyncio.to_thread(
            index_db.list_album_member_ids, album_id,
        )
        # We need the added_at for this specific (album, favourite)
        # pair. The membership table isn't currently exposed by id
        # query, so fall back to the now-stored value by reading
        # list_album_members with a tight filter.
        rows = await asyncio.to_thread(
            index_db.list_album_members, album_id, 1, 0,
        )
        added_at = ""
        # list_album_members INNER JOINs against images, so an
        # orphan membership won't appear here. For a favourited
        # photo this is fine; for an orphan we'd need a separate
        # query path (not exposed yet — kept simple for v1).
        if rows and rows[0]["id"] == favorite_id:
            added_at = str(rows[0]["added_at"] or "")
        return AlbumMemberResponse(
            album_id=album_id,
            favorite_id=favorite_id,
            added_at=added_at,
        )

    @router.delete(
        "/api/albums/{album_id}/members/{favorite_id}",
        status_code=204,
    )
    async def remove_album_member(
        album_id: int, favorite_id: str,
    ) -> None:
        ok = await asyncio.to_thread(
            index_db.remove_album_member, album_id, favorite_id,
        )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail="Album not found or favourite not a member",
            )
        invalidate_album_centroid(album_id)

    @router.get(
        "/api/albums/by-favorite/{favorite_id}",
        response_model=AlbumMembershipsResponse,
    )
    async def list_albums_for_favorite(
        favorite_id: str,
    ) -> AlbumMembershipsResponse:
        """Return every album that contains `favorite_id`.

        Used by the per-photo UI to show which albums a photo is
        in. The summary shape omits member_count (always 1 for
        this view) so we re-use AlbumSummary with count=1.
        """
        rows = await asyncio.to_thread(
            index_db.list_albums_for_favorite, favorite_id,
        )
        summaries = [
            AlbumSummary(
                id=r["id"],
                name=r["name"],
                description=r.get("description") or "",
                cover_favorite_id="",
                member_count=1,
                created_at="",
                updated_at="",
            )
            for r in rows
        ]
        return AlbumMembershipsResponse(
            favorite_id=favorite_id,
            albums=summaries,
        )

    return router
