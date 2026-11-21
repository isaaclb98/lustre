"""
search/routers/thumbnails.py — thumbnail serving endpoint.

Serves pre-generated WebP thumbnails from THUMBNAIL_DIR.
Falls back to 404 if thumbnail doesn't exist (frontend uses blurhash).

Path pattern: {THUMBNAIL_DIR}/{prefix}/{point_id}.webp
  - prefix = first 2 chars of point_id (256 buckets)
  - ~8KB per image at 256px WebP q50

Round-perf (issue #2): supports an optional ?w= query param so the
frontend can ask for a sized variant for its srcset. Returns 404 if
the sized variant doesn't exist on disk so the browser falls back to
the canonical 256px file (which is always present).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from search.config import THUMBNAIL_DIR, THUMBNAIL_SIZE

logger = logging.getLogger(__name__)


def build_thumbnails_router() -> APIRouter:
    """Build the thumbnail serving router."""
    router = APIRouter()

    @router.get("/thumb/{point_id}")
    async def get_thumbnail(
        point_id: str,
        # Round-perf (issue #2): the frontend's srcset asks for a target width
        # so the browser can pick the smallest variant that fits its
        # current rendered size. We don't generate new variants on
        # demand (that'd be a slow first-hit and burn disk) — instead
        # we look for a pre-generated sibling and 404 if it doesn't
        # exist so the browser falls back to the canonical 256px file.
        # Pre-migration cap was 256. The indexer now produces a
        # single 384px variant (matches the so400m model input
        # resolution), so the cap moves up to 384 to match.
        #
        # The cap is the deployment's THUMBNAIL_SIZE: the indexer
        # renders the canonical thumbnail at that long edge, so a
        # request beyond it asks for an asset that does not exist.
        # The dev stack renders 512 (the C-RADIOv4 model's input
        # resolution), production 384 — a hardcoded 384 rejected the
        # dev stack's own request with a 422.
        w: int | None = Query(None, ge=64, le=THUMBNAIL_SIZE),
    ) -> FileResponse:
        """
        Serve a pre-generated WebP thumbnail, optionally sized.

        Args:
            point_id: Qdrant point ID (32-char hex)
            w: Optional target width in pixels. Looks for
                `{THUMBNAIL_DIR}/{prefix}/{point_id}.w{w}.webp`.
                Returns 404 if missing; the browser then loads the
                canonical 256px file.

        Returns:
            WebP file with immutable cache headers

        Raises:
            404 if thumbnail doesn't exist (frontend falls back to blurhash)
        """
        # Validate point_id format (32-char hex or UUID with hyphens)
        clean_id = point_id.replace("-", "")
        if len(clean_id) != 32 or not all(c in "0123456789abcdef" for c in clean_id):
            raise HTTPException(status_code=400, detail="Invalid point_id format")

        # Compute thumbnail path using two-level prefix
        prefix = point_id[:2]
        thumb_path: Path
        if w is not None:
            sized = Path(THUMBNAIL_DIR) / prefix / f"{point_id}.w{w}.webp"
            thumb_path = (
                sized if sized.exists()
                else Path(THUMBNAIL_DIR) / prefix / f"{point_id}.webp"
            )
        else:
            thumb_path = Path(THUMBNAIL_DIR) / prefix / f"{point_id}.webp"

        if not thumb_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")

        # Immutable cache: thumbnails are deterministic by point_id
        # and never change unless the image is reindexed (which changes the point_id)
        return FileResponse(
            path=thumb_path,
            media_type="image/webp",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "Access-Control-Allow-Origin": "*",
            },
        )

    return router
