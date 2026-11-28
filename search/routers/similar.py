"""
search/routers/similar.py — /api/similar/{id} (§B2 step 10).

GET /api/similar/{point_id}?limit=&offset=:
    Most-similar photos: nearest neighbours of `point_id` in the
    embedding space. Backs the Lightbox "Most similar" button.

Two round-trips to Qdrant: retrieve the source point's vector,
then `search` with that vector as the query. The source vector
already lives in Qdrant so we don't re-encode (which would burn
a SigLIP2 forward pass per click). The source photo is excluded
from the results so the user doesn't see the photo they just
clicked as the top hit.

Why the `limit + 1` over-fetch + trim pattern: the server-side
`exclude_ids` filter is best-effort — Qdrant drops the source if
it falls in the filter set but keeps it if some other filter
(collection, payload) silently removes the source from the index.
The extra row gives us headroom; the trim on the client side
guarantees we never return more than `limit` rows.

Tests pin:
- The endpoint returns the documented SearchResponse shape.
- Qdrant errors surface as 502.
- A non-existent source photo returns 404.
- The source photo is excluded from results (Layer 1 — server-side
  `exclude_ids`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from lustre_kernel.spaces import PICTURE_SPACE
from search.image_resolver import resolve_url
from search.models import DiversityMetadata, SearchResponse, SearchResult
from search.qdrant_client import SearchHit

logger = logging.getLogger(__name__)


def _results_from_hits(
    hits: list[SearchHit],
    *,
    web_ui_url: str,
    favorite_ids: set[str] | None = None,
    dislike_ids: set[str] | None = None,
) -> list[SearchResult]:
    """Build SearchResult objects from raw Qdrant hits.

    `favorite_ids` and `dislike_ids` are pre-resolved when the caller
    already has them (avoids a second DB round-trip on the hot
    /api/search path); when omitted, the caller is responsible for
    resolving them — the similar router passes the live lookup.
    """
    return [
        SearchResult(
            id=h.id,
            path=h.path,
            score=h.score,
            score_str=f"{h.score:.3f}",
            url=resolve_url(h.id, web_ui_url),
            is_favorite=h.id in (favorite_ids or set()),
            is_disliked=h.id in (dislike_ids or set()),
            # LQIP from the Qdrant payload (set at index time).
            blurhash=(h.payload or {}).get("blurhash"),
            # Dimensions for the photo-card caption row.
            width=(h.payload or {}).get("width"),
            height=(h.payload or {}).get("height"),
        )
        for h in hits
    ]


def build_similar_router(
    *,
    qdrant: Any,
    cfg: Any,
    index_db: Any,
) -> APIRouter:
    """Build the similar router with the live dependencies."""
    router = APIRouter()

    @router.get("/api/similar/{point_id}", response_model=SearchResponse)
    async def similar_photos(
        point_id: str,
        limit: int = Query(
            cfg.top_k_default,
            description="max similar photos to return",
            ge=1,
            le=200,
        ),
        offset: int = Query(
            0, description="offset into the result set", ge=0,
        ),
    ) -> SearchResponse:
        """Most-similar photos: nearest neighbours of `point_id`."""
        started = time.monotonic()
        # The picture→picture space. DINOv3's summary, not SigLIP2's
        # global summary: DINOv3 is self-supervised on visual
        # structure, so its space ranks photographs by how they LOOK
        # rather than by what they depict. Both the retrieve and the
        # search must use the same space — a query vector is only
        # meaningful in the space it was drawn from.
        try:
            pair = await asyncio.to_thread(
                qdrant.retrieve_with_vector, point_id, PICTURE_SPACE,
            )
        except (ConnectionError, OSError) as e:
            logger.warning(
                "Qdrant unreachable for /api/similar/%s: %s", point_id, e,
            )
            raise HTTPException(status_code=502, detail="Qdrant unreachable") from e
        if pair is None:
            raise HTTPException(status_code=404, detail="Photo not found")
        vec, _hit = pair
        try:
            hits, has_more = await asyncio.to_thread(
                qdrant.search,
                vec,
                limit + 1,  # +1 because we exclude the source below
                offset,
                None,  # no collection whitelist
                None,  # no allowed_ids
                [point_id],  # exclude the source photo itself
                None,  # no path filter
                True,  # with_payload
                None,  # search_params
                PICTURE_SPACE,  # search the picture→picture space
            )
        except (ConnectionError, OSError) as e:
            logger.warning(
                "Qdrant search failed for /api/similar/%s: %s", point_id, e,
            )
            raise HTTPException(status_code=502, detail="Qdrant unreachable") from e
        # Trim the extra +1 we fetched for exclusion safety.
        hits = hits[:limit]

        # Resolve fav/dis sets once, in parallel, for the result list.
        ids = [h.id for h in hits]
        fav_set, dis_set = await asyncio.gather(
            asyncio.to_thread(index_db.favorite_id_set, ids),
            asyncio.to_thread(index_db.dislike_id_set, ids),
        )
        results = _results_from_hits(
            hits, web_ui_url=cfg.web_ui_url,
            favorite_ids=fav_set, dislike_ids=dis_set,
        )
        return SearchResponse(
            query="",
            positives=[],
            negatives=[],
            diverse=False,
            diversity=DiversityMetadata(),
            surprise=False,
            view="grid",
            centroid=None,
            centroids=[],
            weights=None,
            results=results,
            took_ms=int((time.monotonic() - started) * 1000),
            offset=offset,
            limit=limit,
            has_more=has_more,
        )

    return router
