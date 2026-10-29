"""
search/routers/admin_orphans.py — admin orphan-review endpoints.

The orphan lifecycle is a two-step, and these endpoints are the
second step.

**Step one (automatic).** The indexer marks points whose photo is
gone from disk, and points belonging to a source that was removed
from the configured source list. The mark is the `orphan_since`
payload field. A marked point is excluded from every read path, so
it stops being served immediately — but its embedding is still
there.

**Step two (operator-confirmed).** These endpoints. An operator
looks at how many orphans there are and which photos they are, then
confirms the purge. Only then are the embeddings deleted, and the
matching user-state rows removed from SQLite.

The split exists because the two halves have very different
consequences. Marking is reversible — a file that comes back, or a
source that is re-added to the source list, clears the marker and
the photo returns. Deleting is not. So the irreversible half waits
for a human to look at it.

Both endpoints are admin-only by virtue of their path; the app
itself is expected to sit behind whatever access control the
deployment provides (the production deployment is reached over
Tailscale).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from search.index_db import IndexDB
from search.qdrant_client import QdrantSearch

logger = logging.getLogger(__name__)


class OrphanSample(BaseModel):
    """One marked orphan, as shown to the operator."""

    id: str
    path: str = ""
    collection: str = ""
    orphan_since: float | None = None


class OrphanListResponse(BaseModel):
    """Body for GET /api/admin/orphans."""

    count: int
    """Total number of marked orphans in the collection."""

    by_source: dict[str, int] = Field(default_factory=dict)
    """How many orphans each source holds."""

    sample: list[OrphanSample] = Field(default_factory=list)
    """A sample of the orphans, for review."""


class OrphanPurgeRequest(BaseModel):
    """Body for POST /api/admin/orphans/purge."""

    source: str | None = None
    """Purge only this source's orphans. None purges every orphan."""

    limit: int | None = Field(default=None, ge=1)
    """Cap on how many points to delete in this call. None = no cap."""

    confirm: bool = False
    """Must be True. The purge is irreversible, so the caller has to
    say so explicitly — there is no default-true path into it."""


class OrphanPurgeResponse(BaseModel):
    """Body for POST /api/admin/orphans/purge."""

    deleted: int
    """Number of Qdrant points deleted."""

    remaining: int
    """How many orphans are still marked after the purge."""

    user_state: dict[str, int] = Field(default_factory=dict)
    """Per-table counts of the SQLite rows that were removed."""

    source: str | None = None


def build_admin_orphans_router(
    *,
    qdrant: QdrantSearch,
    index_db: IndexDB | None,
) -> APIRouter:
    """Build the admin orphan router.

    `index_db` is optional so tests that don't need a real DB can
    pass None; production passes the live one so the purge can clean
    up the matching user-state rows.
    """
    router = APIRouter(prefix="/api/admin/orphans")

    @router.get("", response_model=OrphanListResponse)
    def list_orphans(
        sample_size: int = Query(
            50,
            ge=1,
            le=500,
            description="How many orphans to include in the sample.",
        ),
    ) -> OrphanListResponse:
        """How many orphans are marked, and which photos are they.

        The review step of the two-step purge. The operator looks at
        the count and the sample here, and only then confirms.
        """
        count = qdrant.count_orphans()
        by_source = qdrant.orphan_source_counts()
        sample = [
            OrphanSample(
                id=item["id"],
                path=item.get("path", ""),
                collection=item.get("collection", ""),
                orphan_since=item.get("orphan_since"),
            )
            for item in qdrant.list_orphans(limit=sample_size)
        ]
        return OrphanListResponse(
            count=count,
            by_source=by_source,
            sample=sample,
        )

    @router.post("/purge", response_model=OrphanPurgeResponse)
    def purge_orphans(req: OrphanPurgeRequest) -> OrphanPurgeResponse:
        """Delete the confirmed orphans from Qdrant and SQLite.

        This is the irreversible half. It requires `confirm: true` in
        the request body — there is no way to reach the delete without
        saying so explicitly.

        The Qdrant delete runs first, and it is scoped by the orphan
        marker: only points carrying `orphan_since` are in scope, so
        a purge can never reach a live photo. The SQLite cleanup runs
        after, for exactly the ids that were deleted.
        """
        if not req.confirm:
            raise HTTPException(
                status_code=400,
                detail=(
                    "the orphan purge is irreversible — set confirm=true to "
                    "delete the marked points"
                ),
            )

        try:
            result = qdrant.delete_orphans(
                source_name=req.source,
                limit=req.limit,
            )
        except Exception as exc:
            logger.exception("orphan purge failed")
            raise HTTPException(
                status_code=500,
                detail=f"orphan purge failed: {exc}",
            ) from exc

        deleted_ids: list[str] = result.get("ids", [])
        user_state: dict[str, int] = {}
        if deleted_ids and index_db is not None:
            try:
                user_state = index_db.delete_user_state_for_ids(deleted_ids)
            except Exception:
                # The Qdrant delete already happened. Failing to clean
                # up SQLite leaves rows pointing at deleted photos —
                # which the orphan purge will catch on its next run.
                # Reporting it is better than hiding it.
                logger.exception("orphan purge: SQLite cleanup failed")
                user_state = {"error": 1}

        remaining = qdrant.count_orphans()
        logger.info(
            "confirmed orphan purge: deleted=%d remaining=%d source=%s",
            result.get("deleted", 0), remaining, req.source,
        )
        return OrphanPurgeResponse(
            deleted=int(result.get("deleted", 0)),
            remaining=remaining,
            user_state=user_state,
            source=req.source,
        )

    return router
