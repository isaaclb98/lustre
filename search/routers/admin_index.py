"""
search/routers/admin_index.py — admin Index endpoints (§C2 in-app indexer).

Exposes the in-process IndexerRunner over HTTP so the UI can:
  - start a job (incremental / rebuild)
  - cancel a running job
  - poll for status / live progress
  - tail the subprocess log (last N lines, with resume cursor)

All four endpoints hit the same singleton `IndexerRunner` owned by
the search app. No locking needed at the router layer — the runner is
thread-safe internally.

The `rebuild` mode wipes the SQLite side store (favorites, dislikes,
albums, saved searches, feedback_events) BEFORE spawning the indexer,
so user data referencing soon-to-be-replaced point IDs is cleared
first. The Qdrant collection wipe happens inside the spawned
local_sync --rebuild subprocess.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from search.index_db import IndexDB
from search.indexer_runner import (
    IndexConflictError,
    IndexerRunner,
)
from search.models import (
    IndexerLogResponse,
    IndexerProgressModel,
    IndexerRunRequest,
    IndexerStatusResponse,
)

logger = logging.getLogger(__name__)


def _status_to_response(status: Any, *, points_count: int | None = None) -> IndexerStatusResponse:
    """Translate the runner's internal `IndexerStatus` to the wire model.

    `points_count` defaults to the runner's own snapshot, but
    callers can override it with a live count from the side store
    (IndexDB.count_images) so the UI sees the real count even when
    the indexer hasn't completed a run yet (e.g. a fresh dev
    container that mounted pre-existing data).
    """
    return IndexerStatusResponse(
        state=status.state.value,
        mode=status.mode,
        job_id=status.job_id,
        pid=status.pid,
        started_at=status.started_at,
        finished_at=status.finished_at,
        last_run_at=status.last_run_at,
        last_error=status.last_error,
        progress=IndexerProgressModel(
            indexed=status.progress.indexed,
            reembedded=status.progress.reembedded,
            skipped=status.progress.skipped,
            errors=status.progress.errors,
            # Round-50: pass through thumb_regens. Default 0 on
            # the wire model makes the field optional from the
            # UI's perspective (the type is non-optional but the
            # default makes the field backward-compatible with
            # older runner responses).
            thumb_regens=status.progress.thumb_regens,
        ),
        points_count=(
            points_count if points_count is not None else status.points_count
        ),
        phase=status.phase,
    )


def build_admin_index_router(
    *,
    indexer_runner: IndexerRunner,
    index_db: IndexDB | None,
) -> APIRouter:
    """Build the admin Index router with the live runner + IndexDB.

    `index_db` is optional so tests that don't need a real DB can
    pass `None`; production passes the live one so the rebuild wipe
    actually clears the side store.
    """
    router = APIRouter(prefix="/api/admin/index")

    def _live_points_count() -> int | None:
        """Return the current count of indexed images.

        Reads through the IndexDB, which caches the corpus size for
        a few seconds. It used to be a local SQLite read of the
        `images` mirror; since that mirror was dropped it is a live
        Qdrant count, and an exact one walks every point. The
        settings page polls the index status every second, so the
        cache is what keeps a poll from counting 2.2M points once a
        second.

        Returns None on any error (DB not initialised, lock
        contention, etc.) so the runner's own snapshot is used
        instead — never lie about counts.
        """
        if index_db is None:
            return None
        try:
            return int(index_db.count_images())
        except Exception as err:  # noqa: BLE001
            logger.debug("count_images failed; falling back to snapshot: %s", err)
            return None

    @router.post("", response_model=IndexerStatusResponse, status_code=202)
    def start_index(req: IndexerRunRequest) -> IndexerStatusResponse:
        """Start a new indexer job. Returns 202 with the initial status.

        409 if a job is already running. The runner rejects
        concurrent starts — the UI must wait for IDLE before
        triggering another.
        """
        # Rebuild wipe: clear the SQLite side store BEFORE spawning
        # the indexer. The Qdrant collection is wiped inside the
        # spawned local_sync --rebuild subprocess (so the wipe
        # happens in the same process that re-creates it).
        if req.mode == "rebuild" and index_db is not None:
            index_db.reset_side_store()

        try:
            indexer_runner.start(req.mode)
        except IndexConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"failed to spawn indexer: {exc}",
            ) from exc

        return _status_to_response(indexer_runner.status(), points_count=_live_points_count())

    @router.post("/cancel", response_model=IndexerStatusResponse)
    def cancel_index() -> IndexerStatusResponse:
        """Send SIGTERM to the running job.

        200 with current status (now IDLE or FAILED). 400 if no
        job was running.
        """
        if not indexer_runner.cancel():
            raise HTTPException(
                status_code=400,
                detail="no indexer job is currently running",
            )
        # Wait briefly for the subprocess to acknowledge and exit.
        # This is mostly so the response carries the post-cancel state.
        indexer_runner.wait_idle(timeout=5.0)
        return _status_to_response(indexer_runner.status(), points_count=_live_points_count())

    @router.get("/status", response_model=IndexerStatusResponse)
    def get_status() -> IndexerStatusResponse:
        """Snapshot of the current job. The UI polls this every 1s
        while the job is RUNNING."""
        return _status_to_response(indexer_runner.status(), points_count=_live_points_count())

    @router.get("/log", response_model=IndexerLogResponse)
    def get_log(
        since_line: int = Query(
            0,
            ge=0,
            description="Return only lines after this 0-based line index.",
        ),
    ) -> IndexerLogResponse:
        """Tail the indexer's stdout/stderr. `next_line` is the cursor
        for the next poll; pass it back as `since_line` to get
        only newer lines."""
        snap = indexer_runner.log(since_line=since_line)
        return IndexerLogResponse(
            lines=snap["lines"],
            next_line=snap["next_line"],
            total=snap["total"],
        )

    return router
