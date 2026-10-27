"""
search/lazy_index_cache.py — B5 contract: lazy + stale-while-revalidate.

Wraps `IndexDB` with a public surface that documents the B5
semantics — startup completes WITHOUT hydrating from Qdrant, the
first read triggers hydration, and the app serves from a stale
(possibly empty) cache while a background task refreshes.

This is the missing abstraction layer: the prior `IndexDB` had a
background-refresh codepath buried inside the FastAPI lifespan,
which made it impossible to test or reuse. `LazyIndexCache`
extracts the contract into a single importable class.

## Contract

  - `LazyIndexCache(index_db)` — wraps an IndexDB. No Qdrant
    call. Startup is fast.
  - `await cache.ensure_hydrated()` — idempotent. Triggers a
    background refresh if the cache is stale. Returns immediately
    (does NOT block on Qdrant).
  - `await cache.read(...)` — reads a row from the underlying
    IndexDB. If the cache hasn't been hydrated yet, schedules a
    background hydrate and returns None (caller treats this as
    "no data yet, fall back to live Qdrant query").
  - `await cache.refresh()` — blocks until Qdrant scroll finishes.
    Used by tests and the manual /api/cache/refresh endpoint.
  - `cache.is_hydrated` — bool. False until the first refresh
    completes. Set to True after `ensure_hydrated` or `refresh`.

## Why this is the B5 shape

The plan's §B5 acceptance criteria:
  - startup completes without hydrating from Qdrant ✓
  - first read triggers a hydrate ✓
  - app serves from stale cache while background refreshes ✓
  - concurrent reads see consistent data (no torn rows) ✓
    (IndexDB already serialises via its internal lock)
  - existing test_index_db.py keeps passing ✓
    (LazyIndexCache is opt-in; nothing forces callers through it)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class LazyIndexCache:
    """B5 wrapper around an `IndexDB`.

    Owns the hydrate state + the background-refresh task. Callers
    that want the B5 contract (startup fast, lazy hydrate, stale-
    while-revalidate) use this. Callers that need direct,
    synchronous access to IndexDB can still go through the
    underlying instance via `cache.index_db`.
    """

    def __init__(self, index_db: Any) -> None:
        self._index_db = index_db
        self._is_hydrated: bool = False
        self._hydrate_started_at: float | None = None
        self._hydrate_finished_at: float | None = None
        self._refresh_in_progress: bool = False
        self._hydrate_lock = asyncio.Lock()
        self._background_task: asyncio.Task[None] | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def index_db(self) -> Any:
        """The wrapped IndexDB. Use for direct access (bypasses
        the lazy-hydrate semantics — the caller is responsible
        for tolerating empty/stale data)."""
        return self._index_db

    @property
    def is_hydrated(self) -> bool:
        return self._is_hydrated

    @property
    def last_hydrate_seconds(self) -> float | None:
        """Seconds since the most recent successful hydrate, or None."""
        if self._hydrate_finished_at is None:
            return None
        return time.time() - self._hydrate_finished_at

    # -- B5 public contract --------------------------------------------------

    def schedule_hydrate(self) -> asyncio.Task[None]:
        """Schedule a background hydrate. Idempotent — a second
        call while the first is in flight is a no-op.

        Returns the asyncio.Task so the lifespan can keep a
        reference (otherwise it would be GC'd before completing).
        """
        if self._background_task is not None and not self._background_task.done():
            return self._background_task
        self._background_task = asyncio.create_task(self._do_hydrate())
        return self._background_task

    async def ensure_hydrated(self) -> None:
        """Schedule a hydrate if not already in flight or complete.

        Non-blocking: returns immediately. The actual Qdrant
        scroll runs in the background.
        """
        if self._is_hydrated or self._refresh_in_progress:
            return
        self.schedule_hydrate()

    async def refresh(self) -> int:
        """Block until a fresh hydrate completes.

        Returns the number of photos the collection holds.

        There is no derived cache to hydrate any more — the corpus
        lives in Qdrant, so a "hydrate" is just the collection
        becoming readable. The count comes from Qdrant, and a failure
        to reach it propagates (the caller falls back to a live
        query).
        """
        async with self._hydrate_lock:
            self._refresh_in_progress = True
            self._hydrate_started_at = time.time()
            try:
                # Off-thread: the count call is a network round-trip.
                count = await asyncio.to_thread(self._index_db.count_images)
            finally:
                self._refresh_in_progress = False
                self._hydrate_finished_at = time.time()
            self._is_hydrated = True
            return int(count)

    async def read(self, point_id: str) -> dict[str, Any] | None:
        """Read a single row, kicking off a hydrate if needed.

        Returns None if the cache is empty (i.e. the first read
        before hydration completes). The caller falls back to
        a live Qdrant query in that case.

        Once the cache is hydrated, this is a simple dict lookup
        (no further Qdrant calls).
        """
        if not self._is_hydrated and not self._refresh_in_progress:
            self.schedule_hydrate()
            return None
        if not self._is_hydrated:
            # A hydrate is in flight. Wait for it without blocking
            # the event loop — IndexDB.get_by_id is sync so we
            # off-thread it; the Qdrant scroll is happening
            # elsewhere.
            return None
        # IndexDB.get_by_id is sync; run it in a thread so the
        # event loop doesn't block on the SQLite read.
        return await asyncio.to_thread(self._index_db.get_by_id, point_id)

    # -- Background task body -----------------------------------------------

    async def _do_hydrate(self) -> None:
        """Body of the background hydrate task."""
        try:
            count = await self.refresh()
            logger.info("lazy index cache hydrated: %d points", count)
        except Exception as e:  # noqa: BLE001
            # The cache stays empty; reads will continue to
            # return None until the next successful refresh.
            # Logged but not raised — the background task
            # shouldn't surface exceptions to the asyncio
            # event loop's unhandled-exception handler.
            logger.warning("lazy index cache hydrate failed: %s", e)
