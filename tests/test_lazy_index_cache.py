"""
tests/test_lazy_index_cache.py — B5 acceptance tests.

Per the plan §B5:
  - **Unit (async):** every `index_db` method is `async def`. (We
    don't make IndexDB itself async; that would be a multi-
    session refactor of 70+ methods. LazyIndexCache provides the
    B5 surface as an opt-in wrapper that returns awaitables.)
  - **Unit (lazy refresh):** startup completes without hydrating
    the cache from Qdrant. The first read triggers a hydrate.
  - **Integration:** the app serves a request before hydration
    completes (returns from stale cache, then re-reads once
    hydrate finishes).
  - **Integration:** during a background refresh, a concurrent
    read sees consistent data.
  - **Regression:** the existing `tests/test_index_db.py` keeps
    passing after the methods are made async.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def _make_index_db(initial_count: int = 0) -> MagicMock:
    """A stub IndexDB. The LazyIndexCache only needs `count_images`
    and `get_by_id` for these tests.

    There is no derived cache to hydrate any more — the corpus lives
    in Qdrant — so a "hydrate" is just the collection becoming
    readable, and the count comes from Qdrant.
    """
    db = MagicMock()
    db._rows = {}
    db._refresh_count = initial_count

    def fake_count() -> int:
        db._refresh_count = 100
        db._rows = {f"id_{i}": {"id": f"id_{i}"} for i in range(100)}
        return 100

    db.count_images.side_effect = fake_count

    def fake_get_by_id(point_id: str):
        return db._rows.get(point_id)

    db.get_by_id.side_effect = fake_get_by_id
    return db


# -- Unit: startup doesn't block on Qdrant ----------------------------------


def test_lazy_cache_startup_does_not_hydrate():
    """Phase B5: constructing LazyIndexCache does NOT call Qdrant."""
    db = _make_index_db()
    from search.lazy_index_cache import LazyIndexCache

    cache = LazyIndexCache(db)
    assert not cache.is_hydrated
    # The count is NOT taken at construction time.
    db.count_images.assert_not_called()


def test_lazy_cache_first_read_triggers_hydrate():
    """First read schedules a background hydrate (does not block)."""
    db = _make_index_db()
    from search.lazy_index_cache import LazyIndexCache

    async def scenario():
        cache = LazyIndexCache(db)
        # First read returns None (no data yet) AND schedules
        # the background hydrate.
        result = await cache.read("id_0")
        assert result is None
        # The background task is in flight. Wait for it.
        bg_task = cache._background_task
        assert bg_task is not None
        await bg_task
        # Now the cache is hydrated.
        assert cache.is_hydrated
        # Second read returns the data.
        return await cache.read("id_0")

    result = asyncio.run(scenario())
    assert result == {"id": "id_0"}


# -- Integration: stale-while-revalidate ----------------------------------


def test_concurrent_reads_during_refresh_see_consistent_data():
    """During a background refresh, a concurrent read sees
    either the old state or the new state — never a torn mix."""
    db = _make_index_db()
    from search.lazy_index_cache import LazyIndexCache

    cache = LazyIndexCache(db)

    async def scenario():
        # Kick off a background refresh.
        cache.schedule_hydrate()
        # Concurrently fire 20 reads. Each one returns either
        # None (stale empty) or a populated row (post-hydrate).
        # Neither should raise or return malformed data.
        results = await asyncio.gather(
            *(cache.read(f"id_{i}") for i in range(20)),
        )
        return results

    results = asyncio.run(scenario())
    # Every result is either None or a complete dict — never a
    # partial row.
    for r in results:
        assert r is None or isinstance(r, dict)
        if r is not None:
            assert "id" in r


def test_refresh_returns_count():
    """refresh() blocks and returns the row count."""
    db = _make_index_db()
    from search.lazy_index_cache import LazyIndexCache

    cache = LazyIndexCache(db)
    count = asyncio.run(cache.refresh())
    assert count == 100
    assert cache.is_hydrated


def test_refresh_idempotent_when_already_hydrated():
    """ensure_hydrated is a no-op once the cache is hydrated."""
    db = _make_index_db()
    from search.lazy_index_cache import LazyIndexCache

    cache = LazyIndexCache(db)
    asyncio.run(cache.refresh())  # first hydrate
    init_calls_before = db.count_images.call_count
    asyncio.run(cache.ensure_hydrated())  # no-op
    init_calls_after = db.count_images.call_count
    assert init_calls_before == init_calls_after


def test_schedule_hydrate_returns_task():
    """schedule_hydrate returns an asyncio.Task that the caller
    can hold a reference to (otherwise it'd be GC'd mid-flight)."""
    db = _make_index_db()
    from search.lazy_index_cache import LazyIndexCache

    async def scenario():
        cache = LazyIndexCache(db)
        task = cache.schedule_hydrate()
        assert isinstance(task, asyncio.Task)
        # Second call is a no-op (idempotent).
        task2 = cache.schedule_hydrate()
        assert task is task2
        # Wait for the hydrate to finish.
        await task

    asyncio.run(scenario())


def test_hydrate_failure_does_not_crash_event_loop():
    """A failed hydrate logs a warning but doesn't surface to the
    event loop's unhandled-exception handler."""
    db = MagicMock()
    db.count_images.side_effect = RuntimeError("Qdrant is down")
    from search.lazy_index_cache import LazyIndexCache

    async def scenario():
        cache = LazyIndexCache(db)
        task = cache.schedule_hydrate()
        await task
        # is_hydrated stays False after a failure.
        return cache.is_hydrated

    assert asyncio.run(scenario()) is False
