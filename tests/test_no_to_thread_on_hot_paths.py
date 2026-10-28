"""
tests/test_no_to_thread_on_hot_paths.py — C1 regression gate.

Per the plan §C1: "A grep-based test fails if [an `asyncio.to_thread`
wrapper] re-appears" on a search-side hot path.

Hot paths = the per-request work inside route handlers in
search/routers/*.py. Excluded from the check:
  - Test code
  - The lazy_index_cache wrapper itself (intentional threading)
  - Background tasks in lifespan (search/app.py)

The test counts the per-route to_thread calls and asserts the
count is at or below the current baseline. A new to_thread on a
per-request hot path that already has one is a budget violation;
a new to_thread in a previously-zero-count route is an even
bigger violation.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTERS_DIR = Path("search/routers")

# Per-route baseline counts. Captured at the time this test was
# added; a regression would add a new call site and exceed the
# budget. If a future PR deliberately adds a to_thread on a hot
# path, the corresponding line should grow in tandem with this
# table. The point of the table is to surface the regression at
# review time, not to forbid the pattern outright.
BASELINE: dict[str, int] = {
    "admin_index.py": 0,
    "albums.py": 15,
    "collections.py": 0,
    "centroids.py": 0,
    "centroids_list.py": 0,
    "centroids_search.py": 0,
    "dislikes.py": 6,
    "favorites.py": 6,
    "for_you.py": 8,  # round‑14: cold‑start + reset_feedback wrap
    # 3: the deck is a cursor into the corpus's id space, so the
    # route makes two network calls on the request path — the facet
    # that sizes the deck, and the scroll that reads the page's ids.
    "random.py": 3,
    "saved_searches.py": 4,
    "search.py": 5,  # 5: count_points wraps qdrant.count for pool-size clamp
    "similar.py": 4,
    "system.py": 7,
    "thumbnails.py": 0,
}

# Routes that MUST have zero to_thread on the request path.
# Any new addition is a C1 violation. The remaining to_thread
# calls in non-zero routes are wrapping single SQLite reads —
# acceptable until those IndexDB methods become async.
NONZERO_ALLOWED: set[str] = {
    "albums.py", "dislikes.py", "favorites.py", "for_you.py",
    "random.py", "saved_searches.py", "search.py", "similar.py",
    "system.py",
}


def _count_to_thread_in(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(re.findall(r"asyncio\.to_thread\(", text))


def test_zero_tolerance_routes_have_no_to_thread():
    """C1 regression: routes NOT in NONZERO_ALLOWED must not introduce
    asyncio.to_thread on the request hot path."""
    offenders: list[tuple[str, int]] = []
    for path in ROUTERS_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        if path.name in NONZERO_ALLOWED:
            continue
        count = _count_to_thread_in(path)
        if count > 0:
            offenders.append((path.name, count))
    assert not offenders, (
        f"C1 violation: zero-tolerance routes have asyncio.to_thread: {offenders}"
    )


def test_no_route_exceeds_baseline():
    """C1 regression: per-route counts must not exceed the baseline.

    The baseline reflects the pre-C1 state; new call sites that
    don't replace an existing one are a budget violation.
    """
    offenders: list[tuple[str, int, int]] = []
    for path in ROUTERS_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        baseline = BASELINE.get(path.name, 0)
        count = _count_to_thread_in(path)
        if count > baseline:
            offenders.append((path.name, count, baseline))
    assert not offenders, (
        f"C1 violation: per-route asyncio.to_thread count grew: {offenders}"
    )


def test_to_thread_only_in_known_routes():
    """C1 regression: no new files in search/routers/ that weren't
    in the baseline table. The point is to surface new files at
    review time so the baseline can be updated deliberately."""
    actual = {p.name for p in ROUTERS_DIR.glob("*.py") if p.name != "__init__.py"}
    expected = set(BASELINE.keys())
    extra = actual - expected
    missing = expected - actual
    assert not extra and not missing, (
        f"C1 regression: search/routers/ has unknown routes. "
        f"extra={sorted(extra)} missing={sorted(missing)}. "
        f"Update tests/test_no_to_thread_on_hot_paths.py BASELINE."
    )


def test_favorite_id_set_uses_batch_query():
    """C1 perf: favorite_id_set is a single IN-clause query, not N
    individual get_by_id calls. The C1 win is a ~10× reduction
    in SQLite round trips for /api/search's per-result fav lookup.
    """
    import inspect

    from search._indexed_helpers import favorite_id_set_sync

    # The body of favorite_id_set_sync should NOT call get_by_id in
    # a loop. Source-level check: no `for pid` / `while pid` over
    # the point_ids list with a get_by_id inside.
    source = inspect.getsource(favorite_id_set_sync)
    assert "for pid" not in source, (
        "C1 regression: favorite_id_set_sync regressed to a per-id "
        "loop. It should be a single IN-clause query."
    )
    assert "while pid" not in source
