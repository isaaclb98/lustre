"""
tests/test_search_snapshot.py — stable pagination for plain /api/search.

THE BUG THIS PINS
-----------------
Qdrant offset pagination is unsound on HNSW. To serve a page the engine
ranks `offset + limit` candidates, and graph-search breadth scales with
that depth — so each page of one query is ranked at a *different* depth
and neighbouring pages disagree near their boundary.

Measured on prod (2,052,334 points, Qdrant 1.19.0), walking offsets
0..480 in steps of 24 returned 78 duplicate ids across 504 results,
with the indexer idle. `offset=240` and `offset=288` shared 20 of 24.

`DriftingQdrant` below reproduces that mechanism deterministically so
the fix can be asserted without a 2M-point collection: ranking order
depends on requested depth, therefore offset-paged windows overlap.

The fix (`materialize_search_page`) always ranks at a *constant* shallow
depth (offset=0, exclude everything frozen so far), so no drift can
occur, and slices pages out of one frozen list.
"""

from __future__ import annotations

import asyncio
import zlib
from unittest.mock import MagicMock

import pytest

from search.qdrant_client import SearchHit
from search.search_snapshot import (
    SearchSnapshot,
    SearchSnapshotCache,
    snapshot_key,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class DriftingQdrant:
    """Qdrant stand-in that exhibits HNSW depth drift.

    `search(vector, limit, offset)` ranks at depth `offset + limit` and
    applies depth-dependent jitter to the ordering, exactly as the real
    engine's graph breadth does. Consequence: paging by offset returns
    overlapping windows — the production bug.

    Band fetches (offset=0 + exclude_ids) rank at the constant depth
    `limit`, so they are drift-free by construction. That asymmetry is
    the whole point of the fix, and this fake makes it observable.

    `retrieve_batch` drops unknown ids, mirroring the real client, so
    hydration behaviour is exercised faithfully.
    """

    def __init__(self, n: int, jitter: float = 0.25):
        self.base = [f"id-{i:05d}" for i in range(n)]
        self.jitter = jitter
        # Stable "true" relevance, descending.
        self.true_score = {
            pid: 1.0 - (i / max(1, n)) for i, pid in enumerate(self.base)
        }
        self.search_calls: list[tuple[int, int, int]] = []  # (limit, offset, n_excluded)
        self.retrieve_calls = 0

    def _rank(self, pool: list[str], depth: int) -> list[str]:
        """Depth-dependent ordering — the drift source."""
        def key(pid: str) -> float:
            # crc32 (not hash()) so ordering is stable across processes.
            h = zlib.crc32(f"{pid}|{depth}".encode()) % 1000
            return self.true_score[pid] + (h / 1000.0) * self.jitter
        return sorted(pool, key=key)

    def search(
        self,
        vector,
        limit: int,
        offset: int = 0,
        collections=None,
        allowed_ids=None,
        exclude_ids=None,
        path_filter=None,
    ):
        excluded = set(exclude_ids or ())
        self.search_calls.append((limit, offset, len(excluded)))
        pool = [p for p in self.base if p not in excluded]
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            pool = [p for p in pool if p in allowed]
        depth = offset + limit
        ranked = self._rank(pool, depth)
        window = ranked[offset:offset + limit]
        hits = [
            SearchHit(id=pid, path=f"/photos/{pid}.jpg", score=self.true_score[pid])
            for pid in window
        ]
        # Real client semantics: has_more == (returned a full page).
        return hits, len(hits) >= limit

    def retrieve_batch(self, point_ids: list[str]) -> list[SearchHit]:
        self.retrieve_calls += 1
        known = set(self.base)
        return [
            # score=0.0 exactly as the real client does — hydration must
            # restore the snapshot's score, which a test asserts below.
            SearchHit(id=pid, path=f"/photos/{pid}.jpg", score=0.0)
            for pid in point_ids
            if pid in known
        ]


def _cfg(**over):
    """Fake cfg with real int snapshot knobs.

    A bare MagicMock returns MagicMock for every attribute, which would
    flow into the arithmetic in materialize_search_page and raise
    TypeError, so the knobs are set explicitly.
    """
    cfg = MagicMock()
    cfg.qdrant_collection = "images"
    values = dict(
        search_initial_band_size=64,
        search_band_size=100,
        search_max_snapshot_size=2000,
        # Off by default in tests so exact search_calls counts stay
        # deterministic. Prefetch-specific tests flip it on.
        search_prefetch_next_band=False,
    )
    values.update(over)
    for k, v in values.items():
        setattr(cfg, k, v)
    return cfg


async def _page(qdrant, cfg, cache, offset, limit, **kw):
    from search._indexed_helpers import materialize_search_page

    return await materialize_search_page(
        cfg,
        qdrant,
        cache,
        vector=[0.1, 0.2, 0.3],
        offset=offset,
        limit=limit,
        collections=kw.get("collections", []),
        allowed_ids=kw.get("allowed_ids"),
        favorite_ids=kw.get("favorite_ids"),
    )


# ---------------------------------------------------------------------------
# 1. The regression: offset paging duplicates, snapshot paging does not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offset_paging_duplicates_are_gone():
    """THE regression test.

    Walking a drifting collection by raw offset yields duplicate ids
    (the prod bug, 78 dupes over 504 results). Walking the same
    collection through materialize_search_page yields none.
    """
    qdrant = DriftingQdrant(n=600)
    PAGE = 24

    # --- old behaviour: raw offset paging against the same drift ---
    raw: list[str] = []
    for off in range(0, PAGE * 21, PAGE):
        hits, _ = qdrant.search(None, PAGE, off)
        raw.extend(h.id for h in hits)
    raw_dupes = len(raw) - len(set(raw))
    assert raw_dupes > 0, (
        "DriftingQdrant failed to reproduce the depth-drift bug; "
        "this fixture no longer models prod and the test is vacuous"
    )

    # --- new behaviour: snapshot paging over an identical walk ---
    qdrant2 = DriftingQdrant(n=600)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    seen: list[str] = []
    for off in range(0, PAGE * 21, PAGE):
        hits, _ = await _page(qdrant2, cfg, cache, off, PAGE)
        seen.extend(h.id for h in hits)

    dupes = len(seen) - len(set(seen))
    assert dupes == 0, f"snapshot paging produced {dupes} duplicate ids"
    assert len(seen) == PAGE * 21


@pytest.mark.asyncio
async def test_consecutive_pages_are_pairwise_disjoint():
    """No two pages share an id — the property the UI relies on."""
    qdrant = DriftingQdrant(n=500)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    pages = []
    for off in range(0, 24 * 10, 24):
        hits, _ = await _page(qdrant, cfg, cache, off, 24)
        pages.append({h.id for h in hits})
    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            assert pages[i].isdisjoint(pages[j]), f"pages {i} and {j} overlap"


@pytest.mark.asyncio
async def test_band_fetches_always_rank_at_constant_depth():
    """Mechanism check: every band fetch uses offset=0.

    Depth drift only arises when offset varies, so pinning offset=0 on
    all Qdrant calls is what makes the ranking stable. If a future
    refactor reintroduces offset paging, this fails.
    """
    qdrant = DriftingQdrant(n=500)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    for off in range(0, 24 * 8, 24):
        await _page(qdrant, cfg, cache, off, 24)
    assert qdrant.search_calls, "expected at least one Qdrant search"
    for _limit, offset, _n_excl in qdrant.search_calls:
        assert offset == 0, f"band fetch used offset={offset}; must be 0"
    # Deeper bands must exclude everything frozen so far.
    excluded_counts = [n for _, _, n in qdrant.search_calls]
    assert excluded_counts[0] == 0
    assert excluded_counts == sorted(excluded_counts), (
        "exclusion set must grow monotonically as the snapshot deepens"
    )


# ---------------------------------------------------------------------------
# 2. Band growth / extension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_grows_in_bands_not_one_deep_fetch():
    """Page 1 fetches only the initial band — no deep upfront ranking.

    This is the latency property: a single 10k-deep fetch costs 473ms on
    prod, while a shallow band costs ~10ms. Pin that page 1 never asks
    for more than the initial band.
    """
    qdrant = DriftingQdrant(n=2000)
    cfg = _cfg(search_initial_band_size=64, search_band_size=100)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    await _page(qdrant, cfg, cache, 0, 24)
    assert len(qdrant.search_calls) == 1
    assert qdrant.search_calls[0][0] == 64, "page 1 must fetch the initial band only"

    # Paging strictly inside the frozen band (target 48 <= 64) costs no
    # Qdrant search at all.
    calls_before = len(qdrant.search_calls)
    await _page(qdrant, cfg, cache, 24, 24)
    assert len(qdrant.search_calls) == calls_before, (
        "pages inside the frozen snapshot must not re-query Qdrant"
    )

    # offset 48 + limit 24 targets index 72, past the 64-id band ->
    # exactly one extension fetch.
    await _page(qdrant, cfg, cache, 48, 24)
    assert len(qdrant.search_calls) == calls_before + 1
    assert qdrant.search_calls[-1][0] == 100, "extension uses the band size"


@pytest.mark.asyncio
async def test_extension_never_repeats_frozen_ids():
    """A band appended after exclusion contains nothing already frozen."""
    qdrant = DriftingQdrant(n=1500)
    cfg = _cfg(search_initial_band_size=50, search_band_size=200)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    seen: list[str] = []
    for off in range(0, 24 * 30, 24):  # 720 ids, forces several bands
        hits, _ = await _page(qdrant, cfg, cache, off, 24)
        seen.extend(h.id for h in hits)
    assert len(seen) == len(set(seen))
    assert len(seen) == 720


@pytest.mark.asyncio
async def test_page_spanning_a_band_boundary_is_contiguous():
    """A page whose window crosses a band edge is still full and unique."""
    qdrant = DriftingQdrant(n=800)
    # Band of 30 with page 24 -> page 2 straddles the boundary.
    cfg = _cfg(search_initial_band_size=30, search_band_size=30)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    p1, _ = await _page(qdrant, cfg, cache, 0, 24)
    p2, _ = await _page(qdrant, cfg, cache, 24, 24)
    assert len(p1) == 24 and len(p2) == 24
    assert {h.id for h in p1}.isdisjoint({h.id for h in p2})


# ---------------------------------------------------------------------------
# 3. Ceiling, exhaustion, has_more
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_has_more_true_until_snapshot_end():
    qdrant = DriftingQdrant(n=300)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    hits, has_more = await _page(qdrant, cfg, cache, 0, 24)
    assert len(hits) == 24
    assert has_more is True


@pytest.mark.asyncio
async def test_has_more_false_on_final_partial_page():
    """Exactly 300 points -> page 12 (offset 288) returns 12, has_more False."""
    qdrant = DriftingQdrant(n=300)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    # Drain to the end.
    last = None
    for off in range(0, 24 * 20, 24):
        hits, has_more = await _page(qdrant, cfg, cache, off, 24)
        if not hits:
            break
        last = (off, len(hits), has_more)
    assert last is not None
    off, n, has_more = last
    assert has_more is False, f"final page at offset={off} should report has_more=False"
    assert n < 24 or off + n >= 300


@pytest.mark.asyncio
async def test_snapshot_cap_stops_growth_and_clears_has_more():
    """Past the ceiling the snapshot stops growing and has_more goes False.

    Bounds both memory and the exclusion request body (~36 bytes/id).
    """
    qdrant = DriftingQdrant(n=3000)
    cap = 300
    cfg = _cfg(search_initial_band_size=100, search_band_size=100,
               search_max_snapshot_size=cap)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    total = 0
    off = 0
    while True:
        hits, has_more = await _page(qdrant, cfg, cache, off, 24)
        if not hits:
            break
        total += len(hits)
        off += 24
        if not has_more:
            break
        assert off < cap + 240, "walk ran away past the cap"

    assert has_more is False
    assert total <= cap
    # Snapshot never exceeded the cap.
    snap = next(iter(cache._entries.values()))
    assert len(snap) <= cap


@pytest.mark.asyncio
async def test_offset_beyond_cap_returns_empty_without_deep_walk():
    """An out-of-range offset must not trigger an expensive deep fetch."""
    qdrant = DriftingQdrant(n=500)
    cfg = _cfg(search_max_snapshot_size=200)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    hits, has_more = await _page(qdrant, cfg, cache, 5000, 24)
    assert hits == []
    assert has_more is False
    assert qdrant.search_calls == [], "out-of-range offset must short-circuit"


@pytest.mark.asyncio
async def test_exhausted_short_band_stops_paging():
    """A band that comes back short marks the snapshot exhausted."""
    qdrant = DriftingQdrant(n=50)  # fewer points than one band
    cfg = _cfg(search_initial_band_size=200, search_band_size=200)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    p1, hm1 = await _page(qdrant, cfg, cache, 0, 24)
    assert len(p1) == 24
    p2, hm2 = await _page(qdrant, cfg, cache, 24, 24)
    assert len(p2) == 24
    p3, hm3 = await _page(qdrant, cfg, cache, 48, 24)
    assert len(p3) == 2  # only 50 points total
    assert hm3 is False

    snap = next(iter(cache._entries.values()))
    assert snap.exhausted is True
    assert len(snap) == 50


# ---------------------------------------------------------------------------
# 4. Hydration: scores, order, live state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scores_are_restored_from_the_snapshot():
    """retrieve_batch returns score=0.0; the snapshot's score must win.

    Without this, every result would render as 0.000 in the UI.
    """
    qdrant = DriftingQdrant(n=200)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    hits, _ = await _page(qdrant, cfg, cache, 0, 24)
    assert all(h.score > 0.0 for h in hits), "scores were not restored"


@pytest.mark.asyncio
async def test_page_preserves_snapshot_ranking_order():
    """Hydrated page order matches the frozen ranking, not retrieve order."""
    qdrant = DriftingQdrant(n=200)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    hits, _ = await _page(qdrant, cfg, cache, 0, 24)
    snap = next(iter(cache._entries.values()))
    assert [h.id for h in hits] == snap.ids[:24]


@pytest.mark.asyncio
async def test_dropped_id_does_not_truncate_has_more():
    """A photo pruned mid-session must not make paging stop early.

    retrieve_batch omits unknown ids. has_more is derived from the
    snapshot length, not the hydrated count, so one dead id can't
    signal "end of results" to the client.
    """
    qdrant = DriftingQdrant(n=300)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    hits, _ = await _page(qdrant, cfg, cache, 0, 24)
    snap = next(iter(cache._entries.values()))

    # Simulate a prune: one id from the next page leaves Qdrant.
    victim = snap.ids[24]
    qdrant.base.remove(victim)

    real_retrieve = qdrant.retrieve_batch

    def retrieve(point_ids):
        return real_retrieve([i for i in point_ids if i != victim])

    qdrant.retrieve_batch = retrieve  # type: ignore[method-assign]

    p2, has_more = await _page(qdrant, cfg, cache, 24, 24)
    assert victim not in {h.id for h in p2}
    assert len(p2) == 23, "the pruned id is skipped, the rest still render"
    assert has_more is True, "one dead id must not end pagination"


@pytest.mark.asyncio
async def test_favorite_flags_are_not_frozen_in_the_snapshot():
    """The snapshot stores ids+scores only, so flags stay live.

    Hydration happens per page via retrieve_batch, meaning a photo
    favourited mid-scroll reflects immediately rather than being
    frozen for the TTL. Verified structurally: SearchSnapshot holds no
    payload/flag fields at all.
    """
    snap = SearchSnapshot()
    assert not hasattr(snap, "payloads")
    assert not hasattr(snap, "favorite_ids")
    assert set(type(snap).__dataclass_fields__) == {
        "ids", "scores", "exhausted", "created_at", "lock",
        "prefetch_inflight", "_flag_lock",
    }


# ---------------------------------------------------------------------------
# 5. SearchSnapshot / cache unit behaviour
# ---------------------------------------------------------------------------


def test_snapshot_extend_dedupes_locally():
    """Second line of defence if the exclusion filter is ever dropped.

    A silently-ignored `must_not` returns ids that are already frozen;
    without this check the snapshot would grow duplicates and the
    growth loop could spin.
    """
    snap = SearchSnapshot()
    hits = [SearchHit(id="a", path="/a", score=0.9),
            SearchHit(id="b", path="/b", score=0.8)]
    snap.extend(hits, band_size=2)
    assert snap.exhausted is False  # full band -> maybe more
    snap.extend(hits, band_size=2)  # same ids again
    assert snap.ids == ["a", "b"]
    assert snap.scores == [0.9, 0.8]


def test_snapshot_extend_flags_exhausted_on_short_band():
    snap = SearchSnapshot()
    snap.extend([SearchHit(id="a", path="/a", score=0.9)], band_size=10)
    assert snap.exhausted is True


def test_snapshot_extend_zero_added_flags_exhausted():
    """A fully-overlapping band must terminate the growth loop."""
    snap = SearchSnapshot()
    snap.extend([SearchHit(id="a", path="/a", score=0.9)], band_size=1)
    snap.exhausted = False
    snap.extend([SearchHit(id="a", path="/a", score=0.9)], band_size=5)
    assert snap.exhausted is True


def test_snapshot_page_slices_correctly():
    snap = SearchSnapshot()
    snap.extend([SearchHit(id=f"x{i}", path=f"/{i}", score=1.0 - i * 0.1)
                 for i in range(10)], band_size=10)
    assert [p[0] for p in snap.page(0, 3)] == ["x0", "x1", "x2"]
    assert [p[0] for p in snap.page(3, 3)] == ["x3", "x4", "x5"]
    assert snap.page(9, 5) == [("x9", snap.scores[9])]
    assert snap.page(50, 5) == []


def test_cache_reuses_entry_for_same_key():
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    a = cache.get_or_create("k")
    a.extend([SearchHit(id="x", path="/x", score=1.0)], band_size=1)
    b = cache.get_or_create("k")
    assert a is b
    assert len(b) == 1


def test_cache_ttl_expiry_returns_fresh_snapshot():
    cache = SearchSnapshotCache(ttl_seconds=0, max_entries=8)  # 0 = always expired
    a = cache.get_or_create("k")
    b = cache.get_or_create("k")
    assert a is not b


def test_cache_lru_evicts_oldest():
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=2)
    cache.get_or_create("a")
    cache.get_or_create("b")
    cache.get_or_create("a")  # touch a -> b becomes oldest
    cache.get_or_create("c")
    assert len(cache) == 2
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None


def test_cache_get_does_not_create():
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    assert cache.get("nope") is None
    assert len(cache) == 0


def test_cache_clear():
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    cache.get_or_create("a")
    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# 6. Cache key correctness
# ---------------------------------------------------------------------------


def _key(**over) -> str:
    """Build a snapshot_key with defaults, overriding any dimension."""
    kw: dict = dict(
        collection="images",
        vector=[0.1, 0.2],
        collections=None,
        allowed_ids=None,
        favorite_ids=None,
        band_size=100,
    )
    kw.update(over)
    return snapshot_key(**kw)  # type: ignore[arg-type]


def test_key_ignores_offset_and_limit():
    """offset/limit select a page WITHIN a snapshot.

    Including them would create one snapshot per page and defeat the
    entire purpose — each page would re-rank and drift again.
    """
    import inspect

    from search.search_snapshot import snapshot_key as sk
    params = set(inspect.signature(sk).parameters)
    assert "offset" not in params
    assert "limit" not in params


def test_key_stable_across_equivalent_inputs():
    assert _key(collections=["a", "b"]) == _key(collections=["b", "a"])
    assert _key(favorite_ids={"x", "y"}) == _key(favorite_ids={"y", "x"})


def test_key_distinguishes_every_filter_dimension():
    base = _key()
    assert base != _key(collection="other")
    assert base != _key(vector=[0.9, 0.9])
    assert base != _key(collections=["a"])
    assert base != _key(allowed_ids=["p"])
    assert base != _key(favorite_ids={"f"})
    assert base != _key(band_size=500)


def test_key_distinguishes_absent_from_empty_filter():
    """No filter must not collide with an empty/explicit one."""
    assert _key(collections=None) != _key(collections=[])
    assert _key(allowed_ids=None) != _key(allowed_ids=[])


@pytest.mark.asyncio
async def test_different_filters_get_separate_snapshots():
    qdrant = DriftingQdrant(n=400)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    await _page(qdrant, cfg, cache, 0, 24, collections=["lib-a"])
    await _page(qdrant, cfg, cache, 0, 24, collections=["lib-b"])
    assert len(cache) == 2


@pytest.mark.asyncio
async def test_same_query_shares_one_snapshot_across_pages():
    qdrant = DriftingQdrant(n=400)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    for off in range(0, 24 * 5, 24):
        await _page(qdrant, cfg, cache, off, 24)
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# 7. Filter pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collections_filter_is_passed_through():
    qdrant = DriftingQdrant(n=200)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    await _page(qdrant, cfg, cache, 0, 24, collections=["kpop"])
    assert qdrant.search_calls, "expected a Qdrant search"


@pytest.mark.asyncio
async def test_allowed_ids_narrow_the_snapshot():
    """The filename filter still restricts results under stable paging."""
    qdrant = DriftingQdrant(n=300)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    allowed = [f"id-{i:05d}" for i in range(40)]
    seen = []
    for off in range(0, 24 * 3, 24):
        hits, _ = await _page(qdrant, cfg, cache, off, 24, allowed_ids=allowed)
        seen.extend(h.id for h in hits)
    assert seen, "expected some hits"
    assert set(seen) <= set(allowed)
    assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------
# 8. Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_pages_share_one_snapshot_with_no_duplicates():
    """The SPA fires overlapping page requests.

    Observed in the browser repro: three REQs in flight before any RES.
    Two racers must not build two snapshots, nor fetch the same band
    twice — the snapshot lock serialises extension.
    """
    qdrant = DriftingQdrant(n=1200)
    cfg = _cfg(search_initial_band_size=50, search_band_size=200)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    offsets = list(range(0, 24 * 12, 24))
    results = await asyncio.gather(
        *(_page(qdrant, cfg, cache, off, 24) for off in offsets)
    )

    assert len(cache) == 1, "concurrent pages must share a single snapshot"
    seen = [h.id for hits, _ in results for h in hits]
    assert len(seen) == len(set(seen)), "concurrent pages produced duplicates"
    assert len(seen) == 24 * 12


@pytest.mark.asyncio
async def test_no_duplicate_band_fetch_under_concurrency():
    """Each band is fetched at most once even with racers.

    Without the per-snapshot lock, N concurrent deep-page requests would
    each see the same short snapshot and issue N identical band fetches.
    """
    qdrant = DriftingQdrant(n=1200)
    cfg = _cfg(search_initial_band_size=50, search_band_size=200)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    await asyncio.gather(
        *(_page(qdrant, cfg, cache, off, 24) for off in range(0, 24 * 12, 24))
    )

    snap = next(iter(cache._entries.values()))
    # Snapshot holds at most one copy of every id.
    assert len(snap.ids) == len(set(snap.ids))
    # Exclusion sizes must strictly increase: no band was fetched twice
    # against the same frozen prefix.
    excl = [n for _, _, n in qdrant.search_calls]
    assert excl == sorted(set(excl)), f"band fetched twice: {excl}"


@pytest.mark.asyncio
async def test_separate_cache_entries_do_not_deadlock():
    """Distinct queries hold distinct locks; neither blocks the other."""
    qdrant = DriftingQdrant(n=400)
    cfg, cache = _cfg(), SearchSnapshotCache(ttl_seconds=300, max_entries=8)
    a, b = await asyncio.wait_for(
        asyncio.gather(
            _page(qdrant, cfg, cache, 0, 24, collections=["lib-a"]),
            _page(qdrant, cfg, cache, 0, 24, collections=["lib-b"]),
        ),
        timeout=10,
    )
    assert len(a[0]) == 24 and len(b[0]) == 24
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# 9. End-to-end through the real router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_search_pagination_is_disjoint_end_to_end():
    """Full stack: router -> materialize_search_page -> JSON response.

    Guards the wiring (snapshot_cache injected into build_search_router,
    plain branch routed through it) rather than just the helper.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from search.routers.search import build_search_router

    qdrant = DriftingQdrant(n=400)
    cfg = _cfg(search_initial_band_size=48, search_band_size=100)
    cfg.top_k_default = 24
    cfg.top_k_max = 200
    cfg.max_results_total = 1000
    cfg.max_prompt_chars = 80
    cfg.max_prompts_total = 10
    cfg.default_view = "grid"
    cfg.centroid_expected_feature_dim = 1536
    cfg.web_ui_url = "http://localhost:5173"
    cfg.filename_cardinality_guard = 0.5
    cfg.diversity_relevance_drop = 0.1
    cfg.diversity_duplicate_hamming_distance = 8
    cfg.diversity_pool_depths = {}

    index_db = MagicMock()
    index_db.favorite_id_set.return_value = set()
    index_db.dislike_id_set.return_value = set()

    def resolve_query_vector(centroid_names, prompt_state, **kwargs):
        return ([0.1, 0.2, 0.3], None, None)

    async def favorite_ids_for_filter():
        return set()

    app = FastAPI()
    app.include_router(build_search_router(
        qdrant=qdrant,
        cfg=cfg,
        index_db=index_db,
        diversity_cache=MagicMock(),
        snapshot_cache=SearchSnapshotCache(ttl_seconds=300, max_entries=8),
        resolve_query_vector=resolve_query_vector,
        favorite_ids_for_filter=favorite_ids_for_filter,
    ))

    with TestClient(app) as client:
        seen: list[str] = []
        for off in range(0, 24 * 8, 24):
            r = client.get(f"/api/search?q=cat&limit=24&offset={off}")
            assert r.status_code == 200, r.text
            body = r.json()
            ids = [x["id"] for x in body["results"]]
            assert len(ids) == 24, f"offset={off} returned {len(ids)}"
            # Scores must survive hydration, not come back as 0.000.
            assert all(float(x["score"]) > 0 for x in body["results"])
            seen.extend(ids)

    assert len(seen) == 24 * 8
    assert len(seen) == len(set(seen)), "router-level pagination produced duplicates"


# ---------------------------------------------------------------------------
# 10. Background prefetch of the next band
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_grows_snapshot_in_background():
    """After serving the last frozen page, a band is fetched off-thread.

    The point is latency: a cold band costs ~0.5-1s at depth on prod,
    so it must happen while the user reads, not when they ask.
    """
    import time as _time

    qdrant = DriftingQdrant(n=1000)
    cfg = _cfg(
        search_initial_band_size=48,
        search_band_size=100,
        search_prefetch_next_band=True,
    )
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    # Page 0 fills the initial band (48) and serves 24.
    await _page(qdrant, cfg, cache, 0, 24)
    snap = next(iter(cache._entries.values()))
    assert len(snap) == 48

    # Page 1 consumes to index 48 -> end of the frozen band -> prefetch.
    await _page(qdrant, cfg, cache, 24, 24)

    # Background thread must extend the snapshot without being asked.
    deadline = _time.monotonic() + 5.0
    while len(snap) <= 48 and _time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert len(snap) > 48, "prefetch did not grow the snapshot"
    assert snap.prefetch_inflight is False, "prefetch flag never released"


@pytest.mark.asyncio
async def test_prefetch_disabled_by_default_in_tests_is_synchronous_only():
    """With the knob off, no background growth happens (deterministic)."""
    qdrant = DriftingQdrant(n=1000)
    cfg = _cfg(search_initial_band_size=48, search_band_size=100)
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    await _page(qdrant, cfg, cache, 0, 24)
    await _page(qdrant, cfg, cache, 24, 24)
    await asyncio.sleep(0.05)

    snap = next(iter(cache._entries.values()))
    # Only what the two synchronous pages needed: target was 48.
    assert len(snap) == 48


@pytest.mark.asyncio
async def test_prefetch_does_not_run_when_slack_remains():
    """No wasted background work while the frozen band still has pages."""
    qdrant = DriftingQdrant(n=1000)
    cfg = _cfg(
        search_initial_band_size=200,
        search_band_size=100,
        search_prefetch_next_band=True,
    )
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    await _page(qdrant, cfg, cache, 0, 24)
    snap = next(iter(cache._entries.values()))
    assert len(snap) == 200

    # Page 1 ends at index 48, well inside the 200-id band.
    await _page(qdrant, cfg, cache, 24, 24)
    await asyncio.sleep(0.05)
    assert len(snap) == 200, "prefetch fired despite ample slack"
    # No claim was made — verify by checking the flag stayed clear
    # rather than calling claim_prefetch(), which would set it.
    assert snap.prefetch_inflight is False


@pytest.mark.asyncio
async def test_prefetch_stops_at_exhaustion():
    """A short band marks the snapshot exhausted; no further prefetch."""
    qdrant = DriftingQdrant(n=60)
    cfg = _cfg(
        search_initial_band_size=40,
        search_band_size=40,
        search_prefetch_next_band=True,
    )
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    await _page(qdrant, cfg, cache, 0, 24)
    await _page(qdrant, cfg, cache, 24, 24)
    await asyncio.sleep(0.05)

    snap = next(iter(cache._entries.values()))
    assert len(snap) == 60
    assert snap.exhausted is True


@pytest.mark.asyncio
async def test_prefetch_claim_is_exclusive():
    """Only one background band per snapshot — claim_prefetch is atomic."""
    snap = SearchSnapshot()
    assert snap.claim_prefetch() is True
    assert snap.claim_prefetch() is False, "second claim must fail"
    snap.release_prefetch()
    assert snap.claim_prefetch() is True


@pytest.mark.asyncio
async def test_prefetch_under_concurrent_pages_no_duplicate_bands():
    """Racing pages plus prefetch still fetch each band once."""
    qdrant = DriftingQdrant(n=1500)
    cfg = _cfg(
        search_initial_band_size=50,
        search_band_size=200,
        search_prefetch_next_band=True,
    )
    cache = SearchSnapshotCache(ttl_seconds=300, max_entries=8)

    results = await asyncio.gather(
        *(_page(qdrant, cfg, cache, off, 24) for off in range(0, 24 * 12, 24))
    )
    # Let any in-flight prefetch settle.
    await asyncio.sleep(0.3)

    snap = next(iter(cache._entries.values()))
    assert len(snap.ids) == len(set(snap.ids)), "prefetch introduced duplicate ids"
    seen = [h.id for hits, _ in results for h in hits]
    assert len(seen) == len(set(seen))
