"""
tests/test_indexed_helpers.py — search/_indexed_helpers.py contract.

Pure (well, I/O-bound) helpers previously closure-bound to
create_app. Lifting them is what lets /api/search and
/api/centroids/{name}/search extract.

Each helper takes index_db + cfg explicitly so callers
(including the routers that will eventually wrap /api/search)
own their dependency surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _fake_cfg():
    c = MagicMock()
    c.web_ui_url = "http://localhost:5173"
    c.filename_cardinality_guard = 0.5
    return c


# --- favorite_id_set_sync ---

def test_favorite_id_set_sync_returns_ids_with_is_favorite_set():
    """Phase C1: single IN-clause query returns the favourited subset."""
    from search._indexed_helpers import favorite_id_set_sync
    db = MagicMock()
    db.favorite_id_set.return_value = {"a", "c"}
    assert favorite_id_set_sync(db, ["a", "b", "c"]) == {"a", "c"}
    db.favorite_id_set.assert_called_once_with(["a", "b", "c"])


def test_favorite_id_set_sync_skips_missing_rows():
    """Phase C1: the IN-clause query naturally filters out ids
    that aren't in the favourites table."""
    from search._indexed_helpers import favorite_id_set_sync
    db = MagicMock()
    db.favorite_id_set.return_value = {"a"}
    assert favorite_id_set_sync(db, ["a", "missing"]) == {"a"}


# --- favorite_id_set (async wrapper) ---

@pytest.mark.asyncio
async def test_favorite_id_set_runs_in_thread():
    """Async wrapper: SQLite read off the event loop.

    After the C1 refactor, the sync impl is a single IN-clause
    query (index_db.favorite_id_set), not a per-id loop.
    """
    from search._indexed_helpers import favorite_id_set
    db = MagicMock()
    db.favorite_id_set.return_value = {"a"}
    result = await favorite_id_set(db, ["a"])
    assert result == {"a"}
    db.favorite_id_set.assert_called_once_with(["a"])


# --- results_from_hits ---

@pytest.mark.asyncio
async def test_results_from_hits_with_pre_resolved_sets_skips_db():
    """Hot path: caller already has fav/dis sets, no DB hit."""
    from search._indexed_helpers import results_from_hits

    cfg = _fake_cfg()
    db = MagicMock()

    h1 = MagicMock(id="a", path="/a.jpg", score=0.9, payload={})
    h2 = MagicMock(id="b", path="/b.jpg", score=0.85, payload={})

    results = await results_from_hits(
        db, cfg=cfg, hits=[h1, h2],
        favorite_ids={"a"}, dislike_ids={"b"},
    )
    assert len(results) == 2
    assert results[0].id == "a"
    assert results[0].is_favorite is True
    assert results[0].is_disliked is False
    assert results[1].is_favorite is False
    assert results[1].is_disliked is True
    # Pre-resolved sets mean no DB call.
    db.get_by_id.assert_not_called()
    db.dislike_id_set.assert_not_called()


@pytest.mark.asyncio
async def test_results_from_hits_resolves_sets_when_missing():
    """Cold path: parallel-fetch fav/dis when caller didn't pre-resolve."""
    from search._indexed_helpers import results_from_hits

    cfg = _fake_cfg()
    db = MagicMock()
    # favorite_id_set's sync impl is now a single IN-clause query.
    db.favorite_id_set.return_value = {"a"}
    db.dislike_id_set.return_value = {"b"}

    h1 = MagicMock(id="a", path="/a.jpg", score=0.9, payload={})
    h2 = MagicMock(id="b", path="/b.jpg", score=0.85, payload={})

    results = await results_from_hits(db, cfg=cfg, hits=[h1, h2])
    assert results[0].is_favorite is True
    assert results[1].is_disliked is True


@pytest.mark.asyncio
async def test_results_from_hits_score_str_is_3dp():
    from search._indexed_helpers import results_from_hits
    cfg = _fake_cfg()
    db = MagicMock()
    h = MagicMock(id="a", path="/a.jpg", score=0.87654, payload={})
    results = await results_from_hits(
        db, cfg=cfg, hits=[h], favorite_ids=set(), dislike_ids=set(),
    )
    assert results[0].score_str == "0.877"


# --- resolve_filename_filter ---

def _fake_qdrant(total: int = 1000, matches: int = 0):
    """A Qdrant stand-in for the native filename filter.

    The filter is evaluated server-side against the native text index,
    so the resolver asks Qdrant for a count rather than asking the
    SQLite FTS5 index for a list of ids.
    """
    qdrant = MagicMock()
    qdrant.count_points.return_value = total
    qdrant.count_filtered.return_value = matches
    return qdrant


@pytest.mark.asyncio
async def test_resolve_filename_filter_empty_pattern_returns_none_none():
    from search._indexed_helpers import resolve_filename_filter
    qdrant = _fake_qdrant()
    cfg = _fake_cfg()
    assert await resolve_filename_filter(qdrant, cfg=cfg, pattern="") == (None, None)
    assert await resolve_filename_filter(qdrant, cfg=cfg, pattern="   ") == (None, None)
    qdrant.count_filtered.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_filename_filter_zero_matches_keeps_the_filter():
    """A pattern that matches nothing is a *narrow* pattern.

    The filter must still be applied — it is what makes the search
    return no results. The old id-list path needed a separate
    zero-match branch because an empty `HasIdCondition` is not the
    same as no filter; the native filter has no such distinction.
    """
    from search._indexed_helpers import resolve_filename_filter
    qdrant = _fake_qdrant(total=1000, matches=0)
    cfg = _fake_cfg()
    cfg.filename_cardinality_guard = 0.5
    condition, err = await resolve_filename_filter(qdrant, cfg=cfg, pattern="xyz")
    assert condition is not None
    assert err is None


@pytest.mark.asyncio
async def test_resolve_filename_filter_high_coverage_skips_filter():
    """Cardinality guard: pattern matches >50% of the corpus → no filter."""
    from search._indexed_helpers import resolve_filename_filter
    qdrant = _fake_qdrant(total=10, matches=6)
    cfg = _fake_cfg()
    cfg.filename_cardinality_guard = 0.5
    condition, err = await resolve_filename_filter(qdrant, cfg=cfg, pattern="common")
    assert condition is None
    assert err is None


@pytest.mark.asyncio
async def test_resolve_filename_filter_low_coverage_returns_a_condition():
    from search._indexed_helpers import resolve_filename_filter
    qdrant = _fake_qdrant(total=1000, matches=2)
    cfg = _fake_cfg()
    cfg.filename_cardinality_guard = 0.5
    condition, err = await resolve_filename_filter(qdrant, cfg=cfg, pattern="specific")
    assert condition is not None
    assert err is None


@pytest.mark.asyncio
async def test_resolve_filename_filter_invalid_pattern_returns_bad_request():
    from search._indexed_helpers import resolve_filename_filter
    qdrant = _fake_qdrant()
    cfg = _fake_cfg()
    condition, err = await resolve_filename_filter(qdrant, cfg=cfg, pattern="*garbage*")
    assert condition is None
    assert err == "bad_request"


@pytest.mark.asyncio
async def test_resolve_filename_filter_empty_pattern_never_counts():
    """The empty pattern short-circuits before any count is asked for."""
    from search._indexed_helpers import resolve_filename_filter
    qdrant = _fake_qdrant()
    cfg = _fake_cfg()
    condition, err = await resolve_filename_filter(qdrant, cfg=cfg, pattern="")
    assert condition is None
    assert err is None
    qdrant.count_points.assert_not_called()
    qdrant.count_filtered.assert_not_called()


def test_surprise_search_returns_shuffled_subset():
    """surprise_search shuffles input order and returns up to k items."""
    from unittest.mock import MagicMock

    from search._indexed_helpers import surprise_search

    hits = [MagicMock(id=f"id_{i}") for i in range(20)]
    result = surprise_search(hits, k=5)
    assert len(result) == 5
    # All returned hits were in the input.
    assert all(h in hits for h in result)


def test_surprise_search_handles_k_larger_than_input():
    from unittest.mock import MagicMock

    from search._indexed_helpers import surprise_search

    hits = [MagicMock(id="a"), MagicMock(id="b")]
    result = surprise_search(hits, k=10)
    assert len(result) == 2


def test_surprise_search_handles_empty_input():
    from search._indexed_helpers import surprise_search
    assert surprise_search([], k=5) == []


def test_search_query_string_includes_q_positives_negatives():
    from search._indexed_helpers import search_query_string
    out = search_query_string(
        q="kittens",
        positives=["cute", "playful"],
        negatives=["blur"],
        collections=[],
    )
    assert "q=kittens" in out
    assert "positives=cute" in out
    assert "positives=playful" in out
    assert "negatives=blur" in out


def test_search_query_string_omits_default_view():
    """When view is the default, it's omitted from canonical URLs."""
    from search._indexed_helpers import search_query_string
    out = search_query_string(q="x", positives=[], negatives=[], collections=[])
    assert "view=" not in out


def test_search_query_string_includes_non_default_view():
    from search._indexed_helpers import search_query_string
    out = search_query_string(
        q="x", positives=[], negatives=[], collections=[], view="feed",
    )
    assert "view=feed" in out


def test_search_query_string_centroids_list_precedence():
    """When centroids= is supplied, single centroid= is ignored."""
    from search._indexed_helpers import search_query_string
    out = search_query_string(
        q="x", positives=[], negatives=[], collections=[],
        centroid="legacy", centroids=["a", "b"],
    )
    assert "centroid=a" in out
    assert "centroid=b" in out
    assert "centroid=legacy" not in out


def test_search_query_string_favorites_flag():
    from search._indexed_helpers import search_query_string
    out = search_query_string(
        q="x", positives=[], negatives=[], collections=[], favorites=True,
    )
    assert "favorites=true" in out


def _fake_cfg(max_prompt_chars: int = 80, max_prompts_total: int = 20):
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.max_prompt_chars = max_prompt_chars
    cfg.max_prompts_total = max_prompts_total
    return cfg


def test_normalize_prompt_state_trims_and_dedupes_positives():
    from search._indexed_helpers import normalize_prompt_state
    cfg = _fake_cfg()
    out = normalize_prompt_state(
        cfg, q="", positives_raw=["kittens", "  kittens  ", "puppies"],
        negatives_raw=[],
    )
    # Dedup is case-insensitive and post-strip.
    assert out.positives == ["kittens", "puppies"]


def test_normalize_prompt_state_drops_overlong_prompts():
    from search._indexed_helpers import normalize_prompt_state
    cfg = _fake_cfg(max_prompt_chars=10)
    out = normalize_prompt_state(
        cfg, q="", positives_raw=["short", "this prompt is way too long"],
        negatives_raw=[],
    )
    assert out.positives == ["short"]


def test_normalize_prompt_state_appends_q_to_positives():
    """q is appended to positives if usable and non-duplicate."""
    from search._indexed_helpers import normalize_prompt_state
    cfg = _fake_cfg()
    out = normalize_prompt_state(
        cfg, q="ducks", positives_raw=["kittens"], negatives_raw=[],
    )
    assert "ducks" in out.positives
    assert out.q == "ducks"


def test_normalize_prompt_state_q_not_duplicated():
    """q must not be added if it's already in positives (case-insensitive)."""
    from search._indexed_helpers import normalize_prompt_state
    cfg = _fake_cfg()
    out = normalize_prompt_state(
        cfg, q="kittens", positives_raw=["kittens"], negatives_raw=[],
    )
    # q should still be set (echo), but positives shouldn't double up.
    assert out.positives.count("kittens") == 1


def test_diversity_page_returns_empty_when_no_favorites_match():
    """When favorites= is set but no candidates match, return empty page."""
    from unittest.mock import MagicMock

    from search._indexed_helpers import diversity_page

    cfg = MagicMock()
    cfg.qdrant_collection = "test_collection"
    cfg.max_results_total = 1000
    cfg.diversity_duplicate_hamming_distance = 8
    cfg.diversity_relevance_drop = 0.1
    cfg.diversity_pool_depths = {}
    qdrant = MagicMock()
    cache = MagicMock()
    cache.get.return_value = None  # miss

    # favorite_ids set but no overlap with allowed_ids
    hits, has_more, meta = diversity_page(
        cfg, qdrant, cache,
        vector=[0.1] * 4,
        effective_limit=20, offset=0,
        collections=[],
        allowed_ids=["a", "b"],
        favorite_ids={"x", "y"},  # none of a/b
        mode="balanced", strength=0.5,
        depth="auto", pool_depth=0,
    )
    # Favorites filter reduces to empty list (no matches in allowed_ids).
    # The route returns empty + applied metadata without hitting qdrant.
    assert hits == []


def test_diversity_page_uses_payload_first_reordering():
    """Verify the payload-first reordering: search payload-only first,
    collapse on the payload, then retrieve_batch_with_vectors for survivors.

    Pins the call sequence so a future refactor that re-introduces the
    fetch-with-vectors path (which transfers ~40% of vectors we discard)
    fails loudly.
    """
    from unittest.mock import MagicMock

    from search._indexed_helpers import diversity_page
    from search.models import DiversityMetadata

    cfg = MagicMock()
    cfg.qdrant_collection = "test_collection"
    cfg.max_results_total = 1000
    cfg.diversity_duplicate_hamming_distance = 8
    cfg.diversity_relevance_drop = 0.1
    cfg.diversity_pool_depths = {}

    # payload-only hits: 3 candidates with distinct content_sha256 AND
    # dhash values that are far apart (>8 bits) so the collapse keeps all
    # three. (All-zero dhash is treated as a missing hash and skipped —
    # use distinctive values here.)
    hit_a = MagicMock(id="a", score=0.9, payload={"dhash": "0f0f0f0f0f0f0f0f",
                                                   "content_sha256": "sha-a"})
    hit_b = MagicMock(id="b", score=0.7, payload={"dhash": "f0f0f0f0f0f0f0f0",
                                                   "content_sha256": "sha-b"})
    hit_c = MagicMock(id="c", score=0.5, payload={"dhash": "aaaaaaaaaaaaaaaa",
                                                   "content_sha256": "sha-c"})
    payload_hits = [hit_a, hit_b, hit_c]

    # retrieve_batch_with_vectors returns pairs in Qdrant-order (not request
    # order) — mix them up to verify re-attachment by id.
    retrieved_pairs = [
        ("b", [0.1] * 4),
        ("a", [0.2] * 4),
        ("c", [0.3] * 4),
    ]

    qdrant = MagicMock()
    qdrant.search.return_value = (payload_hits, False)
    qdrant.retrieve_batch_with_vectors.return_value = retrieved_pairs

    cache = MagicMock()
    cache.get.return_value = None  # miss

    hits, has_more, meta = diversity_page(
        cfg, qdrant, cache,
        vector=[0.5] * 4,
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None, favorite_ids=None,
        mode="balanced", strength=0.5,
        depth="auto", pool_depth=3,
    )

    # 1. Payload-only search called first (with_vectors=False).
    method_names = [c[0] for c in qdrant.method_calls]
    assert "search" in method_names, \
        f"payload-only search should be called, got: {method_names}"
    assert "search_with_vectors" not in method_names, \
        "search_with_vectors must NOT be called — payload-first reordering"

    # 2. retrieve_batch_with_vectors called with the survivor ids (all 3,
    #    since the payload-hits have distinct content_sha256).
    assert "retrieve_batch_with_vectors" in method_names
    # Find the call args for retrieve_batch_with_vectors
    retrieve_calls = [c for c in qdrant.method_calls
                       if c[0] == "retrieve_batch_with_vectors"]
    assert len(retrieve_calls) == 1, retrieve_calls
    survivor_ids = retrieve_calls[0][1][0]  # positional args
    assert sorted(survivor_ids) == ["a", "b", "c"]

    # 3. Order of operations: search before retrieve.
    search_pos = method_names.index("search")
    retrieve_pos = method_names.index("retrieve_batch_with_vectors")
    assert search_pos < retrieve_pos, \
        "search (payload-only) must run before retrieve_batch_with_vectors"

    # 4. Result has the right shape: hits sliced from ranking, has_more
    #    computed from ranking length, metadata surfaced.
    assert len(hits) == 3
    assert has_more is False
    assert isinstance(meta, DiversityMetadata)


def test_diversity_page_payload_first_survives_orphan_survivor():
    """retrieve_batch_with_vectors can drop ids the server doesn't know
    about. Re-attachment must skip them, not crash.
    """
    from unittest.mock import MagicMock

    from search._indexed_helpers import diversity_page

    cfg = MagicMock()
    cfg.qdrant_collection = "test_collection"
    cfg.max_results_total = 1000
    cfg.diversity_duplicate_hamming_distance = 8
    cfg.diversity_relevance_drop = 0.1
    cfg.diversity_pool_depths = {}

    hit_a = MagicMock(id="a", score=0.9, payload={"dhash": "0f0f0f0f0f0f0f0f",
                                                   "content_sha256": "sha-a"})
    hit_b = MagicMock(id="b", score=0.7, payload={"dhash": "f0f0f0f0f0f0f0f0",
                                                   "content_sha256": "sha-b"})
    qdrant = MagicMock()
    qdrant.search.return_value = ([hit_a, hit_b], False)
    # 'b' was dropped server-side (orphan). rank_diverse gets only 'a'.
    qdrant.retrieve_batch_with_vectors.return_value = [("a", [0.2] * 4)]
    cache = MagicMock(); cache.get.return_value = None

    hits, _, _ = diversity_page(
        cfg, qdrant, cache,
        vector=[0.5] * 4,
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None, favorite_ids=None,
        mode="balanced", strength=0.5,
        depth="auto", pool_depth=2,
    )
    assert len(hits) == 1
    assert hits[0].id == "a"


def test_diversity_page_uses_cache_hit():
    """Cached entries skip Qdrant and return the cached page."""
    from unittest.mock import MagicMock

    from search._indexed_helpers import diversity_page
    from search.models import DiversityMetadata

    cfg = MagicMock()
    cfg.qdrant_collection = "test_collection"
    qdrant = MagicMock()
    cache = MagicMock()
    cached_meta = DiversityMetadata(
        requested=True, applied=True, mode="balanced",
        strength=0.5, depth="auto", pool_depth=100,
        candidate_count=100, result_count=100,
        duplicate_images_collapsed=0,
    )
    cached_hits = [MagicMock(id=f"id_{i}") for i in range(20)]
    cache.get.return_value = MagicMock(
        hits=cached_hits, stats=cached_meta,
    )

    hits, has_more, meta = diversity_page(
        cfg, qdrant, cache,
        vector=[0.1] * 4,
        effective_limit=5, offset=10,
        collections=[], allowed_ids=None, favorite_ids=None,
        mode="balanced", strength=0.5, depth="auto", pool_depth=100,
    )
    # Cache hit returns the cached page, qdrant not called.
    assert len(hits) == 5
    assert qdrant.search_with_vectors.call_count == 0
