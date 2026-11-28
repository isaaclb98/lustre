"""
tests/test_centroid_diversity_page.py

Unit tests for `centroid_diversity_page` — the Layer-2-aware diversity
ranking helper behind /api/centroids/{name}/search?diversity=...

The API-level fixture (test_centroids_api.py) uses STATIC centroids, which
have no seed_ids, so the two-layer near-dup exclusion (the whole reason
this helper exists separately from `diversity_page`) never exercises there.
These tests drive it with a mocked Qdrant + real rank_diverse so Layer 1
(exclude_ids passthrough) and Layer 2 (seed near-dup drop) are both pinned.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from search._indexed_helpers import centroid_diversity_page
from lustre_kernel.spaces import TEXT_SPACE


def _cfg():
    cfg = MagicMock()
    cfg.qdrant_collection = "test_collection"
    cfg.max_results_total = 1000
    cfg.diversity_duplicate_hamming_distance = 8
    cfg.diversity_relevance_drop = 0.1
    cfg.diversity_pool_depths = {}
    return cfg


def _hit(hid, score, dhash, sha):
    h = MagicMock()
    h.id = hid
    h.score = score
    h.payload = {"dhash": dhash, "content_sha256": sha}
    return h


def test_no_seeds_skips_layer2():
    """Static centroid (no seed_ids): Layer 2 is skipped, exclude_ids=None,
    and all candidates flow through rank_diverse."""
    cfg = _cfg()
    qa = _hit("a", 0.9, "0f0f0f0f0f0f0f0f", "sha-a")
    qb = _hit("b", 0.7, "f0f0f0f0f0f0f0f0", "sha-b")
    qc = _hit("c", 0.5, "aaaaaaaaaaaaaaaa", "sha-c")
    pairs = [
        (qa, [1.0, 0.0, 0.0, 0.0]),
        (qb, [0.0, 1.0, 0.0, 0.0]),
        (qc, [0.0, 0.0, 1.0, 0.0]),
    ]

    qdrant = MagicMock()
    qdrant.search_with_vectors.return_value = (pairs, True)

    cache = MagicMock()
    cache.get.return_value = None  # miss

    hits, has_more, meta = centroid_diversity_page(
        cfg, qdrant, cache,
        vector=[1.0, 0.0, 0.0, 0.0],
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None,
        seed_ids=[],  # static centroid
        mode="balanced", strength=0.5, depth="auto", pool_depth=3,
    )

    # Layer 1 exclude_ids arg is None when there are no seeds.
    call_args = qdrant.search_with_vectors.call_args
    assert call_args.args[5] is None or call_args.kwargs.get("exclude_ids") is None
    # Layer 2 never runs (no seed vectors fetched).
    qdrant.retrieve_batch_with_vectors.assert_not_called()
    assert meta.applied is True
    assert {h.id for h in hits} == {"a", "b", "c"}
    cache.put.assert_called_once()


def test_seeds_passed_as_layer1_exclude_ids():
    """seed_ids reach Qdrant as the exclude_ids positional (Layer 1)."""
    cfg = _cfg()
    qa = _hit("a", 0.9, "0f0f0f0f0f0f0f0f", "sha-a")
    pairs = [(qa, [1.0, 0.0, 0.0, 0.0])]

    qdrant = MagicMock()
    qdrant.search_with_vectors.return_value = (pairs, False)
    # Layer 2: seed vector far from every candidate → nothing dropped.
    qdrant.retrieve_batch_with_vectors.return_value = [("seed-1", [0.0, 0.0, 0.0, 1.0])]

    cache = MagicMock()
    cache.get.return_value = None

    centroid_diversity_page(
        cfg, qdrant, cache,
        vector=[1.0, 0.0, 0.0, 0.0],
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None,
        seed_ids=["seed-1"],
        mode="balanced", strength=0.5, depth="auto", pool_depth=3,
    )

    args, kwargs = qdrant.search_with_vectors.call_args
    # exclude_ids is the 6th positional in the helper's call.
    exclude = args[5] if len(args) > 5 else kwargs.get("exclude_ids")
    assert exclude == ["seed-1"]
    # Layer 2 fetched the seed vectors.
    # The seed vectors are read in the centroid's own space — the
    # helper defaults to the text space.
    qdrant.retrieve_batch_with_vectors.assert_called_once_with(
        ["seed-1"], TEXT_SPACE,
    )


def test_layer2_drops_seed_near_duplicate():
    """A candidate that is a near-duplicate of a seed vector is dropped
    before ranking (Layer 2). The surviving candidate set excludes it."""
    cfg = _cfg()
    # Two candidates: 'a' is an exact copy of the seed (cos 1.0), 'b' is
    # orthogonal. Layer 2 must drop 'a'.
    seed_vec = [1.0, 0.0, 0.0, 0.0]
    qa = _hit("a", 0.99, "0f0f0f0f0f0f0f0f", "sha-a")
    qb = _hit("b", 0.50, "f0f0f0f0f0f0f0f0", "sha-b")
    pairs = [(qa, list(seed_vec)), (qb, [0.0, 1.0, 0.0, 0.0])]

    qdrant = MagicMock()
    qdrant.search_with_vectors.return_value = (pairs, False)
    qdrant.retrieve_batch_with_vectors.return_value = [("seed-1", list(seed_vec))]

    cache = MagicMock()
    cache.get.return_value = None

    hits, has_more, meta = centroid_diversity_page(
        cfg, qdrant, cache,
        vector=[0.9, 0.1, 0.0, 0.0],
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None,
        seed_ids=["seed-1"],
        mode="balanced", strength=0.5, depth="auto", pool_depth=3,
    )

    kept = {h.id for h in hits}
    assert "a" not in kept, "Layer 2 should have dropped the seed near-dup"
    assert kept == {"b"}


def test_cache_hit_slices_without_reranking():
    """A cache hit returns a slice of the stored ranking and does not
    call Qdrant at all."""
    from search.diversity import DiversityRanking, DiversityStats

    cfg = _cfg()
    stored = [
        _hit("a", 0.9, "0f0f0f0f0f0f0f0f", "sha-a"),
        _hit("b", 0.7, "f0f0f0f0f0f0f0f0", "sha-b"),
        _hit("c", 0.5, "aaaaaaaaaaaaaaaa", "sha-c"),
    ]
    stats = DiversityStats(
        requested=True, applied=True, mode="balanced", strength=0.5,
        candidate_count=3, result_count=3, depth="auto", pool_depth=3,
    )

    qdrant = MagicMock()
    cache = MagicMock()
    cache.get.return_value = DiversityRanking(hits=stored, stats=stats)

    hits, has_more, meta = centroid_diversity_page(
        cfg, qdrant, cache,
        vector=[1.0, 0.0, 0.0, 0.0],
        effective_limit=2, offset=0,
        collections=[], allowed_ids=None,
        seed_ids=["seed-1"],
        mode="balanced", strength=0.5, depth="auto", pool_depth=3,
    )

    qdrant.search_with_vectors.assert_not_called()
    assert [h.id for h in hits] == ["a", "b"]
    assert has_more is True  # 'c' remains at offset 2
    assert meta.applied is True


def test_empty_pool_returns_empty_applied():
    """No candidates → empty page, applied metadata, cached."""
    cfg = _cfg()
    qdrant = MagicMock()
    qdrant.search_with_vectors.return_value = ([], False)

    cache = MagicMock()
    cache.get.return_value = None

    hits, has_more, meta = centroid_diversity_page(
        cfg, qdrant, cache,
        vector=[1.0, 0.0, 0.0, 0.0],
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None,
        seed_ids=[],
        mode="balanced", strength=0.5, depth="auto", pool_depth=3,
    )

    assert hits == []
    assert has_more is False
    assert meta.applied is True
    cache.put.assert_called_once()


def test_none_cache_still_ranks():
    """diversity_cache=None (router built without it) → ranking still
    works, just uncached."""
    cfg = _cfg()
    qa = _hit("a", 0.9, "0f0f0f0f0f0f0f0f", "sha-a")
    pairs = [(qa, [1.0, 0.0, 0.0, 0.0])]

    qdrant = MagicMock()
    qdrant.search_with_vectors.return_value = (pairs, False)

    hits, has_more, meta = centroid_diversity_page(
        cfg, qdrant, None,  # no cache
        vector=[1.0, 0.0, 0.0, 0.0],
        effective_limit=10, offset=0,
        collections=[], allowed_ids=None,
        seed_ids=[],
        mode="balanced", strength=0.5, depth="auto", pool_depth=3,
    )

    assert [h.id for h in hits] == ["a"]
    assert meta.applied is True
