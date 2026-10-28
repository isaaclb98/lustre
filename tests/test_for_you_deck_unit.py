"""
tests/test_for_you_deck_unit.py — Unit tests for For You Gumbel deck.

Verifies the cache-key derivation, the Gumbel sampling on a
taste-ranked pool, the cache lifecycle, and validation guards.
Does NOT require Qdrant — the build path is exercised against a
fake qdrant object.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import pytest

from search.for_you_deck import (
    DEFAULT_POOL_SIZE,
    ForYouDeckCache,
    POOL_SIZE_MAX,
    POOL_SIZE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    build_for_you_deck,
    compute_for_you_cache_key,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeHit:
    """Mimics the Qdrant hit shape that `recommend()` returns."""

    id: str
    score: float


class _FakeQdrant:
    """
    Minimal QdrantClient wrapper for unit tests. Records all calls
    so assertions can inspect them, and returns a configurable
    pool of hits ordered by score descending.
    """

    def __init__(
        self,
        pool: list[_FakeHit] | None = None,
        return_empty: bool = False,
    ):
        self._pool: list[_FakeHit] = list(pool or [])
        self.recommend_calls: list[dict[str, Any]] = []
        self.return_empty: bool = return_empty

    def recommend(
        self,
        *,
        positive: list[str],
        negative: list[str],
        limit: int,
        with_payload: bool = True,
    ) -> list[_FakeHit]:
        self.recommend_calls.append(
            {
                "positive": list(positive),
                "negative": list(negative),
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        if self.return_empty:
            return []
        return self._pool[:limit]


def _make_pool(n: int, score_start: float = 0.9, score_step: float = -0.001) -> list[_FakeHit]:
    """Build a taste-ranked pool: top score first, decaying."""
    return [
        _FakeHit(id=f"id-{i:05d}", score=score_start + i * score_step)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# compute_for_you_cache_key
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    def test_same_inputs_same_key(self):
        k1 = compute_for_you_cache_key(
            fav_ids=["a", "b"],
            dis_ids=["c"],
            temperature=0.05,
            pool_size=2000,
            seed=42,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=["a", "b"],
            dis_ids=["c"],
            temperature=0.05,
            pool_size=2000,
            seed=42,
        )
        assert k1 == k2
        assert len(k1) == 16

    def test_different_seed_different_key(self):
        k1 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=[], temperature=0.05, pool_size=2000, seed=1,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=[], temperature=0.05, pool_size=2000, seed=2,
        )
        assert k1 != k2

    def test_different_temperature_different_key(self):
        k1 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=[], temperature=0.0, pool_size=2000, seed=1,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=[], temperature=0.1, pool_size=2000, seed=1,
        )
        assert k1 != k2

    def test_different_pool_size_different_key(self):
        k1 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=[], temperature=0.05, pool_size=1000, seed=1,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=[], temperature=0.05, pool_size=2000, seed=1,
        )
        assert k1 != k2

    def test_different_fav_ids_different_key(self):
        k1 = compute_for_you_cache_key(
            fav_ids=["a"], dis_ids=[], temperature=0.05, pool_size=2000, seed=1,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=["b"], dis_ids=[], temperature=0.05, pool_size=2000, seed=1,
        )
        assert k1 != k2

    def test_different_dis_ids_different_key(self):
        k1 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=["a"], temperature=0.05, pool_size=2000, seed=1,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=[], dis_ids=["b"], temperature=0.05, pool_size=2000, seed=1,
        )
        assert k1 != k2

    def test_field_separator_collision_resistant(self):
        # A comma in one field must not collide with the same
        # comma joining different fields. The unit-separator
        # joiner in compute_for_you_cache_key prevents that.
        k1 = compute_for_you_cache_key(
            fav_ids=["a,b"], dis_ids=[], temperature=0.05, pool_size=2000, seed=1,
        )
        k2 = compute_for_you_cache_key(
            fav_ids=["a"], dis_ids=["b"], temperature=0.05, pool_size=2000, seed=1,
        )
        assert k1 != k2


# ---------------------------------------------------------------------------
# build_for_you_deck
# ---------------------------------------------------------------------------


class TestBuildDeck:
    def test_zero_temperature_returns_top_k_in_score_order(self):
        # At T=0 the Gumbel draw is deterministic and returns
        # the top-k in score order. The pool is sorted
        # descending by score (id-0 highest, id-N-1 lowest), so
        # the first N deck entries are id-0..id-(N-1).
        pool = _make_pool(100)
        qdrant = _FakeQdrant(pool)
        ids, scores = build_for_you_deck(
            fav_ids=["fav1"],
            dis_ids=[],
            qdrant=qdrant,
            pool_size=100,
            deck_size=10,
            temperature=0.0,
            seed=42,
        )
        assert len(ids) == 10
        assert len(scores) == 10
        assert ids[0] == "id-00000"
        assert ids[9] == "id-00009"
        # Scores are non-increasing (deterministic top-K).
        for a, b in pairwise(scores):
            assert a >= b

    def test_nonzero_temperature_does_not_always_return_top_k(self):
        # At T=1.0 the Gumbel noise dominates the score
        # differences (~0.001), so the deck is not the same as
        # the deterministic top-K. We assert at least one
        # non-top element sneaks in across many trials.
        pool = _make_pool(500, score_start=0.9, score_step=-0.001)
        non_top_seen = 0
        for seed in range(20):
            qdrant = _FakeQdrant(pool)
            ids, _ = build_for_you_deck(
                fav_ids=["fav1"],
                dis_ids=[],
                qdrant=qdrant,
                pool_size=500,
                deck_size=50,
                temperature=1.0,
                seed=seed,
            )
            top10 = {f"id-{i:05d}" for i in range(10)}
            non_top_seen += sum(1 for i in ids if i not in top10)
        # Over 20 trials × 50 draws = 1000 picks, the
        # non-top-10 share should be substantial. With T=1.0
        # the noise is much larger than the score spread
        # (0.001) so the deck is close to uniform: 1000 - 200
        # = 800 non-top picks at minimum.
        assert non_top_seen > 700

    def test_same_seed_same_deck(self):
        pool = _make_pool(200)
        ids1, scores1 = build_for_you_deck(
            fav_ids=["fav1"], dis_ids=[],
            qdrant=_FakeQdrant(pool), pool_size=200,
            deck_size=50, temperature=0.5, seed=7,
        )
        ids2, scores2 = build_for_you_deck(
            fav_ids=["fav1"], dis_ids=[],
            qdrant=_FakeQdrant(pool), pool_size=200,
            deck_size=50, temperature=0.5, seed=7,
        )
        assert ids1 == ids2
        assert scores1 == scores2

    def test_different_seed_different_deck_at_higher_t(self):
        # T=0.5 is high enough that the Gumbel noise is
        # comparable to score differences, so two seeds produce
        # different decks.
        pool = _make_pool(200)
        ids1, _ = build_for_you_deck(
            fav_ids=[], dis_ids=[],
            qdrant=_FakeQdrant(pool), pool_size=200,
            deck_size=50, temperature=0.5, seed=1,
        )
        ids2, _ = build_for_you_deck(
            fav_ids=[], dis_ids=[],
            qdrant=_FakeQdrant(pool), pool_size=200,
            deck_size=50, temperature=0.5, seed=2,
        )
        assert ids1 != ids2

    def test_deck_size_capped_to_pool_size(self):
        pool = _make_pool(5)
        ids, scores = build_for_you_deck(
            fav_ids=[], dis_ids=[],
            qdrant=_FakeQdrant(pool), pool_size=5,
            deck_size=100,  # larger than pool
            temperature=0.0, seed=1,
        )
        assert len(ids) == 5
        assert len(scores) == 5

    def test_passes_positive_and_negative_to_recommend(self):
        qdrant = _FakeQdrant(_make_pool(10))
        build_for_you_deck(
            fav_ids=["fav1", "fav2"],
            dis_ids=["dis1"],
            qdrant=qdrant,
            pool_size=10,
            deck_size=5,
            temperature=0.0,
            seed=1,
        )
        assert len(qdrant.recommend_calls) == 1
        call = qdrant.recommend_calls[0]
        assert call["positive"] == ["fav1", "fav2"]
        assert call["negative"] == ["dis1"]
        assert call["limit"] == 10

    def test_empty_recommend_raises(self):
        qdrant = _FakeQdrant(return_empty=True)
        with pytest.raises(RuntimeError, match="no candidates"):
            build_for_you_deck(
                fav_ids=["fav1"], dis_ids=[],
                qdrant=qdrant, pool_size=10,
                deck_size=5, temperature=0.0, seed=1,
            )

    @pytest.mark.parametrize("bad_temp", [-0.1, 1.5, 100.0])
    def test_invalid_temperature_rejected(self, bad_temp):
        qdrant = _FakeQdrant(_make_pool(10))
        with pytest.raises(ValueError, match="temperature"):
            build_for_you_deck(
                fav_ids=[], dis_ids=[],
                qdrant=qdrant, pool_size=10,
                deck_size=5, temperature=bad_temp, seed=1,
            )

    def test_zero_pool_size_rejected(self):
        qdrant = _FakeQdrant(_make_pool(10))
        with pytest.raises(ValueError, match="pool_size"):
            build_for_you_deck(
                fav_ids=[], dis_ids=[],
                qdrant=qdrant, pool_size=0,
                deck_size=5, temperature=0.0, seed=1,
            )

    def test_score_order_preserved_in_returned_pairs(self):
        # scores[i] must correspond to point_ids[i]. The pool
        # is sorted by score descending; after a T=0 draw the
        # first deck entry has the highest score, the second
        # the second-highest, etc.
        pool = _make_pool(20, score_start=0.5, score_step=-0.01)
        ids, scores = build_for_you_deck(
            fav_ids=[], dis_ids=[],
            qdrant=_FakeQdrant(pool), pool_size=20,
            deck_size=10, temperature=0.0, seed=1,
        )
        # T=0 returns deterministic top-K, so scores are
        # non-increasing across the deck.
        for a, b in pairwise(scores):
            assert a >= b


# ---------------------------------------------------------------------------
# ForYouDeckCache
# ---------------------------------------------------------------------------


class TestForYouDeckCache:
    def test_put_then_get_round_trip(self):
        cache = ForYouDeckCache()
        key = cache.put("k1", ["a", "b", "c"], [0.9, 0.8, 0.7])
        result = cache.get(key)
        assert result is not None
        cached_key, entry = result
        assert cached_key == key
        assert entry.point_ids == ("a", "b", "c")
        assert entry.scores == (0.9, 0.8, 0.7)

    def test_get_missing_key_returns_none(self):
        cache = ForYouDeckCache()
        assert cache.get("nope") is None

    def test_get_empty_key_returns_none(self):
        cache = ForYouDeckCache()
        cache.put("k1", ["a"], [0.5])
        assert cache.get("") is None

    def test_put_validates_length_match(self):
        cache = ForYouDeckCache()
        with pytest.raises(ValueError, match="same length"):
            cache.put("k1", ["a", "b"], [0.5])

    def test_clear_drops_all_entries(self):
        cache = ForYouDeckCache()
        cache.put("k1", ["a"], [0.5])
        cache.put("k2", ["b"], [0.6])
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get("k1") is None

    def test_ttl_expiry(self):
        cache = ForYouDeckCache(ttl_s=0.05)  # 50ms
        cache.put("k1", ["a"], [0.5])
        assert cache.get("k1") is not None
        import time as _time

        _time.sleep(0.1)
        assert cache.get("k1") is None


# ---------------------------------------------------------------------------
# Bounds — round-trip with the surprise_deck constants
# ---------------------------------------------------------------------------


class TestBounds:
    def test_temperature_bounds_match_home_page(self):
        # The /for-you "Surprise" slider reuses the home-page
        # bounds, by design. If the home page ever raises
        # TEMPERATURE_MAX, /for-you goes with it.
        assert TEMPERATURE_MIN == 0.0
        assert TEMPERATURE_MAX == 1.0
        assert POOL_SIZE_MIN == 200
        assert POOL_SIZE_MAX == 50_000

    def test_default_pool_size_is_sensible(self):
        # Smaller than the home-page default because the
        # candidate pool is taste-filtered, not text-cosine.
        # A 2_000-point taste pool is plenty for variety.
        assert 500 <= DEFAULT_POOL_SIZE <= 5000
