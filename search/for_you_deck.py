"""
search/for_you_deck.py — For You Gumbel deck.

Rounds 88: /for-you now uses the same Gumbel-max mechanism as the
home-page Surprise Me mode, but the candidate pool is sourced from
Qdrant recommend(positive=likes, negative=dislikes) rather than
cosine(text_query, photo). The result: at low temperature the user
sees a curated walk through the top of their taste neighborhood; at
high temperature the deck approaches a uniform random sample. The
temperature slider lets the user pick their position on that
spectrum.

The previous implementation uniformly shuffled the top 1% of library
by taste proximity, then returned 30 at a time. The uniform
shuffle destroyed the ranking signal — the user could not tell
the difference from /random. Gumbel preserves the ranking as a
controllable knob.

This module owns:
  - `compute_for_you_cache_key` — sha256 over
    (fav_ids, dis_ids, temperature, pool_size, seed). Doubles as
    the session id.
  - `build_for_you_deck` — calls qdrant.recommend(), runs
    gumbel_top_k, returns (point_ids, scores) in deck order.
  - `ForYouDeckCache` — process-local map of cache_key → deck.
    Same shape as `SurpriseDeckCache` but a separate instance so
    the home-page cache and the for-you cache don't share TTLs.
  - Re-uses `gumbel_top_k` from `search.surprise_deck` (the math
    is identical; only the pool source differs).

Validation guards:
  - temperature must be in [TEMPERATURE_MIN, TEMPERATURE_MAX].
  - pool_size must be in [POOL_SIZE_MIN, POOL_SIZE_MAX]. The upper
    bound matches the home-page slider cap — the Qdrant 30s
    server-side timeout fires for larger pools on a 2M corpus.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from search.surprise_deck import (
    DECK_TTL_S,
    DEFAULT_DECK_SIZE,
    gumbel_top_k,
)
from search.surprise_deck import (
    POOL_SIZE_MAX as _HOMEPAGE_POOL_MAX,
)
from search.surprise_deck import (
    POOL_SIZE_MIN as _HOMEPAGE_POOL_MIN,
)
from search.surprise_deck import (
    TEMPERATURE_MAX as _HOMEPAGE_TEMPERATURE_MAX,
)
from search.surprise_deck import (
    TEMPERATURE_MIN as _HOMEPAGE_TEMPERATURE_MIN,
)

__all__ = [
    "DEFAULT_DECK_SIZE",
    "DEFAULT_POOL_SIZE",
    "POOL_SIZE_MAX",
    "POOL_SIZE_MIN",
    "TEMPERATURE_MAX",
    "TEMPERATURE_MIN",
    "ForYouDeckCache",
    "build_for_you_deck",
    "compute_for_you_cache_key",
]

# Round-88: the temperature / pool bounds mirror the home-page
# Surprise Me sliders. The user controls a "Surprise" knob on
# /for-you that maps to the same Gumbel-max mechanism, so the
# bounds are intentionally identical to keep the UI consistent.
TEMPERATURE_MIN = _HOMEPAGE_TEMPERATURE_MIN  # 0.0
TEMPERATURE_MAX = _HOMEPAGE_TEMPERATURE_MAX  # 1.0
POOL_SIZE_MIN = _HOMEPAGE_POOL_MIN  # 200
POOL_SIZE_MAX = _HOMEPAGE_POOL_MAX  # 50_000 (Qdrant HNSW limit)

# Default pool size for /for-you. Smaller than the home-page default
# (5_000) because the candidate pool is taste-filtered by Qdrant
# recommend() rather than text-cosine — a 2_000-point taste pool
# already covers the "your taste neighborhood" thoroughly.
DEFAULT_POOL_SIZE = 2_000


def compute_for_you_cache_key(
    *,
    fav_ids: list[str],
    dis_ids: list[str],
    temperature: float,
    pool_size: int,
    seed: int,
) -> str:
    """
    Deterministic 16-char cache key for a For You deck.

    Two requests with the same key get the same deck. The key
    doubles as the session id passed back by the client. Fields
    are joined with `\\x1f` (ASCII unit separator) so a value
    containing `|` or `,` cannot collide with a different
    field's value.

    Args:
        fav_ids: Liked-photo ids (positive examples for Qdrant
            recommend). Order matters (it does not change the
            taste vector since recommend averages them, but the
            cache key needs order to be deterministic).
        dis_ids: Disliked-photo ids (negative examples).
        temperature: Gumbel temperature.
        pool_size: Number of recommend() candidates before
            Gumbel sampling.
        seed: Caller-supplied random seed. Pass a fresh random
            value on every page reload for a fresh deck; reuse
            the same value for paginated scroll within a single
            page-mount.
    """
    etag_input = "\x1f".join(
        [
            ",".join(fav_ids),
            ",".join(dis_ids),
            f"{temperature:.6f}",
            str(pool_size),
            str(seed),
        ]
    ).encode("utf-8")
    return hashlib.sha256(etag_input).hexdigest()[:16]


def build_for_you_deck(
    *,
    fav_ids: list[str],
    dis_ids: list[str],
    qdrant: Any,
    pool_size: int,
    deck_size: int,
    temperature: float,
    seed: int,
) -> tuple[list[str], list[float]]:
    """
    Materialise a fresh Gumbel deck for the /for-you feed.

    Calls `qdrant.recommend(positive=fav_ids, negative=dis_ids,
    limit=pool_size)` to get the top `pool_size` candidates by
    taste proximity, then runs Gumbel-max to draw `deck_size`
    distinct indices. The candidate pool is the full set of
    recommend() results in score order; the deck is a re-ordering
    biased toward higher scores by the temperature.

    Args:
        fav_ids: Liked photo ids. Empty list is allowed; the
            caller is expected to have already short-circuited
            the cold-start case (zero-vector fallback) before
            reaching this function.
        dis_ids: Disliked photo ids.
        qdrant: A QdrantClient wrapper with a `recommend()` method.
        pool_size: Number of recommend() candidates. Validated
            by the caller.
        deck_size: Final deck length. Capped to `pool_size` if
            larger.
        temperature: Gumbel temperature in
            [TEMPERATURE_MIN, TEMPERATURE_MAX]. T=0 returns the
            top `deck_size` candidates in score order; higher T
            mixes in more randomness.
        seed: Random seed for the Gumbel draw. Pass an int;
            gets converted to a numpy Generator.

    Returns:
        (point_ids, scores) — both lists of length `deck_size`.
        `scores[i]` is the cosine similarity to the taste
        vector for the photo with id `point_ids[i]`. The
        ordering is the Gumbel draw order, not score order.

    Raises:
        RuntimeError: if `recommend()` returns zero candidates.
            The caller is expected to short-circuit on an empty
            pool and return a friendly response.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1; got {pool_size}")
    if not (TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX):
        raise ValueError(
            f"temperature must be in [{TEMPERATURE_MIN}, "
            f"{TEMPERATURE_MAX}]; got {temperature}"
        )

    # with_payload=False: the pool is a source of ids and scores for
    # the Gumbel draw. The page that is actually returned is
    # hydrated separately, so pulling 20k-50k payloads here is waste.
    hits = qdrant.recommend(
        positive=list(fav_ids),
        negative=list(dis_ids),
        limit=pool_size,
        with_payload=False,
    )
    if not hits:
        raise RuntimeError("recommend returned no candidates")

    # Take scores in the order recommend() returned them. Hits
    # come back sorted by score descending, so the index→score
    # mapping matches the Gumbel draw semantics — the top of
    # the pool has the highest score and is most likely to be
    # drawn at low T.
    scores: list[float] = [float(h.score or 0.0) for h in hits]
    ids: list[str] = [h.id for h in hits]

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    k = min(deck_size, len(scores))
    picked_indices = gumbel_top_k(scores, k, temperature, rng)

    deck_ids = [ids[i] for i in picked_indices]
    deck_scores = [scores[i] for i in picked_indices]
    return deck_ids, deck_scores


# ---------------------------------------------------------------------------
# Process-local deck cache.
# ---------------------------------------------------------------------------


@dataclass
class _DeckEntry:
    """One cached For You deck: list of (point_id, score) pairs + timestamp.

    Scores are the raw cosine similarities from the recommend()
    call that produced the deck. They survive across page
    requests via this cache so the UI can display them on hover —
    the hydration path (qdrant.retrieve_batch) doesn't have
    access to the score from a query, only the cached value.
    """

    created_at: float
    point_ids: tuple[str, ...]
    scores: tuple[float, ...]


class ForYouDeckCache:
    """
    Process-local map of cache_key → pre-generated deck.

    Deliberately NOT shared across processes — a deck is cheap
    to rebuild (one Qdrant recommend + numpy Gumbel draw), and
    cross-process state would mean sticky sessions on a load
    balancer, which is the opposite of what the user wants.

    Mirrors `SurpriseDeckCache` in `search/surprise_deck.py`:
    same TTL semantics, same in-memory dict, same explicit
    `clear()` for tests. The cache_key (not a token) is used as
    the session id, so the same query+params+seed always hits
    the same cache entry.
    """

    def __init__(self, ttl_s: float = DECK_TTL_S):
        self._decks: dict[str, _DeckEntry] = {}
        self._ttl_s = ttl_s

    def get(self, cache_key: str) -> tuple[str, _DeckEntry] | None:
        """Look up a live deck by cache key. Returns None if the
        key is missing or the deck has expired (entry is removed
        on expiry)."""
        if not cache_key:
            return None
        entry = self._decks.get(cache_key)
        if entry is None:
            return None
        if (time.monotonic() - entry.created_at) >= self._ttl_s:
            del self._decks[cache_key]
            return None
        return cache_key, entry

    def put(
        self,
        cache_key: str,
        point_ids: Iterable[str],
        scores: Iterable[float],
    ) -> str:
        """Materialise a fresh deck under the given key. Returns
        the same key (the session id is the cache key). The
        `point_ids` and `scores` iterables must be the same length
        and in the same order — `scores[i]` is the cosine
        similarity to the taste vector for the photo with id
        `point_ids[i]`."""
        ids = tuple(point_ids)
        scs = tuple(scores)
        if len(ids) != len(scs):
            raise ValueError(
                f"point_ids ({len(ids)}) and scores ({len(scs)}) "
                "must be the same length"
            )
        self._decks[cache_key] = _DeckEntry(
            created_at=time.monotonic(),
            point_ids=ids,
            scores=scs,
        )
        return cache_key

    def clear(self) -> None:
        """Drop all decks. Used by tests."""
        self._decks.clear()

    def __len__(self) -> int:
        return len(self._decks)
