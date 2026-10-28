"""
search/surprise_deck.py — Surprise Me deck generator + per-query cache.

The Surprise Me feature pre-generates a fixed-size list of photos drawn
from a Gumbel-perturbed softmax over the deep Qdrant ranking. The user
scrolls through their private randomised deck; subsequent pages
reference the cached deck by session id (no Qdrant round-trip).

This module owns:
  - `gumbel_top_k` — pure numpy helper. Takes a score array and a
    sampling temperature, returns a length-k array of indices into
    the original array, sampled without replacement from the
    softmax distribution `p_i ∝ exp(score_i / temperature)`.
  - `SurpriseDeckCache` — process-local map of cache_key → deck.
    Mirrors `RandomSessionStore` in `search/random_session.py`
    (session TTL, in-memory dict, explicit `clear()` for tests).
  - The "cache_key" itself: a sha256 hash over every search param
    that influences the deck (q, positives, negatives, centroids,
    weights, collections, favorites, filename, temperature,
    pool_size, seed). Used as the session id so the ETag and the
    cache lookup are one mechanism.

Gumbel-max sampling: for each candidate draw an independent Gumbel
sample `g = -log(-log(U))` where U is uniform in (0, 1). The
argmax of `score + T * g` is a sample from the softmax
distribution. Iterating with the chosen index masked to -inf gives
k distinct samples without replacement — O(k * n) numpy argmax
calls, ~tens of ms for n=10000, k=2400.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from search.qdrant_client import SearchHit

import numpy as np


def gumbel_top_k(
    scores: Sequence[float],
    k: int,
    temperature: float,
    rng: np.random.Generator,
) -> list[int]:
    """
    Sample k distinct indices from `scores` under a softmax distribution.

    The probability of index i is `exp(scores[i] / T) / Z` where
    Z is the normalising constant. Indices are returned in the
    order they were drawn (not sorted by score). The sample is
    without replacement.

    Args:
        scores: 1-D score array. Does not need to be normalised.
        k: Number of distinct indices to return. Must satisfy
           0 <= k <= len(scores).
        temperature: Gumbel noise scale. T=0 is deterministic
            (always the top-k); T→∞ is uniform random.
        rng: numpy Generator used to draw uniform noise. Pass a
            seeded generator for reproducible decks.

    Returns:
        List of `k` distinct indices into `scores`, in draw order.
    """
    n = len(scores)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}]; got {k}")
    if k == 0:
        return []
    if temperature <= 0:
        # Deterministic: return the top-k by score in draw order.
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k]
        return [int(i) for i in order]
    arr = np.asarray(scores, dtype=np.float64).copy()
    picked: list[int] = []
    gumbel = -np.log(-np.log(rng.uniform(low=1e-12, high=1 - 1e-12, size=n)))
    for _ in range(k):
        perturbed = arr + temperature * gumbel
        idx = int(np.argmax(perturbed))
        picked.append(idx)
        arr[idx] = -np.inf
    return picked


def compute_cache_key(
    *,
    q: str,
    positives: list[str],
    negatives: list[str],
    centroids: list[str],
    weights: list[float] | None,
    collections: list[str],
    favorites: bool,
    filename: str,
    temperature: float,
    pool_size: int,
    seed: int,
) -> str:
    """
    Deterministic 16-char cache key for a Surprise Me deck.

    The key is a sha256 over every search param that influences
    which photos end up in the deck. Two requests with the same
    key get the same deck; the key doubles as the session id
    passed back by the client (so the ETag and the cache lookup
    are the same value — no separate session store needed).

    Fields are joined with `\\x1f` (ASCII unit separator) so a
    value containing `|` or `,` cannot collide with a different
    field's value. See `routers/search.py` ETag block for the
    matching pattern.
    """
    weights_str = ",".join(f"{w:.6f}" for w in (weights or []))
    etag_input = "\x1f".join(
        [
            q,
            ",".join(positives),
            ",".join(negatives),
            ",".join(centroids),
            weights_str,
            ",".join(collections),
            str(favorites),
            filename,
            f"{temperature:.6f}",
            str(pool_size),
            str(seed),
        ]
    ).encode("utf-8")
    return hashlib.sha256(etag_input).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Process-local deck cache.
# ---------------------------------------------------------------------------


DEFAULT_DECK_SIZE = 2400
DEFAULT_TEMPERATURE = 0.05
DEFAULT_POOL_SIZE = 5000
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0
POOL_SIZE_MIN = 200
# Upper bound for the pool-size slider. Beyond ~75k the
# Qdrant server-side HNSW search crosses its 30s timeout
# for a 2M-point corpus (the cost scales with `limit`:
# scoring + sorting + payload transfer for the candidate
# set). 50k keeps the first-page latency under 10s on
# the current corpus; the route still clamps the actual
# pool to the corpus size when the slider value exceeds
# it, so this is a UX ceiling, not a correctness one.
POOL_SIZE_MAX = 50_000

DECK_TTL_S = 30 * 60


@dataclass
class _DeckEntry:
    """One cached Surprise Me deck: list of (point_id, score) pairs + timestamp.

    Scores are the raw cosine similarities from the pool search that
    produced the deck. They survive across page requests via this
    cache so the UI can display them on hover — the hydration path
    (qdrant.retrieve_batch) doesn't have access to the score from
    a query, only the cached value.
    """

    created_at: float
    point_ids: tuple[str, ...]
    scores: tuple[float, ...]


class SurpriseDeckCache:
    """
    Process-local map of cache_key → pre-generated deck.

    Deliberately NOT shared across processes — a deck is cheap
    to rebuild (one Qdrant search + numpy Gumbel draw), and
    cross-process state would mean sticky sessions on a load
    balancer, which is the opposite of what Surprise Me wants.

    Mirrors `RandomSessionStore` in `search/random_session.py`:
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
        similarity of the photo with id `point_ids[i]`."""
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



# ---------------------------------------------------------------------------
# Deck-builder — the async helper that ties Qdrant + Gumbel + cache together.
# Moved from `search/routers/search.py` in round-93 PR 2
# (`refactor/plan-002-module-layout`). Lives here because it composes
# SurpriseDeckCache + gumbel_top_k; the cross-router import
# (`from search.routers.search import _build_surprise_deck`) was a smell.
# ---------------------------------------------------------------------------


async def build_surprise_deck(
    *,
    qdrant: Any,
    vec: list[float],
    collections: list[str],
    surprise_allowed_ids: list[str] | None,
    surprise_pool: int,
    surprise_temperature: float,
    surprise_seed: int,
    surprise_session: str | None,
    surprise_cache_key: str,
    surprise_deck_cache: SurpriseDeckCache,
    effective_limit: int,
    offset: int,
    exclude_ids: list[str] | None = None,
    path_filter: Any | None = None,
) -> tuple[list[SearchHit], bool, list[str]]:
    """
    Build or fetch the Gumbel-sampled Surprise Me deck and return
    the requested page.

    The cache key doubles as the session id: same params → same
    key → same deck. On a cache hit, no Qdrant search and no
    Gumbel draw happen — we just slice the cached deck.

    Returns `(hits_for_this_page, has_more, full_deck_ids)`.
    `full_deck_ids` is the entire deck (not just this page) so
    the caller can compute `session_total` for the response.
    """
    cached = (
        surprise_deck_cache.get(surprise_cache_key)
        if surprise_session and surprise_session == surprise_cache_key
        else None
    )
    if cached is not None:
        _, entry = cached
        deck_ids = list(entry.point_ids)
        deck_scores = list(entry.scores)
    else:
        # with_payload=False: the pool is only a source of ids and
        # scores for the Gumbel draw. The page that is actually
        # returned is hydrated separately (retrieve_batch below), so
        # pulling 20k-50k payloads here is pure waste — it costs
        # time and memory on both sides of the wire for data nobody
        # reads.
        pool_hits, _ = await asyncio.to_thread(
            functools.partial(
                qdrant.search,
                with_payload=False,
            ),
            vec,
            surprise_pool,
            0,
            collections or None,
            surprise_allowed_ids,
            exclude_ids,
            path_filter,
        )
        scores = [h.score for h in pool_hits]
        picked_indices = gumbel_top_k(
            scores,
            k=min(DEFAULT_DECK_SIZE, len(pool_hits)),
            temperature=surprise_temperature,
            rng=np.random.default_rng(surprise_seed),
        )
        deck_ids = [pool_hits[i].id for i in picked_indices]
        # Capture the raw cosine of each picked photo. The hydration
        # path (qdrant.retrieve_batch) doesn't compute scores — it
        # only returns id+payload — so we need to set them here
        # otherwise the UI shows 0.0 on hover.
        deck_scores = [float(scores[i]) for i in picked_indices]
        surprise_deck_cache.put(surprise_cache_key, deck_ids, deck_scores)

    start = offset
    end = start + effective_limit
    page_ids = deck_ids[start:end]
    has_more = end < len(deck_ids)
    if not page_ids:
        return [], has_more, deck_ids
    hydrated = await asyncio.to_thread(qdrant.retrieve_batch, page_ids)
    by_id = {h.id: h for h in hydrated}
    # Use the deck order so a refresh of the same query+seed
    # gives the same page. Override the (zeroed) score with the
    # value captured during the pool search so the UI sees the
    # real cosine similarity.
    page_score_by_id = {
        deck_ids[i]: deck_scores[i]
        for i in range(start, min(end, len(deck_ids)))
    }
    hits: list[SearchHit] = []
    for pid in page_ids:
        h = by_id.get(pid)
        if h is None:
            continue
        h.score = page_score_by_id[pid]
        hits.append(h)
    return hits, has_more, deck_ids


__all__ = [
    "DEFAULT_DECK_SIZE",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_TEMPERATURE",
    "POOL_SIZE_MAX",
    "POOL_SIZE_MIN",
    "TEMPERATURE_MAX",
    "TEMPERATURE_MIN",
    "SurpriseDeckCache",
    "build_surprise_deck",
    "compute_cache_key",
    "gumbel_top_k",
]
