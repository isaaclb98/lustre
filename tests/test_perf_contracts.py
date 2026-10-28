"""Performance contracts for the surprise deck, the pool count, and the prune.

Each of these pins a specific bad implementation that was measured
and fixed. They are not style checks — each one fails if the
implementation goes back to the slow shape.

  - `gumbel_top_k` was O(k*n): 2,400 iterations over a 50,000-element
    array, ~960 MB of array churn, 118ms on the event loop. The O(n)
    form is 3ms and bit-for-bit identical.
  - `count_points(exact=True)` was 0.277s per surprise request to
    cap the pool size. The approximate count is 0.012s.
"""
from __future__ import annotations

import inspect

import numpy as np


# ---------------------------------------------------------------------------
# the Gumbel draw
# ---------------------------------------------------------------------------


def _reference_gumbel_top_k(scores, k, temperature, rng):
    """The original O(k*n) implementation, kept as the oracle."""
    n = len(scores)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}]; got {k}")
    if k == 0:
        return []
    if temperature <= 0:
        order = np.argsort(
            -np.asarray(scores, dtype=np.float64), kind="stable"
        )[:k]
        return [int(i) for i in order]
    arr = np.asarray(scores, dtype=np.float64).copy()
    picked: list[int] = []
    gumbel = -np.log(
        -np.log(rng.uniform(low=1e-12, high=1 - 1e-12, size=n))
    )
    for _ in range(k):
        perturbed = arr + temperature * gumbel
        idx = int(np.argmax(perturbed))
        picked.append(idx)
        arr[idx] = -np.inf
    return picked


def test_gumbel_matches_the_reference_implementation() -> None:
    """The O(n) draw must return exactly what the O(k*n) one did.

    The Gumbel noise is drawn once and each pick is masked with
    -inf, so the result is "the k largest keys, in descending key
    order" — computable with one argpartition instead of k passes
    over the array. The order matters: it is the deck's display
    order, and the scores decide which photos are in it.
    """
    from search.surprise_deck import gumbel_top_k

    for trial in range(60):
        rng = np.random.default_rng(trial)
        n = int(rng.integers(2, 300))
        k = int(rng.integers(1, n + 1))
        temperature = float(rng.uniform(0.001, 1.0))
        scores = np.random.default_rng(trial + 5).random(n).tolist()
        got = gumbel_top_k(
            scores,
            k=k,
            temperature=temperature,
            rng=np.random.default_rng(trial),
        )
        want = _reference_gumbel_top_k(
            scores,
            k=k,
            temperature=temperature,
            rng=np.random.default_rng(trial),
        )
        assert got == want, f"the draw differs at trial {trial}"


def test_gumbel_does_not_reallocate_per_pick() -> None:
    """The draw must not loop over the pool once per pick.

    The old implementation allocated a fresh n-element array on
    every one of the k picks. This checks the shape of the code
    rather than the timing, so it does not go flaky on a loaded
    machine.
    """
    from search import surprise_deck

    src = inspect.getsource(surprise_deck.gumbel_top_k)
    assert "argpartition" in src, "the O(n) selection is gone"
    assert "for _ in range(k)" not in src, (
        "the draw loops over the pool once per pick again — that is "
        "the O(k*n) implementation"
    )


def test_gumbel_zero_temperature_is_the_top_k() -> None:
    from search.surprise_deck import gumbel_top_k

    scores = [0.1, 0.9, 0.5, 0.7]
    got = gumbel_top_k(
        scores, k=2, temperature=0.0, rng=np.random.default_rng(1)
    )
    assert got == [1, 3]


# ---------------------------------------------------------------------------
# the pool count
# ---------------------------------------------------------------------------


def test_count_points_supports_an_approximate_count() -> None:
    """The count must be able to skip the exact pass.

    The exact count walks every point: 0.277s on the 2.2M-point
    collection. The approximate one reads the segment metadata:
    0.012s. The Surprise Me route only uses the number to cap the
    pool size, where being out by a few hundred points is
    irrelevant.
    """
    from search.qdrant_client import QdrantSearch

    sig = inspect.signature(QdrantSearch.count_points)
    assert "exact" in sig.parameters
    assert sig.parameters["exact"].default is True


def test_the_surprise_route_uses_the_approximate_count() -> None:
    """The route that caps the pool must not pay for an exact count."""
    from search.routers import search as search_router

    src = inspect.getsource(search_router)
    assert "count_points, exact=False" in src, (
        "the surprise route is doing an exact point count on every "
        "request just to cap the pool size"
    )
