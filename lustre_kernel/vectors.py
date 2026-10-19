"""
lustre_kernel.vectors

Vector arithmetic primitives used by both the search side (centroid
math) and the indexer side (payload validation). No I/O, no model
dependencies, no QdrantClient.

Pure functions only. The kernel invariant: this module never imports
anything except stdlib + (eventually) numpy.
"""

from __future__ import annotations

import math

__all__ = ["l2_normalize", "mean_vector", "vector_cosine"]


def l2_normalize(values: list[float]) -> list[float]:
    """Return a unit-norm copy of `values`.

    A zero vector returns a zero vector (avoids division by zero).
    A non-zero vector is rescaled so `sqrt(sum(v**2)) == 1.0` within
    float epsilon.

    Used by both encoder wrappers (post-embed normalization) and by
    centroid math in `search/centroids.py`.
    """
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return list(values)
    return [v / norm for v in values]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return the per-dimension mean of `vectors`, then L2-normalize.

    An empty input returns an empty list. A single-input returns the
    L2-normalized copy of that input. Non-uniform dimensions raise
    ValueError.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError(f"non-uniform vector dim: expected {dim}, got {len(v)}")
    sums = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    mean = [s / len(vectors) for s in sums]
    return l2_normalize(mean)


def vector_cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-dim vectors.

    Both inputs are assumed L2-normalized; the result is the dot
    product in [-1, 1]. If inputs are not normalized, the result is
    the cosine of the angle but scaled by `||a|| * ||b||`.
    """
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))
