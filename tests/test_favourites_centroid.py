"""Tests for the favourites dynamic centroid + DynamicCentroidRegistry.

Two layers:
- Unit tests for the registry (compute, invalidate, cache).
- End-to-end tests through the FastAPI app (mark favourite,
  /api/centroids list, /api/centroids/favourites/search).
"""
from __future__ import annotations

import math

import pytest

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search.centroids import (
    DynamicCentroidRegistry,
    DynamicCentroidSpec,
)
from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text


def _vec(seed: str) -> list[float]:
    """Return a deterministic unit-length vector for use in tests."""
    e = _mock_embed(seed)
    norm = math.sqrt(sum(x * x for x in e))
    return [x / norm for x in e]


def test_registry_register_and_get_vector():
    reg = DynamicCentroidRegistry()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return ([1.0, 0.0, 0.0], 5, ["seed-a", "seed-b"])

    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x", compute_fn=compute,
    ))
    assert reg.names() == ["t"]
    assert reg.cached_n_images("t") is None
    assert reg.is_empty("t") is False

    vec, n, seed_ids = reg.get_vector("t")
    assert vec == [1.0, 0.0, 0.0]
    assert n == 5
    assert seed_ids == ["seed-a", "seed-b"]
    assert calls["n"] == 1
    assert reg.cached_n_images("t") == 5
    assert reg.is_empty("t") is False

    vec2, n2, seed_ids2 = reg.get_vector("t")
    assert vec2 == vec
    assert n2 == 5
    assert seed_ids2 == ["seed-a", "seed-b"]
    assert calls["n"] == 1


def test_registry_invalidate_triggers_recompute():
    reg = DynamicCentroidRegistry()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return ([float(calls["n"])], 1, [f"seed-{calls['n']}"])

    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x", compute_fn=compute,
    ))
    assert reg.get_vector("t")[0] == [1.0]
    reg.invalidate("t")
    assert reg.get_vector("t")[0] == [2.0]
    assert calls["n"] == 2


def test_registry_invalidate_unknown_name_is_noop():
    reg = DynamicCentroidRegistry()
    reg.invalidate("not-registered")


def test_registry_unregister_removes_spec():
    """unregister drops the spec from _by_name so list() stops returning it."""
    reg = DynamicCentroidRegistry()
    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x",
        compute_fn=lambda: ([1.0, 0.0], 3, ["a", "b", "c"]),
    ))
    assert reg.names() == ["t"]
    # Prime the cache so we can verify it gets dropped too.
    assert reg.get_vector("t") == ([1.0, 0.0], 3, ["a", "b", "c"])
    assert reg.cached_n_images("t") == 3

    reg.unregister("t")

    assert reg.names() == []
    assert reg.get_vector("t") is None
    assert reg.cached_n_images("t") is None
    # And the spec lookup is gone (used by /api/centroids to
    # render the row).
    assert reg.get_spec("t") is None


def test_registry_unregister_unknown_name_is_noop():
    """unregister on an unregistered name doesn't raise. Lets the
    DELETE-album path call it unconditionally."""
    reg = DynamicCentroidRegistry()
    reg.unregister("never-registered")
    assert reg.names() == []


def test_registry_unregister_clears_dirty_flag():
    """After invalidate, the name is in _dirty; unregister must
    clear that too so a subsequent register doesn't see a stale
    dirty marker."""
    reg = DynamicCentroidRegistry()
    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x",
        compute_fn=lambda: ([1.0], 1, ["x"]),
    ))
    reg.invalidate("t")
    reg.unregister("t")
    # Re-register — next get_vector should compute exactly once.
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return ([float(calls["n"])], 1, [f"s-{calls['n']}"])

    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x", compute_fn=compute,
    ))
    assert reg.get_vector("t")[0] == [1.0]
    assert calls["n"] == 1


def test_registry_compute_returns_none_is_empty():
    reg = DynamicCentroidRegistry()
    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x",
        compute_fn=lambda: None,
    ))
    assert reg.get_vector("t") is None
    assert reg.is_empty("t") is True
    assert reg.cached_n_images("t") is None


def test_registry_compute_exception_is_swallowed():
    reg = DynamicCentroidRegistry()
    state = {"should_raise": True}

    def compute():
        if state["should_raise"]:
            raise RuntimeError("boom")
        return ([1.0], 1, ["seed-1"])

    reg.register(DynamicCentroidSpec(
        name="t", label="t", description="", source="x", compute_fn=compute,
    ))
    assert reg.get_vector("t") is None

    state["should_raise"] = False
    reg.invalidate("t")
    vec, n, seed_ids = reg.get_vector("t")
    assert vec == [1.0]
    assert n == 1
    assert seed_ids == ["seed-1"]


def test_registry_get_vector_unknown_name_returns_none():
    reg = DynamicCentroidRegistry()
    assert reg.get_vector("nope") is None


def test_registry_list_returns_sorted():
    reg = DynamicCentroidRegistry()
    for name in ["b", "a", "c"]:
        reg.register(DynamicCentroidSpec(
            name=name, label=name, description="", source="x",
            compute_fn=lambda: ([1.0], 1, ["s"]),
        ))
    assert [s.name for s in reg.list_all()] == ["a", "b", "c"]


@pytest.fixture
def fav_app(tmp_path):
    import uuid

    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from search.app import create_app
    from search.config import Config
    from search.qdrant_client import QdrantSearch

    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test_fav_centroid",
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=50,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base=str(tmp_path),
        path_prefix="",
        web_ui_url="http://localhost:8000",
        log_level="WARNING",
        index_db_path=str(tmp_path / "images.db"),
        test_mode=True,
    )

    client = QdrantClient(location=":memory:")
    upsert.ensure_collection(client, cfg.qdrant_collection, dim=VECTOR_DIM)
    seed_ids = {
        "a": str(uuid.uuid4()),
        "b": str(uuid.uuid4()),
        "c": str(uuid.uuid4()),
    }
    # The near-dup filter (Layer 2 in `search_by_centroid`) calibrates
    # its threshold from the seed set's OWN intra-cluster pairwise
    # distance — "drop candidates tighter than the tightest typical
    # seed-seed pair." That calibration assumes the favourites form a
    # tight cluster (real users favourite similar photos); with the
    # raw `_mock_embed` output the three keys "a"/"b"/"c" hash to
    # near-orthogonal vectors and the threshold overshoots, dropping
    # the only candidate left after Layer-1's id-exclude. Build
    # vectors explicitly: a and b are tight neighbours (the
    # favourites — same vibe), c sits clearly outside the cluster so
    # the calibrated threshold correctly keeps it. The point ids
    # themselves are still drawn from `_mock_embed` so unrelated
    # assertions stay stable.
    import numpy as np
    base = np.asarray(_mock_embed("seed"), dtype=np.float32)
    base /= float(np.linalg.norm(base))
    def _close_to(base, jitter: float) -> list[float]:
        noise = np.random.RandomState(0).normal(scale=jitter, size=base.shape).astype(np.float32)
        v = base + noise
        v /= float(np.linalg.norm(v))
        return v.tolist()
    vecs = {
        "a": _close_to(base, jitter=0.05),
        "b": _close_to(base, jitter=0.05),
        "c": _close_to(-base, jitter=0.05),
    }
    items = [
        (seed_ids[k], vecs[k],
         {"id": seed_ids[k], "path": f"/photos/{k}.jpg",
          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"})
        for k in seed_ids
    ]
    upsert.upsert_batch(client, cfg.qdrant_collection, items, wait=True)

    qdrant = QdrantSearch(
        client=client, collection=cfg.qdrant_collection, timeout_ms=2000,
    )

    app = create_app(cfg=cfg, qdrant=qdrant)
    app.state.test_seed_ids = seed_ids
    app.state.qdrant_client = client
    app.state.qdrant_collection = cfg.qdrant_collection
    # Eagerly init the cache from Qdrant so tests don't race with
    # the lifespan handler. TestClient runs lifespan asynchronously;
    # the init_from_qdrant call may not have finished by the time
    # the first test request arrives.
    import search.app as _app_mod
    _app_mod._index_db.init_from_qdrant()
    return TestClient(app)


def test_api_centroids_includes_dynamic_favourites(fav_app):
    resp = fav_app.get("/api/centroids")
    assert resp.status_code == 200
    data = resp.json()
    assert "dynamic_centroids" in data
    fav = next(
        (d for d in data["dynamic_centroids"] if d["name"] == "favourites"),
        None,
    )
    assert fav is not None
    # Round‑29: label kept as "Favourites (legacy alias)" for back-compat
    # with older clients that matched on the exact string.
    assert fav["label"] == "Favourites (legacy alias)"
    assert fav["source"] == "favourites"
    assert fav["n_images"] is None






def test_search_by_favourites_centroid_uses_computed_vector(fav_app):
    """The route uses the dynamic centroid as the query vector AND
    applies the two-layer near-duplicate exclusion: the favourite
    itself (and any vector-near copy of it) must NOT be in the
    results. Pre-fix this test asserted `fav_id in ids`; the
    near-duplicate exclusion is exactly the new behaviour we're
    adding, so the assertion flips. We still want to see distinct
    photos in the result list, so we mark a second photo as a
    favourite too — b/c are the photos the user wants to see
    related to the favourite's vibe.
    """
    fav_id = fav_app.app.state.test_seed_ids["a"]
    other_id = fav_app.app.state.test_seed_ids["b"]
    third_id = fav_app.app.state.test_seed_ids["c"]
    fav_app.post(f"/api/favorites/{fav_id}")
    fav_app.post(f"/api/favorites/{other_id}")
    fav_app.get("/api/centroids")

    resp = fav_app.get("/api/centroids/favourites/search?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == "favourites"
    ids = {r["id"] for r in data["results"]}
    # The favourites themselves are excluded by Layer 1
    # (exact-id `must_not`); a near-dup vector would be excluded by
    # Layer 2. Neither should appear.
    assert fav_id not in ids
    assert other_id not in ids
    # The third photo (not favourited) is a distinct vector and
    # should still come back. If this fails the test is saying the
    # over-fetch dropped everything; the favourite-centroid search
    # would be returning nothing for the user, which is the same
    # failure mode as the old "results echo the inputs" bug.
    assert third_id in ids


def test_search_by_unknown_centroid_returns_404(fav_app):
    resp = fav_app.get("/api/centroids/not-a-real-centroid/search?limit=5")
    assert resp.status_code == 404


def test_mark_favorite_invalidates_centroid_cache(fav_app):
    import search.app as _app_mod

    fav_id = fav_app.app.state.test_seed_ids["a"]
    fav_app.post(f"/api/favorites/{fav_id}")
    fav_app.get("/api/centroids/favourites/search?limit=1")
    assert _app_mod._dynamic_centroids.is_empty("favourites") is False

    fav_app.post(f"/api/favorites/{fav_app.app.state.test_seed_ids['b']}")
    assert _app_mod._dynamic_centroids.cached_n_images("favourites") is None


def test_unmark_favorite_invalidates_centroid_cache(fav_app):
    import search.app as _app_mod

    fav_id = fav_app.app.state.test_seed_ids["a"]
    fav_app.post(f"/api/favorites/{fav_id}")
    fav_app.get("/api/centroids/favourites/search?limit=1")
    assert _app_mod._dynamic_centroids.cached_n_images("favourites") == 1

    fav_app.delete(f"/api/favorites/{fav_id}")
    assert _app_mod._dynamic_centroids.cached_n_images("favourites") is None


def test_search_by_favourites_centroid_when_empty_returns_404(fav_app):
    resp = fav_app.get("/api/centroids/favourites/search?limit=5")
    assert resp.status_code == 404
    assert "no data yet" in resp.json()["detail"]


def test_orphaned_favourite_excluded_from_centroid(fav_app):
    import search.app as _app_mod
    

    fav_id = fav_app.app.state.test_seed_ids["a"]
    fav_app.post(f"/api/favorites/{fav_id}")
    fav_app.get("/api/centroids")
    data = fav_app.get("/api/centroids").json()
    fav = next(d for d in data["dynamic_centroids"] if d["name"] == "favourites")
    assert fav["n_images"] == 1

    qdrant_client = fav_app.app.state.qdrant_client
    collection = fav_app.app.state.qdrant_collection
    from qdrant_client.http import models as qmodels
    qdrant_client.delete(
        collection_name=collection,
        points_selector=qmodels.PointIdsList(points=[fav_id]),
        wait=True,
    )
    _app_mod._invalidate_likes_centroid()
    fav_app.get("/api/centroids")
    data = fav_app.get("/api/centroids").json()
    fav = next(d for d in data["dynamic_centroids"] if d["name"] == "favourites")
    assert fav["n_images"] is None


# ---------------------------------------------------------------------------
# Near-duplicate seed exclusion tests
# ---------------------------------------------------------------------------
#
# Three layers of testing:
#   1. Unit tests for `calibrate_near_dup_threshold` and
#      `filter_near_duplicates` (the pure helpers in centroids.py).
#   2. A compute-returns-seed-ids test on the registry.
#   3. End-to-end test through the route, using deterministic
#      synthetic vectors (not the random-unit-vector mock_embed)
#      so the threshold calibration behaves predictably: a tight
#      seed cluster yields a tight threshold, and a near-duplicate
#      vector is dropped while a distinct one is kept.


def test_compute_returns_seed_ids():
    """DynamicCentroidSpec.compute_fn returns (vector, n, seed_ids).

    The third element feeds the route's near-duplicate exclusion —
    without it, the route would have no way to know which point
    ids were used to build the centroid. This test pins the
    contract that compute fns return the seed ids.
    """
    reg = DynamicCentroidRegistry()
    fav_ids_captured: list[str] = []

    def compute():
        fav_ids_captured.append("a")
        fav_ids_captured.append("b")
        return ([1.0, 0.0], 2, ["a", "b"])

    reg.register(DynamicCentroidSpec(
        name="fav", label="fav", description="", source="x",
        compute_fn=compute,
    ))
    vec, n, seed_ids = reg.get_vector("fav")
    assert vec == [1.0, 0.0]
    assert n == 2
    assert seed_ids == ["a", "b"]


def test_calibrate_threshold_is_constant():
    """Round‑29: the threshold is a fixed 0.02 (near-duplicate on
    SigLIP2), NOT a percentile of the seed set's pairwise
    distances. Pinning this here so the calibration can't
    silently regress to a percentile-based value."""
    import math

    from search.centroids import calibrate_near_dup_threshold

    # Two seeds 0.01 rad apart (cosine distance ~ 5e-5) — tight.
    a = [1.0, 0.0, 0.0]
    b = [math.cos(0.01), math.sin(0.01), 0.0]
    # Two seeds 0.5 rad apart (cosine distance ~ 0.12) — loose.
    c = [1.0, 0.0, 0.0]
    d = [math.cos(0.5), math.sin(0.5), 0.0]

    # Both seed sets return the same constant threshold.
    tight = calibrate_near_dup_threshold([a, b])
    loose = calibrate_near_dup_threshold([c, d])
    assert tight == loose == 0.02


def test_calibrate_threshold_empty_returns_zero():
    """No seeds at all → threshold is 0.0 (the filter short-circuits
    anyway, but we keep the empty-input contract)."""
    from search.centroids import calibrate_near_dup_threshold

    assert calibrate_near_dup_threshold(None) == 0.0
    assert calibrate_near_dup_threshold([]) == 0.0


def test_calibrate_threshold_single_seed_returns_constant():
    """A single seed photo gets the constant 0.02 threshold (was
    0.0 before round‑29; the new contract is constant for any
    non-empty seed set)."""
    from search.centroids import calibrate_near_dup_threshold

    assert calibrate_near_dup_threshold([[1.0, 0.0]]) == 0.02


def test_calibrate_threshold_constant_for_non_unit_inputs():
    """Defensive: non-unit-length inputs are still fine — the
    threshold is a constant, so renormalisation doesn't matter.
    (Previously this test asserted renormalisation was applied
    before computing the percentile; now it's a no-op assertion
    that the constant is independent of input magnitudes.)"""
    import math

    from search.centroids import calibrate_near_dup_threshold
    a_unit = [1.0, 0.0, 0.0]
    b_unit = [math.cos(0.1), math.sin(0.1), 0.0]
    a = [2.0 * x for x in a_unit]
    b = [0.5 * x for x in b_unit]
    expected = calibrate_near_dup_threshold([a_unit, b_unit])
    actual = calibrate_near_dup_threshold([a, b])
    assert actual == expected == 0.02


def test_filter_drops_within_cluster_keeps_outside():
    """A near-duplicate of a seed (vector inside the seed cluster)
    is dropped; a distinct vector outside the cluster is kept."""
    import math

    from search.centroids import (
        calibrate_near_dup_threshold,
        filter_near_duplicates,
    )
    a = [1.0, 0.0, 0.0]
    b = [math.cos(0.01), math.sin(0.01), 0.0]  # 0.01 rad from a
    # near-dup: tiny perturbation of `a` (well within the cluster)
    near_dup = [math.cos(0.001), math.sin(0.001), 0.0]
    # distinct: orthogonal to the cluster
    distinct = [0.0, 0.0, 1.0]

    threshold = calibrate_near_dup_threshold([a, b])
    keep = filter_near_duplicates(
        [near_dup, distinct], [a, b], threshold,
    )
    assert keep == [False, True]


def test_filter_no_seeds_keeps_everything():
    """No seeds → no exclusion. Defensive: the route would short
    out before calling this, but the helper itself should be safe."""
    from search.centroids import filter_near_duplicates

    candidates = [[1.0, 0.0], [0.0, 1.0]]
    assert filter_near_duplicates(candidates, [], 0.0) == [True, True]


def test_filter_dim_mismatch_raises():
    """A dim mismatch between candidates and seeds raises — the
    only failure mode that could silently manifest as "everything
    kept" or "everything dropped" if we let it through."""
    import pytest

    from search.centroids import filter_near_duplicates
    with pytest.raises(ValueError):
        filter_near_duplicates(
            [[1.0, 0.0]], [[0.0, 1.0, 0.0]], 0.0,
        )


def test_search_by_favourites_excludes_near_duplicate(fav_app):
    """End-to-end: a near-duplicate of a favourited photo (same
    direction, different id) must NOT appear in the search results
    — even though Layer 1 only excludes exact ids.

    Uses deterministic synthetic vectors (not the random-unit
    mock_embed) so the threshold calibration settles at a tight
    value matching the seed cluster scale, not the random-vector
    scale.

    Setup:
      - 3 photos forming a tight seed cluster (a, b, c): angles
        0, 0.01, 0.02 rad on the x-y plane.
      - 1 near-dup photo (a_dup): angle 0.0005 rad from `a`
        (well inside the cluster, distinct id).
      - 1 distinct photo (z): along the z-axis, completely
        outside the cluster.
      - Favourite a, b, c.
      - Search and verify: a, b, c excluded (Layer 1); a_dup
        excluded (Layer 2); z included (distinct, kept).
    """
    import math
    import uuid

    from indexer import upsert
    from indexer.upsert import VECTOR_DIM

    cluster_radius = 0.01  # rad — tight cluster
    near_dup_offset = 0.0005  # rad from `a`, well inside cluster
    test_ids = fav_app.app.state.test_seed_ids
    # Wipe the random mock_embed seeds and replace with controlled vectors.
    qdrant_client = fav_app.app.state.qdrant_client
    collection = fav_app.app.state.qdrant_collection
    from qdrant_client.http import models as qmodels
    qdrant_client.delete(
        collection_name=collection,
        points_selector=qmodels.PointIdsList(
            points=list(test_ids.values()),
        ),
        wait=True,
    )

    # Build a unit-length vector in VECTOR_DIM dims that sits at
    # `theta` radians from a unit-length baseline. Energy is in the
    # first two dims; the rest are zero, so the cosine similarity
    # is determined entirely by those two dims (cosine distance is
    # angle-independent of the zero padding).
    def vec_at_angle(theta: float) -> list[float]:
        v = [0.0] * VECTOR_DIM
        v[0] = math.cos(theta)
        v[1] = math.sin(theta)
        return v

    seed_a_id = test_ids["a"]
    seed_b_id = test_ids["b"]
    seed_c_id = test_ids["c"]
    near_dup_id = str(uuid.uuid4())
    distinct_id = str(uuid.uuid4())

    items = [
        (seed_a_id, vec_at_angle(0.0),
         {"id": seed_a_id, "path": "/photos/seed_a.jpg",
          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (seed_b_id, vec_at_angle(cluster_radius),
         {"id": seed_b_id, "path": "/photos/seed_b.jpg",
          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (seed_c_id, vec_at_angle(2 * cluster_radius),
         {"id": seed_c_id, "path": "/photos/seed_c.jpg",
          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (near_dup_id, vec_at_angle(near_dup_offset),
         {"id": near_dup_id, "path": "/photos/near_dup.jpg",
          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
    ]
    # The "distinct" vector is a unit vector along dim 2 — orthogonal
    # to the seed cluster (which lives in dims 0/1), so its cosine
    # distance to every seed is ~1.0 (well outside any threshold the
    # seed cluster would calibrate to).
    distinct_vec = [0.0] * VECTOR_DIM
    distinct_vec[2] = 1.0
    items.append(
        (distinct_id, distinct_vec,
         {"id": distinct_id, "path": "/photos/distinct.jpg",
          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
    )
    upsert.upsert_batch(qdrant_client, collection, items, wait=True)

    # Favourite the three cluster seeds.
    fav_app.post(f"/api/favorites/{seed_a_id}")
    fav_app.post(f"/api/favorites/{seed_b_id}")
    fav_app.post(f"/api/favorites/{seed_c_id}")

    resp = fav_app.get("/api/centroids/favourites/search?limit=5")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    ids = {r["id"] for r in data["results"]}

    # Layer 1 — exact ids excluded at the filter.
    assert seed_a_id not in ids
    assert seed_b_id not in ids
    assert seed_c_id not in ids
    # Layer 2 — the near-duplicate (different id, near-vector) is
    # dropped by the calibrated threshold.
    assert near_dup_id not in ids, (
        "near-duplicate seed should be excluded by Layer 2"
    )
    # The distinct photo (orthogonal to the cluster) is kept —
    # this is the entire point of the feature: distinct photos
    # surface, duplicates don't.
    assert distinct_id in ids, (
        f"distinct photo should be in results; got ids={ids}"
    )

