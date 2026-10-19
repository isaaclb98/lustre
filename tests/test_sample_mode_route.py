"""
tests/test_sample_mode_route.py — integration coverage for
?mode=sample on /api/centroids/{name}/search.

The static-centroid fixture in test_centroids_api.py doesn't
exercise the sample path (a .pt file has no source population,
and the route 400s on that case by design). These tests use a
tiny in-memory dynamic centroid registered against the
in-memory Qdrant, so the full K-of-N re-pick + Qdrant
retrieve_batch_with_vectors + sample_centroid() pipeline runs.

Coverage:
  - ?mode=centroid (default) returns 200 with results
  - ?mode=sample returns 200 with a (likely different) results list
  - ?mode=sample against a static .pt centroid 400s with a
    clear error
  - ?mode=garbage 400s
  - sample_k=0 400s
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _centroid_fixture import (
    CENTROID_CAR_ID,
    CENTROID_CAT_ID,
    CENTROID_DOG_ID,
    WUXIA_CENTROID,
    save_centroid,
)
from lustre_kernel.registry import MockEmbedder
from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

# Three sample-ids large enough that the default K=10 actually
# subsamples. We mock the dynamic centroid's compute_fn to return
# these ids; the route re-fetches their vectors from Qdrant
# during the sample path.
_DYNAMIC_NAME = "test-likes"
_mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text


@pytest.fixture
def app_with_dynamic(qdrant_in_memory, nas_base, tmp_path):
    """
    Same Qdrant as the static fixture (3 points: cat/dog/car),
    plus a dynamic centroid registered under "test-likes" with
    those same 3 ids as the seed set. (3 is below the default
    K=10, which exercises the `n <= k` fallback in
    sample_centroid — every "sample" call returns the full set.)
    """
    centroids_dir = tmp_path / "centroids"
    centroids_dir.mkdir()
    save_centroid(centroids_dir / "siglip2_2026-06-17_wuxia.pt", WUXIA_CENTROID)

    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection=qdrant_in_memory.collection,
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=35,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base=str(nas_base),
        path_prefix="",
        web_ui_url="http://localhost:8000",
        log_level="WARNING",
        test_mode=True,
        centroids_dir=str(centroids_dir),
        centroid_expected_model="siglip2",
        centroid_expected_feature_dim=VECTOR_DIM,
    )

    client = qdrant_in_memory.client
    upsert.ensure_collection(
        client, qdrant_in_memory.collection, dim=VECTOR_DIM
    )
    items = [
        (CENTROID_CAT_ID, _mock_embed("cat"),
         {"path": "cat.jpg", "blurhash": "", "width": 16, "height": 16}),
        (CENTROID_DOG_ID, _mock_embed("dog"),
         {"path": "dog.jpg", "blurhash": "", "width": 16, "height": 16}),
        (CENTROID_CAR_ID, _mock_embed("car"),
         {"path": "car.jpg", "blurhash": "", "width": 16, "height": 16}),
    ]
    upsert.upsert_batch(client, qdrant_in_memory.collection, items, wait=True)

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    # Register a dynamic centroid on the same app instance. The
    # registry is exposed via app_mod._dynamic_centroids after
    # create_app runs; we poke it directly to inject our seed
    # list. This mirrors how the favourites/likes centroids get
    # registered in production.
    from search.centroids import DynamicCentroidSpec
    registry = getattr(app_mod, "_dynamic_centroids", None)
    assert registry is not None, "create_app should set _dynamic_centroids"

    def compute() -> tuple[list[float], int, list[str]] | None:
        ids = [CENTROID_CAT_ID, CENTROID_DOG_ID, CENTROID_CAR_ID]
        return (
            _mock_embed("cat+dog+car"),
            len(ids),
            ids,
        )

    registry.register(
        DynamicCentroidSpec(
            name=_DYNAMIC_NAME,
            label="Test likes",
            description="in-test dynamic centroid for sample-mode tests",
            compute_fn=compute,
            source="test",
        )
    )

    with TestClient(app) as tc:
        yield tc
    app_mod.reset_for_tests()


def test_centroid_mode_default_returns_200(app_with_dynamic):
    resp = app_with_dynamic.get(
        f"/api/centroids/{_DYNAMIC_NAME}/search?limit=5"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    # The dynamic centroid has 3 seeds and a tiny Qdrant, so we
    # get at most 3 non-seed results (we exclude the seeds).
    assert isinstance(body["results"], list)


def test_sample_mode_returns_200(app_with_dynamic):
    resp = app_with_dynamic.get(
        f"/api/centroids/{_DYNAMIC_NAME}/search?limit=5&mode=sample"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_sample_mode_explicit_k(app_with_dynamic):
    # sample_k only takes effect when mode=sample. The validator
    # requires 1 <= sample_n <= sample_k, so when overriding k to a
    # value smaller than the default N (3), n must be passed too.
    resp = app_with_dynamic.get(
        f"/api/centroids/{_DYNAMIC_NAME}/search?limit=5&mode=sample&sample_k=2&sample_n=2"
    )
    assert resp.status_code == 200


def test_sample_mode_400s_on_static_centroid(app_with_dynamic):
    # The static fixture saved WUXIA_CENTROID as a .pt file. Sample
    # mode has no source population to draw from, so the route
    # rejects it with 400 + a clear message.
    resp = app_with_dynamic.get(
        f"/api/centroids/{WUXIA_CENTROID}/search?limit=5&mode=sample"
    )
    assert resp.status_code == 400
    body = resp.json()
    # The error envelope used by `bad_request()` exposes `detail`
    # in either the top-level or the JSONResponse body. Match
    # either shape so we don't break if the helper changes.
    detail = body.get("detail") or body.get("error", {}).get("detail", "")
    assert "static centroid" in detail.lower() or "sample" in detail.lower()


def test_unknown_mode_400s(app_with_dynamic):
    resp = app_with_dynamic.get(
        f"/api/centroids/{_DYNAMIC_NAME}/search?limit=5&mode=garbage"
    )
    assert resp.status_code == 400


def test_zero_sample_k_400s(app_with_dynamic):
    resp = app_with_dynamic.get(
        f"/api/centroids/{_DYNAMIC_NAME}/search?limit=5&mode=sample&sample_k=0"
    )
    assert resp.status_code == 400