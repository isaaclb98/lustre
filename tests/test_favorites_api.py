from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

CAT_ID = "11111111-1111-1111-1111-111111111111"
DOG_ID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def app_with_qdrant(qdrant_in_memory, nas_base):
    from PIL import Image

    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

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
    )
    upsert.ensure_collection(qdrant_in_memory.client, qdrant_in_memory.collection, dim=VECTOR_DIM)
    upsert.upsert_batch(
        qdrant_in_memory.client,
        qdrant_in_memory.collection,
        [
            (CAT_ID, _mock_embed("cat"), {"id": CAT_ID, "path": str(nas_base / "cat.jpg"), "mtime": 1, "size": 10, "indexed_at": "2026-01-01T00:00:00Z"}),
            (DOG_ID, _mock_embed("dog"), {"id": DOG_ID, "path": str(nas_base / "dog.jpg"), "mtime": 2, "size": 20, "indexed_at": "2026-01-01T00:00:00Z"}),
        ],
        wait=True,
    )
    Image.new("RGB", (8, 8), (255, 0, 0)).save(nas_base / "cat.jpg")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(nas_base / "dog.jpg")

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        yield client
    app_mod.reset_for_tests()


def test_post_favorite_marks_photo(app_with_qdrant):
    resp = app_with_qdrant.post(f"/api/favorites/{CAT_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == CAT_ID
    assert data["favorited_at"]


def test_post_favorite_missing_id_returns_404(app_with_qdrant):
    resp = app_with_qdrant.post("/api/favorites/missing")
    assert resp.status_code == 404


def test_delete_favorite_unmarks(app_with_qdrant):
    assert app_with_qdrant.post(f"/api/favorites/{CAT_ID}").status_code == 200
    resp = app_with_qdrant.delete(f"/api/favorites/{CAT_ID}")
    assert resp.status_code == 204
    assert app_with_qdrant.get("/api/favorites").json()["total"] == 0


def test_delete_non_favorite_returns_404(app_with_qdrant):
    resp = app_with_qdrant.delete(f"/api/favorites/{CAT_ID}")
    assert resp.status_code == 404


def test_get_favorites_paginates(app_with_qdrant):
    app_with_qdrant.post(f"/api/favorites/{CAT_ID}")
    app_with_qdrant.post(f"/api/favorites/{DOG_ID}")
    resp = app_with_qdrant.get("/api/favorites?limit=1&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["limit"] == 1
    assert len(data["favorites"]) == 1


def test_api_favorites_default_limit_is_35(app_with_qdrant):
    app_with_qdrant.post(f"/api/favorites/{CAT_ID}")
    response = app_with_qdrant.get("/api/favorites")
    assert response.status_code == 200
    assert response.json()["limit"] == 35




def test_search_favorites_filter(app_with_qdrant):
    app_with_qdrant.post(f"/api/favorites/{DOG_ID}")
    resp = app_with_qdrant.get("/api/search?q=cat&favorites=true")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()["results"]]
    assert DOG_ID in ids
    assert CAT_ID not in ids
