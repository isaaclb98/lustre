"""API tests for the saved-searches endpoints.

The endpoints under test are:
  POST   /api/saved-searches             (create)
  GET    /api/saved-searches             (list with limit/offset)
  GET    /api/saved-searches/{id}        (single fetch)
  DELETE /api/saved-searches/{id}        (delete)
  GET    /saved                          (HTML page)

These run against the same in-memory Qdrant + tmp-NAS fixture the
favourites and albums tests use — the saved-searches endpoints
don't touch Qdrant at all, but reusing the fixture gives us a
working IndexDB with the right schema and the right test client.

Saved searches only store prompt shape, so we never need image
fixtures here; a clean app is enough.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

# A single Qdrant collection with one fake photo. The saved-searches
# endpoints don't read Qdrant, but the fixture wires one up so the
# app's startup path runs cleanly.
CAT_ID = "11111111-1111-1111-1111-111111111111"


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
        top_k_default=50,
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
        ],
        wait=True,
    )
    Image.new("RGB", (8, 8), (255, 0, 0)).save(nas_base / "cat.jpg")

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        yield client
    app_mod.reset_for_tests()


# ---------------- create ----------------


def test_create_returns_full_row(app_with_qdrant):
    resp = app_with_qdrant.post(
        "/api/saved-searches",
        json={
            "name": "red-dress-no-manikin",
            "positives": ["red dress", "studio lighting"],
            "negatives": ["manikin"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "red-dress-no-manikin"
    assert data["positives"] == ["red dress", "studio lighting"]
    assert data["negatives"] == ["manikin"]
    assert isinstance(data["id"], int)
    assert data["created_at"]


def test_create_empty_payload_returns_400(app_with_qdrant):
    """A saved search with no prompts is useless — reject up front."""
    resp = app_with_qdrant.post(
        "/api/saved-searches",
        json={"name": "empty", "positives": [], "negatives": []},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_create_empty_name_returns_400(app_with_qdrant):
    resp = app_with_qdrant.post(
        "/api/saved-searches",
        json={"name": "   ", "positives": ["x"], "negatives": []},
    )
    assert resp.status_code == 400


def test_create_long_name_returns_400(app_with_qdrant):
    resp = app_with_qdrant.post(
        "/api/saved-searches",
        json={"name": "x" * 200, "positives": ["x"], "negatives": []},
    )
    assert resp.status_code == 400


def test_create_strips_whitespace_prompts(app_with_qdrant):
    resp = app_with_qdrant.post(
        "/api/saved-searches",
        json={
            "name": "strip-test",
            "positives": ["  red dress  ", ""],
            "negatives": ["\tmanikin\n"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["positives"] == ["red dress"]
    assert data["negatives"] == ["manikin"]


def test_create_duplicate_name_returns_409(app_with_qdrant):
    body = {"name": "dup", "positives": ["a"], "negatives": []}
    assert app_with_qdrant.post("/api/saved-searches", json=body).status_code == 201
    resp = app_with_qdrant.post("/api/saved-searches", json=body)
    assert resp.status_code == 409
    assert resp.json()["code"] == "conflict"


# ---------------- list ----------------


def test_list_returns_empty_array_initially(app_with_qdrant):
    resp = app_with_qdrant.get("/api/saved-searches")
    assert resp.status_code == 200
    data = resp.json()
    assert data["saved_searches"] == []
    assert data["total"] == 0


def test_list_paginates(app_with_qdrant):
    for i in range(5):
        app_with_qdrant.post(
            "/api/saved-searches",
            json={"name": f"s-{i}", "positives": [f"p{i}"], "negatives": []},
        )
    resp = app_with_qdrant.get("/api/saved-searches?limit=2&offset=0")
    data = resp.json()
    assert data["total"] == 5
    assert data["limit"] == 2
    assert data["offset"] == 0
    assert len(data["saved_searches"]) == 2


def test_list_validates_limit_bounds(app_with_qdrant):
    assert app_with_qdrant.get("/api/saved-searches?limit=0").status_code == 400
    assert app_with_qdrant.get("/api/saved-searches?limit=99999").status_code == 400
    assert app_with_qdrant.get("/api/saved-searches?offset=-1").status_code == 400


# ---------------- single fetch ----------------


def test_get_by_id_returns_row(app_with_qdrant):
    created = app_with_qdrant.post(
        "/api/saved-searches",
        json={"name": "solo", "positives": ["x"], "negatives": []},
    ).json()
    resp = app_with_qdrant.get(f"/api/saved-searches/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


def test_get_missing_returns_404(app_with_qdrant):
    assert app_with_qdrant.get("/api/saved-searches/99999").status_code == 404


# ---------------- delete ----------------


def test_delete_returns_204_and_subsequent_get_is_404(app_with_qdrant):
    created = app_with_qdrant.post(
        "/api/saved-searches",
        json={"name": "doomed", "positives": ["x"], "negatives": []},
    ).json()
    resp = app_with_qdrant.delete(f"/api/saved-searches/{created['id']}")
    assert resp.status_code == 204
    assert app_with_qdrant.get(f"/api/saved-searches/{created['id']}").status_code == 404


def test_delete_missing_returns_404(app_with_qdrant):
    assert app_with_qdrant.delete("/api/saved-searches/99999").status_code == 404


# ---------------- HTML page ----------------



