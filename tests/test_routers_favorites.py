"""
tests/test_routers_favorites.py — favourites router contract (§B2 step 3).

Pins the favourites router module's contract: the factory function
returns an APIRouter with the three documented endpoints. Tests
run against a MagicMock-backed IndexDB without the full
create_app(); the integration is verified by the existing
test_favorites_api.py suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def fake_index_db():
    db = MagicMock()
    db.mark_favorite.return_value = None
    db.unmark_favorite.return_value = None
    db.get_by_id.return_value = {
        "id": "abc", "path": "/p.jpg", "favorited_at": "2026-08-22T00:00:00Z",
        "is_favorite": 1, "width": 800, "height": 600,
    }
    db.list_favorites.return_value = [
        {"id": "abc", "path": "/p.jpg", "favorited_at": "2026-08-22T00:00:00Z",
         "is_favorite": 1, "width": 800, "height": 600},
    ]
    db.count_favorites.return_value = 1
    return db


@pytest.fixture
def fake_cfg():
    c = MagicMock()
    c.top_k_default = 35
    c.default_view = "grid"
    c.web_ui_url = "http://localhost:5173"
    return c


@pytest.fixture
def invalidate():
    return MagicMock()


def test_build_router_returns_api_router(fake_index_db, fake_cfg, invalidate):
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    assert isinstance(router, APIRouter)


def test_router_routes_registered(fake_index_db, fake_cfg, invalidate):
    """Exactly the three favourites endpoints."""
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    paths = {getattr(r, "path", None) or str(r) for r in router.routes}
    assert "/api/favorites" in paths
    assert "/api/favorites/{point_id}" in paths


def test_mark_favorite_invalidates_caches(fake_index_db, fake_cfg, invalidate):
    """POST /api/favorites/{id} invalidates the centroid exactly once."""
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.post("/api/favorites/abc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "abc"
    assert invalidate.call_count == 1


def test_unmark_favorite_404_when_not_favourited(
    fake_index_db, fake_cfg, invalidate,
):
    """DELETE on a non-favourited photo returns 404."""
    from search.routers.favorites import build_favorites_router

    fake_index_db.get_by_id.return_value = {
        "id": "x", "is_favorite": 0,
    }
    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.delete("/api/favorites/x")
    assert resp.status_code == 404


def test_list_favorites_accepts_large_limits(
    fake_index_db, fake_cfg, invalidate,
):
    """Large limits are accepted — no upper cap.

    Previously this endpoint capped limit at 1000 (returning
    400 above that). With infinite-scroll pagination the
    caller can request any limit; the server clamps offset
    against total in the actual handler, not at the
    validation layer.
    """
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/favorites?limit=99999")
    assert resp.status_code == 200


def test_list_favorites_offset_negative(
    fake_index_db, fake_cfg, invalidate,
):
    """GET /api/favorites?offset=-1 → 400."""
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/favorites?offset=-1")
    assert resp.status_code == 400


def test_list_favorites_default_response_shape(
    fake_index_db, fake_cfg, invalidate,
):
    """GET /api/favorites returns FavoritesListResponse shape."""
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/favorites")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["favorites"][0]["id"] == "abc"


def test_list_favorites_as_results_search_shape(
    fake_index_db, fake_cfg, invalidate,
):
    """`as_results=true` returns SearchResponse-compatible shape."""
    from search.routers.favorites import build_favorites_router

    router = build_favorites_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/favorites?as_results=true")
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert body["results"][0]["is_favorite"] is True