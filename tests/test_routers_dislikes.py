"""
tests/test_routers_dislikes.py — dislikes router contract (§B2 step 4).

Pins the dislikes router module's contract: the factory function
returns an APIRouter with the three documented endpoints. Tests
run against a MagicMock-backed IndexDB without the full
create_app(); the integration is verified by the existing
test_dislikes.py suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def fake_index_db():
    db = MagicMock()
    db.mark_dislike.return_value = None
    db.record_feedback.return_value = None
    db.unmark_dislike.return_value = None
    db.list_dislikes.return_value = [
        {"id": "abc", "path": "/p.jpg", "width": 800, "height": 600,
         "disliked_at": "2026-08-22T00:00:00Z"},
    ]
    db.count_dislikes.return_value = 1
    db.list_favorite_ids.return_value = ["abc"]
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


def _build(fake_index_db, fake_cfg, invalidate):
    from search.routers.dislikes import build_dislikes_router
    router = build_dislikes_router(
        index_db=fake_index_db,
        cfg=fake_cfg,
        invalidate_likes_centroid=invalidate,
    )
    app = FastAPI()
    app.include_router(router)
    return app


def test_mark_dislike_calls_db_and_invalidates(fake_index_db, fake_cfg, invalidate):
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.post("/api/dislikes/abc?source=search")
    assert resp.status_code == 204
    fake_index_db.mark_dislike.assert_called_once_with("abc", "search")
    fake_index_db.record_feedback.assert_called_once_with("abc", "dislike", "search")
    assert invalidate.call_count == 1


def test_mark_dislike_defaults_source_to_manual(fake_index_db, fake_cfg, invalidate):
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.post("/api/dislikes/abc")
    assert resp.status_code == 204
    fake_index_db.mark_dislike.assert_called_once_with("abc", "manual")


def test_unmark_dislike_calls_db_and_invalidates(fake_index_db, fake_cfg, invalidate):
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.delete("/api/dislikes/abc")
    assert resp.status_code == 204
    fake_index_db.unmark_dislike.assert_called_once_with("abc")
    assert invalidate.call_count == 1


def test_list_dislikes_default_shape(fake_index_db, fake_cfg, invalidate):
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.get("/api/dislikes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["limit"] == 35
    assert data["offset"] == 0
    assert data["has_more"] is False
    assert data["items"][0]["id"] == "abc"


def test_list_dislikes_as_results_marks_favorite(fake_index_db, fake_cfg, invalidate):
    """as_results=true must reflect the live favourites set so the
    grid's heart icon stays correct."""
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.get("/api/dislikes?as_results=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] is None
    # abc is in the favourites set returned by fake_index_db; the
    # tile's heart should be filled.
    assert data["results"][0]["is_favorite"] is True
    assert data["results"][0]["id"] == "abc"


def test_list_dislikes_accepts_large_limits(fake_index_db, fake_cfg, invalidate):
    """Large limits are accepted — no upper cap.

    Previously this endpoint capped limit at 1000 (returning
    400 above that). With infinite-scroll pagination the
    caller can request any limit; the server clamps offset
    against total in the actual handler, not at the
    validation layer.
    """
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.get("/api/dislikes?limit=9999")
    assert resp.status_code == 200


def test_list_dislikes_negative_offset_returns_400(fake_index_db, fake_cfg, invalidate):
    app = _build(fake_index_db, fake_cfg, invalidate)
    with TestClient(app) as client:
        resp = client.get("/api/dislikes?offset=-1")
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"
