"""
tests/test_routers_system.py — system router contract (§B2 step 1).

Pins the system router module's contract: the factory function
`build_system_router(...)` returns an `APIRouter` that handles
`/healthz` and `/api/cache/status` with the documented response
shapes. The router must be importable and testable without the full
`create_app()` machinery.

When the rest of §B2 lands (search, favorites, albums,
photos, saved-searches, centroids), this test
file is the template: one module per group, one test file per group.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def fake_qdrant():
    q = MagicMock()
    q.healthz.return_value = True
    q.qdrant_point_count = MagicMock(return_value=42)
    return q


@pytest.fixture
def fake_cfg():
    c = MagicMock()
    c.test_mode = False
    c.index_db_refresh_interval_seconds = 300
    c.path_liveness_ttl_seconds = 60
    return c


@pytest.fixture
def fake_index_db():
    db = MagicMock()
    db.qdrant_point_count.return_value = 42
    db.count_images.return_value = 40
    db.last_refresh_time.return_value = 1724280000.0  # 2024-08-22T00:00:00Z as epoch float
    return db


def test_build_system_router_returns_api_router(
    fake_qdrant, fake_cfg, fake_index_db,
):
    """`build_system_router(...)` returns an `APIRouter` instance."""
    from search.routers.system import build_system_router

    router = build_system_router(
        qdrant=fake_qdrant,
        cfg=fake_cfg,
        index_db=fake_index_db,
        path_liveness_cache={},
        path_liveness_cache_max=128,
    )
    assert isinstance(router, APIRouter)


def test_system_router_handles_healthz(fake_qdrant, fake_cfg, fake_index_db):
    """`GET /healthz` returns `{"qdrant": bool, "test_mode": bool}`."""
    from search.routers.system import build_system_router

    router = build_system_router(
        qdrant=fake_qdrant,
        cfg=fake_cfg,
        index_db=fake_index_db,
        path_liveness_cache={},
        path_liveness_cache_max=128,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"qdrant": True, "test_mode": False}


def test_system_router_handles_cache_status(
    fake_qdrant, fake_cfg, fake_index_db,
):
    """`GET /api/cache/status` returns operator-visibility payload."""
    from search.routers.system import build_system_router

    cache: dict = {"/some/path": 1.0}
    router = build_system_router(
        qdrant=fake_qdrant,
        cfg=fake_cfg,
        index_db=fake_index_db,
        path_liveness_cache=cache,
        path_liveness_cache_max=128,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/cache/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qdrant_count"] == 42
    assert body["index_db_count"] == 40
    assert body["drift"] == 2  # 42 - 40
    assert body["path_liveness_cache_size"] == 1
    assert body["path_liveness_cache_max"] == 128


def test_system_router_healthz_uses_to_thread_for_qdrant(
    fake_qdrant, fake_cfg, fake_index_db,
):
    """`qdrant.healthz()` is invoked (under `asyncio.to_thread` semantics
    in the live app; tested here by direct call)."""
    from search.routers.system import build_system_router

    router = build_system_router(
        qdrant=fake_qdrant,
        cfg=fake_cfg,
        index_db=fake_index_db,
        path_liveness_cache={},
        path_liveness_cache_max=128,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        client.get("/healthz")
    fake_qdrant.healthz.assert_called_once()


def test_system_router_handles_drift_unknown(
    fake_qdrant, fake_cfg, fake_index_db,
):
    """When `qdrant_count == -1` (Qdrant unreachable), `drift == "unknown"`."""
    from search.routers.system import build_system_router

    fake_index_db.qdrant_point_count.return_value = -1
    router = build_system_router(
        qdrant=fake_qdrant,
        cfg=fake_cfg,
        index_db=fake_index_db,
        path_liveness_cache={},
        path_liveness_cache_max=128,
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/cache/status")
    body = resp.json()
    assert body["drift"] == "unknown"


def test_system_router_does_not_register_other_endpoints(
    fake_qdrant, fake_cfg, fake_index_db,
):
    """The system router owns exactly `/healthz`, `/api/system/status`,
    and `/api/cache/status`. Adding endpoints here is the right way to
    expand the system surface — but each new endpoint adds work to the
    startup-time introspection. If you add a 4th endpoint, this test
    is the place to update."""
    from search.routers.system import build_system_router

    router = build_system_router(
        qdrant=fake_qdrant,
        cfg=fake_cfg,
        index_db=fake_index_db,
        path_liveness_cache={},
        path_liveness_cache_max=128,
    )
    paths = {r.path for r in router.routes}
    assert paths == {"/healthz", "/api/system/status", "/api/cache/status"}