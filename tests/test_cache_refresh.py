"""Tests for the POST /api/cache/refresh endpoint.

There is no cache to refresh any more. The photo corpus, its
metadata, and the native text index all live in Qdrant; this
SQLite database holds only the user's own state. What the endpoint
still does is the one thing that needed a trigger: dropping
user-state rows that point at photos the indexer has pruned.

The liveness check is against Qdrant, so a favourite is only ever
dropped when the COLLECTION says the photo is gone.
"""
from __future__ import annotations

import pytest

from indexer import upsert
from indexer.upsert import VECTOR_DIM


@pytest.fixture
def refresh_app(tmp_path, monkeypatch):
    """A FastAPI app wired to an in-memory Qdrant with three seeded
    points and a SQLite side store. Mirrors the search_api fixture
    shape so the refresh path is exercised end-to-end.
    """
    # The fix for the UNC-payload bug (2026-08-15) calls resolve_local
    # before _is_path_alive. The fixture's seeded paths don't match
    # the production prefix, so resolve_local would return None and
    # drop every row. Mock it to a no-op identity so the seeded paths
    # round-trip through the filter.
    from pathlib import Path as _Path

    from fastapi.testclient import TestClient

    # Lazy liveness check (added in fix/dual-store-cleanup) calls
    # Path(path).exists() on every row the random route returns.
    # The fixture seeds paths like /photos/a.jpg directly via Qdrant
    # payloads — they're not real files on disk, so the liveness
    # check would filter every row out. Mock the helper to return
    # True so the seeded rows survive the /api/random read path.
    from search import app as _app_mod
    from search.app import create_app
    from search.config import Config
    from search.qdrant_client import QdrantSearch
    monkeypatch.setattr(
        _app_mod,
        "resolve_local",
        lambda path, *a, **kw: _Path(path) if path else None,
    )
    monkeypatch.setattr(_app_mod, "_is_path_alive", lambda path: True)

    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test_refresh",
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

    # Seed three points in Qdrant. Qdrant in-memory requires UUID
    # point ids, so generate one per row instead of using "a"/"b"/"c".
    import uuid

    from qdrant_client import QdrantClient
    client = QdrantClient(location=":memory:")
    upsert.ensure_collection(client, cfg.qdrant_collection, dim=VECTOR_DIM)
    seed_ids = {
        "a": str(uuid.uuid4()),
        "b": str(uuid.uuid4()),
        "c": str(uuid.uuid4()),
        "d": str(uuid.uuid4()),
    }
    seed = [
        (seed_ids["a"], {"id": seed_ids["a"], "path": "/photos/a.jpg",
                          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (seed_ids["b"], {"id": seed_ids["b"], "path": "/photos/b.jpg",
                          "collection": "kpop", "indexed_at": "2026-01-01T00:00:00Z"}),
        (seed_ids["c"], {"id": seed_ids["c"], "path": "/photos/c.jpg",
                          "collection": "portrait", "indexed_at": "2026-01-01T00:00:00Z"}),
    ]
    from lustre_kernel.registry import MockEmbedder
    _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text
    upsert.upsert_batch(
        client, cfg.qdrant_collection,
        [(pid, _mock_embed(pid), payload) for pid, payload in seed],
        wait=True,
    )

    qdrant = QdrantSearch(
        client=client, collection=cfg.qdrant_collection, timeout_ms=2000,
    )

    app = create_app(cfg=cfg, qdrant=qdrant)
    # Make seed_ids and the raw QdrantClient discoverable to tests
    # via app state. Tests that add points mid-test reach in via
    # app.state.qdrant_client; tests that target a UUID reach in via
    # app.state.test_seed_ids.
    app.state.test_seed_ids = seed_ids
    app.state.qdrant_client = client
    app.state.qdrant_collection = cfg.qdrant_collection
    return TestClient(app)


@pytest.fixture
def refresh_app_state(refresh_app):
    """Exposes the in-memory seed_id mapping to tests that need to
    target a specific UUID (e.g. to favourite one).
    """
    return {"seed_ids": refresh_app.app.state.test_seed_ids}


def test_api_cache_refresh_reports_ok(refresh_app):
    """The endpoint answers with the purge result."""
    resp = refresh_app.get("/api/cache/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "purged" in data


def test_api_cache_refresh_purges_orphaned_favourites(
    refresh_app, refresh_app_state,
):
    """A favourite whose photo left the collection is purged."""
    seed_ids = refresh_app_state["seed_ids"]

    # Favourite a photo, then remove it from the collection.
    resp = refresh_app.post(f"/api/favorites/{seed_ids['a']}")
    assert resp.status_code in (200, 201)

    refresh_app.app.state.qdrant_client.delete(
        collection_name=refresh_app.app.state.qdrant_collection,
        points_selector=[seed_ids["a"]],
        wait=True,
    )

    resp = refresh_app.get("/api/cache/refresh")
    assert resp.status_code == 200
    assert resp.json()["purged"]["favorites"] == 1

    # And it is really gone.
    body = refresh_app.get("/api/favorites").json()
    assert seed_ids["a"] not in {f["id"] for f in body.get("favorites", [])}


def test_api_cache_refresh_preserves_favourites(
    refresh_app, refresh_app_state,
):
    """A favourite whose photo is still in the collection survives."""
    seed_ids = refresh_app_state["seed_ids"]

    resp = refresh_app.post(f"/api/favorites/{seed_ids['a']}")
    assert resp.status_code in (200, 201)

    resp = refresh_app.get("/api/cache/refresh")
    assert resp.status_code == 200
    assert resp.json()["purged"]["favorites"] == 0

    body = refresh_app.get("/api/favorites").json()
    assert seed_ids["a"] in {f["id"] for f in body.get("favorites", [])}


def test_api_cache_refresh_never_deletes_when_qdrant_unreachable(
    refresh_app, refresh_app_state, monkeypatch,
):
    """THE safety property.

    If the liveness check cannot be answered, nothing is deleted.
    A user-state row is authoritative; a failed lookup is not
    evidence that the photo is gone.
    """
    from search import index_db as index_db_mod

    seed_ids = refresh_app_state["seed_ids"]
    resp = refresh_app.post(f"/api/favorites/{seed_ids['a']}")
    assert resp.status_code in (200, 201)

    def _boom(_self, _ids):
        raise RuntimeError("qdrant unreachable")

    # Scope the patch to the refresh call only — the verification
    # below reads the store back through a path that also asks
    # Qdrant for liveness.
    with monkeypatch.context() as mp:
        mp.setattr(index_db_mod.IndexDB, "_live_point_ids", _boom)
        # The endpoint surfaces the failure rather than pretending the
        # rows are orphans.
        with pytest.raises(RuntimeError):
            refresh_app.get("/api/cache/refresh")

    body = refresh_app.get("/api/favorites").json()
    assert seed_ids["a"] in {f["id"] for f in body.get("favorites", [])}


def test_api_cache_refresh_accepts_the_legacy_full_flag(refresh_app):
    """An old caller passing `full=true` does not break.

    The parameter is accepted and ignored — there is no cache to
    rebuild.
    """
    resp = refresh_app.get("/api/cache/refresh?full=true")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_cache_refresh_purges_orphaned_dislikes(
    refresh_app, refresh_app_state,
):
    """A dislike whose photo left the collection is purged too."""
    seed_ids = refresh_app_state["seed_ids"]

    resp = refresh_app.post(f"/api/dislikes/{seed_ids['b']}")
    assert resp.status_code in (200, 201, 204)

    refresh_app.app.state.qdrant_client.delete(
        collection_name=refresh_app.app.state.qdrant_collection,
        points_selector=[seed_ids["b"]],
        wait=True,
    )

    resp = refresh_app.get("/api/cache/refresh")
    assert resp.status_code == 200
    assert resp.json()["purged"]["dislikes"] == 1
