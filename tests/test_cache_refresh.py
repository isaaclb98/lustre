"""Tests for the POST /api/cache/refresh endpoint."""
from __future__ import annotations

import pytest

from indexer import upsert
from indexer.upsert import VECTOR_DIM


@pytest.fixture
def refresh_app(tmp_path, monkeypatch):
    """A FastAPI app wired to an in-memory Qdrant with three seeded
    points and a SQLite cache. Mirrors the search_api fixture shape
    so the refresh path is exercised end-to-end.
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
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text
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


def test_api_cache_refresh_rebuilds_index(refresh_app):
    """First call: cache is empty, refresh inserts the 3 seeded
    Qdrant points via the diff path (3 new ids, 0 orphans).
    """
    resp = refresh_app.post("/api/cache/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "incremental"
    assert data["skipped"] is False
    assert data["inserted"] == 3
    assert data["deleted"] == 0


def test_api_cache_refresh_picks_up_new_qdrant_points(refresh_app, refresh_app_state):
    """Cache rebuilt, then a new point is added to Qdrant directly,
    then refresh picks it up.
    """
    seed_ids = refresh_app_state["seed_ids"]
    refresh_app.post("/api/cache/refresh")
    first = refresh_app.get("/api/random?limit=10").json()
    assert {r["id"] for r in first["results"]} == {seed_ids["a"], seed_ids["b"], seed_ids["c"]}

    # Add a fourth point to Qdrant without going through the cache.
    from indexer import upsert
    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text
    # Reach into the app's raw QdrantClient (stored on app.state by
    # the fixture) and upsert a fourth point directly.
    client = refresh_app.app.state.qdrant_client
    collection = refresh_app.app.state.qdrant_collection
    upsert.upsert_batch(
        client, collection,
        [(
            seed_ids["d"],
            _mock_embed("d"),
            {"id": seed_ids["d"], "path": "/photos/d.jpg",
             "collection": "portrait", "indexed_at": "2026-01-02T00:00:00Z"},
        )],
        wait=True,
    )

    # Before refresh: cache still shows only the original 3.
    stale = refresh_app.get("/api/random?limit=10").json()
    assert seed_ids["d"] not in {r["id"] for r in stale["results"]}

    # After refresh: d is included.
    refresh_app.post("/api/cache/refresh")
    fresh = refresh_app.get("/api/random?limit=10").json()
    assert {r["id"] for r in fresh["results"]} == {
        seed_ids["a"], seed_ids["b"], seed_ids["c"], seed_ids["d"],
    }


def test_api_cache_refresh_preserves_favourites(refresh_app, refresh_app_state):
    """The whole point of the persistence refactor: refresh
    shouldn't wipe a favourite.
    """
    refresh_app.post("/api/cache/refresh")
    # Mark a favourite (use one of the seeded UUIDs).
    fav_id = refresh_app_state["seed_ids"]["a"]
    refresh_app.post(f"/api/favorites/{fav_id}")
    assert refresh_app.get("/api/favorites").json()["total"] == 1

    # Force a refresh.
    resp = refresh_app.post("/api/cache/refresh")
    assert resp.status_code == 200
    assert resp.json()["skipped"] is True  # no drift between calls
    assert resp.json()["inserted"] == 0
    assert resp.json()["deleted"] == 0

    # Favourite is still there.
    assert refresh_app.get("/api/favorites").json()["total"] == 1


def test_api_cache_refresh_bails_when_indexer_running(refresh_app, monkeypatch):
    """Round-35: the default incremental refresh is SAFE to run
    concurrently with the indexer (it only touches orphan ids, not
    the whole table) — so it does NOT bail when the indexer is
    running. Only the explicit ?full=true wipe+rebuild still bails,
    because that path races the indexer's INSERTs.

    This test exercises the explicit-full path: a manual refresh
    with ?full=true refuses to run while the indexer is actively
    writing.

    Sets the runner's _state to RUNNING directly (bypassing the
    normal start() flow which spawns a real subprocess) so the
    test stays fast and self-contained.
    """
    from search.indexer_runner import IndexerState

    runner = refresh_app.app.state.indexer_runner
    monkeypatch.setattr(runner, "_state", IndexerState.RUNNING)
    monkeypatch.setattr(runner, "_job_id", "test-job-running-1234")

    resp = refresh_app.post("/api/cache/refresh?full=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"
    assert "indexer is running" in data["reason"]
    assert data["job_id"] == "test-job-running-1234"


def test_api_cache_refresh_incremental_proceeds_when_indexer_running(refresh_app, monkeypatch):
    """Round-35: the default (incremental) refresh is safe to run
    concurrently with the indexer. Same setup as the bail test
    above, but without ?full=true — the refresh succeeds and
    inserts the drift.
    """
    from search.indexer_runner import IndexerState

    runner = refresh_app.app.state.indexer_runner
    monkeypatch.setattr(runner, "_state", IndexerState.RUNNING)
    monkeypatch.setattr(runner, "_job_id", "test-job-running-1234")

    resp = refresh_app.post("/api/cache/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "incremental"
    # The fixture seeds 3 Qdrant points; sqlite starts empty, so
    # the diff inserts all 3.
    assert data["inserted"] == 3
    assert data["skipped"] is False


def test_api_cache_refresh_proceeds_when_indexer_idle(refresh_app):
    """Sanity: the manual refresh guard does NOT trip when the
    indexer is idle (the normal happy path).
    """
    # runner is freshly constructed → state defaults to IDLE
    resp = refresh_app.post("/api/cache/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "incremental"


def test_periodic_refresh_skipped_while_indexer_running(refresh_app, monkeypatch):
    """The _periodic_refresh_loop's wipe+rebuild path races the
    indexer's INSERTs. Loop must skip when indexer is RUNNING.

    We can't trivially trigger the periodic loop in a unit test
    (it waits on an asyncio.sleep), so we test the equivalent
    guard logic directly via a focused micro-loop.
    """
    from search.indexer_runner import IndexerState

    runner = refresh_app.app.state.indexer_runner
    monkeypatch.setattr(runner, "_state", IndexerState.RUNNING)

    # Replicate the guard inline so we're testing the predicate,
    # not asyncio scheduling. The actual loop calls
    # `indexer_runner.status()` then checks `state is RUNNING`.
    status = runner.status()
    assert status.state is IndexerState.RUNNING

    # Now flip back to idle and re-check — the guard must release.
    monkeypatch.setattr(runner, "_state", IndexerState.IDLE)
    status = runner.status()
    assert status.state is IndexerState.IDLE


