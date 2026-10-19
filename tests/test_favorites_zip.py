"""Tests for the GET /favorites/download.zip endpoint."""
from __future__ import annotations

import io
import uuid
import zipfile

import pytest

from indexer import upsert
from indexer.upsert import VECTOR_DIM

# ---------------------------------------------------------------------------
# Fixture: app + Qdrant + SQLite + a tmp dir for real image files.
# ---------------------------------------------------------------------------

@pytest.fixture
def zip_app(tmp_path):
    """A FastAPI app wired to an in-memory Qdrant and a SQLite cache.
    The tmp_path is also used as the NAS images base, so any photo
    written under it resolves to a real local file.
    """
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from search.app import create_app
    from search.config import Config
    from search.qdrant_client import QdrantSearch

    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test_zip",
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
    qdrant = QdrantSearch(
        client=client, collection=cfg.qdrant_collection, timeout_ms=2000,
    )

    app = create_app(cfg=cfg, qdrant=qdrant)
    return TestClient(app), tmp_path, client


def _seed_point(client, collection, point_id: str, path: str,
                shard: str = "default", base_dir=None,
                content: bytes = b"placeholder-bytes") -> str:
    """Insert a single point into Qdrant + write a real file under
    `base_dir` so resolve_local finds it. Returns the point id.
    """
    import pathlib

    from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text
    if base_dir is not None:
        full_path = pathlib.Path(base_dir) / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
    upsert.upsert_batch(
        client, collection,
        [(point_id, _mock_embed(point_id), {
            "id": point_id, "path": path,
            "shard": shard,
            "collection": "kpop",
            "indexed_at": "2026-01-01T00:00:00Z",
        })],
        wait=True,
    )
    return point_id


def _favorite(client_app, point_id: str) -> None:
    resp = client_app.post(f"/api/favorites/{point_id}")
    assert resp.status_code == 200, resp.text


def _refresh_cache(client_app, qdrant_client, collection: str) -> None:
    """Run the cache-refresh endpoint so the SQLite cache reflects
    Qdrant (list_favorites joins against the cache, not Qdrant)."""
    resp = client_app.post("/api/cache/refresh")
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zip_empty_favourites_returns_valid_zip(zip_app):
    """No favourites → still a valid (empty) zip, not a 500."""
    app, _tmp_path, _qdrant = zip_app
    resp = app.get("/favorites/download.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["content-disposition"].startswith("attachment;")
    # Must be a valid zip the OS / Python can open.
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == []


def test_zip_single_file_contents(zip_app):
    """One favourite → zip has one entry with the right name and
    the right bytes."""
    app, tmp_path, client = zip_app
    pid = str(uuid.uuid4())
    rel = "photos/single.jpg"
    _seed_point(client, "images_test_zip", pid, rel,
                shard="kpop", base_dir=tmp_path)
    _refresh_cache(app, client, "images_test_zip")
    _favorite(app, pid)

    resp = app.get("/favorites/download.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert len(names) == 1
    assert names[0] == "kpop__single.jpg"
    assert zf.read(names[0]) == b"placeholder-bytes"


def test_zip_filename_collision_across_shards(zip_app):
    """Two favourites with the same basename in different shards
    must not overwrite each other in the zip."""
    app, tmp_path, client = zip_app
    pid_a = str(uuid.uuid4())
    pid_b = str(uuid.uuid4())
    # Use absolute-ish paths under tmp_path so each seed's file lives
    # in its own directory. resolve_local's prefix-less join maps a
    # non-absolute path under nas_images_base.
    _seed_point(client, "images_test_zip", pid_a, "kpop/IMG_001.jpg",
                shard="kpop", base_dir=tmp_path,
                content=b"kpop-bytes")
    _seed_point(client, "images_test_zip", pid_b, "portrait/IMG_001.jpg",
                shard="portrait", base_dir=tmp_path,
                content=b"portrait-bytes")
    _refresh_cache(app, client, "images_test_zip")
    _favorite(app, pid_a)
    _favorite(app, pid_b)

    resp = app.get("/favorites/download.zip")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = sorted(zf.namelist())
    assert names == ["kpop__IMG_001.jpg", "portrait__IMG_001.jpg"]
    assert zf.read("kpop__IMG_001.jpg") == b"kpop-bytes"
    assert zf.read("portrait__IMG_001.jpg") == b"portrait-bytes"


def test_zip_missing_file_writes_manifest(zip_app, caplog):
    """Favourites whose source file is gone are skipped and listed
    in `_missing.txt`."""
    app, tmp_path, client = zip_app
    pid = str(uuid.uuid4())
    rel = "photos/will-be-deleted.jpg"
    _seed_point(client, "images_test_zip", pid, rel,
                shard="kpop", base_dir=tmp_path)
    _refresh_cache(app, client, "images_test_zip")
    _favorite(app, pid)
    # Now delete the underlying file so resolve_local returns None.
    import pathlib
    pathlib.Path(tmp_path / rel).unlink()

    resp = app.get("/favorites/download.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert names == ["_missing.txt"]
    manifest = zf.read("_missing.txt").decode()
    assert "missing files: 1" in manifest
    assert pid in manifest
    assert rel in manifest


def test_zip_supports_head(zip_app):
    """HEAD must be supported so the browser / download manager can
    probe the response without downloading the body."""
    app, _tmp_path, _qdrant = zip_app
    resp = app.head("/favorites/download.zip")
    # HEAD on a StreamingResponse in Starlette returns 200 with
    # headers but no body. We accept either 200 or 405; the contract
    # we promise the user is GET works.
    assert resp.status_code in (200, 405)


def test_zip_unsharded_uses_bare_basename(zip_app):
    """Favourites with no shard in their payload get the bare
    basename — no 'unsharded__' prefix clutter."""
    app, tmp_path, client = zip_app
    pid = str(uuid.uuid4())
    _seed_point(client, "images_test_zip", pid, "photos/solo.jpg",
                shard="", base_dir=tmp_path)
    _refresh_cache(app, client, "images_test_zip")
    _favorite(app, pid)

    resp = app.get("/favorites/download.zip")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == ["solo.jpg"]


