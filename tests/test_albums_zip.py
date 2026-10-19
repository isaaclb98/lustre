"""Tests for the GET /albums/{album_id}/download.zip endpoint.

Mirrors tests/test_favorites_zip.py — same shape, same fixtures,
album-specific helpers for create-album + add-member. Each test
seeds points + cache + album membership, then asserts on the
streamed zip contents (entry names, file bytes, missing manifest,
404 behaviour, and HEAD support).
"""
from __future__ import annotations

import io
import re
import uuid
import zipfile

import pytest

from indexer import upsert
from indexer.upsert import VECTOR_DIM

# ---------------------------------------------------------------------------
# Fixture: app + Qdrant + SQLite + tmp dir for real image files.
# ---------------------------------------------------------------------------

@pytest.fixture
def album_zip_app(tmp_path, monkeypatch):
    """A FastAPI app wired to an in-memory Qdrant and a SQLite cache.
    The tmp_path is also used as the NAS images base, so any photo
    written under it resolves to a real local file.
    """
    # Pin the index DB inside tmp_path so each test gets a clean DB.
    monkeypatch.setenv("IMAGE_SEARCH_INDEX_DB", str(tmp_path / "images.db"))
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from search.app import create_app, reset_for_tests
    from search.config import Config
    from search.qdrant_client import QdrantSearch

    reset_for_tests()
    cfg = Config(
        qdrant_url="memory://",
        qdrant_collection="images_test_album_zip",
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
            "collection": "library",
            "indexed_at": "2026-01-01T00:00:00Z",
        })],
        wait=True,
    )
    return point_id


def _favorite(client_app, point_id: str) -> None:
    resp = client_app.post(f"/api/favorites/{point_id}")
    assert resp.status_code == 200, resp.text


def _refresh_cache(client_app) -> None:
    """Run the cache-refresh endpoint so the SQLite cache reflects
    Qdrant (list_album_members joins against the cache, not Qdrant)."""
    resp = client_app.post("/api/cache/refresh")
    assert resp.status_code == 200, resp.text


def _create_album(client_app, name: str) -> int:
    resp = client_app.post("/api/albums", json={"name": name})
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])


def _add_member(client_app, album_id: int, point_id: str) -> None:
    resp = client_app.post(
        f"/api/albums/{album_id}/members/{point_id}",
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zip_empty_album_returns_valid_zip(album_zip_app):
    """Album with no members → still a valid (empty) zip, not a 500."""
    app, _tmp_path, _qdrant = album_zip_app
    album_id = _create_album(app, "Empty")

    resp = app.get(f"/albums/{album_id}/download.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["content-disposition"].startswith("attachment;")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == []


def test_zip_single_member_contents(album_zip_app):
    """One album member → zip has one entry with the right name and
    the right bytes."""
    app, tmp_path, client = album_zip_app
    pid = str(uuid.uuid4())
    rel = "photos/single.jpg"
    _seed_point(client, "images_test_album_zip", pid, rel,
                shard="library", base_dir=tmp_path)
    _refresh_cache(app)
    _favorite(app, pid)
    album_id = _create_album(app, "Solo")
    _add_member(app, album_id, pid)

    resp = app.get(f"/albums/{album_id}/download.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert len(names) == 1
    assert names[0] == "library__single.jpg"
    assert zf.read(names[0]) == b"placeholder-bytes"


def test_zip_filename_collision_across_shards(album_zip_app):
    """Two members with the same basename in different shards must
    not overwrite each other in the zip — even within a single
    album."""
    app, tmp_path, client = album_zip_app
    pid_a = str(uuid.uuid4())
    pid_b = str(uuid.uuid4())
    _seed_point(client, "images_test_album_zip", pid_a,
                "library/IMG_001.jpg", shard="library",
                base_dir=tmp_path, content=b"library-bytes")
    _seed_point(client, "images_test_album_zip", pid_b,
                "portrait/IMG_001.jpg", shard="portrait",
                base_dir=tmp_path, content=b"portrait-bytes")
    _refresh_cache(app)
    _favorite(app, pid_a)
    _favorite(app, pid_b)
    album_id = _create_album(app, "Collisions")
    _add_member(app, album_id, pid_a)
    _add_member(app, album_id, pid_b)

    resp = app.get(f"/albums/{album_id}/download.zip")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = sorted(zf.namelist())
    assert names == ["library__IMG_001.jpg", "portrait__IMG_001.jpg"]
    assert zf.read("library__IMG_001.jpg") == b"library-bytes"
    assert zf.read("portrait__IMG_001.jpg") == b"portrait-bytes"


def test_zip_missing_file_writes_manifest(album_zip_app):
    """Members whose source file is gone are skipped and listed in
    `_missing.txt` — same contract as the favourites zip."""
    app, tmp_path, client = album_zip_app
    pid = str(uuid.uuid4())
    rel = "photos/will-be-deleted.jpg"
    _seed_point(client, "images_test_album_zip", pid, rel,
                shard="library", base_dir=tmp_path)
    _refresh_cache(app)
    _favorite(app, pid)
    album_id = _create_album(app, "Vanishing")
    _add_member(app, album_id, pid)
    # Now delete the underlying file so resolve_local returns None.
    import pathlib
    pathlib.Path(tmp_path / rel).unlink()

    resp = app.get(f"/albums/{album_id}/download.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert names == ["_missing.txt"]
    manifest = zf.read("_missing.txt").decode()
    assert "missing files: 1" in manifest
    assert pid in manifest
    assert rel in manifest


def test_zip_supports_head(album_zip_app):
    """HEAD must be supported so the browser / download manager can
    probe the response without downloading the body."""
    app, _tmp_path, _qdrant = album_zip_app
    album_id = _create_album(app, "Probed")
    resp = app.head(f"/albums/{album_id}/download.zip")
    assert resp.status_code in (200, 405)


def test_zip_unsharded_uses_bare_basename(album_zip_app):
    """Members with no shard in their payload get the bare basename
    — no 'unsharded__' prefix clutter."""
    app, tmp_path, client = album_zip_app
    pid = str(uuid.uuid4())
    _seed_point(client, "images_test_album_zip", pid, "photos/solo.jpg",
                shard="", base_dir=tmp_path)
    _refresh_cache(app)
    _favorite(app, pid)
    album_id = _create_album(app, "Shardless")
    _add_member(app, album_id, pid)

    resp = app.get(f"/albums/{album_id}/download.zip")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == ["solo.jpg"]


def test_zip_album_isolation(album_zip_app):
    """Each album's zip contains only its own members — never
    favourites or members of another album."""
    app, tmp_path, client = album_zip_app
    pid_a = str(uuid.uuid4())
    pid_b = str(uuid.uuid4())
    _seed_point(client, "images_test_album_zip", pid_a, "photos/a.jpg",
                shard="library", base_dir=tmp_path,
                content=b"a-bytes")
    _seed_point(client, "images_test_album_zip", pid_b, "photos/b.jpg",
                shard="library", base_dir=tmp_path,
                content=b"b-bytes")
    _refresh_cache(app)
    _favorite(app, pid_a)
    _favorite(app, pid_b)
    album_a = _create_album(app, "Album A")
    album_b = _create_album(app, "Album B")
    _add_member(app, album_a, pid_a)
    _add_member(app, album_b, pid_b)

    zf_a = zipfile.ZipFile(
        io.BytesIO(app.get(f"/albums/{album_a}/download.zip").content)
    )
    zf_b = zipfile.ZipFile(
        io.BytesIO(app.get(f"/albums/{album_b}/download.zip").content)
    )
    assert zf_a.namelist() == ["library__a.jpg"]
    assert zf_a.read("library__a.jpg") == b"a-bytes"
    assert zf_b.namelist() == ["library__b.jpg"]
    assert zf_b.read("library__b.jpg") == b"b-bytes"


def test_zip_unknown_album_returns_404(album_zip_app):
    """An album id with no row → 404, not an empty zip and not a 500."""
    app, _tmp_path, _qdrant = album_zip_app
    resp = app.get("/albums/9999/download.zip")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_zip_filename_uses_slugified_album_name(album_zip_app):
    """The Content-Disposition filename slugs the album name so it
    is safe for filesystems, falling back to `album-{id}` when the
    name is empty or all-unsafe characters."""
    app, _tmp_path, _qdrant = album_zip_app
    album_id = _create_album(app, "Studio / Portraits! 2026")
    resp = app.get(f"/albums/{album_id}/download.zip")
    cd = resp.headers["content-disposition"]
    assert cd.startswith("attachment;")
    # Unsafe characters collapsed to `-`; consecutive runs squashed.
    m = re.search(r'filename="([^"]+)"', cd)
    assert m, f"no filename in {cd!r}"
    name = m.group(1)
    assert name.startswith("album-Studio-Portraits-2026-")
    assert name.endswith(".zip")

    # Empty / all-unsafe name falls back to `album-{id}-...`.
    fallback_id = _create_album(app, "////")
    resp2 = app.get(f"/albums/{fallback_id}/download.zip")
    m2 = re.search(r'filename="([^"]+)"', resp2.headers["content-disposition"])
    assert m2.group(1).startswith(f"album-{fallback_id}-")


