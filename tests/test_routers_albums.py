"""
tests/test_routers_albums.py — albums router contract (§B2 step 12).

Pins the albums router's contract: factory returns an APIRouter
with the 8 documented endpoints. Integration is verified by
tests/test_albums_api.py + tests/test_albums_zip.py + tests/test_favourites_centroid.py
which exercise the real wiring end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build(index_db, cfg, *, register=None, unregister=None, invalidate=None):
    from search.routers.albums import build_albums_router
    router = build_albums_router(
        index_db=index_db,
        cfg=cfg,
        register_album_centroid=register or MagicMock(),
        unregister_album_centroid=unregister or MagicMock(),
        invalidate_album_centroid=invalidate or MagicMock(),
    )
    app = FastAPI()
    app.include_router(router)
    return app


def _fake_cfg():
    c = MagicMock()
    c.top_k_default = 35
    return c


def _fake_index_db():
    db = MagicMock()
    db.list_albums.return_value = []
    db.get_album.return_value = None
    db.list_album_members.return_value = []
    db.count_album_members.return_value = 0
    db.list_albums_for_favorite.return_value = []
    return db




# --- GET /api/albums/{id} ---

def test_album_detail_carries_the_blurhash_of_each_member():
    """The album's members answer with their blurhash.

    The Qdrant payload already carries it and the query already
    hydrates the whole payload, so this costs no extra I/O — but the
    response model used to drop it. A presentation surface renders a
    photo per tile, and without a placeholder a photo that has not
    arrived yet shows as a grey block: a hole in the wall, scrolling
    past. On the drift this reads as pop-in.
    """
    db = MagicMock()
    db.get_album.return_value = {
        "id": 7,
        "name": "Album",
        "description": "",
        "cover_favorite_id": "",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
    }
    db.list_album_members.return_value = [
        {
            "id": "point-a",
            "path": "/nas/a.jpg",
            "added_at": "2026-01-01",
            "blurhash": "LEHV6nWB2yk8pyo0adR*.7kCMdnj",
        },
        {
            "id": "point-b",
            "path": "/nas/b.jpg",
            "added_at": "2026-01-01",
            "blurhash": None,
        },
    ]
    db.count_album_members.return_value = 2
    db.list_albums.return_value = []

    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/albums/7")

    assert resp.status_code == 200
    members = resp.json()["members"]
    assert members[0]["blurhash"] == "LEHV6nWB2yk8pyo0adR*.7kCMdnj"
    # A photo with no blurhash answers null, not a missing key — the
    # frontend reads it either way, but a stable shape is a stable
    # contract.
    assert "blurhash" in members[1]
    assert members[1]["blurhash"] is None

# --- POST /api/albums ---

def test_create_album_returns_summary_and_registers_centroid():
    db = _fake_index_db()
    db.create_album.return_value = 42
    db.list_albums.return_value = [
        {"id": 42, "name": "faves", "description": "stuff",
         "cover_favorite_id": "", "member_count": 0,
         "created_at": "2026-08-22", "updated_at": "2026-08-22"},
    ]
    register = MagicMock()
    app = _build(db, _fake_cfg(), register=register)
    with TestClient(app) as client:
        resp = client.post(
            "/api/albums", json={"name": "faves", "description": "stuff"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 42
    assert data["name"] == "faves"
    register.assert_called_once_with(42)


def test_create_album_duplicate_name_returns_400():
    db = _fake_index_db()
    db.create_album.side_effect = ValueError("duplicate name")
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.post(
            "/api/albums", json={"name": "faves", "description": ""},
        )
    assert resp.status_code == 400


# --- GET /api/albums ---

def test_list_albums_returns_albums_list():
    db = _fake_index_db()
    db.list_albums.return_value = [
        {"id": 1, "name": "a", "description": "", "cover_favorite_id": "",
         "member_count": 3, "created_at": "2026-01-01", "updated_at": "2026-01-02"},
        {"id": 2, "name": "b", "description": "", "cover_favorite_id": "",
         "member_count": 5, "created_at": "2026-01-03", "updated_at": "2026-01-04"},
    ]
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/albums")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["albums"]) == 2
    assert data["albums"][0]["name"] == "a"
    assert data["albums"][1]["member_count"] == 5


# --- GET /api/albums/{id} ---

def test_get_album_returns_detail_with_paged_members():
    db = _fake_index_db()
    db.get_album.return_value = {
        "id": 7, "name": "test", "description": "desc",
        "cover_favorite_id": "cov", "created_at": "2026-01-01",
        "updated_at": "2026-01-02",
    }
    db.list_album_members.return_value = [
        {"id": "a", "path": "/a.jpg", "added_at": "2026-01-02"},
        {"id": "b", "path": "/b.jpg", "added_at": "2026-01-03"},
    ]
    db.count_album_members.return_value = 2
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/albums/7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 7
    assert len(data["members"]) == 2
    assert data["member_total"] == 2


def test_get_album_unknown_id_returns_404():
    db = _fake_index_db()
    db.get_album.return_value = None
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/albums/999")
    assert resp.status_code == 404


# --- PATCH /api/albums/{id} ---

def test_update_album_partial_name_only_passes_description_through():
    """PATCH with only `name` set passes description through unchanged.

    The route does NOT read the existing description; whatever the
    client sent (None here) is forwarded to rename_album. Preserving
    the existing row's description on partial updates would require
    a separate read; current behavior is documented as "caller sends
    the full intended state".
    """
    db = _fake_index_db()
    db.rename_album.return_value = True
    db.list_albums.return_value = [
        {"id": 1, "name": "new", "description": "",
         "cover_favorite_id": "", "member_count": 0,
         "created_at": "x", "updated_at": "z"},
    ]
    register = MagicMock()
    app = _build(db, _fake_cfg(), register=register)
    with TestClient(app) as client:
        resp = client.patch("/api/albums/1", json={"name": "new"})
    assert resp.status_code == 200
    # rename_album(name="new", description=None) — caller sends full state.
    assert db.rename_album.call_args.args[1:] == ("new", None)
    register.assert_called_once_with(1)


def test_update_album_partial_description_only_reads_existing_name():
    """PATCH with only `description` set reads the existing name to
    satisfy rename_album's "both args or neither" signature."""
    db = _fake_index_db()
    db.get_album.return_value = {
        "id": 1, "name": "keep-name", "description": "old",
        "cover_favorite_id": "", "created_at": "x", "updated_at": "y",
    }
    db.rename_album.return_value = True
    db.list_albums.return_value = [
        {"id": 1, "name": "keep-name", "description": "new",
         "cover_favorite_id": "", "member_count": 0,
         "created_at": "x", "updated_at": "z"},
    ]
    register = MagicMock()
    app = _build(db, _fake_cfg(), register=register)
    with TestClient(app) as client:
        resp = client.patch("/api/albums/1", json={"description": "new"})
    assert resp.status_code == 200
    # rename_album reads existing name="keep-name", description="new".
    assert db.rename_album.call_args.args[1:] == ("keep-name", "new")


def test_update_album_empty_body_returns_400():
    db = _fake_index_db()
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.patch("/api/albums/1", json={})
    assert resp.status_code == 400


def test_update_album_unknown_returns_404():
    db = _fake_index_db()
    db.get_album.return_value = None  # partial-update lookup misses
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.patch("/api/albums/999", json={"description": "x"})
    assert resp.status_code == 404


def test_update_album_duplicate_name_returns_400():
    db = _fake_index_db()
    db.rename_album.side_effect = ValueError("duplicate")
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.patch(
            "/api/albums/1", json={"name": "new", "description": "x"},
        )
    assert resp.status_code == 400


# --- DELETE /api/albums/{id} ---

def test_delete_album_unregisters_centroid():
    db = _fake_index_db()
    db.delete_album.return_value = True
    unregister = MagicMock()
    app = _build(db, _fake_cfg(), unregister=unregister)
    with TestClient(app) as client:
        resp = client.delete("/api/albums/1")
    assert resp.status_code == 204
    unregister.assert_called_once_with(1)


def test_delete_album_unknown_returns_404():
    db = _fake_index_db()
    db.delete_album.return_value = False
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.delete("/api/albums/999")
    assert resp.status_code == 404


# --- POST /api/albums/{id}/members/{fid} ---

def test_add_member_invalidates_centroid():
    db = _fake_index_db()
    db.add_album_member.return_value = True
    db.list_album_members.return_value = [
        {"id": "fid-1", "path": "/x.jpg", "added_at": "2026-08-22"},
    ]
    invalidate = MagicMock()
    app = _build(db, _fake_cfg(), invalidate=invalidate)
    with TestClient(app) as client:
        resp = client.post("/api/albums/1/members/fid-1")
    assert resp.status_code == 200
    assert resp.json()["favorite_id"] == "fid-1"
    invalidate.assert_called_once_with(1)


def test_add_member_already_member_returns_404():
    db = _fake_index_db()
    db.add_album_member.return_value = False
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.post("/api/albums/1/members/fid-1")
    assert resp.status_code == 404


# --- DELETE /api/albums/{id}/members/{fid} ---

def test_remove_member_invalidates_centroid():
    db = _fake_index_db()
    db.remove_album_member.return_value = True
    invalidate = MagicMock()
    app = _build(db, _fake_cfg(), invalidate=invalidate)
    with TestClient(app) as client:
        resp = client.delete("/api/albums/1/members/fid-1")
    assert resp.status_code == 204
    invalidate.assert_called_once_with(1)


def test_remove_member_not_a_member_returns_404():
    db = _fake_index_db()
    db.remove_album_member.return_value = False
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.delete("/api/albums/1/members/fid-1")
    assert resp.status_code == 404


# --- GET /api/albums/by-favorite/{fid} ---

def test_list_albums_for_favorite_returns_summaries_with_count_1():
    db = _fake_index_db()
    db.list_albums_for_favorite.return_value = [
        {"id": 1, "name": "a", "description": ""},
        {"id": 2, "name": "b", "description": "with-desc"},
    ]
    app = _build(db, _fake_cfg())
    with TestClient(app) as client:
        resp = client.get("/api/albums/by-favorite/fid-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["favorite_id"] == "fid-1"
    assert len(data["albums"]) == 2
    # The by-favorite view always reports member_count=1.
    assert all(a["member_count"] == 1 for a in data["albums"])
