"""Tests for the album CRUD + membership API.

Albums are user-curated collections of favourites. The favourites
table is the implicit default album; every endpoint here is for
user-created albums only. Membership is independent of favourites
status — a photo can be in an album without being favourited.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config

CAT_ID = "11111111-1111-1111-1111-111111111111"
DOG_ID = "22222222-2222-2222-2222-222222222222"
BIRD_ID = "33333333-3333-3333-3333-333333333333"


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
            (BIRD_ID, _mock_embed("bird"), {"id": BIRD_ID, "path": str(nas_base / "bird.jpg"), "mtime": 3, "size": 30, "indexed_at": "2026-01-01T00:00:00Z"}),
        ],
        wait=True,
    )
    Image.new("RGB", (8, 8), (255, 0, 0)).save(nas_base / "cat.jpg")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(nas_base / "dog.jpg")
    Image.new("RGB", (8, 8), (0, 0, 255)).save(nas_base / "bird.jpg")

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        yield client
    app_mod.reset_for_tests()


# ---------------- create ----------------


def test_create_album_returns_summary(app_with_qdrant):
    resp = app_with_qdrant.post(
        "/api/albums",
        json={"name": "studio portraits", "description": "natural light only"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "studio portraits"
    assert data["description"] == "natural light only"
    assert data["member_count"] == 0
    assert data["cover_favorite_id"] == ""
    assert "id" in data
    assert data["id"] > 0


def test_create_album_rejects_duplicate_name(app_with_qdrant):
    app_with_qdrant.post("/api/albums", json={"name": "studio"})
    resp = app_with_qdrant.post("/api/albums", json={"name": "studio"})
    assert resp.status_code == 400
    assert "already exists" in resp.json()["detail"]


def test_create_album_rejects_empty_name(app_with_qdrant):
    resp = app_with_qdrant.post("/api/albums", json={"name": "   "})
    assert resp.status_code == 400


def test_create_album_trims_whitespace(app_with_qdrant):
    resp = app_with_qdrant.post(
        "/api/albums", json={"name": "  trimmed  "},
    )
    assert resp.json()["name"] == "trimmed"


# ---------------- list ----------------


def test_list_albums_returns_empty_initially(app_with_qdrant):
    resp = app_with_qdrant.get("/api/albums")
    assert resp.status_code == 200
    assert resp.json()["albums"] == []


def test_list_albums_returns_all_ordered_by_name(app_with_qdrant):
    app_with_qdrant.post("/api/albums", json={"name": "zebra"})
    app_with_qdrant.post("/api/albums", json={"name": "alpha"})
    app_with_qdrant.post("/api/albums", json={"name": "mango"})
    resp = app_with_qdrant.get("/api/albums")
    names = [a["name"] for a in resp.json()["albums"]]
    assert names == ["alpha", "mango", "zebra"]


def test_list_albums_first_member_id_empty_when_no_members(app_with_qdrant):
    resp = app_with_qdrant.post("/api/albums", json={"name": "empty"})
    album_id = resp.json()["id"]
    listed = app_with_qdrant.get("/api/albums").json()["albums"]
    assert listed[0]["id"] == album_id
    assert listed[0]["first_member_id"] == ""
    assert listed[0]["member_count"] == 0


def test_list_albums_first_member_id_is_most_recently_added(app_with_qdrant):
    """first_member_id is the most RECENT photo *added* to the album
    (album_memberships.added_at DESC), not the oldest."""
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "ordered"},
    ).json()["id"]
    # Add in order: cat, dog, bird. Each add uses _utc_now() so the
    # added_at timestamps are strictly increasing on the millisecond.
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{DOG_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{BIRD_ID}")

    listed = app_with_qdrant.get("/api/albums").json()["albums"]
    assert listed[0]["first_member_id"] == BIRD_ID
    assert listed[0]["member_count"] == 3


def test_list_albums_first_member_id_updates_after_removal(app_with_qdrant):
    """If the most-recently-added member is removed, the next-newest
    member becomes the first_member_id."""
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "trim"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{DOG_ID}")
    app_with_qdrant.delete(f"/api/albums/{album_id}/members/{DOG_ID}")

    listed = app_with_qdrant.get("/api/albums").json()["albums"]
    assert listed[0]["first_member_id"] == CAT_ID


def test_list_albums_first_member_id_tie_break_is_deterministic(app_with_qdrant):
    """Two members with the same added_at resolve to a stable photo.

    Two photos added in the same microsecond get identical
    `added_at` values. Without a tie-break, SQLite picks
    arbitrarily between them and the album card cover can flicker
    between reloads. The tie-break is `favorite_id DESC` — the
    lexicographically largest id wins, deterministically.
    """
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "tie"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{DOG_ID}")

    # Force the tie: same added_at for both members.
    with app_with_qdrant.app.state.index_db._lock:
        app_with_qdrant.app.state.index_db._conn.execute(
            "UPDATE album_memberships SET added_at = ? WHERE album_id = ?",
            ("2026-01-01T00:00:00.000000+00:00", album_id),
        )
        app_with_qdrant.app.state.index_db._conn.commit()

    listed = app_with_qdrant.get("/api/albums").json()["albums"]
    # DOG_ID > CAT_ID lexicographically, so DESC picks DOG_ID.
    assert listed[0]["first_member_id"] == DOG_ID

    # Repeated reads must agree — the cover doesn't flicker.
    for _ in range(3):
        again = app_with_qdrant.get("/api/albums").json()["albums"]
        assert again[0]["first_member_id"] == DOG_ID


# ---------------- get detail ----------------


def test_get_album_returns_members(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    # Add members via API
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{DOG_ID}")

    resp = app_with_qdrant.get(f"/api/albums/{album_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == album_id
    assert data["name"] == "test"
    assert data["member_total"] == 2
    member_ids = {m["id"] for m in data["members"]}
    assert member_ids == {CAT_ID, DOG_ID}


def test_get_album_api_default_limit_is_35(app_with_qdrant, monkeypatch):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "api default limit"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")

    index_db = app_mod.get_index_db()
    original = index_db.list_album_members
    observed = {}

    def record_limit(album, limit, offset):
        observed["limit"] = limit
        return original(album, limit, offset)

    monkeypatch.setattr(index_db, "list_album_members", record_limit)
    response = app_with_qdrant.get(f"/api/albums/{album_id}")

    assert response.status_code == 200
    assert observed["limit"] == 35




def test_get_album_unknown_returns_404(app_with_qdrant):
    resp = app_with_qdrant.get("/api/albums/9999")
    assert resp.status_code == 404


# ---------------- update ----------------


def test_update_album_renames(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "old name"},
    ).json()["id"]
    resp = app_with_qdrant.patch(
        f"/api/albums/{album_id}", json={"name": "new name"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "new name"


def test_update_album_rejects_duplicate_name(app_with_qdrant):
    a = app_with_qdrant.post("/api/albums", json={"name": "first"}).json()
    app_with_qdrant.post("/api/albums", json={"name": "second"})
    resp = app_with_qdrant.patch(
        f"/api/albums/{a['id']}", json={"name": "second"},
    )
    assert resp.status_code == 400


def test_update_album_unknown_returns_404(app_with_qdrant):
    resp = app_with_qdrant.patch(
        "/api/albums/9999", json={"name": "x"},
    )
    assert resp.status_code == 404


def test_update_album_rejects_empty_body(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    resp = app_with_qdrant.patch(f"/api/albums/{album_id}", json={})
    assert resp.status_code == 400


# ---------------- delete ----------------


def test_delete_album_cascades_memberships(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "doomed"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")

    resp = app_with_qdrant.delete(f"/api/albums/{album_id}")
    assert resp.status_code == 204
    # Album gone
    assert app_with_qdrant.get(f"/api/albums/{album_id}").status_code == 404
    # List is empty
    assert app_with_qdrant.get("/api/albums").json()["albums"] == []


def test_delete_album_removes_centroid_from_registry(app_with_qdrant):
    """DELETE /api/albums/{id} must drop the album's centroid from
    /api/centroids immediately, not just on next process restart.
    Regression test — `invalidate` alone leaves the spec in
    `_by_name` so `list()` keeps returning it."""
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "doomed"},
    ).json()["id"]

    # Centroid present after create.
    resp = app_with_qdrant.get("/api/centroids")
    assert resp.status_code == 200
    names_before = {d["name"] for d in resp.json()["dynamic_centroids"]}
    assert f"album:{album_id}" in names_before

    # Delete — centroid should disappear immediately.
    app_with_qdrant.delete(f"/api/albums/{album_id}")
    resp = app_with_qdrant.get("/api/centroids")
    assert resp.status_code == 200
    names_after = {d["name"] for d in resp.json()["dynamic_centroids"]}
    assert f"album:{album_id}" not in names_after


def test_delete_album_centroid_search_404_after_delete(app_with_qdrant):
    """After delete, /api/centroids/album:{id}/search should 404 —
    the centroid is gone, not just stale."""
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "doomed"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")

    # Centroid-anchored search works while album exists.
    resp = app_with_qdrant.get(
        f"/api/centroids/album:{album_id}/search?limit=5"
    )
    assert resp.status_code == 200

    # After delete, the name is unknown — 404.
    app_with_qdrant.delete(f"/api/albums/{album_id}")
    resp = app_with_qdrant.get(
        f"/api/centroids/album:{album_id}/search?limit=5"
    )
    assert resp.status_code == 404


def test_delete_album_unknown_returns_404(app_with_qdrant):
    resp = app_with_qdrant.delete("/api/albums/9999")
    assert resp.status_code == 404


# ---------------- member ops ----------------


def test_add_member_is_idempotent(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    r1 = app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    assert r1.status_code == 200
    # Adding again still returns 200, no error
    r2 = app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    assert r2.status_code == 200
    # Member count is still 1
    assert app_with_qdrant.get(f"/api/albums/{album_id}").json()["member_total"] == 1


def test_add_member_unknown_album_returns_404(app_with_qdrant):
    resp = app_with_qdrant.post(f"/api/albums/9999/members/{CAT_ID}")
    assert resp.status_code == 404


def test_remove_member_is_idempotent(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    # Remove a photo that was never added — still 204
    resp = app_with_qdrant.delete(f"/api/albums/{album_id}/members/{CAT_ID}")
    assert resp.status_code == 204


def test_remove_member_unknown_album_returns_404(app_with_qdrant):
    resp = app_with_qdrant.delete(f"/api/albums/9999/members/{CAT_ID}")
    assert resp.status_code == 404


def test_member_count_updates_correctly(app_with_qdrant):
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{DOG_ID}")
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{BIRD_ID}")
    assert app_with_qdrant.get(f"/api/albums/{album_id}").json()["member_total"] == 3
    app_with_qdrant.delete(f"/api/albums/{album_id}/members/{DOG_ID}")
    assert app_with_qdrant.get(f"/api/albums/{album_id}").json()["member_total"] == 2


def test_photo_can_be_in_multiple_albums(app_with_qdrant):
    """Many-to-many membership: same photo in 2 albums."""
    a = app_with_qdrant.post("/api/albums", json={"name": "a"}).json()
    b = app_with_qdrant.post("/api/albums", json={"name": "b"}).json()
    app_with_qdrant.post(f"/api/albums/{a['id']}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{b['id']}/members/{CAT_ID}")
    assert app_with_qdrant.get(f"/api/albums/{a['id']}").json()["member_total"] == 1
    assert app_with_qdrant.get(f"/api/albums/{b['id']}").json()["member_total"] == 1


def test_photo_in_album_independent_of_favourites(app_with_qdrant):
    """A photo can be in an album without being favourited, and
    unfavouriting does NOT remove it from albums.
    """
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    # Add to album WITHOUT favouriting first
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    assert app_with_qdrant.get(f"/api/albums/{album_id}").json()["member_total"] == 1

    # Favourite then unfavourite — album membership should survive
    app_with_qdrant.post(f"/api/favorites/{CAT_ID}")
    app_with_qdrant.delete(f"/api/favorites/{CAT_ID}")
    assert app_with_qdrant.get(f"/api/albums/{album_id}").json()["member_total"] == 1


# ---------------- by-favorite ----------------


def test_list_albums_for_favorite(app_with_qdrant):
    a = app_with_qdrant.post("/api/albums", json={"name": "a"}).json()
    b = app_with_qdrant.post("/api/albums", json={"name": "b"}).json()
    app_with_qdrant.post("/api/albums", json={"name": "c"})  # not in any membership

    app_with_qdrant.post(f"/api/albums/{a['id']}/members/{CAT_ID}")
    app_with_qdrant.post(f"/api/albums/{b['id']}/members/{CAT_ID}")

    resp = app_with_qdrant.get(f"/api/albums/by-favorite/{CAT_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["favorite_id"] == CAT_ID
    names = sorted(al["name"] for al in data["albums"])
    assert names == ["a", "b"]


def test_list_albums_for_favorite_empty(app_with_qdrant):
    resp = app_with_qdrant.get(f"/api/albums/by-favorite/{CAT_ID}")
    assert resp.status_code == 200
    assert resp.json()["albums"] == []


# ---------------- Default Album is unaffected by album ops ----------------


def test_default_album_unchanged_by_album_crud(app_with_qdrant):
    """Favouriting behaviour is independent of albums. The
    favourites table and endpoints are unaffected by album ops.
    """
    # Favourite two photos
    app_with_qdrant.post(f"/api/favorites/{CAT_ID}")
    app_with_qdrant.post(f"/api/favorites/{DOG_ID}")
    assert app_with_qdrant.get("/api/favorites").json()["total"] == 2

    # Create an album and add one of them
    album_id = app_with_qdrant.post("/api/albums", json={"name": "test"}).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")

    # Favourites count is still 2
    assert app_with_qdrant.get("/api/favorites").json()["total"] == 2
    # Album has 1
    assert app_with_qdrant.get(f"/api/albums/{album_id}").json()["member_total"] == 1

    # Delete the album — favourites count unchanged
    app_with_qdrant.delete(f"/api/albums/{album_id}")
    assert app_with_qdrant.get("/api/favorites").json()["total"] == 2


# ---------------- centroid integration ----------------


def test_album_registers_centroid_on_create(app_with_qdrant):
    """Creating an album registers it as a dynamic centroid."""
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]

    centroids = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    names = [c["name"] for c in centroids]
    assert f"album:{album_id}" in names


def test_album_centroid_updates_on_member_change(app_with_qdrant):
    """Adding a member invalidates the centroid cache so the next
    read recomputes. We verify by checking that the centroid has
    the correct n_images after a member add.
    """
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]

    # Initially empty centroid — n_images is None (not yet computed)
    centroids_before = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    album_centroid = next(
        c for c in centroids_before if c["name"] == f"album:{album_id}"
    )
    assert album_centroid["n_images"] is None

    # Add a member, then re-fetch — should now report n_images=1
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")
    centroids_after = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    album_centroid = next(
        c for c in centroids_after if c["name"] == f"album:{album_id}"
    )
    assert album_centroid["n_images"] == 1


def test_album_centroid_uses_centroid_key_id(app_with_qdrant):
    """The centroid key is id-based, not name-based, so renames
    don't break search references.
    """
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "old name"},
    ).json()["id"]
    centroid_name_before = f"album:{album_id}"
    centroids = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    assert centroid_name_before in [c["name"] for c in centroids]

    # Rename — centroid key still works
    app_with_qdrant.patch(
        f"/api/albums/{album_id}", json={"name": "new name"},
    )
    centroids_after = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    # Centroid key unchanged
    assert centroid_name_before in [c["name"] for c in centroids_after]
    # Label updated
    album_centroid = next(
        c for c in centroids_after if c["name"] == centroid_name_before
    )
    assert "new name" in album_centroid["label"]


def test_album_centroid_search_works(app_with_qdrant):
    """An album's centroid can be used as a search primitive via
    /api/search?centroid=album:{id}, same shape as the favourites
    centroid.
    """
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "cat album"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")

    resp = app_with_qdrant.get(f"/api/search?centroid=album:{album_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["centroid"] == f"album:{album_id}"
    assert len(data["results"]) > 0


def test_album_delete_invalidates_centroid_cache(app_with_qdrant):
    """Deleting an album unregisters its centroid entirely — the
    spec disappears from `/api/centroids` immediately rather than
    staying registered with an empty cached value until process
    restart. See `test_delete_album_removes_centroid_from_registry`
    and `test_delete_album_centroid_search_404_after_delete` for
    the user-visible assertions; this test covers the
    DynamicCentroidRegistry-level behavior via the same code path.
    """
    album_id = app_with_qdrant.post(
        "/api/albums", json={"name": "test"},
    ).json()["id"]
    app_with_qdrant.post(f"/api/albums/{album_id}/members/{CAT_ID}")

    # Trigger compute so the cached value exists, then verify the
    # spec is fully removed on delete.
    centroids = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    assert any(c["name"] == f"album:{album_id}" for c in centroids)

    app_with_qdrant.delete(f"/api/albums/{album_id}")

    centroids_after = app_with_qdrant.get("/api/centroids").json()["dynamic_centroids"]
    assert not any(c["name"] == f"album:{album_id}" for c in centroids_after)