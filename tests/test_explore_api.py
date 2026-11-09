"""Tests for the Explore API — the per-album feedback loop.

Explore is a Qdrant `recommend()` round whose POSITIVES are the
album's member photos and whose NEGATIVES are the photos the user
down-voted during the session. These tests pin the contract:

- a round serves the album's photos, with the album's own members
  excluded from the results
- a down-vote is session-scoped: it becomes a negative for the next
  round and never reaches the `dislikes` table
- an up-vote is permanent: the photo joins the album
- the seen-set keeps a photo from being served twice
- dead member ids are filtered before they reach `recommend()`, which
  404s the whole call when any seed id is missing
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
DEAD_ID = "44444444-4444-4444-4444-444444444444"

# Photos that live in the library but are NOT album members. Explore
# serves "more like the album", so the results have to come from
# outside the album — these are what a round returns. There are
# enough of them that a second round still has fresh candidates
# after the first round has been served.
OUTSIDE_IDS = [
    "55555555-5555-5555-5555-555555555555",
    "66666666-6666-6666-6666-666666666666",
    "77777777-7777-7777-7777-777777777777",
    "88888888-8888-8888-8888-888888888888",
    "99999999-9999-9999-9999-999999999999",
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "cccccccc-cccc-cccc-cccc-cccccccccccc",
]


@pytest.fixture
def explore_client(qdrant_in_memory, nas_base):
    from PIL import Image

    from lustre_kernel.registry import MockEmbedder

    _mock_embed = MockEmbedder(
        dim=VECTOR_DIM, resolution=384
    ).embed_text

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
    upsert.ensure_collection(
        qdrant_in_memory.client, qdrant_in_memory.collection, dim=VECTOR_DIM
    )
    members = [
        (CAT_ID, "cat"),
        (DOG_ID, "dog"),
        (BIRD_ID, "bird"),
    ]
    outsiders = [
        (OUTSIDE_IDS[i], f"outside {i}") for i in range(len(OUTSIDE_IDS))
    ]
    points = []
    for point_id, caption in members + outsiders:
        filename = f"{caption.replace(' ', '_')}.jpg"
        points.append(
            (
                point_id,
                _mock_embed(caption),
                {
                    "id": point_id,
                    "path": str(nas_base / filename),
                    "mtime": 1,
                    "size": 10,
                    "indexed_at": "2026-01-01T00:00:00Z",
                },
            )
        )
    upsert.upsert_batch(
        qdrant_in_memory.client,
        qdrant_in_memory.collection,
        points,
        wait=True,
    )
    for _point_id, caption in members + outsiders:
        filename = f"{caption.replace(' ', '_')}.jpg"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(nas_base / filename)

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        yield client
    app_mod.reset_for_tests()


def _make_album(client, member_ids, name="Explore Test"):
    """Create an album containing `member_ids`."""
    res = client.post("/api/albums", json={"name": name})
    assert res.status_code == 200, res.text
    album_id = res.json()["id"]
    for member_id in member_ids:
        res = client.post(
            f"/api/albums/{album_id}/members/{member_id}"
        )
        assert res.status_code == 200, res.text
    return album_id


# ---------------- the built-in Favourites album ----------------


def _favourite(client, point_id):
    """Mark a photo as a favourite."""
    res = client.post(f"/api/favorites/{point_id}")
    assert res.status_code == 200, res.text


def test_round_returns_results_for_the_favourites_album(explore_client):
    """The built-in album explores the same way a user album does."""
    for point_id in (CAT_ID, DOG_ID, BIRD_ID):
        _favourite(explore_client, point_id)

    res = explore_client.get("/api/explore/favourites/round")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["results"], "the round should serve photos"
    # The favourites themselves are excluded — the round is "more
    # like these", not a copy of them.
    served = {r["id"] for r in body["results"]}
    assert not served & {CAT_ID, DOG_ID, BIRD_ID}


def test_an_upvote_on_the_favourites_album_favourites_the_photo(explore_client):
    """An up-vote on the built-in album marks the photo a favourite.

    The built-in album's membership IS the favourites table, so
    "add it to the album" and "favourite it" are the same act.
    """
    for point_id in (CAT_ID, DOG_ID, BIRD_ID):
        _favourite(explore_client, point_id)

    res = explore_client.get("/api/explore/favourites/round?limit=1")
    assert res.status_code == 200, res.text
    served_id = res.json()["results"][0]["id"]

    before = explore_client.get("/api/favorites?limit=99999").json()
    assert served_id not in {
        r["id"] for r in before.get("favorites", [])
    }, "the served photo should not already be a favourite"

    res = explore_client.post(
        "/api/explore/favourites/vote",
        json={"photo_id": served_id, "vote": "up"},
    )
    assert res.status_code == 200, res.text

    after = explore_client.get("/api/favorites?limit=99999").json()
    assert served_id in {r["id"] for r in after.get("favorites", [])}


def test_a_downvote_on_the_favourites_album_never_reaches_the_favourites(explore_client):
    """A down-vote stays session-scoped — it is not a favourite."""
    for point_id in (CAT_ID, DOG_ID, BIRD_ID):
        _favourite(explore_client, point_id)

    res = explore_client.get("/api/explore/favourites/round?limit=1")
    served_id = res.json()["results"][0]["id"]

    res = explore_client.post(
        "/api/explore/favourites/vote",
        json={"photo_id": served_id, "vote": "down"},
    )
    assert res.status_code == 200, res.text

    after = explore_client.get("/api/favorites?limit=99999").json()
    assert served_id not in {r["id"] for r in after.get("favorites", [])}


def test_an_unknown_explore_source_is_rejected(explore_client):
    """A source key that names nothing gets a 400, not a crash."""
    res = explore_client.get("/api/explore/nonsense/round")
    assert res.status_code == 400
    res = explore_client.post(
        "/api/explore/nonsense/vote",
        json={"photo_id": CAT_ID, "vote": "down"},
    )
    assert res.status_code == 400


def test_a_favourites_session_is_not_reusable_for_an_album(explore_client):
    """A session belongs to one source; it cannot be pointed at another."""
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    for point_id in (CAT_ID, DOG_ID, BIRD_ID):
        _favourite(explore_client, point_id)

    res = explore_client.get("/api/explore/favourites/round")
    session = res.json()["session_id"]

    res = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?session={session}"
    )
    assert res.status_code == 400


# ---------------- the round ----------------


def test_round_returns_results_for_an_album(explore_client):
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    res = explore_client.get(f"/api/explore/album%3A{album_id}/round")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["results"], "expected at least one result"
    assert body["session_id"]
    # The album's own members are excluded from the results — the
    # user is hunting for more like the album, not the album.
    served = {r["id"] for r in body["results"]}
    assert not (served & {CAT_ID, DOG_ID, BIRD_ID})


def test_round_excludes_the_albums_own_members(explore_client):
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    res = explore_client.get(f"/api/explore/album%3A{album_id}/round")
    served = {r["id"] for r in res.json()["results"]}
    assert CAT_ID not in served
    assert DOG_ID not in served
    assert BIRD_ID not in served


def test_round_rejects_an_empty_album(explore_client):
    album_id = _make_album(explore_client, [])
    res = explore_client.get(f"/api/explore/album%3A{album_id}/round")
    assert res.status_code == 400
    assert "no photos" in res.json()["detail"].lower()


def test_round_rejects_a_session_from_another_album(explore_client):
    album_a = _make_album(explore_client, [CAT_ID], name="Explore A")
    album_b = _make_album(explore_client, [DOG_ID], name="Explore B")
    first = explore_client.get(f"/api/explore/album%3A{album_a}/round")
    session = first.json()["session_id"]
    res = explore_client.get(
        f"/api/explore/album%3A{album_b}/round?session={session}"
    )
    assert res.status_code == 400
    assert "different album" in res.json()["detail"]


def test_round_pages_within_the_materialised_round(explore_client):
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    first = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?limit=1"
    ).json()
    assert first["session_id"]
    session = first["session_id"]
    assert first["has_more"] is True

    # The second page comes from the SAME materialised round — the
    # server slices the cached id list rather than re-querying
    # Qdrant with an offset (which would duplicate rows).
    second = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?session={session}&limit=1&offset=1"
    ).json()
    assert second["results"]
    assert second["results"][0]["id"] != first["results"][0]["id"]
    assert second["session_total"] == first["session_total"]


def test_round_rejects_a_limit_out_of_range(explore_client):
    album_id = _make_album(explore_client, [CAT_ID])
    res = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?limit=0"
    )
    assert res.status_code == 422


# ---------------- the vote ----------------


def test_downvote_is_session_scoped_and_not_a_dislike(explore_client):
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    round_res = explore_client.get(f"/api/explore/album%3A{album_id}/round")
    session = round_res.json()["session_id"]
    target = round_res.json()["results"][0]["id"]

    res = explore_client.post(
        f"/api/explore/album%3A{album_id}/vote?session={session}",
        json={"photo_id": target, "vote": "down"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["downvotes"] == 1

    # The permanent dislike table is untouched — an exploratory veto
    # is not "never show me this again".
    dislikes = explore_client.get("/api/dislikes").json()
    served_ids = {r["id"] for r in dislikes.get("results", [])}
    assert target not in served_ids


def test_downvote_becomes_a_negative_for_the_next_round(explore_client):
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    round_res = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?limit=2"
    )
    session = round_res.json()["session_id"]
    target = round_res.json()["results"][0]["id"]

    explore_client.post(
        f"/api/explore/album%3A{album_id}/vote?session={session}",
        json={"photo_id": target, "vote": "down"},
    )
    # The next round is rebuilt with the new negative, so the
    # down-voted photo is not served again.
    after = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?session={session}&limit=2"
    ).json()
    served = {r["id"] for r in after["results"]}
    assert target not in served


def test_a_round_is_not_served_twice(explore_client):
    """The seen-set keeps a photo from coming back in a later round."""
    album_id = _make_album(explore_client, [CAT_ID, DOG_ID, BIRD_ID])
    first = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?limit=3"
    ).json()
    session = first["session_id"]
    first_ids = {r["id"] for r in first["results"]}

    # A vote invalidates the round, so the next round is rebuilt.
    explore_client.post(
        f"/api/explore/album%3A{album_id}/vote?session={session}",
        json={"photo_id": next(iter(first_ids)), "vote": "down"},
    )
    second = explore_client.get(
        f"/api/explore/album%3A{album_id}/round?session={session}&limit=3"
    ).json()
    second_ids = {r["id"] for r in second["results"]}
    assert not (first_ids & second_ids)


def test_upvote_adds_the_photo_to_the_album(explore_client):
    album_id = _make_album(explore_client, [CAT_ID])
    round_res = explore_client.get(f"/api/explore/album%3A{album_id}/round")
    session = round_res.json()["session_id"]
    target = round_res.json()["results"][0]["id"]

    res = explore_client.post(
        f"/api/explore/album%3A{album_id}/vote?session={session}",
        json={"photo_id": target, "vote": "up"},
    )
    assert res.status_code == 200, res.text

    detail = explore_client.get(f"/api/albums/{album_id}").json()
    member_ids = {m["id"] for m in detail["members"]}
    assert target in member_ids


def test_vote_rejects_an_unknown_action(explore_client):
    album_id = _make_album(explore_client, [CAT_ID])
    res = explore_client.post(
        f"/api/explore/album%3A{album_id}/vote",
        json={"photo_id": CAT_ID, "vote": "sideways"},
    )
    assert res.status_code == 400
    assert "up" in res.json()["detail"]


def test_vote_requires_a_photo_id(explore_client):
    album_id = _make_album(explore_client, [CAT_ID])
    res = explore_client.post(
        f"/api/explore/album%3A{album_id}/vote",
        json={"vote": "down"},
    )
    assert res.status_code == 400
    assert "photo_id" in res.json()["detail"]


# ---------------- the liveness filter ----------------


def test_a_dead_album_member_does_not_break_the_round(explore_client):
    """A membership row can outlive its photo.

    `recommend()` is all-or-nothing on its seed ids — one missing id
    404s the whole call. The round filters dead ids before seeding,
    so one dead member must not take the feature down.
    """
    album_id = _make_album(explore_client, [CAT_ID, DEAD_ID])
    res = explore_client.get(f"/api/explore/album%3A{album_id}/round")
    assert res.status_code == 200, res.text
    assert res.json()["results"]
