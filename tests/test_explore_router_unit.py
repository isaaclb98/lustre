"""Unit tests for the Explore round query construction.

These pin the mechanic itself — what goes into the Qdrant
`recommend()` call as positives, negatives, and exclusions — against
a fake Qdrant that records every call. No real Qdrant, no HTTP.

The contract:

- POSITIVES are the album's member photos — LIVE, so an up-vote
  (which joins the album) counts in the next round.
- NEGATIVES are the session's down-votes.
- EXCLUSIONS are the photos the user ACTED ON plus the album's own
  members, so a judged photo never comes back and the album does
  not fill the results.
- A VOTE never redraws the round — the photos the user is looking
  at stay put. Only an explicit advance draws the next one.
- Dead member ids are filtered before they reach `recommend()`,
  which 404s the whole call when any seed id is missing.
- The HNSW breadth is pinned, so the round's latency is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from search.explore import EXPLORE_ROUND_SIZE
from search.routers.explore import HNSW_EF, build_explore_router


@dataclass
class _FakeHit:
    id: str
    score: float = 0.9
    path: str = "/photos/x.jpg"
    payload: dict[str, Any] = field(default_factory=dict)


class _FakeQdrant:
    """Records every recommend() call and returns a canned pool."""

    def __init__(self, pool: list[str] | None = None):
        self.pool = list(pool or [])
        self.recommend_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[list[str]] = []

    def recommend(
        self,
        *,
        positive: list[str],
        negative: list[str],
        limit: int,
        exclude_ids: list[str] | None = None,
        with_payload: bool = True,
        search_params: Any | None = None,
    ) -> list[_FakeHit]:
        self.recommend_calls.append(
            {
                "positive": list(positive),
                "negative": list(negative),
                "limit": limit,
                "exclude_ids": list(exclude_ids or []),
                "with_payload": with_payload,
                "search_params": search_params,
            }
        )
        excluded = set(exclude_ids or [])
        # The score is the rank-based similarity the real recommend()
        # returns: the best hit scores highest and the rest descend
        # from there, counted over the hits the call actually
        # returned.
        kept = [pid for pid in self.pool if pid not in excluded]
        hits = [
            _FakeHit(id=pid, score=0.9 - i * 0.01)
            for i, pid in enumerate(kept)
        ]
        return hits[:limit]

    def retrieve_batch(self, point_ids: list[str]) -> list[_FakeHit]:
        """Mirrors the real wrapper: unknown ids are skipped, and the
        hits carry NO score.

        The real `QdrantSearch.retrieve_batch` builds each hit with
        `score=0.0` — a retrieve is a lookup, not a similarity query.
        The fake has to reproduce that, or a test that checks the
        round's scores would pass against a fake that behaves nothing
        like the real client.
        """
        self.retrieve_calls.append(list(point_ids))
        known = set(self.pool)
        return [
            _FakeHit(id=pid, score=0.0)
            for pid in point_ids
            if pid in known
        ]


class _FakeIndexDB:
    """Minimal IndexDB surface for the router."""

    def __init__(self, members: dict[int, list[str]] | None = None):
        self.members = members or {}
        self.added: list[tuple[int, str]] = []
        self.favorites: set[str] = set()
        self.dislikes: set[str] = set()

    def list_album_member_ids(self, album_id: int) -> list[str]:
        return list(self.members.get(album_id, []))

    def add_album_member(self, album_id: int, favorite_id: str) -> bool:
        self.added.append((album_id, favorite_id))
        self.members.setdefault(album_id, []).append(favorite_id)
        return True

    def favorite_id_set(self, ids: list[str]) -> set[str]:
        return {i for i in ids if i in self.favorites}

    def dislike_id_set(self, ids: list[str]) -> set[str]:
        return {i for i in ids if i in self.dislikes}


class _FakeCfg:
    web_ui_url = "http://localhost:8000"


def _make_router(index_db, qdrant, invalidated: list[int] | None = None):
    router = build_explore_router(
        index_db=index_db,
        qdrant=qdrant,
        cfg=_FakeCfg(),
        invalidate_album_centroid=(
            (lambda album_id: invalidated.append(album_id))
            if invalidated is not None
            else None
        ),
    )
    return router


def _client(router):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_the_album_members_are_the_positives():
    index_db = _FakeIndexDB({1: ["a", "b", "c"]})
    qdrant = _FakeQdrant(pool=["a", "b", "c", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    res = client.get("/api/explore/album%3A1/round")
    assert res.status_code == 200, res.text
    call = qdrant.recommend_calls[0]
    assert call["positive"] == ["a", "b", "c"]
    assert call["negative"] == []


def test_the_downvotes_become_the_negatives():
    index_db = _FakeIndexDB({1: ["a", "b"]})
    qdrant = _FakeQdrant(pool=["a", "b", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    first = client.get("/api/explore/album%3A1/round").json()
    session = first["session_id"]
    target = first["results"][0]["id"]
    client.post(
        f"/api/explore/album%3A1/vote?session={session}",
        json={"photo_id": target, "vote": "down"},
    )
    client.get(f"/api/explore/album%3A1/round?session={session}&advance=true")

    second_call = qdrant.recommend_calls[1]
    assert target in second_call["negative"]
    assert second_call["negative"] == [target]


def test_the_seen_set_and_the_album_are_excluded():
    index_db = _FakeIndexDB({1: ["a", "b"]})
    qdrant = _FakeQdrant(pool=["a", "b", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    client.get("/api/explore/album%3A1/round?limit=2")
    call = qdrant.recommend_calls[0]
    excluded = set(call["exclude_ids"])
    # The album's own members never appear in the results — the
    # user is hunting for more like the album, not the album.
    assert {"a", "b"} <= excluded


def test_only_the_photos_the_user_judged_are_excluded():
    """The seen-set is the acted-on set, not the served set.

    Serving a photo is not a verdict on it. A photo the user
    scrolled past without voting on stays a candidate for a later
    round; the ones they judged never come back.
    """
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    first = client.get("/api/explore/album%3A1/round?limit=2").json()
    session = first["session_id"]
    served = [r["id"] for r in first["results"]]

    # The user judges one of the two they were shown.
    client.post(
        f"/api/explore/album%3A1/vote?session={session}",
        json={"photo_id": served[0], "vote": "down"},
    )
    client.get(f"/api/explore/album%3A1/round?session={session}&advance=true")

    excluded = set(qdrant.recommend_calls[1]["exclude_ids"])
    # The judged photo is excluded — it must never come back.
    assert served[0] in excluded
    # The one they merely looked at is still a candidate.
    assert served[1] not in excluded


def test_an_upvoted_photo_is_excluded_from_the_next_round():
    """An up-vote joins the album, so it is a member — and members
    are excluded from the results."""
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    first = client.get("/api/explore/album%3A1/round").json()
    session = first["session_id"]
    target = first["results"][0]["id"]

    client.post(
        f"/api/explore/album%3A1/vote?session={session}",
        json={"photo_id": target, "vote": "up"},
    )
    client.get(f"/api/explore/album%3A1/round?session={session}&advance=true")

    call = qdrant.recommend_calls[1]
    # It joined the album, so it is a positive of the next round...
    assert target in call["positive"]
    # ...and a member, so it is excluded from the results.
    assert target in set(call["exclude_ids"])


def test_dead_member_ids_are_filtered_before_the_recommend():
    """One dead member must not 404 the whole recommend call.

    `recommend()` is all-or-nothing on its seed ids. A membership row
    can outlive its photo, so the dead id has to be dropped before
    it is used as a seed.
    """
    index_db = _FakeIndexDB({1: ["a", "b", "dead"]})
    # The fake Qdrant only knows about "a" and "b".
    qdrant = _FakeQdrant(pool=["a", "b", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    res = client.get("/api/explore/album%3A1/round")
    assert res.status_code == 200, res.text
    call = qdrant.recommend_calls[0]
    assert call["positive"] == ["a", "b"]
    assert "dead" not in call["positive"]


def test_dead_downvotes_are_filtered_too():
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    first = client.get("/api/explore/album%3A1/round").json()
    session = first["session_id"]
    # Down-vote a photo that is not in the collection.
    client.post(
        f"/api/explore/album%3A1/vote?session={session}",
        json={"photo_id": "ghost", "vote": "down"},
    )
    client.get(f"/api/explore/album%3A1/round?session={session}&advance=true")
    call = qdrant.recommend_calls[1]
    assert call["negative"] == []


def test_the_round_asks_for_the_configured_round_size():
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    client.get("/api/explore/album%3A1/round")
    assert qdrant.recommend_calls[0]["limit"] == EXPLORE_ROUND_SIZE
    assert EXPLORE_ROUND_SIZE == 240


def test_the_hnsw_breadth_is_pinned():
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    client.get("/api/explore/album%3A1/round")
    params = qdrant.recommend_calls[0]["search_params"]
    assert params is not None
    assert params.hnsw_ef == HNSW_EF
    assert HNSW_EF == 256


def test_an_upvote_adds_to_the_album_and_invalidates_the_centroid():
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    invalidated: list[int] = []
    client = _client(_make_router(index_db, qdrant, invalidated))

    first = client.get("/api/explore/album%3A1/round").json()
    session = first["session_id"]
    target = first["results"][0]["id"]
    res = client.post(
        f"/api/explore/album%3A1/vote?session={session}",
        json={"photo_id": target, "vote": "up"},
    )
    assert res.status_code == 200, res.text
    assert (1, target) in index_db.added
    # The album's membership changed, so its cached centroid is stale.
    assert invalidated == [1]


def test_the_round_is_reused_until_the_client_advances():
    """The round survives votes. Only an advance draws the next one.

    The photos the user is looking at must not change under them
    while they vote — so a vote records the feedback and leaves the
    materialised round exactly where it is.
    """
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    first = client.get("/api/explore/album%3A1/round").json()
    session = first["session_id"]
    # Paging within the round does NOT re-query Qdrant — the round
    # is materialised once and sliced.
    client.get(f"/api/explore/album%3A1/round?session={session}&offset=1&limit=1")
    assert len(qdrant.recommend_calls) == 1

    # A vote does not invalidate it either.
    client.post(
        f"/api/explore/album%3A1/vote?session={session}",
        json={"photo_id": "x", "vote": "down"},
    )
    client.get(f"/api/explore/album%3A1/round?session={session}")
    assert len(qdrant.recommend_calls) == 1

    # Only an explicit advance draws the next round.
    client.get(f"/api/explore/album%3A1/round?session={session}&advance=true")
    assert len(qdrant.recommend_calls) == 2


def test_the_round_scores_survive_the_page_hydration():
    """The tiles must show the score the round earned, not 0.

    The page's payloads are fetched with `retrieve_batch`, which
    returns no score — every hit it produces has a score of 0. The
    round's own `recommend()` call is the only source of the
    similarity, so the score has to be carried across the
    hydration. Without that, every tile in the Explore grid displays
    "0.000".
    """
    index_db = _FakeIndexDB({1: ["a"]})
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z"])
    client = _client(_make_router(index_db, qdrant))

    res = client.get("/api/explore/album%3A1/round?limit=3")
    assert res.status_code == 200, res.text
    body = res.json()
    results = body["results"]
    assert results

    # The fake's recommend() hands out 0.90, 0.89, 0.88, 0.87 in
    # pool order; the album's own member "a" is excluded, so the
    # first three results are 0.90, 0.89 and 0.88.
    assert [r["score"] for r in results] == [0.9, 0.89, 0.88]
    assert all(r["score"] > 0 for r in results)
    assert [r["score_str"] for r in results] == ["0.900", "0.890", "0.880"]

    # And the page's payload fetch really did go through the
    # score-less path — otherwise this test would pass vacuously.
    assert qdrant.retrieve_calls, "expected a payload hydration call"


def test_a_paged_result_keeps_its_round_score():
    """The score belongs to the round, so page 2 carries page 2's
    scores — not page 1's, and not zeros."""
    index_db = _FakeIndexDB({1: ["a"]})
    # Five non-member ids, so the round is long enough to page.
    qdrant = _FakeQdrant(pool=["a", "x", "y", "z", "w", "v"])
    client = _client(_make_router(index_db, qdrant))

    first = client.get("/api/explore/album%3A1/round?limit=2").json()
    session = first["session_id"]
    second = client.get(
        f"/api/explore/album%3A1/round?session={session}&limit=2&offset=2"
    ).json()

    # The album's own member "a" is excluded, so the round's five
    # results score 0.90 down to 0.86 in rank order.
    assert [r["score"] for r in first["results"]] == [0.9, 0.89]
    assert [r["score"] for r in second["results"]] == [0.88, 0.87]
    assert all(r["score"] > 0 for r in second["results"])
