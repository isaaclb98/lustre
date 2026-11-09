"""
test_v2_smoke.py — comprehensive happy-path test exercising every
backend route the v2 frontend depends on.

Built on the real FastAPI app from `_build_demo_app`, so it sees
the same in-memory Qdrant + IndexDB setup the dev_server uses.
This catches regressions that pure stub tests miss: route
mapping, query-param binding, response shapes, and the
interaction between SearchResponse's diversity block and the
caller-side serialization.

Per the v2 testing strategy Layer 1 + 2: this is the behavioural
test layer; tests/test_openapi_stability.py pins the contract.

Run with:
    .venv-test/bin/python -m pytest tests/test_v2_smoke.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Set env BEFORE importing — matches dev_server invariants.
os.environ.setdefault("SEARCH_TEST_MODE", "1")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MODEL_NAME", "hf-hub:timm/ViT-SO400M-16-SigLIP2-384")
os.environ.setdefault("NAS_IMAGES_BASE", "/tmp/_is_v2_smoke_nas")  # noqa: S108

_REPO = Path(__file__).resolve().parent.parent
_NAS = Path("/tmp/_is_v2_smoke_nas")  # noqa: S108
_NAS.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def client():
    """Build the real FastAPI app with a 20-photo in-memory Qdrant.

    Imported as `search.dev_server` (NOT by adding `search/` to
    sys.path — that shadows the stdlib `random` module that
    `tempfile` uses for its `Random` import).
    """
    from search.dev_server import _build_demo_app

    app = _build_demo_app(count=20)
    return TestClient(app)


# ---------- health ----------

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200


# ---------- search ----------

def test_search_requires_prompts(client):
    r = client.get("/api/search")
    assert r.status_code in (400, 422)


def test_search_with_positive_returns_results(client):
    r = client.get("/api/search", params={"positives": ["beach"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "results" in body
    assert "took_ms" in body
    assert isinstance(body["results"], list)


def test_search_pagination_limit_and_offset(client):
    r1 = client.get(
        "/api/search",
        params={"positives": ["beach"], "limit": 3, "offset": 0}
    )
    r2 = client.get(
        "/api/search",
        params={"positives": ["beach"], "limit": 3, "offset": 3}
    )
    b1, b2 = r1.json(), r2.json()
    # If we have ≥6 results, the two pages' first IDs must differ.
    if len(b1["results"]) >= 3 and len(b2["results"]) >= 1:
        ids1 = {r["id"] for r in b1["results"]}
        ids2 = {r["id"] for r in b2["results"]}
        assert ids1.isdisjoint(ids2), (
            "Pagination overlap — page 2 contains page 1 ids"
        )


def test_search_response_shape_has_diversity_block(client):
    r = client.get(
        "/api/search",
        params={"positives": ["beach"], "diversity_mode": "auto"}
    )
    body = r.json()
    assert "diversity" in body
    assert "mode" in body["diversity"]
    assert "applied" in body["diversity"]


# ---------- random ----------

def test_random_returns_results(client):
    r = client.get("/api/random", params={"limit": 4})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) <= 4


def test_random_items_have_required_fields(client):
    r = client.get("/api/random", params={"limit": 2})
    body = r.json()
    for item in body["results"]:
        for field in ("id", "path", "score", "url"):
            assert field in item, f"Random item missing {field}"


# ---------- saved searches (the spec's MVP) ----------

def test_create_saved_search_then_list_then_delete(client):
    # create
    body = {
        "name": "v2-smoke-set",
        "positives": ["beach", "sunset"],
        "negatives": ["people"]
    }
    r = client.post("/api/saved-searches", json=body)
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["name"] == "v2-smoke-set"
    assert "positives" in created and "negatives" in created
    sid = created["id"]

    # list
    r = client.get("/api/saved-searches")
    assert r.status_code == 200
    listed = r.json()
    names = [s["name"] for s in listed["saved_searches"]]
    assert "v2-smoke-set" in names

    # delete (cleanup so this test is rerunnable)
    r = client.delete(f"/api/saved-searches/{sid}")
    assert r.status_code == 204


# ---------- albums ----------

def test_albums_listing_includes_demo_albums(client):
    """`_build_demo_app` seeds 2 demo albums; verify they're visible
    via the API (regression: cover_favorite_id is a string, not int)."""
    r = client.get("/api/albums")
    assert r.status_code == 200
    body = r.json()
    assert "albums" in body
    assert len(body["albums"]) >= 1
    for a in body["albums"]:
        assert "id" in a
        assert "name" in a
        assert "cover_favorite_id" in a


def test_album_detail_returns_members(client):
    r = client.get("/api/albums")
    assert r.status_code == 200
    aid = r.json()["albums"][0]["id"]
    r2 = client.get(f"/api/albums/{aid}")
    assert r2.status_code == 200
    body = r2.json()
    assert "members" in body
    assert isinstance(body["members"], list)


# ---------- photo bytes ----------

def test_photo_raw_serves_bytes(client):
    """Pick a known demo photo via /api/search and fetch its raw
    bytes through /photo/{id}/raw. Note: /api/random relies on the
    IndexDB cache which isn't populated by `_build_demo_app`, so
    we go through the search endpoint instead.
    """
    r = client.get("/api/search", params={"positives": ["beach"], "limit": 1})
    body = r.json()
    assert body["results"], "/api/search returned no results for 'beach'"
    pid = body["results"][0]["id"]
    raw = client.get(f"/photo/{pid}/raw")
    assert raw.status_code == 200
    bytes_body = raw.content
    assert len(bytes_body) > 100, f"photo too small: {len(bytes_body)} bytes"
