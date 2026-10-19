"""
test_openapi_stability.py — fast contracts that catch API drift.

Per docs/lustre-v2-testing.md (Layer 1 + Layer 2):
  - The frontend treats `openapi.json` as its compile-time and
    runtime source of truth (openapi-typescript + zod).
  - This module guarantees the backend surface the frontend
    depends on. Each test asserts a narrow, named contract.

Adding a new endpoint?  Add a corresponding test here so the
frontend has a contract to bind to. Removing or renaming? Update
or remove the matching test (and `npm run gen`).

These tests do NOT replace behavioural tests in test_search_api.py
etc. They are a tighter, faster sub-suite that runs as part of
the regular test loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Set the env vars BEFORE importing the app so config.load() sees them.
os.environ.setdefault("SEARCH_TEST_MODE", "1")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MODEL_NAME", "hf-hub:timm/ViT-gopt-16-SigLIP2-384")

from search.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_OPENAPI = REPO_ROOT / "frontend" / "openapi.json"


@pytest.fixture(scope="module")
def openapi_spec():
    """The frontend's pinned OpenAPI snapshot."""
    import json

    return json.loads(FRONTEND_OPENAPI.read_text())


@pytest.fixture(scope="module")
def live_spec():
    """The current backend's OpenAPI spec — produced by create_app."""
    return create_app().openapi()


# ---------- path surface ----------

EXPECTED_PATHS = {
    "/api/search",
    "/api/random",
    "/api/saved-searches",
    "/api/saved-searches/{saved_id}",
    "/api/albums",
    "/api/albums/{album_id}",
    "/api/favorites",
    "/api/favorites/{point_id}",
    "/api/for-you/feed",
    "/api/dislikes",
    "/api/dislikes/{point_id}",
    "/api/collections",
    "/api/centroids",
    "/api/centroids/{name}/search",
    "/photo/{point_id}/raw",
    "/healthz",
}


@pytest.mark.parametrize("path", sorted(EXPECTED_PATHS))
def test_expected_endpoint_present(openapi_spec, path):
    """Every endpoint the frontend relies on is present.

    A backend route rename that silently shifts a path goes
    undetected by request-level tests if no test happens to
    fire that endpoint. This test is explicit.
    """
    assert path in openapi_spec["paths"], (
        f"Frontend depends on {path!r} but it is missing from openapi.json. "
        "Add the route, run `npm run gen`, and update the frontend endpoints."
    )


# ---------- schema surface ----------

# Required schemas the frontend has TypeScript types for.
EXPECTED_SCHEMAS = {
    "SearchResult",
    "DiversityMetadata",
    "SearchResponse",
    "SavedSearch",
    "SavedSearchCreateRequest",
    "SavedSearchListResponse",
    "AlbumSummary",
    "AlbumsListResponse",
    "AlbumDetailResponse",
}


@pytest.mark.parametrize("schema", sorted(EXPECTED_SCHEMAS))
def test_expected_schema_present(openapi_spec, schema):
    assert schema in openapi_spec["components"]["schemas"], (
        f"Frontend depends on schema {schema!r} but it is missing."
    )


# ---------- SavedSearch field contract ----------
# The backend switched "prompts"/"negative_prompts" to
# "positives"/"negatives" recently. This test pins those names
# so the frontend never silently references the old fields.


def test_saved_search_has_positives_negatives(openapi_spec):
    schema = openapi_spec["components"]["schemas"]["SavedSearch"]
    props = set((schema.get("properties") or {}).keys())
    assert "positives" in props, "SavedSearch.positives is required by the frontend"
    assert "negatives" in props, "SavedSearch.negatives is required by the frontend"
    # The legacy field names must NOT appear — that would re-introduce drift.
    assert "prompts" not in props, (
        "SavedSearch.prompts is a legacy field. Frontend expects `positives`."
    )
    assert "negative_prompts" not in props, (
        "SavedSearch.negative_prompts is a legacy field. Frontend expects `negatives`."
    )


def test_saved_search_create_request_has_positives_negatives(openapi_spec):
    schema = openapi_spec["components"]["schemas"]["SavedSearchCreateRequest"]
    props = set((schema.get("properties") or {}).keys())
    assert "positives" in props
    assert "negatives" in props


# ---------- SearchResponse field contract ----------


def test_search_response_has_results_and_timing(openapi_spec):
    schema = openapi_spec["components"]["schemas"]["SearchResponse"]
    props = set((schema.get("properties") or {}).keys())
    for f in ("results", "took_ms", "limit"):
        assert f in props, f"SearchResponse.{f} is required by the frontend"


# ---------- live spec equals pinned snapshot ----------
# When this fails, the right answer is:
#   1. Run `npm run gen` to refresh the frontend snapshot.
#   2. If the change is intentional, commit BOTH the snapshot
#      and the regenerated types. Otherwise revert the route.


def test_live_spec_paths_match_pinned_snapshot(openapi_spec, live_spec):
    """Every path+method in the pinned snapshot must exist in live spec.

    We don't try to diff full JSON: the backend's parameter
    descriptions drift between releases in benign ways. The
    per-path-tests above already cover fields the frontend
    depends on. This is a coarse outer envelope.
    """
    expected_paths = set(openapi_spec["paths"].keys())
    actual_paths = set(live_spec["paths"].keys())
    missing = expected_paths - actual_paths
    extra = actual_paths - expected_paths  # noqa: F841
    assert not missing, f"These paths are pinned but no longer exist: {missing}"
    # extras are tolerated — they may be internal endpoints the
    # frontend never binds to.
    for path in expected_paths & actual_paths:
        exp = set(openapi_spec["paths"][path].keys())
        act = set(live_spec["paths"][path].keys())
        assert exp <= act, (
            f"{path} lost a method. expected {exp}, live has {act}"
        )
