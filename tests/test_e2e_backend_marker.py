"""The disposable-backend marker.

The frontend's destructive E2E tests (the ones that clear every
favourite and every user album to assert on an empty state) must
refuse to run against anything but a disposable backend. They decide
that by checking for the ``x-lustre-e2e-mode: 1`` header the backend
sets when it is started with ``LUSTRE_E2E_MODE=1``.

On 2026-11-09 the E2E suite was pointed at the production backend by
mistake and deleted all 21 of Isaac's user albums. These tests pin the
marker's behaviour so the guard can never silently stop working.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import search.app as app_mod
from search.config import Config


@pytest.fixture
def app_with_qdrant(qdrant_in_memory, nas_base):
    """A minimal app wired to an in-memory Qdrant."""
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

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as client:
        yield client
    app_mod.reset_for_tests()


def test_marker_present_in_e2e_mode(app_with_qdrant, monkeypatch):
    """With LUSTRE_E2E_MODE=1 every response carries the marker."""
    monkeypatch.setenv("LUSTRE_E2E_MODE", "1")
    res = app_with_qdrant.get("/api/albums?limit=5")
    assert res.status_code == 200
    assert res.headers.get("x-lustre-e2e-mode") == "1"


def test_marker_absent_without_e2e_mode(app_with_qdrant, monkeypatch):
    """Without the env var the backend announces nothing.

    This is the case that matters: a production backend must NOT look
    disposable, or the E2E guard would wave it through.
    """
    monkeypatch.delenv("LUSTRE_E2E_MODE", raising=False)
    res = app_with_qdrant.get("/api/albums?limit=5")
    assert res.status_code == 200
    assert res.headers.get("x-lustre-e2e-mode") is None


def test_marker_is_not_set_by_a_truthy_non_one_value(
    app_with_qdrant, monkeypatch
):
    """Only the exact value "1" counts.

    A stray LUSTRE_E2E_MODE=true or =yes must not enable it, so the
    marker cannot be switched on by accident.
    """
    for value in ("true", "yes", "0", ""):
        monkeypatch.setenv("LUSTRE_E2E_MODE", value)
        res = app_with_qdrant.get("/api/albums?limit=5")
        assert res.headers.get("x-lustre-e2e-mode") is None, (
            f"LUSTRE_E2E_MODE={value!r} must not enable the marker"
        )


def test_marker_present_on_error_responses(app_with_qdrant, monkeypatch):
    """The marker is added by middleware, so 404s carry it too."""
    monkeypatch.setenv("LUSTRE_E2E_MODE", "1")
    res = app_with_qdrant.get("/api/this-route-does-not-exist")
    assert res.status_code == 404
    assert res.headers.get("x-lustre-e2e-mode") == "1"


def test_e2e_mode_is_read_per_request(app_with_qdrant, monkeypatch):
    """The middleware reads the env var at request time, not at startup.

    A test runner that sets the variable after the app is built still
    gets the marker — and one that unsets it again stops getting it.
    """
    monkeypatch.delenv("LUSTRE_E2E_MODE", raising=False)
    res = app_with_qdrant.get("/api/albums?limit=5")
    assert res.headers.get("x-lustre-e2e-mode") is None

    monkeypatch.setenv("LUSTRE_E2E_MODE", "1")
    res = app_with_qdrant.get("/api/albums?limit=5")
    assert res.headers.get("x-lustre-e2e-mode") == "1"

    monkeypatch.delenv("LUSTRE_E2E_MODE", raising=False)
    res = app_with_qdrant.get("/api/albums?limit=5")
    assert res.headers.get("x-lustre-e2e-mode") is None


def test_env_var_is_not_leaked_into_the_openapi_schema(
    app_with_qdrant, monkeypatch
):
    """The marker is a response header, not a documented API field."""
    monkeypatch.setenv("LUSTRE_E2E_MODE", "1")
    res = app_with_qdrant.get("/openapi.json")
    assert res.status_code == 200
    assert "x-lustre-e2e-mode" not in res.text
