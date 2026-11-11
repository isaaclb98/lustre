"""
tests/test_theme.py — the LUSTRE_THEME contract.

The frontend is one static build shared by every deployment, so the
theme cannot be baked into it. Instead the backend stamps the
active theme onto the <html> element of the SPA shell it serves:

    <html lang="en" data-theme="dark">

and the SPA reads it back — CSS picks the palette off the same
attribute, and the grid geometry follows it in JS. Switching
`LUSTRE_THEME` is therefore a redeploy, not a rebuild.

These tests pin both halves of that contract:

  - the env var is validated strictly (a typo must fail loudly
    rather than silently shipping the wrong theme),
  - the shell the backend serves carries the marker.

The light theme is the default, and it must stay the default: an
unset LUSTRE_THEME has to produce the current look, byte for byte.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# The env var
# ---------------------------------------------------------------------------


def test_theme_defaults_to_light():
    """An unset LUSTRE_THEME is the light theme."""
    from search.config import load

    assert load().theme == "light"


def test_theme_accepts_dark():
    """LUSTRE_THEME=dark selects the dark theme."""
    from search.config import load

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LUSTRE_THEME", "dark")
        assert load().theme == "dark"


def test_theme_is_case_insensitive():
    """Operators type LUSTRE_THEME=Dark; that is not a typo."""
    from search.config import load

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LUSTRE_THEME", "Dark")
        assert load().theme == "dark"


def test_unknown_theme_is_a_startup_error():
    """An unrecognised theme must fail loudly, not fall back silently.

    A typo in the operator's .env would otherwise ship the wrong
    theme with no sign that anything went wrong.
    """
    from search.config import load

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LUSTRE_THEME", "midnight")
        with pytest.raises(ValueError, match="LUSTRE_THEME"):
            load()


# ---------------------------------------------------------------------------
# The marker on the served shell
# ---------------------------------------------------------------------------


@pytest.fixture
def shell_dir(tmp_path):
    """A minimal static dir with the SPA shell in it."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n  <head><title>lustre</title></head>\n'
        "  <body><div></div></body>\n</html>\n",
        encoding="utf-8",
    )
    return static


def _client(static_dir):
    """A TestClient over an app serving the given static dir."""
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from search.app import create_app, reset_for_tests
    from search.qdrant_client import QdrantSearch

    reset_for_tests()
    client = QdrantClient(location=":memory:")
    qdrant = QdrantSearch(client=client, collection="theme_test", timeout_ms=2000)
    return TestClient(create_app(qdrant=qdrant))


def test_served_shell_carries_the_dark_marker(shell_dir, monkeypatch):
    """The shell the backend serves carries data-theme="dark"."""
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.setenv("LUSTRE_THEME", "dark")

    with _client(shell_dir) as http:
        resp = http.get("/")

    assert resp.status_code == 200
    assert 'data-theme="dark"' in resp.text


def test_served_shell_carries_the_light_marker_by_default(shell_dir, monkeypatch):
    """The default theme is stamped too — the marker is always present.

    The SPA reads the marker to decide the palette AND the grid
    geometry, so "no marker" has to mean a definite theme rather
    than an absent value. Stamping 'light' makes the contract
    explicit: one attribute, always there.
    """
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.delenv("LUSTRE_THEME", raising=False)

    with _client(shell_dir) as http:
        resp = http.get("/")

    assert resp.status_code == 200
    assert 'data-theme="light"' in resp.text


def test_index_html_and_root_serve_the_same_marker(shell_dir, monkeypatch):
    """/ and /index.html are the same shell, so they carry the same marker."""
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.setenv("LUSTRE_THEME", "dark")

    with _client(shell_dir) as http:
        root = http.get("/")
        index = http.get("/index.html")

    assert root.status_code == index.status_code == 200
    assert 'data-theme="dark"' in root.text
    assert 'data-theme="dark"' in index.text


def test_marker_is_replaced_not_duplicated(shell_dir, monkeypatch):
    """A shell that already carries a marker gets exactly one.

    The static build is theme-neutral, but a shell that already
    declares a theme (a hand-edited one, or a build that stamped
    one) must not end up with two data-theme attributes — the
    browser would take the first and ignore the rest.
    """
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.setenv("LUSTRE_THEME", "dark")

    shell = shell_dir / "index.html"
    shell.write_text(
        '<!doctype html>\n<html lang="en" data-theme="light">\n'
        "  <head><title>lustre</title></head>\n"
        "  <body><div></div></body>\n</html>\n",
        encoding="utf-8",
    )

    with _client(shell_dir) as http:
        resp = http.get("/")

    assert resp.text.count("data-theme=") == 1
    assert 'data-theme="dark"' in resp.text


def test_shell_revalidates_so_a_theme_change_is_picked_up(shell_dir, monkeypatch):
    """The shell must not be cached stale across a theme change.

    The marker is baked into the shell's bytes, so a cached copy
    from before the switch would keep serving the old theme.
    """
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.setenv("LUSTRE_THEME", "dark")

    with _client(shell_dir) as http:
        resp = http.get("/")

    cc = resp.headers.get("cache-control", "").lower()
    assert "must-revalidate" in cc or "no-store" in cc
    assert resp.headers.get("etag")


def test_api_paths_are_not_swallowed_by_the_shell(shell_dir, monkeypatch):
    """The theme change must not affect the API 404 behaviour."""
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))

    with _client(shell_dir) as http:
        resp = http.get("/api/definitely-not-a-route")

    assert resp.status_code == 404
    assert "data-theme" not in resp.text
