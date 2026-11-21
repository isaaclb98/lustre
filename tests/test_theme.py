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


# ----- The masonry marker (the grid's layout mode) -----

def test_masonry_is_off_by_default():
    from search.config import load

    assert load().masonry is False


def test_masonry_follows_the_env_var(monkeypatch):
    from search.config import load

    monkeypatch.setenv("LUSTRE_MASONRY", "1")
    assert load().masonry is True
    monkeypatch.setenv("LUSTRE_MASONRY", "0")
    assert load().masonry is False


def test_the_thumbnail_size_is_published_on_the_shell():
    """The shell must carry the deployment's thumbnail size.

    The grid's `srcset` has to state the width the image IS, and the
    thumbnail's size is a deployment-level number (dev renders 512,
    production 384). The backend publishes it on the shell, the same
    way the theme and the layout mode travel.
    """
    from search.layout import inject_thumbnail_size

    html = b"<html lang=\"en\"><head></head></html>"
    stamped = inject_thumbnail_size(html, 512)
    assert b'data-thumbnail-size="512"' in stamped
    # The rest of the tag survives.
    assert b'<html lang="en"' in stamped


def test_the_thumbnail_size_replaces_rather_than_duplicates():
    from search.layout import inject_thumbnail_size

    html = b'<html lang="en" data-thumbnail-size="384">'
    stamped = inject_thumbnail_size(html, 512)
    assert stamped.count(b"data-thumbnail-size") == 1
    assert b'data-thumbnail-size="512"' in stamped


def test_the_thumbnail_size_follows_the_env_var(monkeypatch):
    from search.config import load

    monkeypatch.setenv("THUMBNAIL_SIZE", "512")
    assert load().thumbnail_size == 512
    monkeypatch.delenv("THUMBNAIL_SIZE")
    assert load().thumbnail_size == 384


def test_the_served_shell_carries_the_masonry_marker(shell_dir, monkeypatch):
    """LUSTRE_MASONRY=1 stamps the marker onto the shell.

    Same contract as the theme: the frontend is one static build
    shared by every deployment, so the layout mode rides on the
    bytes the shell is served as.
    """
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.setenv("LUSTRE_MASONRY", "1")

    with _client(shell_dir) as http:
        resp = http.get("/")

    assert resp.status_code == 200
    assert 'data-masonry="1"' in resp.text


def test_the_served_shell_has_no_masonry_marker_by_default(shell_dir, monkeypatch):
    """No marker means the uniform grid — the app's current look."""
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.delenv("LUSTRE_MASONRY", raising=False)

    with _client(shell_dir) as http:
        resp = http.get("/")

    assert resp.status_code == 200
    assert "data-masonry" not in resp.text


def test_the_masonry_marker_is_replaced_not_duplicated(shell_dir, monkeypatch):
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.setenv("LUSTRE_MASONRY", "1")

    with _client(shell_dir) as http:
        resp = http.get("/")

    assert resp.text.count("data-masonry") == 1


def test_the_masonry_marker_is_covered_by_the_etag(shell_dir, monkeypatch):
    """A shell cached under the square-grid ETag must not be served
    to a deployment that has since switched to masonry."""
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))
    monkeypatch.delenv("LUSTRE_MASONRY", raising=False)

    with _client(shell_dir) as http:
        square = http.get("/")

    monkeypatch.setenv("LUSTRE_MASONRY", "1")
    with _client(shell_dir) as http:
        masonry = http.get("/")

    assert square.headers["etag"] != masonry.headers["etag"]


def test_api_paths_are_not_swallowed_by_the_shell(shell_dir, monkeypatch):
    """The theme change must not affect the API 404 behaviour."""
    monkeypatch.setenv("FRONTEND_DIR", str(shell_dir))

    with _client(shell_dir) as http:
        resp = http.get("/api/definitely-not-a-route")

    assert resp.status_code == 404
    assert "data-theme" not in resp.text
