"""
tests/test_routers_thumbnails.py — thumbnail router contract.

Pins:
  - canonical `/thumb/{id}` returns the 256-px WebP
  - `/thumb/{id}?w=N` returns a pre-generated sized variant when on disk
  - `/thumb/{id}?w=N` falls back to the canonical 256-px file when the
    requested variant is missing (never 404 just because a sized
    sibling hasn't been generated yet — the browser will flock to the
    canonical if it asks for a width we never indexed)
  - `/thumb/{id}?w=N` returns 404 only when the canonical is also
    missing (true miss — frontend blurhash fallback)
  - `/thumb/{id}?w=N` rejects out-of-range widths with 422
  - validation: invalid point_id still 400s
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


def _build(thumb_dir: Path):
    """Build a minimal FastAPI app with the thumbnail router mounted and
    THUMBNAIL_DIR pointed at the supplied tmp directory."""
    import search.routers.thumbnails as router_mod
    router_mod.THUMBNAIL_DIR = str(thumb_dir)
    from search.routers.thumbnails import build_thumbnails_router
    app = FastAPI()
    app.include_router(build_thumbnails_router())
    return app


def _write_thumbnail(thumb_dir: Path, point_id: str, width: int = 384) -> Path:
    """Materialise a WebP thumbnail (and optionally a sized sibling)
    for the given point_id. Returns the canonical path.

    Post the model-variant migration plan, the canonical thumbnail
    is 384×384 (matches so400m input resolution). Tests that
    explicitly want the pre-migration 256 can pass `width=256`.
    """
    prefix = point_id[:2]
    (thumb_dir / prefix).mkdir(parents=True, exist_ok=True)
    canonical = thumb_dir / prefix / f"{point_id}.webp"
    Image.new("RGB", (width, width), color="red").save(
        canonical, "WEBP", quality=50
    )
    return canonical


@pytest.fixture
def thumb_dir(tmp_path, monkeypatch):
    """Pin THUMBNAIL_DIR at the env level so the endpoint picks up our
    tmp directory even if the module-level constant was bound before
    the test loaded."""
    monkeypatch.setenv("THUMBNAIL_DIR", str(tmp_path))
    # Module-level constant was already evaluated with the real env,
    # so re-patch it in the router module too.
    import search.routers.thumbnails as router_mod
    monkeypatch.setattr(router_mod, "THUMBNAIL_DIR", str(tmp_path))
    return tmp_path


# ----- canonical -----


class TestCanonicalThumbnail:
    def test_returns_webp_for_canonical(self, thumb_dir):
        _write_thumbnail(thumb_dir, "aabbccddeeff00112233445566778899")
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get("/thumb/aabbccddeeff00112233445566778899")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"
        # The response REVALIDATES rather than claiming immutability:
        # the thumbnail is addressed by a point id (a hash of the
        # photo's path), so a re-embed rewrites the file at the same
        # URL, and an immutable response would keep the old bytes.
        assert "immutable" not in resp.headers["cache-control"]
        assert "must-revalidate" in resp.headers["cache-control"]
        assert resp.headers["access-control-allow-origin"] == "*"

    def test_a_conditional_request_answers_304(self, thumb_dir):
        """A client holding the current bytes is told 304.

        The revalidating cache header only pays off if the server
        actually answers the conditional request — a plain Response
        does not do that by itself, so the route handles it.
        """
        point_id = "aabbccddeeff00112233445566778899"
        _write_thumbnail(thumb_dir, point_id)
        app = _build(thumb_dir)
        with TestClient(app) as client:
            first = client.get(f"/thumb/{point_id}")
            etag = first.headers["etag"]
            second = client.get(
                f"/thumb/{point_id}", headers={"If-None-Match": etag}
            )
        assert first.status_code == 200
        assert second.status_code == 304
        assert second.content == b""

    def test_the_etag_changes_when_the_thumbnail_is_rewritten(self, thumb_dir):
        """A regenerated thumbnail must not be served from a stale cache.

        The point id is a hash of the photo's PATH, so re-embedding
        the same photo rewrites the thumbnail at the same URL. The
        ETag has to change, or a client that revalidates would be
        told 304 and keep the old bytes.
        """
        point_id = "aabbccddeeff00112233445566778899"
        _write_thumbnail(thumb_dir, point_id)
        app = _build(thumb_dir)
        with TestClient(app) as client:
            first = client.get(f"/thumb/{point_id}")
            first_etag = first.headers["etag"]

            # Rewrite the thumbnail, as a re-embed would.
            time.sleep(0.01)
            _write_thumbnail(thumb_dir, point_id)

            second = client.get(f"/thumb/{point_id}")
            assert second.headers["etag"] != first_etag

    def test_returns_404_when_no_thumbnail(self, thumb_dir):
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get("/thumb/00000000000000000000000000000000")
        assert resp.status_code == 404


# ----- sized variants -----


class TestSizedVariants:
    def test_sized_variant_returned_when_present(self, thumb_dir):
        point_id = "11111111111111111111111111111111"
        _write_thumbnail(thumb_dir, point_id, 384)
        # Drop a 384-px sibling alongside. Post the model-variant
        # migration plan, the only pre-generated variant is 384
        # (matching so400m input resolution); pre-migration this
        # wrote a 240 sibling.
        Image.new("RGB", (384, 384), color="blue").save(
            thumb_dir / point_id[:2] / f"{point_id}.w384.webp", "WEBP", quality=50
        )
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get(f"/thumb/{point_id}?w=384")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"
        # Verify we got the *sized* file, not the canonical — the body
        # bytes should differ.
        sized_bytes = (thumb_dir / point_id[:2] / f"{point_id}.w384.webp").read_bytes()
        canonical_bytes = (thumb_dir / point_id[:2] / f"{point_id}.webp").read_bytes()
        assert resp.content == sized_bytes
        assert resp.content != canonical_bytes

    def test_falls_back_to_canonical_when_variant_missing(self, thumb_dir):
        """If ?w=120 is requested but no w120 sibling is on disk, the
        endpoint serves the canonical 256-px file (the browser then
        re-decodes to its CSS-pixel size). This keeps the bandwidth win
        gradual: pre-existing indexes work without any backfill."""
        point_id = "22222222222222222222222222222222"
        _write_thumbnail(thumb_dir, point_id, 256)
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get(f"/thumb/{point_id}?w=120")
        assert resp.status_code == 200
        canonical_bytes = (thumb_dir / point_id[:2] / f"{point_id}.webp").read_bytes()
        assert resp.content == canonical_bytes

    def test_404_when_neither_canonical_nor_variant(self, thumb_dir):
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get("/thumb/33333333333333333333333333333333?w=120")
        assert resp.status_code == 404


# ----- input validation -----


class TestValidation:
    def test_rejects_out_of_range_width_below_minimum(self, thumb_dir):
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get("/thumb/44444444444444444444444444444444?w=32")
        # FastAPI's Query(ge=64) returns 422, not 400.
        assert resp.status_code == 422

    def test_rejects_out_of_range_width_above_maximum(self, thumb_dir):
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get("/thumb/55555555555555555555555555555555?w=512")
        assert resp.status_code == 422

    def test_accepts_a_published_variant_wider_than_the_canonical(self, thumb_dir, monkeypatch):
        """A published variant is a legitimate width.

        The deployment advertises its variant ladder on the shell,
        and the browser asks for those widths. Capping the endpoint
        at the CANONICAL size rejected a variant with a 422 — and
        the browser treats a 422 as a broken image, so the tile went
        blank instead of falling back to the canonical.

        So the cap covers every width the deployment publishes.
        """
        monkeypatch.setenv("THUMBNAIL_SIZE", "384")
        monkeypatch.setenv("THUMBNAIL_VARIANT_SIZES", "240,480")
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get(
                "/thumb/66666666666666666666666666666666?w=480"
            )
        # NOT 422. The variant may not be on disk, in which case the
        # endpoint falls back to the canonical — but it must not
        # refuse the width outright.
        assert resp.status_code != 422
        assert resp.status_code in (200, 404)

    def test_still_rejects_a_width_beyond_every_published_size(self, thumb_dir, monkeypatch):
        """The cap still exists: a width no deployment publishes is
        still refused."""
        monkeypatch.setenv("THUMBNAIL_SIZE", "384")
        monkeypatch.setenv("THUMBNAIL_VARIANT_SIZES", "240,480")
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get(
                "/thumb/77777777777777777777777777777777?w=4096"
            )
        assert resp.status_code == 422

    def test_rejects_invalid_point_id(self, thumb_dir):
        """Existing behaviour: point_id must be 32-char hex (with or
        without hyphens). Shorter ids 400 — sanity check that we
        didn't break the existing validator."""
        app = _build(thumb_dir)
        with TestClient(app) as client:
            resp = client.get("/thumb/not-hex")
        assert resp.status_code == 400
