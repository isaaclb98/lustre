"""
tests/test_thumbnails_unit.py — Unit tests for indexer/thumbnails.py.

Thumbnail generation: 384×384 WebP q50, stored at
{THUMBNAIL_DIR}/{prefix}/{point_id}.webp with a 2-char prefix
shard to avoid one-directory-per-2M-files.

Critical for the indexer — without thumbnails, /thumb/{id} returns
404 and the gallery breaks. This is the exact module that had the
generation-skip bug I found earlier (local_sync.py never called it).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest  # noqa: F401  # used by tests that conditionally skip
from PIL import Image

from indexer.thumbnails import (
    THUMBNAIL_DIR,
    THUMBNAIL_QUALITY,
    THUMBNAIL_SIZE,
    THUMBNAIL_SIZES,
    CHROMA_SUBSAMPLING,
    _HAS_PYVIPS,
    _PIL_FORCE,
    _SRGB_PROFILE_BYTES,
    _use_pyvips,
    compute_thumbnail,
    generate_thumbnail_for_path,
    sized_thumbnail_path,
    thumbnail_path,
)


# ----- thumbnail_path -----

class TestThumbnailPath:
    """The path layout is deterministic from the point_id."""

    def test_basic_path_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        p = thumbnail_path("abc123def456")
        assert p == tmp_path / "ab" / "abc123def456.webp"
        assert p.suffix == ".webp"

    def test_prefix_is_first_two_chars(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        assert thumbnail_path("abcdef").parent.name == "ab"
        assert thumbnail_path("0123456").parent.name == "01"
        assert thumbnail_path("zz_yyyy").parent.name == "zz"

    def test_id_with_dash_or_underscore_works(self, tmp_path, monkeypatch):
        """Common id characters (dash, underscore) pass through cleanly."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        p = thumbnail_path("abc-def_ghi")
        assert p.name == "abc-def_ghi.webp"

    def test_uses_env_var(self, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", "/custom/thumb/dir")
        p = thumbnail_path("abc123")
        assert str(p).startswith("/custom/thumb/dir")


# ----- compute_thumbnail -----

class TestComputeThumbnail:
    """Generate a WebP thumbnail from a PIL Image."""

    def test_writes_webp_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (1024, 768), color="blue")
        result = compute_thumbnail(img, "point-abc-123")
        assert result is not None
        assert result.exists()
        assert result.suffix == ".webp"

    def test_creates_prefix_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (100, 100), color="red")
        compute_thumbnail(img, "abc123")
        # The "ab" prefix directory should have been created
        assert (tmp_path / "ab").is_dir()

    def test_returns_path_matching_thumbnail_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (100, 100), color="red")
        result = compute_thumbnail(img, "abc123")
        assert result == thumbnail_path("abc123")

    def test_thumbnail_keeps_the_source_aspect_ratio(self, tmp_path, monkeypatch):
        """Output is native aspect: the long edge is capped, the short
        edge follows the source. No crop, no squash, no letterbox."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (1024, 768), color="green")
        result = compute_thumbnail(img, "abc123")
        with Image.open(result) as thumb:
            assert thumb.size == (384, 288)

    def test_smaller_image_is_not_upscaled(self, tmp_path, monkeypatch):
        """A source already within the cap keeps its dimensions."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (64, 64), color="red")
        result = compute_thumbnail(img, "small123")
        with Image.open(result) as thumb:
            assert thumb.size == (64, 64)

    def test_square_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (512, 512), color="red")
        result = compute_thumbnail(img, "square12")
        with Image.open(result) as thumb:
            assert thumb.size == (384, 384)

    def test_portrait_image_stays_portrait(self, tmp_path, monkeypatch):
        """Tall portrait: the long edge is capped, the width follows."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (400, 800), color="red")
        result = compute_thumbnail(img, "tall1234")
        with Image.open(result) as thumb:
            assert thumb.size == (192, 384)

    def test_landscape_image_stays_landscape(self, tmp_path, monkeypatch):
        """Wide landscape: the long edge is capped, the height follows."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (800, 400), color="red")
        result = compute_thumbnail(img, "wide1234")
        with Image.open(result) as thumb:
            assert thumb.size == (384, 192)

    def test_no_crop_the_whole_picture_survives(self, tmp_path, monkeypatch):
        """The thumbnail is the whole picture, not a centre crop.

        A red band across the TOP of the source must still be at the
        top of the thumbnail -- under the old square centre-crop it
        was cropped away entirely.
        """
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (400, 800), color="blue")
        img.paste(Image.new("RGB", (400, 100), color="red"), (0, 0))
        result = compute_thumbnail(img, "crop123")
        with Image.open(result) as thumb:
            assert thumb.size == (192, 384)
            top = thumb.convert("RGB").crop((0, 0, 192, 20))
            bottom = thumb.convert("RGB").crop((0, 364, 192, 384))
            assert top.getpixel((96, 10))[0] > 150, "the red top band was cropped away"
            assert bottom.getpixel((96, 10))[2] > 150, "the blue bottom was cropped away"

    def test_nothing_is_cropped_away(self, tmp_path, monkeypatch):
        """The whole picture survives — nothing is cropped.

        A red band across the middle of a portrait source must still
        be in the middle of the thumbnail. The old square centre-crop
        would have kept it but thrown away the top and bottom.
        """
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (100, 400), color="blue")
        # Paint the center column red
        for y in range(150, 250):
            for x in range(100):
                img.putpixel((x, y), (255, 0, 0))
        result = compute_thumbnail(img, "center12")
        with Image.open(result) as thumb:
            # Portrait source -> portrait thumbnail, long edge capped.
            assert thumb.size == (96, 384)
            # The red band is still at the vertical middle.
            r, g, b = thumb.getpixel((48, 192))
            assert r > g  # red dominates over blue
            # And the top of the picture is the source's top (blue).
            r, g, b = thumb.getpixel((48, 4))
            assert b > r

    def test_rgba_image_converted(self, tmp_path, monkeypatch):
        """RGBA images should still produce valid WebP output."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        result = compute_thumbnail(img, "rgba1234")
        assert result.exists()
        with Image.open(result) as thumb:
            assert thumb.format == "WEBP"

    def test_grayscale_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("L", (100, 100), color=128)
        result = compute_thumbnail(img, "gray12345")
        assert result.exists()

    def test_does_not_mutate_input(self, tmp_path, monkeypatch):
        """The function should work on a copy, not mutate the caller's image."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (1024, 768), color="red")
        original_size = img.size
        compute_thumbnail(img, "naming1")
        assert img.size == original_size

    def test_overwrites_existing_thumbnail(self, tmp_path, monkeypatch):
        """Re-running for the same point_id should overwrite, not error."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        result1 = compute_thumbnail(img1, "same12345")
        result2 = compute_thumbnail(img2, "same12345")
        assert result1 == result2
        assert result2.exists()

    def test_failure_returns_none(self, tmp_path, monkeypatch):
        """If save fails, return None and don't crash."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        # Use a mock image whose .save() raises
        bad_img = MagicMock()
        bad_img.copy.return_value.copy.return_value.thumbnail.side_effect = None
        bad_img.copy.return_value.thumbnail.side_effect = OSError("disk full")

        result = compute_thumbnail(bad_img, "fail1234")
        assert result is None

    def test_failure_logs_warning(self, tmp_path, monkeypatch, caplog):
        """Failed generation should log a warning with the point_id."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        bad_img = MagicMock()
        bad_img.copy.return_value.thumbnail.side_effect = OSError("disk full")

        import logging
        with caplog.at_level(logging.WARNING, logger="indexer.thumbnails"):
            compute_thumbnail(bad_img, "fail1234")

        assert any("fail1234" in r.message for r in caplog.records)


# ----- generate_thumbnail_for_path -----

class TestGenerateThumbnailForPath:
    """Generate thumbnail from a source image path (used by local_sync)."""

    def test_basic_generation(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        # Create a source image
        src = tmp_path / "source.jpg"
        Image.new("RGB", (200, 200), color="red").save(src)
        # Point id should be derived from the path
        result = generate_thumbnail_for_path(
            Image.open(src),
            src,
            shard="",
        )
        assert result is not None
        assert result.exists()
        assert result.suffix == ".webp"

    def test_with_shard(self, tmp_path, monkeypatch):
        """Different shards should produce different point_ids for the same path."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        src = tmp_path / "photo.jpg"
        Image.new("RGB", (100, 100), color="red").save(src)

        result1 = generate_thumbnail_for_path(Image.open(src), src, shard="")
        result2 = generate_thumbnail_for_path(Image.open(src), src, shard="shard-a")

        # Different shards → different file paths
        assert result1 != result2
        assert result1.exists()
        assert result2.exists()

    def test_id_matches_upsert_logic(self, tmp_path, monkeypatch):
        """The point_id used should match what upsert.id_for produces."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        from indexer.upsert import id_for

        src = tmp_path / "photo.jpg"
        Image.new("RGB", (100, 100), color="red").save(src)

        result = generate_thumbnail_for_path(
            Image.open(src), src, shard="test-shard"
        )

        # The thumbnail file name should be the id_for(...) of this path
        expected_id = id_for(src, "test-shard")
        assert result.name == f"{expected_id}.webp"


# ----- Module constants -----

class TestThumbnailConstants:
    """Constants are stable (referenced from indexer/local_sync.py)."""

    def test_thumbnail_size_is_384(self):
        # Post the model-variant migration plan, the thumbnail is
        # 384×384 (matching the so400m model input resolution). Pre-
        # migration this was 256×256.
        assert THUMBNAIL_SIZE == 384

    def test_thumbnail_target_size_caps_the_long_edge(self):
        """The sizing policy: long edge capped, aspect preserved."""
        from indexer.thumbnails import thumbnail_target_size

        # Portrait: the long edge is the height.
        assert thumbnail_target_size(1080, 1350) == (307, 384)
        # Landscape: the long edge is the width.
        assert thumbnail_target_size(1920, 1080) == (384, 216)
        # Square.
        assert thumbnail_target_size(800, 800) == (384, 384)
        # Already within the cap -> untouched.
        assert thumbnail_target_size(200, 100) == (200, 100)
        # An explicit size overrides the default.
        assert thumbnail_target_size(1080, 1350, 500) == (400, 500)
        # Degenerate input.
        assert thumbnail_target_size(0, 0) == (384, 384)

    def test_thumbnail_quality_is_80(self):
        # Post the 256→384 / q50→q80 thumbnail pipeline rework
        # (commit 44bac8c deleted docs/archive/thumbnail-pipeline.md
        # which had argued for q50). Current prod code uses q80.
        assert THUMBNAIL_QUALITY == 80

    def test_chroma_subsampling_is_4_4_4(self):
        # Forced 4:4:4 chroma for the libvips/Immich parity reason:
        # PIL's default at q<95 is 4:2:0, which subsamples the
        # chroma channels and mutes saturated color edges. Pin
        # the value so a future maintainer doesn't regress to
        # the PIL default.
        assert CHROMA_SUBSAMPLING == "4:4:4"

    def test_srgb_profile_is_embedded(self, tmp_path, monkeypatch):
        """The output WebP must carry the sRGB ICC profile so color-
        managed browsers and tools render with explicit sRGB
        intent. PIL strips ICC profiles on crop/resize by default;
        this test pins the embed behavior."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (256, 256), color="red")
        result = compute_thumbnail(img, "iccembed1")
        assert result is not None
        with Image.open(result) as thumb:
            assert "icc_profile" in thumb.info
            # Same 588-byte sRGB blob the module cached at load.
            assert thumb.info["icc_profile"] == _SRGB_PROFILE_BYTES

    def test_linear_light_resize_preserves_extremes(self, tmp_path, monkeypatch):
        """Linear-light resize is mathematically correct but the
        gamma round-trip on solid colors must NOT shift them. A
        solid-red 256x256 should round-trip to solid red (255,0,0)
        within 1 unit per channel — anything more means the
        gamma conversion or the uint8 clip is broken."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        # 2x2 solid red, upscaled to a real photo-like size so the
        # resize has meaningful content to work with.
        src = Image.new("RGB", (2, 2), color=(255, 0, 0))
        src = src.resize((512, 512), Image.Resampling.NEAREST)
        result = compute_thumbnail(src, "linrsz001")
        assert result is not None
        with Image.open(result) as thumb:
            # Sample the center — should still be solid red.
            r, g, b = thumb.getpixel((192, 192))
            assert r == 255
            assert g <= 1
            assert b <= 1

    def test_thumbnail_dir_uses_env_or_default(self, monkeypatch):
        monkeypatch.delenv("THUMBNAIL_DIR", raising=False)
        # Default is /app/data/thumbnails (container path)
        assert THUMBNAIL_DIR == "/app/data/thumbnails" or "thumbnail" in THUMBNAIL_DIR


# ----- Round-perf (issue #2): sized variants for the frontend srcset -----


class TestSizedThumbnailPath:
    """sized_thumbnail_path() mirrors the look-up the search router does
    when the frontend asks for `?w={width}`. The path layout MUST stay
    in sync or every srcset request 404s."""

    def test_path_uses_prefix_bucket(self):
        p = sized_thumbnail_path("abc123def456", 240)
        assert p.parent.name == "ab"

    def test_path_includes_width(self):
        p = sized_thumbnail_path("abc123def456", 360)
        assert p.name == "abc123def456.w360.webp"

    def test_different_widths_produce_different_files(self):
        a = sized_thumbnail_path("abc123", 240)
        b = sized_thumbnail_path("abc123", 480)
        assert a != b


class TestComputeThumbnailWritesSizedVariants:
    """compute_thumbnail() should write the downscaled siblings alongside
    the canonical 256px file so the frontend srcset can pick the smallest
    variant that fits the rendered tile."""

    def test_writes_all_sized_variants(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (1024, 1024), color="purple")
        compute_thumbnail(img, "varsize1")
        for w in THUMBNAIL_SIZES:
            assert sized_thumbnail_path("varsize1", w).exists(), f"missing w={w}"

    def test_sized_variants_are_smaller_than_canonical(self, tmp_path, monkeypatch):
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (1024, 1024), color="orange")
        compute_thumbnail(img, "sizechk1")
        canonical = thumbnail_path("sizechk1")
        with Image.open(canonical) as t:
            canonical_size = t.size[0]
        for w in THUMBNAIL_SIZES:
            assert w < canonical_size, f"variant w={w} should be < canonical {canonical_size}"
            with Image.open(sized_thumbnail_path("sizechk1", w)) as v:
                assert v.size == (w, w)

    def test_sized_variants_are_skipped_if_larger_than_source(self, tmp_path, monkeypatch):
        """If the canonical 256px source is itself smaller than a sized
        variant, we'd be upscaling — that's worse than the canonical
        file. Skip those variants and the endpoint's 404-fallback serves
        the canonical instead."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        # Patch the variant list to include an "impossible" 999-px variant
        # alongside a small one, then check only the small one is written.
        monkeypatch.setattr(
            "indexer.thumbnails.THUMBNAIL_SIZES", (200, 999)
        )
        img = Image.new("RGB", (1024, 1024), color="blue")
        compute_thumbnail(img, "skipbig1")
        assert sized_thumbnail_path("skipbig1", 200).exists()
        assert not sized_thumbnail_path("skipbig1", 999).exists()


# ----- pyvips (libvips) path -----


class TestPyvipsPath:
    """The libvips path runs when pyvips is importable AND a source
    path is provided. Behavior contract:
      - writes a 384x384 WebP at the canonical path
      - embeds the sRGB ICC profile
      - Q=80 default
    Skipped on hosts that don't have pyvips (most CI runners).
    """

    def test_use_pyvips_reflects_availability(self, monkeypatch):
        # Default: pyvips decides. monkeypatch _PIL_FORCE True → False.
        monkeypatch.setattr("indexer.thumbnails._PIL_FORCE", True)
        assert _use_pyvips() is False
        monkeypatch.setattr("indexer.thumbnails._PIL_FORCE", False)
        # Result depends on whether pyvips is importable; just
        # check that the function respects the toggle.
        assert _use_pyvips() is (_HAS_PYVIPS and not _PIL_FORCE)

    def test_pyvips_path_runs_when_source_path_provided(
        self, tmp_path, monkeypatch
    ):
        if not _HAS_PYVIPS:
            import pytest

            pytest.skip("pyvips not importable on this host")
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        # Write a real source JPEG to disk so the pyvips path
        # has something to load.
        src = tmp_path / "src.jpg"
        Image.new("RGB", (640, 480), color=(180, 60, 220)).save(
            src, "JPEG", quality=85
        )
        # Caller has both a PIL Image (for the PIL path) and
        # a source path (for the pyvips path); the path wins.
        pil_image = Image.new("RGB", (16, 16), color="red")
        result = compute_thumbnail(pil_image, "vipstest1", source_path=src)
        assert result is not None
        assert result.exists()
        with Image.open(result) as thumb:
            # The source is 640x480; the long edge is capped at 384
            # and the aspect ratio is preserved.
            assert thumb.size == (384, 288)
            assert thumb.format == "WEBP"
            # libvips auto-4:4:4 + ICC embed: the file MUST
            # carry the sRGB profile (matches what the PIL
            # path embeds; the load + reload test below proves
            # the bytes round-trip).
            assert "icc_profile" in thumb.info
            assert thumb.info["icc_profile"] == _SRGB_PROFILE_BYTES

    def test_pil_path_runs_when_source_path_missing(
        self, tmp_path, monkeypatch
    ):
        """No source_path → PIL fallback, even when pyvips is
        available. The indexer always has a source path so this
        path is for unit tests and other in-memory callers."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        img = Image.new("RGB", (512, 512), color="orange")
        # No source_path argument → PIL path.
        result = compute_thumbnail(img, "pilpath1")
        assert result is not None
        with Image.open(result) as thumb:
            assert thumb.size == (384, 384)
            assert thumb.info.get("icc_profile") == _SRGB_PROFILE_BYTES

    def test_pil_path_runs_when_pyvips_forced_off(
        self, tmp_path, monkeypatch
    ):
        """_PIL_FORCE = True bypasses pyvips even with a source
        path. The test-suite toggle so PIL helpers are exercised
        on hosts that have libvips installed (i.e. this one)."""
        monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
        monkeypatch.setattr("indexer.thumbnails._PIL_FORCE", True)
        src = tmp_path / "src.jpg"
        Image.new("RGB", (640, 480), color=(40, 200, 80)).save(
            src, "JPEG", quality=85
        )
        result = compute_thumbnail(
            Image.new("RGB", (16, 16), color="red"),
            "pilforce1",
            source_path=src,
        )
        assert result is not None
        with Image.open(result) as thumb:
            # The PIL path works from the IN-MEMORY image (16x16),
            # not the source path — and a source already within the
            # cap is not upscaled.
            assert thumb.size == (16, 16)
            # PIL path also embeds the profile.
            assert thumb.info.get("icc_profile") == _SRGB_PROFILE_BYTES