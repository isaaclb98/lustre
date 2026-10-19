"""
tests/test_image_loader_unit.py — Unit tests for indexer/image_loader.py.

Image loading + preprocessing for embedding. Critical path because
every indexed photo goes through this.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image

from lustre_kernel.registry import ModelNotFoundError
from indexer.image_loader import (
    LoaderError,
    SIGLIP_MEAN,
    SIGLIP_STD,
    _LOAD_TIMEOUT_S,
    _default_resolution,
    letterbox_resize,
    load,
    load_image_pil,
    to_chw_float,
)


# ----- Module constants -----

class TestModuleConstants:
    """Constants used by the loader."""

    def test_siglip_mean(self):
        assert SIGLIP_MEAN == (0.5, 0.5, 0.5)

    def test_siglip_std(self):
        assert SIGLIP_STD == (0.5, 0.5, 0.5)

    def test_default_load_timeout_is_positive(self):
        assert _LOAD_TIMEOUT_S > 0

    def test_default_resolution_is_positive(self):
        assert _default_resolution() > 0


# ----- LoaderError -----

class TestLoaderError:
    """The exception raised on load failures."""

    def test_constructs_with_path_and_reason(self):
        err = LoaderError(Path("/img.jpg"), "file not found")
        assert isinstance(err, Exception)

    def test_message_contains_path(self):
        err = LoaderError(Path("/img.jpg"), "file not found")
        assert "/img.jpg" in str(err)
        assert "file not found" in str(err)


# ----- load_image_pil -----

class TestLoadImagePil:
    """Load a PIL Image from disk."""

    def test_loads_jpeg(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="red").save(img_path, "JPEG")
        loaded = load_image_pil(img_path)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    def test_loads_png(self, tmp_path):
        img_path = tmp_path / "test.png"
        Image.new("RGB", (100, 100), color="blue").save(img_path, "PNG")
        loaded = load_image_pil(img_path)
        assert loaded.mode == "RGB"

    def test_loads_webp(self, tmp_path):
        img_path = tmp_path / "test.webp"
        Image.new("RGB", (100, 100), color="green").save(img_path, "WEBP")
        loaded = load_image_pil(img_path)
        assert loaded.mode == "RGB"

    def test_rgba_converted_to_rgb(self, tmp_path):
        """RGBA images should be converted to RGB (SigLIP2 expects 3 channels)."""
        img_path = tmp_path / "rgba.png"
        Image.new("RGBA", (50, 50), color=(255, 0, 0, 128)).save(img_path, "PNG")
        loaded = load_image_pil(img_path)
        assert loaded.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, tmp_path):
        img_path = tmp_path / "gray.png"
        Image.new("L", (50, 50), color=128).save(img_path, "PNG")
        loaded = load_image_pil(img_path)
        assert loaded.mode == "RGB"

    def test_missing_file_raises(self, tmp_path):
        from indexer.image_loader import LoaderError
        missing = tmp_path / "nope.jpg"
        with pytest.raises(LoaderError):
            load_image_pil(missing)

    def test_corrupted_file_raises(self, tmp_path):
        """Corrupted image data should raise, not return garbage."""
        from indexer.image_loader import LoaderError
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        with pytest.raises(LoaderError):
            load_image_pil(bad)


# ----- letterbox_resize -----

class TestLetterboxResize:
    """Resize preserving aspect ratio with letterbox padding."""

    def test_square_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = letterbox_resize(img, size=200)
        # Aspect ratio 1:1 → fits exactly at 200x200
        assert result.size == (200, 200)

    def test_wide_image_letterboxed(self):
        img = Image.new("RGB", (400, 200), color="red")
        result = letterbox_resize(img, size=200)
        # Aspect ratio 2:1 → 200x100 with letterbox padding
        assert result.size == (200, 200)
        # The actual content is centered (100px tall in middle)

    def test_tall_image_letterboxed(self):
        img = Image.new("RGB", (200, 400), color="red")
        result = letterbox_resize(img, size=200)
        assert result.size == (200, 200)

    def test_no_resize_when_already_correct_size(self):
        img = Image.new("RGB", (256, 256), color="red")
        result = letterbox_resize(img, size=256)
        assert result.size == (256, 256)

    def test_returns_rgb_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = letterbox_resize(img, size=200)
        assert result.mode == "RGB"

    def test_input_image_is_resized(self, tmp_path):
        """Note: PIL.Image.thumbnail() mutates in place.

        The function uses thumbnail() under the hood, which is
        documented to mutate the input image. This test pins that
        behavior so callers know to pass a copy if they want to
        preserve the original.
        """
        img = Image.new("RGB", (400, 200), color="red")
        original_size = img.size
        letterbox_resize(img, size=200)
        # The input WAS mutated (size changed to 200x100)
        assert img.size != original_size
        assert img.size == (200, 100)

    def test_different_size_values(self):
        """Different size parameters produce correctly-sized outputs."""
        img = Image.new("RGB", (300, 300), color="red")
        for size in [128, 256, 384, 512]:
            result = letterbox_resize(img, size=size)
            assert result.size == (size, size)


# ----- load -----

class TestLoad:
    """Full load pipeline: open → EXIF → RGB → letterbox."""

    def test_basic_load(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (300, 200), color="red").save(img_path, "JPEG")
        # Round‑30: load() returns (img, source_w, source_h).
        loaded_img, _sw, _sh = load(img_path)
        # Should be letterboxed to model's resolution
        assert loaded_img.size == (_default_resolution(), _default_resolution())

    def test_load_returns_rgb(self, tmp_path):
        img_path = tmp_path / "test.png"
        Image.new("RGBA", (200, 200), color=(0, 0, 255, 255)).save(img_path, "PNG")
        loaded_img, _sw, _sh = load(img_path)
        assert loaded_img.mode == "RGB"

    def test_load_with_specific_model(self, tmp_path):
        """Loading with a different model name should use that model's resolution."""
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (300, 200), color="red").save(img_path, "JPEG")

        from lustre_kernel.registry import get as registry_get
        # Use a model that has a different resolution
        for model_name in ["ViT-L-16-SigLIP2-256", "ViT-B-16-SigLIP2-256"]:
            _logger = logging.getLogger(__name__)
            try:
                spec = registry_get(model_name)
                loaded_img, _sw, _sh = load(img_path, model_name=model_name)
                assert loaded_img.size == (spec.resolution, spec.resolution)
            except (KeyError, ModelNotFoundError) as exc:
                # Model not registered in test env — skip. Logged
                # at debug so a missing-model setup is easy to
                # diagnose without spamming test output.
                _logger.debug("skipping load test for unregistered model %s: %s", model_name, exc)

    def test_load_missing_file_raises(self, tmp_path):
        from indexer.image_loader import LoaderError
        with pytest.raises(LoaderError):
            load(tmp_path / "missing.jpg")


# ----- to_chw_float -----

class TestToChwFloat:
    """Convert HWC PIL image to CHW float list for model input."""

    def test_returns_list_of_lists(self):
        img = Image.new("RGB", (4, 4), color="red")
        result = to_chw_float(img)
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_three_channels(self):
        """Output has 3 channels (C dimension = 3)."""
        img = Image.new("RGB", (4, 4), color="red")
        result = to_chw_float(img)
        # 3 channels: [R_plane, G_plane, B_plane]
        assert len(result) == 3

    def test_spatial_dims_match(self):
        """Each channel plane has HxW shape."""
        img = Image.new("RGB", (4, 4), color="red")
        result = to_chw_float(img)
        for channel in result:
            assert len(channel) == 4  # height
            assert len(channel[0]) == 4  # width

    def test_values_are_floats(self):
        img = Image.new("RGB", (2, 2), color="red")
        result = to_chw_float(img)
        for channel in result:
            for row in channel:
                for v in row:
                    assert isinstance(v, float)

    def test_red_pixel_has_red_channel_one(self):
        """A pure red image should have R=1.0 in the red channel plane."""
        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        result = to_chw_float(img)
        red_channel = result[0]  # First channel = R
        for row in red_channel:
            for v in row:
                assert abs(v - 1.0) < 0.01  # Allow tiny float error

    def test_green_pixel_has_red_channel_zero(self):
        """A pure green image should have R=0.0 in the red channel plane."""
        img = Image.new("RGB", (2, 2), color=(0, 255, 0))
        result = to_chw_float(img)
        red_channel = result[0]
        for row in red_channel:
            for v in row:
                assert abs(v) < 0.01

    def test_normalization_applied(self):
        """Values should be normalized (not 0-255 range)."""
        img = Image.new("RGB", (2, 2), color=(128, 128, 128))
        result = to_chw_float(img)
        # Normalized values should be in [-1, 1] range, not [0, 255]
        for channel in result:
            for row in channel:
                for v in row:
                    assert -1.5 <= v <= 1.5  # Allow some float margin

    def test_different_sizes(self):
        """to_chw_float should work with arbitrary image sizes."""
        for size in [(2, 2), (10, 5), (5, 10)]:
            img = Image.new("RGB", size, color="blue")
            result = to_chw_float(img)
            assert len(result) == 3
            assert len(result[0]) == size[1]  # height
            assert len(result[0][0]) == size[0]  # width


# ----- Module imports -----

class TestImageLoaderImports:
    """The module's public API."""

    def test_public_functions(self):
        from indexer import image_loader
        assert callable(image_loader.load)
        assert callable(image_loader.load_image_pil)
        assert callable(image_loader.letterbox_resize)
        assert callable(image_loader.to_chw_float)