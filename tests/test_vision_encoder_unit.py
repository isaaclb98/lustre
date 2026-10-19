"""
tests/test_vision_encoder_unit.py — Unit tests for indexer/vision_encoder.py.

VisionEncoder is a thin wrapper around the model registry. Tests
use the mock-1536 entry (no actual ML model needed).
"""
from __future__ import annotations

import math

from PIL import Image

from indexer.vision_encoder import (
    DEFAULT_MODEL_NAME,
    MOCK_MODEL_NAME,
    VisionEncoder,
)


# ----- Module constants -----

class TestModuleConstants:
    """Default and mock model names."""

    def test_default_model_name(self):
        assert DEFAULT_MODEL_NAME == "ViT-so400m-patch16-384"

    def test_mock_model_name(self):
        assert MOCK_MODEL_NAME == "mock-1536"

    def test_default_is_registered(self):
        """DEFAULT_MODEL_NAME should be in the kernel registry."""
        from lustre_kernel.registry import get
        spec = get(DEFAULT_MODEL_NAME)
        assert spec is not None


# ----- VisionEncoder construction -----

class TestVisionEncoderConstruction:
    """Basic constructor behavior."""

    def test_default_construction(self):
        encoder = VisionEncoder()
        assert encoder.device == "cpu"
        assert encoder.test_mode is False

    def test_explicit_arch(self):
        encoder = VisionEncoder(arch=MOCK_MODEL_NAME, test_mode=True)
        assert encoder is not None

    def test_test_mode_uses_mock(self):
        """test_mode=True selects the mock-1536 entry."""
        encoder = VisionEncoder(test_mode=True)
        assert encoder.test_mode is True

    def test_device_attribute(self):
        encoder = VisionEncoder(device="cpu", test_mode=True)
        assert encoder.device == "cpu"

    def test_pretrained_ignored(self):
        """The pretrained arg is accepted but ignored."""
        encoder = VisionEncoder(pretrained="ignored-value", test_mode=True)
        # No assertion needed — just shouldn't raise
        assert encoder is not None


# ----- Properties -----

class TestVisionEncoderProperties:
    """dim and resolution properties."""

    def test_dim_is_int(self):
        encoder = VisionEncoder(test_mode=True)
        assert isinstance(encoder.dim, int)
        assert encoder.dim > 0

    def test_resolution_is_int(self):
        encoder = VisionEncoder(test_mode=True)
        assert isinstance(encoder.resolution, int)
        assert encoder.resolution > 0

    def test_dim_with_mock_is_1536(self):
        """Mock-1536's dim tracks the active prod variant.

        Pre-migration this asserted a literal 1536. Post-migration
        the mock spec's dim is resolved from `DEFAULT_MODEL` so the
        mock matches whichever variant is the prod default —
        so400m/16-384 today, 1152-dim. The test now asserts the
        contract ("mock dim equals registry dim") rather than a
        specific literal.
        """
        from lustre_kernel.registry import get as _registry_get
        from search.config import DEFAULT_MODEL
        expected_dim = _registry_get(DEFAULT_MODEL).dim
        encoder = VisionEncoder(test_mode=True)
        assert encoder.dim == expected_dim


# ----- embed_one -----

class TestEmbedOne:
    """Single-image embedding."""

    def test_returns_list_of_floats(self):
        encoder = VisionEncoder(test_mode=True)
        img = Image.new("RGB", (100, 100), color="red")
        result = encoder.embed_one(img)
        assert isinstance(result, list)
        assert len(result) == encoder.dim
        for v in result:
            assert isinstance(v, float)

    def test_unit_normalized(self):
        """Mock encoder produces unit-norm vectors."""
        encoder = VisionEncoder(test_mode=True)
        img = Image.new("RGB", (100, 100), color="red")
        vec = encoder.embed_one(img)
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01

    def test_deterministic(self):
        """Same input → same output."""
        encoder = VisionEncoder(test_mode=True)
        img = Image.new("RGB", (100, 100), color="red")
        v1 = encoder.embed_one(img)
        v2 = encoder.embed_one(img)
        assert v1 == v2

    def test_different_images_different_vectors(self):
        encoder = VisionEncoder(test_mode=True)
        red = Image.new("RGB", (100, 100), color="red")
        blue = Image.new("RGB", (100, 100), color="blue")
        v1 = encoder.embed_one(red)
        v2 = encoder.embed_one(blue)
        assert v1 != v2

    def test_works_with_grayscale(self):
        encoder = VisionEncoder(test_mode=True)
        img = Image.new("L", (100, 100), color=128)
        result = encoder.embed_one(img)
        assert len(result) == encoder.dim

    def test_works_with_rgba(self):
        encoder = VisionEncoder(test_mode=True)
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        result = encoder.embed_one(img)
        assert len(result) == encoder.dim


# ----- embed_batch -----

class TestEmbedBatch:
    """Batch embedding."""

    def test_batch_of_one(self):
        encoder = VisionEncoder(test_mode=True)
        imgs = [Image.new("RGB", (100, 100), color="red")]
        result = encoder.embed_batch(imgs)
        assert len(result) == 1

    def test_batch_of_many(self):
        encoder = VisionEncoder(test_mode=True)
        imgs = [
            Image.new("RGB", (100, 100), color="red"),
            Image.new("RGB", (100, 100), color="blue"),
            Image.new("RGB", (100, 100), color="green"),
        ]
        result = encoder.embed_batch(imgs)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == encoder.dim

    def test_empty_batch(self):
        encoder = VisionEncoder(test_mode=True)
        result = encoder.embed_batch([])
        assert result == []

    def test_batch_results_are_lists(self):
        encoder = VisionEncoder(test_mode=True)
        imgs = [
            Image.new("RGB", (100, 100), color="red"),
            Image.new("RGB", (100, 100), color="blue"),
        ]
        result = encoder.embed_batch(imgs)
        for vec in result:
            assert isinstance(vec, list)
            for v in vec:
                assert isinstance(v, float)


# ----- Module imports -----

class TestModuleImports:
    """Public API is importable."""

    def test_vision_encoder_importable(self):
        from indexer.vision_encoder import VisionEncoder
        assert VisionEncoder is not None

    def test_default_constants_exported(self):
        from indexer.vision_encoder import DEFAULT_MODEL_NAME, MOCK_MODEL_NAME
        assert DEFAULT_MODEL_NAME == "ViT-so400m-patch16-384"
        assert MOCK_MODEL_NAME == "mock-1536"


# ----- Real image file -----

class TestEmbedFromRealFile:
    """embed_one with a real PIL Image loaded from disk."""

    def test_embed_from_jpeg(self, tmp_path):
        encoder = VisionEncoder(test_mode=True)
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="red").save(img_path, "JPEG")
        img = Image.open(img_path)
        vec = encoder.embed_one(img)
        assert len(vec) == encoder.dim

    def test_embed_from_png(self, tmp_path):
        encoder = VisionEncoder(test_mode=True)
        img_path = tmp_path / "test.png"
        Image.new("RGB", (100, 100), color="green").save(img_path, "PNG")
        img = Image.open(img_path)
        vec = encoder.embed_one(img)
        assert len(vec) == encoder.dim