"""
tests/conftest.py

Shared pytest fixtures:
  - test mode env vars (SEARCH_TEST_MODE=1, NAS_IMAGES_BASE=tmp_path)
  - in-memory Qdrant client wired into a search.app
  - tiny test PNG fixtures generated on the fly
  - reset text encoder singleton between tests
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `import search` works without
# the package being installed. Tests should work without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Real-world photo corpus used by tests that need actual JPEG/PNG/WebP/JFIF
# bitstreams (vs. the synthesized `fixture_images` above). Skips if the dir
# isn't present — e.g. on a CI runner without the dev box's NAS mirror.
REAL_PHOTOS_DIR = Path("/home/ubuntu/tmp/pics")


# Set env vars BEFORE any search.* import so config.load() sees them.
os.environ.setdefault("SEARCH_TEST_MODE", "1")
# QDRANT_URL default is a port nothing is listening on, so any test
# that forgets to use the qdrant_in_memory fixture will fail-fast
# with a connection error instead of silently hitting the prod
# Qdrant sidecar. The old default (http://localhost:6333) was the
# prod port, and reconcile_variant_from_qdrant would have called
# delete_collection() on the prod collection on a dim mismatch —
# this is the failure that wiped 2,104,430 points on 2026-10-19.
# Real tests that need a Qdrant use the in-memory fixture; the
# qdrant_in_memory client takes a :memory: location and ignores
# QDRANT_URL entirely.
os.environ.setdefault("QDRANT_URL", "http://localhost:16333")
os.environ.setdefault("SIGLIP_VARIANT", "so400m/16-384")
os.environ.setdefault("MODEL_NAME", "ViT-so400m-patch16-384")
os.environ.setdefault("TOP_K_DEFAULT", "35")
os.environ.setdefault("TOP_K_MAX", "200")


# Register a deterministic "test" model entry so existing tests that
# call `build_payload(..., model_name="test", ...)` continue to work
# after the registry refactor (§A3). Real-model entries are registered
# lazily by `lustre_kernel.registry._try_register_real_models` on
# first call to `get_default_registry()`.
@pytest.fixture(scope="session", autouse=True)
def _register_test_model():
    from lustre_kernel.registry import (
        MockEmbedder,
        ModelSpec,
        get_default_registry,
    )
    # Source the test dim from the registry too — pre-migration this
    # fixture hardcoded 1536 (gopt's dim); post-migration we use the
    # same dim as the active prod variant so any test that compares
    # fixture vectors against registry-resolved dim stays correct.
    from indexer.upsert import VECTOR_DIM
    registry = get_default_registry()
    registry.register(ModelSpec(
        name="test",
        dim=VECTOR_DIM,
        resolution=384,
        revision="test-r0",
        text=MockEmbedder(dim=VECTOR_DIM, resolution=384),
        vision=MockEmbedder(dim=VECTOR_DIM, resolution=384),
    ))
    yield


@pytest.fixture
def qdrant_in_memory():
    """A QdrantClient in :memory: mode, function-scoped (fresh per test)."""
    from qdrant_client import QdrantClient

    from search.qdrant_client import QdrantSearch

    client = QdrantClient(location=":memory:")
    yield QdrantSearch(client=client, collection="images_test", timeout_ms=2000)


@pytest.fixture
def fixture_images(tmp_path: Path) -> Path:
    """
    Create a small set of tiny PNG images for tests.
    Returns the directory path.
    """
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # 5 distinct images with different sizes so id_for produces distinct ids.
    for i, (size, color) in enumerate(
        [
            (16, (255, 0, 0)),
            (24, (0, 255, 0)),
            (32, (0, 0, 255)),
            (40, (255, 255, 0)),
            (48, (255, 0, 255)),
        ]
    ):
        img = Image.new("RGB", (size, size), color)
        img.save(images_dir / f"img_{i:02d}.png")
    # A hidden file that should be skipped.
    (images_dir / ".hidden.jpg").write_bytes(b"not an image")
    # A Thumbs.db that should be skipped.
    (images_dir / "Thumbs.db").write_bytes(b"junk")
    # A non-image file.
    (images_dir / "notes.txt").write_text("hello")
    return images_dir


@pytest.fixture
def nas_base(tmp_path: Path) -> Path:
    """Empty NAS base for resolve_local tests."""
    p = tmp_path / "nas"
    p.mkdir()
    return p


@pytest.fixture(autouse=True)
def _set_nas_base_env(nas_base: Path, monkeypatch):
    """Wire NAS_IMAGES_BASE so config.load() succeeds."""
    monkeypatch.setenv("NAS_IMAGES_BASE", str(nas_base))


# Re-export the shared centroid test fixture so pytest discovers it.
# The fixture + constants + helper live in `_centroid_fixture.py`
# (a regular module that's also importable from test files). See
# that file's docstring for why we don't put them directly here.
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from _centroid_fixture import app_with_centroids  # noqa: F401  (fixture re-export)


@pytest.fixture(autouse=True)
def _reset_text_encoder():
    """Drop the encoder singleton between tests so test_mode is respected."""
    from search import text_encoder

    text_encoder.reset_encoder_for_tests()
    text_encoder.clear_cache()
    yield
    text_encoder.reset_encoder_for_tests()
    text_encoder.clear_cache()
