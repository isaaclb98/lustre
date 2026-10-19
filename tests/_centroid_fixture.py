"""
tests/_centroid_fixture.py

Shared pytest fixture for the centroid tests. Lives in a regular
module (not conftest.py) so test files can `from _centroid_fixture
import ...` for the constants too — conftest.py fixtures are
discoverable but the file itself isn't importable as a Python
module.

The fixture is automatically discovered by pytest because it lives
in a module whose name starts with `test_`... wait, no, this is
`_centroid_fixture.py`. Pytest auto-discovers fixtures from any
file matching `test_*.py` or `conftest.py` by default, but it can
also pick up fixtures from any imported module if the file is on
sys.path. For fixtures to be auto-discovered, they should be in
`tests/conftest.py` or `tests/test_*.py`.

Workaround: define the fixture here as a function, and re-export
it from conftest.py via pytest's `pytest_plugins` mechanism, or
just call the factory function explicitly in each test file.

Simplest path: keep the fixture as a regular `@pytest.fixture`
function here, then re-import it into conftest.py so pytest's
auto-discovery picks it up. The conftest re-exports are just
`from _centroid_fixture import app_with_centroids` plus constants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config
from lustre_kernel.registry import MockEmbedder; _mock_embed = MockEmbedder(dim=VECTOR_DIM, resolution=384).embed_text

# Stable test ids (also used as Qdrant point ids — must be valid UUIDs).
CENTROID_CAT_ID = "11111111-1111-1111-1111-111111111111"
CENTROID_DOG_ID = "22222222-2222-2222-2222-222222222222"
CENTROID_CAR_ID = "33333333-3333-3333-3333-333333333333"

# Centroid names (also used as the in-file `name` field).
WUXIA_CENTROID = "wuxia_female_leads"
NOIR_CENTROID = "noir_cinematography"


def save_centroid(
    path: Path,
    name: str,
    *,
    model: str = "siglip2",
    feature_dim: int = VECTOR_DIM,
) -> None:
    """Build a minimal valid centroid .pt file for tests."""
    torch.save(
        {
            "centroid": torch.randn(feature_dim),
            "name": name,
            "model": model,
            "model_type": model,
            "model_id": None,
            "feature_dim": feature_dim,
            "n_images": 25,
            "extracted_at": "2026-06-17T00:00:00",
        },
        path,
    )


@pytest.fixture
def app_with_centroids(qdrant_in_memory, nas_base, tmp_path):
    """
    FastAPI app wired to:
      - in-memory Qdrant (3 points: cat, dog, car)
      - mock text encoder
      - CENTROIDS_DIR = tmp_path/centroids with two valid .pt files
        and one mismatched file (wrong model/dim) that should be
        silently skipped at load.
    """
    centroids_dir = tmp_path / "centroids"
    centroids_dir.mkdir()
    save_centroid(centroids_dir / "siglip2_2026-06-17_wuxia.pt", WUXIA_CENTROID)
    save_centroid(centroids_dir / "siglip2_2026-06-17_noir.pt", NOIR_CENTROID)
    save_centroid(
        centroids_dir / "dinov3_junk.pt", "dinov3_junk",
        model="dinov3", feature_dim=4096,
    )

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
        centroids_dir=str(centroids_dir),
        centroid_expected_model="siglip2",
        centroid_expected_feature_dim=VECTOR_DIM,
    )

    client = qdrant_in_memory.client
    upsert.ensure_collection(client, qdrant_in_memory.collection, dim=VECTOR_DIM)
    items = [
        (CENTROID_CAT_ID, _mock_embed("cat"),
         {"id": CENTROID_CAT_ID, "path": str(nas_base / "cat.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"}),
        (CENTROID_DOG_ID, _mock_embed("dog"),
         {"id": CENTROID_DOG_ID, "path": str(nas_base / "dog.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"}),
        (CENTROID_CAR_ID, _mock_embed("car"),
         {"id": CENTROID_CAR_ID, "path": str(nas_base / "car.jpg"), "collection": "general", "indexed_at": "2026-01-01T00:00:00Z"}),
    ]
    upsert.upsert_batch(client, qdrant_in_memory.collection, items, wait=True)

    Image.new("RGB", (16, 16), (255, 0, 0)).save(nas_base / "cat.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(nas_base / "dog.jpg")

    app_mod.reset_for_tests()
    app = app_mod.create_app(cfg=cfg, qdrant=qdrant_in_memory)
    with TestClient(app) as tc:
        yield tc
    app_mod.reset_for_tests()
