"""
benchmarks/conftest.py — shared fixtures for benchmark tests.

These are isolated from `tests/conftest.py` so a benchmark run doesn't
interfere with the regular test suite's env-var side effects.

Goals:
  - In-memory Qdrant (`location=":memory:"`) for repeatability across hosts.
  - Mock encoder (no model download, no GPU required).
  - Synthetic JPEG corpus generated via PIL — no real photos needed.
  - Resource tracking via `resource.getrusage` (no external deps).
"""

from __future__ import annotations

import os
import random
from collections.abc import Generator
from pathlib import Path

import pytest

# Set env vars BEFORE any search.* import so config.load() sees them.
os.environ.setdefault("SEARCH_TEST_MODE", "1")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")  # unused; we pass :memory:
os.environ.setdefault("MODEL_NAME", "C-RADIOv4-SO400M")
os.environ.setdefault("TOP_K_DEFAULT", "35")
os.environ.setdefault("TOP_K_MAX", "200")


@pytest.fixture
def qdrant_in_memory() -> Generator[object, None, None]:
    """Fresh in-memory Qdrant per test."""
    from qdrant_client import QdrantClient

    from search.qdrant_client import QdrantSearch

    client = QdrantClient(location=":memory:")
    yield QdrantSearch(client=client, collection="bench_images", timeout_ms=2000)


@pytest.fixture
def synth_corpus(tmp_path: Path, request: pytest.FixtureRequest) -> Generator[Path, None, None]:
    """Generate N small synthetic JPEGs in a temp dir.

    Usage:
        @pytest.mark.parametrize("synth_corpus", [200], indirect=True)
        def test_x(synth_corpus): ...

    Or request a custom size:
        def test_x(synth_corpus):
            corpus = synth_corpus  # default 200
    """
    from PIL import Image

    n = getattr(request, "param", 200)
    images_dir = tmp_path / "corpus"
    images_dir.mkdir()

    rng = random.Random(0xCAFE)  # deterministic across runs
    palettes = [
        ((255, 100, 100), (50, 0, 0)),
        ((100, 255, 100), (0, 50, 0)),
        ((100, 100, 255), (0, 0, 50)),
        ((255, 200, 100), (100, 50, 0)),
        ((200, 100, 255), (50, 0, 100)),
    ]
    for i in range(n):
        size = rng.randint(64, 256)
        top, bottom = rng.choice(palettes)
        img = Image.new("RGB", (size, size), top)
        # gradient bottom half
        for y in range(size // 2, size):
            t = (y - size // 2) / max(1, size // 2)
            r = int(top[0] * (1 - t) + bottom[0] * t)
            g = int(top[1] * (1 - t) + bottom[1] * t)
            b = int(top[2] * (1 - t) + bottom[2] * t)
            for x in range(size):
                img.putpixel((x, y), (r, g, b))
        img.save(images_dir / f"img_{i:05d}.jpg", "JPEG", quality=70)

    yield images_dir


@pytest.fixture
def resource_snapshot():
    """Capture peak RSS via `resource.getrusage` (Linux/macOS)."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"rss_kb": usage.ru_maxrss, "utime_s": usage.ru_utime, "stime_s": usage.ru_stime}


@pytest.fixture
def elapsed_ms():
    """Returns a callable that yields elapsed milliseconds since instantiation."""
    import time

    t0 = time.perf_counter()

    def _now() -> float:
        return (time.perf_counter() - t0) * 1000.0

    return _now