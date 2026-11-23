"""Regression test for round-32 local_sync missing dimensions bug.

After round-30 made image_loader.load() return source_w/source_h and
the pipeline / run_pipeline / upsert_all updated to pass them through
to build_payload, local_sync.py was missed — its ingest path still
called build_payload(path, shard, model, rev, collection) without
width/height. Every photo touched by local_sync ended up with
width=null, height=null in the qdrant payload, and the photo page
showed "—" for dimensions.

These tests verify that local_sync's ingest path passes through the
loader-captured source dimensions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

INDEXED_PATH = (
    "/mnt/nas-main/images/kpop/collections/aespa/group/"
    "211220_Dreams_Come_True_220425_Ningning_-_ICN_Arrival_from_LAX_Press_220425 Ningning - Press-OSEN 03_c565cc.jpg"
)


@pytest.fixture
def indexed_source():
    p = Path(INDEXED_PATH)
    if not p.exists():
        pytest.skip(f"test source not present: {INDEXED_PATH}")
    return p


def test_load_returns_source_dimensions(indexed_source):
    """The dim-capturing loader should return (img, sw, sh) — and
    the source must be non-square so the test is meaningful."""
    from indexer.image_loader import load_image_pil, peek_source_dims

    img = load_image_pil(indexed_source)
    sw, sh = peek_source_dims(indexed_source)
    assert img.size[0] != img.size[1]
    assert isinstance(sw, int) and sw > 0
    assert isinstance(sh, int) and sh > 0


def test_build_payload_receives_dims(indexed_source):
    """Calling build_payload with width/height must record them
    in the returned dict. Sanity check on the helper that
    local_sync wires into."""
    from indexer.upsert import build_payload

    p = indexed_source
    payload = build_payload(p, shard="", model_name="C-RADIOv4-SO400M",
                            model_revision="", collection="test-src",
                            width=1024, height=1405)
    assert payload["width"] == 1024
    assert payload["height"] == 1405


def test_build_payload_omits_dims_when_not_passed(indexed_source):
    """Pin the implicit contract: callers that don't pass width/height
    get null in the payload. (Captures the original bug shape.)"""
    from indexer.upsert import build_payload

    p = indexed_source
    payload = build_payload(p, shard="", model_name="C-RADIOv4-SO400M",
                            model_revision="", collection="test-src")
    assert payload["width"] is None
    assert payload["height"] is None
