"""
Regression tests for the lazy-liveness check on the /photo/{id}/raw
and /photo/{id} routes.

When the indexer stored photo paths as Windows UNC strings (e.g.
`\\\\nas\\share\\images\\foo.jpg`) and the search app runs on Linux
with PATH_PREFIX set to the same UNC and NAS_IMAGES_BASE pointing at
the corresponding NFS mount, the lazy-liveness check must consult
the *resolved* local path, not the raw payload path. The raw UNC
path is never alive on the Linux server and was silently short-
circuiting every request to 404.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from indexer import upsert
from indexer.upsert import VECTOR_DIM
from search import app as app_mod
from search.config import Config
from search.qdrant_client import QdrantSearch
from lustre_kernel.spaces import TEXT_SPACE

# Production-shape constants: payload path is a UNC, path_prefix is
# the same UNC, NAS_IMAGES_BASE is the local Linux mount.
PREFIX = "\\\\nas\\share\\images"
PAYLOAD_PATH = "\\\\nas\\share\\images\\kpop\\0849.jpg"
POINT_ID = "11111111-1111-1111-1111-111111111111"
COLLECTION = "images_test_unc"

# Minimal 1x1 JPEG so FileResponse has bytes to serve and the
# content-type test can assert on the JPEG SOI marker.
MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
    b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
    b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n"
    b"\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijst"
    b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
    b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
    b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
    b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
    b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa3\xff\xd9"
)


def _build(tmp_path: Path, *, delete_local_file: bool = False):
    """Build a fresh app + in-memory Qdrant + sqlite cache.

    Returns (client, image_path_on_disk). Writes a real JPEG under
    `tmp_path/images/kpop/0849.jpg` so the resolve target exists,
    unless delete_local_file is True (used for the 404 case).
    """
    nas_base = tmp_path / "images"
    image_dir = nas_base / "kpop"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "0849.jpg"
    if not delete_local_file:
        image_path.write_bytes(MINIMAL_JPEG)

    cfg = Config(
        qdrant_url="http://localhost:6333",  # ignored; we pass our own qdrant below
        qdrant_collection=COLLECTION,
        qdrant_api_key=None,
        model_name="mock",
        model_revision="",
        device="cpu",
        top_k_default=50,
        top_k_max=200,
        query_timeout_ms=2000,
        nas_images_base=str(nas_base),
        path_prefix=PREFIX,
        web_ui_url="http://localhost:8000",
        log_level="WARNING",
        index_db_path=str(tmp_path / "images.db"),
        test_mode=True,
    )

    qclient = QdrantClient(location=":memory:")
    upsert.ensure_collection(qclient, cfg.qdrant_collection, dim=VECTOR_DIM)
    qclient.upsert(
        collection_name=cfg.qdrant_collection,
        points=[
            qmodels.PointStruct(
                id=POINT_ID,
                vector={TEXT_SPACE: [0.0] * VECTOR_DIM},
                payload={"path": PAYLOAD_PATH, "collection": "kpop"},
            )
        ],
    )
    qdrant = QdrantSearch(
        client=qclient, collection=cfg.qdrant_collection, timeout_ms=2000
    )

    app = app_mod.create_app(cfg, qdrant=qdrant)
    client = TestClient(app)
    return client, image_path


def test_photo_raw_serves_image_when_payload_path_is_unc(tmp_path):
    """The /raw endpoint must serve the resolved local image, not 404."""
    client, image_path = _build(tmp_path)

    # Sanity: the file genuinely exists on disk.
    assert image_path.exists(), "test setup: local file MUST exist"

    resp = client.get(f"/photo/{POINT_ID}/raw")

    assert resp.status_code == 200, (
        f"expected 200 with image bytes, got {resp.status_code} "
        f"({resp.text!r}) — the lazy-liveness check on the raw UNC "
        "payload path is short-circuiting before resolve_local runs."
    )
    assert resp.headers["content-type"].startswith("image/")
    assert resp.content[:2] == b"\xff\xd8"  # JPEG SOI
    assert resp.content == image_path.read_bytes()




def test_photo_raw_404s_when_resolved_local_file_is_missing(tmp_path):
    """A missing file should still 404 — the fix must not pass everything."""
    client, _ = _build(tmp_path, delete_local_file=True)

    resp = client.get(f"/photo/{POINT_ID}/raw")

    assert resp.status_code == 404, (
        f"missing file should 404, got {resp.status_code} "
        "— the fix is letting through dead paths."
    )
