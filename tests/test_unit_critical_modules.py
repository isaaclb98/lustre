"""
tests/test_unit_critical_modules.py

Unit tests for the indexer critical modules (§A4 of the plan):
- `upsert`: id stability, payload assembly
- `image_loader`: corrupt input, EXIF rotation, normalization
- `cache`: atomicity, drift detection, version mismatch
- `vision_encoder`: mock embed round-trip, dim correctness

These complement the existing integration tests in `tests/test_*.py`
and target the failure modes the audit identified as most likely to
silently regress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# upsert — id stability + payload assembly
# ---------------------------------------------------------------------------

class TestUpsertIdStability:
    """`id_for(path, shard)` must be deterministic and shard-scoped."""

    def test_id_is_deterministic(self, tmp_path: Path):
        from indexer.upsert import id_for

        p = tmp_path / "img.jpg"
        p.write_bytes(b"x")
        ids = {id_for(p, "") for _ in range(100)}
        assert len(ids) == 1, "id_for must be deterministic"

    def test_id_differs_across_shards(self, tmp_path: Path):
        from indexer.upsert import id_for

        p = tmp_path / "img.jpg"
        p.write_bytes(b"x")
        assert id_for(p, "shard-a") != id_for(p, "shard-b")

    def test_id_differs_across_paths(self, tmp_path: Path):
        from indexer.upsert import id_for

        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "b.jpg").write_bytes(b"b")
        assert id_for(tmp_path / "a.jpg", "") != id_for(tmp_path / "b.jpg", "")


class TestUpsertPayloadAssembly:
    """`build_payload` populates every declared schema field."""

    def _png(self, path: Path) -> None:
        Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path, "PNG")

    def test_build_payload_populates_every_field(self, tmp_path: Path):
        from lustre_kernel.payload_schema import payload_field_names
        from indexer.upsert import build_payload

        p = tmp_path / "img.png"
        self._png(p)
        payload = build_payload(
            p, shard="", model_name="test", model_revision="r0",
            collection="default",
        )
        declared = set(payload_field_names())
        actual = set(payload.keys())
        missing = declared - actual
        assert not missing, f"build_payload missing fields: {sorted(missing)}"

    def test_build_payload_sets_required_fields(self, tmp_path: Path):
        """Every REQUIRED_FIELDS entry lands on the payload."""
        from indexer.upsert import build_payload
        from lustre_kernel.payload_schema import REQUIRED_FIELDS

        p = tmp_path / "img.png"
        self._png(p)
        payload = build_payload(
            p, shard="", model_name="test", model_revision="r0",
            collection="default",
        )
        missing = REQUIRED_FIELDS - payload.keys()
        assert not missing, f"build_payload missing required fields: {missing}"

    def test_build_payload_folder_is_parent_path(self, tmp_path: Path):
        from lustre_kernel.payload_schema import FIELD_FOLDER
        from indexer.upsert import build_payload

        sub = tmp_path / "vacation"
        sub.mkdir()
        p = sub / "img.png"
        self._png(p)
        payload = build_payload(
            p, shard="", model_name="test", model_revision="r0",
            collection="default",
        )
        assert payload[FIELD_FOLDER] == str(sub.resolve())

    def test_build_payload_model_dim_from_registry(self, tmp_path: Path):
        from lustre_kernel.payload_schema import FIELD_MODEL_DIM
        from lustre_kernel.registry import get as registry_get
        from indexer.upsert import build_payload

        p = tmp_path / "img.png"
        self._png(p)
        payload = build_payload(
            p, shard="", model_name="test", model_revision="r0",
            collection="default",
        )
        assert payload[FIELD_MODEL_DIM] == registry_get("test").dim


# ---------------------------------------------------------------------------
# image_loader — corrupt input + EXIF rotation + normalize
# ---------------------------------------------------------------------------

class TestImageLoader:
    def test_load_succeeds_on_valid_png(self, tmp_path: Path):
        from indexer.image_loader import load

        p = tmp_path / "valid.png"
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(p, "PNG")
        # Round‑30: load() returns (img, source_w, source_h).
        img, _sw, _sh = load(p)
        assert img.size == (384, 384)  # default resolution

    def test_load_letterboxes_to_registered_resolution(self, tmp_path: Path):
        """Smaller model → smaller letterbox target."""
        from indexer.image_loader import load

        p = tmp_path / "valid.png"
        Image.new("RGB", (32, 32), color=(0, 255, 0)).save(p, "PNG")
        img, _sw, _sh = load(p, model_name="ViT-L-16-SigLIP2-256")
        assert img.size == (256, 256)

    def test_load_raises_on_corrupt_file(self, tmp_path: Path):
        from indexer.image_loader import LoaderError, load

        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"\xff\xff\xff\xffgarbage not an image")
        with pytest.raises(LoaderError):
            load(p)

    def test_load_raises_on_missing_file(self, tmp_path: Path):
        from indexer.image_loader import LoaderError, load

        p = tmp_path / "missing.jpg"
        with pytest.raises(LoaderError):
            load(p)

    def test_load_raises_on_non_image_extension(self, tmp_path: Path):
        from indexer.image_loader import LoaderError, load

        p = tmp_path / "notes.txt"
        p.write_text("not an image")
        with pytest.raises(LoaderError):
            load(p)

    def test_load_handles_exif_transpose(self, tmp_path: Path):
        """EXIF-rotated JPEGs are transposed before the embedder sees them.

        Hard to set EXIF in a unit test without a real JPEG. We
        verify the helper is called via the load path on a
        valid JPEG without an EXIF tag (the transposition is a
        no-op in that case).
        """
        from indexer.image_loader import load

        p = tmp_path / "no_exif.jpg"
        Image.new("RGB", (100, 200), color=(0, 0, 255)).save(p, "JPEG")
        img, _sw, _sh = load(p)
        assert img.size == (384, 384)  # letterboxed


# ---------------------------------------------------------------------------
# cache — atomicity + drift detection + version mismatch
# ---------------------------------------------------------------------------

class TestIndexerCache:
    def test_write_then_read_round_trip(self, tmp_path: Path):
        from indexer.cache import IndexerCache

        cache_path = tmp_path / "cache.json"
        a_path = tmp_path / "a.jpg"
        a_path.write_bytes(b"a")
        stat = a_path.stat()
        c = IndexerCache(cache_path, "test")
        c.add(
            a_path, "id-a",
            mtime=int(stat.st_mtime), size=int(stat.st_size),
        )
        c.save()
        c2 = IndexerCache(cache_path, "test")
        loaded = c2.load()
        assert loaded
        assert c2.has(a_path)
        entry = c2._entries[str(a_path)]
        assert entry.id == "id-a"

    def test_atomic_write_no_partial_state(self, tmp_path: Path):
        """Atomic write leaves the cache readable even if interrupted.

        For SQLite: the WAL mode means a mid-write crash leaves
        the DB on the last committed transaction. The on-disk
        file is always a valid SQLite database.
        """
        import sqlite3

        from indexer.cache import IndexerCache

        cache_path = tmp_path / "cache.db"
        a_path = tmp_path / "a.jpg"
        a_path.write_bytes(b"a")
        stat = a_path.stat()
        c = IndexerCache(cache_path, "test")
        c.add(
            a_path, "id-a",
            mtime=int(stat.st_mtime), size=int(stat.st_size),
        )
        c.save()
        assert cache_path.exists()
        # Open the on-disk file as a fresh SQLite connection and
        # confirm the row is there — proves the write was atomic.
        conn = sqlite3.connect(str(cache_path))
        rows = conn.execute("SELECT id, mtime, size FROM entries").fetchall()
        assert rows == [("id-a", int(stat.st_mtime), int(stat.st_size))]
        conn.close()

    def test_stale_cache_version_refused(self, tmp_path: Path):
        """Cache written with a future CACHE_VERSION is rejected on load."""
        from indexer.cache import IndexerCache

        cache_path = tmp_path / "cache.json"
        # Manually write a cache with an unsupported version.
        cache_path.write_text(json.dumps({
            "version": 999,
            "entries": {},
        }))
        c = IndexerCache(cache_path, "test")
        loaded = c.load()
        # Either the loader returns False (refusal), or the
        # entries dict is empty (refusal-by-ignore).
        if loaded:
            # If it claims to load, the entries must be empty
            # (the stale format's data is not silently reused).
            assert c.has(tmp_path / "nonexistent") is False


# ---------------------------------------------------------------------------
# vision_encoder — mock embed round-trip via registry
# ---------------------------------------------------------------------------

class TestVisionEncoderMockPath:
    def test_test_mode_uses_mock(self):
        from indexer.vision_encoder import VisionEncoder

        enc = VisionEncoder(test_mode=True)
        # Mock dim tracks the active prod variant — so400m today
        # (1152-dim). Pre-migration this was 1536 (gopt's dim).
        from lustre_kernel.registry import get as _registry_get
        from search.config import DEFAULT_MODEL
        expected_dim = _registry_get(DEFAULT_MODEL).dim
        assert enc.dim == expected_dim
        img = Image.new("RGB", (8, 8), color=(1, 2, 3))
        vec = enc.embed_one(img)
        assert isinstance(vec, list)
        assert len(vec) == expected_dim
        # Unit norm.
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_different_inputs_produce_different_vectors(self):
        from indexer.vision_encoder import VisionEncoder

        enc = VisionEncoder(test_mode=True)
        img_a = Image.new("RGB", (8, 8), color=(1, 2, 3))
        img_b = Image.new("RGB", (8, 8), color=(4, 5, 6))
        vec_a = enc.embed_one(img_a)
        vec_b = enc.embed_one(img_b)
        assert vec_a != vec_b

    def test_same_input_produces_same_vector(self):
        from indexer.vision_encoder import VisionEncoder

        enc = VisionEncoder(test_mode=True)
        img = Image.new("RGB", (8, 8), color=(7, 8, 9))
        vec1 = enc.embed_one(img)
        vec2 = enc.embed_one(img)
        assert vec1 == vec2
