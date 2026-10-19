"""
tests/test_blurhash.py — unit tests for indexer/blurhash.py.

Covers:
  - compute_blurhash: happy path, missing file, dimension tweaks
  - is_valid_blurhash: structural validator (length + printable ASCII)
  - payload wire-in: build_payload sets blurhash on the Qdrant point

We synthesize a real PNG inside the test rather than ship binary
fixtures — `tests/` is binary-free to keep the repo light, and
synthesis is fast (~10ms for a 64x64 gradient).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Synthesize a real test image at module import time so every test
# function can reuse it without a per-test fixture overhead.
def _synth_png(path: Path, w: int = 64, h: int = 64) -> Path:
    from PIL import Image
    img = Image.new("RGB", (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), (x * 4, y * 4, (x + y) * 2))
    img.save(path, "PNG")
    return path


@pytest.fixture(scope="module")
def sample_png(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("blurhash") / "gradient.png"
    return _synth_png(p)


@pytest.fixture(scope="module")
def tiny_png(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("blurhash") / "tiny.png"
    return _synth_png(p, w=4, h=4)


# ---------- compute_blurhash ----------

def test_compute_blurhash_returns_valid_string(sample_png):
    from indexer.blurhash import compute_blurhash, is_valid_blurhash
    h = compute_blurhash(sample_png)
    assert h is not None
    assert isinstance(h, str)
    assert is_valid_blurhash(h)


def test_compute_blurhash_components_affect_output(sample_png):
    from indexer.blurhash import compute_blurhash
    # Different component counts produce different hashes (the
    # compression-level differs even when the colors are similar).
    h_low = compute_blurhash(sample_png, x_components=3, y_components=2)
    h_high = compute_blurhash(sample_png, x_components=8, y_components=8)
    assert h_low is not None and h_high is not None
    assert h_low != h_high


def test_compute_blurhash_returns_none_for_missing_file(tmp_path):
    from indexer.blurhash import compute_blurhash
    missing = tmp_path / "does_not_exist.png"
    assert compute_blurhash(missing) is None


def test_compute_blurhash_returns_none_for_non_image(tmp_path):
    from indexer.blurhash import compute_blurhash
    # A text file is not a valid image — must return None, not raise.
    text_file = tmp_path / "note.txt"
    text_file.write_text("hello world\n")
    assert compute_blurhash(text_file) is None


def test_compute_blurhash_handles_tiny_image(tiny_png):
    from indexer.blurhash import compute_blurhash
    # Even 4x4 images produce a hash (the encoder clamps gracefully).
    h = compute_blurhash(tiny_png, x_components=3, y_components=2)
    assert h is not None


def test_compute_blurhash_returns_none_for_invalid_encoder_output(sample_png, monkeypatch):
    """Defense-in-depth: if the upstream encoder returns a string
    that fails the structural validator (length / printable-ASCII),
    treat the encode as a failure and return None instead of
    storing poison in the Qdrant payload.

    The real `blurhash` library is stable, but a future encoder
    change or a numpy / Pillow interaction edge case could produce
    malformed output — this guards the indexer against that path
    without taking a hard dependency on the library's correctness.
    """
    import blurhash as _bh

    def fake_encode(*_args, **_kwargs):
        # 1 printable-ASCII char — well below the 6-char minimum
        # legal blurhash length, so the validator rejects it.
        return "x"

    monkeypatch.setattr(_bh, "encode", fake_encode)
    from indexer.blurhash import compute_blurhash
    assert compute_blurhash(sample_png) is None


def test_compute_blurhash_returns_none_for_non_ascii_encoder_output(sample_png, monkeypatch):
    """Variant of the defense-in-depth test: even a string of valid
    length with a NUL byte must be rejected, because the validator
    requires printable ASCII only."""
    import blurhash as _bh

    def fake_encode(*_args, **_kwargs):
        # 30 printable chars + 1 NUL byte — passes the length check,
        # fails the printable-ASCII check.
        return ("a" * 30) + "\x00"

    monkeypatch.setattr(_bh, "encode", fake_encode)
    from indexer.blurhash import compute_blurhash
    assert compute_blurhash(sample_png) is None


def test_compute_blurhash_returns_none_when_encoder_returns_non_string(sample_png, monkeypatch):
    """Variant: the validator also rejects non-string returns. A
    future encoder that accidentally returns bytes (or None, or an
    int) must not get stored as `payload.blurhash`."""
    import blurhash as _bh

    def fake_encode(*_args, **_kwargs):
        return b"\x00\x01\x02\x03"  # bytes, not str

    monkeypatch.setattr(_bh, "encode", fake_encode)
    from indexer.blurhash import compute_blurhash
    assert compute_blurhash(sample_png) is None


# ---------- is_valid_blurhash ----------

def test_is_valid_blurhash_accepts_typical_payload(sample_png):
    from indexer.blurhash import compute_blurhash, is_valid_blurhash
    h = compute_blurhash(sample_png)
    assert is_valid_blurhash(h) is True


def test_is_valid_blurhash_rejects_empty_and_none():
    from indexer.blurhash import is_valid_blurhash
    assert is_valid_blurhash("") is False
    assert is_valid_blurhash(None) is False  # type: ignore[arg-type]


def test_is_valid_blurhash_rejects_wrong_types():
    from indexer.blurhash import is_valid_blurhash
    assert is_valid_blurhash(42) is False  # type: ignore[arg-type]
    assert is_valid_blurhash(b"\x00\x01") is False  # type: ignore[arg-type]


def test_is_valid_blurhash_rejects_too_short_and_too_long():
    from indexer.blurhash import is_valid_blurhash
    # The absolute minimum legal blurhash is 6 chars (1×1 components):
    # 2 prefix chars + 4 DC chars + 0 AC chars = 6. Anything shorter
    # is structurally impossible and must be rejected. The default
    # (4×3) produces 28 chars; 8×8 (the largest we ever call with) is
    # 132 chars. 200 is a sanity cap.
    assert is_valid_blurhash("a" * 5) is False
    assert is_valid_blurhash("a" * 6) is True
    assert is_valid_blurhash("a" * 200) is True
    assert is_valid_blurhash("a" * 201) is False


def test_is_valid_blurhash_rejects_non_printable():
    from indexer.blurhash import is_valid_blurhash
    # Looks plausible length-wise but contains a NUL byte (binary blob).
    bad = "a" * 30 + "\x00" + "b" * 30
    assert is_valid_blurhash(bad) is False


# ---------- payload wire-in ----------

def test_build_payload_includes_blurhash(sample_png, monkeypatch):
    """Smoke-test the wire-in: every point we upsert should have a
    `blurhash` field on its payload."""
    from qdrant_client import QdrantClient

    from indexer import upsert

    # Use an in-memory Qdrant so the wire-in doesn't touch disk.
    client = QdrantClient(location=":memory:")
    upsert.ensure_collection(client, "test_blurhash")
    payload = upsert.build_payload(
        sample_png,
        shard="",
        model_name="test",
        model_revision="r0",
        collection="default",
    )
    assert "blurhash" in payload
    assert payload["blurhash"] is None or isinstance(payload["blurhash"], str)
    # We synthesized a valid PNG — blurhash must compute.
    assert payload["blurhash"] is not None
    # Use the active model's dim — collection was just created with it.
    # Hardcoding 1536 here would break when L/16-256 (1024) is active.
    from lustre_kernel.registry import get_active_model_spec
    active_dim = get_active_model_spec().dim
    upsert.upsert_batch(client, "test_blurhash", [(payload["id"], [0.0] * active_dim, payload)])
    # Round-trip via Qdrant: the stored payload must include blurhash.
    recs, _ = client.scroll("test_blurhash", with_payload=True, limit=1)
    assert len(recs) == 1
    assert "blurhash" in recs[0].payload
    assert recs[0].payload["blurhash"]
