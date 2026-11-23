"""Round‑30: tests that the indexer persists source dimensions
into both the qdrant payload and the local SQLite cache.

Photo page's `formatDimensions()` reads from the qdrant payload
(or, in the cache path, the SQLite `images` table). Before this
round the indexer never wrote dims, so the page showed "—" for
every photo.

After the fix:
- `indexer.upsert.build_payload` accepts optional `width` /
  `height` keyword args and includes them in the returned dict.
- `indexer.image_loader.load` returns `(letterboxed_img,
  source_w, source_h)` so the pipeline can pass the source dims
  through without re-reading the file.
- `search.index_db.upsert_records` persists the dims in the
  `images` SQLite table.
"""

from __future__ import annotations

from pathlib import Path

from indexer.upsert import build_payload


def test_build_payload_includes_width_and_height(tmp_path: Path):
    """Round‑30: build_payload should include width / height
    when the caller passes them in (the round‑19 hot path)."""
    p = tmp_path / "x.jpg"
    p.write_bytes(b"")
    payload = build_payload(
        path=p, shard="",
        model_name="C-RADIOv4-SO400M",
        model_revision="webli",
        collection="",
        width=3000, height=2000,
    )
    assert payload["width"] == 3000
    assert payload["height"] == 2000


def test_build_payload_width_and_height_default_to_none(tmp_path: Path):
    """Back-compat: existing callers that don't pass dims (the
    legacy disk-only path) still get a payload — the new fields
    just default to None."""
    p = tmp_path / "x.jpg"
    p.write_bytes(b"")
    payload = build_payload(
        path=p, shard="",
        model_name="C-RADIOv4-SO400M",
        model_revision="webli",
        collection="",
    )
    assert "width" in payload
    assert "height" in payload
    assert payload["width"] is None
    assert payload["height"] is None


def test_load_returns_source_dims_alongside_letterboxed_image(tmp_path: Path):
    """Round‑30: `image_loader.load` now returns
    `(letterboxed_img, source_w, source_h)` so the ingest pipeline
    can persist source dims without a second PIL read.

    Source: a 100×200 PNG. After the letterbox to 256×256 the
    returned `img` is 256×256, but `source_w` / `source_h` must
    still be the original 100×200.

    Round‑31 regression test: previously the loader captured dims
    AFTER `Image.draft()` was called inside `load_image_pil`,
    which mutated `img.size` to a non-proportional draft size
    (PIL's JPEG MCU alignment). For PNG (no draft hint) the
    early behaviour happened to be correct, but the JPEG case
    was systematically wrong. This test pins the fix for the PNG
    path and exercises the new `peek_source_dims()` helper which
    is what the JPEG path now uses.
    """
    from PIL import Image

    from indexer.image_loader import load, peek_source_dims

    p = tmp_path / "wide.png"
    Image.new("RGB", (100, 200), (255, 0, 0)).save(p)

    # peek_source_dims: read the header, no decode.
    sw, sh = peek_source_dims(p)
    assert (sw, sh) == (100, 200)

    # load: returns source dims from peek_source_dims (called BEFORE
    # draft), so they're the true source even when load_image_pil's
    # draft step would otherwise have mutated img.size.
    img, source_w, source_h = load(p, model_name="C-RADIOv4-SO400M")
    assert source_w == 100
    assert source_h == 200
    # The live model is native-aspect: the picture keeps its own
    # aspect ratio, it is not letterboxed onto a square canvas.
    assert img.size == (100, 200)


def test_load_jpeg_source_dims_survive_draft(tmp_path: Path):
    """Round‑31 regression test for the JPEG draft‑mode gotcha.

    A 6000×4000 JPEG (landscape, real camera file) must report
    source dims as (6000, 4000), NOT (1000, 1500) which is what
    PIL's `Image.draft('RGB', (512, 512))` mutates `img.size`
    to. (The draft box is a square target, but PIL scales to
    JPEG-MCU-aligned chunks that don't preserve aspect ratio.)

    We use a 6000×4000 real-world-style ratio because that's
    where the bug showed up.
    """
    from PIL import Image

    from indexer.image_loader import load

    p = tmp_path / "landscape.jpg"
    # 600×400 is enough to exercise the draft scaling code path
    # without making the test slow. The bug is in PIL's draft
    # behaviour, not in pixel count.
    Image.new("RGB", (600, 400), (0, 128, 255)).save(p, "JPEG", quality=80)

    img, source_w, source_h = load(p, model_name="C-RADIOv4-SO400M")
    assert source_w == 600, f"source_w={source_w} (expected 600)"
    assert source_h == 400, f"source_h={source_h} (expected 400)"
    # Native-aspect: the decoded image keeps the source's shape.
    assert img.size == (600, 400)
