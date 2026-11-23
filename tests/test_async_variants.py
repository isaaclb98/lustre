"""
The canonical thumbnail is written inline; the SIZED VARIANTS are
written off the indexer's critical path.

Measured on prod: the three WebP encodes cost ~195 ms/photo against
a 21 ms decode, and each variant lands on NFS at ~30 ms. Writing
them inline kept the GPU idle ~68% of a run. The variants are
display-only, and the serving endpoint falls back to the canonical
when one is missing — so deferring them costs a slightly larger
image, never a broken one.

These tests pin the split: the canonical exists as soon as
`generate_canonical_and_model_input` returns, the variants do not
exist until `generate_variants_for_path` runs, and both end up on
disk with the right sizes.
"""
from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("pyvips")


@pytest.fixture
def ladder_env(tmp_path, monkeypatch):
    monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_DIR", str(tmp_path))
    monkeypatch.setattr("indexer.thumbnails.THUMBNAIL_SIZE", 640)
    monkeypatch.setattr(
        "indexer.thumbnails.THUMBNAIL_VARIANT_SIZES", (240, 360)
    )
    src = tmp_path / "src.jpg"
    Image.new("RGB", (1280, 960), color=(40, 120, 200)).save(
        src, "JPEG", quality=92
    )
    return tmp_path, src


def test_the_canonical_is_written_and_the_variants_are_not(ladder_env):
    """The split, in one assertion each."""
    from indexer.thumbnails import (
        generate_canonical_and_model_input,
        sized_thumbnail_path,
        thumbnail_path,
    )

    tmp_path, src = ladder_env
    out, model_input, sw, sh = generate_canonical_and_model_input(
        src, "abc123", model_input_size=640
    )

    assert out is not None
    assert out == thumbnail_path("abc123")
    assert out.exists(), "the canonical must exist immediately"
    assert model_input is not None
    assert (sw, sh) == (1280, 960)

    # The variants are NOT written by the canonical call.
    for size in (240, 360):
        assert not sized_thumbnail_path("abc123", size).exists(), (
            f"the {size} variant must not be written on the critical path"
        )


def test_the_model_input_is_the_canonical_pixels(ladder_env):
    """The embedding must be of the picture the user sees.

    Same contract as before the split: the model input IS the
    canonical's pixels, taken before the WebP encode so nothing is
    recompressed on the way to the GPU.
    """
    from indexer.thumbnails import (
        generate_canonical_and_model_input,
        thumbnail_path,
    )

    tmp_path, src = ladder_env
    _, model_input, _, _ = generate_canonical_and_model_input(
        src, "reuse123", model_input_size=640
    )
    assert model_input is not None
    assert max(model_input.size) == 640

    with Image.open(thumbnail_path("reuse123")) as canonical:
        assert canonical.size == model_input.size
        c = canonical.convert("RGB")
        m = model_input.convert("RGB")
        for x, y in ((10, 10), (100, 50), (300, 200)):
            assert abs(c.getpixel((x, y))[2] - m.getpixel((x, y))[2]) <= 3


def test_the_variants_are_written_on_demand(ladder_env):
    """The async call writes exactly the ladder's variants."""
    from indexer.thumbnails import (
        generate_canonical_and_model_input,
        generate_variants_for_path,
        sized_thumbnail_path,
    )

    tmp_path, src = ladder_env
    generate_canonical_and_model_input(src, "v1", model_input_size=640)

    written = generate_variants_for_path(src, "v1")
    assert written == 2, "expected the 240 and 360 variants"

    for size in (240, 360):
        p = sized_thumbnail_path("v1", size)
        assert p.exists(), f"the {size} variant is missing"
        with Image.open(p) as im:
            assert max(im.size) == size


def test_the_variants_keep_the_source_aspect_ratio(ladder_env):
    """A variant is native-aspect, like the canonical — never cropped.

    A square variant next to a native-aspect canonical would make
    the same photo cropped in the small asset and whole in the large
    one, and the masonry's no-crop promise would hold only for the
    tiles that happened to get the canonical.
    """
    from indexer.thumbnails import (
        generate_canonical_and_model_input,
        generate_variants_for_path,
        sized_thumbnail_path,
        thumbnail_path,
    )

    tmp_path, src = ladder_env
    generate_canonical_and_model_input(src, "ar1", model_input_size=640)
    generate_variants_for_path(src, "ar1")

    with Image.open(thumbnail_path("ar1")) as canonical:
        c_size = canonical.size
    with Image.open(sized_thumbnail_path("ar1", 360)) as variant:
        v_size = variant.size

    assert c_size[0] != c_size[1], "the source is not square"
    assert v_size[0] != v_size[1], "the variant must not be square"
    # Same aspect ratio as the canonical.
    assert abs((c_size[0] / c_size[1]) - (v_size[0] / v_size[1])) < 0.02


def test_a_missing_variant_is_detected(ladder_env):
    """The self-heal must notice a missing VARIANT, not just the canonical.

    With the variants written asynchronously, a photo can have its
    canonical on disk and one of its variants still queued (or lost
    to a crash). The canonical alone is not sufficient evidence.
    """
    from indexer.local_sync import _missing_variant
    from indexer.thumbnails import (
        generate_canonical_and_model_input,
        generate_variants_for_path,
        sized_thumbnail_path,
    )

    tmp_path, src = ladder_env
    generate_canonical_and_model_input(src, "m1", model_input_size=640)

    # Canonical present, variants absent -> a variant is missing.
    assert _missing_variant("m1") is True

    generate_variants_for_path(src, "m1")
    assert _missing_variant("m1") is False

    # One variant deleted -> still detected.
    sized_thumbnail_path("m1", 240).unlink()
    assert _missing_variant("m1") is True


def test_the_writer_drains_before_the_run_reports_done(tmp_path, monkeypatch):
    """A queued variant must be on disk by the time the writer closes.

    Otherwise a caller that sees "done" could immediately request a
    variant that has not been written yet.
    """
    from indexer.local_sync import _VariantWriter

    written: list[str] = []

    def _fake_variants(path, point_id):
        written.append(point_id)
        return 2

    monkeypatch.setattr(
        "indexer.local_sync.generate_variants_for_path", _fake_variants
    )

    writer = _VariantWriter(workers=2)
    for i in range(5):
        writer.submit(tmp_path / f"p{i}.jpg", f"id{i}")
    writer.close()

    assert sorted(written) == [f"id{i}" for i in range(5)]
    assert writer.written == 10, "5 photos x 2 variants"
    assert writer.failed == 0
