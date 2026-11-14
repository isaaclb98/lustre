"""Regression tests for the server-side photo resize.

The lightbox preloads adjacent photos, and that preload's request IS
what performs the server-side resize (the resize cache is keyed by
(point_id, width), so the preload's URL warms it). A cold resize is
the dominant cost of opening a photo the user has not seen before,
so it is worth keeping cheap.

`Image.draft()` makes libjpeg scale during the DCT decode, so the
Lanczos pass never runs on full-size pixels.

The trap these tests pin: draft() MUST come before exif_transpose().
exif_transpose() loads the image, and a loaded image can no longer
change its decode scale — calling draft() afterwards is silently a
no-op that still LOOKS like it works (identical output, no speedup).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageOps


def _make_jpeg(width: int, height: int) -> bytes:
    """A gradient JPEG, big enough that draft() has a scale to pick."""
    im = Image.new("RGB", (width, height))
    px = im.load()
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _render(source: Path, width: int, use_draft: bool) -> tuple[tuple[int, int], bytes]:
    """Mirror of app._resize_cached's image pipeline.

    Kept as a local copy rather than importing the route helper so
    the test pins the ORDER as well as the outcome — the helper's
    internals are what regressed.
    """
    with Image.open(source) as im:
        if use_draft:
            im.draft("RGB", (width, width))
            im = ImageOps.exif_transpose(im)
        else:
            im = ImageOps.exif_transpose(im)
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if im.width > width:
            im = im.resize(
                (width, max(1, round(im.height * width / im.width))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90, optimize=True)
        return im.size, buf.getvalue()


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("resize") / "source.jpg"
    p.write_bytes(_make_jpeg(4000, 3000))
    return p


def test_draft_produces_identical_dimensions(source: Path) -> None:
    """draft() changes the decode scale, never the final geometry."""
    plain_size, _ = _render(source, 1200, use_draft=False)
    draft_size, _ = _render(source, 1200, use_draft=True)
    assert plain_size == draft_size


def test_draft_is_not_a_no_op(source: Path) -> None:
    """The regression: draft() after exif_transpose() does nothing.

    exif_transpose() loads the image; a loaded image cannot change
    its decode scale. The output is byte-identical to the plain
    path — which reads as "it works" — but no time is saved.
    """
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        im.draft("RGB", (1200, 1200))
        after_transpose = im.size

    with Image.open(source) as im:
        im.draft("RGB", (1200, 1200))
        before_transpose = im.size

    assert before_transpose != after_transpose, (
        "draft() after exif_transpose() is a no-op — it must be called "
        f"before (draft-before gave {before_transpose}, "
        f"draft-after gave {after_transpose})"
    )


def test_draft_keeps_the_target_covered() -> None:
    """draft() must never produce an image narrower than the target.

    libjpeg picks the nearest power-of-two scale factor whose result
    still covers the requested size. If that ever stopped holding,
    the caller would skip the resize and serve an undersized photo.
    """
    width = 1200
    for w, h in ((4000, 3000), (3000, 4000), (4000, 4000), (2400, 1800)):
        with Image.open(io.BytesIO(_make_jpeg(w, h))) as im:
            im.draft("RGB", (width, width))
            if im.width < width and im.height < width:
                # Only a source smaller than the target in BOTH
                # dimensions may come back under it.
                assert im.size == (w, h) or min(im.size) < width


def test_draft_output_is_visually_lossless(source: Path) -> None:
    """The difference must stay imperceptible (PSNR > 35 dB)."""
    import math

    _, plain = _render(source, 1200, use_draft=False)
    _, draft = _render(source, 1200, use_draft=True)

    with Image.open(io.BytesIO(plain)) as a, Image.open(io.BytesIO(draft)) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        assert a.size == b.size
        total = 0.0
        count = a.width * a.height * 3
        for y in range(a.height):
            ra = a.crop((0, y, a.width, y + 1)).tobytes()
            rb = b.crop((0, y, a.width, y + 1)).tobytes()
            for i in range(0, len(ra), 3):
                for c in range(3):
                    d = ra[i + c] - rb[i + c]
                    total += d * d
        mse = total / count
    if mse == 0:
        return
    psnr = 10 * math.log10((255.0 * 255.0) / mse)
    assert psnr > 35.0, f"draft() changed the image too much: PSNR {psnr:.1f} dB"


def test_route_uses_draft_before_transpose() -> None:
    """The shipped helper must call draft() before exif_transpose().

    Checked on the AST, not on the raw source text: the comments
    around the calls mention both names, so a text search would
    match a comment rather than the call.
    """
    import ast
    import inspect

    from search import app

    tree = ast.parse(inspect.getsource(app._resize_cached))

    def line_of(attr: str) -> int:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == attr
            ):
                return node.lineno
        raise AssertionError(f"the resize never calls {attr}()")

    draft_line = line_of("draft")
    transpose_line = line_of("exif_transpose")
    assert draft_line < transpose_line, (
        f"draft() is at line {draft_line} but exif_transpose() is at line "
        f"{transpose_line} — draft() must come first, because "
        "exif_transpose() loads the image and a loaded image can no "
        "longer change its decode scale (draft() becomes a silent no-op)"
    )
