"""
The dims the indexer records must be the dims the photo DISPLAYS at.

A camera held sideways writes the frame in the camera's own landscape
buffer and records the rotation in the EXIF orientation tag. The
picture is only portrait once that tag is applied.

The indexer applied the tag to the PIXELS but reported the BUFFER's
dimensions, so the payload said `5472 x 3648` for a portrait photo.
The grid lays each tile out from the recorded ratio and then
cover-crops the upright picture into it, which cuts the head off the
photo — the thumbnail on disk was correct the whole time, the numbers
that described it were not.

These tests pin the invariant: the dims a photo reports are the dims
of the picture the viewer sees, and they agree with the thumbnail the
indexer writes beside them.
"""
from __future__ import annotations


import inspect
from pathlib import Path

import pytest
from PIL import Image

from indexer.image_loader import (
    display_size,
    load_image_pil_with_dims,
    orientation_of,
    peek_source_dims,
)


def _write_photo(path: Path, size: tuple[int, int], orientation: int | None) -> Path:
    """A JPEG whose BUFFER is `size`, tagged with an EXIF orientation."""
    img = Image.new("RGB", size, (200, 40, 90))
    if orientation is not None:
        exif = img.getexif()
        exif[274] = orientation
        img.save(path, quality=95, exif=exif)
    else:
        img.save(path, quality=95)
    return path


# ----- the rule -----

class TestDisplaySize:
    """The orientation values that rotate the frame swap the axes."""

    @pytest.mark.parametrize("orientation", [1, 2, 3, 4])
    def test_an_unrotated_orientation_keeps_the_axes(self, orientation: int) -> None:
        assert display_size(200, 100, orientation) == (200, 100)

    @pytest.mark.parametrize("orientation", [5, 6, 7, 8])
    def test_a_rotated_orientation_swaps_the_axes(self, orientation: int) -> None:
        assert display_size(200, 100, orientation) == (100, 200)

    def test_no_orientation_at_all_keeps_the_axes(self) -> None:
        assert display_size(200, 100, None) == (200, 100)


# ----- the loader -----

class TestLoaderReportsDisplayDims:
    """`load_image_pil_with_dims` is what the payload is built from."""

    def test_a_sideways_photo_reports_its_displayed_axes(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "sideways.jpg", (200, 100), 8)
        img, w, h = load_image_pil_with_dims(photo)
        assert (w, h) == (100, 200)
        # The pixels were already transposed, and the dims describe
        # exactly that picture.
        assert img.size == (100, 200)

    def test_an_upright_photo_reports_its_own_axes(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "upright.jpg", (200, 100), None)
        _, w, h = load_image_pil_with_dims(photo)
        assert (w, h) == (200, 100)

    def test_the_dims_describe_the_picture_the_loader_returns(
        self, tmp_path: Path
    ) -> None:
        """The invariant: the reported dims match the decoded image."""
        for orientation in (None, 1, 6, 8):
            photo = _write_photo(
                tmp_path / f"o{orientation}.jpg", (200, 100), orientation
            )
            img, w, h = load_image_pil_with_dims(photo)
            assert (w, h) == img.size, orientation

    def test_peek_source_dims_reports_the_same_axes(self, tmp_path: Path) -> None:
        """The header-only read has to agree with the decode."""
        photo = _write_photo(tmp_path / "sideways.jpg", (200, 100), 8)
        assert peek_source_dims(photo) == (100, 200)

    def test_orientation_of_reads_the_tag(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "sideways.jpg", (200, 100), 8)
        with Image.open(photo) as img:
            assert orientation_of(img) == 8

    def test_orientation_of_a_plain_photo_is_none(self, tmp_path: Path) -> None:
        photo = _write_photo(tmp_path / "plain.jpg", (200, 100), None)
        with Image.open(photo) as img:
            assert orientation_of(img) is None


# ----- the libvips path -----

class _StubVipsImage:
    """A libvips image's shape, without libvips.

    `pyvips.Image.new_from_file()` does not apply the EXIF
    orientation tag, so the buffer's axes are the camera's. The
    thumbnail path reads the dims off the image, and it has to ask
    the tag too.
    """

    def __init__(self, width: int, height: int, orientation: int | None) -> None:
        self.width = width
        self.height = height
        self._orientation = orientation

    def get(self, field: str) -> int:
        if field == "orientation" and self._orientation is not None:
            return self._orientation
        raise ValueError(f"no field {field}")


class TestLibvipsDisplaySize:
    def test_a_rotated_image_reports_its_displayed_axes(self) -> None:
        from indexer.thumbnails import _libvips_display_size

        assert _libvips_display_size(_StubVipsImage(200, 100, 8)) == (100, 200)

    def test_an_image_without_the_tag_keeps_its_axes(self) -> None:
        from indexer.thumbnails import _libvips_display_size

        assert _libvips_display_size(_StubVipsImage(200, 100, None)) == (200, 100)


class TestThePyvipsBranchAsksTheTag:
    """The hot path's dims come from the helper, not from the buffer."""

    def test_the_pyvips_branch_does_not_report_the_raw_axes(self) -> None:
        from indexer import thumbnails

        src = inspect.getsource(thumbnails.generate_canonical_and_model_input)
        for raw in ("src.width", "src.height"):
            assert raw not in src, (
                "generate_canonical_and_model_input reads the libvips "
                "buffer's own axes — those are the camera's, not the photo's"
            )

    def test_the_branch_calls_the_display_size_helper(self) -> None:
        from indexer import thumbnails

        src = inspect.getsource(thumbnails.generate_canonical_and_model_input)
        assert "_libvips_display_size" in src
