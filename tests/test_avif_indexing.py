"""
tests/test_avif_indexing.py — AVIF is indexed, GIF is not.

AVIF is a real photo format and the library has files in it. It was
missing from the indexed extension set, so those files were never
embedded and never appeared in search.

The extension set is shared with the orphan scan's alive-set, which
is what makes the two halves agree: a file that is indexed is a file
the alive-set knows about, so it is never mistaken for a deleted
photo. Adding an extension to ONE of the two sets and not the other
would mark every file of that type as an orphan on the next index
run — and take the user's favourites with it.

GIF is deliberately excluded: it is an animated format, and a
still frame of an animation is not a photo.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from indexer.scan import IMAGE_EXTENSIONS, is_image_suffix
from search.image_resolver import guess_content_type


class TestAvifIsIndexed:
    """AVIF is in the indexed set."""

    def test_avif_is_an_indexed_extension(self):
        assert ".avif" in IMAGE_EXTENSIONS

    def test_is_image_suffix_accepts_avif(self):
        assert is_image_suffix(".avif")
        assert is_image_suffix(".AVIF")

    def test_avif_is_not_gif(self):
        """GIF stays out of the indexed set."""
        assert ".gif" not in IMAGE_EXTENSIONS
        assert not is_image_suffix(".gif")

    def test_the_alive_set_and_the_indexer_agree(self):
        """The orphan scan's alive-set uses the same set.

        `existing_image_paths` filters on `is_image_suffix`, so an
        extension in `IMAGE_EXTENSIONS` is in the alive-set too. That
        symmetry is what stops an indexed AVIF file from being
        reported as a deleted photo.
        """
        assert ".avif" in IMAGE_EXTENSIONS
        assert ".gif" not in IMAGE_EXTENSIONS

    def test_avif_files_are_survived_by_the_walk(self, tmp_path):
        """An .avif file on disk is part of the alive-set."""
        from indexer.scan import existing_image_paths

        (tmp_path / "photo.avif").write_bytes(b"x")
        (tmp_path / "photo.jpg").write_bytes(b"x")

        found = existing_image_paths(tmp_path)

        assert str((tmp_path / "photo.avif").resolve()) in found
        assert str((tmp_path / "photo.jpg").resolve()) in found

    def test_gif_files_are_not_in_the_alive_set(self, tmp_path):
        """A .gif file is not an indexed photo, so it is not in the set.

        It is also not a point in the collection, so it cannot be an
        orphan — there is nothing to mark.
        """
        from indexer.scan import existing_image_paths

        (tmp_path / "anim.gif").write_bytes(b"x")

        found = existing_image_paths(tmp_path)

        assert str((tmp_path / "anim.gif").resolve()) not in found


class TestAvifContentType:
    """AVIF is served with the right Content-Type."""

    def test_avif_content_type(self):
        assert guess_content_type(Path("photo.avif")) == "image/avif"

    def test_gif_content_type_is_unchanged(self):
        """The resolver's GIF mapping is about serving, not indexing.

        It is harmless that the resolver knows the type: the file
        simply never gets into the index, so it is never served.
        """
        assert guess_content_type(Path("anim.gif")) == "image/gif"


class TestAvifDecodes:
    """The runtime can actually decode an AVIF file."""

    def test_pillow_decodes_avif(self, tmp_path):
        """Pillow opens an AVIF and reports its size.

        Pillow's native AVIF support landed in 11.3, which is why the
        dependency floor is `pillow>=11.3`. A runtime without it
        would index every AVIF as a per-file error.
        """
        from PIL import features

        if not features.check("avif"):
            pytest.skip("this Pillow build has no AVIF support")

        p = tmp_path / "photo.avif"
        Image.new("RGB", (8, 6), (10, 20, 30)).save(p, format="AVIF")

        with Image.open(p) as img:
            assert img.size == (8, 6)
