"""
The indexer must read each source photo ONCE.

The hot path used to read the file four times per photo: the libvips
decode for the canonical thumbnail, the variant writer's own libvips
decode, the content hash's byte read and the dhash's PIL open. Each
of those was a full transfer of the source over the NAS — ~8 MB for a
30 MP phone photo — and the reads were the run's ceiling: the GPU sat
at 0% utilisation while the indexer waited on the network.

One read now feeds everything: the decode, the sized variants, the
recorded dimensions, the content hash and the perceptual hash.

The values must not change: the dhash is still PIL's hash of the
full-resolution picture, and the content hash is still the sha256 of
the file's bytes. The dhash drives the near-duplicate collapse, so a
hash computed from a different picture would silently change what the
search considers a duplicate.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from indexer import thumbnails as T
from indexer import upsert as U

MODEL = "C-RADIOv4-SO400M"


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """A portrait photo with an EXIF rotation, like a phone's."""
    p = tmp_path / "sideways.jpg"
    img = Image.new("RGB", (640, 480), (30, 120, 200))
    exif = img.getexif()
    exif[274] = 8
    img.save(p, quality=92, exif=exif)
    return p


class _OpenCounter:
    """Counts how many times the source file is opened."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0

    def note(self, candidate) -> None:
        try:
            if Path(candidate) == self.path:
                self.count += 1
        except TypeError:
            pass


def _count_source_opens(monkeypatch, path: Path, run) -> int:
    counter = _OpenCounter(path)

    import pathlib

    real_read_bytes = pathlib.Path.read_bytes

    def counting_read_bytes(self):
        if self == path:
            counter.count += 1
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", counting_read_bytes)

    real_pil_open = Image.open

    def counting_pil_open(fp, *args, **kwargs):
        try:
            if Path(fp) == path:
                counter.count += 1
        except TypeError:
            pass
        return real_pil_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counting_pil_open)

    pyvips = pytest.importorskip("pyvips")
    real_new_from_file = pyvips.Image.new_from_file

    def counting_new_from_file(filename, *args, **kwargs):
        if str(filename) == str(path):
            counter.count += 1
        return real_new_from_file(filename, *args, **kwargs)

    monkeypatch.setattr(pyvips.Image, "new_from_file", counting_new_from_file)

    run()
    return counter.count


def test_the_source_is_opened_once(tmp_path, monkeypatch):
    """The whole prepare is ONE open of the source."""
    photo = tmp_path / "sideways.jpg"
    img = Image.new("RGB", (640, 480), (30, 120, 200))
    exif = img.getexif()
    exif[274] = 8
    img.save(photo, quality=92, exif=exif)

    monkeypatch.setattr(T, "THUMBNAIL_DIR", str(tmp_path / "thumbs"))
    point_id = U.id_for(photo, "")

    def run():
        # The whole hot path: prepare, write the variants, build the
        # payload. Every one of those used to open the source again.
        prepared = T.prepare_photo(
            photo,
            point_id,
            model_input_size=640,
            want_variants=True,
            want_fingerprints=True,
        )
        assert prepared.model_input is not None
        T.write_variants_from_images(prepared.variants, point_id)
        payload = U.build_payload(
            photo,
            "",
            MODEL,
            "",
            "data",
            width=prepared.source_width,
            height=prepared.source_height,
            fingerprints=prepared.fingerprints,
            blurhash=prepared.blurhash,
        )
        assert payload["width"] == 480

    opens = _count_source_opens(monkeypatch, photo, run)
    assert opens == 1, f"the source was opened {opens} times"


def test_the_dhash_is_unchanged(tmp_path, monkeypatch):
    """The perceptual hash is still PIL's hash of the full-res picture.

    The dhash groups near-duplicates in the search results, so a hash
    computed from a different picture would change what the search
    considers a duplicate.
    """
    from indexer.fingerprints import compute_fingerprints

    photo = tmp_path / "sideways.jpg"
    img = Image.new("RGB", (640, 480), (30, 120, 200))
    exif = img.getexif()
    exif[274] = 8
    img.save(photo, quality=92, exif=exif)

    # what the indexer produced before: PIL opening the FILE
    before = compute_fingerprints(photo)

    # what it produces now: PIL opening the BYTES
    raw = photo.read_bytes()
    with Image.open(io.BytesIO(raw)) as pil:
        pil.load()
        after = compute_fingerprints(pil, sha_bytes=raw)

    assert after["dhash"] == before["dhash"]
    assert after["content_sha256"] == before["content_sha256"]

    # The blurhash moves to the same decode too, and its value must
    # not change: the client renders it as the photo's placeholder.
    from indexer.blurhash import compute_blurhash

    assert compute_blurhash(pil) == compute_blurhash(photo)


def test_the_thumbnail_pixels_are_unchanged(tmp_path, monkeypatch):
    """Decoding the bytes gives the same thumbnail as decoding the file."""
    photo = tmp_path / "sideways.jpg"
    img = Image.new("RGB", (640, 480), (30, 120, 200))
    exif = img.getexif()
    exif[274] = 8
    img.save(photo, quality=92, exif=exif)

    pyvips = pytest.importorskip("pyvips")
    from_file = pyvips.Image.new_from_file(str(photo))
    from_bytes = pyvips.Image.new_from_buffer(photo.read_bytes(), "")

    a = from_file.thumbnail_image(640, height=640)
    b = from_bytes.thumbnail_image(640, height=640)
    assert a.write_to_memory() == b.write_to_memory()
    assert (a.width, a.height) == (b.width, b.height)


def test_the_recorded_dims_are_the_display_dims(tmp_path, monkeypatch):
    """The prepare reports the dims the photo DISPLAYS at."""
    photo = tmp_path / "sideways.jpg"
    img = Image.new("RGB", (640, 480), (30, 120, 200))
    exif = img.getexif()
    exif[274] = 8
    img.save(photo, quality=92, exif=exif)

    monkeypatch.setattr(T, "THUMBNAIL_DIR", str(tmp_path / "thumbs"))
    prepared = T.prepare_photo(
        photo, U.id_for(photo, ""), model_input_size=640
    )
    # The buffer is 640x480 and the orientation rotates it: the photo
    # displays at 480x640.
    assert (prepared.source_width, prepared.source_height) == (480, 640)
