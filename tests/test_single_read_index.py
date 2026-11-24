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

The content hash is still the sha256 of the file's bytes. The
perceptual hash and the blurhash are computed from the CANONICAL
THUMBNAIL's pixels instead of from a full-resolution decode: that
decode was ~354 ms per 30 MP photo, the largest CPU item in the run.
The dhash therefore hashes the 640-pixel picture rather than the 30 MP
one — it still groups near-duplicates, but it is not the same
function of the source as the hash the corpus was indexed with before
this change, and a photo indexed on either side of it can compare
unequal to its own copy.
"""
from __future__ import annotations

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


def test_the_dhash_comes_from_the_canonical(tmp_path, monkeypatch):
    """The perceptual hash is the hash of the CANONICAL's pixels.

    Not of the full-resolution picture: that decode is what this
    change removes. The hash still groups near-duplicates, but it is a
    different function of the source than the full-resolution hash, so
    the corpus is indexed from scratch with the one it now produces.
    """
    from indexer.fingerprints import compute_fingerprints, dhash

    photo = tmp_path / "structured.jpg"
    img = Image.new("RGB", (1200, 900), (30, 120, 200))
    # structure, so the two hashes really differ
    for x in range(0, 1200, 17):
        for y in range(0, 900, 13):
            img.putpixel((x, y), (x % 255, y % 255, (x * y) % 255))
    img.save(photo, quality=92)

    monkeypatch.setattr(T, "THUMBNAIL_DIR", str(tmp_path / "thumbs"))
    point_id = U.id_for(photo, "")

    prepared = T.prepare_photo(
        photo,
        point_id,
        model_input_size=640,
        want_variants=True,
        want_fingerprints=True,
    )
    assert prepared.fingerprints is not None

    # the full-resolution hash, for comparison
    before = compute_fingerprints(photo)

    # the canonical's pixels — the same ones the indexer hashes:
    # libvips' resize of the source, taken BEFORE the lossy WebP
    # encode (reading the encoded file back would cost a second read
    # of the source).
    pyvips = pytest.importorskip("pyvips")
    raw = photo.read_bytes()
    canonical = T._vips_to_pil(
        pyvips.Image.new_from_buffer(raw, "").thumbnail_image(640, height=640)
    )

    assert prepared.fingerprints["dhash"] == dhash(canonical)
    assert prepared.fingerprints["content_sha256"] == before["content_sha256"]

    # The hash DRIFTS from the full-resolution one — measured on real
    # photos, 33% of them differ, by 1.3 bits on average — so it must
    # stay well inside the distance at which the search collapses two
    # photos as duplicates. It does: the drift is never close to it.
    import dataclasses

    from indexer.fingerprints import hamming_distance
    from search.config import Config

    threshold = next(
        f.default
        for f in dataclasses.fields(Config)
        if f.name == "diversity_duplicate_hamming_distance"
    )
    drift = hamming_distance(
        prepared.fingerprints["dhash"], before["dhash"]
    )
    assert drift is not None
    assert drift < threshold, (
        f"the canonical's hash drifted {drift} bits from the "
        f"full-resolution one; the collapse threshold is {threshold}"
    )

    # the blurhash is the placeholder the client shows: it is now the
    # picture the user actually sees.
    from indexer.blurhash import compute_blurhash

    assert prepared.blurhash == compute_blurhash(canonical)


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
