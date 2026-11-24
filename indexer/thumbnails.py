"""
indexer/thumbnails.py — generate WebP thumbnails at index time.

Thumbnails keep the SOURCE's aspect ratio — never cropped,
squashed or letterboxed. The LONG edge is capped at THUMBNAIL_SIZE
(the default 384); the short edge follows the source. WebP q80
4:4:4. Storage layout: {THUMBNAIL_DIR}/{prefix}/{point_id}.webp

Two-level prefix (first 2 chars of point_id) avoids putting 2M files
in one directory. ~8K files per bucket at 2M scale.

Post the model-variant migration plan: the thumbnail is the same
resolution as the so400m model input (384×384). The embedder still
loads the original JPEG directly — see `image_loader.py`'s
`_resize_for_embedding` — but indexing now writes a single 384px
display asset. The canonical 256px file is no longer written; the
sized-variant tuple is empty.

Why a single 384px variant (vs. the pre-migration 240/180/120 trio):
the model's input resolution is 384, and 384px is enough to render
crisp up to 192 CSS pixels at 2× DPR — covers most desktop + tablet
tile sizes. The pre-migration 240 was already too small to be
sharp at 192 CSS px × 2× DPR; 480 would be overkill for the common
rendering path and waste storage. Single-variant keeps the on-disk
footprint manageable and lets the frontend use a single hardcoded
`?w=384` request without consulting a srcset.

Why 384 and not a new 240/360/480 trio: the storage win is real
(~22 GB → ~20 GB on a 902k corpus) and the rendering win is
neutral (the new 384px is sharp where the old 240 was blurry).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from PIL import Image, ImageCms

from indexer.image_loader import display_size
from indexer.upsert import id_for

if TYPE_CHECKING:
    import pyvips as _pyvips_typing  # noqa: F401

# pyvips (libvips bindings) is the primary thumbnail pipeline. It
# does all four libvips behaviors as a package: 4:4:4 chroma
# subsampling at Q>=80, linear-light resize, ICC profile embed,
# and 10-20x the throughput of the PIL path. If pyvips is
# importable we route through it; if not (libvips system package
# missing), we fall back to a slower PIL pipeline that does the
# same four things in pure Python. The fallback is in the module
# for two reasons: (1) the indexer never breaks on a host that's
# missing the system libvips package; (2) tests can monkeypatch
# _PIL_FORCE = True to exercise the PIL path under CI conditions
# that don't have libvips installed (most CI runners).
try:
    import pyvips as _pyvips_runtime  # noqa: F401

    _HAS_PYVIPS = True
except ImportError:
    _HAS_PYVIPS = False

THUMBNAIL_DIR = os.environ.get("THUMBNAIL_DIR", "/app/data/thumbnails")
# The thumbnail's LONG edge, in pixels. The short edge follows the
# source's aspect ratio: the thumbnail is never cropped, squashed or
# letterboxed. A portrait photo produces a portrait thumbnail.
#
# This is the DISPLAY size: how big the served thumbnail is. It is
# NOT the model's input size — the C-RADIOv4 embedder reads its own
# MODEL_INPUT_SIZE (see `lustre_kernel/_radio_models.py`). They were
# the same number until the 640 embed took the GPU off the PCIe bus
# (Xid 79): welding them meant paying GPU cost for a display size.
# Both still show the SAME framing — the source at its native aspect
# ratio, never cropped — so only the resolution differs.
#
# Generating a thumbnail is pure CPU work (libvips; this module
# imports no torch), so raising this number costs disk and encode
# time, never VRAM.
#
# The default is unchanged from the SigLIP2 era (384 = the old model
# input) so a deployment that does not set it keeps its current
# storage footprint; the geometry change alone makes the files
# slightly SMALLER for non-square sources.
THUMBNAIL_SIZE: int = int(os.environ.get("THUMBNAIL_SIZE", "640"))

# The DISPLAY thumbnail's smaller variants, as long edges.
#
# These are for a dense view mode: the grid renders a tile at a
# fixed CSS width, and a tile of 192 CSS px needs a much smaller
# asset than a tile of 360. Serving the 640 canonical to a 192px
# tile wastes the bandwidth AND the decode cost for pixels the
# browser scales straight back down.
#
# Each variant is the SAME native-aspect photo at a smaller long
# edge, written as a `w{size}.webp` sibling of the canonical. The
# thumbnail router serves `?w={size}` from that sibling, and falls
# back to the canonical when it is absent — so a variant that was
# never generated is a slower response, never a broken image.
#
# Empty by default: a deployment opts in. The values are the long
# edges, and a variant at or above the canonical is skipped (it
# would be an upscale).
def parse_variant_sizes(raw: str) -> tuple[int, ...]:
    """Parse the variant ladder from its environment value.

    A comma-separated list of long edges. Entries that are not
    positive integers are ignored rather than raising: a typo in a
    deployment's environment should cost the deployment its smaller
    variants, not its whole thumbnail pipeline.

    The result is sorted and de-duplicated, so the ladder is a
    strict sequence however the deployment spelled it.
    """
    sizes = {
        int(part)
        for part in raw.split(",")
        if part.strip().isdigit() and int(part) > 0
    }
    return tuple(sorted(sizes))


THUMBNAIL_VARIANT_SIZES: tuple[int, ...] = parse_variant_sizes(
    os.environ.get("THUMBNAIL_VARIANT_SIZES", "240,360")
)


def thumbnail_target_size(
    width: int, height: int, size: int | None = None
) -> tuple[int, int]:
    """Target dimensions for a native-aspect thumbnail.

    The image is scaled so its LONG edge is `size`, and the short edge
    follows the source's aspect ratio. Never upscales a source that
    is already smaller than the cap.

    This is the same rule Immich applies to its non-cropping resize
    (`fit: 'inside'`): the picture fits inside a size x size box with
    its aspect preserved, so the long edge is the cap. Immich's own
    thumbnail is a square centre-crop instead (`fit: 'outside'`), which
    is the behaviour this replaces.

    The cap is on the LONG edge rather than the short one because it
    bounds the file size: a short-edge cap lets an extreme panorama
    grow without limit, while a long-edge cap keeps every thumbnail
    within size x size.
    """
    if size is None:
        size = THUMBNAIL_SIZE
    if width <= 0 or height <= 0:
        return (size, size)
    scale = size / max(width, height)
    if scale >= 1.0:
        # Already within the cap — keep the source dimensions.
        return (width, height)
    return (max(1, round(width * scale)), max(1, round(height * scale)))
# Quality 80 — top of the Instagram / Google Photos / Apple Photos
# range. Preserves fine detail (faces, fabric, foliage) without the
# ~20% byte bloat of going to 85+. Below 70 starts eating high-
# frequency texture. Round-36 had this at 50 which kept broad color
# but crushed detail — every tile looked like a watercolor when
# viewed on a HiDPI display.
THUMBNAIL_QUALITY = 80
# Force 4:4:4 chroma subsampling on the WebP encode. PIL's
# default for quality<95 is 4:2:0, which subsamples the chroma
# channels and visibly mutes saturated colors at high-contrast
# edges (lips, fabric folds, neon signs). libvips/sharp
# (Immich, PhotoPrism) force 4:4:4 at quality>=80; we adopt the
# same threshold for parity with the libvips pipeline. Output
# bytes grow ~10-20% per file; visual gain is concentrated on
# already-saturated content (skin tones and low-saturation
# backgrounds are unaffected).
CHROMA_SUBSAMPLING = "4:4:4"


# Resize in linear-light sRGB, not gamma-encoded sRGB. PIL's
# LANCZOS resize averages pixels in the sRGB (gamma-encoded)
# colorspace, which is mathematically wrong: averaging two sRGB
# values that differ in brightness does NOT yield the correct
# mid-tone (the average of `pow(x, 2.4)` ≠ `pow(avg(x), 2.4)`).
# The libvips pipeline (Immich, PhotoPrism) resizes in linear
# light and converts back; we mirror that here.
#
# The visual difference on K-pop photos is small (<0.3% on
# saturation/value/contrast vs. gamma-encoded resize) — see
# the measurement in references/thumbnail-color-pipeline.md
# — but the principle is right and the cost is ~3× the
# resize time for the extra gamma round-trip. Linear resize
# only runs at index time, not on serving.
_LINEAR_RESIZE = True

# Force the PIL path even when pyvips is importable. The unit
# tests set this to True so the PIL helpers are exercised in
# CI environments that don't have libvips installed; the
# indexer never sets it. Defaults to False (use pyvips when
# available).
_PIL_FORCE = False


# sRGB ICC profile bytes. PIL strips ICC profiles from PIL
# Images on crop/resize by default; without an explicit
# `icc_profile=` argument to `save()` the output WebP is
# untagged. Browsers default-interpret untagged WebP as sRGB,
# which is correct for Telegram JPEGs (no profile on input),
# but a tagged sRGB profile makes the intent explicit and
# rounds-trips cleanly through any tool that respects ICC
# (color-managed browsers, image editors, PhotoPrism/Immich
# side-by-side comparisons). The 588 bytes is the standard
# sRGB profile from ImageCms; cached at module load so the
# per-thumbnail encode doesn't rebuild it.
_SRGB_PROFILE_BYTES: bytes = ImageCms.ImageCmsProfile(
    ImageCms.createProfile("sRGB")
).tobytes()

# Cached path to the sRGB profile as a real file. pyvips'
# write_to_file wants a file path for the profile argument
# (not bytes), so we drop the bytes to a temp file once at
# module load. Deleted at process exit; the file is small
# (588 bytes) so the disk I/O cost is negligible.
_SRGB_PROFILE_PATH: str | None = None


def _ensure_srgb_profile_path() -> str:
    """Return the on-disk path of the sRGB ICC profile, writing it
    once per process if needed. pyvips' write_to_file accepts a
    file path for the `profile` argument; we don't want to call
    ImageCms in the hot path."""
    global _SRGB_PROFILE_PATH
    if _SRGB_PROFILE_PATH is None:
        import tempfile

        # NamedTemporaryFile with delete=False so the file
        # survives the with-block; the OS will clean up
        # /tmp on reboot. We never delete it explicitly —
        # the bytes are tiny and a stale copy is harmless.
        fd, path = tempfile.mkstemp(suffix='.icc', prefix='srgb-')
        os.write(fd, _SRGB_PROFILE_BYTES)
        os.close(fd)
        _SRGB_PROFILE_PATH = path
    return _SRGB_PROFILE_PATH


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded channel to linear-light, per IEC 61966-2-1.

    The sRGB curve has a linear segment near black (channel <= 0.04045)
    and a power-2.4 segment above. Inverting it is what linear-light
    resampling needs to operate on physically-meaningful values.
    """
    return np.where(
        channel <= 0.04045,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(channel: np.ndarray) -> np.ndarray:
    """Convert linear-light channel back to sRGB-encoded.

    Inverse of `_srgb_to_linear`. The output range is [0, 1] in
    float; round/clip to uint8 before re-assembling the image.
    """
    return np.where(
        channel <= 0.0031308,
        channel * 12.92,
        1.055 * (np.maximum(channel, 0.0) ** (1.0 / 2.4)) - 0.055,
    )


def _resize_in_linear_light(
    thumb: Image.Image, size: tuple[int, int]
) -> Image.Image:
    """Resize `thumb` to `size` in linear-light sRGB, then convert back.

    PIL's RGB resize averages sRGB-encoded pixel values, which is the
    mathematically-wrong place to interpolate. This helper does the
    full round-trip:
      1. uint8 RGB  →  float32 linear-light  (per channel)
      2. LANCZOS resize each channel independently
      3. linear-light  →  sRGB-encoded
      4. uint8 RGB

    Channels are split during the resize because PIL's LANCZOS works
    on 2D planes, not 3-channel tensors. Splitting is 3× the
    resize() calls but each is a tiny per-channel op; the total
    cost is ~3-5ms per thumbnail on a typical CPU.

    Input is expected to be RGB (the canonical thumb path calls
    `.convert("RGB")` before this). Output is RGB.
    """
    arr = np.asarray(thumb, dtype=np.float32) / 255.0
    out_w, out_h = size

    # Per-channel linear-light resize.
    resized_channels: list[np.ndarray] = []
    for c in range(3):
        linear = _srgb_to_linear(arr[:, :, c])
        # 16-bit container for the LANCZOS pass; PIL's LANCZOS on
        # float32 isn't supported across all Pillow versions and
        # the precision loss at 8-bit is visible on the dark
        # shadows that linearization reveals.
        lin16 = Image.fromarray(
            np.clip(linear * 65535.0, 0, 65535).astype(np.uint16),
            mode="I;16",
        )
        resized = lin16.resize((out_w, out_h), Image.Resampling.LANCZOS)
        resized_channels.append(
            np.asarray(resized, dtype=np.float32) / 65535.0
        )

    # Stack, convert back to sRGB, clip, return as RGB uint8.
    linear_stack = np.stack(resized_channels, axis=-1)
    srgb_stack = _linear_to_srgb(linear_stack)
    srgb_u8 = np.clip(np.round(srgb_stack * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(srgb_u8, mode="RGB")


logger = logging.getLogger(__name__)


def thumbnail_path(point_id: str) -> Path:
    """Compute deterministic path from point ID."""
    prefix = point_id[:2]
    return Path(THUMBNAIL_DIR) / prefix / f"{point_id}.webp"


def sized_thumbnail_path(point_id: str, width: int) -> Path:
    """Compute deterministic path for a sized variant.

    Matches the look-up `search/routers/thumbnails.py` does when the
    frontend asks for `?w={width}`. We always write the canonical
    384px file *and* these siblings so the endpoint's 404-fallback to
    the canonical file is a true safety net, not a routine path.
    """
    prefix = point_id[:2]
    return Path(THUMBNAIL_DIR) / prefix / f"{point_id}.w{width}.webp"


def _use_pyvips() -> bool:
    """True when the pyvips (libvips) path should be used.

    pyvips is preferred because it does all four libvips behaviors
    (4:4:4 chroma, linear-light resize, ICC profile embed, fast
    C-based encode) as a package. The PIL path is the same set of
    behaviors implemented in pure Python + numpy, used when libvips
    isn't available (CI runners, dev machines without the system
    package installed). The unit tests force the PIL path with
    `_PIL_FORCE = True` so the PIL helpers get exercised in any
    environment.
    """
    return _HAS_PYVIPS and not _PIL_FORCE


def _write_canonical_pyvips(src: Any, point_id: str) -> tuple[Path, Any]:
    """Write ONLY the canonical thumbnail.

    `src` is an already-open libvips image, so the caller can reuse
    the same decode for the model's input.

    The canonical is the ONE asset the indexer cannot defer: it is
    what the embedder consumes (the model input IS the canonical's
    pixels when MODEL_INPUT_SIZE == THUMBNAIL_SIZE), and it is the
    fallback the serving endpoint returns when a sized variant is
    missing.

    Returns the canonical path and the resized canonical image.
    """
    out_path = thumbnail_path(point_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and resize to fit INSIDE a THUMBNAIL_SIZE box, keeping
    # the source's aspect ratio. No crop: the thumbnail shows the
    # whole picture, so a portrait stays portrait.
    thumb = src.thumbnail_image(
        THUMBNAIL_SIZE,
        height=THUMBNAIL_SIZE,
    )
    # WebP encode. libvips auto-selects 4:4:4 chroma at Q>=80
    # (matches Immich/PhotoPrism). effort=4 mirrors the
    # PIL `method=4` setting (slowest encoder, best size).
    # The `profile` argument embeds the sRGB ICC profile
    # in the output.
    thumb.write_to_file(
        str(out_path),
        Q=THUMBNAIL_QUALITY,
        effort=4,
        profile=_ensure_srgb_profile_path(),
    )

    return out_path, thumb


def _write_variants_pyvips(src: Any, point_id: str) -> dict[int, Any]:
    """Write the sized display variants (everything below the canonical).

    Split out from the canonical write so the caller can run it
    OFF the indexer's critical path. The variants are display-only
    assets — the serving endpoint falls back to the canonical when
    one is missing — so nothing downstream depends on them being
    on disk by the time the embed finishes.

    NO crop: a variant keeps the source's aspect ratio, exactly
    like the canonical. `crop="centre"` here would make the variant
    a SQUARE while the canonical is native-aspect — so the same
    photo would be cropped in the small asset and whole in the large
    one, and the masonry's no-crop promise would hold only for the
    tiles that happened to get the canonical.
    """
    resized: dict[int, Any] = {}
    for size in THUMBNAIL_VARIANT_SIZES:
        if size >= THUMBNAIL_SIZE:
            continue
        variant = src.thumbnail_image(
            size,
            height=size,
        )
        resized[size] = variant
        variant_path = sized_thumbnail_path(point_id, size)
        variant.write_to_file(
            str(variant_path),
            Q=THUMBNAIL_QUALITY,
            effort=4,
            profile=_ensure_srgb_profile_path(),
        )
    return resized


def _write_ladder_pyvips(src: Any, point_id: str) -> tuple[Path, dict[int, Any]]:
    """Write the canonical thumbnail + every sized variant.

    The canonical and its variants are written in one call, which is
    what the PIL fallback path and the standalone regen path use.
    The indexer's hot path uses `_write_canonical_pyvips` +
    `_write_variants_pyvips` separately so the variants can run
    asynchronously.

    Returns the canonical path and the resized images keyed by the
    size they were made at.
    """
    out_path, thumb = _write_canonical_pyvips(src, point_id)
    resized: dict[int, Any] = {THUMBNAIL_SIZE: thumb}
    resized.update(_write_variants_pyvips(src, point_id))
    return out_path, resized


def _compute_thumbnail_pyvips(path: Path, point_id: str) -> Path | None:
    """libvips path: thumbnail_image + WebP encode with ICC + 4:4:4.

    Uses `pyvips.Image.thumbnail_image()` for the smart-crop +
    resize in one call (Lanczos 3-lobe, linear-light, color-managed
    internally — these are the libvips defaults we want). The WebP
    save uses Q=80, embeds the sRGB ICC profile via the `profile`
    argument. libvips's WebP writer automatically uses 4:4:4 chroma
    at Q>=80 (no explicit flag, unlike PIL).

    Args:
        path: source image file path. pyvips reads from disk
            directly (no PIL intermediate), which is part of why
            it is faster than the PIL path.
        point_id: Qdrant point ID (sha1 hash) — used to compute
            the deterministic on-disk path.
    """
    # Late import so the module loads on hosts without pyvips;
    # `_use_pyvips()` already gated us.
    import pyvips as _pv

    try:
        img: Any = _pv.Image.new_from_file(str(path))
        out_path, _resized = _write_ladder_pyvips(img, point_id)
        return out_path
    except _pv.Error as e:
        logger.warning("libvips failed for %s: %s", point_id, e)
        return None
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("Failed to generate thumbnail for %s: %s", point_id, e)
        return None


def _vips_to_pil(img: Any) -> Image.Image:
    """A libvips image as a PIL RGB image, without a file round-trip."""
    if img.hasalpha():
        img = img.flatten(background=[255, 255, 255])
    if img.bands != 3:
        img = img.extract_band(0, n=3)
    return Image.frombuffer(
        "RGB", (img.width, img.height), img.write_to_memory()
    )


def _libvips_display_size(src: Any) -> tuple[int, int]:
    """The DISPLAY dimensions of a libvips image, honouring EXIF orientation.

    `pyvips.Image.new_from_file()` does NOT apply the EXIF orientation
    tag — only `thumbnail_image()` does, and only since libvips 8.15.
    So the buffer's own axes are the camera's, and a sideways photo
    reads as landscape until the tag is applied.

    The tag is in the image's metadata: libvips exposes the EXIF
    orientation as the `orientation` field.
    """
    orientation: int | None = None
    try:
        orientation = int(src.get("orientation"))
    except Exception:  # noqa: BLE001 - the field is absent on most images
        orientation = None
    return display_size(src.width, src.height, orientation)


class PreparedPhoto(NamedTuple):
    """Everything ONE read of a photo's bytes produced.

    `variants` maps a display size to its libvips image. Those images
    are lazily bound to the source bytes, so `source_bytes` has to stay
    alive until they are written.
    """

    model_input: Image.Image | None
    source_width: int | None
    source_height: int | None
    fingerprints: dict[str, str | None] | None
    blurhash: str | None
    variants: dict[int, Any]
    source_bytes: bytes | None


def _fingerprints_and_blurhash_from_pixels(
    canonical: Image.Image, raw: bytes
) -> tuple[dict[str, str | None], str | None]:
    """The content hash, the perceptual hash and the blurhash, from ONE read.

    The content hash is the sha256 of the file's bytes — the same
    value as before, and the only thing that still needs the whole
    file.

    The perceptual hash and the blurhash are computed from the
    CANONICAL THUMBNAIL's pixels rather than from a full-resolution
    decode. That decode was ~354 ms per 30 MP photo — the largest
    CPU item in the run, and the reason the card sat idle waiting for
    it.

    The dhash is therefore a hash of the 640-pixel picture, not of the
    30 MP one. It still groups near-duplicates, but a photo indexed
    before this change and a copy indexed after it can compare
    unequal: the hash function is not uniform across a corpus that
    spans the change. The blurhash moves with it, and for a
    placeholder that is a better match anyway — it is now the picture
    the user actually sees.
    """
    from indexer.blurhash import compute_blurhash
    from indexer.fingerprints import compute_fingerprints

    fingerprints = compute_fingerprints(canonical, sha_bytes=raw)
    blurhash = compute_blurhash(canonical)
    return fingerprints, blurhash


def prepare_photo(
    path: Path,
    point_id: str,
    *,
    model_input_size: int | None = None,
    want_variants: bool = True,
    want_fingerprints: bool = False,
) -> PreparedPhoto:
    """Prepare one photo from ONE read of its bytes.

    Everything comes from that read: the canonical thumbnail, the
    embedder's input, the sized display variants, the recorded
    dimensions, the content hash and the perceptual hash.

    The hot path used to read the file FOUR times — the libvips
    decode, the variant writer's own decode, the content hash's byte
    read and the dhash's PIL open — and each of those was a full
    transfer of the source over the NAS. On a 30 MP phone photo that
    is ~8 MB per read, and the reads were the run's ceiling: the GPU
    sat at 0% while the indexer waited on the network.

    Nothing about the output changes. libvips decodes the bytes to the
    same image it decodes the file to (verified byte-identical
    thumbnails), the variant resizes are the same resizes, and the
    dhash is still PIL's hash of the full-resolution picture, so every
    payload value is the value it was.
    """
    if model_input_size is None or not _use_pyvips():
        # A letterbox model wants the picture AS DECODED — it pads the
        # picture into a square itself — and the PIL path is the one
        # that has it.
        return _prepare_photo_pil(
            path, point_id, model_input_size=model_input_size
        )

    import pyvips as _pv

    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return PreparedPhoto(None, None, None, None, None, {}, None)

    try:
        src: Any = _pv.Image.new_from_buffer(raw, "")
        _out_path, thumb = _write_canonical_pyvips(src, point_id)
        source_w, source_h = _libvips_display_size(src)

        # The canonical's pixels — what the embedder consumes and what
        # the fingerprints are computed from. Taken before the WebP
        # encode, so nothing is recompressed on the way to the GPU.
        canonical_pil: Image.Image | None = None

        model_input = None
        if model_input_size is not None:
            if model_input_size == THUMBNAIL_SIZE:
                model_input = _vips_to_pil(thumb)
            else:
                model_input = _vips_to_pil(
                    src.thumbnail_image(
                        model_input_size, height=model_input_size
                    )
                )

        variants: dict[int, Any] = {}
        if want_variants:
            for size in THUMBNAIL_VARIANT_SIZES:
                if size >= THUMBNAIL_SIZE:
                    continue
                variants[size] = src.thumbnail_image(size, height=size)

        fingerprints = None
        blurhash = None
        if want_fingerprints:
            if canonical_pil is None:
                canonical_pil = _vips_to_pil(thumb)
            fingerprints, blurhash = _fingerprints_and_blurhash_from_pixels(
                canonical_pil, raw
            )
        return PreparedPhoto(
            model_input, source_w, source_h, fingerprints, blurhash, variants, raw
        )
    except _pv.Error as e:
        logger.warning("libvips failed for %s: %s", point_id, e)
        return PreparedPhoto(None, None, None, None, None, {}, None)
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(
            "Failed to generate thumbnail for %s: %s", point_id, e
        )
        return PreparedPhoto(None, None, None, None, None, {}, None)


def _prepare_photo_pil(
    path: Path,
    point_id: str,
    *,
    model_input_size: int | None = None,
) -> PreparedPhoto:
    """The PIL fallback: one decode, no libvips.

    PIL's path never had the second decode (the variants come from
    the same image), so it needs none of the one-read machinery.
    """
    from indexer.image_loader import load_image_pil_with_dims

    try:
        image, source_w, source_h = load_image_pil_with_dims(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return PreparedPhoto(None, None, None, None, None, {}, None)

    _pil_out = _compute_thumbnail_pil(image, point_id)

    # The fingerprints come from the canonical's pixels, exactly as in
    # the libvips path, so the perceptual hash is the same function of
    # the picture whichever pipeline produced it.
    fingerprints = None
    blurhash = None
    try:
        canonical = _canonical_pil_from_pil(image)
        fingerprints, blurhash = _fingerprints_and_blurhash_from_pixels(
            canonical, path.read_bytes()
        )
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("fingerprint failed for %s: %s", point_id, e)

    model_input = None
    if model_input_size is None:
        # Letterbox model: the picture AS DECODED — it pads the
        # picture into a square itself.
        model_input = image
    else:
        target = thumbnail_target_size(
            image.width, image.height, model_input_size
        )
        try:
            if _LINEAR_RESIZE:
                model_input = _resize_in_linear_light(
                    image.convert("RGB"), target
                )
            else:
                model_input = image.resize(
                    target, Image.Resampling.LANCZOS
                )
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("model input resize failed for %s: %s", point_id, e)
    return PreparedPhoto(
        model_input, source_w, source_h, fingerprints, blurhash, {}, None
    )


def write_variants_from_images(
    variants: dict[int, Any], point_id: str
) -> int:
    """Write already-resized display variants. Returns the count."""
    written = 0
    for size, variant in variants.items():
        variant_path = sized_thumbnail_path(point_id, size)
        variant.write_to_file(
            str(variant_path),
            Q=THUMBNAIL_QUALITY,
            effort=4,
            profile=_ensure_srgb_profile_path(),
        )
        written += 1
    return written


def generate_canonical_and_model_input(
    path: Path,
    point_id: str,
    *,
    model_input_size: int | None = None,
) -> tuple[Path | None, Image.Image | None, int | None, int | None]:
    """Write ONLY the canonical thumbnail; return the model's input.

    The canonical is the one asset the indexer cannot defer: the
    embedder consumes its pixels, and the serving endpoint falls
    back to it when a sized variant is missing.

    The sized display variants are written separately by
    `generate_variants_for_path`, which the indexer runs OFF the
    critical path. Measured on prod: the three WebP encodes are
    ~195 ms/photo against a 21 ms decode, and the variant writes
    land on NFS at ~30 ms each — so writing them inline kept the
    GPU idle ~68% of a run.

    Returns `(thumbnail_path, model_input, source_width,
    source_height)`. `model_input` is None when `model_input_size`
    is None or the source cannot be prepared.
    """
    if _use_pyvips():
        # Late import so the module loads on hosts without pyvips.
        import pyvips as _pv

        try:
            src: Any = _pv.Image.new_from_file(str(path))
            out_path, thumb = _write_canonical_pyvips(src, point_id)
            source_w, source_h = _libvips_display_size(src)

            model_input = None
            if model_input_size is not None:
                if model_input_size == THUMBNAIL_SIZE:
                    # The model input IS the canonical's pixels,
                    # taken before the WebP encode so nothing is
                    # recompressed.
                    model_input = _vips_to_pil(thumb)
                else:
                    model_input = _vips_to_pil(
                        src.thumbnail_image(
                            model_input_size, height=model_input_size
                        )
                    )
            return out_path, model_input, source_w, source_h
        except _pv.Error as e:
            logger.warning("libvips failed for %s: %s", point_id, e)
            return None, None, None, None
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(
                "Failed to generate thumbnail for %s: %s", point_id, e
            )
            return None, None, None, None

    # PIL fallback: decode once, use the decode for both the ladder
    # and the model input.
    from indexer.image_loader import load_image_pil_with_dims

    try:
        image, source_w, source_h = load_image_pil_with_dims(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None, None, None, None

    pil_out: Path | None = _compute_thumbnail_pil(image, point_id)

    model_input = None
    if model_input_size is not None:
        target = thumbnail_target_size(
            image.width, image.height, model_input_size
        )
        try:
            if _LINEAR_RESIZE:
                model_input = _resize_in_linear_light(
                    image.convert("RGB"), target
                )
            else:
                model_input = image.resize(
                    target, Image.Resampling.LANCZOS
                )
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("model input resize failed for %s: %s", point_id, e)
    return pil_out, model_input, source_w, source_h


def generate_variants_for_path(path: Path, point_id: str) -> int:
    """Write the sized display variants for `path`. Returns the count.

    Runs off the indexer's critical path: the variants are
    display-only assets, and the serving endpoint falls back to the
    canonical when one is missing.

    Re-decodes the source (the canonical's decode is long gone by
    the time this runs, and libvips images are lazily bound to their
    file). A source that cannot be decoded yields 0 variants, and
    the canonical — already on disk — still serves.
    """
    if not THUMBNAIL_VARIANT_SIZES:
        return 0

    if _use_pyvips():
        import pyvips as _pv

        try:
            src: Any = _pv.Image.new_from_file(str(path))
            resized = _write_variants_pyvips(src, point_id)
            return len(resized)
        except _pv.Error as e:
            logger.warning("libvips failed for %s: %s", point_id, e)
            return 0
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(
                "Failed to generate variants for %s: %s", point_id, e
            )
            return 0

    from indexer.image_loader import load_image_pil_with_dims

    try:
        image, _sw, _sh = load_image_pil_with_dims(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return 0

    written = 0
    for size in THUMBNAIL_VARIANT_SIZES:
        if size >= THUMBNAIL_SIZE:
            continue
        target = thumbnail_target_size(image.width, image.height, size)
        variant_path = sized_thumbnail_path(point_id, size)
        variant_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if _LINEAR_RESIZE:
                variant_img = _resize_in_linear_light(
                    image.convert("RGB"), target
                )
            else:
                variant_img = image.resize(target, Image.Resampling.LANCZOS)
            variant_img.save(
                variant_path,
                format="WEBP",
                quality=THUMBNAIL_QUALITY,
            )
            written += 1
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(
                "Failed to write variant %s for %s: %s", size, point_id, e
            )
    return written


def generate_ladder_and_model_input(
    path: Path,
    point_id: str,
    *,
    model_input_size: int | None = None,
) -> tuple[Path | None, Image.Image | None, int | None, int | None]:
    """Write the display ladder for `path` and return the model's input.

    ONE decode of the source produces both: every display size AND
    the image the embedder consumes. The model input is the source at
    its native aspect ratio with its long edge at `model_input_size` —
    the same picture the display serves, at the model's resolution.

    When the display ladder contains a variant of exactly that size,
    the model input IS that variant's pixels (taken before the WebP
    encode, so nothing is recompressed): the photo is decoded once
    and resized once instead of twice, and the embedding cannot
    drift from the asset the user sees.

    Returns `(thumbnail_path, model_input, source_width,
    source_height)`. `model_input` is None when `model_input_size` is
    None or the source cannot be prepared.
    """
    if _use_pyvips():
        # Late import so the module loads on hosts without pyvips.
        import pyvips as _pv

        try:
            src: Any = _pv.Image.new_from_file(str(path))
            out_path, resized = _write_ladder_pyvips(src, point_id)
            source_w, source_h = src.width, src.height

            model_input = None
            if model_input_size is not None:
                if model_input_size in resized:
                    model_input = _vips_to_pil(resized[model_input_size])
                else:
                    model_input = _vips_to_pil(
                        src.thumbnail_image(
                            model_input_size, height=model_input_size
                        )
                    )
            return out_path, model_input, source_w, source_h
        except _pv.Error as e:
            logger.warning("libvips failed for %s: %s", point_id, e)
            return None, None, None, None
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(
                "Failed to generate thumbnail for %s: %s", point_id, e
            )
            return None, None, None, None

    # PIL fallback: decode once, use the decode for both the ladder
    # and the model input.
    from indexer.image_loader import load_image_pil_with_dims

    try:
        image, source_w, source_h = load_image_pil_with_dims(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None, None, None, None

    out_path = _compute_thumbnail_pil(image, point_id)

    model_input = None
    if model_input_size is not None:
        target = thumbnail_target_size(
            image.width, image.height, model_input_size
        )
        try:
            if _LINEAR_RESIZE:
                model_input = _resize_in_linear_light(
                    image.convert("RGB"), target
                )
            else:
                model_input = image.resize(
                    target, Image.Resampling.LANCZOS
                )
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("model input resize failed for %s: %s", point_id, e)
    return out_path, model_input, source_w, source_h



def compute_thumbnail(
    image: Image.Image, point_id: str, source_path: Path | None = None
) -> Path | None:
    """Generate 384px WebP thumbnail from a PIL Image (or file path).

    The pyvips (libvips) path is the primary encoder; PIL is the
    fallback. The pyvips path requires a file path, so if the
    caller has one, pass `source_path=` and we route to libvips.
    With `source_path=` the caller's `image` is used ONLY for the
    model input (the caller already decoded it); the thumbnail itself
    is produced by libvips reading the file directly.

    Args:
        image: PIL Image already opened by the pipeline. Used by
            the PIL path; ignored when `source_path` is given and
            libvips is available.
        point_id: Qdrant point ID (sha1 hash).
        source_path: optional source file path. When provided
            AND libvips is importable, the libvips path runs
            (no PIL intermediate, ~10x faster). When None, the
            PIL path runs.

    Returns:
        Path to the written thumbnail file, or None on failure.
    """
    if _use_pyvips() and source_path is not None:
        return _compute_thumbnail_pyvips(source_path, point_id)
    return _compute_thumbnail_pil(image, point_id)


def compute_model_input(
    image: Image.Image,
    source_path: Path | None = None,
    *,
    model_input_size: int,
) -> Image.Image | None:
    """The image the embedder should consume for `source_path`.

    The model input is the SOURCE at its native aspect ratio with
    its long edge at `model_input_size` — the same framing the
    display thumbnail has, at the model's own resolution.

    Returns None when the input cannot be produced (the caller then
    falls back to feeding the decoded source).
    """
    if _use_pyvips() and source_path is not None:
        try:
            import pyvips as _pv

            src: Any = _pv.Image.new_from_file(str(source_path))
            resized = src.thumbnail_image(
                model_input_size, height=model_input_size
            )
            return _vips_to_pil(resized)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "model input from %s failed: %s", source_path, exc
            )
            return None

    target = thumbnail_target_size(image.width, image.height, model_input_size)
    try:
        if _LINEAR_RESIZE:
            return _resize_in_linear_light(image.convert("RGB"), target)
        return image.resize(target, Image.Resampling.LANCZOS)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("model input resize failed: %s", exc)
        return None


def _canonical_pil_from_pil(image: Image.Image) -> Image.Image:
    """The canonical thumbnail's pixels, in the PIL pipeline.

    Mirrors `_compute_thumbnail_pil`'s resize so the fingerprints
    computed from this picture match the ones the libvips path
    produces from the same source.
    """
    target = thumbnail_target_size(image.width, image.height)
    if _LINEAR_RESIZE:
        return _resize_in_linear_light(image.convert("RGB"), target)
    return image.resize(target, Image.Resampling.LANCZOS)


def _compute_thumbnail_pil(image: Image.Image, point_id: str) -> Path | None:
    """PIL fallback path: numpy-based linear-light resize + WebP save.

    Mirrors what libvips does, in pure Python + numpy + PIL. Used
    when pyvips isn't importable (libvips system package missing)
    or when `_PIL_FORCE = True` (the test-suite toggle).
    """
    try:
        out_path = thumbnail_path(point_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Work on a copy to avoid mutating the pipeline's image.
        # Resize to the native-aspect target — no crop, no letterbox:
        # the long edge becomes THUMBNAIL_SIZE and the short edge
        # follows the source's aspect ratio.
        thumb = image.copy()
        # Unpack explicitly: a malformed image raises ValueError here,
        # which the except below catches, rather than a TypeError
        # from a star-unpack of a bogus size.
        src_w, src_h = thumb.size
        target = thumbnail_target_size(src_w, src_h)
        # Resize in linear-light sRGB (see _resize_in_linear_light
        # for why). Gamma correction is only meaningful at the
        # resampling step, where neighboring pixels are
        # weighted-averaged.
        if _LINEAR_RESIZE:
            thumb_rgb = thumb.convert("RGB")
            thumb = _resize_in_linear_light(thumb_rgb, target)
        else:
            thumb = thumb.resize(target, Image.Resampling.LANCZOS)
        # Force 4:4:4 chroma subsampling on the WebP encode. PIL's
        # default for quality<95 is 4:2:0, which subsamples the
        # chroma channels and visibly mutes saturated colors at
        # high-contrast edges (lips, fabric folds, neon signs).
        # libvips/sharp (Immich, PhotoPrism) force 4:4:4 at
        # quality>=80; we adopt the same threshold for parity with
        # the libvips pipeline. Output bytes grow ~10-20% per
        # file; visual gain is concentrated on already-saturated
        # content (skin tones and low-saturation backgrounds are
        # unaffected). Embed the sRGB ICC profile so the WebP is
        # explicitly color-managed (browsers with strict color
        # management render tagged sRGB slightly more accurately
        # than untagged; image editors round-trip cleanly).
        thumb.save(
            out_path,
            "WEBP",
            quality=THUMBNAIL_QUALITY,
            method=4,
            chroma_subsampling=CHROMA_SUBSAMPLING,
            icc_profile=_SRGB_PROFILE_BYTES,
        )

        # Round-perf (issue #2): write the downscaled siblings for the
        # frontend srcset. The 256-px canonical is always written
        # first; siblings are extra variants the browser may pick.
        #
        # All advertised sizes are smaller than the canonical edge
        # (we control THUMBNAIL_VARIANT_SIZES), so the loop below always runs
        # at least once. If a future maintainer adds a variant that's
        # >= the canonical, we'd be upscaling — which is worse than
        # serving the canonical file. The endpoint's 404-fallback to
        # the canonical handles that automatically; we skip the write
        # to avoid garbage files on disk.
        for size in THUMBNAIL_VARIANT_SIZES:
            if size >= THUMBNAIL_SIZE:
                continue
            # Same linear-light resize as the canonical thumb so
            # the srcset doesn't have a per-size color shift. The
            # variant keeps the canonical's aspect ratio — only the
            # long edge shrinks.
            variant_target = thumbnail_target_size(src_w, src_h, size)
            if _LINEAR_RESIZE:
                variant = _resize_in_linear_light(thumb, variant_target)
            else:
                variant = thumb.resize(variant_target, Image.Resampling.LANCZOS)
            variant_path = sized_thumbnail_path(point_id, size)
            # Same 4:4:4 chroma + sRGB profile as the canonical;
            # sized variants are an edge case (the ladder is
            # currently empty) but the encode should match the
            # canonical for consistent color rendering across
            # the srcset.
            variant.save(
                variant_path,
                "WEBP",
                quality=THUMBNAIL_QUALITY,
                method=4,
                chroma_subsampling=CHROMA_SUBSAMPLING,
                icc_profile=_SRGB_PROFILE_BYTES,
            )

        return out_path
    except (OSError, ValueError, RuntimeError) as e:
        # Thumbnail decode/save can fail on corrupt images,
        # missing files, or PIL/encoder issues. The caller expects
        # None on any failure (no payload = no placeholder), but
        # narrow to realistic failure modes rather than catch-all.
        logger.warning("Failed to generate thumbnail for %s: %s", point_id, e)
        return None


def generate_thumbnail_for_path(
    image: Image.Image, path: Path, shard: str = ""
) -> Path | None:
    """Generate thumbnail for a given source image path (used by local_sync).

    Computes the point_id from the path (same as Qdrant upsert), then
    generates the thumbnail. This ensures the thumbnail path matches
    what the search side will look for.

    When pyvips is importable, the file `path` is passed through to
    libvips directly (no PIL intermediate, ~10x faster than the
    numpy-backed PIL path). When pyvips is not importable, the
    in-memory `image` is used with the PIL fallback.
    """
    point_id = id_for(path, shard)
    return compute_thumbnail(image, point_id, source_path=path)
