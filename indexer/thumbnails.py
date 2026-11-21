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
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageCms

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
# This is the same number the C-RADIOv4 embedder uses for its input's
# long edge — the model's input IS the thumbnail's geometry, so the
# embedding sees exactly the framing the thumbnail shows. See
# `lustre_kernel/_radio_models.py`.
#
# The default is unchanged from the SigLIP2 era (384 = the old model
# input) so a deployment that does not set it keeps its current
# storage footprint; the geometry change alone makes the files
# slightly SMALLER for non-square sources.
THUMBNAIL_SIZE: int = int(os.environ.get("THUMBNAIL_SIZE", "384"))


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
# Sized variants the frontend srcset advertises. Post-migration
# this is empty: a single 384px asset covers the common rendering
# sizes (up to 192 CSS px at 2× DPR = 384 px at the bitmap level)
# without the storage overhead of three variants. The frontend
# requests `?w=384` and the router serves the canonical 384px file.
THUMBNAIL_SIZES: tuple[int, ...] = ()


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
            this is faster than the PIL path.
        point_id: Qdrant point ID (sha1 hash) — used to compute
            the deterministic on-disk path.
    """
    # Late import so the module loads on hosts without pyvips;
    # `_use_pyvips()` already gated us.
    import pyvips as _pv

    out_path = thumbnail_path(point_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Load and resize to fit INSIDE a THUMBNAIL_SIZE box, keeping
        # the source's aspect ratio. No crop: the thumbnail shows the
        # whole picture, so a portrait stays portrait.
        img: Any = _pv.Image.new_from_file(str(path))
        thumb = img.thumbnail_image(
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

        # Sized variants. Same pipeline at smaller dimensions.
        for size in THUMBNAIL_SIZES:
            if size >= THUMBNAIL_SIZE:
                continue
            variant = img.thumbnail_image(
                size,
                height=size,
                crop="centre",
            )
            variant_path = sized_thumbnail_path(point_id, size)
            variant.write_to_file(
                str(variant_path),
                Q=THUMBNAIL_QUALITY,
                effort=4,
                profile=_ensure_srgb_profile_path(),
            )

        return out_path
    except _pv.Error as e:
        logger.warning("libvips failed for %s: %s", point_id, e)
        return None
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("Failed to generate thumbnail for %s: %s", point_id, e)
        return None


def compute_thumbnail(
    image: Image.Image, point_id: str, source_path: Path | None = None
) -> Path | None:
    """Generate 384px WebP thumbnail from a PIL Image (or file path).

    The pyvips (libvips) path is the primary encoder; PIL is the
    fallback. The pyvips path requires a file path, so if the
    caller has one, pass `source_path=` and we route to libvips.
    Without a path, the PIL path runs (it's slower but always
    available). Both paths produce visually equivalent output:
    4:4:4 chroma, linear-light resize, sRGB ICC profile embed.

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
        # (we control THUMBNAIL_SIZES), so the loop below always runs
        # at least once. If a future maintainer adds a variant that's
        # >= the canonical, we'd be upscaling — which is worse than
        # serving the canonical file. The endpoint's 404-fallback to
        # the canonical handles that automatically; we skip the write
        # to avoid garbage files on disk.
        for size in THUMBNAIL_SIZES:
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
            # sized variants are an edge case (THUMBNAIL_SIZES is
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
