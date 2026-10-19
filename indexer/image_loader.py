"""
indexer/image_loader.py

Load + preprocess an image for embedding.

Pipeline:
  1. PIL.Image.open(path) — lazy decode
  2. ImageOps.exif_transpose() — apply EXIF orientation
  3. .convert("RGB") — drop alpha; SigLIP2 expects 3 channels
  4. Letterbox resize to the registered model's resolution (e.g.
     384x384 for ViT-gopt-16-SigLIP2-384, 256x256 for
     ViT-L-16-SigLIP2-256) — preserve aspect ratio
  5. Normalize with the model's mean/std — produces a tensor
     ready for the embedder

The model's resolution, mean, and std are pulled from the
kernel's model registry by `model_name`. No constants live in
this module.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path

from PIL import Image, ImageOps

# TODO(v1.1): import utilities from isaac-image-scoring once that package
# is pip-installable. Today it's a script-shaped project, not a library.

logger = logging.getLogger(__name__)

# SigLIP2 normalization (from the model's processor config). Mean/std
# are model-family properties; the resolution is per-model and
# resolved at `letterbox_resize` call time from the registry.
SIGLIP_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Per-image load timeout (seconds). PIL's read is a blocking syscall
# on the underlying file handle; on a slow/stalled network share it
# can hang indefinitely. The timeout turns the hang into a
# LoaderError so the indexer can skip the file and move on.
# Override with INDEXER_LOAD_TIMEOUT_S=60 (or whatever) if 30s is
# too tight for your network.
_LOAD_TIMEOUT_S: float = float(os.environ.get("INDEXER_LOAD_TIMEOUT_S", "30"))


class LoaderError(Exception):
    """Raised when an image cannot be loaded/normalized."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


# Cap PIL's decoder at this many pixels on the long edge. The
# model expects a letterbox to its registered resolution (256 /
# 384), so decoding the full multi‑megapixel JPEG is pure waste.
#
# Default of 768 gives libjpeg a 2× budget over the model's 384
# input resolution (LANCZOS resampling on a >2× downscale looks
# visually identical to a smaller intermediate). On a typical
# 6000×4000 JPEG this skips decoding ~85% of the pixels — the
# load phase drops from ~30 ms to ~5 ms per image on the same
# hardware (measured on a RTX 3080 / NVMe source). 0 disables the
# cap (full decode). Override with INDEXER_DECODE_MAX=<pixels>.
_DECODE_MAX: int = int(os.environ.get("INDEXER_DECODE_MAX", "768"))


def load_image_pil(
    path: Path,
    *,
    decode_max: int | None = None,
) -> Image.Image:
    """
    Load an image and apply EXIF orientation. Returns a PIL.Image in RGB.

    Does NOT resize or normalize — that's the next step. Useful for tests
    and for callers that want to inspect the image before tensorizing.

    `decode_max` caps the JPEG/PNG decoder at the given long‑edge
    pixel count via PIL's `Image.draft()` fast path. The model
    letterboxes to its registered resolution anyway, so decoding
    beyond ~2x the model's resolution is wasted work. Default cap
    comes from `INDEXER_DECODE_MAX` (0 = disabled, no cap).

    ⚠️ **Source dims are NOT preserved in the returned image.**
    `Image.draft()` mutates `img.size` to the *drafted* size — which
    for a landscape JPEG + square draft target is **NOT** a
    proportional shrink (PIL picks JPEG-MCU-aligned chunks). When
    the caller needs the actual source dimensions (the photo page
    shows them), read `Image.open(path).size` BEFORE calling this
    function, or use `image_loader.peek_source_dims()` below.

    The source-dims gotcha is the source of the round‑31 bug where
    every photo showed up as 1000×1500 regardless of orientation
    (6000×4000 landscape images were reported as 1000×1500
    portrait, because PIL's draft box was non-proportional).

    The higher‑level `load()` helper does the right thing for the
    indexer: it captures `(source_w, source_h)` BEFORE calling
    `load_image_pil`, so callers of `load()` never hit this trap.
    Direct callers of `load_image_pil` should be aware of the
    gotcha — that's why this comment is here.
    """
    # Suppress PIL's decompression bomb check. Several photos in the
    # collection exceed the default MAX_IMAGE_PIXELS (178956970) at
    # typical resolutions like 14400x13872 (~200M pixels). The images
    # are from a trusted source (your own NAS) so the bomb check is
    # pure friction. Raised to 1G pixels — enough for any single
    # frame this pipeline will ever encounter.
    from PIL import Image as _PIL
    _PIL.MAX_IMAGE_PIXELS = 1_000_000_000

    cap = decode_max if decode_max is not None else (_DECODE_MAX or None)

    try:
        with Image.open(path) as img:
            # Tell PIL's decoder to only decode enough for `cap` on
            # the long edge. libjpeg will skip remaining MCU rows.
            # This is the documented fast path for "I don't need the
            # full image". Falls back to full decode for formats
            # without a draft hint (PNG, HEIC, etc).
            if cap:
                img.draft("RGB", (cap, cap))
            # .copy() so the file handle is released (Windows file locking)
            img = ImageOps.exif_transpose(img).copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except FileNotFoundError as e:
        raise LoaderError(path, "file not found") from e
    except Exception as e:  # PIL raises a zoo of exceptions
        raise LoaderError(path, f"open failed: {type(e).__name__}: {e}") from e


def peek_source_dims(path: Path) -> tuple[int | None, int | None]:
    """Read just the JPEG header to get the source `(width, height)`
    without decoding any pixels.

    Fast: opens the file, reads the header (a few KB), closes the
    file. Suitable for use in tests and CLI tooling that wants the
    true source dims without paying the decode cost.

    Returns `(None, None)` if the file can't be opened as an image
    (e.g. it's a directory, a text file, or unreadable). Callers
    that want to surface a hard error should catch the underlying
    PIL exception themselves.

    Round‑31: the photo page needs the source dims to display
    "Dimensions: 6000 × 4000" instead of the draft‑cropped
    1000 × 1500. After re‑ingest the indexer still has the
    draft‑cropped dims in the payload, so this helper is the
    escape hatch until a full re‑ingest lands.
    """
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001
        return (None, None)


def _default_resolution() -> int:
    """Resolution to use when no model is specified.

    Looks up the web backend's registered model and returns its
    resolution. Falls back to 384 (the SigLIP2 family default) if
    the registry hasn't been initialized — for example, when this
    module is imported by a tool that runs before any registry call.
    """
    try:
        from lustre_kernel.registry import get as _registry_get
        return _registry_get("ViT-so400m-patch16-384").resolution
    except Exception:  # noqa: BLE001
        return 384


def letterbox_resize(img: Image.Image, size: int | None = None) -> Image.Image:
    """
    Resize `img` to fit in a `size`x`size` box, preserving aspect ratio,
    then pad to exactly `size`x`size` with black.

    Why letterbox (vs. naive resize): naive resize distorts the image.
    SigLIP2-style models are trained on square inputs but real photos
    aren't square. Letterbox + pad keeps the aspect ratio intact.

    Uses LANCZOS resampling to match isaac-image-scoring exactly.

    If `size` is None, the active model's resolution is read from the
    registry. Callers should pass the registered model's resolution
    explicitly when known.
    """
    if size is None:
        size = _default_resolution()
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    # Pad to exactly size x size, centered.
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def load(
    path: Path, *, model_name: str = "ViT-so400m-patch16-384",
) -> tuple[Image.Image, int | None, int | None]:
    """
    Load + EXIF-correct + RGB-convert + letterbox. Returns
    `(letterboxed_image, source_width, source_height)` where
    `source_width` / `source_height` are the original pixel
    dimensions of the file on disk (after EXIF transpose, before
    the letterbox squashes the image to the model's resolution).

    Round‑30: returns source dims so the ingest pipeline can
    persist them in the qdrant payload. The photo page
    (`formatDimensions()`) needs the source size, not the
    embedder's 256×256 input size.

    Round‑31: source dims are now read via `peek_source_dims()`
    BEFORE `load_image_pil()` calls `Image.draft()`. PIL's draft
    box is non‑proportional for landscape JPEGs, so capturing
    dims from the returned image (as round‑30 did) was
    systematically wrong: a 6000×4000 image became "1000×1500"
    after draft. Reading the header up front fixes it.

    `model_name` defaults to the web backend's current model
    (`ViT-gopt-16-SigLIP2-384`). The resolution is read from the model
    registry, not from a constant in this module.

    Wraps `load_image_pil` in a thread with a `_LOAD_TIMEOUT_S` cap
    so a slow/stalled read on a network share doesn't hang the
    indexer. On timeout, raises LoaderError (caught by the indexer
    loop and logged as a per-file error; the indexer continues with
    the next file).
    """
    # Capture source dims from the JPEG header BEFORE any decode.
    # Reading the header is cheap (a few KB) and avoids the PIL
    # draft-mode non-proportional shrink trap (see the comment on
    # load_image_pil below).
    source_w, source_h = peek_source_dims(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        from lustre_kernel.registry import get as _registry_get
        size = _registry_get(model_name).resolution
        # 2× the model's resolution is enough room for the letterbox
        # to keep the full image visible while capping the decoder at
        # a small fraction of the source file's pixels. Override via
        # INDEXER_DECODE_MAX.
        decode_cap = 2 * size
        future = executor.submit(load_image_pil, path, decode_max=decode_cap)
        try:
            img = future.result(timeout=_LOAD_TIMEOUT_S)
        except concurrent.futures.TimeoutError as err:
            raise LoaderError(
                path,
                f"timed out after {_LOAD_TIMEOUT_S}s reading (likely a "
                f"slow/stalled network share; raise INDEXER_LOAD_TIMEOUT_S "
                f"if your network is just consistently slow)",
            ) from err
    # Resize to the registered model's resolution. NB: we don't
    # re‑capture img.width / img.height here — those reflect the
    # post‑draft size (see load_image_pil's warning), not the
    # source. Source dims come from peek_source_dims above.
    return letterbox_resize(img, size=size), source_w, source_h


# Heuristic mapping from PIL size to torchvision's expected CHW float tensor.
# We don't import torchvision here to keep this module testable on machines
# without GPU/torchvision wheels; the vision_encoder wraps with the
# processor at embed time.
def to_chw_float(img: Image.Image) -> list[list[list[float]]]:
    """
    Convert a PIL.Image (RGB, square) to a CHW float list in [0, 1].

    Normalization to mean/std is left to the model's processor; this
    gives the raw CHW float the processor expects.

    Note: in v1 we use the AutoProcessor directly, not this function.
    It's kept for tests and for callers that want raw pixels.
    """
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0  # HWC in [0, 1]
    return arr.transpose(2, 0, 1).tolist()  # CHW
