"""
lustre_kernel._radio_models

C-RADIOv4-SO400M registry entry.

NVIDIA's C-RADIOv4 is a vision-only ViT backbone distilled from
several teachers (SigLIP2-g, DINOv3, SAM3). The checkpoint ships
"teacher adaptors": projections that map the backbone's summary
token into each teacher's own embedding space. The `siglip2-g`
adaptor additionally carries SigLIP2-g's text tower and tokenizer,
so a single forward pass produces text-aligned vectors that are
directly comparable with text queries.

This module registers the model lazily, so importing the kernel
never pulls torch or downloads weights.

Three differences from the SigLIP2 path that the rest of the
codebase has to know about:

- **Native aspect ratio.** RADIO's ViT interpolates its position
  embeddings to whatever resolution it is handed, so the input does
  not have to be square. We resize the long edge to the model's
  resolution and snap both edges to a multiple of the patch size
  (16) instead of letterboxing onto a square canvas. The snapshot
  is therefore cropped-free and bar-free — the black bars the
  SigLIP2 letterbox adds are gone.

- **Variable-size batches.** Because the input is not square, a
  batch is only stackable when every image in it has the same
  shape. `embed_images` groups the images by shape and runs one
  forward pass per group. With the long edge pinned, the number of
  distinct shapes is bounded by the patch grid, so the grouping
  overhead is negligible.

- **Text alignment comes from the adaptor's teacher.** The text
  tower is SigLIP2-g, loaded by the adaptor itself — not by
  open_clip. The vector dim is SigLIP2-**giant**'s joint-space dim
  (1536), which is not the 1152 the SigLIP2 `so400m` variant
  produces — so a C-RADIOv4 collection is a different dim AND a
  different space from every SigLIP2 collection.

NVIDIA's C-RADIO code is licensed under the NVIDIA Source Code
License for RADIO, not an open-source licence, so none of it is
vendored here. It is fetched at runtime by `torch.hub` into
`TORCH_HOME` and executed in place.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from typing import Any

from PIL import Image

from lustre_kernel.registry import ModelSpec, Registry

__all__ = ["MODEL_INPUT_SIZE", "register_into"]

logger = logging.getLogger(__name__)

# The GitHub repo torch.hub clones. NVIDIA's; fetched at runtime.
RADIO_HUB_REPO = "NVlabs/RADIO"

# The checkpoint version key. Resolves to
# nvidia/C-RADIOv4-SO400M (the SO400M backbone) on the HF hub.
RADIO_VERSION = "c-radio_v4-so400m"

# The teacher adaptor whose space we index in. `siglip2-g` is the
# only C-RADIOv4 adaptor that also carries a text tower, so it is
# the one that supports text search.
RADIO_ADAPTOR = "siglip2-g"

# Patch size of the SO400M checkpoint. The ViT requires both input
# edges to be a multiple of this.
RADIO_PATCH_SIZE = 16

# The ViT's input long edge — a GPU budget decision, NOT the display
# thumbnail's size.
#
# Read here, at module level, so every consumer (the registry, the
# tests, the reconciler) sees the SAME number. The kernel cannot
# import the indexer — the architecture test forbids it — so this is
# the one place the setting is read.
#
# It was THUMBNAIL_SIZE until the 640 embed took the GPU off the
# PCIe bus (Xid 79). The ViT's activation memory scales with its
# input resolution, so a DISPLAY size was being paid for in VRAM.
# THUMBNAIL_SIZE now governs the display asset alone, and costs
# nothing but disk and encode time.
MODEL_INPUT_SIZE = int(os.environ.get("MODEL_INPUT_SIZE", "512"))


class RadioEmbedder:
    """Embedder for C-RADIOv4-SO400M with the siglip2-g adaptor.

    Lazy-loads the checkpoint on the first `embed_*` call, exactly
    like `OpenClipEmbedder`, so registry init stays cheap and the
    kernel stays importable on a host with no GPU.

    Not thread-safe by default; the load guard uses a lock and
    concurrent inference is serialised through the forward pass.
    """

    def __init__(
        self,
        *,
        dim: int = 1536,
        long_edge: int = 512,
        revision: str = "c-radio-v4-so400m",
    ) -> None:
        self._dim = dim
        self._long_edge = long_edge
        self._revision = revision
        # Typed as Any: the checkpoint is a torch.nn.Module whose shape
        # is only known at runtime, and mypy would otherwise infer
        # `None` from the initial assignment and treat every use as
        # unreachable.
        self._model: Any = None
        self._device = self._resolve_device()
        self._lock = threading.Lock()

    # -- Embedder protocol ------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def resolution(self) -> int:
        """Target length of the image's LONG edge, in pixels.

        The short edge follows from the source aspect ratio. For the
        SigLIP2 models this field is the side of a square canvas;
        for RADIO it is the long edge of a native-aspect image.
        """
        return self._long_edge

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None  # noqa: S101 — narrowed by _ensure_loaded
        if not texts:
            return []
        adaptor = self._model.adaptors[RADIO_ADAPTOR]
        tokens = adaptor.tokenizer(list(texts))
        tokens = _move_to_device(tokens, self._device)
        import torch

        with torch.no_grad():
            features = adaptor.encode_text(tokens, normalize=True)
        vectors: list[list[float]] = features.float().cpu().tolist()
        return vectors

    def embed_image(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None  # noqa: S101 — narrowed by _ensure_loaded
        if not images:
            return []
        import torch

        bad = [i for i, img in enumerate(images) if not isinstance(img, Image.Image)]
        if bad:
            raise TypeError(f"non-PIL images at indices {bad}")

        tensors = [self._prepare(img) for img in images]

        # A tensor can only be stacked with tensors of the same
        # shape. Native-aspect images almost never share one, so
        # group by shape and run one forward pass per group.
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, tensor in enumerate(tensors):
            groups.setdefault(tuple(tensor.shape[-2:]), []).append(idx)

        out: list[list[float] | None] = [None] * len(tensors)
        log_vram = os.environ.get("LOG_EMBED_VRAM", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if log_vram and self._device == "cuda":
            from contextlib import suppress

            with suppress(Exception):
                torch.cuda.reset_peak_memory_stats()
        for indices in groups.values():
            batch = torch.stack([tensors[i] for i in indices]).to(self._device)
            with torch.no_grad():
                vis_output = self._model(batch)
                summary = vis_output[RADIO_ADAPTOR][0]
            summary = summary.float()
            if summary.dim() == 3:
                summary = summary.squeeze(1)
            summary = summary / summary.norm(dim=-1, keepdim=True)
            values = summary.cpu().tolist()
            for slot, vector in zip(indices, values, strict=True):
                out[slot] = vector
        if log_vram and self._device == "cuda":
            from contextlib import suppress as _suppress

            with _suppress(Exception):
                import sys as _sys

                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                # The peak over the WHOLE call, not one group: the
                # groups are per-shape, so a single group's peak
                # would under-report what the card actually held.
                print(
                    f"[embed_vram] photos={len(images)} groups={len(groups)} "
                    f"device={self._device} peak={peak_mb:.0f}MB "
                    f"resolution={self._long_edge}",
                    file=_sys.stderr,
                    flush=True,
                )
        return [v for v in out if v is not None]

    # -- Internals ---------------------------------------------------------

    def _resolve_device(self) -> str:
        device = os.environ.get("DEVICE", "cpu").strip().lower()
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                return "cuda"
        return "cpu"

    def _ensure_loaded(self) -> None:
        # The fast path avoids the lock: the model is loaded once and
        # every later call finds it already there.
        loaded = self._model
        if loaded is not None:
            return
        with self._lock:
            # Re-check under the lock: another thread may have
            # finished loading while this one waited for it.
            loaded = self._model
            if loaded is not None:
                return
            import torch

            logger.info(
                "Loading C-RADIOv4 (%s) with the %s adaptor from %s",
                RADIO_VERSION, RADIO_ADAPTOR, RADIO_HUB_REPO,
            )
            model = torch.hub.load(  # type: ignore[no-untyped-call]
                RADIO_HUB_REPO,
                "radio_model",
                version=RADIO_VERSION,
                progress=True,
                skip_validation=True,
                adaptor_names=[RADIO_ADAPTOR],
                trust_repo=True,
            )
            model = model.to(self._device).eval()
            self._model = model
            logger.info("C-RADIOv4 loaded on %s", self._device)

    def _prepare(self, image: Image.Image) -> Any:
        """Resize to native aspect and convert to a float tensor.

        The long edge becomes `resolution`; the short edge follows
        the source aspect ratio. Both edges are then snapped to a
        multiple of the patch size, because the ViT's position
        embedding interpolation requires it.

        The tensor is in [0, 1] — RADIO's own input conditioner
        applies the CLIP mean/std normalisation on top.
        """
        import numpy as np
        import torch

        img = image.convert("RGB")
        width, height = img.size
        scale = self._long_edge / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        step = RADIO_PATCH_SIZE
        new_width = max(step, round(new_width / step) * step)
        new_height = max(step, round(new_height / step) * step)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        array = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def _move_to_device(tokens: Any, device: str) -> Any:
    """Move a tokenizer batch to `device`, whatever shape it comes in.

    The adaptor's tokenizer returns a transformers `BatchEncoding`
    (a dict of tensors) on the HF path. BatchEncoding has a
    `.to()`; a plain dict does not.
    """
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in tokens.items()}
    return tokens


def register_into(registry: Registry) -> None:
    """Register the C-RADIOv4-SO400M entry.

    The RADIO path needs runtime bits the old SigLIP2 loader did
    not (`einops` for torch.hub's dependency check, `git` for the
    clone), so its import is guarded.

    The input's long edge is MODEL_INPUT_SIZE, not a number of
    this module's own and NOT the display thumbnail's size.

    These were the same number until the 640 embed took the GPU
    off the PCIe bus (Xid 79). The ViT's cost scales with its
    INPUT resolution — the patch grid grows with the square of the
    edge, and so does the activation memory — so raising the
    model's input to 640 pushed the embedder's peak to 7014 MiB and
    made it 1.19x slower, while the DISPLAY thumbnail it also
    forced was pure CPU work (libvips; no torch in that module at
    all). Welding them meant paying GPU cost for a display size.

    So they are separate numbers now: MODEL_INPUT_SIZE is the
    ViT's input and is a GPU budget decision; THUMBNAIL_SIZE is
    the display asset and costs nothing but disk and encode time.
    The embedding still sees the SAME framing as the thumbnail —
    both are the source at its native aspect ratio — only the
    resolution differs.

    Read from the environment because the kernel cannot import the
    indexer (the architecture test forbids it).
    """
    embedder = RadioEmbedder(dim=1536, long_edge=MODEL_INPUT_SIZE)
    registry.register(
        ModelSpec(
            name="C-RADIOv4-SO400M",
            dim=1536,
            resolution=MODEL_INPUT_SIZE,
            revision="c-radio-v4-so400m",
            text=embedder,
            vision=embedder,
            preprocess="native",
        )
    )
