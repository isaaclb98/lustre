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

from lustre_kernel.limits import MAX_EMBED_BATCH_SIZE
from lustre_kernel.registry import ModelSpec, Registry
from lustre_kernel.spaces import PICTURE_SPACE, SPACE_ADAPTORS, TEXT_SPACE

__all__ = ["MODEL_INPUT_SIZE", "register_into"]

logger = logging.getLogger(__name__)

# The GitHub repo torch.hub clones. NVIDIA's; fetched at runtime.
RADIO_HUB_REPO = "NVlabs/RADIO"

# The checkpoint version key. Resolves to
# nvidia/C-RADIOv4-SO400M (the SO400M backbone) on the HF hub.
RADIO_VERSION = "c-radio_v4-so400m"

# The teacher adaptor whose summary is the TEXT space.
RADIO_ADAPTOR = SPACE_ADAPTORS[TEXT_SPACE]

# The teacher adaptor whose summary is the PICTURE space.
RADIO_PICTURE_ADAPTOR = SPACE_ADAPTORS[PICTURE_SPACE]

# The dim each space's summary has, measured from the checkpoint's
# adaptor heads (the final projection's output width):
#   _heads.siglip2-g.final.2.weight  -> (1536, 1520)
#   _heads.dino_v3_7b.final.2.weight -> (4096, 1520)
RADIO_TEXT_DIM = 1536
RADIO_PICTURE_DIM = 4096


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
        device: str | None = None,
        adaptors: Sequence[str] | None = None,
    ) -> None:
        self._dim = dim
        self._long_edge = long_edge
        self._revision = revision
        # Which teacher adaptors to load. The TEXT space needs
        # `siglip2-g` (it carries the text tower); the PICTURE space
        # needs `dino_v3_7b`. Loading BOTH costs one backbone, not
        # two — every adaptor's head projects the SAME backbone
        # features — and the DINOv3-only load was measured to
        # produce BIT-IDENTICAL vectors to the both-adaptors load
        # (max_abs_diff 0.0).
        self._adaptors: tuple[str, ...] = tuple(
            adaptors if adaptors is not None else (RADIO_ADAPTOR,)
        )
        # Typed as Any: the checkpoint is a torch.nn.Module whose shape
        # is only known at runtime, and mypy would otherwise infer
        # `None` from the initial assignment and treat every use as
        # unreachable.
        self._model: Any = None
        self._device = self._resolve_device(device)
        self._lock = threading.Lock()


        # The largest batch this embedder will put in ONE forward
        # pass. The indexer clamps its own CLI flag and env var, but
        # a caller that asks the embedder directly never passes
        # through that clamp — so the ceiling is enforced here too.
        # The activation memory of the ViT scales with the batch, and
        # the card is a 10 GiB RTX 3080: the 640-input run at batch 16
        # left ~90 MiB free, which is the condition that took the card
        # off the PCIe bus (Xid 79).
        #
        # `embed_images` groups by shape and stacks each group into a
        # single forward pass, so without this the group size — not
        # the caller's request — decides the peak. A group is at most
        # as large as the batch handed in, but a direct caller could
        # hand in 32 images of one shape and put all 32 in one pass.
        self.embed_batch_size = max(
            1,
            min(
                int(os.environ.get("EMBED_BATCH_SIZE", str(MAX_EMBED_BATCH_SIZE))),
                MAX_EMBED_BATCH_SIZE,
            ),
        )

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
        """Embed images into the TEXT space (siglip2-g).

        Kept as the single-space API for the callers that only need
        one space. `embed_images_multi` returns every loaded space.
        """
        return self.embed_images_multi(images)[TEXT_SPACE]

    def embed_images_multi(
        self, images: Sequence[Image.Image],
    ) -> dict[str, list[list[float]]]:
        """Embed images into EVERY loaded space, from ONE forward pass.

        Returns `{space_name: [vector, ...]}`. Each adaptor's head
        projects the SAME backbone features into its own space, so
        loading both adaptors costs one backbone, not two.

        The keys are the SPACE names (`""` for the text space,
        `"dinov3"` for the picture space), not the adaptor names.
        """
        self._ensure_loaded()
        assert self._model is not None  # noqa: S101 — narrowed by _ensure_loaded
        spaces = self._space_for_adaptor()
        if not images:
            return {name: [] for name in spaces}
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

        # space name -> one slot per input image, filled as the
        # groups complete.
        out: dict[str, list[list[float] | None]] = {
            name: [None] * len(tensors) for name in spaces
        }
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
            # Chunk the group: a shape group can be larger than the
            # batch ceiling (every photo in the corpus can share one
            # shape), and the whole group would otherwise go into a
            # single forward pass.
            for start in range(0, len(indices), self.embed_batch_size):
                chunk = indices[start : start + self.embed_batch_size]
                batch = torch.stack([tensors[i] for i in chunk]).to(self._device)
                with torch.no_grad():
                    vis_output = self._model(batch)
                    # One forward, every adaptor's summary: each
                    # head projects the SAME backbone features into
                    # its own teacher's space.
                    summaries = {
                        adaptor: vis_output[adaptor][0].float()
                        for adaptor in spaces.values()
                    }
                for space_name, adaptor in spaces.items():
                    summary = summaries[adaptor]
                    if summary.dim() == 3:
                        summary = summary.squeeze(1)
                    summary = summary / summary.norm(dim=-1, keepdim=True)
                    values = summary.cpu().tolist()
                    for slot, vector in zip(chunk, values, strict=True):
                        out[space_name][slot] = vector
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
        return {
            name: [v for v in slots if v is not None]
            for name, slots in out.items()
        }

    # -- Internals ---------------------------------------------------------

    def _space_for_adaptor(self) -> dict[str, str]:
        """SPACE name -> adaptor name, for the ADAPTORS THIS INSTANCE LOADED.

        Filtered from `SPACE_ADAPTORS`. A space whose adaptor is not
        loaded is absent from the result — the caller asked for a
        space the model cannot produce, and silently returning an
        empty list would be a wrong-space answer rather than an
        error.
        """
        return {
            space: adaptor
            for space, adaptor in SPACE_ADAPTORS.items()
            if adaptor in self._adaptors
        }

    def _resolve_device(self, requested: str | None = None) -> str:
        """Resolve the device this embedder runs on.

        `requested` wins when given. Otherwise the DEVICE env var
        decides, which is how the indexer selects its device (its
        --device flag writes DEVICE before the registry is built).
        """
        if requested is None:
            requested = os.environ.get("DEVICE", "cpu")
        device = requested.strip().lower()
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
                "Loading C-RADIOv4 (%s) with the %s adaptor(s) from %s",
                RADIO_VERSION, "+".join(self._adaptors), RADIO_HUB_REPO,
            )
            model = torch.hub.load(  # type: ignore[no-untyped-call]
                RADIO_HUB_REPO,
                "radio_model",
                version=RADIO_VERSION,
                progress=True,
                skip_validation=True,
                adaptor_names=list(self._adaptors),
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
    # The text tower and the vision tower are SEPARATE embedders,
    # and they can sit on different devices.
    #
    # The API and the indexer share this container: the API answers
    # text queries, the indexer (a child process) embeds photos. Both
    # would otherwise resolve the same DEVICE env var and land on the
    # GPU — so the API's first text query would pull the whole model
    # onto the card and take memory away from a running embed. The
    # indexer's --device flag writes DEVICE, so the vision tower has
    # to follow DEVICE; the text tower follows TEXT_ENCODER_DEVICE,
    # which defaults to cpu.
    #
    # The two towers are never loaded in the same process (the API
    # only embeds text, the indexer only embeds images), so this
    # does not double the resident weights.
    #
    # The TEXT tower needs only the siglip2-g adaptor — that is the
    # one carrying SigLIP2-g's text tower. Loading the dino_v3_7b
    # adaptor there would add weights for nothing: the API never
    # embeds an image.
    text_embedder = RadioEmbedder(
        dim=RADIO_TEXT_DIM,
        long_edge=MODEL_INPUT_SIZE,
        device=os.environ.get("TEXT_ENCODER_DEVICE", "cpu"),
        adaptors=(RADIO_ADAPTOR,),
    )
    # The VISION tower writes BOTH spaces, so it loads BOTH
    # adaptors. They share one backbone: every adaptor's head
    # projects the SAME backbone features into its own teacher's
    # space, so this is ONE forward pass producing two vectors, not
    # two passes.
    #
    # Measured on the 3080 at 640 input, batch 4: loading both
    # adaptors peaks at 5250 MB (vs 5078 MB for siglip2 alone) and
    # runs 9.22 photos/s (vs 10.23). The picture space costs
    # ~100 MB of weights and ~10% throughput — the backbone, not the
    # adaptors, is the cost.
    vision_embedder = RadioEmbedder(
        dim=RADIO_TEXT_DIM,
        long_edge=MODEL_INPUT_SIZE,
        adaptors=(RADIO_ADAPTOR, RADIO_PICTURE_ADAPTOR),
    )
    registry.register(
        ModelSpec(
            name="C-RADIOv4-SO400M",
            dim=RADIO_TEXT_DIM,
            resolution=MODEL_INPUT_SIZE,
            revision="c-radio-v4-so400m",
            text=text_embedder,
            vision=vision_embedder,
            preprocess="native",
            # The picture→picture space: DINOv3's summary, in the
            # same collection as the text space under a named
            # vector. 4096-dim — the DINOv3-7B teacher's own width.
            extra_spaces={PICTURE_SPACE: RADIO_PICTURE_DIM},
        )
    )
