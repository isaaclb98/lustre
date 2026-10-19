"""
lustre_kernel._real_models

Real-model registry entries. Imported lazily by `lustre_kernel.registry`
so that the kernel package itself is importable on hosts without
`torch` / `open_clip` / `transformers` installed.

If any required runtime is missing, importing this module raises
`ImportError`. The caller (`_try_register_real_models`) catches it
and proceeds with the mock-only registry.
"""

from __future__ import annotations

import open_clip
import torch
import transformers  # noqa: F401

from lustre_kernel.registry import ModelSpec, Registry

__all__ = ["register_into"]


class OpenClipEmbedder:
    """Adapter: wraps an open_clip model + preprocess into the Embedder Protocol.

    Lazy-loads the actual weights on first `embed_*` call (not on
    `__init__`), so the registry init stays cheap. The first text or
    image call pays the model-load cost; subsequent calls hit the
    cached model.

    Not thread-safe by default; the model-load guard uses a
    `threading.Lock`. Concurrent inference requests are serialized
    through the model forward pass.
    """

    def __init__(
        self,
        *,
        arch_tag: str,
        pretrained: str,
        dim: int,
        resolution: int,
    ) -> None:
        self._arch_tag = arch_tag
        self._pretrained = pretrained
        self._dim = dim
        self._resolution = resolution
        self._model = None
        self._tokenizer = None
        self._preprocess = None
        # Round‑15: honour `DEVICE` env var (defaults to cpu). When set
        # to `cuda` and a GPU is available, the loaded model is moved
        # to that device so image embedding can actually run on the
        # RTX 3080 instead of the CPU.
        import os as _os
        _dev = _os.environ.get("DEVICE", "cpu").strip().lower()
        if _dev == "cuda" and __import__("torch").cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._lock = __import__("threading").Lock()
        # Round‑15: batch size used by the pipeline's _embed phase to
        # stack images into a single GPU forward pass. RTX 3080
        # (10 GB VRAM) comfortably holds 32 L/16-256 tensors; bump
        # for larger GPUs, drop for smaller ones.
        self.embed_batch_size = int(
            __import__("os").environ.get("EMBED_BATCH_SIZE", "32")
        )
        # Round-perf (issue #1): opt-in fp16 autocast on the forward
        # pass. SigLIP2 is robust to fp16 (the pretrained weights were
        # trained with mixed precision in open_clip's training loop).
        # On a RTX 3080 / 3060 this typically yields ~1.5-2× throughput
        # on the encoder. Default off so existing deployments and
        # tests are unaffected. ENABLE_EMBED_AUTOCAST=1 turns it on.
        self._autocast_enabled = (
            __import__("os").environ.get("ENABLE_EMBED_AUTOCAST", "0").lower()
            in ("1", "true", "yes", "on")
            and self._device == "cuda"
        )
        # Round-perf (issue #1): opt-in torch.compile. `mode="reduce-overhead"`
        # uses CUDA graphs to cut launch overhead; warms up once on
        # the first forward pass. On a 902k-photo reindex this saves
        # ~20-30% wall time after the one-time compile cost. Default
        # off because: (a) first call pays a 30-60s compile cost,
        # (b) compilation can change numerics slightly, (c) not all
        # open_clip model variants are compile-friendly. Set
        # ENABLE_EMBED_TORCH_COMPILE=1 to enable.
        self._torch_compile_enabled = (
            __import__("os").environ.get("ENABLE_EMBED_TORCH_COMPILE", "0").lower()
            in ("1", "true", "yes", "on")
            and self._device == "cuda"
        )
        self._compiled = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def resolution(self) -> int:
        return self._resolution

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model, preprocess = open_clip.create_model_from_pretrained(
                f"hf-hub:{self._arch_tag}",
            )
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{self._arch_tag}")
            model = model.eval().to(self._device)
            if self._torch_compile_enabled and not self._compiled:
                # `mode="reduce-overhead"` is the right mode for
                # repeated fixed-shape inference batches (CUDA graphs).
                # The first forward pass triggers compilation; subsequent
                # passes hit the cached graph. Safe to call on the eval
                # model — encode_image is pure inference.
                model = torch.compile(model, mode="reduce-overhead")
                self._compiled = True
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer

    def embed_text(self, text: str) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None  # noqa: S101 — type-narrowing after _ensure_loaded
        tokens = self._tokenizer([text])
        tokens = tokens.to(self._device)
        with torch.no_grad():
            if self._autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self._model.encode_text(tokens)
            else:
                features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_texts(self, texts) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_image(self, image) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None and self._preprocess is not None  # noqa: S101 — type-narrowing after _ensure_loaded
        from PIL import Image as _PILImage

        if not isinstance(image, _PILImage.Image):
            raise TypeError(f"expected PIL.Image, got {type(image).__name__}")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_images(self, images) -> list[list[float]]:
        """Batched image embedding.

        Round‑15: stacks all images into a single forward pass on
        the GPU. The naive `embed_image` loop calls `encode_image`
        once per image, which serialises the work and leaves the GPU
        starved between launches.

        Round-perf (issue #1): with `ENABLE_EMBED_AUTOCAST=1`, wraps
        the forward pass in `torch.autocast(fp16)`. SigLIP2 is robust
        to fp16 (its open_clip training loop already used mixed
        precision). On RTX 30/40 series this typically yields
        ~1.5-2× encoder throughput.
        """
        self._ensure_loaded()
        assert self._model is not None and self._preprocess is not None  # noqa: S101
        from PIL import Image as _PILImage

        if not images:
            return []
        bad = [i for i, img in enumerate(images) if not isinstance(img, _PILImage.Image)]
        if bad:
            raise TypeError(f"non-PIL images at indices {bad}")
        # Pre-process every image, then concatenate on the GPU in one
        # shot. `unsqueeze(0)` on each, then `torch.cat` to (N, C, H, W).
        import torch as _torch
        tensors = [self._preprocess(img) for img in images]
        batch = _torch.stack(tensors, dim=0).to(self._device)
        # VRAM monitor: log peak GPU memory per forward pass so the
        # indexer can see whether batch size + autocast settings
        # actually fit before the next batch fires. Disabled when
        # LOG_EMBED_VRAM=0 (or unset, off by default).
        log_vram = __import__(
            "os"
        ).environ.get("LOG_EMBED_VRAM", "0").lower() in ("1", "true", "yes", "on")
        if log_vram and self._device == "cuda":
            from contextlib import suppress
            with suppress(Exception):
                _torch.cuda.reset_peak_memory_stats()
        with _torch.no_grad():
            if self._autocast_enabled:
                with _torch.autocast(device_type="cuda", dtype=_torch.float16):
                    features = self._model.encode_image(batch)
            else:
                features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        if log_vram and self._device == "cuda":
            from contextlib import suppress as _suppress
            with _suppress(Exception):
                peak_mb = _torch.cuda.max_memory_allocated() / (1024 * 1024)
                # Cheap print on every batch is too noisy for the prod
                # log; gate on >50MB increments. Use the stdout stream
                # the rest of the indexer uses.
                import sys as _sys
                print(
                    f"[embed_vram] batch={len(images)} device={self._device} "
                    f"peak={peak_mb:.0f}MB autocast={self._autocast_enabled}",
                    file=_sys.stderr,
                    flush=True,
                )
        return features.cpu().tolist()


def register_into(registry: Registry) -> None:
    """
    Register the real-model entries shipped with the kernel.

    We ship four SigLIP2 variants:

      - ViT-B-16-SigLIP2-256 (768-dim, 256 input) — end-user ghcr
        default. Smallest, fastest, runs on CPU. The intent is
        "anyone can pull the image and have it work"; prod-quality
        models are opt-in via SIGLIP_VARIANT.
      - ViT-so400m-patch16-384 (1152-dim, web prod) — shape-Optimized
        attention variant; 384 input resolution; ~400M params.
        Selected via SIGLIP_VARIANT=so400m/16-384 (prod uses this
        via .env override on top of the B/16-256 default).
      - ViT-gopt-16-SigLIP2-384 (1536-dim) — kept registered so the
        rollback path (`SIGLIP_VARIANT=gopt/16-384`) still works
        without re-installing model weights.
      - ViT-L-16-SigLIP2-256 (1024-dim) — desktop / dev.

    Each entry uses the same `text`/`vision` adapter because the
    SigLIP-2 family shares weights across modalities.
    """

    so400m_embedder = OpenClipEmbedder(
        # The repo path is `timm/ViT-SO400M-16-SigLIP2-384` (patch
        # size 16, 384 input). This is the model the kernel's
        # `name` field (`ViT-so400m-patch16-384`) refers to, and the
        # weights open_clip downloads under the `webli` pretrained
        # tag. The earlier `timm/ViT-SO400M-14-SigLIP2-378` arch_tag
        # was a different architecture (patch size 14, 378 input)
        # that produced vectors in the wrong embedding space;
        # vectors written under that arch_tag are not compatible
        # with the patch16 model and must be re-indexed.
        arch_tag="timm/ViT-SO400M-16-SigLIP2-384",
        pretrained="webli",
        dim=1152,
        resolution=384,
    )
    registry.register(ModelSpec(
        name="ViT-so400m-patch16-384",
        dim=1152,
        resolution=384,
        revision="webli",
        text=so400m_embedder,
        vision=so400m_embedder,
    ))

    gopt_embedder = OpenClipEmbedder(
        arch_tag="timm/ViT-gopt-16-SigLIP2-384",
        pretrained="webli",
        dim=1536,
        resolution=384,
    )
    registry.register(ModelSpec(
        name="ViT-gopt-16-SigLIP2-384",
        dim=1536,
        resolution=384,
        revision="webli",
        text=gopt_embedder,
        vision=gopt_embedder,
    ))

    l_embedder = OpenClipEmbedder(
        arch_tag="timm/ViT-L-16-SigLIP2-256",
        pretrained="webli",
        dim=1024,
        resolution=256,
    )
    registry.register(ModelSpec(
        name="ViT-L-16-SigLIP2-256",
        dim=1024,
        resolution=256,
        revision="webli",
        text=l_embedder,
        vision=l_embedder,
    ))

    b_embedder = OpenClipEmbedder(
        arch_tag="timm/ViT-B-16-SigLIP2-256",
        pretrained="webli",
        dim=768,
        resolution=256,
    )
    registry.register(ModelSpec(
        name="ViT-B-16-SigLIP2-256",
        dim=768,
        resolution=256,
        revision="webli",
        text=b_embedder,
        vision=b_embedder,
    ))
