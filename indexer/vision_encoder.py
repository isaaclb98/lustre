"""
indexer/vision_encoder.py

Thin wrapper around the model registry (§A3). The actual SigLIP2
loading lives in `lustre_kernel._real_models.OpenClipEmbedder`;
this module exists only to preserve the historical
`VisionEncoder(...)` API for callers (`local_sync.py`, the benchmark
suite) and to centralize test-mode dispatch to the registry's
`mock-1536` entry.

The mock encoder (`MockEmbedder` in the registry) replaces the
previous `_mock_image_embed` function. The `test_mode` flag is now
interpreted as "use the mock-1536 registry entry instead of the
real model."
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default model name registered with the kernel. Callers may override
# via the constructor.
DEFAULT_MODEL_NAME: str = "ViT-so400m-patch16-384"
# Mock entry registered by `lustre_kernel.registry.get_default_registry`.
MOCK_MODEL_NAME: str = "mock-1536"


class VisionEncoder:
    """
    Adapter: routes `embed_batch` / `embed_one` to the embedder
    registered for `model_name` in the model registry.

    The `device` parameter is accepted for backward compatibility; it
    is forwarded to the registered embedder if that embedder is a
    `OpenClipEmbedder`. Mock embedders ignore it.

    The `test_mode` flag selects the `mock-1536` registry entry
    regardless of `model_name`.
    """

    def __init__(
        self,
        arch: str = DEFAULT_MODEL_NAME,
        pretrained: str = "webli",
        device: str = "cpu",
        *,
        test_mode: bool = False,
    ) -> None:
        # `arch` is treated as the model name in the registry. The
        # legacy `pretrained` arg is accepted but ignored — the
        # registry entry pins its own revision.
        self._requested_arch = arch
        self._pretrained = pretrained
        self.device = device
        self.test_mode = test_mode

        from lustre_kernel.registry import get as _registry_get

        if test_mode:
            try:
                self._embedder = _registry_get(MOCK_MODEL_NAME).vision
            except Exception:  # noqa: BLE001
                # Mock not registered (shouldn't happen with the
                # default-registry fixture). Fall back to looking up
                # the requested arch — the embedder may itself be a
                # mock registered by a test fixture.
                self._embedder = _registry_get(arch).vision
        else:
            self._embedder = _registry_get(arch).vision

    @property
    def dim(self) -> int:
        return self._embedder.dim

    @property
    def resolution(self) -> int:
        return self._embedder.resolution

    def embed_batch(self, images):
        """Embed a batch of PIL images. Returns a list of unit-norm vectors."""
        return list(self._embedder.embed_images(list(images)))

    def embed_one(self, image) -> list[float]:
        return self._embedder.embed_image(image)
