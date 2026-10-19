"""
lustre_kernel.registry

The model registry. The only place in the codebase that knows a
model's name, dim, resolution, revision, or how to embed with it.

Rules (from docs/backend-refactor-plan.md §4.3):

- No file outside this module references a model's dim, resolution,
  arch tag, revision, or call site.
- Every embedder call site in the codebase goes through
  `registry.get(name).text.embed_text(...)` or
  `registry.get(name).vision.embed_image(...)`. No direct
  `VisionEncoder(...)`, no `text_encoder.get_encoder()`, no
  `open_clip.create_model_and_transforms(...)` outside this module.
- Adding a model is a registry entry plus a config flag, nothing else.
- Removing a model is a registry deletion plus a check that no
  collection in any environment still references it.
- Real-model registrations are conditionally imported. The kernel
  package itself is importable on a CPU-only host with no torch /
  open_clip / transformers installed.

Concrete entries shipped today:

- `ViT-gopt-16-SigLIP2-384` — the web backend's current model
  (PyTorch + open_clip). Real registration is conditional.
- `mock-1536` — deterministic mock for tests and benchmarks. No
  real-model dependencies; safe to import on any host.
- `ViT-L-16-SigLIP2-256` — desktop product's model. Real
  registration is conditional; lives behind the same gate as the
  web's gopt entry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

__all__ = [
    "Embedder",
    "MockEmbedder",
    "ModelNotFoundError",
    "ModelSpec",
    "Registry",
    "get",
    "get_default_registry",
    "mock_embedder",
    "register",
]


@runtime_checkable
class Embedder(Protocol):
    """The only model-specific interface the codebase knows about.

    Every model — real or mock — implements this Protocol. Consumers
    receive an `Embedder` from the registry and call its methods; they
    never construct or import a model-specific class.
    """

    @property
    def dim(self) -> int: ...

    @property
    def resolution(self) -> int: ...

    def embed_text(self, text: str) -> list[float]: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_image(self, image: Image.Image) -> list[float]: ...

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]: ...


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    `text` and `vision` are independent `Embedder` instances. They
    share `dim` (a single model uses one embedding space for both
    text and image). `resolution` is the input image size the vision
    tower expects.
    """

    name: str
    dim: int
    resolution: int
    revision: str
    text: Embedder
    vision: Embedder


class ModelNotFoundError(KeyError):
    """Raised when a model name has no registry entry."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"no model registered with name {self.name!r}"


class Registry:
    """In-memory map of model name → ModelSpec.

    Process-local. Persistence (e.g. remembering the active model
    across restarts) is not the registry's job; the application
    configuration system reads the active model name and calls
    `get(name)` to obtain its spec.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        """Add or replace a spec by name. Idempotent."""
        self._specs[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove a spec by name. Raises `ModelNotFoundError` if missing."""
        if name not in self._specs:
            raise ModelNotFoundError(name)
        del self._specs[name]

    def get(self, name: str) -> ModelSpec:
        """Look up a spec by name. Raises `ModelNotFoundError` if missing."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ModelNotFoundError(name) from exc

    def names(self) -> list[str]:
        """Sorted list of registered model names."""
        return sorted(self._specs)

    def has(self, name: str) -> bool:
        return name in self._specs


# --- Default registry (process-global singleton) -----------------------

_DEFAULT_REGISTRY: Registry | None = None


def get_default_registry() -> Registry:
    """Return the process-global default registry, initialized lazily.

    On first call, registers the `mock-1536` mock encoder so that any
    consumer (test or production) can call `get("mock-1536")` without
    a prior `register(...)`. Real-model registrations are conditional
    and may add more entries on first call if the relevant runtime is
    available.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = Registry()
        # Order matters: real models MUST be registered before
        # `_patch_mock_dim` runs, because the provider callback
        # queries `registry.get(DEFAULT_MODEL).dim` and that
        # raises ModelNotFoundError if the real spec isn't there.
        _try_register_real_models(_DEFAULT_REGISTRY)
        _patch_mock_dim()
        _DEFAULT_REGISTRY.register(_MOCK_1536_SPEC)
    return _DEFAULT_REGISTRY


def register(spec: ModelSpec) -> None:
    """Register a spec on the default registry."""
    get_default_registry().register(spec)


def get(name: str) -> ModelSpec:
    """Look up a spec on the default registry."""
    return get_default_registry().get(name)


# ---------------------------------------------------------------------------
# Active-model resolution (variant name → model name → spec)
#
# This is the single source of truth for the SIGLIP_VARIANT env var.
# Both `search/config.py` and `indexer/upsert.py` call this helper
# instead of duplicating the variant → model name mapping.
# ---------------------------------------------------------------------------

# Variant name → registered model name. Adding a model means adding
# it here AND in `_real_models.register_into` / `_MOCK_*_SPEC` above.
VARIANT_TO_MODEL: dict[str, str] = {
    "B/16-256": "ViT-B-16-SigLIP2-256",
    "L/16-256": "ViT-L-16-SigLIP2-256",
    "gopt/16-384": "ViT-gopt-16-SigLIP2-384",
    # so400m/16-384 — shape-optimised attention variant; 1152-dim.
    # 384 input resolution matches the SigLIP2 family default. The
    # model-variant migration plan adds this as the new prod default.
    "so400m/16-384": "ViT-so400m-patch16-384",
}

DEFAULT_VARIANT = "B/16-256"  # end-user ghcr default (smallest, CPU-runnable).
# Prod keeps so400m via SIGLIP_VARIANT or MODEL_NAME override.


def resolve_model_name(variant: str) -> str:
    """Map a SIGLIP2 variant name to its registered model name.

    Raises ValueError for unknown variants — same contract as
    `search.config.get_siglip_variant` so callers get a clear
    error message listing valid options.
    """
    if variant not in VARIANT_TO_MODEL:
        raise ValueError(
            f"Unknown SIGLIP variant {variant!r}. "
            f"Known: {sorted(VARIANT_TO_MODEL.keys())}"
        )
    return VARIANT_TO_MODEL[variant]


def get_active_model_spec() -> ModelSpec:
    """Return the ModelSpec for the currently configured variant.

    Reads SIGLIP_VARIANT from env (defaulting to DEFAULT_VARIANT),
    resolves it to a registered model name, and looks up the spec.
    Both the search app and the indexer use this so there's one
    place to change the active model.
    """
    import os

    variant = os.environ.get("SIGLIP_VARIANT", DEFAULT_VARIANT)
    model_name = resolve_model_name(variant)
    return get(model_name)


# --- Mock embedder ------------------------------------------------------

class MockEmbedder:
    """Deterministic Embedder implementation for tests and benchmarks.

    Generates a unit-norm vector of `dim` floats from a seed derived
    from the input. Same input → same vector (deterministic).
    Different inputs → different vectors (collision-resistant).

    No torch, no open_clip, no model weights. Safe to import on any
    host.
    """

    def __init__(self, dim: int = 1536, resolution: int = 384) -> None:
        self._dim = dim
        self._resolution = resolution

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def resolution(self) -> int:
        return self._resolution

    def embed_text(self, text: str) -> list[float]:
        return _deterministic_vector(text, self._dim)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_image(self, image: Image.Image) -> list[float]:
        # Image identity: bytes of the underlying buffer at a point
        # in time. Deterministic enough for tests; the benchmark
        # suite uses fixed fixtures.
        try:
            seed = image.tobytes()
        except Exception:  # noqa: BLE001 — PIL fallback: use empty bytes, the deterministic generator handles either
            seed = b""
        return _deterministic_vector(seed, self._dim)

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        return [self.embed_image(img) for img in images]


def _deterministic_vector(seed: object, dim: int) -> list[float]:
    """Build a unit-norm vector of `dim` floats from a hash of `seed`.

    Mirrors the math used by `indexer/vision_encoder._mock_image_embed`
    (sha512 → repeat to fill `dim` → L2-normalize). The two mocks are
    intentionally compatible: a test that hashes the same string gets
    the same vector regardless of which mock is in play.
    """
    import hashlib
    digest = hashlib.sha512(repr(seed).encode()).digest()  # 64 bytes
    raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


import math  # placed after the function defs that use it; ruff is fine

# The mock spec's dim used to be hardcoded at 1536 to match gopt-16.
# After the model-variant migration plan, the prod variant is
# so400m/16-384 (1152-dim) — but the mock exists to mimic the
# prod model's vector space so tests can encode queries + store
# vectors at the same dim. Hardcoding 1536 here would force every
# test that uses the mock embedder to also build a 1536-dim
# collection, which would diverge from the production collection
# dim.
#
# The mock dim is patched lazily by `_patch_mock_dim()`, which is
# invoked from `get_default_registry()`. The patcher reads the
# active dim from `_get_active_mock_dim`, a callable that the
# application (search package) registers at import time. This
# keeps the kernel pure of any `search` / `indexer` imports
# (enforced by `tests/test_architecture.py`).
_MOCK_1536_SPEC = ModelSpec(
    name="mock-1536",
    dim=1536,  # patched below via _patch_mock_dim()
    resolution=384,
    revision="mock-1",
    text=MockEmbedder(dim=1536, resolution=384),
    vision=MockEmbedder(dim=1536, resolution=384),
)


# Callable injected by the application to tell the kernel which
# dim the mock should mimic. Returns the active variant's vector
# dimension (1152 for so400m/16-384 today). Defaults to 1536
# (gopt's dim) if not registered — so the kernel is usable in
# isolation for tests that don't care about the active variant.
_active_mock_dim_provider: Callable[[], int] | None = None

# Re-entry guard for `_resolve_mock_dim`. Set True while the
# resolver is in flight (so `_try_register_real_models` →
# `get_default_registry()` → `_patch_mock_dim` → `_resolve_mock_dim`
# chain doesn't recurse infinitely if the chain loops back).
# Process-global — tests don't share threads across boundaries
# that matter here.
_RESOLVE_IN_PROGRESS: bool = False


def register_mock_dim_provider(provider: Callable[[], int]) -> None:
    """Install a callable that returns the active mock dim.

    The application (`search/__init__.py`) registers its
    `get_active_mock_dim()` here on import. The kernel uses it
    during `_patch_mock_dim()` to align the mock spec with the
    prod variant. Tests that want to override the dim for a
    single process can call this directly with a custom callable.
    """
    global _active_mock_dim_provider
    _active_mock_dim_provider = provider


def _resolve_mock_dim() -> int:
    """Resolve the mock spec's dim via the registered provider.

    Falls back to 1536 if no provider is registered — that's the
    pre-migration default and stays correct for any test that
    doesn't care about the prod variant.

    Order of operations in `get_default_registry`: real-model
    specs are registered first, then `_patch_mock_dim` runs and
    calls `_resolve_mock_dim` to read the active dim. So by the
    time we're called, the real-model spec is in the registry
    and the provider's `registry.get(DEFAULT_MODEL).dim`
    succeeds.
    """
    global _RESOLVE_IN_PROGRESS
    if _active_mock_dim_provider is None:
        return 1536
    if _RESOLVE_IN_PROGRESS:
        return 1536
    try:
        _RESOLVE_IN_PROGRESS = True
        result = _active_mock_dim_provider()
        if not result:
            raise ValueError("provider returned falsy dim")
        return int(result)
    except (TypeError, ValueError) as e:
        print(f"[kernel] _resolve_mock_dim fallback to 1536: {type(e).__name__}: {e}")
        return 1536
    finally:
        _RESOLVE_IN_PROGRESS = False


def _patch_mock_dim() -> None:
    """Rewrite the dim in `_MOCK_1536_SPEC` and its MockEmbedder
    children to match the active prod variant.

    Called from `get_default_registry` immediately before
    registering the mock spec. ModelSpec is frozen, so we use
    `object.__setattr__` to bypass the freeze.
    """
    dim = _resolve_mock_dim()
    fresh = MockEmbedder(dim=dim, resolution=384)
    object.__setattr__(_MOCK_1536_SPEC, "dim", dim)
    object.__setattr__(_MOCK_1536_SPEC, "text", fresh)
    object.__setattr__(_MOCK_1536_SPEC, "vision", fresh)

# Public name for the mock registry entry. Exposed at module
# level so `search/text_encoder.py` can refer to the same name
# without duplicating the literal (which would drift if we ever
# rename the mock spec).
_MOCK_REGISTRY_NAME: str = _MOCK_1536_SPEC.name


def mock_embedder() -> MockEmbedder:
    """Return a fresh MockEmbedder for callers that need one directly.

    Most code goes through `registry.get("mock-1536").vision`. This
    helper exists for tests that want to construct a mock with a
    non-default dim.
    """
    return MockEmbedder()


# --- Real-model registration (conditional) -----------------------------

def _try_register_real_models(registry: Registry) -> None:
    """Register real-model entries if the relevant runtime is importable.

    The kernel package must remain importable on hosts without torch /
    open_clip / transformers. Each real-model registration is wrapped
    in a try/except so that a missing runtime silently skips that
    entry. This function is called once on default-registry init; it
    is not called from per-request paths.
    """
    # The web backend's gopt-16 model. Real implementation lives in
    # `lustre_kernel._real_models` and is imported lazily so
    # that importing the kernel does not import torch.
    try:
        from lustre_kernel import _real_models

        _real_models.register_into(registry)
    except ImportError:
        # Real-model backend not installed. The mock entry remains
        # available; production deployments that need real models
        # install the corresponding extra (`pip install
        # lustre-kernel[open_clip]`).
        pass
