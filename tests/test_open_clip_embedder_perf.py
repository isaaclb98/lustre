"""
tests/test_open_clip_embedder_perf.py — Round-perf (issue #1) coverage.

Pins the env-var gates for two opt-in GPU perf knobs:

  - `ENABLE_EMBED_AUTOCAST=1` → wraps forward pass in fp16 autocast
    (gated on cuda device; on cpu it's silently ignored)
  - `ENABLE_EMBED_TORCH_COMPILE=1` → applies torch.compile on model load
    (same cuda gating)

These tests inspect the embedder's `_autocast_enabled` /
`_torch_compile_enabled` flags without actually loading weights
(constructor is lazy on model load). The env-var reading happens in
`__init__`, so we just patch the env vars then construct a fresh
embedder — no module reload needed (and reloading kernel modules
would poison other tests' registry state).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def make_embedder():
    """Return a factory that constructs OpenClipEmbedder under current env."""
    from lustre_kernel._real_models import OpenClipEmbedder

    def _make(**kwargs):
        return OpenClipEmbedder(
            arch_tag="ViT-gopt-16-SigLIP2-384",
            pretrained="webli",
            dim=1536,
            resolution=384,
            **kwargs,
        )
    return _make


class TestAutocastGate:
    """`ENABLE_EMBED_AUTOCAST` toggles autocast on the forward pass."""

    def test_default_is_off(self, make_embedder, monkeypatch):
        monkeypatch.delenv("ENABLE_EMBED_AUTOCAST", raising=False)
        embedder = make_embedder()
        assert embedder._autocast_enabled is False

    def test_enabled_on_cuda(self, make_embedder, monkeypatch):
        """With ENABLE_EMBED_AUTOCAST=1 and a cuda device, autocast on."""
        monkeypatch.setenv("ENABLE_EMBED_AUTOCAST", "1")
        monkeypatch.setenv("DEVICE", "cuda")
        # Patch torch.cuda.is_available so the cuda branch is taken
        with patch("torch.cuda.is_available", return_value=True):
            embedder = make_embedder()
            assert embedder._device == "cuda"
            assert embedder._autocast_enabled is True

    def test_disabled_on_cpu_even_with_env(self, make_embedder, monkeypatch):
        """Autocast env var only applies when device is cuda."""
        monkeypatch.setenv("ENABLE_EMBED_AUTOCAST", "1")
        monkeypatch.setenv("DEVICE", "cpu")
        embedder = make_embedder()
        assert embedder._device == "cpu"
        assert embedder._autocast_enabled is False

    def test_truthy_values(self, make_embedder, monkeypatch):
        """1, true, yes, on (case-insensitive) all enable."""
        for truthy in ("1", "true", "TRUE", "True", "yes", "on"):
            monkeypatch.setenv("ENABLE_EMBED_AUTOCAST", truthy)
            monkeypatch.setenv("DEVICE", "cuda")
            with patch("torch.cuda.is_available", return_value=True):
                embedder = make_embedder()
                assert embedder._autocast_enabled is True, (
                    f"truthy value {truthy!r} should enable autocast"
                )

    def test_falsy_values(self, make_embedder, monkeypatch):
        """0, false, no, off, empty, garbage → all disable."""
        for falsy in ("0", "false", "no", "off", "", "garbage"):
            monkeypatch.setenv("ENABLE_EMBED_AUTOCAST", falsy)
            monkeypatch.setenv("DEVICE", "cuda")
            with patch("torch.cuda.is_available", return_value=True):
                embedder = make_embedder()
                assert embedder._autocast_enabled is False, (
                    f"falsy value {falsy!r} should not enable autocast"
                )


class TestTorchCompileGate:
    """`ENABLE_EMBED_TORCH_COMPILE` toggles torch.compile on model load."""

    def test_default_is_off(self, make_embedder, monkeypatch):
        monkeypatch.delenv("ENABLE_EMBED_TORCH_COMPILE", raising=False)
        embedder = make_embedder()
        assert embedder._torch_compile_enabled is False
        assert embedder._compiled is False

    def test_enabled_on_cuda(self, make_embedder, monkeypatch):
        monkeypatch.setenv("ENABLE_EMBED_TORCH_COMPILE", "1")
        monkeypatch.setenv("DEVICE", "cuda")
        with patch("torch.cuda.is_available", return_value=True):
            embedder = make_embedder()
            assert embedder._device == "cuda"
            assert embedder._torch_compile_enabled is True
            # Compilation happens lazily on first forward, not at init
            assert embedder._compiled is False

    def test_disabled_on_cpu_even_with_env(self, make_embedder, monkeypatch):
        """Compile env var only applies when device is cuda."""
        monkeypatch.setenv("ENABLE_EMBED_TORCH_COMPILE", "1")
        monkeypatch.setenv("DEVICE", "cpu")
        embedder = make_embedder()
        assert embedder._device == "cpu"
        assert embedder._torch_compile_enabled is False


class TestDecodeMaxDefault:
    """`INDEXER_DECODE_MAX` default changed from 0 → 768.

    Reads `_DECODE_MAX` from the module — the value is captured at
    import time (env read at module top-level), so reloading the
    module is the only way to test different defaults. We reload in
    an isolated fashion (just this one module) to avoid poisoning
    the broader import state.
    """

    def test_default_is_768(self, monkeypatch):
        monkeypatch.delenv("INDEXER_DECODE_MAX", raising=False)
        import importlib
        import sys
        for mod_name in list(sys.modules.keys()):
            if mod_name == "indexer.image_loader":
                del sys.modules[mod_name]
        import indexer.image_loader as loader
        importlib.reload(loader)
        assert loader._DECODE_MAX == 768

    def test_env_override_works(self, monkeypatch):
        monkeypatch.setenv("INDEXER_DECODE_MAX", "1024")
        import importlib
        import sys
        for mod_name in list(sys.modules.keys()):
            if mod_name == "indexer.image_loader":
                del sys.modules[mod_name]
        import indexer.image_loader as loader
        importlib.reload(loader)
        assert loader._DECODE_MAX == 1024

    def test_env_zero_disables_cap(self, monkeypatch):
        """Setting to 0 disables the cap (full decode)."""
        monkeypatch.setenv("INDEXER_DECODE_MAX", "0")
        import importlib
        import sys
        for mod_name in list(sys.modules.keys()):
            if mod_name == "indexer.image_loader":
                del sys.modules[mod_name]
        import indexer.image_loader as loader
        importlib.reload(loader)
        assert loader._DECODE_MAX == 0
