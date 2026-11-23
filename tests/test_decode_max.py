"""
tests/test_decode_max.py — the decode-size cap.

`INDEXER_DECODE_MAX` caps the pixel budget of a single decode, so a
huge source photo does not have to be fully decoded before it is
downscaled. The cap is read at module import time, so the tests
reload `indexer.image_loader` in isolation.

This file used to also cover the `ENABLE_EMBED_AUTOCAST` and
`ENABLE_EMBED_TORCH_COMPILE` gates. Those gates belonged to the
open_clip embedder, which is gone; the live C-RADIOv4 embedder
implements neither, so there is nothing left to pin.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def reloaded_image_loader(monkeypatch):
    """Yield a freshly-imported `indexer.image_loader`, then restore.

    `_DECODE_MAX` is read from the environment at module import
    time, so testing a different value means re-importing the
    module. Re-importing means a NEW module object with NEW class
    objects — `LoaderError` in particular. Any other test module
    that did `from indexer.image_loader import LoaderError` at
    import time holds the OLD class, and `pytest.raises(LoaderError)`
    then fails to catch the new one.

    The original module object is untouched by the re-import (the
    reload happens in a fresh namespace), so restoring it in
    `sys.modules` puts every later test back on the class identity
    it captured at import time.
    """
    original = sys.modules.get("indexer.image_loader")

    def _reload(env_value: str | None):
        if env_value is None:
            monkeypatch.delenv("INDEXER_DECODE_MAX", raising=False)
        else:
            monkeypatch.setenv("INDEXER_DECODE_MAX", env_value)
        sys.modules.pop("indexer.image_loader", None)
        module = importlib.import_module("indexer.image_loader")
        return module

    yield _reload

    if original is not None:
        sys.modules["indexer.image_loader"] = original
    else:
        sys.modules.pop("indexer.image_loader", None)


class TestDecodeMaxDefault:
    """`INDEXER_DECODE_MAX` default changed from 0 -> 768.

    Reads `_DECODE_MAX` from the module — the value is captured at
    import time (env read at module top-level), so re-importing the
    module is the only way to test different defaults.
    """

    def test_default_is_768(self, reloaded_image_loader):
        loader = reloaded_image_loader(None)
        assert loader._DECODE_MAX == 768

    def test_env_override_works(self, reloaded_image_loader):
        loader = reloaded_image_loader("1024")
        assert loader._DECODE_MAX == 1024

    def test_env_zero_disables_cap(self, reloaded_image_loader):
        """Setting to 0 disables the cap (full decode)."""
        loader = reloaded_image_loader("0")
        assert loader._DECODE_MAX == 0
