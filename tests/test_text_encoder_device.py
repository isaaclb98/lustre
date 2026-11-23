"""
The API's text tower and the indexer's vision tower must be able to
sit on DIFFERENT devices.

The API and the indexer share one container: the API answers text
queries, the indexer (a child process) embeds photos. Both resolve
the DEVICE env var, so before this was split the API's first text
query pulled the whole model onto the GPU and took memory away from
a running embed — on a card where that difference is the difference
between a run that finishes and an Xid 79 that takes the card off
the PCIe bus.

The text tower follows TEXT_ENCODER_DEVICE (cpu by default); the
vision tower follows DEVICE.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_registry():
    """Build a registry with the real radio model, under the current env.

    The default registry is a process-global singleton, and the
    embedders resolve their device when they are BUILT. So each test
    needs its own registry, built after the env is set.
    """
    from lustre_kernel import _radio_models
    from lustre_kernel.registry import Registry

    def _build() -> Registry:
        registry = Registry()
        _radio_models.register_into(registry)
        return registry

    return _build


def _cuda_is_available() -> bool:
    """Whether the card is visible, without importing torch here.

    `torch` may not be imported outside the kernel (the architecture
    test enforces it), so ask the kernel's own embedder instead —
    it resolves "cuda" to "cpu" when the card is not there, which is
    exactly the fallback being tested.
    """
    from lustre_kernel._radio_models import RadioEmbedder

    return RadioEmbedder(dim=1536, long_edge=480, device="cuda")._device == "cuda"


def test_the_text_tower_stays_off_the_gpu(monkeypatch, fresh_registry):
    """DEVICE=cuda must not drag the text tower onto the GPU."""
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.delenv("TEXT_ENCODER_DEVICE", raising=False)

    spec = fresh_registry().get("C-RADIOv4-SO400M")
    assert spec.text._device == "cpu", (
        f"the text tower resolved to {spec.text._device!r} — the API's text "
        f"search would take VRAM from a running embed"
    )


def test_the_text_tower_follows_its_own_env_var(monkeypatch, fresh_registry):
    """TEXT_ENCODER_DEVICE is honoured for the text tower."""
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "cpu")

    spec = fresh_registry().get("C-RADIOv4-SO400M")
    assert spec.text._device == "cpu"


def test_the_vision_tower_follows_device(monkeypatch, fresh_registry):
    """The vision tower follows DEVICE — the indexer's --device flag."""
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "cpu")

    spec = fresh_registry().get("C-RADIOv4-SO400M")
    # "cuda" only if the card is actually visible; otherwise the
    # embedder falls back to cpu, which is the safe direction.
    expected = "cuda" if _cuda_is_available() else "cpu"
    assert spec.vision._device == expected


def test_the_two_towers_are_separate_embedders(monkeypatch, fresh_registry):
    """They must not be the same object.

    One object would mean one device — the two towers could not be
    split, and the split above would be a no-op.
    """
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "cpu")

    spec = fresh_registry().get("C-RADIOv4-SO400M")
    assert spec.text is not spec.vision


def test_the_api_text_encoder_runs_on_cpu(monkeypatch):
    """The end-to-end check: the API's encoder is on the CPU.

    This is the shape that actually bit: `TextEncoder(device="cpu")`
    was constructed with a CPU request, and the embedder it was
    handed still landed on the GPU.

    Test mode is off for this one — in test mode the encoder is the
    mock, which has no device at all.
    """
    monkeypatch.delenv("SEARCH_TEST_MODE", raising=False)
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "cpu")
    monkeypatch.setenv("MODEL_NAME", "C-RADIOv4-SO400M")
    monkeypatch.setenv("SIGLIP_VARIANT", "c-radio-v4/so400m")

    from search import text_encoder

    text_encoder.reset_encoder_for_tests()
    try:
        encoder = text_encoder.get_encoder()
        assert encoder._embedder._device == "cpu", (
            f"the API's text encoder resolved to {encoder._embedder._device!r}"
        )
    finally:
        text_encoder.reset_encoder_for_tests()
