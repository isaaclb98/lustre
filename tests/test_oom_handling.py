"""
Tests for CUDA OOM handling in the indexer.

Context: the embed call sat inside a bare `except Exception` that logged
"embed failed", counted the batch as errors, and continued. A VRAM OOM
was therefore indistinguishable from a corrupt JPEG — the run limped to
completion and reported a corpus with holes in it, and the only evidence
was one log line per batch.

An OOM is a run-level failure: every remaining batch would fail the same
way. The run must stop and say so.

These tests also cover the runner's handling of a "failed" progress
event: its comment promised the child's reason would be captured, but
the code never read the `error` field, so the status API reported only
"indexer exited with code 1".
"""
from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from indexer import local_sync as local_sync_mod
from indexer.local_sync import _describe_oom, _is_cuda_oom

COLLECTION = "images_test_oom"


# ── Detection ────────────────────────────────────────────────────────────
class _FakeOOM(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError."""


_FakeOOM.__name__ = "OutOfMemoryError"


def test_torch_style_oom_is_detected() -> None:
    """torch.cuda.OutOfMemoryError has the class name OutOfMemoryError."""
    assert _is_cuda_oom(_FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB")) is True


def test_runtime_error_with_cuda_oom_text_is_detected() -> None:
    """Some CUDA paths raise a plain RuntimeError with the OOM text."""
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert _is_cuda_oom(exc) is True


def test_ordinary_errors_are_not_oom() -> None:
    """A corrupt JPEG must NOT be treated as an OOM — it stays a per-file error."""
    assert _is_cuda_oom(ValueError("invalid JPEG data")) is False
    assert _is_cuda_oom(OSError("No space left on device")) is False
    assert _is_cuda_oom(MemoryError("host RAM exhausted")) is False


def test_host_ram_oom_is_not_a_cuda_oom() -> None:
    """A CPU-only host OOMs with MemoryError, not a CUDA OOM."""
    assert _is_cuda_oom(MemoryError("Unable to allocate 2.00 GiB")) is False


def test_describe_oom_mentions_the_batch_size_guidance() -> None:
    """The message must tell the operator what to change."""
    desc = _describe_oom(_FakeOOM("CUDA out of memory"))
    assert "OutOfMemoryError" in desc
    assert "INDEXER_BATCH_SIZE" in desc
    assert "EMBED_BATCH_SIZE" in desc


# ── The run aborts instead of limping on ─────────────────────────────────
def _make_png(tmp_path: Path, name: str) -> Path:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow not available")
    p = tmp_path / name
    Image.new("RGB", (16, 16), (10, 90, 200)).save(p)
    return p


@pytest.fixture
def qdrant_raw():
    # An in-memory Qdrant is discarded with the client — no cleanup
    # delete needed (and calling one would trip the destructive-call
    # architecture guard for no reason).
    return QdrantClient(location=":memory:")


def _run_with_raising_encoder(monkeypatch, raw, source_dir: Path, exc: Exception) -> tuple[int, list[dict]]:
    """Run local_sync with an encoder that raises `exc` on every embed.

    Returns (exit_code, emitted_progress_events).
    """
    monkeypatch.setattr(local_sync_mod, "make_client", lambda _: raw)

    class _BoomEncoder:
        def __init__(self, *a, **kw):
            pass

        def embed_batch(self, images):
            raise exc

    monkeypatch.setattr(local_sync_mod, "VisionEncoder", _BoomEncoder)

    events: list[dict] = []

    def _capture(payload: dict) -> None:
        events.append(payload)

    monkeypatch.setattr(local_sync_mod, "_emit_progress", _capture)

    rc = local_sync_mod.main([
        "--source", str(source_dir), "--source-name", "x",
        "--qdrant-collection", COLLECTION,
        "--device", "cpu",
        "--model", "mock-1536",
        "--batch-size", "2",
    ])
    return rc, events


def test_oom_aborts_the_run_with_a_failure_event(monkeypatch, qdrant_raw, tmp_path) -> None:
    """A CUDA OOM stops the run and emits a 'failed' event naming the cause."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        _make_png(src, f"{i}.png")

    rc, events = _run_with_raising_encoder(
        monkeypatch, qdrant_raw, src, _FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB"),
    )

    assert rc == 1, "an OOM must exit non-zero, not report success"
    failed = [e for e in events if e.get("event") == "failed"]
    assert len(failed) == 1, "exactly one failure event"
    assert failed[0]["stage"] == "embed"
    assert "out of memory" in failed[0]["error"].lower()
    assert "INDEXER_BATCH_SIZE" in failed[0]["error"]


def test_oom_does_not_index_any_points(monkeypatch, qdrant_raw, tmp_path) -> None:
    """The run must not leave a half-embedded corpus behind."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        _make_png(src, f"{i}.png")

    rc, _ = _run_with_raising_encoder(
        monkeypatch, qdrant_raw, src, _FakeOOM("CUDA out of memory"),
    )

    assert rc == 1
    if qdrant_raw.collection_exists(COLLECTION):
        count = qdrant_raw.count(collection_name=COLLECTION, exact=True).count
        assert count == 0, "no points should have been written"


def test_a_non_oom_error_is_still_a_per_file_error(monkeypatch, qdrant_raw, tmp_path) -> None:
    """A corrupt-image error keeps the old behaviour: counted, run continues."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        _make_png(src, f"{i}.png")

    rc, events = _run_with_raising_encoder(
        monkeypatch, qdrant_raw, src, ValueError("cannot identify image file"),
    )

    # Not an OOM -> no 'failed' event, the run completes as before.
    assert not [e for e in events if e.get("event") == "failed"]
    assert rc == 0


# ── The runner surfaces the child's reason ───────────────────────────────
def test_runner_captures_the_failed_event_error() -> None:
    """The runner must put the child's reason into last_error.

    Its own comment claimed it did this ("Use the last error from the
    most recent 'failed' progress event if we captured one"), but
    _handle_event never read the `error` field — so the status API
    reported only "indexer exited with code 1".
    """
    from search.indexer_runner import IndexerRunner

    runner = IndexerRunner(command_factory=lambda mode: ["true"])
    runner._handle_event({
        "event": "failed",
        "stage": "embed",
        "error": "CUDA out of memory: free VRAM now: 120 MiB",
        "indexed": 10,
        "reembedded": 0,
        "skipped": 0,
        "errors": 8,
    })

    assert runner._last_error is not None
    assert "out of memory" in runner._last_error.lower()
    assert "free VRAM" in runner._last_error


def test_runner_without_an_error_field_falls_back_to_the_exit_code() -> None:
    """A failed event with no 'error' must not blank the status."""
    from search.indexer_runner import IndexerRunner

    runner = IndexerRunner(command_factory=lambda mode: ["true"])
    runner._handle_event({"event": "failed", "indexed": 0, "errors": 3})

    assert runner._last_error is None
