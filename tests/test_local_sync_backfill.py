"""
tests/test_local_sync_backfill.py

Layer 1 — backfill mode (--reblurhash / --refingerprint) of local_sync.

The backfill walks an existing Qdrant collection and rewrites only
the requested payload field per point, reading the source file
from disk. No re-embedding; the 1536-dim vector stays untouched.

These tests guard:
  * --reblurhash and --refingerprint are mutually exclusive
  * Points with the right field already set are skipped (idempotent)
  * Points under a different `source` payload value are left alone
    (the Windows backfill should never rewrite other-machine points)
  * Missing-source-file paths are recorded as failed, not crashed on

The in-memory Qdrant client from the fixture is injected into
`local_sync.make_client` via monkeypatch so the test owns the same
Qdrant instance the CLI is reading from.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from indexer import local_sync as local_sync_mod
from indexer.blurhash import compute_blurhash, is_valid_blurhash
from indexer.upsert import VECTOR_DIM
from lustre_kernel.spaces import TEXT_SPACE

COLLECTION = "images_test_backfill"


def _pid(name: str) -> str:
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"backfill-{name}"))


def _make_png(tmp_path: Path, name: str) -> Path:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow not available")
    p = tmp_path / name
    Image.new("RGB", (16, 16), (128, 200, 64)).save(p)
    return p


def _seed_point(raw, name: str, path_str: str, source: str, blurhash=None, dhash=None, sha=None) -> None:
    """Insert a point with a tiny dummy vector; blurhash/fingerprints
    can be null to simulate a backfill target."""
    raw.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(
            id=_pid(name),
            vector={TEXT_SPACE: [0.0] * VECTOR_DIM},
            payload={
                "id": _pid(name), "path": path_str, "shard": "",
                "collection": source, "mtime": 100, "size": 200,
                "indexed_at": "2026-01-01T00:00:00+00:00",
                # nullable fields the backfill will rewrite:
                "blurhash": blurhash, "dhash": dhash,
                "content_sha256": sha,
            },
        )],
        wait=True,
    )


def _get_point(raw, name: str) -> dict:
    pts = raw.retrieve(collection_name=COLLECTION, ids=[_pid(name)], with_payload=True)
    assert pts, f"point {name} not found"
    return pts[0].payload


@pytest.fixture
def backfill_env(tmp_path):
    raw = QdrantClient(location=":memory:")
    from indexer.sync_meta import ensure_sync_collections
    ensure_sync_collections(raw, COLLECTION)
    yield {"raw": raw, "tmp": tmp_path}


def _run_with_fixture_client(monkeypatch, raw, args):
    """Patch local_sync.make_client to return the fixture's in-memory
    Qdrant, then invoke main(). Returns the exit code.

    Tests that exercise the embed path inject `--model mock-1536` so
    the registry's deterministic mock is used; the real
    SigLIP2-384 model is multi-GB and hangs on CPU. The path-shape,
    payload-shape, and prune-set invariants are identical regardless
    of which embedder is used, so the mock is sufficient.
    """
    monkeypatch.setattr(local_sync_mod, "make_client", lambda a: raw)
    if "--model" not in args:
        args = ["--model", "mock-1536"] + list(args)
    return local_sync_mod.main(args)


def test_reblurhash_and_refingerprint_are_mutually_exclusive(monkeypatch, backfill_env, tmp_path):
    src = tmp_path / "img"
    src.mkdir()
    (src / "a.png").write_bytes(b"x")
    rc = _run_with_fixture_client(
        monkeypatch, backfill_env["raw"],
        ["--source", str(src), "--source-name", "x",
         "--qdrant-collection", COLLECTION,
         "--reblurhash", "--refingerprint"],
    )
    assert rc == 2


def test_reblurhash_backfills_missing_blurhash(monkeypatch, backfill_env, tmp_path):
    raw = backfill_env["raw"]
    img = _make_png(tmp_path, "a.png")
    _seed_point(raw, "a", str(img), source="x", blurhash=None, dhash=None, sha=None)

    rc = _run_with_fixture_client(
        monkeypatch, raw,
        ["--source", str(tmp_path), "--source-name", "x",
         "--qdrant-collection", COLLECTION, "--reblurhash"],
    )
    assert rc == 0
    payload = _get_point(raw, "a")
    assert payload.get("blurhash") is not None, "blurhash should have been written"
    assert is_valid_blurhash(payload["blurhash"]), (
        f"backfilled blurhash failed is_valid_blurhash: {payload['blurhash']!r}"
    )


def test_reblurhash_skips_already_correct(monkeypatch, backfill_env, tmp_path):
    raw = backfill_env["raw"]
    img = _make_png(tmp_path, "a.png")
    expected = compute_blurhash(img)
    _seed_point(raw, "a", str(img), source="x", blurhash=expected)

    rc = _run_with_fixture_client(
        monkeypatch, raw,
        ["--source", str(tmp_path), "--source-name", "x",
         "--qdrant-collection", COLLECTION, "--reblurhash"],
    )
    assert rc == 0
    payload = _get_point(raw, "a")
    assert payload.get("blurhash") == expected


def test_refingerprint_backfills_dhash_and_sha(monkeypatch, backfill_env, tmp_path):
    raw = backfill_env["raw"]
    img = _make_png(tmp_path, "a.png")
    _seed_point(raw, "a", str(img), source="x", blurhash=None, dhash=None, sha=None)

    rc = _run_with_fixture_client(
        monkeypatch, raw,
        ["--source", str(tmp_path), "--source-name", "x",
         "--qdrant-collection", COLLECTION, "--refingerprint"],
    )
    assert rc == 0
    payload = _get_point(raw, "a")
    assert payload.get("dhash"), f"dhash should have been written, got {payload.get('dhash')!r}"
    assert payload.get("content_sha256"), "content_sha256 should have been written"


def test_backfill_does_not_touch_other_source(monkeypatch, backfill_env, tmp_path):
    """The backfill should only rewrite points whose payload.source
    matches one of the --source-name values. Points from a different
    pipeline (e.g. a different machine) must be left alone — the
    Windows runner doesn't know how to load their files anyway."""
    raw = backfill_env["raw"]
    img = _make_png(tmp_path, "a.png")
    _seed_point(raw, "a", str(img), source="x", blurhash=None)
    _seed_point(raw, "b", "/some/other/machine/path/b.png", source="other-lib",
                blurhash=None, dhash=None, sha=None)

    rc = _run_with_fixture_client(
        monkeypatch, raw,
        ["--source", str(tmp_path), "--source-name", "x",
         "--qdrant-collection", COLLECTION, "--reblurhash"],
    )
    assert rc == 0
    assert _get_point(raw, "a").get("blurhash") is not None, "x source should be backfilled"
    assert _get_point(raw, "b").get("blurhash") is None, "other-lib source must NOT be touched"


def test_backfill_handles_missing_file(monkeypatch, backfill_env, tmp_path):
    """A point whose payload.path no longer exists on disk should be
    recorded as failed (or skipped) without crashing the whole run."""
    raw = backfill_env["raw"]
    _seed_point(raw, "a", str(tmp_path / "does-not-exist.png"),
                source="x", blurhash=None)
    rc = _run_with_fixture_client(
        monkeypatch, raw,
        ["--source", str(tmp_path), "--source-name", "x",
         "--qdrant-collection", COLLECTION, "--reblurhash"],
    )
    # We don't assert a specific exit code here — the run might return
    # 0 (only "failed" counter incremented) or 1 (any failure means
    # non-zero). Both are acceptable; the key is "no crash".
    assert rc in (0, 1)
    assert _get_point(raw, "a") is not None


def test_prune_one_source_does_not_delete_other_source(monkeypatch, backfill_env, tmp_path):
    """Isaac's exact scenario: run with ONLY kpop/collections managed
    and --prune. Points belonging to kpop/data (not in this run's
    source_names) must survive, even though their files aren't in the
    collections walk."""
    raw = backfill_env["raw"]
    # managed source: kpop/collections, with a live file at an
    # absolute host path (same path the indexer would have stored)
    collections_dir = tmp_path / "kpop" / "collections"
    collections_dir.mkdir(parents=True)
    live = _make_png(tmp_path, "live.png")
    live_path = collections_dir / "live.png"
    live_path.write_bytes(live.read_bytes())

    # unmanaged source: data — a point whose file does NOT exist on
    # disk. It must NOT be pruned because it's out of this run's scope.
    data_path = tmp_path / "kpop" / "data" / "missing.png"
    _seed_point(raw, "data_gone", str(data_path), source="kpop/data",
                blurhash=None, dhash=None, sha=None)

    # managed source point that IS alive (also seed, for completeness)
    _seed_point(raw, "coll_live", str(live_path), source="kpop/collections",
                blurhash=None, dhash=None, sha=None)

    rc = _run_with_fixture_client(
        monkeypatch, raw,
        ["--source", str(collections_dir), "--source-name", "kpop/collections",
         "--qdrant-collection", COLLECTION,
         "--device", "cpu", "--prune"],
    )
    assert rc == 0
    # unmanaged source survives even though its file is gone
    pts = raw.retrieve(collection_name=COLLECTION, ids=[_pid("data_gone")], with_payload=True)
    assert pts, "kpop/data point must NOT be pruned by a kpop/collections-only run"
    # managed source live point survives
    assert _get_point(raw, "coll_live") is not None
