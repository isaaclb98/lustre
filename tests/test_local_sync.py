"""
tests/test_local_sync.py

Layer 1 — blurhash and source-name presence in local_sync payloads.

`indexer.local_sync` is the new unified sync+embed CLI. It calls
`upsert.build_payload` for every photo, which already computes
blurhash + fingerprints. This test guards that contract so a future
refactor of build_payload can't silently drop blurhash (the client
side uses it for instant placeholders during image load — dropping
it would cause a visible UX regression).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indexer import local_sync as local_sync_mod
from indexer.upsert import VECTOR_DIM, build_payload


def _make_png(tmp_path: Path, name: str = "a.png", size: tuple[int, int] = (16, 16)) -> Path:
    """Write a tiny valid PNG so blurhash can actually compute a hash."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow not available")
    p = tmp_path / name
    Image.new("RGB", size, (255, 128, 0)).save(p)
    return p


def test_build_payload_includes_blurhash(tmp_path: Path) -> None:
    """Every local_sync upsert goes through upsert.build_payload; the
    blurhash field must be present and valid (non-empty, printable
    ASCII, within the structural bounds checked by `is_valid_blurhash`).
    """
    png = _make_png(tmp_path)
    payload = build_payload(png, shard="", model_name="ViT-gopt-16-SigLIP2-384",
                           model_revision="", collection="kpop/data")
    assert "blurhash" in payload, (
        "build_payload must include 'blurhash' — local_sync relies on it for LQIP"
    )
    bh = payload["blurhash"]
    assert bh is not None and isinstance(bh, str), f"blurhash should be a non-empty string, got {bh!r}"
    # Validate via the same guard the blurhash module uses.
    from indexer.blurhash import is_valid_blurhash
    assert is_valid_blurhash(bh), f"blurhash {bh!r} failed is_valid_blurhash check"


def test_build_payload_uses_source_name(tmp_path: Path) -> None:
    """The source-name flag flows through to the payload.collection field,
    which the search UI uses for filtering. If local_sync ever drops
    or misnames this field, source filters break silently.
    """
    png = _make_png(tmp_path)
    for source_name in ("kpop/data", "kpop/collections", "my-lib"):
        payload = build_payload(png, shard="", model_name="ViT-gopt-16-SigLIP2-384",
                               model_revision="", collection=source_name)
        assert payload.get("collection") == source_name, (
            f"source mismatch: passed {source_name!r}, got {payload.get('collection')!r}"
        )


def test_vector_dim_matches_siglip2() -> None:
    """The vector dim is shared between the embedder and the Qdrant
    collection config. If either side drifts, the in-memory Qdrant
    backend will reject upserts with a shape error.

    Post-migration: VECTOR_DIM is sourced from the model registry
    keyed on DEFAULT_MODEL (so400m-patch16-384). This test now
    pins the *contract* (dim matches the registry) rather than a
    specific literal — any future variant change will update
    VECTOR_DIM automatically, and this assertion stays correct.
    """
    from lustre_kernel.registry import get as _registry_get
    from search.config import DEFAULT_MODEL
    expected_dim = _registry_get(DEFAULT_MODEL).dim
    assert expected_dim == VECTOR_DIM, (
        f"VECTOR_DIM={VECTOR_DIM} should match registry dim for "
        f"{DEFAULT_MODEL} ({expected_dim})"
    )


def test_local_sync_rejects_missing_source_dir(tmp_path: Path) -> None:
    """Smoke test: local_sync's argparse should reject a non-existent
    --source with exit code 2 (matches the convention used by
    indexer.py for the same check). Catches accidental removal of
    the validation when refactoring.
    """
    bogus = tmp_path / "does-not-exist"
    rc = local_sync_mod.main(
        ["--source", str(bogus), "--source-name", "x", "--dry-run"]
    )
    assert rc == 2, f"expected exit 2 for missing source, got {rc}"


def test_local_sync_rejects_source_name_count_mismatch(tmp_path: Path) -> None:
    """`--source-name` count must be 0, 1, or equal to `--source` count.
    Otherwise the pairing is ambiguous and the user almost certainly
    typed the wrong thing. Catches the regression where we drop
    the count check.
    """
    src = tmp_path / "real"
    src.mkdir()
    rc = local_sync_mod.main([
        "--source", str(src), "--source-name", "a", "--source-name", "b", "--dry-run",
    ])
    assert rc == 2, f"expected exit 2 for --source-name count mismatch, got {rc}"
