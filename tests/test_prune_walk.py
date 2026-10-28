"""The prune's alive set: a contract on how it is built.

The prune compares the set of existing files under the source dirs
against the `path` payload of every point in the collection. It used
to build that set with `Path.rglob("*")` plus a per-entry
`is_file()` — the exact anti-pattern `indexer/scan.py` was written
to avoid — and it put every file in the tree into the set, when only
image paths are ever compared against it.

On the NAS the walk ran at 445-1,483 entries/s over 2.35M entries,
so the prune spent ~56 minutes walking before a single photo could
be embedded. `existing_image_paths` walks the same tree with
`os.scandir` and keeps only the images: 16.9x faster on a real
directory (37.7s -> 2.2s for 34,200 files).
"""
from __future__ import annotations

import inspect
from pathlib import Path


# ---------------------------------------------------------------------------
# the prune's alive set
# ---------------------------------------------------------------------------


def test_existing_image_paths_collects_only_images(tmp_path: Path) -> None:
    """The prune's alive set only needs the image paths.

    The set is compared against the `path` payload of every point in
    the collection, and that payload is always an image path
    (`str(path.resolve())` in upsert.py). Collecting the thumbnails,
    sidecars and NFO files that sit next to the photos is pure
    waste — and on the NAS it is most of the walk.
    """
    from indexer.scan import existing_image_paths

    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "thumb.bin").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpeg").write_bytes(b"x")
    (sub / "sidecar.nfo").write_bytes(b"x")

    found = existing_image_paths(tmp_path)
    names = {Path(p).name for p in found}
    assert names == {"a.jpg", "b.png", "c.jpeg"}, names
    assert all(Path(p).is_absolute() for p in found)


def test_existing_image_paths_matches_the_prunes_path_format(
    tmp_path: Path
) -> None:
    """The set's paths must match the payload's path format.

    upsert.py stores `str(path.resolve())` in the payload; the prune
    compares that string against this set. A different format means
    every point looks like an orphan.
    """
    from indexer.scan import existing_image_paths

    (tmp_path / "a.jpg").write_bytes(b"x")
    found = existing_image_paths(tmp_path)
    assert found == {str((tmp_path / "a.jpg").resolve())}


def test_existing_image_paths_handles_a_missing_directory(
    tmp_path: Path
) -> None:
    from indexer.scan import existing_image_paths

    assert existing_image_paths(tmp_path / "nope") == set()


def test_existing_image_paths_skips_hidden_files(tmp_path: Path) -> None:
    from indexer.scan import existing_image_paths

    (tmp_path / ".hidden.jpg").write_bytes(b"x")
    (tmp_path / "real.jpg").write_bytes(b"x")
    found = existing_image_paths(tmp_path)
    assert {Path(p).name for p in found} == {"real.jpg"}


def test_the_prune_does_not_use_rglob() -> None:
    """The prune must not walk the tree with `Path.rglob` again.

    `indexer/scan.py` exists because `Path.rglob("*")` plus a
    per-entry `is_file()` costs an extra stat syscall per entry, and
    the prune was doing exactly that while collecting every file in
    the tree. On the NAS the walk ran at 445-1,483 entries/s and
    delayed every photo from being embedded.
    """
    import ast

    from indexer import upsert

    tree = ast.parse(inspect.getsource(upsert))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "prune_missing"
    )
    # Look at the code, not the prose: a comment explaining what was
    # replaced is not a call.
    called = {
        node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "rglob" not in called, (
        "the prune is walking the tree with rglob again"
    )
    assert "is_file" not in called, (
        "the prune is doing a per-entry stat again"
    )
    src = inspect.getsource(upsert)
    start = src.index("def prune_missing")
    end = src.find("\ndef ", start + 10)
    assert "existing_image_paths" in src[start:end if end > 0 else len(src)]
