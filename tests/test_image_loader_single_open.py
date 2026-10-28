"""The indexer must open each photo once, not twice.

`peek_source_dims()` and `load_image_pil()` each opened the file, so
the indexer paid two network round trips per photo. Measured on the
NAS: 19.7ms for the header open, 28.6ms for the decode open, 41.9ms
for the pair — the second open was about half the cost of loading a
photo, and the load stage is the pipeline's documented ceiling
("load = 21 ms ... load is the ceiling").

`load_image_pil_with_dims()` does both from one open.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (640, 480), (120, 30, 200)).save(p, quality=90)
    return p


def test_load_image_pil_with_dims_returns_the_image_and_its_size(
    photo: Path,
) -> None:
    from indexer.image_loader import load_image_pil_with_dims

    img, w, h = load_image_pil_with_dims(photo)
    assert img.size == (640, 480)
    assert (w, h) == (640, 480)
    assert img.mode == "RGB"


def test_the_dims_match_peek_source_dims(photo: Path) -> None:
    """The values must be identical to the separate header read.

    The payload stores these dims and the photo page displays them,
    so a difference would change what the UI shows.
    """
    from indexer.image_loader import load_image_pil_with_dims, peek_source_dims

    _, w, h = load_image_pil_with_dims(photo)
    assert (w, h) == peek_source_dims(photo)


def test_the_file_is_opened_only_once(photo: Path, monkeypatch) -> None:
    """One `Image.open` per photo, not two.

    The count is what matters, not the timing — a timing test goes
    flaky on a loaded machine.
    """
    from indexer import image_loader

    opens: list[Path] = []
    real_open = image_loader.Image.open

    def counting_open(fp, *args, **kwargs):
        opens.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(image_loader.Image, "open", counting_open)
    image_loader.load_image_pil_with_dims(photo)
    assert len(opens) == 1, f"the photo was opened {len(opens)} times"


def test_the_batch_loader_uses_the_single_open_helper() -> None:
    """The concurrent batch loader must not call the two helpers.

    `local_sync._load_batch_concurrent` is the path the indexer
    actually takes, and it is where the double open lived.
    """
    from indexer import local_sync

    src = inspect.getsource(local_sync._load_batch_concurrent)
    assert "load_image_pil_with_dims" in src
    assert "peek_source_dims" not in src
    assert "load_image_pil(" not in src


def test_the_load_helper_uses_the_single_open_helper() -> None:
    """The single-path `load()` helper too."""
    from indexer import image_loader

    tree = ast.parse(inspect.getsource(image_loader.load))
    # `executor.submit(fn, ...)` references the function rather than
    # calling it, so look at every name the body mentions.
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert "load_image_pil_with_dims" in names
    assert "peek_source_dims" not in names
    assert "load_image_pil" not in names


def test_the_helpers_are_not_called_in_sequence() -> None:
    """A guard against the two-open shape coming back.

    `peek_source_dims` and `load_image_pil` both open the file; a
    caller that uses both pays for two opens.
    """
    from indexer import image_loader

    tree = ast.parse(inspect.getsource(image_loader))
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        called = {
            node.func.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "peek_source_dims" in called and "load_image_pil" in called:
            pytest.fail(
                f"{fn.name} calls both peek_source_dims and "
                "load_image_pil — that is two file opens per photo"
            )
