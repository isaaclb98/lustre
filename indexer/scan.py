"""
indexer/scan.py

Folder walking for the indexer.

Critical invariant: we return a list (snapshot), not a generator.
On Windows/NTFS and SMB mounts, `os.scandir` re-yields renamed files
mid-iteration — see the lesson in
~/.openclaw/workspace-task/memory/self-improving/2026-06-04-find-images-snapshot.md

The indexer renames/mutates (writes sidecars, etc.) — so we must
materialize the file set BEFORE any iteration that could trigger
mid-walk renames. Snapshot, then iterate.

Speed: uses `os.scandir` directly and filters on DirEntry attributes
(is_file, name, suffix) — no `Path.is_file()` per file, which would be
a redundant `os.stat` syscall. On a Windows + SMB mount, the naive
`Path.rglob` + `Path.is_file()` approach is ~2x slower because of the
extra stat call per file.

Cooperative cancel (round 31): `snapshot()` accepts a `should_cancel`
callback that's polled every `cancel_check_every` files. On True,
the walk aborts with `ScanCancelled`. local_sync wires this to its
SIGTERM-driven `_is_cancelled()` so cancel is responsive even during
the filesystem walk — important for million-file libraries where the
walk alone takes minutes.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ScanCancelled(Exception):
    """Raised by `snapshot()` when its `should_cancel` callback returns True.

    Lets the caller distinguish a user-driven abort from a normal
    empty-walk return value. The indexer catches this and exits with
    code 130 (SIGTERM convention) plus a "cancelled" progress event.
    """

# Image extensions we know how to embed.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".heic", ".heif"}
)

# Names that should never be embedded.
SKIP_NAMES: frozenset[str] = frozenset(
    {"thumbs.db", ".ds_store"}
)


def is_image_suffix(suffix: str) -> bool:
    """True if `suffix` (e.g. ".jpg") is a known image extension."""
    return suffix.lower() in IMAGE_EXTENSIONS


def should_skip_name(name: str) -> bool:
    """True if `name` is junk (hidden, OS metadata, etc.)."""
    if name.startswith("."):  # hidden files, ._foo (macOS resource forks)
        return True
    return name.lower() in SKIP_NAMES


def is_image(path: Path) -> bool:
    """True if `path` has a known image extension."""
    return is_image_suffix(path.suffix)


def should_skip(path: Path) -> bool:
    """True if `path` is a junk file (hidden, OS metadata, etc.)."""
    return should_skip_name(path.name)


def _format_eta(seconds: float) -> str:
    """Format a duration as mm:ss (or h:mm:ss past an hour)."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def snapshot(
    source: Path,
    progress_every: int = 10_000,
    expected_total: int | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
    cancel_check_every: int = 5_000,
) -> list[Path]:
    """
    Return a stable, sorted list of image paths under `source`.

    Materialized as a list (not a generator) so that subsequent
    iteration is safe against mid-walk directory mutations.

    Recurses through subdirectories. Filters by extension and skips
    junk files.

    `progress_every` controls how often a progress line is logged
    (every N image files found). `expected_total` is an optional
    estimate of the total image count (e.g. from Qdrant's per-source
    count) used to print a time-to-completion estimate alongside the
    running rate; without it the line shows elapsed time instead.

    `should_cancel` (optional) is a zero-arg callable polled every
    `cancel_check_every` files; if it returns True the walk aborts
    with `ScanCancelled`. Defaults to None (no cancellation —
    backwards compatible with all existing callers).
    """
    if not source.exists():
        raise FileNotFoundError(f"source path does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"source is not a directory: {source}")

    logger.info("scan: walking %s ...", source)
    if expected_total is not None:
        logger.info("scan: expecting ~%d image files for this source", expected_total)
    results: list[Path] = []
    state = {
        "count": 0,
        "next": progress_every,
        "t0": time.monotonic(),
        "next_cancel": cancel_check_every,
    }
    try:
        _walk_into(
            source, results, state, progress_every, expected_total,
            should_cancel, cancel_check_every,
        )
    except _ScanAbort:
        # Translated here (not in _walk_into) so the cancel sentinel
        # stays an internal detail.
        raise ScanCancelled() from None
    results.sort()
    return results


class _ScanAbort(Exception):
    """Internal sentinel — raised inside _walk_into, translated to
    `ScanCancelled` by `snapshot()`. Callers should never see this."""


def _walk_into(
    source: Path,
    out: list[Path],
    state: dict,
    progress_every: int,
    expected_total: int | None,
    should_cancel: Callable[[], bool] | None = None,
    cancel_check_every: int = 5_000,
) -> None:
    """
    Recursive scandir-based walk. Appends matching Path objects to `out`.

    Uses DirEntry attributes (is_file, name, suffix) which are already
    cached by the underlying syscall — no extra stat() per entry.

    `state` carries the running file count, the next threshold to log
    at, and the walk start time; `progress_every` is the increment
    between thresholds; `expected_total` (optional) drives the ETA.

    `should_cancel` (optional) is polled every `cancel_check_every`
    files. On True, raises `_ScanAbort` which `snapshot()` translates
    to `ScanCancelled`. Checks happen at the same place as progress
    logging — once per `cancel_check_every` matched images — to avoid
    the syscall overhead of a per-file poll.
    """
    try:
        with os.scandir(source) as it:
            for entry in it:
                try:
                    if entry.is_file():
                        name = entry.name
                        if should_skip_name(name):
                            continue
                        if is_image_suffix(Path(name).suffix):
                            out.append(Path(entry.path))
                            state["count"] += 1
                            if state["count"] >= state["next"]:
                                state["next"] += progress_every
                                _log_scan_progress(state, source, expected_total)
                            # Cooperative cancel: only check at the
                            # same cadence as progress logging so
                            # we're not adding per-file overhead.
                            # cancel_check_every defaults to 5000
                            # which gives ~1s responsiveness at the
                            # observed ~5000 files/s SMB rate.
                            if (
                                should_cancel is not None
                                and state["count"] >= state["next_cancel"]
                            ):
                                state["next_cancel"] += cancel_check_every
                                if should_cancel():
                                    raise _ScanAbort()
                    elif entry.is_dir(follow_symlinks=False):
                        # Recurse into subdirectories. Skip symlinks to
                        # avoid loops on weird NAS layouts.
                        _walk_into(
                            Path(entry.path), out, state, progress_every,
                            expected_total, should_cancel, cancel_check_every,
                        )
                except OSError:
                    # Permission errors, broken links, etc. — skip silently.
                    continue
                except _ScanAbort:
                    raise
    except OSError:
        # Top-level source unreadable. surface via the FileNotFoundError
        # checks in snapshot() before we ever get here; if it does fail
        # mid-walk, just stop — the partial result is still useful.
        pass


def existing_image_paths(
    source: Path,
    should_cancel: Callable[[], bool] | None = None,
    cancel_check_every: int = 5_000,
) -> set[str]:
    """The set of resolved image paths under `source`.

    The prune compares this set against the `path` payload of every
    point in the collection, and that payload is always an image path
    (`str(path.resolve())` in upsert.py) — so only image files need
    to be in the set. Collecting every file under the tree (the
    thumbnails, sidecars and NFO files that sit next to the photos)
    is pure waste, and on the NAS it is most of the walk.

    Uses `os.scandir` and the DirEntry attributes, the same way
    `snapshot()` does: `Path.rglob("*")` plus a per-entry `is_file()`
    costs an extra stat syscall per entry, and `resolve()` on an SMB
    mount can cost a round trip per path component.
    """
    found: set[str] = set()
    count = 0
    next_cancel = cancel_check_every

    def walk(directory: Path) -> None:
        nonlocal count, next_cancel
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            name = entry.name
                            if should_skip_name(name):
                                continue
                            if not is_image_suffix(Path(name).suffix):
                                continue
                            found.add(str(Path(entry.path).resolve()))
                            count += 1
                            if (
                                should_cancel is not None
                                and count >= next_cancel
                            ):
                                next_cancel += cancel_check_every
                                if should_cancel():
                                    raise _ScanAbort()
                        elif entry.is_dir(follow_symlinks=False):
                            walk(Path(entry.path))
                    except OSError:
                        continue
                    except _ScanAbort:
                        raise
        except OSError:
            pass

    try:
        walk(source)
    except _ScanAbort:
        raise ScanCancelled(f"cancelled while walking {source}") from None
    return found


def _log_scan_progress(state: dict, source: Path, expected_total: int | None) -> None:
    """Emit one heartbeat line: count, rate, and ETA (if known) or elapsed."""
    elapsed = time.monotonic() - state["t0"]
    count = state["count"]
    rate = count / elapsed if elapsed > 0 else 0.0
    if expected_total is not None and rate > 0:
        remaining = max(0, expected_total - count) / rate
        logger.info(
            "scan: %d image files so far (%.0f/s, ETA %s) walking %s",
            count, rate, _format_eta(remaining), source,
        )
    else:
        logger.info(
            "scan: %d image files so far (%.0f/s, %s elapsed) walking %s",
            count, rate, _format_eta(elapsed), source,
        )
