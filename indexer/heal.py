"""
indexer/heal.py

Qdrant-direct reconciliation CLI for finding points whose source file
no longer exists under a filesystem tree.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from indexer.scan import is_image_suffix, should_skip_name
from lustre_kernel.qdrant_url import client_kwargs as _qdrant_client_kwargs

load_dotenv()


@dataclass
class DiskFile:
    path: str
    mtime: int
    size: int


@dataclass
class QdrantPoint:
    id: str
    path: str
    mtime: int | None = None
    size: int | None = None


@dataclass
class HealReport:
    source: Path
    collection: str
    total_points: int = 0
    total_files: int = 0
    orphans: list[QdrantPoint] = field(default_factory=list)
    new_files: list[DiskFile] = field(default_factory=list)
    modified_files: list[DiskFile] = field(default_factory=list)
    outside_scope: list[QdrantPoint] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="heal",
        description="Reconcile a photo directory against Qdrant.",
    )
    parser.add_argument("source", type=Path, nargs="?", default=None, help="source directory to walk (auto-detected from Qdrant paths if omitted)")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="images")
    parser.add_argument("--apply", action="store_true", help="delete orphan Qdrant points")
    parser.add_argument("--verbose", action="store_true", help="show every orphan path")
    parser.add_argument("--quiet", action="store_true", help="suppress live progress output (final report still prints)")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args(argv)


def make_client(args: argparse.Namespace) -> QdrantClient:
    return QdrantClient(**_qdrant_client_kwargs(
        url=args.qdrant_url,
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=30,
    ))


def reconcile(
    client: Any,
    source: Path | None,
    collection: str,
    batch_size: int = 1000,
    quiet: bool = False,
) -> HealReport:
    report = HealReport(source=source or Path(), collection=collection)
    if not quiet:
        print(f"Scrolling Qdrant collection '{collection}'...", flush=True)
    points = _scroll_points(client, collection, batch_size, quiet=quiet)
    report.total_points = len(points)
    if source is None:
        detected = _detect_source_dir([p.path for p in points if p.path])
        if detected is None:
            raise ValueError(
                "Could not auto-detect source dir: Qdrant collection is empty "
                "or all paths have no common ancestor. Pass the source dir explicitly."
            )
        print(f"Auto-detected source dir from Qdrant paths: {detected}", flush=True)
        source = detected
        report.source = source
    if not source.exists():
        raise FileNotFoundError(
            f"Auto-detected source dir does not exist on this machine: {source}. "
            f"Pass the source dir explicitly if your local mount differs."
        )
    if not source.is_dir():
        raise NotADirectoryError(f"Auto-detected source is not a directory: {source}")
    if not quiet:
        print(f"Walking {source}...", flush=True)
    files = _walk_files(source, report.errors, quiet=quiet)
    report.total_files = len(files)

    # Resolve the source dir to its canonical form so the in-scope check
    # works regardless of whether the user passed a drive-letter path
    # (Z:\...) or a UNC path (\\server\share\...). On Windows,
    # Path.resolve() returns the UNC equivalent of a drive-letter path
    # when the drive is mapped. resolve() on the source is cheap — one call.
    try:
        canonical_source: Path = source.resolve()
    except (OSError, RuntimeError):
        canonical_source = source

    # Walked files are already canonical (path.resolve() in _walk_files).
    files_by_path = {f.path: f for f in files}

    # In-scope check FIRST using the canonical source, with a cheap string
    # prefix compare — no per-point resolve() call. This is the hot path:
    # we need to know which Qdrant points are under the walked tree
    # before deciding which to canonicalize. resolve() on a missing file
    # also raises, so doing it after in-scope filtering means we only
    # resolve paths that have a real chance of mattering.
    in_scope_points: list[QdrantPoint] = []
    outside_scope_points: list[QdrantPoint] = []
    no_path_points: list[QdrantPoint] = []
    for p in points:
        if not p.path:
            no_path_points.append(p)
        elif _is_under(p.path, canonical_source):
            in_scope_points.append(p)
        else:
            outside_scope_points.append(p)

    # Build the canonical map ONLY for in-scope points (a much smaller
    # subset than the full collection). Each resolve() is a filesystem
    # syscall; doing it once per in-scope point is fine, doing it once
    # per Qdrant point in a 1M+ collection is not. resolve() may raise
    # for missing files (orphans) — we fall back to the raw path in
    # that case so the orphan check still detects them.
    points_by_canonical: dict[str, QdrantPoint] = {}
    for p in in_scope_points:
        points_by_canonical[_canonical_path(p.path)] = p

    walked_paths = set(files_by_path.keys())

    # Orphans: in-scope points whose canonical file path isn't in the walked set.
    orphans: list[QdrantPoint] = []
    for p in in_scope_points:
        if _canonical_path(p.path) not in walked_paths:
            orphans.append(p)
    orphans.extend(no_path_points)
    report.orphans = sorted(orphans, key=lambda p: p.path)

    # New files: walked files not in any in-scope Qdrant point.
    new_files: list[DiskFile] = []
    for path, f in files_by_path.items():
        if path not in points_by_canonical:
            new_files.append(f)
    report.new_files = sorted(new_files, key=lambda f: f.path)

    # Modified: in-scope Qdrant points with mtime/size mismatch.
    modified: list[DiskFile] = []
    for path, f in files_by_path.items():
        point = points_by_canonical.get(path)
        if point is None or not _is_under(point.path, canonical_source):
            continue
        if point.mtime != f.mtime or point.size != f.size:
            modified.append(f)
    report.modified_files = sorted(modified, key=lambda f: f.path)

    report.outside_scope = sorted(outside_scope_points, key=lambda p: p.path)
    return report


def _detect_source_dir(paths: list[str]) -> Path | None:
    """Return the longest common directory of all paths, or None if no common ancestor.

    Used as a fallback when the user runs heal without an explicit source dir:
    heal figures out the indexed root from the Qdrant payload paths themselves
    and walks that.
    """
    if not paths:
        return None
    try:
        common = os.path.commonpath(paths)
    except ValueError:
        # commonpath raises ValueError if paths have no common prefix
        # (e.g., different drives on Windows, or some are empty).
        return None
    if not common or common == ".":
        return None
    return Path(common)


def _is_under(path: str, root: Path) -> bool:
    """True if `path` is a descendant of `root` (case-insensitive, cross-format).

    Normalizes both sides to forward-slash form for comparison so UNC paths
    (`\\\\server\\share\\...`) and drive-letter paths (`Z:\\...`) compare
    correctly when they refer to the same tree. Case-insensitive to match
    Windows filesystem semantics.
    """
    try:
        root_str = str(root).replace("\\", "/").rstrip("/")
        path_norm = path.replace("\\", "/")
    except (AttributeError, TypeError):
        return False
    if not root_str:
        return False
    return path_norm.lower().startswith(root_str.lower() + "/")


def delete_orphans(
    client: Any,
    collection: str,
    orphans: list[QdrantPoint],
    batch_size: int = 1000,
    quiet: bool = False,
) -> int:
    deleted = 0
    total = len(orphans)
    for i in range(0, total, batch_size):
        ids = [p.id for p in orphans[i:i + batch_size]]
        if not ids:
            continue
        client.delete(collection_name=collection, points_selector=ids, wait=True)
        deleted += len(ids)
        if not quiet:
            print(f"  Deleted {deleted:,} / {total:,} orphan points", flush=True)
    return deleted


def render_report(report: HealReport, apply: bool = False, verbose: bool = False) -> str:
    outside = len(report.outside_scope)
    in_scope = report.total_points - outside
    lines = [
        f'Healing {report.source} against Qdrant collection "{report.collection}"',
        f"  {report.total_points:,} Qdrant points ({in_scope:,} in scope, {outside:,} outside scope), {report.total_files:,} files on disk",
    ]
    shown = report.orphans if verbose else report.orphans[:10]
    extra = len(report.orphans) - len(shown)
    suffix = ""
    if report.orphans and not verbose:
        suffix = f'   (first {len(shown)} shown, "...and {extra} more")' if extra else f"   (first {len(shown)} shown)"
    lines.append(f"Orphans (in Qdrant, not on disk): {len(report.orphans):,}{suffix}")
    for point in shown:
        lines.append(f"  {point.path}")
    lines.append(f"New files (on disk, not in Qdrant): {len(report.new_files):,}")
    lines.append(f"Modified files (mtime/size changed): {len(report.modified_files):,}")
    if report.outside_scope:
        lines.append(f"Outside scope (Qdrant points not under {report.source}): {outside:,}")
    if report.orphans:
        if apply:
            lines.append(f"Deleted {len(report.orphans):,} orphans.")
        else:
            lines.append(f"Run --apply to delete the {len(report.orphans):,} orphans.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source is not None:
        if not args.source.exists():
            print(f"source does not exist: {args.source}", file=sys.stderr)
            return 2
        if not args.source.is_dir():
            print(f"source is not a directory: {args.source}", file=sys.stderr)
            return 2
    print(f"Using Qdrant at {args.qdrant_url}, collection '{args.collection}'", flush=True)
    try:
        client = make_client(args)
        report = reconcile(client, args.source, args.collection, args.batch_size, quiet=args.quiet)
        for error in report.errors:
            print(error, file=sys.stderr)
        if args.apply and report.orphans:
            if not args.quiet:
                print(f"Deleting {len(report.orphans):,} orphan points...", flush=True)
            deleted = delete_orphans(client, args.collection, report.orphans, args.batch_size, quiet=args.quiet)
        else:
            deleted = 0
        print(render_report(report, apply=bool(args.apply and deleted), verbose=args.verbose))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Qdrant heal failed: {exc}", file=sys.stderr)
        return 1


def _scroll_points(client: Any, collection: str, batch_size: int, quiet: bool = False) -> list[QdrantPoint]:
    points: list[QdrantPoint] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload or {}
            points.append(QdrantPoint(
                id=str(point.id),
                path=str(payload.get("path") or ""),
                mtime=_optional_int(payload.get("mtime")),
                size=_optional_int(payload.get("size")),
            ))
        if not quiet:
            print(f"  Scrolled {len(points):,} points", flush=True)
        if offset is None:
            break
    return points


def _walk_files(source: Path, errors: list[str], quiet: bool = False) -> list[DiskFile]:
    files: list[DiskFile] = []
    for root, dirs, names in os.walk(source, followlinks=False):
        dirs[:] = [d for d in dirs if not should_skip_name(d)]
        for name in names:
            if should_skip_name(name) or not is_image_suffix(Path(name).suffix):
                continue
            path = Path(root) / name
            try:
                stat = path.stat()
            except OSError as exc:
                errors.append(f"could not stat {path}: {exc}")
                continue
            files.append(DiskFile(
                path=str(path.resolve()),
                mtime=int(stat.st_mtime),
                size=int(stat.st_size),
            ))
            if not quiet and len(files) % 1000 == 0:
                print(f"  Walked {len(files):,} files", flush=True)
    if not quiet:
        print(f"  Walked {len(files):,} files", flush=True)
    files.sort(key=lambda f: f.path)
    return files


def _canonical_path(path: str) -> str:
    """Resolve to canonical form; fall back to the input if resolve fails.

    On Windows, Path.resolve() returns the UNC equivalent of a drive-letter
    path when the drive is mapped, so this normalises Z:\\foo to \\server\\share\\foo
    when both refer to the same file. resolve() can raise for missing files
    (orphans); we fall back to the raw path in that case so the orphan check
    can still work — if a Qdrant point's file is missing, we want to detect
    that, not silently lose it.
    """
    if not path:
        return path
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        return path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
