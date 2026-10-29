#!/usr/bin/env python3
"""
Manage Lustre's photo sources within a single Qdrant collection.

A "source" is one logical photo library — a directory of photos that
the indexer walks and tags with a `collection` payload value. The
search UI exposes them as filter chips; `GET /api/collections` lists
them with point counts.

This tool adds and removes sources:

    source_admin.py list
    source_admin.py add kpopping /nas/kpopping
    source_admin.py remove kpopping --dry-run
    source_admin.py remove kpopping --apply

Adding a source:

  1. the directory must exist and be inside the mounted photo root
  2. it is indexed under its own `collection` payload value
  3. it becomes a filter chip in the UI automatically

Removing a source:

  1. its points are deleted from the collection
  2. the remaining sources are untouched — the delete is scoped by a
     `collection` payload filter, and the guard refuses any filter
     that could match everything

Both operations are safe to repeat. The indexer is incremental, so
adding an already-indexed directory re-embeds only what changed.

Every destructive step requires `--apply`. Without it the tool
reports what WOULD happen and changes nothing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from lustre_kernel.destructive_guard import (
    DestructiveFilterError,
    delete_source_points,
)
from lustre_kernel.qdrant_url import client_kwargs


def make_client(url: str, api_key: str | None) -> QdrantClient:
    return QdrantClient(**client_kwargs(url=url, api_key=api_key, timeout=60))


def _model_for_dim(dim: int) -> str | None:
    """Return the registered model that produces vectors of `dim`.

    The collection's stored dim decides the model — not a default.
    Indexing with a mismatched model writes vectors the collection
    cannot hold.
    """
    from lustre_kernel.registry import get_default_registry

    registry = get_default_registry()
    for name in (
        "ViT-so400m-patch16-384",
        "ViT-gopt-16-SigLIP2-384",
        "ViT-L-16-SigLIP2-256",
        "ViT-B-16-SigLIP2-256",
    ):
        try:
            if registry.get(name).dim == dim:
                return name
        except Exception:  # noqa: BLE001 — a model not registered in this env
            continue
    return None


def _live_sources(client: QdrantClient, collection: str) -> dict[str, int]:
    """Return the sources that actually hold points, with their counts.

    The facet API returns a zero-count entry for a source whose points
    were all deleted. Such a source is gone — treating it as present
    would make `add` refuse to re-add it and `remove` report it as
    still there. Only sources with points count as existing.
    """
    hits = client.facet(
        collection_name=collection,
        key="collection",
        limit=100,
        exact=True,
    ).hits
    return {h.value: h.count for h in hits if h.count > 0}


def list_sources(client: QdrantClient, collection: str) -> int:
    """Print every source in the collection with its point count."""
    # The facet API aggregates the payload keyword index server-side:
    # one round trip, O(distinct values), no collection scan.
    counts = _live_sources(client, collection)
    total = 0
    print(f"{'SOURCE':<24} {'POINTS':>12}")
    print("-" * 38)
    for name in sorted(counts):
        print(f"{name:<24} {counts[name]:>12,}")
        total += counts[name]
    print("-" * 38)
    print(f"{'TOTAL':<24} {total:>12,}")
    return 0


def add_source(
    client: QdrantClient,
    collection: str,
    name: str,
    directory: Path,
    photos_root: Path,
    dry_run: bool,
    url: str,
    api_key: str | None,
) -> int:
    """Register and index a new source directory.

    The indexer writes the `collection` payload value from the
    directory's basename by default, so the source name and the
    directory name are kept in step: the name is derived from the
    directory unless it is given explicitly.
    """
    resolved = directory.resolve()
    if not resolved.is_dir():
        print(f"error: {resolved} is not a directory", file=sys.stderr)
        return 2

    # The indexer stores paths as the container sees them, and the
    # search app resolves them under PHOTOS_DIR. A source outside the
    # mounted photo root would index fine and then fail to resolve at
    # serve time, so refuse it here.
    root = photos_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        print(
            f"error: {resolved} is outside the photo root {root}. "
            "The indexer stores paths under the mount; a source "
            "outside it would index but not resolve at serve time.",
            file=sys.stderr,
        )
        return 2

    existing = _live_sources(client, collection)
    if name in existing:
        print(f"error: source {name!r} already exists", file=sys.stderr)
        return 2

    # Derive the source name from the directory basename — that is what
    # the indexer does when no --source-name is passed.
    derived = resolved.name
    if name != derived:
        print(
            f"note: source name {name!r} differs from the directory "
            f"basename {derived!r}. The indexer uses the basename "
            "unless --source-name is passed; pass the same name to the "
            "indexer or the chip in the UI will not match.",
            file=sys.stderr,
        )

    if dry_run:
        print(f"would index {resolved} as source {name!r} into {collection!r}")
        return 0

    # The model MUST match the collection's vector dim, or the
    # indexer writes vectors the collection cannot hold. Read the dim
    # off the live collection and derive the model from it, rather than
    # trusting a default.
    dim = client.get_collection(collection_name=collection).config.params.vectors.size
    model = _model_for_dim(dim)
    if model is None:
        print(
            f"error: collection {collection!r} has vector dim {dim}, which "
            "no registered model produces. Refusing to index.",
            file=sys.stderr,
        )
        return 2

    print(f"indexing {resolved} as source {name!r} into {collection!r}")
    print(f"collection dim {dim} -> model {model}")

    # The indexer is invoked as a module so it runs with the same
    # interpreter and the same installed deps as the search app.
    argv = [
        sys.executable, "-m", "indexer.local_sync",
        "--json-progress",
        "--source", str(resolved),
        "--source-name", name,
        "--model", model,
        "--device", os.environ.get("INDEXER_DEVICE", "cuda"),
        "--qdrant-url", url,
        "--qdrant-collection", collection,
    ]
    if api_key:
        argv += ["--qdrant-api-key", api_key]
    print(f"running: {' '.join(argv)}")
    rc = subprocess.call(argv)
    if rc != 0:
        print(f"error: indexer exited with code {rc}", file=sys.stderr)
        return rc
    return 0


def remove_source(
    client: QdrantClient,
    collection: str,
    name: str,
    apply: bool,
) -> int:
    """Delete one source's points, leaving every other source intact."""
    existing = _live_sources(client, collection)
    if name not in existing:
        print(f"error: no source named {name!r} in {collection!r}", file=sys.stderr)
        print(f"known sources: {', '.join(sorted(existing)) or '(none)'}", file=sys.stderr)
        return 2

    try:
        result = delete_source_points(
            client, collection, name, dry_run=not apply,
        )
    except DestructiveFilterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if apply:
        print(
            f"deleted {result['deleted']:,} point(s) for source {name!r} "
            f"from {collection!r}"
        )
        remaining = client.count(collection_name=collection, exact=True).count
        print(f"{remaining:,} point(s) remain in the collection")
    else:
        print(
            f"would delete {result['matched']:,} point(s) for source "
            f"{name!r} from {collection!r}. Re-run with --apply to do it."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="source_admin",
        description="Add and remove photo sources in a Lustre collection.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        help="Qdrant endpoint (default: $QDRANT_URL or http://127.0.0.1:6333)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QDRANT_API_KEY"),
        help="Qdrant API key (default: $QDRANT_API_KEY)",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "images"),
        help="collection name (default: $QDRANT_COLLECTION or images)",
    )
    parser.add_argument(
        "--photos-root",
        default=os.environ.get("PHOTOS_DIR", "/nas"),
        help="the mounted photo root; sources must live inside it "
             "(default: $PHOTOS_DIR or /nas)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list sources with point counts")

    p_add = sub.add_parser("add", help="index a new source directory")
    p_add.add_argument("name", help="source name (the `collection` payload value)")
    p_add.add_argument("directory", type=Path, help="directory to index")
    p_add.add_argument(
        "--dry-run", action="store_true",
        help="report what would happen without indexing",
    )

    p_rm = sub.add_parser("remove", help="delete a source's points")
    p_rm.add_argument("name", help="source name to remove")
    p_rm.add_argument(
        "--apply", action="store_true",
        help="actually delete (without it, only reports the count)",
    )

    args = parser.parse_args(argv)
    client = make_client(args.url, args.api_key)

    if args.command == "list":
        return list_sources(client, args.collection)
    if args.command == "add":
        return add_source(
            client, args.collection, args.name, args.directory,
            Path(args.photos_root), args.dry_run, args.url, args.api_key,
        )
    if args.command == "remove":
        return remove_source(client, args.collection, args.name, args.apply)
    return 2


if __name__ == "__main__":
    sys.exit(main())
