"""
Guard against a destructive startup path reaching a live service.

The 2026-10-19 and 2026-10-25 incidents both had the same shape: a
destructive default (auto-drop the Qdrant collection on a variant
mismatch) that was reachable from an ordinary code path, combined
with a test fixture whose default endpoint pointed at the production
sidecar. A routine `pytest` run then wiped a multi-gigabyte
production index in milliseconds.

Two rules are enforced here, because either one alone is not enough:

  1. No module on the search-side import path may CALL a
     destructive Qdrant collection operation. The indexer's
     `--rebuild` flag is the one legitimate exception — an explicit
     operator action, not a startup side effect — so it is
     allowlisted by path.

  2. The test fixture's default QDRANT_URL must point at a port
     that nothing listens on, so a test that forgets the in-memory
     fixture fails fast instead of reaching a live sidecar.

Both rules are checked against the SOURCE — a static AST scan and a
regex — not by running anything against a live service. The whole
point is that this check must never need a live service to be
meaningful.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Destructive Qdrant collection operations. Any of these reachable
# from a startup/import path is a data-loss hazard.
DESTRUCTIVE_CALLS = frozenset({
    "delete_collection",
    "delete_payload",
    "delete_points",
    "delete_vectors",
    "recreate_collection",
})

# Paths allowed to contain a destructive call. Each entry is
# (path, reason) so the allowlist itself is auditable.
ALLOWLIST = {
    "indexer/local_sync.py": (
        "the --rebuild flag is an explicit operator action that "
        "wipes and re-indexes the collection on purpose"
    ),
}

# Directories that hold vendored or generated code.
SKIP_DIRS = frozenset({
    ".venv", "node_modules", "build", ".svelte-kit", ".git",
    "dist", "__pycache__",
})


def _destructive_calls(path: Path) -> list[str]:
    """Return the destructive CALLS in `path`, by AST inspection.

    A text scan cannot distinguish a call from prose — the
    reconciler's own docstring describes the historical
    `delete_collection()` call, and a docstring is not a call.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in DESTRUCTIVE_CALLS:
            continue
        hits.append(
            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
            f"{func.attr}(...)"
        )
    return hits


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def test_no_destructive_qdrant_call_on_the_search_import_path() -> None:
    """The search side must never drop a collection on its own."""
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "search").rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWLIST:
            continue
        violations.extend(_destructive_calls(path))
    assert not violations, (
        "Destructive Qdrant call(s) reachable from the search import "
        "path. A startup/import path must never drop a collection — "
        "raise and let the operator decide, or gate it behind an "
        "explicit flag:\n  " + "\n  ".join(violations)
    )


def test_destructive_calls_are_confined_to_the_allowlist() -> None:
    """No destructive Qdrant call outside the audited allowlist."""
    violations: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if _is_skipped(path):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWLIST:
            continue
        violations.extend(_destructive_calls(path))
    assert not violations, (
        "Destructive Qdrant call(s) outside the allowlist:\n  "
        + "\n  ".join(violations)
    )


def test_allowlist_entries_still_exist_and_are_justified() -> None:
    """The allowlist is not a dumping ground."""
    for rel, reason in ALLOWLIST.items():
        path = REPO_ROOT / rel
        assert path.exists(), (
            f"allowlist entry {rel} no longer exists — remove it"
        )
        assert reason.strip(), f"allowlist entry {rel} has no reason"


def test_test_fixture_default_qdrant_url_points_at_a_dead_port() -> None:
    """The fixture default must not reach a live Qdrant."""
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    match = re.search(
        r'setdefault\(\s*"QDRANT_URL"\s*,\s*"([^"]+)"', conftest
    )
    assert match is not None, (
        "tests/conftest.py no longer sets a QDRANT_URL default — a "
        "missing default means the ambient environment wins, which is "
        "how the prod sidecar got reached in the first place"
    )
    url = match.group(1)
    port = url.rsplit(":", 1)[-1]
    assert port != "6333", (
        f"tests/conftest.py defaults QDRANT_URL to {url!r} — that is "
        "the production Qdrant port. A test that forgets the "
        "in-memory fixture would then reach the live sidecar."
    )
    assert port.startswith("16"), (
        f"tests/conftest.py defaults QDRANT_URL to {url!r}; the dead "
        "port should be in the 16xxx range, which is reserved for "
        "test-only services"
    )


def test_reconciler_raises_instead_of_deleting() -> None:
    """The startup reconciler must refuse, not destroy."""
    from search import config

    assert hasattr(config, "QdrantVariantMismatch"), (
        "search.config must expose QdrantVariantMismatch — the "
        "reconciler is supposed to raise on a variant mismatch"
    )
    hits = _destructive_calls(Path(config.__file__))
    assert not hits, (
        "search/config.py still calls a destructive Qdrant operation "
        "— the reconciler must raise QdrantVariantMismatch and "
        f"preserve the collection: {hits}"
    )


def test_reconciler_is_reachable_from_config_load() -> None:
    """The reconciler runs inside config.load() — which is why a
    plain `config.load()` in a test was enough to wipe the prod
    collection. This test pins the reachability so nobody removes
    the call without noticing that the guard above depends on it."""
    from search import config

    source = Path(config.__file__).read_text(encoding="utf-8")
    assert "reconcile_variant_from_qdrant(" in source
    # It must be called from load(), not merely defined.
    tree = ast.parse(source)
    load_fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "load"),
        None,
    )
    assert load_fn is not None, "search/config.py has no load()"
    called = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "reconcile_variant_from_qdrant"
        for n in ast.walk(load_fn)
    )
    assert called, (
        "config.load() no longer calls reconcile_variant_from_qdrant() "
        "— if the reconciler moved out of load(), the variant check "
        "is no longer enforced at startup"
    )


@pytest.mark.parametrize("variant,expected_dim", [
    ("B/16-256", 768),
    ("L/16-256", 1024),
    ("gopt/16-384", 1536),
    ("so400m/16-384", 1152),
])
def test_variant_dims_are_unique_so_mismatches_are_detectable(
    variant: str, expected_dim: int,
) -> None:
    """Each variant must map to its own dim — the dim is what the
    reconciler compares, so two variants sharing a dim would make a
    mismatch undetectable."""
    from search import config

    assert config.SIGLIP_VARIANTS[variant][1] == expected_dim
    dims = [spec[1] for spec in config.SIGLIP_VARIANTS.values()]
    assert len(dims) == len(set(dims)), (
        f"variant dims are not unique: {dims} — the reconciler "
        "compares dims, so duplicates hide a real mismatch"
    )
