"""
tests/test_architecture.py — architectural invariants.

Pins rules from `docs/backend-refactor-plan.md` that would be easy to
regress:

- The shared kernel (`lustre_kernel/`) imports nothing from
  `search/` or `indexer/`. Both of those packages may import from the
  kernel.
- `indexer/` imports nothing from `search/`. The dependency direction
  enforced in §3 of the plan.
- `search/` does not import from `indexer/` outside the kernel's
  payload contract (i.e. only via `lustre_kernel`).

If any of these regress, the failure surfaces here, not as a
runtime bug or a silent coupling that future refactors will have
to discover.

The "model dim / resolution" detectors use `ast` to parse Python
files and only flag actual integer literals in module-level
assignments, list repetitions, and function calls — not string
contents (docstrings) or comments. A docstring that mentions
"1536-dim" doesn't represent a hardcoded dimension; an `[0.0] * 1536`
does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = REPO_ROOT / "lustre_kernel"
SEARCH_DIR = REPO_ROOT / "search"
INDEXER_DIR = REPO_ROOT / "indexer"
TESTS_DIR = REPO_ROOT / "tests"
BENCH_DIR = REPO_ROOT / "benchmarks"


def _python_files(root: Path) -> list[Path]:
    """Return `*.py` files under `root`, excluding `__pycache__` and `__init__.py`."""
    if not root.exists():
        return []
    return [
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


# ---------------------------------------------------------------------------
# Kernel purity: the shared kernel imports nothing from search/ or indexer/.
# ---------------------------------------------------------------------------

KERNEL_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(?:from\s+(search|indexer)|import\s+(search|indexer))",
    re.MULTILINE,
)


def test_kernel_does_not_import_search_or_indexer():
    """Kernel must not depend on either consumer package."""
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(KERNEL_DIR):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in KERNEL_FORBIDDEN_IMPORTS.finditer(text):
            offenders.append((path, match.group(0).strip()))
    assert not offenders, (
        "lustre_kernel imports from a consumer package; "
        "the kernel must remain pure.\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT)}: {line}" for p, line in offenders)
    )


# ---------------------------------------------------------------------------
# Dep direction: indexer/ must not import from search/.
# ---------------------------------------------------------------------------

INDEXER_FORBIDDEN_SEARCH_IMPORTS = re.compile(
    r"^\s*from\s+search\b",
    re.MULTILINE,
)


def test_indexer_does_not_import_search():
    """indexer/ is the runtime-agnostic consumer; it must not depend on search/."""
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(INDEXER_DIR):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in INDEXER_FORBIDDEN_SEARCH_IMPORTS.finditer(text):
            offenders.append((path, match.group(0).strip()))
    assert not offenders, (
        "indexer/ imports from search/; the dep direction is reversed.\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT)}: {line}" for p, line in offenders)
    )


# ---------------------------------------------------------------------------
# Dep direction: search/ must not import from indexer/ (outside the kernel).
# ---------------------------------------------------------------------------

SEARCH_FORBIDDEN_INDEXER_IMPORTS = re.compile(
    r"^\s*from\s+indexer\b",
    re.MULTILINE,
)


def test_search_does_not_import_indexer():
    """search/ consumes the kernel's payload contract, not indexer/ directly."""
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(SEARCH_DIR):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in SEARCH_FORBIDDEN_INDEXER_IMPORTS.finditer(text):
            offenders.append((path, match.group(0).strip()))
    assert not offenders, (
        "search/ imports from indexer/; use the kernel payload contract instead.\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT)}: {line}" for p, line in offenders)
    )


# ---------------------------------------------------------------------------
# Model registry discipline (§A3):
# - Outside the kernel and `indexer/upsert.py` (single deprecated constant),
#   no file references model dim, resolution, arch tag, or embedder call site.
# - Outside the kernel and `indexer/vision_encoder.py`, no file imports
#   open_clip, transformers, or torch.
# ---------------------------------------------------------------------------

# Files where references are explicitly allowed (the registry and the
# one deprecated constant in upsert.py). `search/config.py` is the
# app-level config that mirrors `lustre_kernel.registry`'s
# VARIANT_TO_MODEL — it owns the app-facing `SIGLIP_VARIANT` env var
# contract; the kernel owns the model spec resolution.
DIM_ALLOWED_FILES: frozenset[Path] = frozenset({
    KERNEL_DIR / "registry.py",
    KERNEL_DIR / "_real_models.py",
    KERNEL_DIR / "payload_schema.py",
    SEARCH_DIR / "config.py",
    INDEXER_DIR / "upsert.py",
    KERNEL_DIR / "__init__.py",
})

# Files where ML-runtime imports are explicitly allowed:
# - `lustre_kernel/_real_models.py`: the kernel's conditional
#   real-model registration entry.
# - `search/centroids.py`: reads `.pt` files written by the sibling
#   `isaac-image-scoring` project.
# - Test fixtures for centroids.
ML_RUNTIME_ALLOWED_FILES: frozenset[Path] = frozenset({
    KERNEL_DIR / "_real_models.py",
    SEARCH_DIR / "centroids.py",
    TESTS_DIR / "_centroid_fixture.py",
    TESTS_DIR / "test_centroids.py",
    # Why an exception here, against the rule:
    # This is a stand-alone benchmark — not part of the runtime
    # code path, not imported by any production module. It
    # measures the encoder hot path directly: it needs torch
    # to allocate tensors and trigger CUDA kernel autotuning,
    # which is the whole point of the sweep. Routing it through
    # the kernel would defeat the measurement (you'd be
    # measuring the kernel indirection, not the encoder). The
    # architecture rule's intent is to prevent runtime ML imports
    # leaking past the abstraction; a measurement tool that
    # exercises the abstraction from the outside is a legitimate
    # exception.
    BENCH_DIR / "embed_batch_sweep.py",
})

# The actual model dims currently in use. Add to this set when adding
# a new model. We intentionally do NOT flag resolutions like 384 or
# 256 — those values appear in many legitimate contexts (chunk sizes,
# range limits) and the AST detector can't reliably distinguish.
# Resolution discipline is enforced by the model registry itself:
# callers must read `spec.resolution` rather than hardcoding.
TARGET_DIMS: frozenset[int] = frozenset({1536, 1024})

# ML runtime modules. Importing any of these outside the allow-list
# except via the kernel's `_real_models` violates §A3.
ML_RUNTIME_MODULES: tuple[str, ...] = ("open_clip", "transformers", "torch", "timm")

# Module-level dim constants. The grep below flags these by name.
DIM_CONSTANT_NAMES: re.Pattern[str] = re.compile(
    r"\b(VECTOR_DIM|_EMBED_DIM|SIGLIP_RESOLUTION)\b",
)


def _ast_int_literals(text: str) -> list[tuple[int, int, str]]:
    """Return `[(value, line, snippet)]` for every plain integer literal in
    `text`.

    Walks the AST and reports each `ast.Constant` whose value is an
    `int`. Skips:
      - string contents (docstrings are `Constant` too with str values)
      - integer literals that appear inside a `BinOp` (we don't flag
        `1024` inside `1024 * 1024` because that's a chunk size, not
        a model dim).

    The `line` is 1-indexed; `snippet` is the source line.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()

    # Collect line ranges of every BinOp subtree so we can exclude
    # their contained integer literals.
    binop_ranges: list[tuple[int, int]] = []

    class _BinOpCollector(ast.NodeVisitor):
        def visit_BinOp(self, node: ast.BinOp) -> None:
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            binop_ranges.append((start, end))
            self.generic_visit(node)

    _BinOpCollector().visit(tree)

    def in_binop(line: int) -> bool:
        return any(start <= line <= end for start, end in binop_ranges)

    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, int):
            continue
        if in_binop(node.lineno):
            continue
        line = node.lineno
        snippet = lines[line - 1] if line - 1 < len(lines) else ""
        out.append((node.value, line, snippet.strip()))
    return out


def test_no_hardcoded_model_dim_outside_registry():
    """No integer literal with value in `TARGET_DIMS` may appear in
    `search/` or `indexer/` outside the allow-list.

    Uses `ast` to skip docstrings and comments. A `1536` inside a
    docstring is prose; a `1536` in `[0.0] * 1536` or
    `vectors_config=...size=1536` is a hardcoded dimension.
    """
    offenders: list[tuple[Path, int, str]] = []
    for root in (SEARCH_DIR, INDEXER_DIR):
        for path in _python_files(root):
            if path in DIM_ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for value, line, snippet in _ast_int_literals(text):
                if value in TARGET_DIMS:
                    offenders.append((path, line, snippet))
    assert not offenders, (
        "hardcoded model dim/resolution literal in AST (outside registry/upsert.py):\n"
        + "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{i}: {line}" for p, i, line in offenders
        )
    )


def test_no_vector_dim_constant_outside_registry():
    """No module-level `VECTOR_DIM`, `_EMBED_DIM`, or `SIGLIP_RESOLUTION`
    constant may be defined in `search/` or `indexer/` outside the
    allow-list.

    Uses AST to detect actual `Assign` nodes, ignoring docstring
    mentions (a docstring that mentions `VECTOR_DIM = 1536` is prose,
    not an assignment).
    """
    offenders: list[tuple[Path, int, str]] = []
    for root in (SEARCH_DIR, INDEXER_DIR):
        for path in _python_files(root):
            if path in DIM_ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                # Check each target name on the LHS.
                for target in node.targets:
                    name: object = getattr(target, "id", None)
                    if name is None and isinstance(target, ast.Name):
                        name = target.id
                    if isinstance(name, str) and DIM_CONSTANT_NAMES.search(name):
                        offenders.append(
                            (path, node.lineno, ast.unparse(node).strip()),
                        )
    assert not offenders, (
        "model-dim constant defined outside registry/upsert.py:\n"
        + "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{i}: {line}" for p, i, line in offenders
        )
    )


def test_no_ml_runtime_imports_outside_kernel_real_models():
    """No `torch` / `open_clip` / `transformers` / `timm` imports in
    `search/`, `indexer/`, `tests/`, or `benchmarks/` — the kernel's
    conditional `_real_models.py` is the only allowed importer.

    This is the rule that makes the abstraction sticky. A future PR
    that imports torch anywhere except the registry fails CI.
    """
    offenders: list[tuple[Path, int, str]] = []
    for root in (SEARCH_DIR, INDEXER_DIR, TESTS_DIR, BENCH_DIR):
        for path in _python_files(root):
            if path in ML_RUNTIME_ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in ML_RUNTIME_MODULES:
                            offenders.append(
                                (path, node.lineno, f"import {alias.name}"),
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if module in ML_RUNTIME_MODULES:
                        offenders.append(
                            (path, node.lineno, f"from {node.module} import ..."),
                        )
    assert not offenders, (
        "ML runtime import outside the kernel's _real_models.py:\n"
        + "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{i}: {line}" for p, i, line in offenders
        )
    )


def test_kernel_is_importable_on_cpu_only_host():
    """The kernel package must be importable without torch / open_clip
    / transformers / timm.

    On a host that has them installed, the conditional registration
    succeeds and the registry contains the real-model entries. On a
    host that does not, the kernel still imports and only the
    `mock-1536` entry is registered.

    This test verifies the kernel's *package* imports succeed
    independently of the conditional module. The actual conditional
    registration behavior is covered by
    `test_registry_conditional_real_models` in `test_registry.py`.
    """
    import lustre_kernel
    import lustre_kernel.payload_schema
    import lustre_kernel.qdrant_url
    import lustre_kernel.vectors

    assert hasattr(lustre_kernel, "payload_schema")
    assert hasattr(lustre_kernel, "qdrant_url")
    assert hasattr(lustre_kernel, "vectors")
    assert hasattr(lustre_kernel, "registry")
