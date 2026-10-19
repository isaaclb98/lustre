"""
lustre_kernel — the shared package both `search/` and `indexer/`
import from.

Hard rule: this package has no dependency on `search/` or `indexer/`.
It contains:

- `qdrant_url` — URL → QdrantClient kwargs helper (moved from
  `search/qdrant_url.py`).
- `payload_schema` — canonical Qdrant point payload field names,
  types, and version constants (moved from `indexer/schema.py`).
- `registry` — ModelSpec, Embedder Protocol, and the model registry
  (the only place model name, dim, resolution, revision, and embedder
  call sites are referenced).
- `vectors` — L2 normalize, mean vector, cosine similarity primitives.

Public surface: every symbol exported via `__all__`. Tests import only
from `__all__` symbols; a regression test enforces this.

No I/O. No HTTP. No FastAPI. No QdrantClient construction. No
filesystem reads. The kernel is pure utilities, registries, and types.
"""

from lustre_kernel import payload_schema, qdrant_url, registry, vectors

__all__ = [
    "payload_schema",
    "qdrant_url",
    "registry",
    "vectors",
]
