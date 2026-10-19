"""
lustre_kernel.qdrant_url

URL → QdrantClient kwargs helper. Moved verbatim from
`search/qdrant_url.py` so that both `search/` and `indexer/` can build
QdrantClient instances without depending on each other.

qdrant-client 1.18 has a footgun: when given a URL like
`https://host` with no explicit port, it falls back to its
hard-coded default of 6333 regardless of scheme. That breaks
deployments behind an HTTPS reverse proxy on :443 (which is
exactly the case for `https://qdrant.aizaku.ca` — Caddy listens
on 443, not 6333). curl works because the user-supplied URL
has no port and curl honors the scheme default; the Python
client doesn't.

This module fixes it by parsing the URL and passing `port=`
explicitly when the URL omits it. Behavior:
  https://host        -> port=443
  https://host:8443   -> port=8443 (URL wins)
  http://host         -> port=6333
  http://host:6333    -> port=6333

gRPC support (added for diversity-fetch latency):
  prefer_grpc=True switches the transport from REST/JSON to the
  Qdrant gRPC API (~3x faster on bulk vector fetches at depth
  5000 because the wire format is protobuf instead of JSON and
  there is no per-message parse cost). When `prefer_grpc=True`,
  `grpc_port` is the port qdrant-client connects to for gRPC
  traffic; defaults to REST port + 1 (6334) when not specified,
  which is the qdrant docker image's default gRPC listener.

  Inside the lustre docker network the search container
  reaches qdrant at `qdrant:6334` once the compose file exposes
  port 6334 from the qdrant service. Both the indexer and the
  search app call this helper, so toggling gRPC here flips the
  transport for every QdrantClient in the project.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

__all__ = ["client_kwargs"]


def client_kwargs(url: str, api_key: str | None = None, timeout: float | None = None,
                  *, prefer_grpc: bool = False, grpc_port: int | None = None,
                  ) -> dict[str, Any]:
    """
    Build kwargs for `QdrantClient(...)`.

    Args:
        url: Full URL including scheme, e.g. `https://qdrant.aizaku.ca`.
        api_key: Optional API key. None for unauthenticated Qdrant.
        timeout: Optional timeout in seconds. None for qdrant-client default.
        prefer_grpc: When True, use qdrant-client's gRPC transport instead
            of REST. Falls back to REST on connection failure (the client
            library handles this internally). Default False.
        grpc_port: gRPC port. Defaults to REST port + 1 (6333 -> 6334),
            which matches the qdrant docker image's default. Ignored
            unless prefer_grpc=True.

    Returns:
        A dict suitable for `QdrantClient(**kwargs)`.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"QDRANT_URL must be http:// or https://, got {url!r}"
        )
    if not parsed.hostname:
        raise ValueError(f"QDRANT_URL has no host: {url!r}")

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 6333

    kwargs: dict[str, Any] = {"url": url, "port": port}
    if api_key:
        kwargs["api_key"] = api_key
    if timeout is not None:
        kwargs["timeout"] = timeout
    if prefer_grpc:
        kwargs["prefer_grpc"] = True
        # Default to REST port + 1 if the caller didn't override.
        kwargs["grpc_port"] = grpc_port if grpc_port is not None else port + 1
    return kwargs
