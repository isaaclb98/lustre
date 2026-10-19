"""
search/__init__.py — package init.

Registers the mock-dim provider with the kernel. The kernel needs
to know the active prod variant's vector dimension so the mock
spec can mimic it — but the kernel must NOT import from `search`
(architecture test enforces this). The provider pattern lets the
app push the active dim into the kernel via a callback instead.

The actual `search.config.DEFAULT_MODEL` lookup happens at the
moment `get_active_mock_dim()` is called — not at import time —
so test environments that override `SIGLIP_VARIANT` BEFORE
calling `get_default_registry()` see the correct dim.
"""
from lustre_kernel.registry import register_mock_dim_provider


def get_active_mock_dim() -> int:
    """Return the dim of the currently-configured prod variant."""
    from lustre_kernel.registry import get as _registry_get
    from search.config import DEFAULT_MODEL
    return _registry_get(DEFAULT_MODEL).dim


register_mock_dim_provider(get_active_mock_dim)
