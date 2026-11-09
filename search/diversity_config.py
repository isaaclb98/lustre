"""
search/diversity_config.py — single source of truth for diversity.

Every page that surfaces the Diversity select (search) routes its
query params through `resolve_diversity()` so the validation,
defaults, and env‑based fallback live in exactly one place.

Knobs:
- `mode`:   off | low | balanced | high
- `depth`:  auto | 500 | 1000 | 2000 | 5000

Defaults come from the environment:
- `DIVERSITY_MODE`    (default: "balanced")
- `DIVERSITY_DEPTH`   (default: "auto")
"""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_MODES = ("off", "low", "balanced", "high")
VALID_DEPTHS = ("auto", "500", "1000", "2000", "5000")


@dataclass(frozen=True)
class Diversity:
    mode: str = "balanced"
    depth: str = "auto"

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"invalid diversity mode {self.mode!r}; expected one of {VALID_MODES}"
            )
        if self.depth not in VALID_DEPTHS:
            raise ValueError(
                f"invalid diversity depth {self.depth!r}; expected one of {VALID_DEPTHS}"
            )

    def with_overrides(
        self,
        mode: str | None = None,
        depth: str | None = None,
    ) -> Diversity:
        """Return a new Diversity with any non-None overrides applied."""
        return Diversity(
            mode=mode if mode is not None else self.mode,
            depth=depth if depth is not None else self.depth,
        )


def load_diversity_from_env() -> Diversity:
    """Build the default Diversity from the environment."""
    return Diversity(
        mode=os.environ.get("DIVERSITY_MODE", "balanced"),
        depth=os.environ.get("DIVERSITY_DEPTH", "auto"),
    )


def resolve_diversity(
    cfg_default: Diversity,
    *,
    mode: str | None = None,
    depth: str | None = None,
) -> Diversity:
    """Resolve a Diversity from query params + the app‑wide default.

    The depth knob is always taken from the app‑wide default: the
    only caller that used to override it was the For You feed, which
    has since been removed.
    """
    resolved_mode = mode if mode in VALID_MODES else cfg_default.mode
    resolved_depth = cfg_default.depth
    return Diversity(mode=resolved_mode, depth=resolved_depth)