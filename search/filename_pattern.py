"""Filename-pattern parsing — the single source of truth.

`?filename=` accepts one of two shapes:

* **exact** — a single token that must match a whole token in the
  photo's path (`chaewon` matches `/photos/kpop/chaewon/2024.jpg`).
* **prefix** — the same, but the token may be a prefix of a path
  token (`chaewon*` also matches `/photos/kpop/chaewonder/x.jpg`).

Anything else is a 400. The rules live here, once, because two
backends consume them:

* Qdrant's native text index (`MatchText` / `MatchPhrase`) — the
  primary implementation.
* SQLite FTS5 (`images_fts`) — kept as a fallback and as the oracle
  the two are diffed against.

Both must agree on what a valid pattern *is*; if they disagreed, the
same URL would mean two different things depending on which backend
answered it.

The rules are deliberately strict. FTS5's query language supports
operators (`AND`, `OR`, `NEAR`, `^`, `-`, quotes) and globs, and the
native text index has no equivalent for most of them. Rather than
silently changing the meaning of a pattern depending on the backend,
we reject anything that isn't a single plain token (optionally with
one trailing `*`) and answer 400.

The accepted charset mirrors what FTS5's bareword grammar actually
accepts, measured against the live index: letters, digits, `_`, and
non-ASCII letters. Everything else is either an FTS5 operator
(meaning the pattern would be parsed as syntax) or a character FTS5
rejects outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A path token, as FTS5's `unicode61` tokenizer would split it:
# runs of letters/digits, with everything else acting as a separator.
# `unicode61` also folds case and (with `remove_diacritics 2`) strips
# diacritics, so `Ángela` and `angela` are the same token.
#
# `_` is included because unicode61 treats it as a separator in the
# *indexed* text but FTS5 accepts it inside a query bareword (it
# tokenises the query the same way it tokenises the document).
_TOKEN_RE = re.compile(r"^[0-9A-Za-zÀ-ɏ_]+$")

# Characters that are meaningful to FTS5's query parser. A pattern
# containing one of them would be interpreted as syntax rather than
# as a literal, so we reject it instead of passing it through.
_FTS5_OPERATORS = frozenset("\"':()+-^,")

# `*` is only allowed as a single trailing character (prefix match).
# A leading or embedded `*` is a glob/suffix request, which neither
# backend supports.
_GLOB_RE = re.compile(r"[*?\[\]]")


@dataclass(frozen=True)
class FilenamePattern:
    """A parsed, validated `?filename=` pattern.

    `body` is the token to match; `prefix` says whether it may match
    a prefix of a path token (the `body*` form) or must match a
    whole token (the plain form).
    """

    body: str
    prefix: bool

    @property
    def fts5_query(self) -> str:
        """The FTS5 `MATCH` expression for this pattern."""
        return f"{self.body}*" if self.prefix else self.body

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.body}*" if self.prefix else self.body


def parse_filename_pattern(pattern: str) -> FilenamePattern | None:
    """Parse and validate a raw `?filename=` pattern.

    Returns `None` when the pattern carries no filter at all (empty,
    whitespace-only, or only a `*`), so the caller can skip the
    filter entirely rather than issuing a match that matches
    everything.

    Raises `ValueError` when the pattern is syntactically invalid.
    The message is user-facing — the API surfaces it as a 400 — so it
    should say what is wrong in terms the user typed, not in terms
    of the backend that rejected it.
    """
    if not pattern or not pattern.strip():
        return None

    # Strip surrounding whitespace and a single layer of quotes so
    # casual typing ("chaewon", '"chaewon"') doesn't trip the
    # validator.
    candidate = pattern.strip().strip('"').strip("'").strip()
    if not candidate:
        return None

    # Glob characters anywhere but a single trailing `*` are not
    # supported by either backend. Reject them loudly: silently
    # returning zero matches would look like a broken filter.
    if _GLOB_RE.search(candidate.removesuffix("*")):
        raise ValueError(
            f"filename pattern only supports a single trailing '*' for "
            f"prefix match (got {pattern!r})"
        )

    prefix = candidate.endswith("*")
    body = candidate[:-1] if prefix else candidate

    # A bare `*` (or `*` plus whitespace) carries no filter.
    if not body:
        return None

    if any(ch.isspace() for ch in body):
        raise ValueError(
            f"filename pattern must be a single token (got {pattern!r})"
        )

    if any(ch in _FTS5_OPERATORS for ch in body):
        raise ValueError(
            f"filename pattern must not contain FTS5 operators (got {pattern!r})"
        )

    if not _TOKEN_RE.match(body):
        raise ValueError(
            f"filename pattern must be a single token of letters or "
            f"digits (got {pattern!r})"
        )

    return FilenamePattern(body=body, prefix=prefix)
