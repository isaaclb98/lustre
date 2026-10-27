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
* SQLite FTS5 (`images_fts`) — the pre-migration oracle; the parity
  harness in scripts/parity_harness.py compared against it
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

# The path tokens inside a pattern body. FTS5's `unicode61` treats
# `_` as a separator in the indexed text, so `img_2024` is the two
# tokens `img` and `2024` — and a query for it is a PHRASE of those
# two tokens, so they must be adjacent in the path.
#
# This matters for the prefix form. FTS5's `img_2024*` means "the
# token `img`, immediately followed by a token starting with
# `2024`". The native prefix index alone cannot express that (it
# ANDs the prefixes with no adjacency), so the leading tokens are
# matched exactly-and-adjacently on the word index and only the
# trailing one is matched as a prefix.
_BODY_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏ]+")


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

    @property
    def tokens(self) -> list[str]:
        """The path tokens inside `body`.

        `_` separates tokens in FTS5's tokenizer, so `img_2024` is
        two tokens. A body with one token is a single-term pattern;
        a body with several is a phrase.
        """
        return _BODY_TOKEN_RE.findall(self.body)

    @property
    def phrase_body(self) -> str:
        """The body as a phrase — its tokens separated by spaces.

        Qdrant's `MatchPhrase` takes the phrase as a single string
        and tokenises it with the index's tokenizer. The `word`
        tokenizer splits on `_` as well as on whitespace, but the
        `prefix` tokenizer does NOT — it treats `img_2024` as one
        token. So the tokens have to be separated explicitly for the
        prefix index to see them as a phrase.

        `twice_sana` → `twice sana`, which the prefix tokenizer
        reads as a two-token phrase: the token `twice` immediately
        followed by a token starting with `sana`. That is exactly
        FTS5's `twice_sana*`.
        """
        return " ".join(self.tokens)

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
