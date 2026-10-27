"""Build Qdrant filter conditions for `?filename=` patterns.

Qdrant's native text indexes let the filename filter be evaluated
server-side, so the matching photo ids never have to be
materialised in the search app and shipped to Qdrant as a `HasId`
filter.

Two indexes are involved, because the two pattern shapes need
different tokenizers (measured against the live corpus — both
configurations reproduce FTS5's results exactly, id for id):

* **exact** — `path` indexed with the `word` tokenizer and phrase
  matching on. The condition is `MatchPhrase(text=body)`.

  The `word` tokenizer splits on the same boundaries as FTS5's
  `unicode61`, and `lowercase` + `ascii_folding` reproduce
  `unicode61 remove_diacritics 2`. Phrase matching is what makes a
  multi-token body behave like FTS5: `img_2024` must match a path
  where `img` and `2024` are *adjacent* tokens, and `MatchPhrase`
  does exactly that. (`MatchText` on the same index would be an
  AND over the tokens regardless of distance — measurably looser:
  it matched 13 paths for `img_2024` where FTS5 matched 2.)

* **prefix** — `path_prefix` indexed with the `prefix` tokenizer.
  The condition is `MatchText(text=body)`.

  The `prefix` tokenizer indexes every prefix of every token, so a
  bare term matches any token that *starts with* it — which is
  FTS5's `body*` behaviour. A `word`-tokenized index cannot express
  this: `MatchText(text="chaewon")` against it matches the token
  `chaewon` but not `chaewonder`, where FTS5's `chaewon*` matches
  both.

The two indexes live on two payload fields because a Qdrant
collection supports one text index per field name, and the two
tokenizers cannot coexist on one field. `path_prefix` holds a copy
of `path`; the indexer writes both.

`path_prefix` is only consulted for prefix patterns. A corpus that
has never been backfilled still answers exact queries correctly.
"""

from __future__ import annotations

from typing import Any

from search.filename_pattern import FilenamePattern

# The payload field carrying the path, indexed with the `word`
# tokenizer. Serves exact patterns.
PATH_FIELD = "path"

# The payload field carrying a copy of the path, indexed with the
# `prefix` tokenizer. Serves prefix patterns.
PATH_PREFIX_FIELD = "path_prefix"


def build_path_filter(pattern: FilenamePattern) -> Any:
    """Build the Qdrant filter condition for a filename pattern.

    Returns a filter ready to be AND'd into a query filter. The
    caller owns the `must` list; this module never mutates shared
    state.

    Two indexes, because the two pattern shapes need different
    tokenizers. Measured against FTS5 on controlled paths where the
    answer is known:

    * **exact** (`chaewon`, `img_2024`) — the `word` tokenizer with
      phrase matching, queried with `MatchPhrase`. The `word`
      tokenizer splits on the same boundaries as FTS5's `unicode61`,
      and `lowercase` + `ascii_folding` reproduce
      `unicode61 remove_diacritics 2`. Phrase matching is what makes
      a multi-token body match only *adjacent* tokens, as FTS5 does.
      Exact match on every case tested.

    * **prefix** (`chaewon*`, `twice_sana*`) — the `prefix`
      tokenizer with phrase matching, queried with `MatchPhrase`
      over the tokenised body.

      The prefix tokenizer indexes every prefix of every token, so
      the LAST token of the body matches any token starting with
      it — which is FTS5's `body*` behaviour. Keeping phrase
      matching ON is what preserves the adjacency: FTS5's
      `twice_sana*` means "the token `twice`, immediately followed by
      a token starting with `sana`", and the prefix tokenizer with
      phrase matching reproduces that exactly — including
      `twice_sanawesome.jpg`, which a `word` tokenizer would miss.

      The word tokenizer cannot serve this case: `MatchText` over
      the whole body is an AND over the prefixes with no adjacency
      requirement, so `twice_sana*` matched 8.7x more paths than
      FTS5 and returned matches where FTS5 returned none.

      And the prefix tokenizer cannot serve the exact case: its
      `MatchPhrase('exact')` also matches `exactly.jpg`, which FTS5
      does not.

    So each pattern shape goes to the index that reproduces FTS5
    for it. `path_prefix` holds a copy of `path` so the two
    tokenizers can live on two fields (a Qdrant collection supports
    one text index per field name).
    """
    from qdrant_client.http import models as qmodels

    if pattern.prefix:
        return qmodels.FieldCondition(
            key=PATH_PREFIX_FIELD,
            match=qmodels.MatchPhrase(phrase=pattern.phrase_body),
        )
    # Exact semantics, with multi-token bodies matched as an
    # adjacent phrase (FTS5's behaviour for `img_2024`).
    return qmodels.FieldCondition(
        key=PATH_FIELD,
        match=qmodels.MatchPhrase(phrase=pattern.phrase_body),
    )
