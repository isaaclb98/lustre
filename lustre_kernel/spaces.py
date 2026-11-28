"""
lustre_kernel.spaces

The named vector spaces a Lustre collection holds.

A collection stores TWO embedding spaces for every point, and
BOTH are NAMED:

- **TEXT space** — the named vector ``"siglip2"``. SigLIP2's joint
  space: an image vector and a text vector live in the SAME space,
  so a text query retrieves an image directly. This is where text
  search happens, and it is the space the ``siglip2-g`` teacher
  adaptor produces.

- **PICTURE space** — the named vector ``"dinov3"``. DINOv3's
  summary space. DINOv3 is self-supervised and has NO text tower,
  so this space is image-only by construction — it can answer
  photo→photo similarity but never text→photo.

Both spaces come from ONE C-RADIOv4 backbone pass: the teacher
adaptors project the SAME backbone features into their own spaces.
So a dual-space index costs one forward, not two.

Both are named rather than leaving the text space as the unnamed
default: a collection with an unnamed vector has NO default vector
to speak of — Qdrant rejects a query against it that names no
space at all ("Not existing vector name error: "), and the unnamed
key reads as "the old single-space schema" in a payload dump.
Naming both makes the schema self-describing, and makes the space
a parameter of every read rather than a default nobody has to
think about.

The names live here — not in ``search/`` and not in ``indexer/`` —
because both the app and the indexer must agree on them, and a
mismatch is a silent wrong-space search rather than a crash.
"""

from __future__ import annotations

__all__ = [
    "PICTURE_SPACE",
    "SPACE_ADAPTORS",
    "TEXT_SPACE",
]

# The named vector: the text-aligned space.
TEXT_SPACE = "siglip2"

# The named vector: the picture→picture space.
PICTURE_SPACE = "dinov3"

# Space name → the RADIO teacher adaptor that produces it.
#
# The adaptor name is the CHECKPOINT's teacher name, not the type:
# `dino_v3_7b` is the teacher, `dino_v3` is its type. torch.hub
# matches `adaptor_names` against the teacher name, so
# `adaptor_names=["dino_v3"]` raises "Unable to find the specified
# adaptor name".
SPACE_ADAPTORS: dict[str, str] = {
    TEXT_SPACE: "siglip2-g",
    PICTURE_SPACE: "dino_v3_7b",
}
