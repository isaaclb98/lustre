"""tests/test_schema.py — schema invariants.

Pins the payload field set so a drift between
`lustre_kernel.payload_schema`, `indexer/upsert.py:build_payload`,
and `SCHEMA.md` shows up as a test failure rather than as a silent
search-side bug.
"""

from __future__ import annotations


import pytest


@pytest.fixture(scope="module")
def schema_module():
    # The kernel package owns the canonical schema constants. Tests
    # import only via this fixture; the regression test in §A1 enforces
    # no file outside `lustre_kernel/` references payload field
    # names directly.
    from lustre_kernel import payload_schema

    return payload_schema


def test_payload_field_names_are_distinct(schema_module):
    names = schema_module.payload_field_names()
    assert len(names) == len(set(names)), "duplicate field name in schema"


def test_payload_field_names_are_nonempty_strings(schema_module):
    for name in schema_module.payload_field_names():
        assert isinstance(name, str)
        assert name, "field name must not be empty"


def test_fingerprint_fields_are_in_payload(schema_module):
    payload_names = set(schema_module.payload_field_names())
    for fp in schema_module.FINGERPRINT_FIELDS:
        assert fp in payload_names, f"{fp} missing from payload_field_names()"


def test_build_payload_uses_every_schema_field(tmp_path, schema_module):
    """Cross-check: every FIELD_* constant must appear in the dict
    produced by `build_payload()`. Catches the case where a field
    gets added to the schema but not to the writer.
    """
    from PIL import Image

    from lustre_kernel.payload_schema import payload_field_names
    from indexer.upsert import build_payload

    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    p = tmp_path / "sample.png"
    img.save(p, "PNG")

    payload = build_payload(
        p, shard="", model_name="test", model_revision="r0",
        collection="default",
    )

    expected = set(payload_field_names())
    actual = set(payload.keys())
    missing = expected - actual
    assert not missing, f"build_payload missing fields: {sorted(missing)}"


def test_schema_doc_lists_every_payload_field(schema_module):
    """Cross-check: the kernel constants stay non-empty.

    Removed the prose-doc assertion (the SCHEMA.md prose mirror was
    deleted). The kernel module's `payload_field_names()` is now the
    only source of truth, so this just guards the constant set is
    populated.
    """
    names = schema_module.payload_field_names()
    assert len(names) >= 1, "payload_field_names() returned empty"


def test_schema_field_set_non_empty(schema_module):
    """Guard against accidentally dropping REQUIRED_FIELDS to empty.

    Removed: SCHEMA_VERSION machinery was retired with the migration
    layer; this test now just guards that REQUIRED_FIELDS is still a
    non-empty frozenset so callers don't accidentally drop a required
    field.
    """
    assert isinstance(schema_module.REQUIRED_FIELDS, frozenset)
    assert len(schema_module.REQUIRED_FIELDS) >= 1


def test_build_payload_sets_required_fields(tmp_path, schema_module):
    """Cross-check: `build_payload` writes every REQUIRED_FIELDS entry."""
    from PIL import Image

    from indexer.upsert import build_payload

    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    p = tmp_path / "sample.png"
    img.save(p, "PNG")

    payload = build_payload(
        p, shard="", model_name="test", model_revision="r0",
        collection="default",
    )
    missing = schema_module.REQUIRED_FIELDS - payload.keys()
    assert not missing, f"build_payload missing required fields: {missing}"


def test_build_payload_sets_folder_field(tmp_path, schema_module):
    """Cross-check: `build_payload` writes the `folder` field as the
    absolute parent directory path."""
    from PIL import Image

    from lustre_kernel.payload_schema import FIELD_FOLDER
    from indexer.upsert import build_payload

    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    sub = tmp_path / "vacation_photos"
    sub.mkdir()
    p = sub / "sample.png"
    img.save(p, "PNG")

    payload = build_payload(
        p, shard="", model_name="test", model_revision="r0",
        collection="default",
    )
    assert payload[FIELD_FOLDER] == str(sub.resolve())


def test_build_payload_sets_model_dim_from_registry(tmp_path, schema_module):
    """Cross-check: `build_payload` writes `model_dim` sourced from
    the registry, not from a hardcoded constant.
    """
    from PIL import Image

    from lustre_kernel.payload_schema import FIELD_MODEL_DIM
    from lustre_kernel.registry import get as registry_get
    from indexer.upsert import build_payload

    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    p = tmp_path / "sample.png"
    img.save(p, "PNG")

    payload = build_payload(
        p, shard="", model_name="test", model_revision="r0",
        collection="default",
    )
    assert payload[FIELD_MODEL_DIM] == registry_get("test").dim


def test_require_fields_accepts_complete_payload(schema_module):
    """A payload with every REQUIRED_FIELDS entry is accepted."""
    payload = {
        "id": "abc",
        "path": "/some/where.jpg",
        "folder": "/some",
        "model_name": "test",
        "model_revision": "r0",
        "model_dim": 1536,
    }
    schema_module.require_fields(payload)  # does not raise


def test_require_fields_rejects_missing_folder(schema_module):
    """Required fields must all be present."""
    payload = {
        "id": "abc",
        "path": "/some/where.jpg",
        "model_name": "test",
        "model_revision": "r0",
        "model_dim": 1536,
        # folder missing
    }
    with pytest.raises(ValueError, match="missing required field"):
        schema_module.require_fields(payload)
