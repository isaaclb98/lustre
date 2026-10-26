"""
Test model variant configuration and validation.

Uses monkeypatch for all env var mutations so each test's changes
are restored automatically — prevents the SIGLIP_VARIANT='invalid'
from leaking into subsequent test modules (test_v2_smoke.py
chokes on it because config.get_siglip_variant() reads the env
at call time, not at import time).
"""
import pytest
from search import config


# ----- Variant lookup -----

def test_default_variant_resolves_to_registered_model(monkeypatch):
    """The default variant must resolve to a real registered model.

    Asserts the contract (default → variant name → model name → spec
    with correct dim), not a specific literal. The default has changed
    multiple times (gopt → L/16 → so400m → B/16-256) and will change
    again; pinning this test to a specific model is exactly the
    fragility we want to avoid.
    """
    monkeypatch.delenv("SIGLIP_VARIANT", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    variant = config.get_siglip_variant()
    model_name = config.get_model_name_for_variant(variant)
    dim = config.get_vector_dim_for_variant(variant)

    # The default variant must be one of the registered variants.
    assert variant in config.SIGLIP_VARIANTS, (
        f"DEFAULT_VARIANT={variant!r} not in SIGLIP_VARIANTS"
    )

    # The resolved model name must be registered with a matching dim
    # in the kernel registry. Catches drift between the variant table
    # and the real-model registration.
    from lustre_kernel.registry import get as _registry_get
    spec = _registry_get(model_name)
    assert spec is not None, (
        f"default model {model_name!r} not in kernel registry — "
        f"check lustre_kernel/_real_models.register_into"
    )
    assert spec.dim == dim, (
        f"registry dim ({spec.dim}) != config dim ({dim}) for {model_name}"
    )


@pytest.mark.parametrize("variant,expected_model,expected_dim", [
    ("B/16-256", "ViT-B-16-SigLIP2-256", 768),
    ("L/16-256", "ViT-L-16-SigLIP2-256", 1024),
    ("gopt/16-384", "ViT-gopt-16-SigLIP2-384", 1536),
    ("so400m/16-384", "ViT-so400m-patch16-384", 1152),
])
def test_all_known_variants(monkeypatch, variant, expected_model, expected_dim):
    """Each registered variant maps to the right model and dim."""
    monkeypatch.setenv("SIGLIP_VARIANT", variant)
    assert config.get_siglip_variant() == variant
    assert config.get_model_name_for_variant(variant) == expected_model
    assert config.get_vector_dim_for_variant(variant) == expected_dim
    assert config.get_vector_dim() == expected_dim


def test_gopt_variant(monkeypatch):
    """gopt variant should be 1536-dim."""
    monkeypatch.setenv("SIGLIP_VARIANT", "gopt/16-384")

    variant = config.get_siglip_variant()
    assert variant == "gopt/16-384"

    model_name = config.get_model_name_for_variant(variant)
    assert model_name == "ViT-gopt-16-SigLIP2-384"

    dim = config.get_vector_dim_for_variant(variant)
    assert dim == 1536


def test_invalid_variant_raises(monkeypatch):
    """Invalid variant should raise ValueError with helpful message."""
    monkeypatch.setenv("SIGLIP_VARIANT", "invalid-variant")

    with pytest.raises(ValueError) as exc_info:
        config.get_siglip_variant()
    assert "Invalid SIGLIP_VARIANT" in str(exc_info.value)
    assert "B/16-256" in str(exc_info.value)
    assert "L/16-256" in str(exc_info.value)
    assert "gopt/16-384" in str(exc_info.value)


@pytest.mark.parametrize("bad_variant", [
    "",
    "b/16-256",                          # case-sensitive
    "L/16-512",                          # valid format but no such variant
    "gopt-16-384",                       # wrong separator
    "ViT-L-16-SigLIP2-256",              # model name, not variant name
    "random",
    "L/16",                              # missing resolution
])
def test_various_invalid_variants_raise(monkeypatch, bad_variant):
    """Various malformed variant names all raise ValueError."""
    monkeypatch.setenv("SIGLIP_VARIANT", bad_variant)
    with pytest.raises(ValueError, match="Invalid SIGLIP_VARIANT"):
        config.get_siglip_variant()


def test_unknown_variant_in_lookup_functions(monkeypatch):
    """get_model_name_for_variant / get_vector_dim_for_variant
    reject unknown variants even when the global env is valid."""
    monkeypatch.setenv("SIGLIP_VARIANT", "L/16-256")
    with pytest.raises(ValueError, match="Unknown variant"):
        config.get_model_name_for_variant("not-a-real-variant")
    with pytest.raises(ValueError, match="Unknown variant"):
        config.get_vector_dim_for_variant("not-a-real-variant")


def test_get_vector_dim_uses_active_variant(monkeypatch):
    """get_vector_dim() reads from the active env-configured variant."""
    monkeypatch.setenv("SIGLIP_VARIANT", "B/16-256")
    assert config.get_vector_dim() == 768

    monkeypatch.setenv("SIGLIP_VARIANT", "gopt/16-384")
    assert config.get_vector_dim() == 1536


def test_siglip_variants_dict_is_complete():
    """SIGLIP_VARIANTS dict should have all four documented variants
    with valid model names and positive dims."""
    assert set(config.SIGLIP_VARIANTS.keys()) == {
        "B/16-256", "L/16-256", "gopt/16-384", "so400m/16-384",
    }
    for variant, (model_name, dim) in config.SIGLIP_VARIANTS.items():
        assert isinstance(model_name, str)
        assert model_name.startswith("ViT-"), f"{variant}: bad model name {model_name!r}"
        # The so400m HF repo is named `timm/ViT-so400m-patch16-384`
        # without a "SigLIP2" suffix (quirk of HF repo naming); the
        # other three have "SigLIP2" in their HF repo names. Accept
        # either, since `_CENTROID_MODEL_COMPAT` is what actually
        # drives the model-family grouping.
        assert ("SigLIP2" in model_name or "so400m" in model_name), (
            f"{variant}: model {model_name!r} is not in the SigLIP2 family"
        )
        assert isinstance(dim, int)
        assert dim > 0
        assert dim in (768, 1024, 1152, 1536), f"{variant}: unexpected dim {dim}"


# ----- Variant reconciliation against Qdrant (round 34) -----


def _dim_via_payloads(client, collection, dim):
    """Helper: upsert one fake point at `dim` so get_collection reports it.

    Qdrant in-memory needs at least one point to expose the vector
    config in get_collection(). We use real vector data (a single
    zero vector) rather than mocking, because the reconciler reads
    info.config.params.vectors which is set at collection creation.
    """
    from qdrant_client.http import models as qm
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    client.upsert(
        collection_name=collection,
        points=[qm.PointStruct(id=1, vector=[0.0] * dim, payload={})],
    )


def _make_client_for_reconcile(**_kw):
    """Real in-memory Qdrant client. Discards url/api_key kwargs.

    The reconciler always passes url=, api_key=, timeout=; in-memory
    Qdrant ignores them. Returning a callable that accepts **kw
    makes the test monkeypatch work uniformly with the production
    call shape.
    """
    from qdrant_client import QdrantClient
    return QdrantClient(location=":memory:")


class TestReconcileVariantFromQdrant:
    """Behavior contract for reconcile_variant_from_qdrant.

    Uses real in-memory Qdrant (no mocks) so the actual Qdrant
    schema, dim semantics, and UnexpectedResponse are tested.
    """

    def test_fresh_install_collection_absent_is_noop(self, monkeypatch):
        """No Qdrant collection → no-op. App boots, user reindexes."""
        monkeypatch.setattr(
            config, "QdrantClient", _make_client_for_reconcile,
        )
        # Never create the collection. Should not raise.
        config.reconcile_variant_from_qdrant(
            "B/16-256",
            qdrant_url="ignored",
            qdrant_api_key=None,
            qdrant_collection="images",
        )

    def test_empty_collection_is_noop(self, monkeypatch):
        """Empty Qdrant collection → no-op."""
        monkeypatch.setattr(config, "QdrantClient", _make_client_for_reconcile)
        # Create collection but add no points.
        from qdrant_client.http import models as qm
        client = _make_client_for_reconcile()
        client.create_collection(
            collection_name="images",
            vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE),
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        config.reconcile_variant_from_qdrant(
            "B/16-256",
            qdrant_url="ignored", qdrant_api_key=None,
            qdrant_collection="images",
        )

    def test_matching_dim_proceeds(self, monkeypatch):
        """Collection dim matches env variant's expected dim → no-op."""
        client = _make_client_for_reconcile()
        _dim_via_payloads(client, "images", dim=1152)  # so400m = 1152
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        config.reconcile_variant_from_qdrant(
            "so400m/16-384",
            qdrant_url="ignored", qdrant_api_key=None,
            qdrant_collection="images",
        )

    def test_dim_mismatch_raises_and_keeps_collection(self, monkeypatch):
        """Collection dim ≠ env variant → raise, do NOT drop.

        Refuse-to-start reconciler. The collection is preserved so
        the operator can recover manually (rename + reindex, fix
        the env, or explicit drop via curl). The old behavior —
        silently dropping the collection on startup — is the bug
        this test exists to prevent.
        """
        client = _make_client_for_reconcile()
        # Old data at 768 (B/16-256). User now wants so400m (1152).
        _dim_via_payloads(client, "images", dim=768)
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                "so400m/16-384",
                qdrant_url="ignored", qdrant_api_key=None,
                qdrant_collection="images",
            )
        # The error message must surface all three recovery paths
        # so the operator doesn't have to dig through docs to
        # figure out what to do next.
        msg = str(excinfo.value)
        assert "Set SIGLIP_VARIANT" in msg, "must suggest the env-var fix path"
        assert "rename" in msg.lower(), "must suggest the rename-for-postmortem path"
        assert "curl -X DELETE" in msg, "must surface the manual-drop command"
        # Collection still exists — that's the whole point.
        info = client.get_collection(collection_name="images")
        assert info is not None
        assert (info.points_count or 0) > 0

    def test_payload_mismatch_raises_and_keeps_collection(self, monkeypatch):
        """Dim matches but per-point payload differs → raise, do NOT drop."""
        client = _make_client_for_reconcile()
        from qdrant_client.http import models as qm
        client.create_collection(
            collection_name="images",
            vectors_config=qm.VectorParams(size=1152, distance=qm.Distance.COSINE),
        )
        # Single point with model_variant=L/16-256 (dim=1024, but we're
        # pretending the indexer wrote the wrong payload for the actual
        # 1152-dim vectors).
        client.upsert(
            collection_name="images",
            points=[qm.PointStruct(
                id=1, vector=[0.0] * 1152,
                payload={"model_variant": "L/16-256"},
            )],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                "so400m/16-384",
                qdrant_url="ignored", qdrant_api_key=None,
                qdrant_collection="images",
            )
        msg = str(excinfo.value)
        assert "L/16-256" in msg, "must report the actual stored variant"
        # Collection still exists.
        info = client.get_collection(collection_name="images")
        assert info is not None
        assert (info.points_count or 0) > 0

    def test_mismatch_error_lists_the_stored_dim_variant(self, monkeypatch):
        """The error message should suggest the variant that matches
        the stored dim, so the operator doesn't have to guess.
        """
        client = _make_client_for_reconcile()
        # Stored at 768 → likely variant is B/16-256.
        _dim_via_payloads(client, "images", dim=768)
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                "so400m/16-384",
                qdrant_url="ignored", qdrant_api_key=None,
                qdrant_collection="images",
            )
        assert "B/16-256" in str(excinfo.value)

    def test_legacy_payload_missing_proceeds(self, monkeypatch):
        """Dim matches, payload missing (legacy data) → proceed."""
        client = _make_client_for_reconcile()
        from qdrant_client.http import models as qm
        client.create_collection(
            collection_name="images",
            vectors_config=qm.VectorParams(size=1152, distance=qm.Distance.COSINE),
        )
        # Single point, no model_variant payload (legacy).
        client.upsert(
            collection_name="images",
            points=[qm.PointStruct(id=1, vector=[0.0] * 1152, payload={})],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        # Should not raise, should not drop.
        config.reconcile_variant_from_qdrant(
            "so400m/16-384",
            qdrant_url="ignored", qdrant_api_key=None,
            qdrant_collection="images",
        )
        # Collection still present.
        info = client.get_collection(collection_name="images")
        assert info.points_count == 1


# ----- DEFAULT_MODEL constant -----

def test_default_model_constant_resolves():
    """DEFAULT_MODEL must match the model for the runtime variant.

    Asserts that DEFAULT_MODEL is consistent with the active variant
    the env declares at import time. The constant is env-driven, so
    it follows SIGLIP_VARIANT — DEFAULT_VARIANT is irrelevant here.
    """
    variant = config.get_siglip_variant()
    expected = config.get_model_name_for_variant(variant)
    assert expected == config.DEFAULT_MODEL, (
        f"DEFAULT_MODEL={config.DEFAULT_MODEL!r} != "
        f"model for active variant {variant!r} ({expected!r})"
    )
