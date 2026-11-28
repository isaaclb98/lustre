"""
Test model variant configuration and validation.

C-RADIOv4-SO400M is the only variant the kernel ships — the
SigLIP2 family is gone. The mismatch coverage this file provided
against four SigLIP2 variants is kept, and now runs against the
one real variant plus collections at OTHER dims (which are still
genuine mismatches and must still be refused).

Uses monkeypatch for all env var mutations so each test's changes
are restored automatically — prevents the SIGLIP_VARIANT='invalid'
from leaking into subsequent test modules (test_v2_smoke.py
chokes on it because config.get_siglip_variant() reads the env
at call time, not at import time).
"""
import pytest
from search import config


# The one variant the kernel ships.
LIVE_VARIANT = "c-radio-v4/so400m"
LIVE_MODEL = "C-RADIOv4-SO400M"
LIVE_DIM = 1536

# A dim the live variant does NOT produce. A collection at this dim
# is a genuine mismatch and must be refused — the same protection
# the old "stored 768, wanted 1152" test provided.
FOREIGN_DIM = 768


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
        f"check lustre_kernel/_radio_models.register_into"
    )
    assert spec.dim == dim, (
        f"registry dim ({spec.dim}) != config dim ({dim}) for {model_name}"
    )


def test_live_variant_maps_to_the_radio_model(monkeypatch):
    """The one variant maps to C-RADIOv4-SO400M at 1536 dims."""
    monkeypatch.setenv("SIGLIP_VARIANT", LIVE_VARIANT)

    variant = config.get_siglip_variant()
    assert variant == LIVE_VARIANT
    assert config.get_model_name_for_variant(variant) == LIVE_MODEL
    assert config.get_vector_dim_for_variant(variant) == LIVE_DIM
    assert config.get_vector_dim() == LIVE_DIM


def test_siglip2_variants_are_gone():
    """The SigLIP2 family must not be selectable any more.

    Each of the four names is asserted ABSENT from the variant
    table. They used to be registered, and that registration is how
    an index run could silently embed with SigLIP2 while the
    operator believed it used C-RADIOv4 — so their absence is a
    property worth pinning, not an accident.
    """
    removed = {
        "B/16-256",
        "L/16-256",
        "gopt/16-384",
        "so400m/16-384",
    }
    assert not (removed & set(config.SIGLIP_VARIANTS.keys())), (
        "SigLIP2 variants are still selectable: "
        f"{sorted(removed & set(config.SIGLIP_VARIANTS.keys()))}"
    )


def test_siglip2_models_are_not_registered():
    """The four SigLIP2 model names must not be in the kernel registry."""
    from lustre_kernel.registry import ModelNotFoundError
    from lustre_kernel.registry import get as _registry_get

    for model_name in (
        "ViT-B-16-SigLIP2-256",
        "ViT-L-16-SigLIP2-256",
        "ViT-gopt-16-SigLIP2-384",
        "ViT-so400m-patch16-384",
    ):
        with pytest.raises(ModelNotFoundError):
            _registry_get(model_name)


def test_invalid_variant_raises(monkeypatch):
    """Invalid variant should raise ValueError with helpful message."""
    monkeypatch.setenv("SIGLIP_VARIANT", "invalid-variant")

    with pytest.raises(ValueError) as exc_info:
        config.get_siglip_variant()
    assert "Invalid SIGLIP_VARIANT" in str(exc_info.value)
    assert LIVE_VARIANT in str(exc_info.value)


@pytest.mark.parametrize("bad_variant", [
    "",
    "b/16-256",                          # case-sensitive
    "so400m/16-384",                     # a removed SigLIP2 variant
    "gopt-16-384",                       # wrong separator
    LIVE_MODEL,                          # model name, not variant name
    "random",
    "c-radio-v4",                        # missing resolution
])
def test_various_invalid_variants_raise(monkeypatch, bad_variant):
    """Various malformed variant names all raise ValueError."""
    monkeypatch.setenv("SIGLIP_VARIANT", bad_variant)
    with pytest.raises(ValueError, match="Invalid SIGLIP_VARIANT"):
        config.get_siglip_variant()


def test_unknown_variant_in_lookup_functions(monkeypatch):
    """get_model_name_for_variant / get_vector_dim_for_variant
    reject unknown variants even when the global env is valid."""
    monkeypatch.setenv("SIGLIP_VARIANT", LIVE_VARIANT)
    with pytest.raises(ValueError, match="Unknown variant"):
        config.get_model_name_for_variant("not-a-real-variant")
    with pytest.raises(ValueError, match="Unknown variant"):
        config.get_vector_dim_for_variant("not-a-real-variant")


def test_get_vector_dim_uses_active_variant(monkeypatch):
    """get_vector_dim() reads from the active env-configured variant."""
    monkeypatch.setenv("SIGLIP_VARIANT", LIVE_VARIANT)
    assert config.get_vector_dim() == LIVE_DIM


def test_siglip_variants_dict_is_complete():
    """SIGLIP_VARIANTS must hold exactly the one live variant, and
    every variant in it must be usable.

    A variant must resolve to a registered model of the same dim,
    and must have a centroid-compat mapping or config.load() refuses
    to boot.
    """
    assert set(config.SIGLIP_VARIANTS.keys()) == {LIVE_VARIANT}
    for variant, (model_name, dim) in config.SIGLIP_VARIANTS.items():
        assert isinstance(model_name, str)
        assert isinstance(dim, int)
        assert dim > 0
        from lustre_kernel.registry import get as _registry_get

        spec = _registry_get(model_name)
        assert spec.dim == dim, (
            f"{variant}: registry dim {spec.dim} disagrees with the "
            f"variant table's {dim}"
        )
        config.centroid_compat_for(model_name)


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


def _two_space_point(dim=LIVE_DIM, picture_dim=None, point_id=1, payload=None):
    """A point carrying BOTH spaces' vectors.

    An upsert replaces the whole vector set, so a point written with
    only the text vector would carry no picture vector at all.
    """
    from qdrant_client.http import models as qm
    if picture_dim is None:
        from lustre_kernel.registry import get as _registry_get

        picture_dim = _registry_get(
            config.get_model_name_for_variant(LIVE_VARIANT)
        ).extra_spaces["dinov3"]
    return qm.PointStruct(
        id=point_id,
        vector={"": [0.0] * dim, "dinov3": [0.0] * picture_dim},
        payload=payload or {},
    )


def _two_space_collection(client, collection, dim=LIVE_DIM, picture_dim=None):
    """Create the collection the way the indexer creates it.

    The live variant writes TWO spaces: the default (unnamed) vector
    is the text space, and `dinov3` is the picture space. A
    single-space collection is a legacy layout — the reconciler
    refuses to start against one, because photo→photo similarity
    would be served from the wrong space.

    The upsert carries BOTH vectors: an upsert replaces the whole
    vector set, so a point written with only the text vector would
    have no picture vector at all.
    """
    from qdrant_client.http import models as qm
    if picture_dim is None:
        from lustre_kernel.registry import get as _registry_get

        picture_dim = _registry_get(
            config.get_model_name_for_variant(LIVE_VARIANT)
        ).extra_spaces["dinov3"]
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "": qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            "dinov3": qm.VectorParams(size=picture_dim, distance=qm.Distance.COSINE),
        },
    )
    return client


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
            LIVE_VARIANT,
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
            vectors_config=qm.VectorParams(size=LIVE_DIM, distance=qm.Distance.COSINE),
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        config.reconcile_variant_from_qdrant(
            LIVE_VARIANT,
            qdrant_url="ignored", qdrant_api_key=None,
            qdrant_collection="images",
        )

    def test_matching_dim_proceeds(self, monkeypatch):
        """Collection dim matches env variant's expected dim → no-op."""
        client = _make_client_for_reconcile()
        _two_space_collection(client, "images", dim=LIVE_DIM)
        client.upsert(
            collection_name="images",
            points=[_two_space_point(dim=LIVE_DIM)],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        config.reconcile_variant_from_qdrant(
            LIVE_VARIANT,
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
        # Old data at a dim the live variant does not produce.
        _dim_via_payloads(client, "images", dim=FOREIGN_DIM)
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                LIVE_VARIANT,
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
        _two_space_collection(client, "images", dim=LIVE_DIM)
        # Single point whose payload claims a different model wrote it.
        client.upsert(
            collection_name="images",
            points=[_two_space_point(
                dim=LIVE_DIM, point_id=1,
                payload={"model_variant": "so400m/16-384"},
            )],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                LIVE_VARIANT,
                qdrant_url="ignored", qdrant_api_key=None,
                qdrant_collection="images",
            )
        msg = str(excinfo.value)
        assert "so400m/16-384" in msg, "must report the actual stored variant"
        # Collection still exists.
        info = client.get_collection(collection_name="images")
        assert info is not None
        assert (info.points_count or 0) > 0

    def test_mismatch_error_lists_the_stored_dim_variant(self, monkeypatch):
        """The error message should suggest the variant that matches
        the stored dim, so the operator doesn't have to guess.
        """
        client = _make_client_for_reconcile()
        # Stored at a dim the live variant does not produce.
        _dim_via_payloads(client, "images", dim=FOREIGN_DIM)
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                LIVE_VARIANT,
                qdrant_url="ignored", qdrant_api_key=None,
                qdrant_collection="images",
            )
        assert str(FOREIGN_DIM) in str(excinfo.value)

    def test_legacy_payload_missing_proceeds(self, monkeypatch):
        """Dim matches, payload missing (legacy data) → proceed."""
        client = _make_client_for_reconcile()
        _two_space_collection(client, "images", dim=LIVE_DIM)
        # Single point, no model_variant payload (legacy).
        client.upsert(
            collection_name="images",
            points=[_two_space_point(dim=LIVE_DIM, point_id=1, payload={})],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        # Should not raise, should not drop.
        config.reconcile_variant_from_qdrant(
            LIVE_VARIANT,
            qdrant_url="ignored", qdrant_api_key=None,
            qdrant_collection="images",
        )
        # Collection still present.
        info = client.get_collection(collection_name="images")
        assert info.points_count == 1

    def test_foreign_dim_without_payload_raises(self, monkeypatch):
        """A dim no variant produces, with no payload to tell which
        model wrote it, must NOT be served.

        The live variant produces 1536. A collection at another dim
        could have been written by any model, so the reconciler
        cannot prove the vectors match the running model — it has to
        refuse.
        """
        client = _make_client_for_reconcile()
        from qdrant_client.http import models as qm
        client.create_collection(
            collection_name="images",
            vectors_config=qm.VectorParams(size=FOREIGN_DIM, distance=qm.Distance.COSINE),
        )
        client.upsert(
            collection_name="images",
            points=[qm.PointStruct(id=1, vector=[0.0] * FOREIGN_DIM, payload={})],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        with pytest.raises(config.QdrantVariantMismatch) as excinfo:
            config.reconcile_variant_from_qdrant(
                LIVE_VARIANT,
                qdrant_url="ignored", qdrant_api_key=None,
                qdrant_collection="images",
            )
        message = str(excinfo.value)
        assert LIVE_VARIANT in message
        assert str(FOREIGN_DIM) in message
        # The collection must survive the refusal.
        info = client.get_collection(collection_name="images")
        assert info.points_count == 1

    def test_matching_payload_proceeds(self, monkeypatch):
        """The live dim whose payload names the live model → the check
        is satisfied and the app boots."""
        client = _make_client_for_reconcile()
        _two_space_collection(client, "images", dim=LIVE_DIM)
        client.upsert(
            collection_name="images",
            points=[_two_space_point(
                dim=LIVE_DIM, point_id=1,
                payload={"model_variant": LIVE_VARIANT},
            )],
        )
        monkeypatch.setattr(
            config, "QdrantClient",
            lambda **kw: client,
        )
        config.reconcile_variant_from_qdrant(
            LIVE_VARIANT,
            qdrant_url="ignored", qdrant_api_key=None,
            qdrant_collection="images",
        )
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
