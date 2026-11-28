"""
Tests for the two-space collection design.

Lustre holds ONE collection with TWO named vectors per point:

- the TEXT space (the default, unnamed vector) — SigLIP2's joint
  space, so a text query retrieves an image directly;
- the PICTURE space (the named vector `dinov3`) — DINOv3's summary
  space, so photo→photo similarity ranks photographs by how they
  look.

Both come from ONE C-RADIOv4 backbone pass: every teacher adaptor
projects the SAME backbone features into its own space.

What these tests pin:

1. The space NAMES are a single shared constant, and the model
   spec declares the picture space's dim.
2. One forward pass produces both spaces (the fake model records
   how many times it was called).
3. A named-vector read returns a DICT, so the space has to be
   selected explicitly — reading without saying which space wants
   a dict where a list is expected.
4. The collection is created with one vector per space.
5. An upsert carries EVERY space's vector — Qdrant replaces the
   whole vector set on upsert, so a text-only upsert would silently
   drop the picture vector.
6. The startup reconciler refuses a collection that is missing the
   picture space.
7. The photo→photo route searches the picture space.

None of these tests load the checkpoint or touch a live Qdrant.
"""
from __future__ import annotations

import pytest
from PIL import Image

from lustre_kernel import registry
from lustre_kernel.spaces import PICTURE_SPACE, SPACE_ADAPTORS, TEXT_SPACE


RADIO_MODEL = "C-RADIOv4-SO400M"


# ----- Space names -----

def test_space_names_are_shared_constants():
    """Both spaces are named: the text space is "siglip2"."""
    assert TEXT_SPACE == "siglip2"
    assert PICTURE_SPACE == "dinov3"


def test_space_adaptors_are_the_checkpoint_teacher_names():
    """The adaptor name is the teacher's NAME, not its type.

    `dino_v3` is the teacher TYPE; `dino_v3_7b` is the teacher
    NAME. torch.hub matches `adaptor_names` against the name, so
    `adaptor_names=["dino_v3"]` raises "Unable to find the
    specified adaptor name".
    """
    assert SPACE_ADAPTORS[TEXT_SPACE] == "siglip2-g"
    assert SPACE_ADAPTORS[PICTURE_SPACE] == "dino_v3_7b"


def test_model_spec_declares_the_picture_space():
    spec = registry.get(RADIO_MODEL)
    assert spec.extra_spaces == {PICTURE_SPACE: 4096}
    # The text space's dim is the spec's own `dim`.
    assert spec.dim == 1536


def test_the_mock_writes_no_extra_spaces():
    """The mock's collection holds just the default vector."""
    spec = registry.get(registry._MOCK_REGISTRY_NAME)
    assert spec.extra_spaces == {}


# ----- One forward pass, two spaces -----

class _FakeDualAdaptorModel:
    """Stands in for the checkpoint with BOTH teacher adaptors.

    Records every forward pass so a test can assert that two
    spaces came out of ONE pass, and gives each adaptor a
    DIFFERENT vector for the same image so the two spaces are
    distinguishable in assertions.
    """

    def __init__(self, text_dim=1536, picture_dim=4096):
        self.text_dim = text_dim
        self.picture_dim = picture_dim
        self.calls = 0
        self.batch_shapes = []

    def __call__(self, batch):
        self.calls += 1
        self.batch_shapes.append(tuple(batch.shape[-2:]))
        n = batch.shape[0]
        channels = batch.reshape(n, 3, -1).mean(dim=2) + 1.0

        def _tile(dim):
            """Per-image content tiled to exactly `dim` floats."""
            base = channels.reshape(n, 3, 1).repeat(1, 1, dim // 3 + 1)
            return base.reshape(n, -1)[:, :dim]

        text = _tile(self.text_dim)
        # A different vector for the same image: the picture space
        # is NOT the text space.
        picture = _tile(self.picture_dim) * 2.0
        return {
            "siglip2-g": (text, None),
            "dino_v3_7b": (picture, None),
        }


@pytest.fixture
def dual_embedder():
    """Returns (embedder, fake_model) with BOTH adaptors loaded."""
    from lustre_kernel._radio_models import RadioEmbedder

    embedder = RadioEmbedder(
        dim=1536,
        long_edge=512,
        adaptors=("siglip2-g", "dino_v3_7b"),
    )
    fake = _FakeDualAdaptorModel()
    embedder._model = fake
    embedder._ensure_loaded = lambda: None
    return embedder, fake


def test_embed_images_multi_returns_both_spaces(dual_embedder):
    embedder, _fake = dual_embedder
    images = [Image.new("RGB", (1200, 800)), Image.new("RGB", (800, 1200))]

    out = embedder.embed_images_multi(images)

    assert set(out) == {TEXT_SPACE, PICTURE_SPACE}
    assert len(out[TEXT_SPACE]) == 2
    assert len(out[PICTURE_SPACE]) == 2
    assert all(len(v) == 1536 for v in out[TEXT_SPACE])
    assert all(len(v) == 4096 for v in out[PICTURE_SPACE])


def test_embed_images_multi_is_one_forward_pass(dual_embedder):
    """Two spaces must not cost two backbone passes.

    Both teacher adaptors project the SAME backbone features, so
    one `model(batch)` call yields both summaries. Two passes
    would double the indexer's cost for nothing.
    """
    embedder, fake = dual_embedder
    images = [
        Image.new("RGB", (1200, 800)),
        Image.new("RGB", (1200, 800)),
        Image.new("RGB", (800, 1200)),
    ]

    embedder.embed_images_multi(images)

    # Two shapes -> two forward passes, not two per space.
    assert fake.calls == 2
    assert fake.batch_shapes == [(336, 512), (512, 336)]


def test_embed_images_multi_preserves_input_order(dual_embedder):
    embedder, _fake = dual_embedder
    images = [
        Image.new("RGB", (1200, 800), (255, 0, 0)),
        Image.new("RGB", (800, 1200), (0, 255, 0)),
        Image.new("RGB", (1200, 800), (0, 0, 255)),
    ]

    out = embedder.embed_images_multi(images)

    text = out[TEXT_SPACE]
    assert text[0] != text[1]
    assert text[0] != text[2]
    # The picture space is ordered the same way.
    picture = out[PICTURE_SPACE]
    assert picture[0] != picture[1]


def test_embed_images_multi_normalises_both_spaces(dual_embedder):
    embedder, _fake = dual_embedder
    out = embedder.embed_images_multi([Image.new("RGB", (1200, 800))])

    for space, vectors in out.items():
        for vector in vectors:
            norm = sum(x * x for x in vector) ** 0.5
            assert abs(norm - 1.0) < 1e-5, space


def test_embed_images_multi_rejects_non_pil(dual_embedder):
    embedder, _fake = dual_embedder
    with pytest.raises(TypeError, match="non-PIL"):
        embedder.embed_images_multi([Image.new("RGB", (10, 10)), "not an image"])


def test_embed_images_multi_on_empty_input(dual_embedder):
    embedder, _fake = dual_embedder
    out = embedder.embed_images_multi([])
    assert out == {TEXT_SPACE: [], PICTURE_SPACE: []}


def test_a_space_whose_adaptor_is_not_loaded_is_absent():
    """Asking for a space the model cannot produce is an absence.

    The siglip2-only embedder has no DINOv3 adaptor, so the
    picture space is simply not in the result. Returning an empty
    list for it would be a wrong-space answer dressed up as data.
    """
    from lustre_kernel._radio_models import RadioEmbedder

    embedder = RadioEmbedder(dim=1536, long_edge=512, adaptors=("siglip2-g",))
    fake = _FakeDualAdaptorModel()
    embedder._model = fake
    embedder._ensure_loaded = lambda: None

    out = embedder.embed_images_multi([Image.new("RGB", (1200, 800))])

    assert set(out) == {TEXT_SPACE}
    assert PICTURE_SPACE not in out


def test_embed_images_still_returns_the_text_space(dual_embedder):
    """The single-space API keeps working."""
    embedder, _fake = dual_embedder
    vectors = embedder.embed_images([Image.new("RGB", (1200, 800))])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1536


# ----- Reading a named vector -----

def test_space_vector_picks_the_named_space():
    from search.qdrant_client import _space_vector

    vec = _space_vector(
        {TEXT_SPACE: [0.1, 0.2], PICTURE_SPACE: [0.3, 0.4, 0.5]}, PICTURE_SPACE,
    )
    assert vec == [0.3, 0.4, 0.5]


def test_space_vector_defaults_to_the_unnamed_vector():
    from search.qdrant_client import _space_vector

    assert _space_vector({TEXT_SPACE: [0.1, 0.2], PICTURE_SPACE: [0.3]}) == [0.1, 0.2]


def test_space_vector_returns_none_for_a_missing_space():
    """A missing vector is not an empty one."""
    from search.qdrant_client import _space_vector

    assert _space_vector({TEXT_SPACE: [0.1]}, "dinov3") is None


def test_space_vector_passes_through_a_flat_vector():
    from search.qdrant_client import _space_vector

    assert _space_vector([0.1, 0.2]) == [0.1, 0.2]


# ----- Collection shape -----

class _Collections:
    """What `client.get_collections()` returns when nothing exists."""

    def __init__(self) -> None:
        self.collections: list = []


def test_ensure_collection_creates_one_vector_per_space():
    """The collection is created with a default vector AND a
    named one — Qdrant cannot add a named vector to an existing
    collection, so this is the only chance to get the shape
    right."""
    from indexer import upsert

    created = {}

    class _Client:
        def get_collections(self):
            return _Collections()

        def create_collection(self, collection_name, vectors_config):
            created["name"] = collection_name
            created["vectors_config"] = vectors_config

    upsert.ensure_collection(
        _Client(), "images", dim=1536,
        extra_spaces={PICTURE_SPACE: 4096},
    )

    vectors = created["vectors_config"]
    assert set(vectors) == {TEXT_SPACE, PICTURE_SPACE}
    assert vectors[TEXT_SPACE].size == 1536
    assert vectors[PICTURE_SPACE].size == 4096


def test_ensure_collection_defaults_to_a_single_space():
    """A caller that does not ask for extra spaces gets the plain
    default vector.

    The search side and the legacy callers create single-space
    collections; putting a 4096-dim DINOv3 space into them would
    make every point written without it carry no picture vector.
    """
    from indexer import upsert

    created = {}

    class _Client:
        def get_collections(self):
            return _Collections()

        def create_collection(self, collection_name, vectors_config):
            created["vectors_config"] = vectors_config

    upsert.ensure_collection(_Client(), "images", dim=1536)

    assert set(created["vectors_config"]) == {TEXT_SPACE}


# ----- The upsert carries every space -----

def test_upsert_batch_sends_a_vector_dict_per_point():
    """Every point's upsert must carry EVERY space's vector.

    An upsert REPLACES the whole vector set (measured on Qdrant
    1.19), so a text-only upsert would silently drop the picture
    vector and leave the point with no picture-space vector at all.
    """
    import ast
    from pathlib import Path

    from indexer import local_sync

    source = Path(local_sync.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # The `vectors = {...}` that feeds the upsert must be a DICT
    # literal keyed by the text space — not a bare vector.
    dict_builds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "vectors"
            for t in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert dict_builds, (
        "indexer/local_sync.py must build a per-space vector dict — "
        "a bare vector would drop the other space's vector on upsert"
    )

    # Its first key is the TEXT space (a Name node referring to the
    # constant, not a string literal).
    keys = {
        k.id for k in dict_builds[0].value.keys if isinstance(k, ast.Name)
    }
    assert "TEXT_SPACE" in keys, keys

    # And the OTHER spaces are added from the embedder's output, so
    # the picture space's vector is not left behind.
    subscripts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and t.value.id == "vectors"
            for t in node.targets
        )
    ]
    assert subscripts, (
        "indexer/local_sync.py must copy every non-text space's "
        "vector into the per-point dict"
    )

    # The values come from the embedder's per-space output.
    source_text = source
    assert "vecs_by_space" in source_text, (
        "the per-point dict must be built from the embedder's "
        "multi-space output"
    )


def test_upsert_batch_accepts_both_vector_shapes():
    """The signature must accept a dict of spaces OR a bare vector.

    The indexer writes a dict (multi-space collections need it);
    the legacy single-space callers write a bare list.
    """
    import inspect

    from indexer import upsert

    sig = inspect.signature(upsert.upsert_batch)
    item_type = str(sig.parameters["items"].annotation)
    assert "dict" in item_type, item_type
    assert "list" in item_type, item_type


def test_upsert_batch_refuses_a_bare_vector_on_a_two_space_collection():
    """A bare vector would silently wipe the other space's vector.

    The REAL server accepts the shape mismatch and drops the
    missing space with no error (measured against Qdrant 1.19) —
    only qdrant-client's local in-memory mode refuses it. So the
    guard has to live in our code, not in the client's.
    """
    from qdrant_client import QdrantClient, models

    from indexer import upsert

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="two_space",
        vectors_config={
            TEXT_SPACE: models.VectorParams(size=4, distance=models.Distance.COSINE),
            PICTURE_SPACE: models.VectorParams(
                size=3, distance=models.Distance.COSINE,
            ),
        },
    )
    with pytest.raises(TypeError, match="bare vector"):
        upsert.upsert_batch(
            client, "two_space",
            [("11111111-1111-1111-1111-111111111111", [1.0, 0, 0, 0], {})],
        )


def test_upsert_batch_refuses_a_partial_space_dict():
    """A dict missing a space would wipe that space's vector."""
    from qdrant_client import QdrantClient, models

    from indexer import upsert

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="two_space",
        vectors_config={
            TEXT_SPACE: models.VectorParams(size=4, distance=models.Distance.COSINE),
            PICTURE_SPACE: models.VectorParams(
                size=3, distance=models.Distance.COSINE,
            ),
        },
    )
    with pytest.raises(ValueError, match="missing"):
        upsert.upsert_batch(
            client, "two_space",
            [("11111111-1111-1111-1111-111111111111",
              {TEXT_SPACE: [1.0, 0, 0, 0]}, {})],
        )


def test_upsert_batch_accepts_a_bare_vector_on_a_single_space_collection():
    """The legacy shape still works where it is valid."""
    from qdrant_client import QdrantClient, models

    from indexer import upsert

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="one_space",
        vectors_config=models.VectorParams(
            size=4, distance=models.Distance.COSINE,
        ),
    )
    upsert.upsert_batch(
        client, "one_space",
        [("11111111-1111-1111-1111-111111111111", [1.0, 0, 0, 0], {})],
        wait=True,
    )
    got = client.retrieve(
        collection_name="one_space",
        ids=["11111111-1111-1111-1111-111111111111"],
        with_vectors=True,
    )
    assert len(got) == 1


# ----- The reconciler refuses a single-space collection -----

class _FakeCollectionInfo:
    def __init__(self, vectors):
        self.config = type(
            "cfg", (), {"params": type("params", (), {"vectors": vectors})()},
        )()
        self.points_count = 10


def test_collection_spaces_reads_every_named_vector():
    from search.config import _qdrant_collection_spaces

    class _P:
        def __init__(self, size):
            self.size = size

    info = _FakeCollectionInfo({TEXT_SPACE: _P(1536), PICTURE_SPACE: _P(4096)})
    assert _qdrant_collection_spaces(info) == {TEXT_SPACE: 1536, PICTURE_SPACE: 4096}


def test_collection_dim_reads_the_text_space():
    """The dim check must read the TEXT space by name.

    The picture space is 4096-dim. Taking whichever vector the
    config lists first would compare 4096 against the expected
    1536 and refuse a perfectly good collection.
    """
    from search.config import _qdrant_collection_dim

    class _P:
        def __init__(self, size):
            self.size = size

    info = _FakeCollectionInfo({TEXT_SPACE: _P(1536), PICTURE_SPACE: _P(4096)})
    assert _qdrant_collection_dim(info) == 1536


def test_check_extra_spaces_rejects_a_missing_picture_space():
    """A collection indexed with the old schema must not boot.

    Its text dim is right, so the dim check passes. Without the
    space check the app would boot and serve photo→photo
    similarity out of the wrong space.
    """
    from search.config import QdrantVariantMismatch, _check_extra_spaces

    class _P:
        def __init__(self, size):
            self.size = size

    info = _FakeCollectionInfo({TEXT_SPACE: _P(1536)})
    with pytest.raises(QdrantVariantMismatch, match="MISSING"):
        _check_extra_spaces(info, "images", "http://qdrant:6333", "c-radio-v4/so400m")


def test_check_extra_spaces_rejects_a_wrong_dim():
    from search.config import QdrantVariantMismatch, _check_extra_spaces

    class _P:
        def __init__(self, size):
            self.size = size

    info = _FakeCollectionInfo({TEXT_SPACE: _P(1536), PICTURE_SPACE: _P(1024)})
    with pytest.raises(QdrantVariantMismatch, match="dim=1024"):
        _check_extra_spaces(info, "images", "http://qdrant:6333", "c-radio-v4/so400m")


def test_check_extra_spaces_accepts_the_full_schema():
    from search.config import _check_extra_spaces

    class _P:
        def __init__(self, size):
            self.size = size

    info = _FakeCollectionInfo({TEXT_SPACE: _P(1536), PICTURE_SPACE: _P(4096)})
    _check_extra_spaces(info, "images", "http://qdrant:6333", "c-radio-v4/so400m")


# ----- The photo→photo route uses the picture space -----

def test_similar_route_uses_the_picture_space():
    """The retrieve AND the search must both use the picture space.

    A query vector is only meaningful in the space it was drawn
    from: the 4096-d DINOv3 vector compared against 1536-d
    SigLIP2 vectors is meaningless, and the search would have to
    reject it.
    """
    from search.routers import similar

    calls = {}

    class _Qdrant:
        def retrieve_with_vector(self, point_id, using=None):
            calls["retrieve"] = (point_id, using)
            return ([0.1] * 4096, _Hit())

        def search(self, vec, limit, offset=0, *args, **kwargs):
            calls["search"] = kwargs
            calls["search_args"] = args
            return ([], False)

    class _Hit:
        def __init__(self) -> None:
            self.id = "p1"
            self.path = "/x.jpg"
            self.score = 0.0
            self.payload: dict = {}

    router = similar.build_similar_router(qdrant=_Qdrant(), cfg=_Cfg(), index_db=_Db())
    assert router is not None

    # The route body is what matters; call it directly through the
    # router's endpoint.
    import asyncio

    endpoint = next(
        r for r in router.routes
        if getattr(r, "path", "") == "/api/similar/{point_id}"
    )
    # The endpoint's defaults are FastAPI `Query` objects, so the
    # limit/offset have to be passed explicitly.
    result = asyncio.run(endpoint.endpoint(point_id="p1", limit=20, offset=0))
    assert result is not None

    assert calls["retrieve"][1] == PICTURE_SPACE
    # The search's `using` — passed positionally in the route.
    assert PICTURE_SPACE in calls["search_args"] or (
        calls["search"].get("using") == PICTURE_SPACE
    )


class _Cfg:
    top_k_default = 20
    web_ui_url = "http://localhost"


class _Db:
    def favorite_id_set(self, ids):
        return set()

    def dislike_id_set(self, ids):
        return set()
