"""
Tests for the C-RADIOv4-SO400M model path.

Three things are worth pinning here:

1. The registry entry exists and carries the right shape. A model
   that is registered but mis-dimensioned would silently produce
   vectors Qdrant cannot store.

2. The image is prepared at NATIVE aspect ratio. This is the whole
   point of the transition: the SigLIP2 path letterboxes onto a
   square canvas, which puts black bars around a portrait photo and
   squashes a landscape one. RADIO's ViT takes the image as it is.

3. `embed_images` batches by shape. Native-aspect images are not
   all the same size, so a plain `torch.stack` would raise; the
   embedder groups them instead.

None of these tests load the checkpoint. The model is exercised
end-to-end by the dev stack, not by the unit suite.
"""
import pytest
from PIL import Image

from lustre_kernel import registry
from search import config


RADIO_MODEL = "C-RADIOv4-SO400M"
RADIO_VARIANT = "c-radio-v4/so400m"


# ----- Registry -----

def test_radio_model_is_registered():
    spec = registry.get(RADIO_MODEL)
    assert spec.name == RADIO_MODEL
    assert spec.dim == 1536
    # The long edge of the native-aspect input, not a square.
    #
    # It is MODEL_INPUT_SIZE, NOT the display thumbnail's size. The
    # two were one number until the 640 embed took the GPU off the
    # PCIe bus (Xid 79): the ViT's cost scales with its INPUT
    # resolution, so welding them meant paying GPU cost for a
    # display size.
    from lustre_kernel._radio_models import MODEL_INPUT_SIZE

    assert spec.resolution == MODEL_INPUT_SIZE
    assert spec.preprocess == "native"


def test_radio_variant_resolves_to_the_radio_model():
    assert registry.resolve_model_name(RADIO_VARIANT) == RADIO_MODEL
    assert registry.VARIANT_TO_MODEL[RADIO_VARIANT] == RADIO_MODEL


def test_radio_variant_is_in_the_config_variant_table():
    assert RADIO_VARIANT in config.SIGLIP_VARIANTS
    model_name, dim = config.SIGLIP_VARIANTS[RADIO_VARIANT]
    assert model_name == RADIO_MODEL
    assert dim == registry.get(RADIO_MODEL).dim


def test_radio_model_has_a_centroid_compat_mapping():
    """`config.load()` raises without one, which fails the boot."""
    tag, dim = config.centroid_compat_for(RADIO_MODEL)
    assert tag == "cradio-v4-siglip2g"
    assert dim == 1536


def test_radio_variant_is_a_different_space_from_the_siglip_family():
    """C-RADIOv4 must not be servable against a SigLIP2 collection.

    Two independent guards have to hold:

    - the DIM differs, so the reconciler's dim check refuses to
      boot against a collection written by the prod model;
    - the variant NAME differs, so the per-point `model_variant`
      payload check refuses too.

    The dim guard is the load-bearing one (a stale collection is
    the realistic mistake); the payload guard is the belt-and-braces
    for a collection that happens to share the dim.
    """
    radio_dim = config.get_vector_dim_for_variant(RADIO_VARIANT)
    siglip_dim = config.get_vector_dim_for_variant("so400m/16-384")
    assert radio_dim == 1536
    assert siglip_dim == 1152
    assert radio_dim != siglip_dim
    assert RADIO_VARIANT != "so400m/16-384"


def test_radio_dim_matches_the_gopt_dim_but_not_its_space():
    """gopt is also 1536-dim — the payload guard is what separates them.

    Two models can share a dim and still be different spaces. This
    is the case where the dim check alone would NOT catch a
    mismatched collection, which is exactly why the reconciler also
    reads the per-point `model_variant` payload.
    """
    radio_dim = config.get_vector_dim_for_variant(RADIO_VARIANT)
    gopt_dim = config.get_vector_dim_for_variant("gopt/16-384")
    assert radio_dim == gopt_dim == 1536
    assert RADIO_VARIANT != "gopt/16-384"


# ----- Native-aspect preprocessing -----

def test_prepare_keeps_the_long_edge_and_snaps_to_the_patch_grid():
    from lustre_kernel._radio_models import RadioEmbedder

    embedder = RadioEmbedder(dim=1536, long_edge=512)

    # Landscape: long edge is the width. 1200x800 is 3:2, so the
    # short edge is 341 — snapped down to the patch grid at 336.
    t = embedder._prepare(Image.new("RGB", (1200, 800)))
    assert t.shape[-1] == 512
    assert t.shape[-2] == 336

    # Portrait: long edge is the height.
    t = embedder._prepare(Image.new("RGB", (800, 1200)))
    assert t.shape[-2] == 512
    assert t.shape[-1] == 336

    # Square stays square.
    t = embedder._prepare(Image.new("RGB", (1000, 1000)))
    assert t.shape[-2] == t.shape[-1] == 512


def test_prepare_edges_are_multiples_of_the_patch_size():
    from lustre_kernel._radio_models import RADIO_PATCH_SIZE, RadioEmbedder

    embedder = RadioEmbedder(dim=1536, long_edge=512)
    for size in [(1023, 767), (640, 480), (999, 333), (2000, 137)]:
        t = embedder._prepare(Image.new("RGB", size))
        assert t.shape[-1] % RADIO_PATCH_SIZE == 0, size
        assert t.shape[-2] % RADIO_PATCH_SIZE == 0, size


def test_prepare_preserves_the_aspect_ratio():
    """No squashing: the output ratio must match the source.

    The letterbox path keeps the ratio too, but inside a square
    canvas. Here there is no canvas — the ratio has to survive on
    its own, within the patch-grid rounding tolerance.
    """
    from lustre_kernel._radio_models import RadioEmbedder

    embedder = RadioEmbedder(dim=1536, long_edge=512)
    for size in [(1200, 800), (800, 1200), (1600, 900), (900, 1600)]:
        t = embedder._prepare(Image.new("RGB", size))
        src_ratio = size[0] / size[1]
        out_ratio = t.shape[-1] / t.shape[-2]
        assert abs(out_ratio - src_ratio) / src_ratio < 0.06, (size, out_ratio)


def test_prepare_returns_a_float_tensor_in_unit_range():
    from lustre_kernel._radio_models import RadioEmbedder

    embedder = RadioEmbedder(dim=1536, long_edge=512)
    t = embedder._prepare(Image.new("RGB", (300, 300), (255, 128, 0)))
    assert t.dtype.is_floating_point
    assert float(t.min()) >= 0.0
    assert float(t.max()) <= 1.0
    assert t.shape[0] == 3  # CHW


# ----- Shape-bucketed batching -----

class _FakeRadioModel:
    """Stands in for the RADIO model.

    Records the shape of every batch it is handed so the test can
    assert that only same-shaped images were stacked together, and
    derives each image's vector from its actual pixel content so
    ordering assertions mean something.
    """

    def __init__(self, dim=1536):
        self.dim = dim
        self.batch_shapes = []
        self.adaptors = {"siglip2-g": self}

    def __call__(self, batch):

        self.batch_shapes.append(tuple(batch.shape[-2:]))
        n = batch.shape[0]
        # Per-image mean, offset so a black image does not normalise
        # to NaN. Tiled to the model dim: two images with the same
        # content get the same vector, different content differs.
        channels = batch.reshape(n, 3, -1).mean(dim=2) + 1.0
        summary = channels.reshape(n, 3, 1).repeat(1, 1, self.dim // 3).reshape(n, self.dim)
        return {"siglip2-g": (summary, None)}


@pytest.fixture
def radio_embedder():
    """Returns (embedder, fake_model).

    The fake stands in for the checkpoint so the batching logic can
    be tested without downloading 1.5 GB of weights.
    """
    from lustre_kernel._radio_models import RadioEmbedder

    embedder = RadioEmbedder(dim=1536, long_edge=512)
    fake = _FakeRadioModel()
    embedder._model = fake  # bypass the checkpoint load
    embedder._ensure_loaded = lambda: None
    return embedder, fake


def test_embed_images_batches_only_same_shaped_images(radio_embedder):
    embedder, fake = radio_embedder
    images = [
        Image.new("RGB", (1200, 800)),
        Image.new("RGB", (800, 1200)),
        Image.new("RGB", (1000, 1000)),
        Image.new("RGB", (1200, 800)),  # same shape as the first
    ]
    vectors = embedder.embed_images(images)

    assert len(vectors) == 4
    assert all(len(v) == 1536 for v in vectors)

    # Three distinct shapes -> three forward passes, each stacked
    # only with images of the same shape. Shapes are (H, W).
    assert fake.batch_shapes == [
        (336, 512),
        (512, 336),
        (512, 512),
    ]


def test_embed_images_returns_vectors_in_input_order(radio_embedder):
    embedder, _fake = radio_embedder
    images = [
        Image.new("RGB", (1200, 800), (255, 0, 0)),
        Image.new("RGB", (800, 1200), (0, 255, 0)),
        Image.new("RGB", (1200, 800), (0, 0, 255)),
    ]
    vectors = embedder.embed_images(images)
    # The two landscapes batch together, the portrait is its own
    # batch. Order must still follow the input, so the red image
    # (first) and the blue one (third) must NOT share a vector.
    assert vectors[0] != vectors[1]
    assert vectors[0] != vectors[2]
    assert vectors[1] != vectors[2]


def test_embed_images_gives_identical_images_identical_vectors(radio_embedder):
    embedder, _fake = radio_embedder
    images = [
        Image.new("RGB", (1200, 800), (12, 34, 56)),
        Image.new("RGB", (800, 1200), (99, 11, 22)),
        Image.new("RGB", (1200, 800), (12, 34, 56)),
    ]
    vectors = embedder.embed_images(images)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_embed_images_normalises(radio_embedder):
    embedder, _fake = radio_embedder
    images = [Image.new("RGB", (1200, 800))]
    (vector,) = embedder.embed_images(images)
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_embed_images_rejects_non_pil(radio_embedder):
    embedder, _fake = radio_embedder
    with pytest.raises(TypeError, match="non-PIL"):
        embedder.embed_images([Image.new("RGB", (10, 10)), "not an image"])


# ----- Image loader policy -----

def test_prepare_for_embedding_follows_the_model_policy():
    """The policy helper: native models keep the picture, the
    SigLIP2 models get the square canvas."""
    from indexer.image_loader import prepare_for_embedding

    img = Image.new("RGB", (400, 800))

    native = prepare_for_embedding(img, RADIO_MODEL)
    assert native is img
    assert native.size == (400, 800)

    letterboxed = prepare_for_embedding(img, "ViT-so400m-patch16-384")
    assert letterboxed.size == (384, 384)


def test_the_indexer_does_not_letterbox_behind_the_policys_back():
    """The embed path must go through the policy, not around it.

    `local_sync` used to call `letterbox_resize()` directly, which
    silently forced the square canvas onto every model — including
    the native-aspect ones. The embeddings in the dev collection
    were letterboxed as a result, while every other code path
    (the loader, the pipeline) honoured the policy.
    """
    import ast
    from pathlib import Path

    from indexer import local_sync

    source = Path(local_sync.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    direct_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "letterbox_resize"
    ]
    assert not direct_calls, (
        "indexer/local_sync.py calls letterbox_resize directly — the "
        "model's own preprocess policy must decide how an image is "
        "prepared for embedding"
    )

    used = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }
    assert "prepare_for_embedding" in used


def test_loader_does_not_letterbox_for_native_models(tmp_path):
    """A native model must get the image as decoded — no black bars."""
    from indexer.image_loader import load

    path = tmp_path / "portrait.jpg"
    Image.new("RGB", (900, 1350), (200, 30, 30)).save(path)

    native, src_w, src_h = load(path, model_name=RADIO_MODEL)
    assert (src_w, src_h) == (900, 1350)
    # Not square, and no padding added.
    assert native.width != native.height
    assert native.size[0] <= 1350


def test_loader_still_letterboxes_for_siglip_models(tmp_path):
    """The SigLIP2 path must not change."""
    from indexer.image_loader import load

    path = tmp_path / "portrait.jpg"
    Image.new("RGB", (900, 1350), (200, 30, 30)).save(path)

    letterboxed, src_w, src_h = load(path, model_name="ViT-so400m-patch16-384")
    assert (src_w, src_h) == (900, 1350)
    assert letterboxed.size == (384, 384)
