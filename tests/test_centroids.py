"""
tests/test_centroids.py

Unit tests for search.centroids.CentroidStore.

We build minimal in-memory torch .pt files for each test rather than
relying on real `isaac-image-scoring` outputs — the store should
treat the .pt file as an opaque blob with a known schema, and we
want the tests to exercise the schema guard, not the upstream
extraction code.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from search.centroids import CentroidStore


def _save_centroid(
    path: Path,
    name: str,
    *,
    model: str = "cradio-v4-siglip2g",
    feature_dim: int = 1536,
    centroid_shape: tuple[int, ...] | None = None,
    n_images: int = 10,
    model_type: str | None = None,
    model_id: str | None = None,
) -> None:
    """
    Build a minimal valid centroid dict and torch.save it. Tests
    can override individual fields to exercise guard branches.
    """
    if centroid_shape is None:
        centroid_shape = (feature_dim,)
    blob = {
        "centroid": torch.randn(*centroid_shape),
        "name": name,
        "model": model,
        "model_type": model_type if model_type is not None else model,
        "model_id": model_id,
        "feature_dim": feature_dim,
        "n_images": n_images,
        "extracted_at": "2026-06-17T00:00:00",
    }
    torch.save(blob, path)


# ------------------------------ load() ------------------------------


def test_load_empty_dir(tmp_path: Path) -> None:
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0
    assert s.list_all() == []
    assert s.count() == 0


def test_load_missing_dir(tmp_path: Path) -> None:
    """A missing dir is fine — store is empty, no error."""
    missing = tmp_path / "does-not-exist"
    s = CentroidStore(missing, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0
    assert s.list_all() == []


def test_load_none_dir() -> None:
    """`centroids_dir=None` is the unset-env-var case — never errors."""
    s = CentroidStore(None, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0


def test_load_one_valid(tmp_path: Path) -> None:
    _save_centroid(tmp_path / "wuxia.pt", "wuxia_female_leads")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 1
    spec = s.get("wuxia_female_leads")
    assert spec is not None
    assert spec.feature_dim == 1536
    assert spec.n_images == 10
    assert spec.model == "cradio-v4-siglip2g"
    assert spec.model_type == "cradio-v4-siglip2g"
    assert spec.model_id is None
    assert spec.extracted_at == "2026-06-17T00:00:00"
    # Vector is materialized as a list[float] of the right length.
    assert isinstance(spec.vector, list)
    assert len(spec.vector) == 1536
    # And is roughly unit-norm (random tensor then re-normalized would
    # be exactly 1.0; raw random has norm sqrt(1536) ~= 39). We don't
    # renormalize on load — the .pt file is expected to hold a
    # unit-norm centroid. Verify the loader preserves the input
    # tensor faithfully by computing its norm.
    norm = math.sqrt(sum(v * v for v in spec.vector))
    # Random tensor of dim 1536 has expected norm ~sqrt(1536) ≈ 39.1
    # Allow a wide range so the test isn't flaky.
    assert 20 < norm < 60


# ------------------------------ guards ------------------------------


def test_skip_dim_mismatch(tmp_path: Path) -> None:
    """A 4096-dim dino v3 centroid is silently skipped against a 1536-dim store."""
    _save_centroid(tmp_path / "dino.pt", "dino_centroid", model="dinov3", feature_dim=4096)
    _save_centroid(tmp_path / "ok.pt", "ok_centroid")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 1
    assert s.get("dino_centroid") is None
    assert s.get("ok_centroid") is not None


def test_skip_model_mismatch(tmp_path: Path) -> None:
    """Same dim but a different model tag is also rejected."""
    _save_centroid(tmp_path / "other.pt", "other_centroid", model="clip-vit-l", feature_dim=1536)
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0
    assert s.get("other_centroid") is None


def test_skip_bad_shape(tmp_path: Path) -> None:
    """A 2D centroid tensor is rejected even if the dim count matches."""
    _save_centroid(
        tmp_path / "bad.pt", "bad_centroid",
        centroid_shape=(1, 1536), feature_dim=1536,
    )
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0


def test_skip_empty_name(tmp_path: Path) -> None:
    _save_centroid(tmp_path / "empty.pt", "")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0


def test_skip_missing_keys(tmp_path: Path) -> None:
    """A blob missing the 'centroid' or 'name' key is rejected, doesn't crash."""
    torch.save({"name": "x", "model": "cradio-v4-siglip2g", "feature_dim": 1536}, tmp_path / "no_centroid.pt")
    torch.save({"centroid": torch.randn(1536), "model": "cradio-v4-siglip2g", "feature_dim": 1536}, tmp_path / "no_name.pt")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    assert s.load() == 0


def test_skip_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "junk.pt").write_bytes(b"not a torch file")
    _save_centroid(tmp_path / "ok.pt", "ok_centroid")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    # Corrupt file skipped, valid one still loads — a single bad file
    # does not abort the load.
    assert s.load() == 1
    assert s.get("ok_centroid") is not None


def test_duplicate_name_keeps_first(tmp_path: Path) -> None:
    """Two files with the same stored name — keep the first by sorted path order."""
    _save_centroid(tmp_path / "a_wuxia.pt", "wuxia", n_images=1)
    _save_centroid(tmp_path / "b_wuxia.pt", "wuxia", n_images=99)
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    spec = s.get("wuxia")
    assert spec is not None
    # Sorted glob → a_wuxia.pt comes first → n_images=1 is kept.
    assert spec.n_images == 1
    assert spec.source_path.name == "a_wuxia.pt"


# ------------------------------ lookups ------------------------------


def test_get_case_insensitive(tmp_path: Path) -> None:
    _save_centroid(tmp_path / "x.pt", "Wuxia_Female_Leads")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    assert s.get("Wuxia_Female_Leads") is not None
    assert s.get("wuxia_female_leads") is not None
    assert s.get("WUXIA_FEMALE_LEADS") is not None


def test_get_missing_returns_none(tmp_path: Path) -> None:
    _save_centroid(tmp_path / "x.pt", "wuxia")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    assert s.get("not_loaded") is None


def test_list_sorted_by_name(tmp_path: Path) -> None:
    _save_centroid(tmp_path / "b.pt", "beta")
    _save_centroid(tmp_path / "a.pt", "alpha")
    _save_centroid(tmp_path / "c.pt", "gamma")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    assert [c.name for c in s.list_all()] == ["alpha", "beta", "gamma"]


def test_as_dict_has_public_fields(tmp_path: Path) -> None:
    _save_centroid(
        tmp_path / "x.pt", "wuxia",
        n_images=42, model_type="ensemble", model_id="facebook/dinov3",
    )
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    d = s.list_all()[0].as_dict()
    assert d["name"] == "wuxia"
    assert d["model"] == "cradio-v4-siglip2g"
    assert d["model_type"] == "ensemble"
    assert d["model_id"] == "facebook/dinov3"
    assert d["feature_dim"] == 1536
    assert d["n_images"] == 42
    assert d["extracted_at"] == "2026-06-17T00:00:00"
    assert d["source_path"].endswith("x.pt")


# ------------------------------ reload ------------------------------


def test_load_is_idempotent(tmp_path: Path) -> None:
    """Calling load() twice with no disk changes returns the same set."""
    _save_centroid(tmp_path / "x.pt", "wuxia")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    n1 = s.count()
    s.load()
    assert s.count() == n1


def test_load_picks_up_new_files(tmp_path: Path) -> None:
    """A second load() after a new file appears loads it (reload semantics)."""
    _save_centroid(tmp_path / "x.pt", "wuxia")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    assert s.count() == 1
    _save_centroid(tmp_path / "y.pt", "noir")
    s.load()
    assert s.count() == 2
    assert s.get("noir") is not None


def test_load_drops_deleted_files(tmp_path: Path) -> None:
    """A reload after a file is removed drops the corresponding entry."""
    _save_centroid(tmp_path / "x.pt", "wuxia")
    _save_centroid(tmp_path / "y.pt", "noir")
    s = CentroidStore(tmp_path, expected_model="cradio-v4-siglip2g", expected_feature_dim=1536)
    s.load()
    assert s.count() == 2
    (tmp_path / "y.pt").unlink()
    s.load()
    assert s.count() == 1
    assert s.get("noir") is None


def test_centroids_compute_module_is_pure():
    """Phase B3 contract: search.centroids_compute is pure."""
    import search.centroids_compute as compute

    # All public compute entry points are available.
    assert callable(compute.blend_centroids)
    assert callable(compute.composite_centroid_name)
    assert callable(compute.calibrate_near_dup_threshold)
    assert callable(compute.filter_near_duplicates)


def test_centroids_service_re_exports_compute_api():
    """Backwards-compat: callers can still import from search.centroids."""
    from search import centroids

    # Pure compute names accessible from the service module too.
    assert centroids.blend_centroids is not None
    assert centroids.calibrate_near_dup_threshold is not None
    assert centroids.composite_centroid_name is not None
    assert centroids.filter_near_duplicates is not None
    # The persistence + state classes live here.
    assert hasattr(centroids, "CentroidStore")
    assert hasattr(centroids, "DynamicCentroidRegistry")
    assert hasattr(centroids, "CentroidSpec")
