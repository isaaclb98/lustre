from __future__ import annotations

import pytest

from search import config


def test_default_result_limit_is_28(monkeypatch, tmp_path):
    monkeypatch.delenv("TOP_K_DEFAULT", raising=False)
    monkeypatch.setenv("SEARCH_TEST_MODE", "1")
    monkeypatch.setenv("NAS_IMAGES_BASE", str(tmp_path))
    monkeypatch.setenv("MODEL_NAME", "C-RADIOv4-SO400M")

    loaded = config.load()

    assert loaded.top_k_default == 28


# ---------------------------------------------------------------------
# Round-35: PHOTOS_DIR is the canonical env var name; NAS_IMAGES_BASE
# is a deprecated alias. The tests below pin both halves of the contract
# so a future refactor can't silently drop the alias (which would break
# existing .env files on upgrade) or make PHOTOS_DIR behave
# differently from NAS_IMAGES_BASE in a way that surprises the user.
# ---------------------------------------------------------------------


def test_photos_dir_is_canonical(monkeypatch, tmp_path):
    monkeypatch.delenv("NAS_IMAGES_BASE", raising=False)
    monkeypatch.setenv("SEARCH_TEST_MODE", "1")
    monkeypatch.setenv("PHOTOS_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_NAME", "C-RADIOv4-SO400M")

    loaded = config.load()

    assert loaded.nas_images_base == str(tmp_path)


def test_nas_images_base_works_as_deprecated_alias(monkeypatch, tmp_path):
    """Existing .env files that set NAS_IMAGES_BASE must still load.
    Emits a DeprecationWarning so the operator knows to rename.
    """
    monkeypatch.delenv("PHOTOS_DIR", raising=False)
    monkeypatch.setenv("SEARCH_TEST_MODE", "1")
    monkeypatch.setenv("NAS_IMAGES_BASE", str(tmp_path))
    monkeypatch.setenv("MODEL_NAME", "C-RADIOv4-SO400M")

    with pytest.warns(DeprecationWarning, match="NAS_IMAGES_BASE is deprecated"):
        loaded = config.load()

    assert loaded.nas_images_base == str(tmp_path)


def test_photos_dir_wins_over_nas_images_base(monkeypatch, tmp_path):
    """If both are set (e.g. an operator forgot to delete one when
    renaming), PHOTOS_DIR wins. Old name is ignored silently — no
    deprecation warning when the new name is also set, because the
    user clearly prefers the new name.
    """
    new_dir = tmp_path / "new"
    old_dir = tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()
    monkeypatch.setenv("SEARCH_TEST_MODE", "1")
    monkeypatch.setenv("PHOTOS_DIR", str(new_dir))
    monkeypatch.setenv("NAS_IMAGES_BASE", str(old_dir))
    monkeypatch.setenv("MODEL_NAME", "C-RADIOv4-SO400M")

    loaded = config.load()

    assert loaded.nas_images_base == str(new_dir)


def test_missing_photos_dir_raises_outside_test_mode(monkeypatch, tmp_path):
    """When neither PHOTOS_DIR nor NAS_IMAGES_BASE is set, config.load()
    raises ValueError (unless SEARCH_TEST_MODE is on, which is the
    contract for the existing test suite).
    """
    monkeypatch.delenv("PHOTOS_DIR", raising=False)
    monkeypatch.delenv("NAS_IMAGES_BASE", raising=False)
    monkeypatch.delenv("SEARCH_TEST_MODE", raising=False)
    monkeypatch.setenv("MODEL_NAME", "C-RADIOv4-SO400M")

    with pytest.raises(ValueError, match="PHOTOS_DIR is required"):
        config.load()
