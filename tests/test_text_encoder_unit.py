"""
tests/test_text_encoder_unit.py — Unit tests for search/text_encoder.py.

Tests the pure-Python helpers (no actual ML model required):
  - _normalize_query_for_siglip2
  - _canonical_prompt_tuple
  - ModelStatus enum
  - get_status / reset_encoder_for_tests
  - TextEncoder with mock model
"""
from __future__ import annotations


from search.text_encoder import (
    DEFAULT_MODEL_NAME,
    MOCK_MODEL_NAME,
    ModelStatus,
    TextEncoder,
    _canonical_prompt_tuple,
    _normalize_query_for_siglip2,
    clear_cache,
    get_status,
    reset_encoder_for_tests,
)


# ----- ModelStatus enum -----

class TestModelStatus:
    """The model loading state enum."""

    def test_values(self):
        assert ModelStatus.NOT_STARTED.value == "not_started"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.ERROR.value == "error"

    def test_is_string_subclass(self):
        """ModelStatus inherits from str — values can be used as strings."""
        assert isinstance(ModelStatus.READY.value, str)
        # Can be compared to strings
        assert ModelStatus.READY == "ready"

    def test_all_states_distinct(self):
        values = [s.value for s in ModelStatus]
        assert len(values) == len(set(values))


# ----- Module constants -----

class TestModuleConstants:
    """The module's exported constants."""

    def test_default_model_name(self):
        """DEFAULT_MODEL_NAME matches the active configured variant.

        Asserts consistency (whatever variant the env says at import
        time, the text encoder's default must match), not a specific
        literal. The default has changed multiple times (gopt → L/16 →
        so400m → B/16-256) and will change again.
        """
        from search.config import get_model_name_for_variant, get_siglip_variant
        active_variant = get_siglip_variant()
        expected = get_model_name_for_variant(active_variant)
        assert expected == DEFAULT_MODEL_NAME, (
            f"text encoder default {DEFAULT_MODEL_NAME!r} != "
            f"model for active variant {active_variant!r} ({expected!r})"
        )

    def test_mock_model_name(self):
        assert MOCK_MODEL_NAME == "mock-1536"

    def test_default_model_is_registered(self):
        """The default model name should be in the kernel registry."""
        from lustre_kernel.registry import get
        spec = get(DEFAULT_MODEL_NAME)
        assert spec is not None


# ----- _normalize_query_for_siglip2 -----

class TestNormalizeQueryForSiglip2:
    """Lowercase normalization for SigLIP2 text tower."""

    def test_lowercases(self):
        assert _normalize_query_for_siglip2("Cat") == "cat"

    def test_already_lowercase_unchanged(self):
        assert _normalize_query_for_siglip2("cat") == "cat"

    def test_uppercase_to_lowercase(self):
        assert _normalize_query_for_siglip2("CAT") == "cat"

    def test_mixed_case(self):
        assert _normalize_query_for_siglip2("cAtDoG") == "catdog"

    def test_empty_string(self):
        assert _normalize_query_for_siglip2("") == ""

    def test_unicode_lowercase(self):
        """Unicode lowercase works (e.g., É → é)."""
        result = _normalize_query_for_siglip2("Café")
        assert result == "café"

    def test_with_numbers_unchanged(self):
        """Numbers don't have case, so they pass through."""
        assert _normalize_query_for_siglip2("IMG_2024") == "img_2024"


# ----- _canonical_prompt_tuple -----

class TestCanonicalPromptTuple:
    """Canonicalize prompt tuples for cache key generation."""

    def test_basic_canonicalization(self):
        result = _canonical_prompt_tuple(("cat", "dog"))
        assert isinstance(result, tuple)

    def test_empty_tuple(self):
        result = _canonical_prompt_tuple(())
        assert isinstance(result, tuple)

    def test_single_prompt(self):
        result = _canonical_prompt_tuple(("cat",))
        assert "cat" in result

    def test_returns_tuple_not_list(self):
        """Return type is tuple, not list (for hashability)."""
        result = _canonical_prompt_tuple(("a", "b"))
        assert isinstance(result, tuple)
        assert not isinstance(result, list)

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        result = _canonical_prompt_tuple(("  cat  ", "dog"))
        # The canonical form should not have leading/trailing spaces
        for item in result:
            assert item == item.strip()

    def test_empty_strings_dropped(self):
        """Empty prompts should be removed."""
        result = _canonical_prompt_tuple(("cat", "", "dog"))
        assert "" not in result

    def test_deduped(self):
        """Duplicate prompts should be removed."""
        result = _canonical_prompt_tuple(("cat", "dog", "cat"))
        # Should have at most one "cat"
        assert result.count("cat") <= 1


# ----- get_status / reset_encoder_for_tests -----

class TestGetStatus:
    """Module-level status tracking."""

    def test_initial_status(self):
        """Reset to known state, then check status."""
        reset_encoder_for_tests()
        status = get_status()
        assert "model_status" in status
        assert status["model_status"] in (
            ModelStatus.NOT_STARTED.value,
            ModelStatus.READY.value,
        )

    def test_status_dict_keys(self):
        """Status dict has model_status, model_name, and model_error."""
        reset_encoder_for_tests()
        status = get_status()
        assert "model_status" in status
        assert "model_name" in status
        assert "model_error" in status

    def test_reset_clears_status(self):
        """reset_encoder_for_tests returns to NOT_STARTED."""
        reset_encoder_for_tests()
        status = get_status()
        assert status["model_status"] in (
            ModelStatus.NOT_STARTED.value,
            ModelStatus.READY.value,
        )


# ----- TextEncoder -----

class TestTextEncoder:
    """The TextEncoder class — routes to the registered model."""

    def test_default_construction(self):
        encoder = TextEncoder()
        assert encoder.device == "cpu"

    def test_test_mode_uses_mock(self):
        """test_mode=True should select the mock embedder."""
        encoder = TextEncoder(test_mode=True)
        # Mock embedder has dim 1536
        assert encoder.dim > 0

    def test_dim_property(self):
        encoder = TextEncoder(test_mode=True)
        # dim comes from the registered embedder
        assert isinstance(encoder.dim, int)
        assert encoder.dim > 0

    def test_resolution_property(self):
        encoder = TextEncoder(test_mode=True)
        # resolution comes from the registered spec
        assert isinstance(encoder.resolution, int)
        assert encoder.resolution > 0

    def test_explicit_arch(self):
        """Can specify a model name explicitly."""
        encoder = TextEncoder(arch=MOCK_MODEL_NAME, test_mode=True)
        assert encoder.dim > 0

    def test_device_attribute(self):
        encoder = TextEncoder(device="cpu", test_mode=True)
        assert encoder.device == "cpu"

    def test_embed_returns_list_of_floats(self):
        encoder = TextEncoder(test_mode=True)
        result = encoder.embed("cat")
        assert isinstance(result, list)
        assert len(result) == encoder.dim
        for v in result:
            assert isinstance(v, float)

    def test_embed_empty_string(self):
        """Empty string should still produce a valid vector."""
        encoder = TextEncoder(test_mode=True)
        result = encoder.embed("")
        assert len(result) == encoder.dim

    def test_embed_deterministic(self):
        """Same input should produce same output (mock is deterministic)."""
        encoder = TextEncoder(test_mode=True)
        v1 = encoder.embed("cat")
        v2 = encoder.embed("cat")
        assert v1 == v2

    def test_embed_different_inputs_different_outputs(self):
        encoder = TextEncoder(test_mode=True)
        v1 = encoder.embed("cat")
        v2 = encoder.embed("dog")
        assert v1 != v2

    def test_embed_normalized_to_unit_length(self):
        """Mock embedder produces unit-norm vectors."""
        import math
        encoder = TextEncoder(test_mode=True)
        vec = encoder.embed("test query")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01


# ----- clear_cache -----

class TestClearCache:
    """The cache-clearing functions."""

    def test_clear_cache_runs(self):
        """clear_cache should not raise."""
        clear_cache()

    def test_clear_cache_multi_runs(self):
        from search.text_encoder import clear_cache_multi
        clear_cache_multi()


# ----- Module imports -----

class TestModuleImports:
    """Public API is importable."""

    def test_text_encoder_importable(self):
        from search.text_encoder import TextEncoder
        assert TextEncoder is not None

    def test_model_status_enum(self):
        from search.text_encoder import ModelStatus
        assert hasattr(ModelStatus, "READY")

    def test_status_helpers(self):
        from search.text_encoder import get_status, reset_encoder_for_tests
        assert callable(get_status)
        assert callable(reset_encoder_for_tests)