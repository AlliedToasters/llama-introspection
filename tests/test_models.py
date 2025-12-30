"""Tests for model configuration (src/llama_introspection/models.py)."""

import pytest


class TestModelShortcuts:
    """Tests for model shortcut configuration."""

    def test_shortcuts_exist(self):
        """MODEL_SHORTCUTS contains expected model sizes."""
        from llama_introspection.models import MODEL_SHORTCUTS

        assert "1B" in MODEL_SHORTCUTS
        assert "8B" in MODEL_SHORTCUTS
        assert "70B" in MODEL_SHORTCUTS
        assert "405B" in MODEL_SHORTCUTS

    def test_shortcuts_are_valid_hf_ids(self):
        """MODEL_SHORTCUTS values are valid HuggingFace model IDs."""
        from llama_introspection.models import MODEL_SHORTCUTS

        for shortcut, model_id in MODEL_SHORTCUTS.items():
            assert "/" in model_id, f"Model ID {model_id} should contain '/'"
            assert "llama" in model_id.lower(), f"Model ID {model_id} should be a Llama model"


class TestRemoteModels:
    """Tests for remote model configuration."""

    def test_remote_models_subset_of_shortcuts(self):
        """REMOTE_MODELS should reference models in MODEL_SHORTCUTS."""
        from llama_introspection.models import MODEL_SHORTCUTS, REMOTE_MODELS

        shortcut_values = set(MODEL_SHORTCUTS.values())
        shortcut_keys = set(MODEL_SHORTCUTS.keys())

        for model in REMOTE_MODELS:
            assert model in shortcut_values or model in shortcut_keys, (
                f"Remote model {model} not found in MODEL_SHORTCUTS"
            )


class TestDefaultConfigs:
    """Tests for default experiment configurations."""

    def test_default_strengths(self):
        """DEFAULT_STRENGTHS is a non-empty list of positive floats."""
        from llama_introspection.models import DEFAULT_STRENGTHS

        assert len(DEFAULT_STRENGTHS) > 0
        assert all(isinstance(s, (int, float)) for s in DEFAULT_STRENGTHS)
        assert all(s > 0 for s in DEFAULT_STRENGTHS)

    def test_default_scale_factors(self):
        """DEFAULT_SCALE_FACTORS is a non-empty list of positive floats."""
        from llama_introspection.models import DEFAULT_SCALE_FACTORS

        assert len(DEFAULT_SCALE_FACTORS) > 0
        assert all(isinstance(s, (int, float)) for s in DEFAULT_SCALE_FACTORS)
        assert all(s > 0 for s in DEFAULT_SCALE_FACTORS)


class TestResolveModel:
    """Tests for model resolution utility."""

    def test_resolve_shortcut(self):
        """resolve_model_id expands shortcuts to full IDs."""
        from llama_introspection.models import resolve_model_id

        assert resolve_model_id("1B") == "meta-llama/Llama-3.2-1B-Instruct"
        assert resolve_model_id("8B") == "meta-llama/Llama-3.1-8B-Instruct"

    def test_resolve_full_id_passthrough(self):
        """resolve_model_id passes through full model IDs unchanged."""
        from llama_introspection.models import resolve_model_id

        full_id = "meta-llama/Llama-3.1-8B-Instruct"
        assert resolve_model_id(full_id) == full_id

    def test_resolve_unknown_raises(self):
        """resolve_model_id raises ValueError for unknown shortcuts."""
        from llama_introspection.models import resolve_model_id

        with pytest.raises(ValueError):
            resolve_model_id("unknown-model")


class TestIsRemoteModel:
    """Tests for remote model detection."""

    def test_is_remote_model_shortcut(self):
        """is_remote_model correctly identifies remote models by shortcut."""
        from llama_introspection.models import is_remote_model

        assert is_remote_model("405B") is True
        assert is_remote_model("1B") is False

    def test_is_remote_model_full_id(self):
        """is_remote_model correctly identifies remote models by full ID."""
        from llama_introspection.models import is_remote_model

        assert is_remote_model("meta-llama/Llama-3.1-405B-Instruct") is True
        assert is_remote_model("meta-llama/Llama-3.2-1B-Instruct") is False
