"""Tests for steering vector computation (src/llama_introspection/steering.py)."""

import pytest
import torch


class TestSteeringVectorResult:
    """Tests for SteeringVectorResult dataclass."""

    def test_init_with_vectors(self):
        """SteeringVectorResult can be initialized with vectors and metadata."""
        from llama_introspection.steering import SteeringVectorResult

        vectors = [torch.randn(1, 64) for _ in range(4)]
        metadata = {"type": "test", "model_slug": "test-model"}

        result = SteeringVectorResult(vectors=vectors, metadata=metadata)

        assert result.num_layers == 4
        assert result.hidden_dim == 64
        assert result.metadata["type"] == "test"

    def test_getitem(self):
        """SteeringVectorResult supports indexing."""
        from llama_introspection.steering import SteeringVectorResult

        vectors = [torch.randn(1, 64) for _ in range(4)]
        result = SteeringVectorResult(vectors=vectors, metadata={})

        assert result[0].shape == (1, 64)
        assert torch.equal(result[2], vectors[2])

    def test_len(self):
        """SteeringVectorResult supports len()."""
        from llama_introspection.steering import SteeringVectorResult

        vectors = [torch.randn(1, 64) for _ in range(4)]
        result = SteeringVectorResult(vectors=vectors, metadata={})

        assert len(result) == 4

    def test_norms(self):
        """SteeringVectorResult.norms() returns L2 norms per layer."""
        from llama_introspection.steering import SteeringVectorResult

        # Create vectors with known norms
        v1 = torch.tensor([[3.0, 4.0]])  # norm = 5
        v2 = torch.tensor([[1.0, 0.0]])  # norm = 1
        result = SteeringVectorResult(vectors=[v1, v2], metadata={})

        norms = result.norms()
        assert len(norms) == 2
        assert pytest.approx(norms[0], rel=1e-5) == 5.0
        assert pytest.approx(norms[1], rel=1e-5) == 1.0


class TestCachePath:
    """Tests for cache path generation."""

    def test_cache_path_deterministic(self, tmp_path):
        """Same inputs produce same cache path."""
        from llama_introspection.steering import get_cache_path

        path1 = get_cache_path(tmp_path, "model-a", "bespoke", positive="p", negative="n")
        path2 = get_cache_path(tmp_path, "model-a", "bespoke", positive="p", negative="n")

        assert path1 == path2

    def test_cache_path_different_for_different_inputs(self, tmp_path):
        """Different inputs produce different cache paths."""
        from llama_introspection.steering import get_cache_path

        path1 = get_cache_path(tmp_path, "model-a", "bespoke", positive="p1", negative="n")
        path2 = get_cache_path(tmp_path, "model-a", "bespoke", positive="p2", negative="n")

        assert path1 != path2

    def test_cache_path_includes_vector_type(self, tmp_path):
        """Cache path filename includes the vector type."""
        from llama_introspection.steering import get_cache_path

        path = get_cache_path(tmp_path, "model", "random", seed=42)

        assert "random" in path.name


class TestNormalizeVectors:
    """Tests for vector normalization utilities."""

    def test_normalize_to_target_norm(self):
        """normalize_vectors scales all vectors to target norm."""
        from llama_introspection.steering import SteeringVectorResult, normalize_vectors

        vectors = [torch.randn(1, 64) for _ in range(3)]
        result = SteeringVectorResult(vectors=vectors, metadata={})

        normalized = normalize_vectors(result, target_norm=10.0)

        for v in normalized.vectors:
            assert pytest.approx(v.norm().item(), rel=1e-5) == 10.0

    def test_normalize_preserves_direction(self):
        """normalize_vectors preserves vector direction."""
        from llama_introspection.steering import SteeringVectorResult, normalize_vectors

        v = torch.tensor([[3.0, 4.0]])
        result = SteeringVectorResult(vectors=[v], metadata={})

        normalized = normalize_vectors(result, target_norm=1.0)

        # Direction should be preserved (cosine similarity = 1)
        cos_sim = torch.nn.functional.cosine_similarity(v, normalized.vectors[0])
        assert pytest.approx(cos_sim.item(), rel=1e-5) == 1.0


class TestMatchVectorNorms:
    """Tests for matching vector norms to a reference."""

    def test_match_norms_layer_by_layer(self):
        """match_vector_norms matches norms from reference layer-by-layer."""
        from llama_introspection.steering import SteeringVectorResult, match_vector_norms

        source = SteeringVectorResult(
            vectors=[torch.randn(1, 64) for _ in range(3)],
            metadata={"type": "random"},
        )
        reference = SteeringVectorResult(
            vectors=[torch.randn(1, 64) * (i + 1) for i in range(3)],
            metadata={"type": "bespoke"},
        )

        matched = match_vector_norms(source, reference)

        for i in range(3):
            src_norm = matched.vectors[i].norm().item()
            ref_norm = reference.vectors[i].norm().item()
            assert pytest.approx(src_norm, rel=1e-5) == ref_norm

    def test_match_norms_layer_count_mismatch_raises(self):
        """match_vector_norms raises on layer count mismatch."""
        from llama_introspection.steering import SteeringVectorResult, match_vector_norms

        source = SteeringVectorResult(vectors=[torch.randn(1, 64)], metadata={})
        reference = SteeringVectorResult(vectors=[torch.randn(1, 64) for _ in range(3)], metadata={})

        with pytest.raises(AssertionError):
            match_vector_norms(source, reference)


class TestComputeRandomVector:
    """Tests for random vector computation."""

    def test_random_vector_reproducible(self, tiny_model):
        """Same seed produces same random vectors."""
        from llama_introspection.steering import compute_random_vector

        model_slug = "test-model"
        result1 = compute_random_vector(tiny_model, model_slug, seed=42)
        result2 = compute_random_vector(tiny_model, model_slug, seed=42)

        for v1, v2 in zip(result1.vectors, result2.vectors):
            assert torch.equal(v1, v2)

    def test_random_vector_different_seeds(self, tiny_model):
        """Different seeds produce different random vectors."""
        from llama_introspection.steering import compute_random_vector

        model_slug = "test-model"
        result1 = compute_random_vector(tiny_model, model_slug, seed=42)
        result2 = compute_random_vector(tiny_model, model_slug, seed=123)

        assert not torch.equal(result1.vectors[0], result2.vectors[0])


class TestGetNumLayers:
    """Tests for layer counting utility."""

    def test_get_num_layers(self, tiny_model):
        """get_num_layers returns correct layer count."""
        from llama_introspection.steering import get_num_layers

        num_layers = get_num_layers(tiny_model)

        assert isinstance(num_layers, int)
        assert num_layers > 0
