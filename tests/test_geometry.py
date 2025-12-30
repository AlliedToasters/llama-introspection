"""Tests for vector geometry computations (src/llama_introspection/geometry.py)."""

import pytest
import torch


class TestVectorNorms:
    """Tests for vector norm computation."""

    def test_l2_norm(self):
        """compute_l2_norm returns correct Euclidean norm."""
        from llama_introspection.geometry import compute_l2_norm

        v = torch.tensor([3.0, 4.0])
        assert pytest.approx(compute_l2_norm(v), rel=1e-5) == 5.0

    def test_l2_norm_batch(self):
        """compute_l2_norm handles batched vectors."""
        from llama_introspection.geometry import compute_l2_norm

        batch = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        norms = compute_l2_norm(batch, dim=-1)

        assert pytest.approx(norms[0].item(), rel=1e-5) == 5.0
        assert pytest.approx(norms[1].item(), rel=1e-5) == 1.0


class TestVectorDistance:
    """Tests for vector distance computation."""

    def test_l2_distance(self):
        """compute_l2_distance returns correct Euclidean distance."""
        from llama_introspection.geometry import compute_l2_distance

        v1 = torch.tensor([0.0, 0.0])
        v2 = torch.tensor([3.0, 4.0])

        assert pytest.approx(compute_l2_distance(v1, v2), rel=1e-5) == 5.0

    def test_l2_distance_same_vector(self):
        """compute_l2_distance returns 0 for identical vectors."""
        from llama_introspection.geometry import compute_l2_distance

        v = torch.tensor([1.0, 2.0, 3.0])
        assert compute_l2_distance(v, v) == 0.0


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_cosine_similarity_parallel(self):
        """Parallel vectors have cosine similarity of 1."""
        from llama_introspection.geometry import compute_cosine_similarity

        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([2.0, 0.0])

        assert pytest.approx(compute_cosine_similarity(v1, v2), rel=1e-5) == 1.0

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors have cosine similarity of 0."""
        from llama_introspection.geometry import compute_cosine_similarity

        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])

        assert pytest.approx(compute_cosine_similarity(v1, v2), abs=1e-5) == 0.0

    def test_cosine_similarity_opposite(self):
        """Opposite vectors have cosine similarity of -1."""
        from llama_introspection.geometry import compute_cosine_similarity

        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0])

        assert pytest.approx(compute_cosine_similarity(v1, v2), rel=1e-5) == -1.0


class TestActivationGeometry:
    """Tests for activation geometry metrics used in experiments."""

    def test_compute_pre_post_norms(self):
        """compute_intervention_geometry returns pre/post norms."""
        from llama_introspection.geometry import compute_intervention_geometry

        pre_activation = torch.randn(1, 64)
        post_activation = torch.randn(1, 64)

        metrics = compute_intervention_geometry(pre_activation, post_activation)

        assert "pre_norm" in metrics
        assert "post_norm" in metrics
        assert "l2_distance" in metrics
        assert metrics["pre_norm"] >= 0
        assert metrics["post_norm"] >= 0

    def test_compute_norm_change_ratio(self):
        """compute_intervention_geometry includes norm change ratio."""
        from llama_introspection.geometry import compute_intervention_geometry

        pre = torch.tensor([[1.0, 0.0]])  # norm = 1
        post = torch.tensor([[2.0, 0.0]])  # norm = 2

        metrics = compute_intervention_geometry(pre, post)

        assert "norm_ratio" in metrics
        assert pytest.approx(metrics["norm_ratio"], rel=1e-5) == 2.0
