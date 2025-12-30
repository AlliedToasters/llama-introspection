"""Tests for trajectory module (activation trajectory analysis).

Test-driven development: These tests define the expected API for the trajectory
module before implementation.
"""

import pytest
import torch
import numpy as np
from typing import Optional

# Import will fail until module is created - that's expected in TDD
try:
    from llama_introspection.trajectory import (
        TrajectoryData,
        compute_trajectory_metrics,
        simulate_random_walks,
        capture_trajectory,
    )
    TRAJECTORY_AVAILABLE = True
except ImportError:
    TRAJECTORY_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_states():
    """Create synthetic layer states for testing metric computation.

    Returns 4 layers of hidden states with known properties.
    Creates a "directed walk" where updates are roughly aligned.
    """
    hidden_dim = 64
    n_layers = 4

    # Create states with aligned updates (directed walk pattern)
    # Start with initial state
    states = []
    current = torch.randn(hidden_dim)
    current = current / current.norm() * 10.0  # Start with norm 10
    states.append(current.clone())

    # Create aligned updates - each update is mostly in same direction
    update_direction = torch.randn(hidden_dim)
    update_direction = update_direction / update_direction.norm()

    for i in range(n_layers - 1):
        # Each update is mostly aligned with update_direction, with small noise
        noise = torch.randn(hidden_dim) * 0.1
        update = update_direction * 5.0 + noise  # Strong alignment, small noise
        current = current + update
        states.append(current.clone())

    return states


@pytest.fixture
def orthogonal_states():
    """Create states where consecutive updates are orthogonal (random walk)."""
    hidden_dim = 64
    n_layers = 4

    states = []
    current = torch.randn(hidden_dim)
    current = current / current.norm() * 10.0  # Start with norm 10
    states.append(current.clone())

    for i in range(n_layers - 1):
        # Generate orthogonal update
        update = torch.randn(hidden_dim)
        # Remove component along current direction
        update = update - (update @ current / (current @ current)) * current
        update = update / update.norm() * 5.0  # Fixed update norm
        current = current + update
        states.append(current.clone())

    return states


@pytest.fixture
def tiny_model():
    """Load tiny model for fast testing.

    Uses llamafactory/tiny-random-Llama-3 which has ~few MB of parameters.
    """
    pytest.importorskip("nnsight")
    from nnsight import LanguageModel

    model = LanguageModel(
        "llamafactory/tiny-random-Llama-3",
        device_map="cpu",
    )
    return model


# =============================================================================
# TestTrajectoryData - Dataclass tests
# =============================================================================

@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="trajectory module not yet implemented")
class TestTrajectoryData:
    """Tests for TrajectoryData dataclass."""

    def test_default_initialization(self):
        """TrajectoryData can be created with default values."""
        data = TrajectoryData()
        assert data.state_norms == []
        assert data.update_norms == []
        assert data.update_alignments == []
        assert data.update_state_alignments == []
        assert data.intervention_type == "baseline"
        assert data.intervention_layer == -1
        assert data.intervention_strength == 0.0

    def test_initialization_with_values(self):
        """TrajectoryData can be initialized with specific values."""
        data = TrajectoryData(
            state_norms=[10.0, 15.0, 20.0],
            update_norms=[5.0, 5.0],
            intervention_type="concept",
            intervention_layer=8,
            intervention_strength=1.5,
            concept="ocean",
        )
        assert data.state_norms == [10.0, 15.0, 20.0]
        assert data.intervention_type == "concept"
        assert data.intervention_layer == 8
        assert data.concept == "ocean"

    def test_norm_growth_ratio(self):
        """TrajectoryData computes norm growth ratio correctly."""
        data = TrajectoryData(state_norms=[10.0, 15.0, 20.0])
        assert data.norm_growth_ratio == pytest.approx(2.0)  # 20/10

    def test_norm_growth_ratio_empty(self):
        """Norm growth ratio returns 1.0 for empty/single state."""
        data = TrajectoryData(state_norms=[])
        assert data.norm_growth_ratio == 1.0

        data = TrajectoryData(state_norms=[10.0])
        assert data.norm_growth_ratio == 1.0

    def test_mean_update_alignment(self):
        """TrajectoryData computes mean update alignment."""
        data = TrajectoryData(update_alignments=[0.5, 0.3, 0.4])
        assert data.mean_update_alignment == pytest.approx(0.4)

    def test_mean_update_alignment_empty(self):
        """Mean update alignment returns 0.0 for empty list."""
        data = TrajectoryData(update_alignments=[])
        assert data.mean_update_alignment == 0.0

    def test_perturbation_metrics(self):
        """TrajectoryData stores intervention geometry metrics."""
        data = TrajectoryData(
            intervention_type="concept",
            pre_norm=100.0,
            post_norm=120.0,
            perturbation_norm=25.0,
            perturbation_cosine=0.8,
        )
        assert data.pre_norm == 100.0
        assert data.post_norm == 120.0
        assert data.perturbation_norm == 25.0
        assert data.perturbation_cosine == 0.8


# =============================================================================
# TestComputeTrajectoryMetrics - Pure computation tests
# =============================================================================

@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="trajectory module not yet implemented")
class TestComputeTrajectoryMetrics:
    """Tests for compute_trajectory_metrics function.

    This function takes a list of layer state tensors and returns computed metrics.
    """

    def test_state_norms_computed(self, synthetic_states):
        """State norms are computed for all layers."""
        metrics = compute_trajectory_metrics(synthetic_states)

        assert len(metrics["state_norms"]) == len(synthetic_states)
        # Verify norms are positive
        assert all(n > 0 for n in metrics["state_norms"])
        # Verify they match torch.norm
        for i, state in enumerate(synthetic_states):
            assert metrics["state_norms"][i] == pytest.approx(state.norm().item(), rel=1e-5)

    def test_update_norms_computed(self, synthetic_states):
        """Update norms ||h_{l+1} - h_l|| are computed."""
        metrics = compute_trajectory_metrics(synthetic_states)

        # Should have n_layers - 1 updates
        assert len(metrics["update_norms"]) == len(synthetic_states) - 1

        # Verify computation
        for i in range(len(synthetic_states) - 1):
            expected = (synthetic_states[i + 1] - synthetic_states[i]).norm().item()
            assert metrics["update_norms"][i] == pytest.approx(expected, rel=1e-5)

    def test_update_alignments_parallel(self, synthetic_states):
        """Update alignments are high (~1) for directed/parallel updates."""
        metrics = compute_trajectory_metrics(synthetic_states)

        # Synthetic states have aligned updates, so alignments should be positive
        assert len(metrics["update_alignments"]) == len(synthetic_states) - 2
        # For our synthetic directed states, alignments should be high
        mean_alignment = np.mean(metrics["update_alignments"])
        assert mean_alignment > 0.5, f"Expected high alignment for directed walk, got {mean_alignment}"

    def test_update_alignments_orthogonal(self, orthogonal_states):
        """Update alignments are low (~0) for orthogonal/random updates."""
        metrics = compute_trajectory_metrics(orthogonal_states)

        # Orthogonal updates should have low alignment
        mean_alignment = np.mean(metrics["update_alignments"])
        assert abs(mean_alignment) < 0.3, f"Expected low alignment for orthogonal walk, got {mean_alignment}"

    def test_update_state_alignments_computed(self, synthetic_states):
        """Update-state alignments (cos(u_l, h_l)) are computed."""
        metrics = compute_trajectory_metrics(synthetic_states)

        # Should have n_layers - 1 alignments
        assert len(metrics["update_state_alignments"]) == len(synthetic_states) - 1
        # Values should be in [-1, 1]
        assert all(-1 <= a <= 1 for a in metrics["update_state_alignments"])

    def test_single_state_returns_empty_updates(self):
        """Single state returns empty update lists."""
        states = [torch.randn(64)]
        metrics = compute_trajectory_metrics(states)

        assert len(metrics["state_norms"]) == 1
        assert metrics["update_norms"] == []
        assert metrics["update_alignments"] == []
        assert metrics["update_state_alignments"] == []

    def test_two_states_returns_one_update(self):
        """Two states returns one update, no alignment."""
        states = [torch.randn(64), torch.randn(64)]
        metrics = compute_trajectory_metrics(states)

        assert len(metrics["state_norms"]) == 2
        assert len(metrics["update_norms"]) == 1
        assert metrics["update_alignments"] == []  # Need 3+ states for alignment
        assert len(metrics["update_state_alignments"]) == 1


# =============================================================================
# TestSimulateRandomWalks - Random walk simulation tests
# =============================================================================

@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="trajectory module not yet implemented")
class TestSimulateRandomWalks:
    """Tests for simulate_random_walks function."""

    def test_output_shape(self):
        """Output has shape [n_walks, n_steps + 1]."""
        update_norms = [5.0, 5.0, 5.0, 5.0]  # 4 updates
        result = simulate_random_walks(
            update_norms=update_norms,
            initial_norm=10.0,
            hidden_dim=64,
            n_walks=50,
        )

        assert result.shape == (50, 5)  # 50 walks, 5 points (initial + 4 updates)

    def test_initial_norm_matches(self):
        """First column should be close to initial_norm."""
        result = simulate_random_walks(
            update_norms=[5.0] * 10,
            initial_norm=100.0,
            hidden_dim=1024,
            n_walks=100,
        )

        # Initial norms should all equal initial_norm
        assert np.allclose(result[:, 0], 100.0)

    def test_random_walk_sqrt_growth(self):
        """Random walk norm grows as sqrt(n) for large n."""
        n_steps = 100
        step_size = 10.0
        hidden_dim = 1000  # High dim for orthogonality

        result = simulate_random_walks(
            update_norms=[step_size] * n_steps,
            initial_norm=0.1,  # Start small to see growth clearly
            hidden_dim=hidden_dim,
            n_walks=500,
        )

        # Expected final norm for random walk: sqrt(sum of ||u||^2) = sqrt(n) * step_size
        expected_growth = np.sqrt(n_steps) * step_size
        actual_mean_final = result[:, -1].mean()

        # Should be within 20% of theoretical (some variance expected)
        assert abs(actual_mean_final - expected_growth) / expected_growth < 0.2

    def test_reproducibility_with_seed(self):
        """Same seed produces same results."""
        kwargs = dict(
            update_norms=[5.0] * 10,
            initial_norm=10.0,
            hidden_dim=64,
            n_walks=10,
            seed=42,
        )

        result1 = simulate_random_walks(**kwargs)
        result2 = simulate_random_walks(**kwargs)

        assert np.allclose(result1, result2)


# =============================================================================
# TestCaptureTrajectory - Model-based tests (baseline)
# =============================================================================

@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="trajectory module not yet implemented")
class TestCaptureTrajectory:
    """Tests for capture_trajectory with real (tiny) model."""

    def test_baseline_returns_trajectory_data(self, tiny_model):
        """capture_trajectory returns TrajectoryData for baseline."""
        result = capture_trajectory(
            model=tiny_model,
            prompt="Hello, world!",
            intervention_type="baseline",
        )

        assert isinstance(result, TrajectoryData)
        assert result.intervention_type == "baseline"
        assert len(result.state_norms) > 0
        assert result.n_layers > 0

    def test_captures_all_layers(self, tiny_model):
        """All layer states are captured."""
        result = capture_trajectory(
            model=tiny_model,
            prompt="Test prompt",
            intervention_type="baseline",
        )

        # tiny-random-Llama-3 has a specific number of layers
        # State norms should equal number of layers
        assert len(result.state_norms) == result.n_layers
        assert len(result.update_norms) == result.n_layers - 1

    def test_prompt_stored(self, tiny_model):
        """Prompt is stored in result."""
        prompt = "This is my test prompt"
        result = capture_trajectory(
            model=tiny_model,
            prompt=prompt,
            intervention_type="baseline",
        )

        assert prompt in result.prompt or result.prompt == prompt[:100]

    def test_hidden_dim_stored(self, tiny_model):
        """Hidden dimension is captured."""
        result = capture_trajectory(
            model=tiny_model,
            prompt="Test",
            intervention_type="baseline",
        )

        assert result.hidden_dim > 0


# =============================================================================
# TestCaptureTrajectoryWithIntervention - Intervention tests
# =============================================================================

@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="trajectory module not yet implemented")
class TestCaptureTrajectoryWithIntervention:
    """Tests for capture_trajectory with interventions."""

    def test_concept_intervention_records_metadata(self, tiny_model):
        """Concept intervention records intervention metadata."""
        # Create a simple steering vector
        # First get hidden dim
        baseline = capture_trajectory(
            model=tiny_model,
            prompt="Test",
            intervention_type="baseline",
        )
        hidden_dim = baseline.hidden_dim

        steering_vector = torch.randn(hidden_dim)

        result = capture_trajectory(
            model=tiny_model,
            prompt="Test prompt",
            intervention_type="concept",
            intervention_layer=baseline.n_layers // 2,
            steering_vector=steering_vector,
            intervention_strength=1.0,
        )

        assert result.intervention_type == "concept"
        assert result.intervention_layer == baseline.n_layers // 2
        assert result.intervention_strength == 1.0

    def test_scale_intervention_records_metadata(self, tiny_model):
        """Scale intervention records metadata."""
        baseline = capture_trajectory(
            model=tiny_model,
            prompt="Test",
            intervention_type="baseline",
        )

        result = capture_trajectory(
            model=tiny_model,
            prompt="Test prompt",
            intervention_type="scale",
            intervention_layer=baseline.n_layers // 2,
            intervention_strength=2.0,  # Scale factor
        )

        assert result.intervention_type == "scale"
        assert result.intervention_strength == 2.0

    def test_intervention_changes_trajectory(self, tiny_model):
        """Intervention produces different trajectory than baseline."""
        prompt = "Hello, how are you?"

        baseline = capture_trajectory(
            model=tiny_model,
            prompt=prompt,
            intervention_type="baseline",
        )

        # Strong random intervention should change trajectory
        steering_vector = torch.randn(baseline.hidden_dim) * 100

        intervened = capture_trajectory(
            model=tiny_model,
            prompt=prompt,
            intervention_type="random",
            intervention_layer=baseline.n_layers // 2,
            steering_vector=steering_vector,
            intervention_strength=1.0,
        )

        # Post-intervention norms should differ
        # Check layer after intervention
        int_layer = baseline.n_layers // 2
        if int_layer < len(baseline.state_norms) and int_layer < len(intervened.state_norms):
            baseline_norm = baseline.state_norms[int_layer]
            intervened_norm = intervened.state_norms[int_layer]
            # They should be different (intervention had effect)
            assert baseline_norm != pytest.approx(intervened_norm, rel=0.01), \
                "Intervention should change activation norms"

    def test_intervention_geometry_captured(self, tiny_model):
        """Pre/post norms and perturbation metrics are captured."""
        baseline = capture_trajectory(
            model=tiny_model,
            prompt="Test",
            intervention_type="baseline",
        )

        steering_vector = torch.randn(baseline.hidden_dim)

        result = capture_trajectory(
            model=tiny_model,
            prompt="Test prompt",
            intervention_type="concept",
            intervention_layer=baseline.n_layers // 2,
            steering_vector=steering_vector,
            intervention_strength=1.0,
        )

        # Should capture intervention geometry
        assert result.pre_norm > 0
        assert result.post_norm > 0
        assert result.perturbation_norm >= 0
        assert -1 <= result.perturbation_cosine <= 1

    def test_random_intervention_type(self, tiny_model):
        """Random intervention uses random vector."""
        baseline = capture_trajectory(
            model=tiny_model,
            prompt="Test",
            intervention_type="baseline",
        )

        # Random intervention generates its own vector
        result = capture_trajectory(
            model=tiny_model,
            prompt="Test prompt",
            intervention_type="random",
            intervention_layer=baseline.n_layers // 2,
            intervention_strength=1.0,
            hidden_dim=baseline.hidden_dim,  # Needed to generate random vector
        )

        assert result.intervention_type == "random"


# =============================================================================
# Integration tests
# =============================================================================

@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="trajectory module not yet implemented")
class TestTrajectoryIntegration:
    """Integration tests combining multiple trajectory functions."""

    def test_metrics_match_trajectory_data(self, synthetic_states):
        """compute_trajectory_metrics output matches TrajectoryData fields."""
        metrics = compute_trajectory_metrics(synthetic_states)

        data = TrajectoryData(
            state_norms=metrics["state_norms"],
            update_norms=metrics["update_norms"],
            update_alignments=metrics["update_alignments"],
            update_state_alignments=metrics["update_state_alignments"],
        )

        assert data.state_norms == metrics["state_norms"]
        assert data.norm_growth_ratio > 1.0  # Directed walk grows

    def test_baseline_vs_random_walk_comparison(self, tiny_model):
        """Baseline trajectory can be compared to simulated random walk."""
        baseline = capture_trajectory(
            model=tiny_model,
            prompt="Explain quantum physics",
            intervention_type="baseline",
        )

        # Simulate random walks with same update norms
        random_walks = simulate_random_walks(
            update_norms=baseline.update_norms,
            initial_norm=baseline.state_norms[0],
            hidden_dim=baseline.hidden_dim,
            n_walks=100,
        )

        # Real trajectory should exist and be comparable
        assert len(baseline.state_norms) == random_walks.shape[1]
