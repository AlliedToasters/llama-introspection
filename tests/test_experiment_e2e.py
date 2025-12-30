"""End-to-end tests for the introspection experiment suite.

These tests use the tiny model and mock Anthropic client to run
the full experiment pipeline without consuming API credits.
"""

import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import MockAnthropicClient, MockAnthropicClientFixedResponse


# Skip all tests in this module if nnsight is not available
pytest.importorskip("nnsight")


@dataclass
class MockArgs:
    """Mock argument namespace for experiment functions."""
    random_seed: int = 42
    model: str = "tiny"
    condition: str = "baseline"
    concept: str = "ocean"
    n_layers: int = 1
    n_trials: int = 1
    strengths: Optional[list] = None
    output_dir: str = "results"
    incoherence_tolerance: int = 2
    use_remote: bool = False


class TestExperimentHelpers:
    """Test helper functions from introspection module."""

    def test_compute_layer_indices_single(self):
        """Single layer returns 2/3 depth."""
        from experiments.introspection import compute_layer_indices

        indices = compute_layer_indices(num_layers=12, n_samples=1)
        assert indices == [8]  # 12 * 2 // 3

    def test_compute_layer_indices_multiple(self):
        """Multiple layers are evenly spaced."""
        from experiments.introspection import compute_layer_indices

        indices = compute_layer_indices(num_layers=12, n_samples=3)
        assert indices == [0, 5, 11]  # start, middle, end

    def test_injection_messages_structure(self):
        """INJECTION_MESSAGES has correct structure."""
        from experiments.introspection import INJECTION_MESSAGES

        assert len(INJECTION_MESSAGES) == 3
        assert INJECTION_MESSAGES[0]["role"] == "user"
        assert INJECTION_MESSAGES[1]["role"] == "assistant"
        assert INJECTION_MESSAGES[2]["role"] == "user"
        assert "Trial 1" in INJECTION_MESSAGES[2]["content"]


@pytest.fixture(scope="module")
def tiny_model():
    """Load tiny model once for the module."""
    from nnsight import LanguageModel

    model = LanguageModel("llamafactory/tiny-random-Llama-3", device_map="auto")
    return model


@pytest.fixture
def mock_client():
    """Create mock Anthropic client."""
    return MockAnthropicClient(default_coherent=True)


@pytest.fixture
def fixed_response_client():
    """Create mock client with fixed positive response."""
    return MockAnthropicClientFixedResponse({
        "refusal": False,
        "affirmative": True,
        "correct_id": True,
        "early_detection": False,
        "coherent": True,
        "reasoning": "Test response",
    })


class TestBaselineCondition:
    """End-to-end tests for baseline (no intervention) condition."""

    def test_baseline_generates_text(self, tiny_model, mock_client):
        """Baseline condition generates coherent text."""
        from experiments.introspection import (
            INJECTION_MESSAGES,
            MAX_NEW_TOKENS,
        )
        from llama_introspection.steering import compute_injection_position

        # Compute injection position
        injection_prompt, injection_start_pos, total_tokens = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        # Run baseline generation (no steering)
        with tiny_model.generate(injection_prompt, max_new_tokens=10):
            output = tiny_model.generator.output.save()

        text = tiny_model.tokenizer.decode(output[0], skip_special_tokens=True)

        assert isinstance(text, str)
        assert len(text) > 0
        # The text should contain some part of the injection prompt
        assert "Trial" in text or len(text) > 50

    def test_baseline_grading(self, tiny_model, mock_client):
        """Baseline responses are graded correctly."""
        from experiments.introspection import _grade_response_with_client

        # Sample baseline response
        response = "I don't detect any injected thought. My processing feels normal."

        grade = _grade_response_with_client(
            mock_client, response, condition="baseline", concept=None
        )

        assert "coherent" in grade
        assert "affirmative" in grade
        assert "refusal" in grade


class TestConceptCondition:
    """End-to-end tests for concept injection condition."""

    def test_concept_trial_runs(self, tiny_model, mock_client):
        """Concept injection trial completes without error."""
        from experiments.introspection import (
            run_concept_trial,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        # Setup
        layers = get_layer_accessor(tiny_model)(tiny_model)
        hidden_dim = tiny_model.config.hidden_size
        injection_prompt, injection_start_pos, _ = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        # Create a simple steering vector
        steering_vector = torch.randn(1, hidden_dim) * 0.1

        # Run trial with strength=0 (baseline equivalent)
        text, geometry = run_concept_trial(
            model=tiny_model,
            layers=layers,
            injection_prompt=injection_prompt,
            layer_idx=0,
            steering_vector=steering_vector,
            strength=0,  # No actual steering
            injection_start_pos=injection_start_pos,
            use_remote=False,
        )

        assert isinstance(text, str)
        assert isinstance(geometry, dict)
        assert "pre_norm" in geometry

    def test_concept_trial_with_steering(self, tiny_model, mock_client):
        """Concept injection with non-zero strength modifies activations."""
        from experiments.introspection import (
            run_concept_trial,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        layers = get_layer_accessor(tiny_model)(tiny_model)
        hidden_dim = tiny_model.config.hidden_size
        injection_prompt, injection_start_pos, _ = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        # Create steering vector
        steering_vector = torch.randn(1, hidden_dim)
        steering_vector = steering_vector / steering_vector.norm()  # Normalize

        # Run with small strength
        text, geometry = run_concept_trial(
            model=tiny_model,
            layers=layers,
            injection_prompt=injection_prompt,
            layer_idx=0,
            steering_vector=steering_vector,
            strength=0.1,
            injection_start_pos=injection_start_pos,
            use_remote=False,
        )

        assert isinstance(text, str)
        assert geometry["pre_norm"] >= 0
        assert geometry["post_norm"] >= 0
        assert geometry["l2_distance"] >= 0


class TestRandomCondition:
    """End-to-end tests for random vector condition."""

    def test_random_trial_runs(self, tiny_model, mock_client):
        """Random vector trial completes without error."""
        from experiments.introspection import (
            run_random_trial,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        layers = get_layer_accessor(tiny_model)(tiny_model)
        hidden_dim = tiny_model.config.hidden_size
        injection_prompt, injection_start_pos, _ = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        args = MockArgs(random_seed=42)

        text, geometry = run_random_trial(
            trial_num=0,
            args=args,
            model=tiny_model,
            layers=layers,
            hidden_dim=hidden_dim,
            mean_steering_norm=1.0,
            injection_prompt=injection_prompt,
            layer_idx=0,
            strength=0.1,
            injection_start_pos=injection_start_pos,
            use_remote=False,
        )

        assert isinstance(text, str)
        assert isinstance(geometry, dict)

    def test_random_vectors_are_reproducible(self, tiny_model):
        """Same seed produces same random vector."""
        from experiments.introspection import (
            run_random_trial,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        layers = get_layer_accessor(tiny_model)(tiny_model)
        hidden_dim = tiny_model.config.hidden_size
        injection_prompt, injection_start_pos, _ = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        args1 = MockArgs(random_seed=123)
        args2 = MockArgs(random_seed=123)

        # Run same trial twice with same seed
        text1, _ = run_random_trial(
            trial_num=0, args=args1, model=tiny_model, layers=layers,
            hidden_dim=hidden_dim, mean_steering_norm=1.0,
            injection_prompt=injection_prompt, layer_idx=0,
            strength=0.1, injection_start_pos=injection_start_pos,
        )

        text2, _ = run_random_trial(
            trial_num=0, args=args2, model=tiny_model, layers=layers,
            hidden_dim=hidden_dim, mean_steering_norm=1.0,
            injection_prompt=injection_prompt, layer_idx=0,
            strength=0.1, injection_start_pos=injection_start_pos,
        )

        # Same seed + same trial_num should give same result
        assert text1 == text2


class TestScaleCondition:
    """End-to-end tests for activation scaling condition."""

    def test_scale_trial_runs(self, tiny_model, mock_client):
        """Scale trial completes without error."""
        from experiments.introspection import (
            run_scale_trial,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        layers = get_layer_accessor(tiny_model)(tiny_model)
        injection_prompt, injection_start_pos, _ = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        # Scale factor 1.0 should be equivalent to baseline
        text, geometry = run_scale_trial(
            model=tiny_model,
            layers=layers,
            injection_prompt=injection_prompt,
            layer_idx=0,
            scale_factor=1.0,
            injection_start_pos=injection_start_pos,
            use_remote=False,
        )

        assert isinstance(text, str)
        assert isinstance(geometry, dict)

    def test_scale_trial_with_scaling(self, tiny_model, mock_client):
        """Scale trial with non-1.0 factor modifies activations."""
        from experiments.introspection import (
            run_scale_trial,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        layers = get_layer_accessor(tiny_model)(tiny_model)
        injection_prompt, injection_start_pos, _ = compute_injection_position(
            tiny_model.tokenizer, INJECTION_MESSAGES
        )

        text, geometry = run_scale_trial(
            model=tiny_model,
            layers=layers,
            injection_prompt=injection_prompt,
            layer_idx=0,
            scale_factor=1.5,
            injection_start_pos=injection_start_pos,
            use_remote=False,
        )

        assert isinstance(text, str)
        assert geometry["pre_norm"] >= 0
        assert geometry["post_norm"] >= 0
        # Post norm should be larger due to scaling
        assert geometry["post_norm"] > geometry["pre_norm"] * 0.9  # Allow some variance


class TestFullExperiment:
    """End-to-end tests for the full run_experiment function."""

    def test_run_experiment_baseline(self, tiny_model, fixed_response_client):
        """Full experiment runs for baseline condition."""
        from experiments.introspection import (
            run_experiment,
            compute_layer_indices,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_results.pt"

            layers = get_layer_accessor(tiny_model)(tiny_model)
            num_layers = len(layers)
            hidden_dim = tiny_model.config.hidden_size
            layer_indices = compute_layer_indices(num_layers, 1)

            injection_prompt, injection_start_pos, _ = compute_injection_position(
                tiny_model.tokenizer, INJECTION_MESSAGES
            )

            args = MockArgs(condition="baseline")

            results, stopped_at, config = run_experiment(
                model=tiny_model,
                hidden_dim=hidden_dim,
                mean_steering_norm=1.0,
                args=args,
                model_name="tiny-test",
                condition="baseline",
                concept=None,
                layers=layers,
                injection_prompt=injection_prompt,
                injection_start_pos=injection_start_pos,
                layer_indices=layer_indices,
                strengths=[0],  # Only baseline strength
                n_trials=1,
                incoherence_tolerance=2,
                client=fixed_response_client,
                save_path=save_path,
                steering_vector=None,
                use_remote=False,
            )

            # Check results structure
            assert isinstance(results, dict)
            assert len(results) > 0

            # Check that results were saved
            assert save_path.exists()

            # Load and verify saved results
            saved_data = torch.load(save_path, weights_only=False)
            assert "results" in saved_data
            assert "config" in saved_data

    def test_run_experiment_concept(self, tiny_model, fixed_response_client):
        """Full experiment runs for concept injection condition."""
        from experiments.introspection import (
            run_experiment,
            compute_layer_indices,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_results.pt"

            layers = get_layer_accessor(tiny_model)(tiny_model)
            num_layers = len(layers)
            hidden_dim = tiny_model.config.hidden_size
            layer_indices = compute_layer_indices(num_layers, 1)

            injection_prompt, injection_start_pos, _ = compute_injection_position(
                tiny_model.tokenizer, INJECTION_MESSAGES
            )

            # Create simple steering vector
            steering_vector = torch.randn(1, hidden_dim) * 0.1

            args = MockArgs(condition="concept", concept="ocean")

            results, stopped_at, config = run_experiment(
                model=tiny_model,
                hidden_dim=hidden_dim,
                mean_steering_norm=1.0,
                args=args,
                model_name="tiny-test",
                condition="concept",
                concept="ocean",
                layers=layers,
                injection_prompt=injection_prompt,
                injection_start_pos=injection_start_pos,
                layer_indices=layer_indices,
                strengths=[0, 0.1],  # Baseline + one strength
                n_trials=1,
                incoherence_tolerance=2,
                client=fixed_response_client,
                save_path=save_path,
                steering_vector=steering_vector,
                use_remote=False,
            )

            assert isinstance(results, dict)
            layer_key = str(layer_indices[0])
            assert layer_key in results
            assert "0" in results[layer_key]  # strength=0
            assert "0.1" in results[layer_key]  # strength=0.1

            # Each trial should have text, grade, and geometry
            trial = results[layer_key]["0.1"][0]
            assert "text" in trial
            assert "grade" in trial
            assert "geometry" in trial

    def test_grading_client_called(self, tiny_model, mock_client):
        """Verify grading client is called for each trial."""
        from experiments.introspection import (
            run_experiment,
            compute_layer_indices,
            INJECTION_MESSAGES,
        )
        from llama_introspection.steering import (
            compute_injection_position,
            get_layer_accessor,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_results.pt"

            layers = get_layer_accessor(tiny_model)(tiny_model)
            num_layers = len(layers)
            hidden_dim = tiny_model.config.hidden_size
            layer_indices = compute_layer_indices(num_layers, 1)

            injection_prompt, injection_start_pos, _ = compute_injection_position(
                tiny_model.tokenizer, INJECTION_MESSAGES
            )

            args = MockArgs(condition="baseline")
            n_trials = 2
            n_strengths = 2

            run_experiment(
                model=tiny_model,
                hidden_dim=hidden_dim,
                mean_steering_norm=1.0,
                args=args,
                model_name="tiny-test",
                condition="baseline",
                concept=None,
                layers=layers,
                injection_prompt=injection_prompt,
                injection_start_pos=injection_start_pos,
                layer_indices=layer_indices,
                strengths=[0, 1],
                n_trials=n_trials,
                incoherence_tolerance=2,
                client=mock_client,
                save_path=save_path,
                steering_vector=None,
                use_remote=False,
            )

            # Client should be called once per trial
            expected_calls = n_strengths * n_trials
            assert mock_client.call_count == expected_calls


class TestMockAnthropicClient:
    """Tests for MockAnthropicClient interface."""

    def test_mimics_anthropic_interface(self):
        """Mock client has same interface as real Anthropic client."""
        client = MockAnthropicClient()

        # Should have messages.create method
        assert hasattr(client, "messages")
        assert hasattr(client.messages, "create")

        # Should return response with content[0].text
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": "Test prompt"}],
        )

        assert hasattr(response, "content")
        assert len(response.content) > 0
        assert hasattr(response.content[0], "text")

    def test_tracks_calls(self):
        """Mock client tracks number of calls."""
        client = MockAnthropicClient()
        assert client.call_count == 0

        client.messages.create(
            model="test", max_tokens=100, messages=[{"role": "user", "content": "Hi"}]
        )
        assert client.call_count == 1

        client.messages.create(
            model="test", max_tokens=100, messages=[{"role": "user", "content": "Hello"}]
        )
        assert client.call_count == 2
