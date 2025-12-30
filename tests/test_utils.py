"""Tests for shared utilities (src/llama_introspection/utils.py)."""

import pytest
import torch


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions is 0."""
        from llama_introspection.utils import compute_kl_divergence

        logits = torch.tensor([1.0, 2.0, 3.0])
        kl = compute_kl_divergence(logits, logits)

        assert pytest.approx(kl, abs=1e-5) == 0.0

    def test_kl_divergence_positive(self):
        """KL divergence is non-negative."""
        from llama_introspection.utils import compute_kl_divergence

        logits_base = torch.tensor([1.0, 2.0, 3.0])
        logits_steered = torch.tensor([3.0, 2.0, 1.0])

        kl = compute_kl_divergence(logits_base, logits_steered)

        assert kl >= 0.0

    def test_kl_divergence_asymmetric(self):
        """KL(P||Q) != KL(Q||P) in general."""
        from llama_introspection.utils import compute_kl_divergence

        # Use asymmetric distributions to show KL is asymmetric
        logits_a = torch.tensor([5.0, 1.0, 0.1])  # peaked at first token
        logits_b = torch.tensor([1.0, 1.0, 1.0])  # uniform

        kl_ab = compute_kl_divergence(logits_a, logits_b)
        kl_ba = compute_kl_divergence(logits_b, logits_a)

        # KL divergence is asymmetric
        assert kl_ab != kl_ba


class TestNucleusKL:
    """Tests for nucleus-filtered KL divergence."""

    def test_nucleus_kl_subset(self):
        """Nucleus KL only considers top-p probability mass."""
        from llama_introspection.utils import compute_kl_divergence

        # Create logits where most probability is in first few tokens
        logits_base = torch.tensor([10.0, 9.0, 1.0, 1.0, 1.0])
        logits_steered = torch.tensor([9.0, 10.0, 1.0, 1.0, 1.0])

        kl_full = compute_kl_divergence(logits_base, logits_steered, p_top=1.0)
        kl_nucleus = compute_kl_divergence(logits_base, logits_steered, p_top=0.9)

        # Both should be computed without error
        assert kl_full >= 0
        assert kl_nucleus >= 0


class TestInjectionPosition:
    """Tests for injection position computation (re-exported from steering)."""

    def test_compute_injection_position(self, tiny_model):
        """compute_injection_position finds correct token position."""
        from llama_introspection.utils import compute_injection_position

        messages = [{"role": "user", "content": "Test message\n\nTrial 1: Begin"}]

        prompt, pos, total = compute_injection_position(
            tiny_model.tokenizer, messages, marker="\n\nTrial 1"
        )

        assert isinstance(prompt, str)
        assert isinstance(pos, int)
        assert isinstance(total, int)
        assert 0 < pos < total

    def test_injection_position_marker_not_found(self, tiny_model):
        """compute_injection_position raises on missing marker."""
        from llama_introspection.utils import compute_injection_position

        messages = [{"role": "user", "content": "No marker here"}]

        with pytest.raises(ValueError, match="Could not find"):
            compute_injection_position(
                tiny_model.tokenizer, messages, marker="NONEXISTENT"
            )
