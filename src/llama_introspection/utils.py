"""Shared utility functions for LLM experiments.

Provides:
- KL divergence computation
- Token transition analysis
"""

from typing import Dict, List

import numpy as np
import torch


def compute_kl_divergence(
    logits_base: torch.Tensor,
    logits_steered: torch.Tensor,
    p_top: float = 1.0,
) -> float:
    """Compute KL(steered || base) for next-token distributions.

    Args:
        logits_base: Logits from unsteered model
        logits_steered: Logits from steered model
        p_top: Nucleus threshold - include tokens that make up top p_top
               cumulative probability in either distribution.
               Set to 1.0 for raw KL over full vocabulary.

    Returns:
        KL divergence value (float)
    """
    # Convert to probabilities
    p_base = torch.softmax(logits_base, dim=-1)
    p_steered = torch.softmax(logits_steered, dim=-1)

    if p_top < 1.0:
        # Nucleus selection for base distribution
        sorted_base, indices_base = torch.sort(p_base, descending=True)
        cumsum_base = torch.cumsum(sorted_base, dim=-1)
        nucleus_mask_base = cumsum_base <= p_top
        nucleus_mask_base[0] = True  # Always include at least the top token
        nucleus_indices_base = indices_base[nucleus_mask_base]

        # Nucleus selection for steered distribution
        sorted_steered, indices_steered = torch.sort(p_steered, descending=True)
        cumsum_steered = torch.cumsum(sorted_steered, dim=-1)
        nucleus_mask_steered = cumsum_steered <= p_top
        nucleus_mask_steered[0] = True
        nucleus_indices_steered = indices_steered[nucleus_mask_steered]

        # Union of both nucleus sets
        union_indices = torch.unique(
            torch.cat([nucleus_indices_base, nucleus_indices_steered])
        )

        # Select only the union tokens
        p_base = p_base[union_indices]
        p_steered = p_steered[union_indices]

        # Renormalize to valid probability distributions
        p_base = p_base / p_base.sum()
        p_steered = p_steered / p_steered.sum()

    # KL divergence
    eps = 1e-10  # Numerical stability
    kl = (p_steered * (torch.log(p_steered + eps) - torch.log(p_base + eps))).sum()

    return kl.item()


def compute_kl_curves(
    all_logits: torch.Tensor,
    alpha_values: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute KL divergence curves from pre-computed logits.

    Args:
        all_logits: [num_alphas, vocab_size] tensor of logits
        alpha_values: array of alpha values corresponding to each row

    Returns:
        dict with 'full' and 'nucleus' KL arrays
    """
    alpha_values = np.array(alpha_values)

    # Find baseline (α closest to 0)
    idx_zero = np.argmin(np.abs(alpha_values))
    logits_base = all_logits[idx_zero]

    kl_full = []
    kl_nucleus = []

    for i in range(len(alpha_values)):
        kl_full.append(compute_kl_divergence(logits_base, all_logits[i], p_top=1.0))
        kl_nucleus.append(compute_kl_divergence(logits_base, all_logits[i], p_top=0.4))

    return {"full": np.array(kl_full), "nucleus": np.array(kl_nucleus)}


def get_token_transitions(
    all_logits: torch.Tensor,
    alpha_values: np.ndarray,
    tokenizer,
) -> List[Dict]:
    """Find where the top predicted token changes along the alpha sweep.

    Args:
        all_logits: [num_alphas, vocab_size] tensor
        alpha_values: array of alpha values
        tokenizer: model tokenizer for decoding

    Returns:
        List of dicts with 'idx', 'alpha', 'token_id', 'token_str' at each transition
    """
    top_tokens = all_logits.argmax(dim=-1)  # [num_alphas]

    transitions = []
    prev_token = None

    for i, (alpha, token_id) in enumerate(zip(alpha_values, top_tokens)):
        token_id = token_id.item()
        if token_id != prev_token:
            token_str = tokenizer.decode([token_id]).strip()
            if not token_str:
                token_str = repr(tokenizer.decode([token_id]))
            transitions.append(
                {
                    "idx": i,
                    "alpha": alpha,
                    "token_id": token_id,
                    "token_str": token_str,
                }
            )
            prev_token = token_id

    return transitions


# Re-export compute_injection_position from steering for convenience
from llama_introspection.steering import compute_injection_position

__all__ = [
    "compute_kl_divergence",
    "compute_kl_curves",
    "get_token_transitions",
    "compute_injection_position",
]
