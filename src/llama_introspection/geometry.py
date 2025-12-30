"""Vector geometry computations for activation analysis.

Provides utilities for computing norms, distances, and similarity metrics
on activation vectors, used for analyzing steering interventions.
"""

from typing import Dict, Optional, Union

import torch


def compute_l2_norm(
    v: torch.Tensor,
    dim: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """Compute L2 (Euclidean) norm of a vector or batch of vectors.

    Args:
        v: Input tensor
        dim: Dimension along which to compute norm. If None, computes
             norm of flattened tensor and returns a scalar.

    Returns:
        L2 norm as float (if dim is None) or tensor of norms
    """
    if dim is None:
        return v.norm().item()
    return torch.norm(v, p=2, dim=dim)


def compute_l2_distance(
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> float:
    """Compute L2 (Euclidean) distance between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        L2 distance as float
    """
    return (v1 - v2).norm().item()


def compute_cosine_similarity(
    v1: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector
        eps: Small epsilon for numerical stability

    Returns:
        Cosine similarity in range [-1, 1]
    """
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()

    dot_product = (v1_flat * v2_flat).sum()
    norm_product = v1_flat.norm() * v2_flat.norm()

    if norm_product < eps:
        return 0.0

    return (dot_product / norm_product).item()


def compute_intervention_geometry(
    pre_activation: torch.Tensor,
    post_activation: torch.Tensor,
) -> Dict[str, float]:
    """Compute geometry metrics for a steering intervention.

    Computes various metrics comparing activations before and after
    an intervention, useful for analyzing the effect of steering vectors.

    Args:
        pre_activation: Activation tensor before intervention
        post_activation: Activation tensor after intervention

    Returns:
        Dictionary with keys:
        - pre_norm: L2 norm of pre-intervention activation
        - post_norm: L2 norm of post-intervention activation
        - l2_distance: L2 distance between pre and post activations
        - norm_ratio: Ratio of post_norm to pre_norm
        - cosine_sim: Cosine similarity between pre and post
    """
    pre_norm = pre_activation.norm().item()
    post_norm = post_activation.norm().item()
    l2_dist = (post_activation - pre_activation).norm().item()

    # Compute norm ratio (handle zero pre_norm)
    if pre_norm > 1e-8:
        norm_ratio = post_norm / pre_norm
    else:
        norm_ratio = float("inf") if post_norm > 1e-8 else 1.0

    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(pre_activation, post_activation)

    return {
        "pre_norm": pre_norm,
        "post_norm": post_norm,
        "l2_distance": l2_dist,
        "norm_ratio": norm_ratio,
        "cosine_sim": cosine_sim,
    }


def compute_trajectory_stats(
    activations: torch.Tensor,
    dim: int = -1,
) -> Dict[str, torch.Tensor]:
    """Compute statistics on activation norms across a trajectory.

    Useful for analyzing how activation norms evolve across layers.

    Args:
        activations: Tensor of shape [..., hidden_dim] or [layers, ..., hidden_dim]
        dim: Dimension along which to compute norms

    Returns:
        Dictionary with mean, std, min, max of norms
    """
    norms = torch.norm(activations, p=2, dim=dim)

    return {
        "mean": norms.mean(),
        "std": norms.std(),
        "min": norms.min(),
        "max": norms.max(),
    }


__all__ = [
    "compute_l2_norm",
    "compute_l2_distance",
    "compute_cosine_similarity",
    "compute_intervention_geometry",
    "compute_trajectory_stats",
]
