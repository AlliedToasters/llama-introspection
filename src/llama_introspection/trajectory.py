"""Trajectory analysis for activation dynamics through layers.

This module provides tools for capturing and analyzing how activations
evolve through transformer layers, with support for interventions
(steering vector injection, scaling).

Key concepts:
- State norms: ||h_l|| at each layer
- Update norms: ||h_{l+1} - h_l|| (step sizes)
- Update alignments: cos(u_l, u_{l+1}) (directedness measure)
- Update-state alignments: cos(u_l, h_l) (overwriting measure)

The "directed vs random walk" hypothesis suggests LLM trajectories
exhibit aligned updates (directed walk), leading to faster norm growth
than random walks with the same step sizes.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrajectoryData:
    """Data from a single forward pass / trajectory through layers.

    Captures activation trajectory metrics and optional intervention metadata.
    """

    # Core trajectory metrics
    state_norms: list[float] = field(default_factory=list)
    update_norms: list[float] = field(default_factory=list)
    update_alignments: list[float] = field(default_factory=list)
    update_state_alignments: list[float] = field(default_factory=list)

    # Intervention metadata
    intervention_type: str = "baseline"  # "baseline", "concept", "random", "scale"
    intervention_layer: int = -1
    intervention_strength: float = 0.0
    concept: Optional[str] = None

    # Intervention geometry (at intervention layer)
    pre_norm: float = 0.0  # Norm before intervention
    post_norm: float = 0.0  # Norm after intervention
    perturbation_norm: float = 0.0  # ||post - pre||
    perturbation_cosine: float = 1.0  # cos(post - pre, pre)
    steering_vector_norm: float = 0.0

    # Model metadata
    n_layers: int = 0
    hidden_dim: int = 0
    prompt: str = ""
    response: str = ""

    @property
    def norm_growth_ratio(self) -> float:
        """Ratio of final to initial state norm."""
        if len(self.state_norms) < 2:
            return 1.0
        return self.state_norms[-1] / self.state_norms[0]

    @property
    def mean_update_alignment(self) -> float:
        """Mean alignment between consecutive updates."""
        if not self.update_alignments:
            return 0.0
        return sum(self.update_alignments) / len(self.update_alignments)


def compute_trajectory_metrics(states: list[torch.Tensor]) -> dict:
    """Compute trajectory metrics from a list of layer state tensors.

    Args:
        states: List of tensors, one per layer. Each should be [hidden_dim]
                or [1, hidden_dim] (will be squeezed).

    Returns:
        Dict with:
        - state_norms: [n_layers] norms of each state
        - update_norms: [n_layers - 1] norms of updates
        - update_alignments: [n_layers - 2] cosines between consecutive updates
        - update_state_alignments: [n_layers - 1] cosines between update and state
    """
    # Ensure states are 1D tensors
    states = [s.squeeze() for s in states]

    result = {
        "state_norms": [],
        "update_norms": [],
        "update_alignments": [],
        "update_state_alignments": [],
    }

    if not states:
        return result

    # State norms
    result["state_norms"] = [s.norm().item() for s in states]

    if len(states) < 2:
        return result

    # Compute updates
    updates = [states[i + 1] - states[i] for i in range(len(states) - 1)]

    # Update norms
    result["update_norms"] = [u.norm().item() for u in updates]

    # Update-state alignments: cos(u_l, h_l)
    for i, (update, state) in enumerate(zip(updates, states[:-1])):
        cos_sim = F.cosine_similarity(
            update.unsqueeze(0), state.unsqueeze(0)
        ).item()
        result["update_state_alignments"].append(cos_sim)

    # Update alignments: cos(u_l, u_{l+1})
    if len(updates) >= 2:
        for i in range(len(updates) - 1):
            cos_sim = F.cosine_similarity(
                updates[i].unsqueeze(0), updates[i + 1].unsqueeze(0)
            ).item()
            result["update_alignments"].append(cos_sim)

    return result


def simulate_random_walks(
    update_norms: list[float],
    initial_norm: float,
    hidden_dim: int,
    n_walks: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate random walks with the same step sizes as a real trajectory.

    In high dimensions, random vectors are nearly orthogonal, so:
    ||h + u||^2 ≈ ||h||^2 + ||u||^2

    This leads to sqrt(n) growth for n steps, vs linear growth for
    aligned (directed) updates.

    Args:
        update_norms: Step sizes to use (from real trajectory)
        initial_norm: Starting state norm
        hidden_dim: Dimensionality of state space
        n_walks: Number of random walks to simulate
        seed: Random seed for reproducibility

    Returns:
        Array of shape [n_walks, n_steps + 1] with norm at each step
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = len(update_norms)
    trajectories = np.zeros((n_walks, n_steps + 1))

    for w in range(n_walks):
        # Start with initial state (random direction, given norm)
        h = np.random.randn(hidden_dim)
        h = h / np.linalg.norm(h) * initial_norm
        trajectories[w, 0] = np.linalg.norm(h)

        # Take steps with random directions but real norms
        for i, update_norm in enumerate(update_norms):
            # Random unit vector
            u = np.random.randn(hidden_dim)
            u = u / np.linalg.norm(u) * update_norm

            h = h + u
            trajectories[w, i + 1] = np.linalg.norm(h)

    return trajectories


def get_layer_accessor(model):
    """Return appropriate layer accessor based on model architecture."""
    model_name = type(model._model).__name__.lower() if hasattr(model, "_model") else ""
    if "gpt2" in model_name:
        return lambda m: m._model.h
    else:
        return lambda m: m.model.layers


def capture_trajectory(
    model,
    prompt: str,
    intervention_type: str = "baseline",
    intervention_layer: int = -1,
    steering_vector: Optional[torch.Tensor] = None,
    intervention_strength: float = 0.0,
    hidden_dim: Optional[int] = None,
    use_remote: bool = False,
    concept: Optional[str] = None,
) -> TrajectoryData:
    """Capture activation trajectory through all layers.

    Supports several intervention types:
    - "baseline": No intervention, just capture trajectory
    - "concept": Add steering_vector * strength at intervention_layer
    - "random": Add random vector * strength at intervention_layer
    - "scale": Multiply activations by strength at intervention_layer

    Args:
        model: nnsight LanguageModel
        prompt: Input prompt
        intervention_type: One of "baseline", "concept", "random", "scale"
        intervention_layer: Layer index for intervention (-1 = no intervention)
        steering_vector: Vector to add (for "concept" type)
        intervention_strength: Multiplier for intervention
        hidden_dim: Hidden dimension (needed for "random" type if generating vector)
        use_remote: Use NDIF remote execution
        concept: Concept name (for metadata)

    Returns:
        TrajectoryData with trajectory metrics and intervention metadata
    """
    # Get layer accessor
    layers = get_layer_accessor(model)(model)
    num_layers = len(layers)

    # Initialize result
    result = TrajectoryData(
        n_layers=num_layers,
        prompt=prompt[:100],
        intervention_type=intervention_type,
        intervention_layer=intervention_layer,
        intervention_strength=intervention_strength,
        concept=concept,
    )

    # Handle random vector generation
    if intervention_type == "random" and steering_vector is None:
        if hidden_dim is None:
            raise ValueError("hidden_dim required for random intervention without steering_vector")
        steering_vector = torch.randn(hidden_dim)
        steering_vector = steering_vector / steering_vector.norm()  # Normalize

    if steering_vector is not None:
        result.steering_vector_norm = steering_vector.norm().item()

    # Capture trajectory
    if intervention_type == "baseline" or intervention_layer < 0:
        # Simple trace without intervention
        result = _capture_baseline(model, layers, num_layers, prompt, use_remote, result)
    else:
        # Trace with intervention
        result = _capture_with_intervention(
            model, layers, num_layers, prompt, use_remote, result,
            intervention_type, intervention_layer, steering_vector, intervention_strength
        )

    return result


def _capture_baseline(
    model,
    layers,
    num_layers: int,
    prompt: str,
    use_remote: bool,
    result: TrajectoryData,
) -> TrajectoryData:
    """Capture baseline trajectory without intervention."""
    with model.trace(remote=use_remote) as tracer:
        layer_outputs = list().save()

        with tracer.invoke(prompt):
            for l in range(num_layers):
                layer_outputs.append(layers[l].output[:, -1, :])

    # Process captured activations
    states = [lo.detach().cpu().squeeze(0) for lo in layer_outputs]
    result.hidden_dim = states[0].shape[-1]
    result.response = "(trace mode - no generation)"

    # Compute metrics
    metrics = compute_trajectory_metrics(states)
    result.state_norms = metrics["state_norms"]
    result.update_norms = metrics["update_norms"]
    result.update_alignments = metrics["update_alignments"]
    result.update_state_alignments = metrics["update_state_alignments"]

    return result


def _capture_with_intervention(
    model,
    layers,
    num_layers: int,
    prompt: str,
    use_remote: bool,
    result: TrajectoryData,
    intervention_type: str,
    intervention_layer: int,
    steering_vector: Optional[torch.Tensor],
    intervention_strength: float,
) -> TrajectoryData:
    """Capture trajectory with intervention at specified layer."""
    with model.trace(remote=use_remote) as tracer:
        layer_outputs = list().save()
        pre_intervention = list().save()
        post_intervention = list().save()

        with tracer.invoke(prompt):
            for l in range(num_layers):
                if l == intervention_layer:
                    # Capture pre-intervention state
                    pre_intervention.append(layers[l].output[:, -1, :].clone())

                    # Apply intervention
                    if intervention_type == "scale":
                        layers[l].output[:, -1, :] *= intervention_strength
                    elif intervention_type in ("concept", "random"):
                        if steering_vector is not None:
                            # Move steering vector to correct device if needed
                            sv = steering_vector
                            if hasattr(layers[l].output, 'device'):
                                sv = sv.to(layers[l].output.device)
                            layers[l].output[:, -1, :] += sv * intervention_strength

                    # Capture post-intervention state
                    post_intervention.append(layers[l].output[:, -1, :].clone())

                # Always capture output
                layer_outputs.append(layers[l].output[:, -1, :])

    # Process captured activations
    states = [lo.detach().cpu().squeeze(0) for lo in layer_outputs]
    result.hidden_dim = states[0].shape[-1]
    result.response = "(trace mode - no generation)"

    # Compute metrics
    metrics = compute_trajectory_metrics(states)
    result.state_norms = metrics["state_norms"]
    result.update_norms = metrics["update_norms"]
    result.update_alignments = metrics["update_alignments"]
    result.update_state_alignments = metrics["update_state_alignments"]

    # Compute intervention geometry
    if pre_intervention and post_intervention:
        pre = pre_intervention[0].detach().cpu().squeeze(0)
        post = post_intervention[0].detach().cpu().squeeze(0)

        result.pre_norm = pre.norm().item()
        result.post_norm = post.norm().item()

        perturbation = post - pre
        result.perturbation_norm = perturbation.norm().item()

        if result.pre_norm > 0 and result.perturbation_norm > 0:
            result.perturbation_cosine = F.cosine_similarity(
                perturbation.unsqueeze(0), pre.unsqueeze(0)
            ).item()

    return result


__all__ = [
    "TrajectoryData",
    "compute_trajectory_metrics",
    "simulate_random_walks",
    "capture_trajectory",
    "get_layer_accessor",
]
