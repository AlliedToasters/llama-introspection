"""LLaMA introspection experiments.

Reproducing and extending Anthropic's introspection paper experiments
on LLaMA-3.1-Instruct models.
"""

__version__ = "0.1.0"

from llama_introspection.trajectory import (
    TrajectoryData,
    compute_trajectory_metrics,
    simulate_random_walks,
    capture_trajectory,
    get_layer_accessor,
)
