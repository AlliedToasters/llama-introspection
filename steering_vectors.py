"""Legacy steering_vectors module - re-exports from llama_introspection.steering.

This file is deprecated. Import from llama_introspection.steering instead.

Note: This module still loads prompts.pt for backwards compatibility with
existing experiment scripts that use BASELINE_WORDS, GENERIC_PROMPT_TEMPLATE,
and ALL_CONCEPTS.
"""

import torch

from llama_introspection.steering import (
    SteeringVectorResult,
    get_cache_path,
    get_layer_accessor,
    get_num_layers,
    compute_bespoke_vector,
    compute_baseline_means,
    compute_generic_vector,
    compute_random_vector,
    normalize_vectors,
    match_vector_norms,
    compute_activation_statistics,
    load_cached_vectors,
    list_cached_vectors,
    compute_injection_position,
    create_ablation_vectors,
    compute_mean_steering_norm,
)

# Load prompts data (project-specific, not part of core library)
_prompts = torch.load("prompts.pt")

BASELINE_WORDS = _prompts["baseline_words"]
GENERIC_PROMPT_TEMPLATE = _prompts["generic_prompt_template"]
ALL_CONCEPTS = list(_prompts["prompts"].keys())

__all__ = [
    # Core classes and functions from llama_introspection.steering
    "SteeringVectorResult",
    "get_cache_path",
    "get_layer_accessor",
    "get_num_layers",
    "compute_bespoke_vector",
    "compute_baseline_means",
    "compute_generic_vector",
    "compute_random_vector",
    "normalize_vectors",
    "match_vector_norms",
    "compute_activation_statistics",
    "load_cached_vectors",
    "list_cached_vectors",
    "compute_injection_position",
    "create_ablation_vectors",
    "compute_mean_steering_norm",
    # Project-specific data
    "BASELINE_WORDS",
    "GENERIC_PROMPT_TEMPLATE",
    "ALL_CONCEPTS",
]
