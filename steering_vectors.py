"""Legacy steering_vectors module - re-exports from llama_introspection.steering.

This file is deprecated. Import from llama_introspection.steering instead.

Note: This module lazily loads prompts.pt for backwards compatibility with
existing experiment scripts that use BASELINE_WORDS, GENERIC_PROMPT_TEMPLATE,
and ALL_CONCEPTS.
"""

from pathlib import Path

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

# Lazily load prompts data (project-specific, not part of core library)
_prompts_cache = None


def _load_prompts():
    global _prompts_cache
    if _prompts_cache is None:
        import torch
        prompts_path = Path(__file__).parent / "prompts.pt"
        if prompts_path.exists():
            _prompts_cache = torch.load(prompts_path, weights_only=False)
        else:
            _prompts_cache = {"baseline_words": [], "generic_prompt_template": "", "prompts": {}}
    return _prompts_cache


def __getattr__(name):
    """Lazy loading for prompts data."""
    if name == "BASELINE_WORDS":
        return _load_prompts()["baseline_words"]
    elif name == "GENERIC_PROMPT_TEMPLATE":
        return _load_prompts()["generic_prompt_template"]
    elif name == "ALL_CONCEPTS":
        return list(_load_prompts()["prompts"].keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
