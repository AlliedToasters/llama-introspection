"""Legacy util module - re-exports from llama_introspection.utils.

This file is deprecated. Import from llama_introspection.utils instead.
"""

# Re-export from llama_introspection.utils
from llama_introspection.utils import (
    compute_kl_divergence,
    compute_kl_curves,
    get_token_transitions,
    compute_injection_position,
)

__all__ = [
    "compute_kl_divergence",
    "compute_kl_curves",
    "get_token_transitions",
    "compute_injection_position",
]
