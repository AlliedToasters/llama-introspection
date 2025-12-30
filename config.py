"""Legacy config module - re-exports from llama_introspection.models.

This file is deprecated. Import from llama_introspection.models instead.
"""

from llama_introspection.models import (
    MODEL_SHORTCUTS,
    REMOTE_MODELS,
    DEFAULT_STRENGTHS,
    DEFAULT_SCALE_FACTORS,
    MAX_NEW_TOKENS,
    resolve_model_id,
    is_remote_model,
)

__all__ = [
    "MODEL_SHORTCUTS",
    "REMOTE_MODELS",
    "DEFAULT_STRENGTHS",
    "DEFAULT_SCALE_FACTORS",
    "MAX_NEW_TOKENS",
    "resolve_model_id",
    "is_remote_model",
]