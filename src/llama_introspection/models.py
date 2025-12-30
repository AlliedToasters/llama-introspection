"""Model configuration and utilities.

Provides model shortcuts, remote model detection, and default experiment parameters.
"""

from typing import Set

MODEL_SHORTCUTS: dict[str, str] = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Llama-3.1-70B-Instruct",
    "405B": "meta-llama/Llama-3.1-405B-Instruct",
}

REMOTE_MODELS: Set[str] = {"405B", "meta-llama/Llama-3.1-405B-Instruct"}

DEFAULT_STRENGTHS: list[float] = [0.5, 1.0, 2.0, 4.0, 8.0]

DEFAULT_SCALE_FACTORS: list[float] = [0.5, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

MAX_NEW_TOKENS: int = 100


def resolve_model_id(model: str) -> str:
    """Resolve a model shortcut to its full HuggingFace model ID.

    Args:
        model: Either a shortcut (e.g., "8B") or a full model ID.

    Returns:
        The full HuggingFace model ID.

    Raises:
        ValueError: If the model is not a known shortcut or valid model ID.
    """
    if model in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[model]
    if "/" in model:
        # Assume it's already a full model ID
        return model
    raise ValueError(
        f"Unknown model: {model}. "
        f"Use a shortcut ({list(MODEL_SHORTCUTS.keys())}) or a full HuggingFace model ID."
    )


def is_remote_model(model: str) -> bool:
    """Check if a model requires remote execution via NDIF.

    Args:
        model: Either a shortcut or full model ID.

    Returns:
        True if the model should be run remotely.
    """
    return model in REMOTE_MODELS or resolve_model_id(model) in REMOTE_MODELS
