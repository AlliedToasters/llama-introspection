"""Pytest configuration and shared fixtures."""

import pytest

# Tiny model for fast unit tests (4.11M params)
TINY_MODEL_ID = "llamafactory/tiny-random-Llama-3"


@pytest.fixture(scope="session")
def tiny_model_id():
    """Return the tiny model ID for tests."""
    return TINY_MODEL_ID


@pytest.fixture(scope="session")
def tiny_model():
    """Load the tiny Llama model for testing.

    Session-scoped to avoid reloading for each test.
    """
    from nnsight import LanguageModel

    model = LanguageModel(TINY_MODEL_ID, device_map="auto")
    return model
