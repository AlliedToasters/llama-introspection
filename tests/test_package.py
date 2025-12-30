"""Smoke tests for package installation."""


def test_package_imports():
    """Verify the package can be imported."""
    import llama_introspection

    assert llama_introspection.__version__ == "0.1.0"


def test_tiny_model_loads(tiny_model):
    """Verify the tiny test model loads correctly."""
    assert tiny_model is not None
    assert hasattr(tiny_model, "generate")
