"""Smoke tests for package installation."""


def test_package_imports():
    """Verify the package can be imported."""
    import llama_introspection

    assert llama_introspection.__version__ == "0.1.0"
