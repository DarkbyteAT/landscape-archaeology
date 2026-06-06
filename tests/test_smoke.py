"""Smoke test: package imports and exposes the documented public surface."""

import pytest

import landscape_archaeology as la


@pytest.mark.unit
def test_package_exports_singular_spectrum():
    # Given: the package as imported.
    # When: we look up the documented public surface.
    # Then: singular_spectrum is exposed and callable.
    assert hasattr(la, "singular_spectrum")
    assert callable(la.singular_spectrum)


@pytest.mark.unit
def test_package_exposes_version():
    # Given: the package as imported.
    # When: we look up __version__.
    # Then: it is a non-empty string.
    assert hasattr(la, "__version__")
    assert isinstance(la.__version__, str)
    assert la.__version__
