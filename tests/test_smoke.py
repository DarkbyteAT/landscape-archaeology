"""Smoke test: package imports, surface exists, scaffold raises NotImplementedError."""

import jax
import jax.numpy as jnp
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
def test_singular_spectrum_scaffold_raises_not_implemented_on_jacobian_binding():
    # Given: a trivial reparameterisation operator (identity, here) and a point.
    def render_fn(params):
        return params

    point = jnp.array([1.0, 2.0, 3.0])

    # When: we call the v0.0.0 scaffold with the Jacobian binding.
    # Then: it raises NotImplementedError, pointing at the Trello board.
    with pytest.raises(NotImplementedError, match="scaffold"):
        la.singular_spectrum(render_fn, point, k=2)


@pytest.mark.unit
def test_singular_spectrum_scaffold_raises_not_implemented_on_hessian_binding():
    # Given: a scalar quadratic loss and its gradient operator.
    def loss_fn(params):
        return jnp.sum(params * params)

    grad_fn = jax.grad(loss_fn)
    point = jnp.array([1.0, 2.0, 3.0])

    # When: we call the v0.0.0 scaffold with the Hessian binding (operator = grad of loss).
    # Then: it raises NotImplementedError, pointing at the Trello board.
    with pytest.raises(NotImplementedError, match="scaffold"):
        la.singular_spectrum(grad_fn, point, k=2)
