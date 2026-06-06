"""Smoke test: package imports, surface exists, scaffold raises NotImplementedError."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import landscape_archaeology as la


@pytest.mark.unit
def test_package_exports_hessian_topk():
    # Given: the package as imported.
    # When: we look up the documented public surface.
    # Then: hessian_topk is exposed and callable.
    assert hasattr(la, "hessian_topk")
    assert callable(la.hessian_topk)


@pytest.mark.unit
def test_hessian_topk_scaffold_raises_not_implemented():
    # Given: a trivial quadratic loss on a single-leaf pytree.
    def loss_fn(params):
        return jnp.sum(params * params)

    params = jnp.array([1.0, 2.0, 3.0])

    # When: we call the v0.0.0 scaffold.
    # Then: it raises NotImplementedError, pointing at the Trello board.
    with pytest.raises(NotImplementedError, match="scaffold"):
        la.hessian_topk(loss_fn, params, k=2)
