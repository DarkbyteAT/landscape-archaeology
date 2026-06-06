"""Diagnose loss-landscape geometry without explicit Hessian materialisation.

landscape-archaeology measures the curvature structure of a scalar loss
function L(params) at a point in parameter space — top-k Hessian eigenspectrum
via Hessian-vector-product power iteration, curvature along arbitrary
directions, perturbation radius to lose ε accuracy, and other flatness /
sharpness proxies.

This is the v0.0.0 scaffold; the public surface below is a placeholder and
returns NotImplementedError. The first real diagnostic lands as the Trello
board's first Doing card.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


def hessian_topk(
    loss_fn: Callable[[PyTree], Float[Array, ""]],
    params: PyTree,
    *,
    k: int = 10,
    num_iterations: int = 50,
) -> Float[Array, "k"]:
    """Top-k Hessian eigenvalues of `loss_fn` at `params`, via HVP power iteration.

    Args:
        loss_fn: a scalar loss function PyTree -> Float[Array, ""].
        params: the point at which to evaluate the spectrum.
        k: number of top eigenvalues to return.
        num_iterations: power-iteration steps per eigenvalue.

    Returns:
        Top-k eigenvalues, descending. Shape (k,).
    """
    raise NotImplementedError(
        "landscape-archaeology v0.0.0 is a scaffold; "
        "see https://trello.com/b/h8iNt8nJ/landscape-archaeology for the first diagnostic card"
    )


__all__ = ["hessian_topk"]
__version__ = "0.0.0"
