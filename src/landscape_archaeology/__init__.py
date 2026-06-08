"""Diagnose operator spectra at a point in parameter space.

landscape-archaeology measures the singular spectrum of a vector-valued
operator at a point — top-k singular values via JVP/VJP power iteration,
with no `m × n` matrix ever materialised.

Two canonical bindings cover the use cases this library targets:

- **Jacobian of a reparameterisation.** Bind ``operator = render_fn``, where
  ``render_fn: PyTree -> PyTree`` is the chain of basis + projection that
  produces a weight pytree from modulation parameters (e.g. loom's render
  bound to an INR basis G and projection P). The spectrum reveals how the
  reparameterisation shapes the implicit gradient on the modulation
  parameters — does G contribute geometric structure beyond a relabelling
  of identity?

- **Hessian of a scalar loss.** Bind ``operator = jax.grad(loss_fn)``, where
  ``loss_fn: PyTree -> Float[Array, ""]``. The Jacobian of the gradient is
  the Hessian, so the singular spectrum of ``jax.grad(loss_fn)`` is the
  spectrum of curvature at the point. Top eigenvalue is the sharpness;
  effective rank, condition number, and other flatness / sharpness proxies
  fall out of the same spectrum.

Both bindings reach the same JVP/VJP power-iteration machinery; the library
does not assume which use case applies, and never inspects ``operator``
beyond calling it.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree

from landscape_archaeology.robustness import (
    delta_from_curve,
    perturbation_curve,
    retraining_noise_floor,
)
from landscape_archaeology.spectral import ht_sr_alpha, radial_fft_alpha


def singular_spectrum(
    operator: Callable[[PyTree], PyTree],
    point: PyTree,
    *,
    k: int = 10,
    num_iterations: int = 50,
    key: Array | None = None,
) -> Float[Array, " k"]:
    r"""Top-k singular spectrum of ``operator`` at ``point``, via JVP/VJP power iteration.

    Block (simultaneous) power iteration on the symmetric operator
    :math:`A = J^T J`, where :math:`J = \\partial \\text{operator} / \\partial \\text{point}`
    is the Jacobian at ``point``. The Jacobian is never materialised: each
    application of :math:`J` uses ``jax.jvp`` and each application of
    :math:`J^T` uses ``jax.vjp``. A length-``k`` orthonormal basis is
    iterated through :math:`A` and re-orthogonalised with a QR factorisation
    at every step; after ``num_iterations`` rounds the diagonal of the
    final ``R`` carries the top-``k`` eigenvalues of :math:`A`, whose
    square roots are the singular values of :math:`J`.

    Two canonical bindings:

    - **Jacobian of a reparameterisation**: bind ``operator = render_fn``, where
      ``render_fn: PyTree -> PyTree`` is e.g. loom's ``render`` bound to a basis
      ``G`` and projection ``P``. The spectrum reveals how ``G``'s geometric
      structure shapes the implicit gradient on the modulation parameters.

    - **Hessian of a scalar loss**: bind ``operator = jax.grad(loss_fn)``,
      where ``loss_fn: PyTree -> Float[Array, ""]``. The spectrum reveals the
      loss landscape's curvature structure at the point — sharpness, top
      eigenvalue, effective rank.

    Both bindings use the same JVP/VJP power-iteration machinery; the only
    difference is what ``operator`` is wired to. The library does not assume
    which use case applies.

    Args:
        operator: any pure function ``PyTree -> PyTree``. The Jacobian of
            ``operator`` at ``point`` is the linear map whose spectrum is
            returned. The library never materialises the Jacobian.
        point: the parameter-space point at which to evaluate the spectrum.
            Defines the linearisation; must match ``operator``'s input shape.
        k: number of top singular values to return.
        num_iterations: power-iteration steps. More steps tighten the
            top-singular-value estimate at a roughly linear cost.
        key: PRNG key for initialising the iteration basis. When ``None``,
            a deterministic ``jax.random.key(0)`` is used so calls are
            reproducible by default.

    Returns:
        Top-k singular values, descending. Shape ``(k,)``. Padded with the
        smaller singular values when ``k`` exceeds the rank of the Jacobian
        — entries below the rank are near-zero (machine epsilon scale).
    """
    flat_point, unravel_input = ravel_pytree(point)
    n = flat_point.shape[0]

    # Probe the operator's output structure once so vjp cotangents can be
    # unravelled back into the right pytree shape.
    out_shape = jax.eval_shape(operator, point)
    out_zeros = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), out_shape)
    _, unravel_output = ravel_pytree(out_zeros)

    def jv(v_flat: Array) -> Array:
        v_pytree = unravel_input(v_flat)
        _, jv_pytree = jax.jvp(operator, (point,), (v_pytree,))
        jv_flat, _ = ravel_pytree(jv_pytree)
        return jv_flat

    def jt_u(u_flat: Array) -> Array:
        _, vjp_fn = jax.vjp(operator, point)
        u_pytree = unravel_output(u_flat)
        (jt_pytree,) = vjp_fn(u_pytree)
        jt_flat, _ = ravel_pytree(jt_pytree)
        return jt_flat

    apply_j = jax.vmap(jv, in_axes=1, out_axes=1)
    apply_jt = jax.vmap(jt_u, in_axes=1, out_axes=1)

    if key is None:
        key = jax.random.key(0)
    v_init = jax.random.normal(key, (n, k), dtype=flat_point.dtype)
    v, _ = jnp.linalg.qr(v_init)

    final_r = jnp.zeros((k, k), dtype=flat_point.dtype)
    for _ in range(num_iterations):
        w = apply_j(v)
        a_v = apply_jt(w)
        v, final_r = jnp.linalg.qr(a_v)

    # After convergence diag(R) holds the eigenvalues of J^T J; take |·| to
    # absorb QR sign conventions, then sqrt for singular values.
    sigmas = jnp.sqrt(jnp.abs(jnp.diag(final_r)))
    return jnp.flip(jnp.sort(sigmas))


__all__ = [
    "delta_from_curve",
    "ht_sr_alpha",
    "perturbation_curve",
    "radial_fft_alpha",
    "retraining_noise_floor",
    "singular_spectrum",
]
__version__ = "0.3.0"
