"""Diagnose operator spectra at a point in parameter space.

landscape-archaeology measures the singular spectrum of a vector-valued
operator at a point — top-k singular values via JVP/VJP power iteration,
with no `N × N` matrix ever materialised.

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
beyond calling it. This is the v0.0.0 scaffold — the public surface below
raises NotImplementedError until the first diagnostic card lands.
"""

from collections.abc import Callable

from jaxtyping import Array, Float, PyTree


def singular_spectrum(
    operator: Callable[[PyTree], PyTree],
    point: PyTree,
    *,
    k: int = 10,
    num_iterations: int = 50,
) -> Float[Array, " k"]:
    """Top-k singular spectrum of ``operator`` at ``point``, via JVP/VJP power iteration.

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
        num_iterations: power-iteration steps per singular value.

    Returns:
        Top-k singular values, descending. Shape ``(k,)``.
    """
    raise NotImplementedError(
        "landscape-archaeology v0.0.0 is a scaffold; "
        "see https://trello.com/b/h8iNt8nJ/landscape-archaeology for the first diagnostic card"
    )


__all__ = ["singular_spectrum"]
__version__ = "0.0.0"
