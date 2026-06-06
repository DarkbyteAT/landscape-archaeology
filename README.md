# landscape-archaeology

A JAX library for measuring the singular spectrum of an operator at a point — without ever materialising the operator's Jacobian.

One verb, two canonical bindings:

- **Jacobian of a reparameterisation** — bind `operator = render_fn` to ask what geometric structure a basis `G` and projection `P` contribute to the implicit gradient on modulation parameters. Does the reparameterisation do more than relabel identity?
- **Hessian of a scalar loss** — bind `operator = jax.grad(loss_fn)` to ask what the loss landscape looks like at this point. The Jacobian of the gradient is the Hessian; its spectrum is curvature.

The library is operator-agnostic. It accepts any pure function `PyTree → PyTree`, a point in its domain, and returns spectral diagnostics. Reparameterisations produced by [`loom`](https://github.com/DarkbyteAT/loom) composed through [`ondes`](https://github.com/DarkbyteAT/ondes) bases are the canonical Jacobian instantiation; scalar losses of any shape are the canonical Hessian instantiation; anything else satisfying the type also works.

See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the design stance.

## What it looks like

```python
import jax
import landscape_archaeology as la

# Jacobian binding — what does the reparameterisation's geometry look like?
def render_fn(params):  # PyTree -> PyTree
    ...

jac_spectrum = la.singular_spectrum(render_fn, params, k=32)

# Hessian binding — what does the loss landscape look like at this point?
def loss_fn(params):  # PyTree -> Float[Array, ""]
    ...

hess_spectrum = la.singular_spectrum(jax.grad(loss_fn), params, k=32)
```

## Install

`landscape-archaeology` is not on PyPI yet. Install from git:

```bash
uv add git+https://github.com/DarkbyteAT/landscape-archaeology
```

## Status

v0.0.0 — scaffold only. The public surface raises `NotImplementedError`; see the [Trello board](https://trello.com/b/h8iNt8nJ/landscape-archaeology) for the first diagnostic card.

## License

MIT
