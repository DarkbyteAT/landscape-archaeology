# CLAUDE.md

@AGENTS.md

## Project Context

A JAX library for measuring the singular spectrum of an operator at a point — without ever materialising the operator's Jacobian. Given any pure `Callable[[PyTree], PyTree]` and a point in its domain, `landscape-archaeology` returns top-k singular values via JVP/VJP power iteration.

Two canonical bindings cover the motivating use cases. **Jacobian of a reparameterisation**: bind `operator = render_fn` (e.g. `loom.render` composed with an `ondes` basis) to interrogate the geometric structure the reparameterisation contributes. **Hessian of a scalar loss**: bind `operator = jax.grad(loss_fn)` — the Jacobian of the gradient is the Hessian, so the same machinery returns curvature diagnostics (sharpness, top eigenvalue, effective rank).

Operator-agnostic. Sibling library to [`loom`](https://github.com/DarkbyteAT/loom) (the reparameterisation substrate) and [`ondes`](https://github.com/DarkbyteAT/ondes) (INR primitives). Downstream of neither; never imports either. The library does not inspect what the operator represents — Jacobian, Hessian, or neither — and never assumes which binding the caller chose.

See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the API stance: one verb, two canonical bindings, measurement-only, no model assumptions baked in.
