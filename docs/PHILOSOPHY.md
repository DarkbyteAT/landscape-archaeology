# landscape-archaeology — Philosophy

## The question

What is `landscape-archaeology` for?

## The verdict

> **`landscape-archaeology` is a library for measuring the geometry of a scalar loss surface at a point, via Hessian-vector-products and their compositions — without ever materialising the Hessian.**

That's it. Not a training tool. Not a sharpness-aware optimiser. Not a generalisation-bound estimator. A *measurement library*. The library is loss-agnostic — it expects a pure function `params → scalar` and a point in parameter space, and returns diagnostics.

Everything else — what loss to use, how the model is parameterised, how the diagnostics are interpreted, what training trajectory produced `params`, what to do with the answer — lives upstream or downstream. The library owns the mechanism (HVPs and the iterative algorithms that build on them), not any specific instantiation of the loss surface.

The library is thin by design. The product is the **contract** — that each public verb answers one geometric question about the loss surface, returns a well-typed result, and never silently materialises an `N × N` matrix.

## Four principles

### 1. One verb per question

Each public function answers a single geometric question about the loss surface and returns a well-typed result. No bundles, no "diagnostic suites", no `LandscapeReport` value type. If the user wants three diagnostics, they call three functions.

**Rules out**: convenience wrappers, result-bundle value types, "summary" verbs that hide which measurements were actually performed.

### 2. Measurement, not modelling

The library does not interpret the loss surface. It does not decide whether a minimum is "sharp enough to generalise poorly", does not name regions, does not classify landscapes. It returns numbers; the caller assigns meaning.

**Rules out**: opinionated thresholds (e.g. "sharpness > 100 means bad"), classification verbs (`is_sharp_minimum`, `is_in_basin`), built-in generalisation predictors.

### 3. No matrix materialisation

Every diagnostic must be expressible as a composition of Hessian-vector-products and `jax.grad` / `jax.jvp` / `jax.vjp`. The Hessian is not stored. If a planned measurement requires the full Hessian, it does not belong in this library.

**Rules out**: dense-Hessian APIs, "approximate Hessian via blocked computation" helpers, anything that allocates `O(N²)` for an `N`-parameter model.

### 4. Loss-agnostic

The library accepts any `Callable[[PyTree], Float[Array, ""]]`. No assumptions about the model architecture, the data distribution, the optimiser, or whether `params` was reparameterised through a renderer. If a measurement needs more context than a scalar loss and a point, it does not belong here.

**Rules out**: model-specific shortcuts (CNN-aware, transformer-aware), data-loader integration, training-history-aware diagnostics.

## What the v1 surface will look like

The exact verbs land as Trello cards. The shape will be roughly:

```python
import landscape_archaeology as la

# Top-k Hessian eigenspectrum via HVP power iteration.
eigvals = la.hessian_topk(loss_fn, params, k=10)

# Curvature along an arbitrary direction (rayleigh quotient).
curvature = la.directional_curvature(loss_fn, params, direction)

# Perturbation radius to lose ε accuracy (random or Hessian-aligned).
radius = la.perturbation_radius(loss_fn, params, epsilon=0.1, ...)
```

Each verb is independent; none depend on the result of another.

## What landscape-archaeology does NOT own

- Loss definitions — caller supplies
- Model architectures — irrelevant; loss is opaque
- Training loops — upstream
- Optimisers (including sharpness-aware ones like SAM) — upstream
- Generalisation predictions — downstream interpretation
- Plotting / visualisation — downstream
- Sweep registries — downstream
- Dense-Hessian APIs — structurally out by Principle 3

## Status

v0.0.0 — scaffold only. The public surface raises `NotImplementedError`. See the Trello board for the first diagnostic card.
