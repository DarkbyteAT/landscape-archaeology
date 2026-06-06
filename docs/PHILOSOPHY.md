# landscape-archaeology — Philosophy

## The question

What is `landscape-archaeology` for?

## The verdict

> **`landscape-archaeology` is a library for measuring the singular spectrum of an operator at a point, via JVP/VJP power iteration — without ever materialising the operator's Jacobian.**

That's it. Not a training tool. Not a sharpness-aware optimiser. Not a generalisation-bound estimator. Not a reparameterisation diagnostic, specifically — a measurement library for operator spectra in general, of which the reparameterisation-Jacobian and loss-Hessian cases are the two motivating bindings.

The library accepts any pure function `operator: PyTree → PyTree` and a point in its domain; it returns spectral diagnostics. Everything else — what the operator is, how the point was produced, what to do with the answer — lives upstream or downstream.

The library is thin by design. The product is the **contract** — that the public verb answers one geometric question (the singular spectrum of the linearisation at a point), returns a well-typed result, and never silently materialises an `N × N` matrix.

## Two canonical bindings, one verb

The library exposes a single public verb — `singular_spectrum` — because the two questions it serves are the same geometric measurement under two different bindings of `operator`:

### Jacobian of a reparameterisation

Bind `operator = render_fn`, where `render_fn: PyTree → PyTree` is the chain of basis + projection that produces a weight pytree from modulation parameters:

```python
import landscape_archaeology as la
import loom, ondes

def render_fn(params):
    return loom.render(P, ondes_basis_G, params)

spectrum = la.singular_spectrum(render_fn, params, k=32)
```

The spectrum answers: does `G` carry geometric structure that makes it more than a relabelling of identity? Heavy-tailed spectra indicate the reparameterisation concentrates gradient signal in a low-dimensional subspace; flat spectra indicate the chain behaves like a high-dimensional identity.

### Hessian of a scalar loss

Bind `operator = jax.grad(loss_fn)`, where `loss_fn: PyTree → Float[Array, ""]`. The Jacobian of the gradient is the Hessian, so the singular spectrum of `jax.grad(loss_fn)` is the spectrum of curvature at the point:

```python
import jax
import landscape_archaeology as la

def loss_fn(params): ...

spectrum = la.singular_spectrum(jax.grad(loss_fn), params, k=32)
```

Top eigenvalue is the sharpness; effective rank, condition number, and other flatness / sharpness proxies fall out of the same spectrum.

### Why one verb, not two

Both bindings reach the same JVP/VJP power-iteration machinery. The library does not inspect `operator` to discover which case applies; it cannot, and does not want to. A two-verb surface (`hessian_topk` plus `jacobian_topk`, say) would double the contract, duplicate the implementation, and impose an ontology — "this is a Hessian, that is a Jacobian" — that the library has no need to know. One verb keeps the contract minimal and the user honest about the binding.

## Four principles

### 1. One verb per question

The geometric question is *what is the top-k singular spectrum of the linearisation at this point?* That is one question. The library exposes one verb to answer it. No bundles, no "diagnostic suites", no `SpectrumReport` value type. If the user wants other diagnostics derived from the spectrum (effective rank, condition number, projection coverage), those land as separate verbs taking the spectrum as input.

**Rules out**: convenience wrappers, result-bundle value types, binding-specific verbs (`hessian_topk`, `jacobian_topk`) that bake in the user's choice of operator.

### 2. Measurement, not modelling

The library returns numbers; the caller assigns meaning. It does not classify spectra as "sharp" or "flat", does not name regions of parameter space, does not predict generalisation.

**Rules out**: opinionated thresholds, classification verbs, generalisation-bound estimators.

### 3. No matrix materialisation

Every diagnostic must be expressible as a composition of JVPs and VJPs (`jax.jvp` / `jax.vjp` / `jax.grad`). No `N × N` matrix is ever stored. If a planned measurement requires the full Jacobian, it does not belong in this library.

**Rules out**: dense-Jacobian APIs, "approximate via blocked computation" helpers, anything `O(N²)` in parameter count.

### 4. Operator-agnostic

The library accepts any `Callable[[PyTree], PyTree]`. No assumptions about what the operator represents — Jacobian of a reparameterisation, gradient of a scalar loss, gradient of a vector-valued auxiliary, whatever. If a measurement needs more context than an operator and a point, it does not belong here.

**Rules out**: binding-specific shortcuts, loss-specific shortcuts, model-architecture-aware shortcuts.

## What the v1 surface will look like

The exact verb signatures and follow-on diagnostics land as Trello cards. The shape will be roughly:

```python
import jax
import landscape_archaeology as la

# Top-k singular spectrum (the one primitive).
spectrum = la.singular_spectrum(operator, point, k=32)

# Spectrum-derived diagnostics (future verbs; each takes a spectrum as input).
# rank = la.effective_rank(spectrum, ...)
# kappa = la.condition_number(spectrum)
```

The Jacobian and Hessian use cases are reached by choice of `operator`; no separate API surface.

## What landscape-archaeology does NOT own

- Operator definitions — caller supplies
- Loss definitions — caller supplies (then passes `jax.grad(loss_fn)`)
- Reparameterisation definitions — caller supplies (then passes `render_fn`)
- Model architectures — irrelevant; the operator is opaque
- Training loops — upstream
- Optimisers (including sharpness-aware ones like SAM) — upstream
- Generalisation predictions — downstream interpretation
- Plotting / visualisation — downstream
- Sweep registries — downstream
- Dense-Jacobian / dense-Hessian APIs — structurally out by Principle 3

## Status

v0.0.0 — scaffold only. The public surface raises `NotImplementedError`. See the Trello board for the first diagnostic card.
