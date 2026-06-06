# CLAUDE.md

@AGENTS.md

## Project Context

A JAX library for diagnosing loss-landscape geometry without explicit Hessian materialisation. Given a scalar loss function `L(params)` and a point in parameter space, `landscape-archaeology` returns curvature diagnostics: top-k Hessian eigenspectrum via Hessian-vector-product power iteration, curvature along arbitrary directions, perturbation radius to lose ε accuracy, and other flatness / sharpness proxies.

Loss-agnostic. Sibling library to [`loom`](https://github.com/DarkbyteAT/loom) (the reparameterisation substrate) and [`ondes`](https://github.com/DarkbyteAT/ondes) (INR primitives). Downstream of neither; never imports either. The library accepts any `Callable[[PyTree], Float[Array, ""]]` as the loss surface to interrogate.

See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the API stance: one verb per question, measurement-only, no model assumptions baked in.
