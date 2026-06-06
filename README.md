# landscape-archaeology

A JAX library for diagnosing the geometry of a scalar loss surface at a point. Given a loss `L(params)` and a point in parameter space, `landscape-archaeology` answers — what does the curvature look like, how sharp is the minimum, and how far can you perturb before accuracy collapses?

Everything happens through Hessian-vector-products: no `N × N` Hessian is ever materialised. The library is loss-agnostic — it expects a pure function `params → scalar` and a point; it returns diagnostics. Loss surfaces produced by [`loom`](https://github.com/DarkbyteAT/loom) reparameterisations of [`ondes`](https://github.com/DarkbyteAT/ondes)-rendered weights are the canonical instantiation, but anything `Callable[[PyTree], Float[Array, ""]]` is a valid target.

See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the API stance.

## What it looks like

```python
import landscape_archaeology as la

# loss_fn is any pure function from params to a scalar loss.
def loss_fn(params):
    ...

spectrum = la.hessian_topk(loss_fn, params, k=10)
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
