# AGENTS.md

## Commands

```bash
uv sync
uv run ruff check src/
uv run ruff format --check src/
uv run pyright src/
uv run pytest tests/ -v
uv run pytest -m unit
```

## Critical Rules

- Python 3.11+
- Plain `def test_*` functions, Given-When-Then structure
- Public surface lives in `src/landscape_archaeology/__init__.py`; one verb per question
- The library is operator-agnostic — accept any `Callable[[PyTree], PyTree]`. The two motivating bindings (Jacobian of a reparameterisation, Hessian-as-Jacobian-of-grad) reach the same machinery; the library does not branch on which one the caller chose
- No explicit Jacobian / Hessian materialisation; everything via `jax.jvp` / `jax.vjp` / `jax.grad` compositions
- No magic-number thresholds on learned quantities; derive float tolerances from `jnp.finfo(dtype).eps`

## Archetypes

For multi-agent reviews and parallel work, draw from the archetype gallery in sibling repos (`loom/AGENTS.md`, `ondes/AGENTS.md`). The relevant personas here are:

- **measurement-purist** — challenges any addition that interprets the operator beyond calling it; the library measures, it does not model. Pushes back hard on binding-specific verbs (`hessian_topk`, `jacobian_topk`) that would re-fragment the API
- **numerics-pedant** — flags float thresholds, eps handling, singular-value convergence criteria, and JVP/VJP conditioning concerns
- **api-minimalist** — keeps the public surface to one verb per question; rejects convenience wrappers, result-bundle value types, and binding-specific shortcuts
