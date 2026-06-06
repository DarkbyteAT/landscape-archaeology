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
- No loss assumptions inside the library — accept any `Callable[[PyTree], Float[Array, ""]]`
- No explicit Hessian materialisation; everything via `jax.jvp` / `jax.vjp` and HVP composition
- No magic-number thresholds on learned quantities; derive float tolerances from `jnp.finfo(dtype).eps`

## Archetypes

For multi-agent reviews and parallel work, draw from the archetype gallery in sibling repos (`loom/AGENTS.md`, `jacobian-spec/AGENTS.md`). The relevant personas here are:

- **measurement-purist** — challenges any addition that interprets the loss surface beyond its scalar value and its derivatives; the library measures, it does not model
- **numerics-pedant** — flags float thresholds, eps handling, eigenvalue-convergence criteria, and HVP conditioning concerns
- **api-minimalist** — keeps the public surface to one verb per question; rejects convenience wrappers and result-bundle value types
