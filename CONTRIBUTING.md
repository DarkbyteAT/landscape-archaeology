# Contributing to landscape-archaeology

## Development Setup

```bash
git clone https://github.com/DarkbyteAT/landscape-archaeology.git
cd landscape-archaeology
uv sync
```

## Code Conventions

- **Python 3.11+** — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- **Google-style docstrings**

## Docstring Style

Google-style docstrings with LaTeX math support:

```python
"""Summary line.

Args:
    param: Description of param.

Returns:
    Description of return value.
"""
```

Use `$...$` for inline math and `$$...$$` for display math in docstrings.

## Quality Gates

```bash
make all    # format-check + lint + typecheck + test
make fix    # auto-fix lint violations
```

Tool configs live in dedicated files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`).

## Testing

- Plain `def test_*` functions — no classes
- Given-When-Then structure
- `tests/` mirrors `src/landscape_archaeology/` layout
- `@pytest.mark.unit` for fast isolated tests
