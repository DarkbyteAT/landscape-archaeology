.PHONY: lint format format-check fix typecheck test all

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

format-check:
	uv run ruff format --check src/

fix:
	uv run ruff check --fix src/

typecheck:
	uv run pyright src/

test:
	uv run pytest tests/ -v || [ $$? -eq 5 ]

all: format-check lint typecheck test
