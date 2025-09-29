OUT ?= ./output.md

.PHONY: check
check:
	uv run ruff check
	uv run ty check
	uv run ruff format

.PHONY: ci
ci:
	uv sync --locked --all-extras --dev
	uv run --frozen pytest --cov-report=term-missing -p no:cacheprovider >> "${OUT}"
	uv cache prune --ci
