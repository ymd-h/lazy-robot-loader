OUT ?= ./output.md

.PHONY: check
check:
	uv run ruff check
	uv run ty check
	uv run ruff format

.PHONY: ci
ci:
	uv sync --locked --all-extras --dev
	echo "```" >> "${OUT}"
	uv run --frozen pytest --cov-report=term-missing -p no:cacheprovider >> "${OUT}"
	echo "```" >> "${OUT}"
	uv cache prune --ci
