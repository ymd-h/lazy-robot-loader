.PHONY: check
check:
	uv run ruff check
	uv run ty check
	uv run ruff format

.PHONY: ci
ci:
	uv sync --locked --all-extras --dev
	uv run ruff check
	uv run ty check
	uv run ruff format --check
	uv run pytest -p no:cacheprovider
	uv cache prune --ci
