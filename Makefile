.PHONY: check
check:
	uv run ruff check
	uv run ty check
	uv run ruff format
