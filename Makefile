OUT ?= ./output.md

.PHONY: check
check:
	uv run ruff check
	uv run ty check
	uv run ruff format

.PHONY: ci
ci:
	uv sync --locked --all-extras --dev
	echo "## Ruff Lint Check\n```\n" >> $OUT
	uv run ruff check >> $OUT
	echo "\n```\n## Ty Type Check\n```\n" >> $OUT
	uv run ty check >> $GITHUB_STEP_SUMMARY
	echo "\n```\n## Ruff Format Check" >> $OUT
	uv run ruff format --check >> $OUT
	echo "\n```\n## pytest\n```\n" >> $OUT
	uv run pytest -p no:cacheprovider >> $OUT
	echo "\n```\n" >> $OUT
	uv cache prune --ci
