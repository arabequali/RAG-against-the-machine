.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size 2000

debug:
	uv run python -m pdb -m src index --max_chunk_size 2000

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	-uv run python3 -m flake8 src/
	-uv run python3 -m mypy src/ --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict
