.PHONY: lint format check

lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

check:
	uv run prek run --all-files
