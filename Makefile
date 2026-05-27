.PHONY: clean format-project lint

format-project:
	uvx pyproject-fmt pyproject.toml || true
	uvx docformatter --in-place --pre-summary-newline --recursive --wrap-summaries 88 --wrap-descriptions 88 src/bonsai_loop/ || true
	uvx ruff format
	uvx ruff check --fix


lint:
	uvx ruff check src/
	uv run --with mypy mypy --ignore-missing-imports src/

clean:
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf dist

