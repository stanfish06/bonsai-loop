.PHONY: clean format-project lint

format-project:
	uvx pyproject-fmt pyproject.toml || true
	uvx docformatter --in-place --pre-summary-newline --recursive --wrap-summaries 88 --wrap-descriptions 88 src/bonsai_loop/ || true
	uv run ruff format || true
	uv run ruff check --fix || true


lint:
	uv run pyright || true
	uv run ruff check src/ || true
	uv run --with mypy mypy --ignore-missing-imports src/ || true
	uv run pyrefly check src/ || true

clean:
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf dist

