.PHONY: clean format-project

format-project:
	uvx pyproject-fmt pyproject.toml || true
	uvx docformatter --in-place --pre-summary-newline --recursive --wrap-summaries 88 --wrap-descriptions 88 src/bonsai_loop/
	uvx ruff format
	uvx ruff check --fix

clean:
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf dist
