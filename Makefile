.PHONY: clean format-project

format-project: 
	uvx pyproject-fmt pyproject.toml || true
	uvx ruff format
	uvx ruff check --fix

clean:
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf dist
