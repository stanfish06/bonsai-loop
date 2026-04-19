# Bonsai loop

Trajectory analysis of human 2D gastruloid data using the package [Bonsai-data-representation](https://github.com/dhdegroot/Bonsai-data-representation.git).

## Setup

Two required packages (`Sanity` and `Bonsai-data-representation`) are included as git submodules:

```bash
git submodule update --init --recursive
```

### Install uv

If uv is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create the virtual environment

```bash
uv sync
```

### Activate venv

```bash
./activate-venv
# or
source activate-venv
```

### Register an IPython kernel *(optional)*

```bash
uv run ipython kernel install --name 'bonsai-loop' --user
```

## Uninstalling uv

```bash
uv cache clean
rm -r "$(uv python dir)"
rm -r "$(uv tool dir)"
rm ~/.local/bin/uv ~/.local/bin/uvx
```
