# Bonsai loop

Trajectory analysis of human 2D gastruloid data using the package [Bonsai-data-representation](https://github.com/dhdegroot/Bonsai-data-representation.git).

## Data

Datasets used for notebooks have been uploaded to zenodo and made public, and the code for downloading data has been added to each notebook.

## Setup

`Bonsai-data-representation` is included as a git submodule under `third_party/`; Sanity comes from the `sanity-py` package:

```bash
git submodule update --init third_party/Bonsai-data-representation
```

`packages/pybonsai` (C++ Bonsai bindings, used by `bonsai_lib.get_pdists_posterior`)

### System requirements

Python 3.12 or 3.13; `uv sync` downloads one if needed.

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

### Git hooks

Formatting and linting run as [lefthook](https://lefthook.dev) hooks:

```bash
lefthook install
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
