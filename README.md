# Bonsai loop

Trajectory analysis of human 2D gastruloid data using the package [Bonsai-data-representation](https://github.com/dhdegroot/Bonsai-data-representation.git).

## Data

Datasets used for notebooks have been uploaded to zenodo and made public, and the code for downloading data has been added to each notebook.

## Setup

Two required packages (`Sanity` and `Bonsai-data-representation`) are included as git submodules:

```bash
git submodule update --init --recursive
```

### System requirements

PyTables (`tables`) builds from source on apple arm chip (might not have wheel):

```bash
brew install hdf5 c-blosc lzo bzip2
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
