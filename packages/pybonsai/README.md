# pybonsai

Python bindings for the C++ Bonsai

## Test

```bash
nix develop -c uv run pytest
```

`tests/test_reference.py` compares against the output of penjing's own command-line tools on a
real tree. It is skipped unless `PYBONSAI_TREE_DIR` and `PYBONSAI_REFERENCE_DIR` are set:

```bash
nix develop -c packages/pybonsai/scripts/make_reference.sh <final-bonsai-folder>
PYBONSAI_TREE_DIR=<final-bonsai-folder> \
  PYBONSAI_REFERENCE_DIR=$PWD/packages/pybonsai/results/reference nix develop -c uv run pytest
```

## Use

```python
from pybonsai import BonsaiOutput

out = BonsaiOutput("path/to/final_bonsai_zscore0.75")  # Python or C++ Bonsai output
tree = out.tree_distances(leaves_only=True)  # sum of branch lengths
post = out.posterior_distances(leaves_only=True, with_uncertainty=True)
residual = post.values - tree.values  # negative: closer than the tree allows
top = out.detect_convergence(leaves_only=True, top_k=50, max_pairs=20_000_000)
```

`values` and `uncertainty` are condensed arrays in `scipy.spatial.distance.squareform` order
over `node_ids`. Both distances are diffusion times.

`bonsai_loop.bonsai_lib.get_pdists_posterior` wraps `posterior_distances` with the same
`cell_ids` contract as `get_pdists_on_tree`.
