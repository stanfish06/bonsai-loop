"""Check bonsai_lib.get_pdists_posterior on a Bonsai output folder.

1. Tree distances from pybonsai agree with get_pdists_on_tree (tree.nwk, pure Python).
2. get_pdists_posterior agrees with penjing's command-line output, when a reference
   folder written by packages/pybonsai/scripts/make_reference.sh is given.
3. Prints how the corrected distance compares with the uncorrected one.

Usage: uv run python scripts/check_posterior_pdists.py [bonsai_output_folder] [reference_dir]
"""

import csv
import sys
from pathlib import Path

import numpy as np
from pybonsai import BonsaiOutput
from scipy.stats import spearmanr

from bonsai_loop.bonsai_lib import (
    get_bonsai_euclidean_distances,  # pyright: ignore[reportAttributeAccessIssue]
    get_pdists_on_tree,
    get_pdists_posterior,
)

# ---- options -------------------------------------------------------------
# final Bonsai output folder to check
BONSAI_OUTPUT_FOLDER = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/stan/Git/micropattern/notebooks/heemskerk_data/Bonsai_48h-1/final_bonsai_zscore0.75"
)
# folder written by packages/pybonsai/scripts/make_reference.sh for the same tree;
# skipped if missing
REFERENCE_DIR = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else Path(__file__).resolve().parents[1] / "packages/pybonsai/results/reference"
)
N_CELLS = 300  # cells sampled for the checks
SEED = 0  # sampling seed
THREADS = 8  # OpenMP threads
TREE_RTOL = 1e-6  # tree.nwk stores rounded branch lengths
FLOAT32_RTOL = 2e-7  # the reference files are float32
# --------------------------------------------------------------------------

output = BonsaiOutput(BONSAI_OUTPUT_FOLDER, topology_only=True)
leaf_ids = [node_id for node_id, leaf in zip(output.node_ids, output.is_sample) if leaf]
rng = np.random.default_rng(SEED)
# shuffled, so the result has to follow cell_ids order rather than tree order
cell_ids = [leaf_ids[i] for i in rng.choice(len(leaf_ids), size=N_CELLS, replace=False)]
print(f"{BONSAI_OUTPUT_FOLDER}\n{len(leaf_ids)} cells, checking {N_CELLS} of them")

# 1. tree distances: C++ reader of edgeInfo.txt vs pure-Python walk of tree.nwk
tree_python = get_pdists_on_tree(BONSAI_OUTPUT_FOLDER / "tree.nwk", cell_ids)
tree_cpp = output.tree_distances(nodes=cell_ids, threads=THREADS)
row_of_id = {node_id: row for row, node_id in enumerate(tree_cpp.node_ids)}
order = np.array([row_of_id[node_id] for node_id in cell_ids], dtype=np.intp)
upper = np.triu_indices(N_CELLS, 1)
# pair (i, j) of cell_ids sits at rows (order[i], order[j]) of the pybonsai result
low = np.minimum(order[upper[0]], order[upper[1]])
high = np.maximum(order[upper[0]], order[upper[1]])
tree_cpp_in_order = tree_cpp.values[
    low * (2 * N_CELLS - low - 1) // 2 + (high - low - 1)
]
np.testing.assert_allclose(tree_cpp_in_order, tree_python, rtol=TREE_RTOL)
print(f"1. tree distances match get_pdists_on_tree (rtol {TREE_RTOL})")

# 2. posterior distances vs penjing's command-line tools
posterior, posterior_sd = get_pdists_posterior(
    BONSAI_OUTPUT_FOLDER, cell_ids, with_uncertainty=True, threads=THREADS
)
if (REFERENCE_DIR / "posterior.npy").is_file():
    with open(REFERENCE_DIR / "posterior_node_ids.tsv", newline="") as handle:
        reference_ids = [
            row["node_id"] for row in csv.DictReader(handle, delimiter="\t")
        ]
    reference_row = {node_id: row for row, node_id in enumerate(reference_ids)}
    rows = np.array([reference_row[node_id] for node_id in cell_ids], dtype=np.int64)
    low = np.minimum(rows[upper[0]], rows[upper[1]])
    high = np.maximum(rows[upper[0]], rows[upper[1]])
    slots = low * (2 * len(reference_ids) - low - 1) // 2 + (high - low - 1)
    for name, got in (
        ("posterior", posterior),
        ("posterior_uncertainty", posterior_sd),
    ):
        reference = np.load(REFERENCE_DIR / f"{name}.npy", mmap_mode="r")[slots]
        np.testing.assert_allclose(got, reference, rtol=FLOAT32_RTOL)
    print(
        f"2. posterior distances and their std match {REFERENCE_DIR} (rtol {FLOAT32_RTOL})"
    )
else:
    print(f"2. skipped: no reference at {REFERENCE_DIR}")

# 3. corrected vs uncorrected distance, both against the tree distance.
# The uncorrected one is in expression units, so only its rank correlation is comparable.
uncorrected = get_bonsai_euclidean_distances(
    final_bonsai_folder=BONSAI_OUTPUT_FOLDER, cell_ids=cell_ids
)
residual = posterior - tree_python
print("3. over the sampled pairs:")
print(f"   tree distance mean                    {tree_python.mean():.4f}")
print(f"   posterior distance mean (corrected)   {posterior.mean():.4f}")
print(
    f"   Spearman tree vs corrected            {spearmanr(tree_python, posterior)[0]:.4f}"
)
print(
    f"   Spearman tree vs uncorrected          {spearmanr(tree_python, uncorrected)[0]:.4f}"
)
print(
    f"   residual (corrected - tree) mean {residual.mean():+.4f}, "
    f"in units of its std {np.mean(residual / posterior_sd):+.2f}"
)
