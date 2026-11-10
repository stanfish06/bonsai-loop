"""Cluster cells into cellstates and write the files Sanity and Bonsai take.

Build the executable first with `make cellstates` (on NixOS: `nix develop -c make cellstates`).
Uses bonsai_loop.cellstates, which also works directly on an AnnData in a notebook.

Usage: uv run python scripts/run_cellstates.py [counts] [out_dir]
"""

import sys
import time
from pathlib import Path

import anndata
import pandas as pd
import scipy.sparse as sp

from bonsai_loop.cellstates import cellstates, export_cellstates

# ---- options -------------------------------------------------------------
# raw counts: a .h5ad file, or a tab-separated genes x cells table with gene ids in the
# first column and cell ids in the header
COUNTS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/stan/Git/micropattern/notebooks/heemskerk_data/adata_timeseries_old_48h-1_raw_expr_hvg.tsv"
)
# output folder; default sits next to the counts
OUT_DIR = Path(
    sys.argv[2] if len(sys.argv) > 2 else COUNTS.with_name(f"{COUNTS.stem}_cellstates")
)
LAYER = None  # .h5ad only: layer with the raw counts; None uses X
KEY = "cellstate"  # obs column for the labels
THREADS = 12  # OpenMP threads; the labels depend on this number
SEED = 1  # random seed
# cellstates with fewer cells are left out of Gene_table.txt; 1 keeps all, 2 drops
# single cells (the Bonsai authors' default for the C++ Bonsai)
MIN_CELLS = 1
EXTRA_ARGS: list[str] = []  # further cellstates options, e.g. ["--coeff-c", "3.0"]
OVERWRITE = False  # replace an earlier export in OUT_DIR
# --------------------------------------------------------------------------

if COUNTS.suffix == ".h5ad":
    adata = anndata.read_h5ad(COUNTS)
else:
    table = pd.read_csv(COUNTS, sep="\t", index_col=0)
    adata = anndata.AnnData(
        sp.csr_matrix(table.to_numpy().T),
        obs=pd.DataFrame(index=table.columns.astype(str)),
        var=pd.DataFrame(index=table.index.astype(str)),
    )
print(f"{COUNTS}: {adata.n_obs} cells x {adata.n_vars} genes")

start = time.time()
# the work folder keeps the input matrix, the log and cellstates' diagnostic files
cellstates(
    adata,
    layer=LAYER,
    key_added=KEY,
    seed=SEED,
    threads=THREADS,
    work_dir=OUT_DIR / "work",
    extra_args=EXTRA_ARGS,
)
sizes = adata.obs[KEY].value_counts()
print(
    f"{len(sizes)} cellstates in {time.time() - start:.0f} s; largest {sizes.iloc[0]} cells, "
    f"{int((sizes == 1).sum())} single cells"
)

export_cellstates(
    adata, OUT_DIR, key=KEY, layer=LAYER, min_cells=MIN_CELLS, overwrite=OVERWRITE
)
if COUNTS.suffix == ".h5ad":
    # the input with the labels added to obs
    adata.write_h5ad(OUT_DIR / "cellstates.h5ad")
print(f"wrote {OUT_DIR}")
print(
    f"aggregated route: uv run python scripts/run_sanity.py {OUT_DIR / 'Gene_table.txt'}"
)
print(
    "premerged route:  uv run python scripts/run_bonsai_old_premerged.py "
    f"<sanity_dir of all cells> {OUT_DIR / 'optimized_clusters.txt'}"
)
