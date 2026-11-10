"""Run Sanity on a count table and write the result as a Sanity folder for Bonsai.

Uses the sanity-py bindings (``pysanity.fit``) and
``bonsai_loop.sanity.export_for_bonsai``.

Usage: uv run python scripts/run_sanity.py [counts_tsv] [out_dir]
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pysanity

from bonsai_loop.sanity import export_for_bonsai

# ---- options -------------------------------------------------------------
# count table: genes x cells, tab-separated, gene ids in the first column, cell ids in the header
COUNTS_TSV = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/stan/Git/micropattern/notebooks/heemskerk_data/adata_timeseries_old_48h-1_raw_expr_hvg.tsv"
)
# output folder; default sits next to the count table
OUT_DIR = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else COUNTS_TSV.with_name(f"{COUNTS_TSV.stem}_sanity")
)
# how Sanity picks each gene's variance: "MAP" (Sanity 2 default, recommended by Bonsai),
# "MLE" (matches runs of older Sanity with `-max_v true`) or "EAP". Bonsai rejects "MARG".
VARIANCE_METHOD = "MAP"
# "txt": read by the Python Bonsai and every bonsai_port version.
# "npy": Sanity's Bonsai output mode, smaller and faster, read by bonsai_port from 4387156 on.
FILE_FORMAT = "txt"
N_THREADS = 12  # OpenMP threads for the fit
DROP_EMPTY = True  # drop genes and cells without counts; Sanity cannot fit them
OVERWRITE = False  # replace Sanity output already in OUT_DIR
VERBOSE = True  # print fit progress
# --------------------------------------------------------------------------

counts = pd.read_csv(COUNTS_TSV, sep="\t", index_col=0)
print(f"{COUNTS_TSV}: {counts.shape[0]} genes x {counts.shape[1]} cells")

empty_genes = counts.sum(axis=1) <= 0
empty_cells = counts.sum(axis=0) <= 0
if empty_genes.any() or empty_cells.any():
    if not DROP_EMPTY:
        raise SystemExit(
            f"{int(empty_genes.sum())} genes and {int(empty_cells.sum())} cells have no counts; "
            "set DROP_EMPTY = True or remove them"
        )
    counts = counts.loc[~empty_genes, ~empty_cells]
    print(
        f"dropped {int(empty_genes.sum())} genes and {int(empty_cells.sum())} cells without counts"
    )

start = time.time()
result = pysanity.fit(
    counts.to_numpy(),
    variance_method=VARIANCE_METHOD,
    n_threads=N_THREADS,
    verbose=VERBOSE,
)
print(f"fit took {time.time() - start:.1f} s ({VARIANCE_METHOD}, {N_THREADS} threads)")

export_for_bonsai(
    result,
    counts.index,
    counts.columns,
    OUT_DIR,
    variance_method=VARIANCE_METHOD,
    file_format=FILE_FORMAT,
    overwrite=OVERWRITE,
)
print(f"wrote {OUT_DIR} ({FILE_FORMAT})")
