"""Run Sanity per batch and merge the batches into one input folder for Bonsai.

Same steps as the Python Bonsai's sanity_batch.py and merge_batches.py, with the Sanity
fits done by sanity-py:

1. Fit Sanity on each batch separately, and once on the counts summed per batch.
2. merge_batches.py centres every gene within each batch, stacks the batches and adds
   back the gene means from the summed-counts fit. It writes features.txt and
   standard_deviations.txt, which scripts/run_bonsai.py reads.

Usage: uv run python scripts/run_sanity_multibatch.py [counts_tsv] [batch_annotation_tsv] [out_dir]
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pysanity

from bonsai_loop.sanity import export_for_bonsai

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- options -------------------------------------------------------------
# count table: genes x cells, tab-separated, gene ids in the first column, cell ids in the header
COUNTS_TSV = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/stan/Git/micropattern/data/multibatch/counts.tsv"
)
# two tab-separated columns without header: cell id, batch id
BATCH_ANNOTATION = Path(
    sys.argv[2] if len(sys.argv) > 2 else COUNTS_TSV.with_name("batch_annotation.tsv")
)
# output folder; default sits next to the count table. It gets three subfolders:
#   sanity/<batch>/      per-batch Sanity folders, plus sanity/total_counts_per_batch/
#   bonsai_in/           merged input for scripts/run_bonsai.py
#   bonsai_out_python/   bonsai_config.yaml for a run with the Python Bonsai
OUT_DIR = Path(
    sys.argv[3]
    if len(sys.argv) > 3
    else COUNTS_TSV.with_name(f"{COUNTS_TSV.stem}_multibatch")
)
DATASET = COUNTS_TSV.parent.name  # dataset name written into bonsai_config.yaml
# how Sanity picks each gene's variance: "MAP" (Sanity 2 default), "MLE" (matches runs of
# older Sanity with `-max_v`) or "EAP". Bonsai rejects "MARG".
VARIANCE_METHOD = "MAP"
N_THREADS = 12  # OpenMP threads for the fits
MERGE = True  # run merge_batches.py after the fits
OVERWRITE = False  # refit batches and rewrite bonsai_in even if they already exist
VERBOSE = False  # print fit progress and merge_batches debug output
# --------------------------------------------------------------------------

TOTAL = "total_counts_per_batch"  # folder name merge_batches.py expects for the summed-counts fit
MERGE_SCRIPT = (
    REPO_ROOT
    / "Bonsai-data-representation"
    / "optional_preprocessing"
    / "batch_correction"
    / "merge_batches.py"
)
sanity_dir = OUT_DIR / "sanity"
bonsai_in = OUT_DIR / "bonsai_in"

counts = pd.read_csv(COUNTS_TSV, sep="\t", index_col=0)
annotation = pd.read_csv(BATCH_ANNOTATION, sep="\t", header=None, index_col=0).iloc[
    :, 0
]
print(f"{COUNTS_TSV}: {counts.shape[0]} genes x {counts.shape[1]} cells")

missing = [cell for cell in counts.columns if cell not in annotation.index]
if missing:
    raise SystemExit(
        f"{len(missing)} cells have no batch in {BATCH_ANNOTATION} (first: {missing[0]})"
    )
batch_of_cell = annotation.loc[counts.columns].astype(str)
batch_ids = sorted(batch_of_cell.unique())
if TOTAL in batch_ids or any("/" in batch_id for batch_id in batch_ids):
    raise SystemExit(f"batch ids must not contain '/' or be named {TOTAL}")
print(f"{len(batch_ids)} batches: " + ", ".join(batch_ids))

# one table per batch, then the per-batch gene totals with the batches as its columns
tables = {
    batch_id: counts.loc[:, (batch_of_cell == batch_id).to_numpy()]
    for batch_id in batch_ids
}
tables[TOTAL] = pd.DataFrame(
    {batch_id: tables[batch_id].sum(axis=1) for batch_id in batch_ids}
)

for name, table in tables.items():
    folder = sanity_dir / name
    # sanity_command.txt is written last, so its presence marks a finished fit
    if (folder / "sanity_command.txt").exists() and not OVERWRITE:
        print(f"{name}: reusing {folder}")
        continue
    # Sanity cannot fit genes or cells without counts; merge_batches.py pads the
    # genes a batch lacks so that Bonsai ignores them for that batch's cells
    empty_genes = table.sum(axis=1) <= 0
    empty_cells = table.sum(axis=0) <= 0
    table = table.loc[~empty_genes, ~empty_cells]
    start = time.time()
    result = pysanity.fit(
        table.to_numpy(dtype="float64"),
        variance_method=VARIANCE_METHOD,
        n_threads=N_THREADS,
        verbose=VERBOSE,
    )
    export_for_bonsai(
        result,
        table.index,
        table.columns,
        folder,
        variance_method=VARIANCE_METHOD,
        file_format="txt",
        overwrite=OVERWRITE,
    )
    print(
        f"{name}: {table.shape[0]} genes x {table.shape[1]} cells fitted in "
        f"{time.time() - start:.0f} s (dropped {int(empty_genes.sum())} genes and "
        f"{int(empty_cells.sum())} cells without counts)"
    )

if not MERGE:
    raise SystemExit(0)
if (bonsai_in / "features.txt").exists() and not OVERWRITE:
    raise SystemExit(f"{bonsai_in} already holds a merged input; set OVERWRITE = True")

start = time.time()
subprocess.run(
    [
        sys.executable,
        str(MERGE_SCRIPT),
        "--dataset",
        DATASET,
        "--sanity_output_folder",
        str(sanity_dir),
        "--batch_annotation_file",
        str(BATCH_ANNOTATION),
        "--bonsai_input_folder",
        str(bonsai_in),
        "--bonsai_results_folder",
        str(OUT_DIR / "bonsai_out_python"),
        "--verbose",
        str(VERBOSE),
    ],  # fmt: skip
    check=True,
)
print(f"merged in {time.time() - start:.0f} s: {bonsai_in}")
print(f"next: uv run python scripts/run_bonsai.py {bonsai_in}")
