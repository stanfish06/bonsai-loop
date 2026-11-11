"""Run Sanity per batch and merge the batches into one input folder for Bonsai.

Same steps as the Python Bonsai's sanity_batch.py and merge_batches.py, with the Sanity
fits done by sanity-py:

1. Fit Sanity on each batch separately, and once on the counts summed per batch.
2. merge_batches.py centres every gene within each batch, stacks the batches and adds
   back the gene means from the summed-counts fit. It writes features.txt and
   standard_deviations.txt, which scripts/run_bonsai.py reads.

Usage:
    uv run python scripts/run_sanity_multibatch.py adata.h5ad --batch-key sample
    uv run python scripts/run_sanity_multibatch.py adata.h5ad --batch-key sample --counts layer --key counts
    uv run python scripts/run_sanity_multibatch.py counts.tsv --batch-annotation batch_annotation.tsv
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pysanity

from bonsai_loop.sanity import bonsai_batch_ids, counts_from_adata, export_for_bonsai

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- options -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Sanity per batch and merge the batches into one input folder for Bonsai.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "input",
    type=Path,
    help="a .h5ad file; any other file is read as a tab-separated genes x cells count "
    "table with gene ids in the first column and cell ids in the header",
)
parser.add_argument(
    "--batch-key", help=".h5ad only: obs column with the batch of each cell"
)
parser.add_argument(
    "--batch-annotation",
    type=Path,
    help="instead of --batch-key: a tab-separated file without header, cell id then "
    "batch id; a count table defaults to batch_annotation.tsv next to it",
)
parser.add_argument(
    "--counts",
    choices=["X", "layer", "obsm"],
    default="X",
    help="where the .h5ad holds the raw counts",
)
parser.add_argument(
    "--key",
    help="name of the layer or obsm entry; required with --counts layer or obsm",
)
parser.add_argument(
    "--out",
    type=Path,
    help="output folder; default <input stem>_multibatch next to the input. It gets "
    "sanity/<batch>/ (per-batch Sanity folders, plus sanity/total_counts_per_batch/), "
    "bonsai_in/ (merged input for scripts/run_bonsai.py) and bonsai_out_python/ "
    "(bonsai_config.yaml for a run with the Python Bonsai)",
)
parser.add_argument(
    "--dataset",
    help="dataset name written into bonsai_config.yaml; default the stem of a .h5ad, "
    "the folder name of a count table",
)
parser.add_argument(
    "--variance-method",
    choices=["MAP", "MLE", "EAP"],
    default="MAP",
    help="how Sanity picks each gene's variance: MAP is the Sanity 2 default, MLE "
    "matches older Sanity with `-max_v true`",
)
parser.add_argument(
    "--threads", type=int, default=12, help="OpenMP threads for the fits"
)
parser.add_argument(
    "--merge",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run merge_batches.py after the fits",
)
parser.add_argument(
    "--allow-non-integer",
    action="store_true",
    help="fit values that are not whole numbers instead of stopping",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="refit batches and rewrite bonsai_in even if they already exist",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="print fit progress and merge_batches debug output",
)
args = parser.parse_args()
# --------------------------------------------------------------------------

TOTAL = "total_counts_per_batch"  # folder name merge_batches.py expects for the summed-counts fit
MERGE_SCRIPT = (
    REPO_ROOT
    / "third_party"
    / "Bonsai-data-representation"
    / "optional_preprocessing"
    / "batch_correction"
    / "merge_batches.py"
)
is_h5ad = args.input.suffix == ".h5ad"
out_dir = args.out or args.input.with_name(f"{args.input.stem}_multibatch")
dataset = args.dataset or (args.input.stem if is_h5ad else args.input.parent.name)
sanity_dir = out_dir / "sanity"
bonsai_in = out_dir / "bonsai_in"

if args.batch_key is not None and args.batch_annotation is not None:
    parser.error("pass --batch-key or --batch-annotation, not both")
if is_h5ad:
    if args.batch_key is None and args.batch_annotation is None:
        parser.error("a .h5ad needs --batch-key or --batch-annotation")
    adata = anndata.read_h5ad(args.input)
else:
    if args.counts != "X" or args.key is not None or args.batch_key is not None:
        parser.error("--counts, --key and --batch-key apply to .h5ad input only")
    table = pd.read_csv(args.input, sep="\t", index_col=0)
    adata = anndata.AnnData(
        table.to_numpy().T,
        obs=pd.DataFrame(index=table.columns.astype(str)),
        var=pd.DataFrame(index=table.index.astype(str)),
    )
try:
    # stops on values that are not raw counts, e.g. a log-normalised X
    matrix, gene_ids = counts_from_adata(
        adata, args.counts, args.key, allow_non_integer=args.allow_non_integer
    )
except (KeyError, ValueError) as error:
    parser.error(f"{args.input}: {error.args[0]}")
counts = pd.DataFrame(matrix, index=gene_ids, columns=adata.obs_names.astype(str))
if not is_h5ad:
    source = "table"
elif args.key is None:
    source = "X"
else:
    source = f"{args.counts} {args.key!r}"
print(f"{args.input} ({source}): {counts.shape[0]} genes x {counts.shape[1]} cells")

# batch of each cell, in the order of the count columns
if args.batch_key is not None:
    if args.batch_key not in adata.obs:
        parser.error(
            f"{args.input} has no obs column {args.batch_key!r}; available: "
            + (", ".join(map(str, adata.obs.columns)) or "none")
        )
    try:
        # numeric batch labels get the column name as a prefix
        batch_of_cell = pd.Series(
            bonsai_batch_ids(pd.Series(adata.obs[args.batch_key]), args.batch_key),
            index=counts.columns,
        )
    except ValueError as error:
        parser.error(str(error))
    annotation_file = out_dir / "batch_annotation.tsv"
else:
    annotation_file = args.batch_annotation or args.input.with_name(
        "batch_annotation.tsv"
    )
    if not annotation_file.is_file():
        parser.error(f"{annotation_file} not found; pass --batch-annotation")
    annotation = pd.read_csv(
        annotation_file, sep="\t", header=None, index_col=0, dtype=str
    ).iloc[:, 0]
    missing = [cell for cell in counts.columns if cell not in annotation.index]
    if missing:
        parser.error(
            f"{len(missing)} cells have no batch in {annotation_file} (first: {missing[0]})"
        )
    batch_of_cell = annotation.loc[counts.columns]

batch_ids = sorted(batch_of_cell.unique())
if len(batch_ids) < 2:
    parser.error(f"found one batch ({batch_ids[0]}); use scripts/run_sanity.py")
if TOTAL in batch_ids or any("/" in batch_id for batch_id in batch_ids):
    parser.error(f"batch ids must not contain '/' or be named {TOTAL}")
as_numbers = np.asarray(
    pd.to_numeric(pd.Series(batch_ids), errors="coerce"), dtype=np.float64
)
if not np.isnan(as_numbers).any():
    # merge_batches.py reads such ids back as numbers and cannot build folder names from them
    parser.error(
        f"batch ids are all numbers ({', '.join(batch_ids)}), which merge_batches.py "
        "cannot handle; rename them, e.g. batch_0"
    )
print(f"{len(batch_ids)} batches: " + ", ".join(batch_ids))
if args.batch_key is not None:
    # merge_batches.py reads the batches from a file
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_of_cell.to_csv(annotation_file, sep="\t", header=False)

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
    if (folder / "sanity_command.txt").exists() and not args.overwrite:
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
        variance_method=args.variance_method,
        n_threads=args.threads,
        verbose=args.verbose,
    )
    export_for_bonsai(
        result,
        table.index,
        table.columns,
        folder,
        variance_method=args.variance_method,
        file_format="txt",
        overwrite=args.overwrite,
    )
    print(
        f"{name}: {table.shape[0]} genes x {table.shape[1]} cells fitted in "
        f"{time.time() - start:.0f} s (dropped {int(empty_genes.sum())} genes and "
        f"{int(empty_cells.sum())} cells without counts)"
    )

if not args.merge:
    raise SystemExit(0)
if (bonsai_in / "features.txt").exists() and not args.overwrite:
    raise SystemExit(f"{bonsai_in} already holds a merged input; pass --overwrite")

start = time.time()
subprocess.run(
    [
        sys.executable,
        str(MERGE_SCRIPT),
        "--dataset",
        dataset,
        "--sanity_output_folder",
        str(sanity_dir),
        "--batch_annotation_file",
        str(annotation_file),
        "--bonsai_input_folder",
        str(bonsai_in),
        "--bonsai_results_folder",
        str(out_dir / "bonsai_out_python"),
        "--verbose",
        str(args.verbose),
    ],  # fmt: skip
    check=True,
)
print(f"merged in {time.time() - start:.0f} s: {bonsai_in}")
print(f"next: uv run python scripts/run_bonsai.py {bonsai_in}")
