"""Run Sanity on the raw counts of an AnnData and write a Sanity folder for Bonsai.

Uses the sanity-py bindings (``pysanity.fit``) and
``bonsai_loop.sanity.export_for_bonsai``.

Usage:
    uv run python scripts/run_sanity.py adata.h5ad                       # counts in X
    uv run python scripts/run_sanity.py adata.h5ad --counts layer --key counts
    uv run python scripts/run_sanity.py adata.h5ad --counts obsm --key raw_counts
    uv run python scripts/run_sanity.py Gene_table.txt                   # count table
"""

import argparse
import time
from pathlib import Path

import anndata
import pandas as pd
import pysanity

from bonsai_loop.sanity import counts_from_adata, export_for_bonsai

# ---- options -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Sanity on the raw counts of an AnnData and write a Sanity folder for Bonsai.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "input",
    type=Path,
    help="a .h5ad file; any other file is read as a tab-separated genes x cells count "
    "table with gene ids in the first column and cell ids in the header",
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
    help="output folder; default <input stem>_sanity next to the input",
)
parser.add_argument(
    "--variance-method",
    choices=["MAP", "MLE", "EAP"],
    default="MAP",
    help="how Sanity picks each gene's variance: MAP is the Sanity 2 default and what "
    "Bonsai recommends, MLE matches older Sanity with `-max_v true`",
)
parser.add_argument(
    "--format",
    choices=["txt", "npy"],
    default="txt",
    help="txt is read by the Python Bonsai and every bonsai_port version; npy is "
    "Sanity's Bonsai output mode, read by bonsai_port from 4387156 on",
)
parser.add_argument(
    "--threads", type=int, default=12, help="OpenMP threads for the fit"
)
parser.add_argument(
    "--drop-empty",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="drop genes and cells without counts, which Sanity cannot fit; "
    "--no-drop-empty stops instead",
)
parser.add_argument(
    "--allow-non-integer",
    action="store_true",
    help="fit values that are not whole numbers instead of stopping",
)
parser.add_argument(
    "--overwrite", action="store_true", help="replace Sanity output already in --out"
)
parser.add_argument(
    "--verbose",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="print fit progress",
)
args = parser.parse_args()
# --------------------------------------------------------------------------

out_dir = args.out or args.input.with_name(f"{args.input.stem}_sanity")
# stop before the fit rather than after it
if not args.overwrite and any(
    (out_dir / name).exists()
    for name in ("sanity_command.txt", "delta.txt", "delta.npy")
):
    parser.error(f"{out_dir} already holds Sanity output; pass --overwrite")

if args.input.suffix == ".h5ad":
    adata = anndata.read_h5ad(args.input)
else:
    if args.counts != "X" or args.key is not None:
        parser.error("--counts and --key apply to .h5ad input only")
    table = pd.read_csv(args.input, sep="\t", index_col=0)
    adata = anndata.AnnData(
        table.to_numpy().T,
        obs=pd.DataFrame(index=table.columns.astype(str)),
        var=pd.DataFrame(index=table.index.astype(str)),
    )
try:
    # stops on values that are not raw counts, e.g. a log-normalised X
    counts, gene_ids = counts_from_adata(
        adata, args.counts, args.key, allow_non_integer=args.allow_non_integer
    )
except (KeyError, ValueError) as error:
    parser.error(f"{args.input}: {error.args[0]}")
cell_ids = adata.obs_names.astype(str)
if args.input.suffix != ".h5ad":
    source = "table"
elif args.key is None:
    source = "X"
else:
    source = f"{args.counts} {args.key!r}"
print(f"{args.input} ({source}): {counts.shape[0]} genes x {counts.shape[1]} cells")
if not gene_ids.equals(adata.var_names.astype(str)):
    print(f"gene ids come from {source}: {gene_ids[0]}, ...")

empty_genes = counts.sum(axis=1) <= 0
empty_cells = counts.sum(axis=0) <= 0
if empty_genes.any() or empty_cells.any():
    if not args.drop_empty:
        raise SystemExit(
            f"{int(empty_genes.sum())} genes and {int(empty_cells.sum())} cells have no "
            "counts; remove them or drop --no-drop-empty"
        )
    counts = counts[~empty_genes][:, ~empty_cells]
    gene_ids, cell_ids = gene_ids[~empty_genes], cell_ids[~empty_cells]
    print(
        f"dropped {int(empty_genes.sum())} genes and {int(empty_cells.sum())} cells without counts"
    )

start = time.time()
result = pysanity.fit(
    counts,
    variance_method=args.variance_method,
    n_threads=args.threads,
    verbose=args.verbose,
)
print(
    f"fit took {time.time() - start:.1f} s ({args.variance_method}, {args.threads} threads)"
)

export_for_bonsai(
    result,
    gene_ids,
    cell_ids,
    out_dir,
    variance_method=args.variance_method,
    file_format=args.format,
    overwrite=args.overwrite,
)
print(f"wrote {out_dir} ({args.format})")
