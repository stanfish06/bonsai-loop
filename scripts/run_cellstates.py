"""Cluster cells into cellstates and write the files Sanity and Bonsai take.

Build the executable first with `make cellstates` (on NixOS: `nix develop -c make cellstates`).
Uses bonsai_loop.cellstates, which also works directly on an AnnData in a notebook.

Usage:
    uv run python scripts/run_cellstates.py adata.h5ad                       # counts in X
    uv run python scripts/run_cellstates.py adata.h5ad --counts layer --key counts
    uv run python scripts/run_cellstates.py adata.h5ad --batch-key sample    # per batch
    uv run python scripts/run_cellstates.py counts.tsv                       # count table
"""

import argparse
import time
from pathlib import Path

import anndata
import pandas as pd
import scipy.sparse as sp

from bonsai_loop.cellstates import cellstates, export_cellstates
from bonsai_loop.sanity import counts_from_adata

# ---- options -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Cluster cells into cellstates and write the files Sanity and Bonsai take.",
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
    "--batch-key",
    help=".h5ad only: obs column with the batch of each cell. The batches are clustered "
    "separately and batch_annotation.tsv is written for scripts/run_sanity_multibatch.py",
)
parser.add_argument(
    "--n-target",
    type=int,
    help="merge the cellstates further until this many are left (per batch with "
    "--batch-key); for data where cellstates alone merges almost nothing",
)
parser.add_argument(
    "--out",
    type=Path,
    help="output folder; default <input stem>_cellstates next to the input",
)
parser.add_argument(
    "--obs-key", default="cellstate", help="obs column that receives the labels"
)
parser.add_argument(
    "--threads",
    type=int,
    default=12,
    help="OpenMP threads; the labels depend on this number",
)
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--min-cells",
    type=int,
    default=1,
    help="cellstates with fewer cells are left out of Gene_table.txt; 1 keeps all, 2 "
    "drops single cells (the Bonsai authors' default for the C++ Bonsai)",
)
parser.add_argument(
    "--extra-args",
    nargs=argparse.REMAINDER,
    default=[],
    help="everything after this goes to the cellstates executable, e.g. --coeff-c 3.0",
)
parser.add_argument(
    "--overwrite", action="store_true", help="replace an earlier export in --out"
)
args = parser.parse_args()
# --------------------------------------------------------------------------

is_h5ad = args.input.suffix == ".h5ad"
out_dir = args.out or args.input.with_name(f"{args.input.stem}_cellstates")
if not args.overwrite and (out_dir / "optimized_clusters.txt").exists():
    parser.error(f"{out_dir} already holds cellstates; pass --overwrite")

if is_h5ad:
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
if args.batch_key is not None and args.batch_key not in adata.obs:
    parser.error(
        f"{args.input} has no obs column {args.batch_key!r}; available: "
        + (", ".join(map(str, adata.obs.columns)) or "none")
    )
try:
    matrix, gene_ids = counts_from_adata(adata, args.counts, args.key)
except (KeyError, ValueError) as error:
    parser.error(f"{args.input}: {error.args[0]}")
# the chosen counts with the cell annotation; the labels are copied back to adata below
work = anndata.AnnData(
    sp.csr_matrix(matrix.T),
    obs=pd.DataFrame(adata.obs).copy(),
    var=pd.DataFrame(index=gene_ids),
)
print(f"{args.input}: {work.n_obs} cells x {work.n_vars} genes")

start = time.time()
# the work folder keeps the input matrix, the log and cellstates' diagnostic files
cellstates(
    work,
    key_added=args.obs_key,
    batch_key=args.batch_key,
    n_target=args.n_target,
    seed=args.seed,
    threads=args.threads,
    work_dir=out_dir / "work",
    extra_args=args.extra_args,
)
sizes = work.obs[args.obs_key].value_counts()
print(
    f"{len(sizes)} cellstates in {time.time() - start:.0f} s; largest {sizes.iloc[0]} cells, "
    f"{int((sizes == 1).sum())} single cells"
)
if args.batch_key is not None:
    per_batch = (
        pd.DataFrame(work.obs)
        .groupby(args.batch_key, observed=True)[args.obs_key]
        .nunique()
    )
    print("per batch: " + ", ".join(f"{b} {n}" for b, n in per_batch.items()))

export_cellstates(
    work,
    out_dir,
    key=args.obs_key,
    batch_key=args.batch_key,
    min_cells=args.min_cells,
    overwrite=args.overwrite,
)
if is_h5ad:
    # the input with the labels added to obs
    adata.obs[args.obs_key] = work.obs[args.obs_key]
    adata.uns[args.obs_key] = work.uns[args.obs_key]
    adata.write_h5ad(out_dir / "cellstates.h5ad")
print(f"wrote {out_dir}")
gene_table = out_dir / "Gene_table.txt"
if args.batch_key is None:
    print(f"aggregated route: uv run python scripts/run_sanity.py {gene_table}")
else:
    print(
        f"aggregated route: uv run python scripts/run_sanity_multibatch.py {gene_table} "
        f"--batch-annotation {out_dir / 'batch_annotation.tsv'}"
    )
print(
    "premerged route:  uv run python scripts/run_bonsai_old_premerged.py "
    f"<sanity_dir of all cells> {out_dir / 'optimized_clusters.txt'}"
)
