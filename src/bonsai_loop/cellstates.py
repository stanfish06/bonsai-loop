"""
Run cellstates on an AnnData and export its result for Sanity and Bonsai.

cellstates groups cells whose counts are statistically indistinguishable. The clustering
itself is the supercellstates executable, built with `make cellstates`; its algorithm
lives in its main function, so it is run as a program rather than called in-process.
"""

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.io import mmwrite

# built by `make cellstates`; the CELLSTATES_BIN environment variable overrides it
DEFAULT_EXECUTABLE = (
    Path(__file__).resolve().parents[2] / "build" / "cellstates" / "cellstates"
)


def _counts(adata: AnnData, layer: str | None) -> sp.csr_matrix:
    """Raw counts as a cells x genes integer matrix."""
    matrix = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError("adata.X is empty; pass the layer that holds the raw counts")
    counts = sp.csr_matrix(matrix)
    # cellstates parses integers only and does not stop on anything else
    if counts.nnz and (
        np.any(counts.data < 0) or np.any(counts.data != np.round(counts.data))
    ):
        where = "adata.X" if layer is None else f"layer {layer!r}"
        raise ValueError(
            f"{where} must hold raw counts (non-negative integers); it holds other values"
        )
    return counts.astype(np.int64)


def cellstates(
    adata: AnnData,
    *,
    layer: str | None = None,
    key_added: str = "cellstate",
    prior: Sequence[float] | None = None,
    seed: int = 1,
    threads: int | None = None,
    work_dir=None,
    executable=None,
    extra_args: Sequence[str] = (),
    timeout: float | None = None,
) -> None:
    """
    Cluster cells into cellstates and store the labels in ``adata.obs``.

    Parameters
    ----------
    adata
        Cells x genes, with raw counts in ``X`` or in ``layers[layer]``.
    layer
        Layer holding the raw counts; ``adata.X`` if None.
    key_added
        ``adata.obs[key_added]`` receives the cellstate label of each cell as a
        categorical ("0", "1", ...). ``adata.uns[key_added]`` receives the
        parameters and the number of cellstates.
    prior
        One prior parameter per gene; None lets cellstates use its default.
    seed
        Random seed. A rerun gives the same labels only with the same seed and
        the same number of threads.
    threads
        OpenMP threads; None uses half the logical cores.
    work_dir
        Folder to keep the input matrix, log and diagnostic files in. By default
        a temporary folder is used and removed afterwards.
    executable
        cellstates executable; defaults to the CELLSTATES_BIN environment
        variable, then to the one built by `make cellstates`.
    extra_args
        Further command-line options, e.g. ``["--coeff-c", "3.0"]``.
    timeout
        Seconds after which the run is stopped and an error raised; None waits.
    """
    program = Path(executable or os.environ.get("CELLSTATES_BIN") or DEFAULT_EXECUTABLE)
    if not program.is_file():
        raise FileNotFoundError(f"{program} not found; run `make cellstates` first")
    counts = _counts(adata, layer)
    empty_cells = int((np.asarray(counts.sum(axis=1)).ravel() == 0).sum())
    if empty_cells:
        raise ValueError(f"{empty_cells} cells have no counts; remove them first")
    if threads is None:
        threads = max(1, (os.cpu_count() or 2) // 2)

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(work_dir) if work_dir is not None else Path(temporary)
        folder.mkdir(parents=True, exist_ok=True)
        # cellstates reads genes x cells and labels the columns in order
        mmwrite(folder / "counts.mtx", counts.T.tocoo(), field="integer")
        command = [
            str(program),
            "--data", str(folder / "counts.mtx"),
            "--out-dir", str(folder),
            "--seed", str(seed),
            "--threads", str(threads),
        ]  # fmt: skip
        if prior is None:
            command += ["--prior", "None"]
        else:
            prior_values = np.asarray(prior, dtype=np.float64)
            if prior_values.shape != (adata.n_vars,):
                raise ValueError(f"prior needs one value per gene ({adata.n_vars})")
            np.savetxt(folder / "prior.txt", prior_values)
            command += ["--prior", str(folder / "prior.txt")]
        command += list(extra_args)

        log_path = folder / "run.log"
        with open(log_path, "w") as log:
            try:
                finished = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"cellstates did not finish within {timeout} s"
                ) from None
        if finished.returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-15:])
            raise RuntimeError(
                f"cellstates exited with code {finished.returncode}:\n{tail}"
            )
        labels = np.loadtxt(folder / "optimized_clusters.txt", dtype=np.int64, ndmin=1)

    if labels.shape != (adata.n_obs,):
        raise RuntimeError(
            f"cellstates returned {labels.size} labels for {adata.n_obs} cells"
        )
    categories = [str(label) for label in np.unique(labels)]
    adata.obs[key_added] = pd.Categorical(labels.astype(str), categories=categories)
    adata.uns[key_added] = {
        "params": {
            "layer": layer,
            "seed": seed,
            "threads": threads,
            "extra_args": list(extra_args),
        },
        "n_cellstates": len(categories),
    }


def export_cellstates(
    adata: AnnData,
    out_dir,
    *,
    key: str = "cellstate",
    layer: str | None = None,
    min_cells: int = 1,
    overwrite: bool = False,
) -> Path:
    """
    Write a cellstate grouping and its aggregated counts for Sanity and Bonsai.

    Files, named as in the Bonsai authors' pipeline:

    - ``optimized_clusters.txt``: the integer label of every cell, one per line in
      ``adata.obs`` order. Bonsai's ``--premerge-groups`` reads it; it needs
      Sanity run on the same cells in the same order.
    - ``Gene_table.txt``: raw counts summed per cellstate, genes x cellstates,
      tab-separated with a ``GeneID`` header. scripts/run_sanity.py reads it, and
      the tree then has cellstates as leaves.
    - ``cell_id_to_cs_id.csv``: cell id and cellstate id per line, without header,
      for the cells in kept cellstates.

    Cellstate ids in the last two files are ``cs_0``, ``cs_1``, ... over the kept
    cellstates in ascending label order, so they differ from the labels once
    ``min_cells`` drops any.

    Parameters
    ----------
    adata
        Cells x genes, with raw counts in ``X`` or in ``layers[layer]``.
    out_dir
        Folder to write; created if missing.
    key
        ``adata.obs`` column with the grouping; any labels work, not only those
        written by :func:`cellstates`.
    layer
        Layer holding the raw counts; ``adata.X`` if None.
    min_cells
        Leave cellstates with fewer cells out of ``Gene_table.txt``. The authors'
        default for the C++ Bonsai drops single-cell cellstates (``min_cells=2``).
    overwrite
        Replace files already in ``out_dir``.

    Returns
    -------
    The output folder.
    """
    if key not in adata.obs:
        raise KeyError(f"adata.obs has no column {key!r}; run cellstates first")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1")
    counts = _counts(adata, layer)
    grouping = adata.obs[key]
    if grouping.isna().to_numpy().any():
        raise ValueError(f"adata.obs[{key!r}] has missing values")

    # integer code per cell: cellstates labels as they are, other labels numbered in category order
    try:
        codes = grouping.astype(str).astype(np.int64).to_numpy()
    except ValueError:
        codes = pd.Categorical(grouping).codes.astype(np.int64)
    labels, cell_to_group, sizes = np.unique(
        codes, return_inverse=True, return_counts=True
    )
    kept = np.flatnonzero(sizes >= min_cells)
    if kept.size == 0:
        raise ValueError(f"no cellstate has at least {min_cells} cells")
    # position among the kept cellstates, -1 for cells of dropped ones
    column_of_group = np.full(labels.size, -1, dtype=np.int64)
    column_of_group[kept] = np.arange(kept.size)
    column_of_cell = column_of_group[cell_to_group]
    cs_ids = [f"cs_{column}" for column in range(kept.size)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("optimized_clusters.txt", "Gene_table.txt", "cell_id_to_cs_id.csv")
    present = [name for name in names if (out_dir / name).exists()]
    if present and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds {', '.join(present)}; pass overwrite=True"
        )

    # sum the kept cells' counts per cellstate with a cells x cellstates indicator
    cells = np.flatnonzero(column_of_cell >= 0)
    indicator = sp.csr_matrix(
        (np.ones(cells.size, dtype=np.int64), (cells, column_of_cell[cells])),
        shape=(adata.n_obs, kept.size),
    )
    summed = (counts.T @ indicator).toarray()
    table = pd.DataFrame(summed, index=adata.var_names, columns=cs_ids)
    table.to_csv(out_dir / "Gene_table.txt", sep="\t", index_label="GeneID")

    np.savetxt(out_dir / "optimized_clusters.txt", codes, fmt="%d")
    pd.Series(
        [cs_ids[column] for column in column_of_cell[cells]],
        index=adata.obs_names[cells],
    ).to_csv(out_dir / "cell_id_to_cs_id.csv", header=False)
    return out_dir
