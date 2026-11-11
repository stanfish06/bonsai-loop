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

from .sanity import bonsai_batch_ids

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
    batch_key: str | None = None,
    n_target: int | None = None,
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
    batch_key
        ``adata.obs`` column with the batch of each cell. The batches are then
        clustered separately, so that no cellstate holds cells of two batches;
        the multi-batch Sanity route needs that. Labels stay unique across
        batches.
    n_target
        Merge the cellstates further until this many are left, with the
        coarsify_cellstates program. Use it when cellstates alone merges almost
        nothing, as on deeply sequenced cells. Applies to each batch separately
        if ``batch_key`` is given. With ``work_dir``, the subfolder ``coarsify``
        keeps the labels of every intermediate number of cellstates as
        ``clustering_<k>_clusters.txt``.
    prior
        One prior parameter per gene; None lets cellstates use its default.
    seed
        Random seed. A rerun gives the same labels only with the same seed and
        the same number of threads.
    threads
        OpenMP threads; None uses half the logical cores.
    work_dir
        Folder to keep the input matrix, log and diagnostic files in, with one
        subfolder per batch if ``batch_key`` is given. By default a temporary
        folder is used and removed afterwards.
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
    coarsify_program = program.with_name("coarsify_cellstates")
    if n_target is not None:
        if n_target < 1:
            raise ValueError("n_target must be at least 1")
        if not coarsify_program.is_file():
            raise FileNotFoundError(
                f"{coarsify_program} not found; run `make cellstates` first"
            )
    counts = _counts(adata, layer)
    empty_cells = int((np.asarray(counts.sum(axis=1)).ravel() == 0).sum())
    if empty_cells:
        raise ValueError(f"{empty_cells} cells have no counts; remove them first")
    if threads is None:
        threads = max(1, (os.cpu_count() or 2) // 2)

    prior_values = None
    if prior is not None:
        prior_values = np.asarray(prior, dtype=np.float64)
        if prior_values.shape != (adata.n_vars,):
            raise ValueError(f"prior needs one value per gene ({adata.n_vars})")

    # cells of each run: all of them, or one run per batch
    if batch_key is None:
        runs = [("", np.arange(adata.n_obs))]
    else:
        if batch_key not in adata.obs:
            raise KeyError(f"adata.obs has no column {batch_key!r}")
        batch_of_cell = bonsai_batch_ids(pd.Series(adata.obs[batch_key]), batch_key)
        runs = [
            (batch, np.flatnonzero(batch_of_cell == batch))
            for batch in sorted(set(batch_of_cell))
        ]

    labels = np.empty(adata.n_obs, dtype=np.int64)
    n_cellstates = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(work_dir) if work_dir is not None else Path(temporary)
        for batch, cells in runs:
            batch_labels = _run_cellstates(
                counts[cells],
                root / batch,
                program=program,
                coarsify_program=coarsify_program,
                n_target=n_target,
                seed=seed,
                threads=threads,
                prior_values=prior_values,
                extra_args=extra_args,
                timeout=timeout,
            )
            # shift past the labels of the previous batches; a single run keeps its labels
            labels[cells] = n_cellstates + batch_labels
            n_cellstates = int(labels[cells].max()) + 1

    categories = [str(label) for label in np.unique(labels)]
    adata.obs[key_added] = pd.Categorical(labels.astype(str), categories=categories)
    adata.uns[key_added] = {
        "params": {
            "layer": layer,
            "batch_key": batch_key,
            "n_target": n_target,
            "seed": seed,
            "threads": threads,
            "extra_args": list(extra_args),
        },
        "n_cellstates": len(categories),
    }


def _run_cellstates(
    counts: sp.csr_matrix,
    folder: Path,
    *,
    program: Path,
    coarsify_program: Path,
    n_target: int | None,
    seed: int,
    threads: int,
    prior_values: np.ndarray | None,
    extra_args: Sequence[str],
    timeout: float | None,
) -> np.ndarray:
    """Run the executable on a cells x genes count matrix; returns one label per
    cell."""
    folder.mkdir(parents=True, exist_ok=True)
    # cellstates reads genes x cells and labels the columns in order
    mmwrite(folder / "counts.mtx", counts.T.tocoo(), field="integer")
    prior_file = "None"
    if prior_values is not None:
        np.savetxt(folder / "prior.txt", prior_values)
        prior_file = str(folder / "prior.txt")
    shared = ["--data", str(folder / "counts.mtx"), "--prior", prior_file, "--threads", str(threads)]  # fmt: skip
    _execute(
        [str(program), *shared, "--out-dir", str(folder), "--seed", str(seed), *extra_args],
        folder / "run.log",
        timeout,
    )  # fmt: skip
    labels_file = folder / "optimized_clusters.txt"
    labels = np.loadtxt(labels_file, dtype=np.int64, ndmin=1)

    if n_target is not None and np.unique(labels).size > n_target:
        # merge the cellstates one at a time down to n_target; every step writes its labels
        coarsify_folder = folder / "coarsify"
        coarsify_folder.mkdir(exist_ok=True)
        _execute(
            [
                str(coarsify_program), *shared,
                "--clusters", str(labels_file),
                "--n-target-clusters", str(n_target),
                "--out-dir", str(coarsify_folder),
            ],
            coarsify_folder / "run.log",
            timeout,
        )  # fmt: skip
        labels = np.loadtxt(
            coarsify_folder / f"clustering_{n_target}_clusters.txt",
            dtype=np.int64,
            ndmin=1,
        )

    n_cells = counts.indptr.size - 1
    if labels.shape != (n_cells,):
        raise RuntimeError(
            f"cellstates returned {labels.size} labels for {n_cells} cells"
        )
    return labels


def _execute(command: list[str], log_path: Path, timeout: float | None) -> None:
    """Run a program with its output in log_path; raise with the log's tail if it
    fails."""
    name = Path(command[0]).name
    with open(log_path, "w") as log:
        try:
            finished = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{name} did not finish within {timeout} s") from None
    if finished.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-15:])
        raise RuntimeError(f"{name} exited with code {finished.returncode}:\n{tail}")


def export_cellstates(
    adata: AnnData,
    out_dir,
    *,
    key: str = "cellstate",
    layer: str | None = None,
    batch_key: str | None = None,
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

    - ``batch_annotation.tsv``, only with ``batch_key``: cellstate id and batch per
      line, tab-separated without header. Numeric batch labels get the column
      name as a prefix (see :func:`bonsai_loop.sanity.bonsai_batch_ids`). scripts/run_sanity_multibatch.py takes
      it with ``Gene_table.txt``.

    Cellstate ids in these files are ``cs_0``, ``cs_1``, ... over the kept
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
    batch_key
        ``adata.obs`` column with the batch of each cell. Every cellstate must
        lie within one batch, as :func:`cellstates` with ``batch_key`` ensures.
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
    names = ["optimized_clusters.txt", "Gene_table.txt", "cell_id_to_cs_id.csv"]
    batch_of_group = None
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"adata.obs has no column {batch_key!r}")
        batch_of_cell = bonsai_batch_ids(pd.Series(adata.obs[batch_key]), batch_key)
        # batch of each cellstate, taken from its first cell
        batch_of_group = batch_of_cell[np.unique(cell_to_group, return_index=True)[1]]
        if (batch_of_group[cell_to_group] != batch_of_cell).any():
            raise ValueError(
                f"some cellstates hold cells of several batches; run cellstates with "
                f"batch_key={batch_key!r}"
            )
        names.append("batch_annotation.tsv")
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
    if batch_of_group is not None:
        pd.Series(batch_of_group[kept], index=cs_ids).to_csv(
            out_dir / "batch_annotation.tsv", sep="\t", header=False
        )
    return out_dir
