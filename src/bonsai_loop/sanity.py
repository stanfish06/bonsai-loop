"""Write sanity-py results in the folder layout Bonsai reads as Sanity output."""

from collections.abc import Iterable
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from pysanity import SanityResult, VarianceMethod

# Sanity version of the kernel bundled in sanity-py, which does not expose it.
# Bonsai reads this from sanity_command.txt and requires at least 2.0.0.
SANITY_KERNEL_VERSION = "2.0.0"

# Bonsai divides Sanity's prior back out of the posterior, which needs a single
# gene variance per gene. MARG marginalises over it, so Bonsai rejects it.
BONSAI_VARIANCE_METHODS = ("MAP", "MLE", "EAP")

FileFormat = Literal["npy", "txt"]
_ARRAY_STEMS = ("delta", "d_delta", "mu", "variance")
_ID_FILES = ("geneID.txt", "cellID.txt")
_COMMAND_FILE = "sanity_command.txt"


def export_for_bonsai(
    result: SanityResult,
    gene_ids: Iterable[str],
    cell_ids: Iterable[str],
    out_dir,
    *,
    variance_method: VarianceMethod,
    file_format: FileFormat = "npy",
    overwrite: bool = False,
) -> Path:
    """
    Write a ``pysanity.fit`` result as a Sanity output folder for Bonsai.

    The folder is what ``libbonsai --input-is-sanity-output --data-dir`` reads:

    - ``delta``: log fold change of each cell from the gene mean, genes x cells
    - ``d_delta``: its standard deviation, genes x cells
    - ``mu``: mean log transcription quotient per gene
    - ``variance``: gene variance per gene
    - ``geneID.txt``, ``cellID.txt``: one id per line, in row and column order
    - ``sanity_command.txt``: Sanity version, variance method and output mode

    Parameters
    ----------
    result
        Output of ``pysanity.fit`` on counts shaped genes x cells.
    gene_ids
        One id per row of the fitted counts.
    cell_ids
        One id per column of the fitted counts.
    out_dir
        Folder to write; created if missing.
    variance_method
        The ``variance_method`` passed to ``pysanity.fit``. The result does not
        record it, and Bonsai accepts only MAP, MLE or EAP.
    file_format
        ``"npy"`` writes Sanity's Bonsai output mode: float32 ``delta.npy`` and
        ``d_delta.npy``, float64 ``mu.npy`` and ``variance.npy``. bonsai_port
        reads it from commit 4387156 on. ``"txt"`` writes Sanity's extended
        text output with six decimals, which older bonsai_port and the Python
        Bonsai also read.
    overwrite
        Replace Sanity output already in ``out_dir``. Files of the other format
        are removed so the folder holds one consistent set.

    Returns
    -------
    The output folder.
    """
    if variance_method not in BONSAI_VARIANCE_METHODS:
        raise ValueError(
            f"Bonsai cannot use Sanity output fitted with variance_method="
            f"{variance_method!r}; refit with one of {BONSAI_VARIANCE_METHODS}"
        )
    if file_format not in ("npy", "txt"):
        raise ValueError(f"file_format must be 'npy' or 'txt', got {file_format!r}")

    delta = np.asarray(result.log_fold_change, dtype=np.float64)
    d_delta = np.sqrt(np.asarray(result.log_fold_change_error_sq, dtype=np.float64))
    mu = np.asarray(result.mean_ltq, dtype=np.float64)
    variance = np.asarray(result.gene_variance, dtype=np.float64)
    n_genes, n_cells = delta.shape
    if (
        d_delta.shape != delta.shape
        or mu.shape != (n_genes,)
        or variance.shape != (n_genes,)
    ):
        raise ValueError("result arrays have inconsistent shapes")

    gene_ids = [str(gene_id) for gene_id in gene_ids]
    cell_ids = [str(cell_id) for cell_id in cell_ids]
    for name, ids, expected in (
        ("gene_ids", gene_ids, n_genes),
        ("cell_ids", cell_ids, n_cells),
    ):
        if len(ids) != expected:
            raise ValueError(
                f"{name} has {len(ids)} entries, the result has {expected}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} contains duplicates")
        # ids are stored one per line
        if any(not item or "\n" in item or "\r" in item for item in ids):
            raise ValueError(f"{name} contains an empty id or a line break")
    for name, array in (
        ("delta", delta),
        ("d_delta", d_delta),
        ("mu", mu),
        ("variance", variance),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    known_files = [
        *(f"{stem}.{suffix}" for stem in _ARRAY_STEMS for suffix in ("npy", "txt")),
        *_ID_FILES,
        _COMMAND_FILE,
    ]
    present = [name for name in known_files if (out_dir / name).exists()]
    if present and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds Sanity output ({', '.join(present)}); pass overwrite=True"
        )
    # Bonsai identifies the folder by sanity_command.txt, so remove it first and
    # write it last: an interrupted export is never read as a complete one
    for name in [_COMMAND_FILE, *present]:
        (out_dir / name).unlink(missing_ok=True)

    if file_format == "npy":
        np.save(out_dir / "delta.npy", delta.astype(np.float32))
        np.save(out_dir / "d_delta.npy", d_delta.astype(np.float32))
        np.save(out_dir / "mu.npy", mu[:, None])
        np.save(out_dir / "variance.npy", variance[:, None])
    else:
        np.savetxt(out_dir / "delta.txt", delta, fmt="%.6f", delimiter="\t")
        np.savetxt(out_dir / "d_delta.txt", d_delta, fmt="%.6f", delimiter="\t")
        np.savetxt(out_dir / "mu.txt", mu, fmt="%.6f")
        np.savetxt(out_dir / "variance.txt", variance, fmt="%.6f")
    (out_dir / "geneID.txt").write_text("".join(f"{gene_id}\n" for gene_id in gene_ids))
    (out_dir / "cellID.txt").write_text("".join(f"{cell_id}\n" for cell_id in cell_ids))

    # header keys follow the Sanity command-line tool; Bonsai stops parsing at the
    # first line that does not start with '#'
    header = [
        f"# Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"# Sanity version: {SANITY_KERNEL_VERSION}",
        f"# Method: {variance_method}",
    ]
    if file_format == "npy":
        header += [
            "# Output mode: bonsai",
            "# Output: Bonsai NPY; delta/d_delta: <f4; mu/variance: <f8; "
            "C-order genes x cells; gene vectors: genes x 1; d_delta: standard deviation",
        ]
    command = f"bonsai_loop.sanity.export_for_bonsai (sanity-py {version('sanity-py')})"
    (out_dir / _COMMAND_FILE).write_text("\n".join([*header, command]) + "\n")
    return out_dir
