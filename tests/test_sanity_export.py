"""Tests for bonsai_loop.sanity: counts_from_adata and export_for_bonsai.

Two tests need external binaries and are skipped unless these are set:
SANITY_BIN (Sanity >= 2.0 command-line tool) and LIBBONSAI_BIN (bonsai_port's
libbonsai, from commit 4387156 on).
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pysanity
import pytest
import scipy.sparse as sp
from anndata import AnnData

from bonsai_loop.sanity import (
    SANITY_KERNEL_VERSION,
    counts_from_adata,
    export_for_bonsai,
)

N_GENES, N_CELLS = 40, 30


@pytest.fixture(scope="module")
def counts():
    # over-dispersed counts with gene and cell effects; every gene and cell has counts
    rng = np.random.default_rng(0)
    gene_level = rng.gamma(2.0, 2.0, size=(N_GENES, 1))
    cell_depth = rng.uniform(0.5, 2.0, size=(1, N_CELLS))
    fold = rng.lognormal(0.0, 0.6, size=(N_GENES, N_CELLS))
    table = rng.poisson(gene_level * cell_depth * fold).astype(np.float64)
    table[table.sum(axis=1) == 0, 0] = 1
    table[0, table.sum(axis=0) == 0] = 1
    return table


@pytest.fixture(scope="module")
def ids():
    return [f"gene_{g}" for g in range(N_GENES)], [f"cell_{c}" for c in range(N_CELLS)]


@pytest.fixture(scope="module")
def result(counts):
    return pysanity.fit(counts, variance_method="MAP", n_threads=2)


def header_of(folder):
    """Keys Bonsai reads from sanity_command.txt: '# key: value' lines before the command."""
    header = {}
    for line in (folder / "sanity_command.txt").read_text().splitlines():
        if not line.startswith("#"):
            break
        key, _, value = line[1:].partition(":")
        header[key.strip().lower()] = value.strip()
    return header


def test_npy_export(result, ids, tmp_path):
    genes, cells = ids
    folder = export_for_bonsai(
        result, genes, cells, tmp_path / "out", variance_method="MAP"
    )

    delta, d_delta = np.load(folder / "delta.npy"), np.load(folder / "d_delta.npy")
    mu, variance = np.load(folder / "mu.npy"), np.load(folder / "variance.npy")
    assert delta.dtype == d_delta.dtype == np.float32
    assert mu.dtype == variance.dtype == np.float64
    assert delta.shape == d_delta.shape == (N_GENES, N_CELLS)
    assert mu.shape == variance.shape == (N_GENES, 1)
    np.testing.assert_array_equal(delta, result.log_fold_change.astype(np.float32))
    # d_delta is a standard deviation; the fit returns the variance
    np.testing.assert_allclose(
        d_delta.astype(np.float64) ** 2, result.log_fold_change_error_sq, rtol=1e-6
    )
    np.testing.assert_array_equal(mu[:, 0], result.mean_ltq)
    np.testing.assert_array_equal(variance[:, 0], result.gene_variance)

    assert (folder / "geneID.txt").read_text().splitlines() == genes
    assert (folder / "cellID.txt").read_text().splitlines() == cells
    header = header_of(folder)
    assert header["sanity version"] == SANITY_KERNEL_VERSION
    assert header["method"] == "MAP"
    assert header["output mode"] == "bonsai"


def test_txt_export(result, ids, tmp_path):
    genes, cells = ids
    folder = export_for_bonsai(
        result, genes, cells, tmp_path / "out", variance_method="EAP", file_format="txt"
    )
    assert not list(folder.glob("*.npy"))
    delta = np.loadtxt(folder / "delta.txt")
    assert delta.shape == (N_GENES, N_CELLS)
    np.testing.assert_allclose(delta, result.log_fold_change, atol=5.1e-7)
    np.testing.assert_allclose(
        np.loadtxt(folder / "d_delta.txt"),
        np.sqrt(result.log_fold_change_error_sq),
        atol=5.1e-7,
    )
    np.testing.assert_allclose(
        np.loadtxt(folder / "mu.txt"), result.mean_ltq, atol=5.1e-7
    )
    np.testing.assert_allclose(
        np.loadtxt(folder / "variance.txt"), result.gene_variance, atol=5.1e-7
    )
    header = header_of(folder)
    assert header["method"] == "EAP"
    # Bonsai takes an "output mode" line as the switch to the NPY files
    assert "output mode" not in header


def test_overwrite(result, ids, tmp_path):
    genes, cells = ids
    folder = export_for_bonsai(
        result, genes, cells, tmp_path / "out", variance_method="MAP"
    )
    with pytest.raises(FileExistsError):
        export_for_bonsai(result, genes, cells, folder, variance_method="MAP")
    # switching format leaves one consistent set of files
    export_for_bonsai(
        result,
        genes,
        cells,
        folder,
        variance_method="MAP",
        file_format="txt",
        overwrite=True,
    )
    assert not list(folder.glob("*.npy"))
    assert "output mode" not in header_of(folder)


def test_rejects_bad_input(result, ids, tmp_path):
    genes, cells = ids
    with pytest.raises(ValueError, match="MARG"):
        export_for_bonsai(result, genes, cells, tmp_path / "a", variance_method="MARG")
    with pytest.raises(ValueError, match="gene_ids has"):
        export_for_bonsai(
            result, genes[:-1], cells, tmp_path / "b", variance_method="MAP"
        )
    with pytest.raises(ValueError, match="duplicates"):
        export_for_bonsai(
            result,
            genes,
            [cells[0], *cells[:-1]],
            tmp_path / "c",
            variance_method="MAP",
        )
    with pytest.raises(ValueError, match="file_format"):
        export_for_bonsai(
            result,
            genes,
            cells,
            tmp_path / "d",
            variance_method="MAP",
            file_format="csv",  # pyright: ignore[reportArgumentType]
        )
    # nothing is written when validation fails
    assert not any((tmp_path / name).exists() for name in "abcd")


@pytest.mark.skipif(
    "SANITY_BIN" not in os.environ, reason="set SANITY_BIN to the Sanity executable"
)
@pytest.mark.parametrize(
    ("file_format", "flags"), [("npy", ["--bonsai-output-mode"]), ("txt", ["-e", "1"])]
)
def test_matches_sanity_cli(counts, result, ids, tmp_path, file_format, flags):
    genes, cells = ids
    table = "\n".join(
        ["\t".join(["GeneID", *cells])]
        + [
            "\t".join([gene, *(str(int(v)) for v in row)])
            for gene, row in zip(genes, counts)
        ]
    )
    (tmp_path / "counts.tsv").write_text(table + "\n")
    (tmp_path / "cli").mkdir()
    subprocess.run(
        [
            os.environ["SANITY_BIN"],
            "-f",
            str(tmp_path / "counts.tsv"),
            "-d",
            str(tmp_path / "cli"),
            "-n",
            "2",
            *flags,
        ],
        check=True,
        capture_output=True,
    )
    folder = export_for_bonsai(
        result,
        genes,
        cells,
        tmp_path / "py",
        variance_method="MAP",
        file_format=file_format,
    )
    for stem in ("delta", "d_delta", "mu", "variance"):
        name = f"{stem}.{file_format}"
        if file_format == "npy":
            np.testing.assert_array_equal(
                np.load(folder / name), np.load(tmp_path / "cli" / name)
            )
        else:
            assert (folder / name).read_bytes() == (
                tmp_path / "cli" / name
            ).read_bytes()
    for name in ("geneID.txt", "cellID.txt"):
        assert (folder / name).read_bytes() == (tmp_path / "cli" / name).read_bytes()
    assert header_of(folder).keys() - {"timestamp"} == header_of(
        tmp_path / "cli"
    ).keys() - {"timestamp"}


@pytest.mark.skipif(
    "LIBBONSAI_BIN" not in os.environ,
    reason="set LIBBONSAI_BIN to bonsai_port's libbonsai",
)
@pytest.mark.parametrize("file_format", ["npy", "txt"])
def test_libbonsai_reads_export(result, ids, tmp_path, file_format):
    genes, cells = ids
    folder = export_for_bonsai(
        result,
        genes,
        cells,
        tmp_path / "out",
        variance_method="MAP",
        file_format=file_format,
    )
    summary = subprocess.run(
        [
            os.environ["LIBBONSAI_BIN"],
            "--data-dir",
            str(folder),
            "--input-is-sanity-output",
            "--filter-summary",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    columns = summary[0].split("\t")
    first = dict(zip(columns, summary[1].split("\t")))
    assert int(first["input_features"]) == N_GENES
    assert int(first["samples"]) == N_CELLS


@pytest.fixture
def adata(counts, ids):
    """Log-normalised X, with the raw counts in a sparse layer and three obsm entries."""
    gene_ids, cell_ids = ids
    cells_by_genes = counts.T
    data = AnnData(
        np.log1p(cells_by_genes),
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=gene_ids),
    )
    data.layers["counts"] = sp.csr_matrix(cells_by_genes.astype(np.float32))
    data.obsm["counts"] = cells_by_genes.astype(np.int64)
    data.obsm["table"] = pd.DataFrame(
        cells_by_genes, index=cell_ids, columns=[f"t_{g}" for g in range(N_GENES)]
    )
    data.obsm["narrow"] = cells_by_genes[:, :5].astype(np.int64)
    return data


def test_counts_from_adata_slots(adata, counts, ids):
    gene_ids, _ = ids
    raw_in_x = AnnData(sp.csr_matrix(counts.T), var=pd.DataFrame(index=gene_ids))
    for data, source, key in [
        (raw_in_x, "X", None),
        (adata, "layer", "counts"),
        (adata, "obsm", "counts"),
    ]:
        values, names = counts_from_adata(data, source, key)
        assert values.dtype == np.float64 and values.flags.c_contiguous
        np.testing.assert_array_equal(values, counts)  # genes x cells
        assert list(names) == gene_ids

    # a DataFrame in obsm names its own columns
    values, names = counts_from_adata(adata, "obsm", "table")
    np.testing.assert_array_equal(values, counts)
    assert list(names) == [f"t_{g}" for g in range(N_GENES)]

    # an obsm array of another width than var gets numbered ids
    values, names = counts_from_adata(adata, "obsm", "narrow")
    np.testing.assert_array_equal(values, counts[:5])
    assert list(names) == [f"narrow_{g}" for g in range(5)]


def test_counts_from_adata_rejects_bad_input(adata):
    with pytest.raises(ValueError, match="whole numbers"):
        counts_from_adata(adata)  # X is log-normalised
    values, _ = counts_from_adata(adata, allow_non_integer=True)
    np.testing.assert_allclose(values, np.asarray(adata.X).T)

    adata.layers["negative"] = -np.asarray(adata.obsm["counts"])
    with pytest.raises(ValueError, match="negative"):
        counts_from_adata(adata, "layer", "negative")
    with pytest.raises(KeyError, match="available: counts"):
        counts_from_adata(adata, "layer", "absent")
    with pytest.raises(ValueError, match="needs a key"):
        counts_from_adata(adata, "obsm")
    with pytest.raises(ValueError, match="key applies"):
        counts_from_adata(adata, "X", "counts")
