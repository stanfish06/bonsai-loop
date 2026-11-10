"""Tests for bonsai_loop.cellstates.

The tests that run the clustering need the executable from `make cellstates` (or the
CELLSTATES_BIN environment variable) and are skipped without it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysanity
import pytest
import scipy.sparse as sp
from anndata import AnnData

from bonsai_loop.cellstates import DEFAULT_EXECUTABLE, cellstates, export_cellstates
from bonsai_loop.sanity import export_for_bonsai

EXECUTABLE = Path(os.environ.get("CELLSTATES_BIN") or DEFAULT_EXECUTABLE)
needs_executable = pytest.mark.skipif(
    not EXECUTABLE.is_file(), reason="build the executable with `make cellstates`"
)

N_PER_GROUP, N_GENES = 30, 40


@pytest.fixture
def two_populations():
    """Two cell populations that express disjoint halves of the genes."""
    rng = np.random.default_rng(0)
    counts = np.zeros((2 * N_PER_GROUP, N_GENES), dtype=np.int64)
    half = N_GENES // 2
    counts[:N_PER_GROUP, :half] = rng.poisson(20, size=(N_PER_GROUP, half))
    counts[N_PER_GROUP:, half:] = rng.poisson(20, size=(N_PER_GROUP, half))
    adata = AnnData(sp.csr_matrix(counts))
    adata.obs_names = [f"cell_{c}" for c in range(2 * N_PER_GROUP)]
    adata.var_names = [f"gene_{g}" for g in range(N_GENES)]
    adata.obs["population"] = ["a"] * N_PER_GROUP + ["b"] * N_PER_GROUP
    return adata


@pytest.fixture
def labelled():
    """Six cells x three genes with hand-set cellstate labels."""
    counts = np.array(
        [[1, 0, 2], [0, 3, 0], [4, 0, 1], [0, 0, 7], [2, 2, 0], [1, 1, 1]],
        dtype=np.int64,
    )
    adata = AnnData(counts)
    adata.obs_names = [f"cell_{c}" for c in range(6)]
    adata.var_names = ["g0", "g1", "g2"]
    adata.obs["cellstate"] = pd.Categorical(["2", "0", "2", "5", "0", "2"])
    return adata


@needs_executable
def test_cellstates_separates_populations(two_populations, tmp_path):
    cellstates(two_populations, threads=2, work_dir=tmp_path / "work")
    labels = two_populations.obs["cellstate"]
    assert isinstance(labels.dtype, pd.CategoricalDtype)
    assert not labels.isna().any()
    # no cellstate holds cells of both populations
    mixed = two_populations.obs.groupby("cellstate", observed=True)[
        "population"
    ].nunique()
    assert (mixed == 1).all()
    info = two_populations.uns["cellstate"]
    assert info["n_cellstates"] == labels.nunique() >= 2
    assert info["params"]["threads"] == 2
    assert {"counts.mtx", "run.log", "optimized_clusters.txt"} <= {
        path.name for path in (tmp_path / "work").iterdir()
    }


@needs_executable
def test_cellstates_reproducible_and_input_forms(two_populations):
    cellstates(two_populations, threads=2, seed=3, key_added="first")
    cellstates(two_populations, threads=2, seed=3, key_added="again")
    assert list(two_populations.obs["first"]) == list(two_populations.obs["again"])

    # integer-valued floats, dense, in a layer: same counts, so the same labels
    two_populations.layers["raw"] = two_populations.X.toarray().astype(np.float32)
    cellstates(two_populations, layer="raw", threads=2, seed=3, key_added="from_layer")
    assert list(two_populations.obs["from_layer"]) == list(two_populations.obs["first"])


def test_cellstates_rejects_bad_input(two_populations, tmp_path):
    with pytest.raises(FileNotFoundError, match="make cellstates"):
        cellstates(two_populations, executable=tmp_path / "missing")
    fake = tmp_path / "program"
    fake.write_text("")
    normalised = two_populations.copy()
    normalised.X = normalised.X.astype(np.float64) / 3.0
    with pytest.raises(ValueError, match="raw counts"):
        cellstates(normalised, executable=fake)
    with_empty_cell = two_populations.copy()
    with_empty_cell.X = sp.csr_matrix(
        np.vstack(
            [two_populations.X.toarray()[:-1], np.zeros((1, N_GENES), dtype=np.int64)]
        )
    )
    with pytest.raises(ValueError, match="no counts"):
        cellstates(with_empty_cell, executable=fake)
    assert "cellstate" not in two_populations.obs


def test_export_files(labelled, tmp_path):
    folder = export_cellstates(labelled, tmp_path / "out")

    # one label per cell, in obs order
    assert (folder / "optimized_clusters.txt").read_text().split() == list("202502")

    table = pd.read_csv(folder / "Gene_table.txt", sep="\t", index_col=0)
    assert table.index.name == "GeneID"
    assert list(table.index) == ["g0", "g1", "g2"]
    # labels 0, 2, 5 become cs_0, cs_1, cs_2
    assert list(table.columns) == ["cs_0", "cs_1", "cs_2"]
    np.testing.assert_array_equal(table["cs_0"], [2, 5, 0])  # cells 1 and 4
    np.testing.assert_array_equal(table["cs_1"], [6, 1, 4])  # cells 0, 2 and 5
    np.testing.assert_array_equal(table["cs_2"], [0, 0, 7])  # cell 3
    assert table.to_numpy().sum() == labelled.X.sum()

    mapping = pd.read_csv(folder / "cell_id_to_cs_id.csv", header=None, index_col=0)[1]
    assert mapping.to_dict() == {
        "cell_0": "cs_1", "cell_1": "cs_0", "cell_2": "cs_1",
        "cell_3": "cs_2", "cell_4": "cs_0", "cell_5": "cs_1",
    }  # fmt: skip

    with pytest.raises(FileExistsError):
        export_cellstates(labelled, folder)


def test_export_min_cells_and_other_labels(labelled, tmp_path):
    # the single-cell cellstate 5 is left out of the table; the labels file keeps every cell
    folder = export_cellstates(labelled, tmp_path / "filtered", min_cells=2)
    table = pd.read_csv(folder / "Gene_table.txt", sep="\t", index_col=0)
    assert list(table.columns) == ["cs_0", "cs_1"]
    assert table.to_numpy().sum() == labelled.X.sum() - 7
    mapping = pd.read_csv(folder / "cell_id_to_cs_id.csv", header=None, index_col=0)[1]
    assert "cell_3" not in mapping.index
    assert (folder / "optimized_clusters.txt").read_text().split() == list("202502")

    # any obs column works as a grouping; text labels are numbered in category order
    labelled.obs["kind"] = ["x", "y", "x", "x", "y", "y"]
    folder = export_cellstates(labelled, tmp_path / "kind", key="kind")
    assert (folder / "optimized_clusters.txt").read_text().split() == list("010011")

    with pytest.raises(KeyError):
        export_cellstates(labelled, tmp_path / "none", key="absent")
    with pytest.raises(ValueError, match="at least"):
        export_cellstates(labelled, tmp_path / "none", min_cells=10)


@needs_executable
def test_cellstates_to_sanity_folder(two_populations, tmp_path):
    """The whole route: cluster, aggregate, fit Sanity on the aggregate, write a Sanity folder."""
    cellstates(two_populations, threads=2)
    folder = export_cellstates(two_populations, tmp_path / "cs")
    table = pd.read_csv(folder / "Gene_table.txt", sep="\t", index_col=0)
    assert table.shape == (N_GENES, two_populations.uns["cellstate"]["n_cellstates"])

    result = pysanity.fit(table.to_numpy(), variance_method="MAP", n_threads=2)
    sanity_folder = export_for_bonsai(
        result, table.index, table.columns, tmp_path / "sanity",
        variance_method="MAP", file_format="txt",
    )  # fmt: skip
    assert (sanity_folder / "cellID.txt").read_text().split() == list(table.columns)
