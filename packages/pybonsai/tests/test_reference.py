"""Compare the binding with penjing's own command-line tools on a real tree.

Set PYBONSAI_TREE_DIR to the Bonsai output folder and PYBONSAI_REFERENCE_DIR to the folder
scripts/make_reference.sh wrote for it. Skipped when either is unset.
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from pybonsai import BonsaiOutput

THREADS = int(os.environ.get("PYBONSAI_THREADS", "8"))
# the tools store float32, so agreement is limited to float32 rounding
FLOAT32_RTOL = 2e-7

pytestmark = pytest.mark.skipif(
    "PYBONSAI_TREE_DIR" not in os.environ or "PYBONSAI_REFERENCE_DIR" not in os.environ,
    reason="set PYBONSAI_TREE_DIR and PYBONSAI_REFERENCE_DIR (see scripts/make_reference.sh)",
)


@pytest.fixture(scope="module")
def reference():
    return Path(os.environ["PYBONSAI_REFERENCE_DIR"])


@pytest.fixture(scope="module")
def output():
    return BonsaiOutput(os.environ["PYBONSAI_TREE_DIR"])


def stored_node_ids(path):
    with open(path, newline="") as handle:
        return [row["node_id"] for row in csv.DictReader(handle, delimiter="\t")]


def test_tree_distances(output, reference):
    got = output.tree_distances(leaves_only=True, threads=THREADS)
    assert got.node_ids == stored_node_ids(reference / "tree_node_ids.tsv")
    np.testing.assert_allclose(
        got.values, np.load(reference / "tree.npy"), rtol=FLOAT32_RTOL
    )


def test_posterior_distances(output, reference):
    got = output.posterior_distances(
        leaves_only=True, threads=THREADS, with_uncertainty=True
    )
    assert got.node_ids == stored_node_ids(reference / "posterior_node_ids.tsv")
    np.testing.assert_allclose(
        got.values, np.load(reference / "posterior.npy"), rtol=FLOAT32_RTOL
    )
    np.testing.assert_allclose(
        got.uncertainty,
        np.load(reference / "posterior_uncertainty.npy"),
        rtol=FLOAT32_RTOL,
    )


def test_convergence_candidates(output, reference):
    with open(reference / "convergence_candidates.tsv", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    n_leaves = int(output.is_sample.sum())
    got = output.detect_convergence(
        leaves_only=True,
        top_k=len(rows),
        max_pairs=n_leaves * (n_leaves - 1) // 2,
        threads=THREADS,
    )
    assert got["first_node_id"] == [row["first_node_id"] for row in rows]
    assert got["second_node_id"] == [row["second_node_id"] for row in rows]
    for column in (
        "tree_distance",
        "posterior_distance",
        "residual",
        "standard_deviation",
        "z_score",
    ):
        np.testing.assert_allclose(
            got[column], [float(row[column]) for row in rows], rtol=1e-12
        )
