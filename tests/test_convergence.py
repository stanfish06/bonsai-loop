"""By-level distances in bonsai_loop.convergence on the Bonsai output vendored in penjing."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pybonsai import BonsaiOutput

from bonsai_loop.bonsai_lib import get_pdists_on_tree_euclidean, get_pdists_posterior
from bonsai_loop.convergence import (
    get_pdists_embedding_by_level,
    get_pdists_on_tree_by_level,
)

PENJING_TREE = (
    Path(__file__).resolve().parents[1]
    / "packages/pybonsai/third_party/penjing/tests/data/bonsai_output/final_bonsai_tree"
)


@pytest.fixture
def tree_and_lookup():
    """Stand-ins with the two things the by-level functions read: edges and levels."""
    if not PENJING_TREE.is_dir():
        pytest.skip(f"{PENJING_TREE} missing; run `git submodule update --init`")
    output = BonsaiOutput(PENJING_TREE)
    id_of_index = dict(zip(output.node_index.tolist(), output.node_ids))
    edges = pd.DataFrame(
        [
            (id_of_index[int(parent)], node_id, length)
            for node_id, parent, length in zip(
                output.node_ids, output.parent_index, output.branch_length
            )
            if parent != -1
        ],
        columns=["source", "target", "dist"],
    )
    tree = SimpleNamespace(get_edge_dataframe=lambda: edges)
    # leaves at level 0, every other node at level 1
    node_data_lookup = {
        node_id: SimpleNamespace(topological_level=0 if is_sample else 1)
        for node_id, is_sample in zip(output.node_ids, output.is_sample)
    }
    return tree, node_data_lookup


def test_tree_distances_by_level(tree_and_lookup):
    tree, node_data_lookup = tree_and_lookup
    dists_t, node_ids = get_pdists_on_tree_by_level(tree, node_data_lookup, "bonsai_t")
    dists_e, node_ids_e = get_pdists_on_tree_by_level(
        tree, node_data_lookup, "euclidean", bonsai_output_folder=PENJING_TREE
    )
    assert node_ids == node_ids_e
    assert all(node_data_lookup[node_id].topological_level == 0 for node_id in node_ids)
    np.testing.assert_allclose(
        dists_e,
        get_pdists_on_tree_euclidean(PENJING_TREE / "tree.nwk", PENJING_TREE, node_ids),
        rtol=1e-7,
    )
    # each branch's posterior distance is its fitted branch length
    np.testing.assert_allclose(dists_e, dists_t, rtol=1e-3)

    with pytest.raises(ValueError, match="bonsai_output_folder"):
        get_pdists_on_tree_by_level(tree, node_data_lookup, "euclidean")


def test_embedding_distances_by_level(tree_and_lookup):
    _, node_data_lookup = tree_and_lookup
    dists, node_ids = get_pdists_embedding_by_level(
        node_data_lookup, PENJING_TREE, level=1
    )
    assert all(node_data_lookup[node_id].topological_level == 1 for node_id in node_ids)
    np.testing.assert_array_equal(dists, get_pdists_posterior(PENJING_TREE, node_ids))
