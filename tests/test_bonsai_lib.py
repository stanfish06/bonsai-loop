"""bonsai_lib distance functions on a hand-made C++ Bonsai output folder."""

from pathlib import Path

import numpy as np
import pytest
from pybonsai import BonsaiOutput
from scipy.spatial.distance import squareform

from bonsai_loop.bonsai_lib import (
    get_bonsai_euclidean_distances,
    get_bonsai_posteriors,
    get_pdists_on_tree,
    get_pdists_on_tree_euclidean,
    get_pdists_posterior,
)

# 64-cell Bonsai output vendored in penjing; pybonsai needs a complete output folder
PENJING_TREE = (
    Path(__file__).resolve().parents[1]
    / "packages/pybonsai/third_party/penjing/tests/data/bonsai_output/final_bonsai_tree"
)

# ((a,b)inner,c)root; rows of the arrays follow node_index
NODE_IDS = ["a", "b", "c", "inner", "root"]
NEWICK = "((a:1.0,b:2.0)inner:0.5,c:1.5)root;"
MEANS = np.array(
    [
        [0.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
        [1.0, 1.0, 1.0],
        [3.0, 2.0, 2.0],
    ]
)


@pytest.fixture
def cpp_folder(tmp_path):
    lines = ["node_index\tnode_id\tis_sample"]
    lines += [f"{i}\t{node_id}\t{int(i < 3)}" for i, node_id in enumerate(NODE_IDS)]
    (tmp_path / "vertex_list.tsv").write_text("\n".join(lines) + "\n")
    (tmp_path / "tree.nwk").write_text(NEWICK + "\n")
    np.save(tmp_path / "posterior_coords.npy", MEANS)
    np.save(tmp_path / "posterior_coords_variances.npy", MEANS / 10)
    return tmp_path


def msd(first: str, second: str) -> float:
    """Mean over genes of the squared difference between two nodes."""
    return float(
        np.mean((MEANS[NODE_IDS.index(first)] - MEANS[NODE_IDS.index(second)]) ** 2)
    )


def test_posteriors_follow_requested_order(cpp_folder):
    means, variances = get_bonsai_posteriors(cpp_folder, ["c", "inner", "a"])
    assert means.shape == (3, 3)
    np.testing.assert_allclose(means.T, MEANS[[2, 3, 0]])
    np.testing.assert_allclose(variances.T, MEANS[[2, 3, 0]] / 10)


def test_posteriors_reject_unknown_node(cpp_folder):
    with pytest.raises(KeyError, match="nope"):
        get_bonsai_posteriors(cpp_folder, ["a", "nope"])


def test_distances(cpp_folder):
    cells = ["c", "a", "b"]
    straight = squareform(get_bonsai_euclidean_distances(cpp_folder, cells))
    assert straight[0, 1] == pytest.approx(msd("c", "a"))
    assert straight[1, 2] == pytest.approx(msd("a", "b"))

    tree_t = squareform(get_pdists_on_tree(cpp_folder / "tree.nwk", cells))
    assert tree_t[1, 2] == pytest.approx(3.0)
    assert tree_t[0, 1] == pytest.approx(1.5 + 0.5 + 1.0)


def test_tree_euclidean_sums_posterior_distances_over_branches():
    if not PENJING_TREE.is_dir():
        pytest.skip(f"{PENJING_TREE} missing; run `git submodule update --init`")
    output = BonsaiOutput(PENJING_TREE)
    row_of_index = {int(index): row for row, index in enumerate(output.node_index)}
    parent_row = [row_of_index.get(int(parent), -1) for parent in output.parent_index]

    def branches_to_root(row):
        """Posterior distance of each branch above a node, keyed by the lower node."""
        branches = {}
        while parent_row[row] != -1:
            branches[row] = output.posterior_distance(
                output.node_ids[row], output.node_ids[parent_row[row]]
            )
            row = parent_row[row]
        return branches

    leaf_rows = np.flatnonzero(output.is_sample)[[0, 7, 30, 63]]
    cells = [output.node_ids[row] for row in leaf_rows]
    tree_e = squareform(
        get_pdists_on_tree_euclidean(PENJING_TREE / "tree.nwk", PENJING_TREE, cells)
    )
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            first, second = (
                branches_to_root(leaf_rows[a]),
                branches_to_root(leaf_rows[b]),
            )
            # branches above both cells are not on the path between them
            expected = sum(
                weight
                for row, weight in {**first, **second}.items()
                if (row in first) != (row in second)
            )
            assert tree_e[a, b] == pytest.approx(expected, rel=1e-7)

    # each branch's posterior distance is its fitted branch length
    tree_t = squareform(get_pdists_on_tree(PENJING_TREE / "tree.nwk", cells))
    np.testing.assert_allclose(tree_e, tree_t, rtol=1e-3)
    # the straight-line posterior distance is a different quantity
    straight = squareform(get_pdists_posterior(PENJING_TREE, cells))
    assert not np.allclose(straight, tree_t, rtol=1e-3)
