"""By-level distances in bonsai_loop.convergence on the Bonsai output vendored in penjing."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pybonsai import BonsaiOutput

from bonsai_loop.bonsai_lib import (
    Layout_Tree,
    get_pdists_on_tree_euclidean,
    get_pdists_posterior,
)
from bonsai_loop.convergence import (
    accumulate_delta_deviation_scores_along_lineage,
    compute_delta_deviation_from_parent,
    compute_tree_node_level_and_label,
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


@pytest.fixture
def rerooted_tree_and_lookup():
    """The vendored tree as Layout_Tree nodes, rerooted at the node above its first leaf."""
    if not PENJING_TREE.is_dir():
        pytest.skip(f"{PENJING_TREE} missing; run `git submodule update --init`")
    output = BonsaiOutput(PENJING_TREE)
    node_index = {node_id: i for i, node_id in enumerate(output.node_ids)}
    tree = Layout_Tree()
    tree.from_newick(
        (PENJING_TREE / "tree.nwk").read_text().strip(), node_id_to_vert_ind=node_index
    )
    tree.root.storeParent()
    first_leaf = output.node_ids[int(np.flatnonzero(output.is_sample)[0])]
    new_root = tree.get_vert_ind_to_node_DF()[node_index[first_leaf]].parentNode
    tree.reset_root(new_root.vert_ind)
    node_data_lookup = compute_tree_node_level_and_label(tree, "topological")
    return tree, node_data_lookup


def deviation(first: str, second: str) -> float:
    """D of two nodes: posterior distance along the tree path minus the direct one."""
    if first == second:
        return 0.0
    nodes = [first, second]
    tree_d = get_pdists_on_tree_euclidean(
        PENJING_TREE / "tree.nwk", PENJING_TREE, nodes
    )
    return float(tree_d[0] - get_pdists_posterior(PENJING_TREE, nodes)[0])


def test_delta_deviation_is_the_change_of_deviation_along_a_branch(
    rerooted_tree_and_lookup,
):
    tree, node_data_lookup = rerooted_tree_and_lookup
    leaves = [nid for nid, nd in node_data_lookup.items() if nd.tree_node.isLeaf]
    references = [leaves[5], leaves[40]]
    compute_delta_deviation_from_parent(
        node_data_lookup,
        reference_node_ids=references,
        normalize_by_branch_length=False,
        bonsai_output_folder=PENJING_TREE,
    )
    # every node but the root has a branch to its parent
    rows = {
        nid: nd.delta_deviation_from_parent
        for nid, nd in node_data_lookup.items()
        if nd.delta_deviation_from_parent is not None
    }
    assert set(node_data_lookup) - set(rows) == {tree.root.nodeId}

    def below(node) -> set[str]:
        found, stack = set(), [node]
        while stack:
            current = stack.pop()
            found.add(current.nodeId)
            stack.extend(current.childNodes)
        return found

    checked = 0
    for child_id in leaves[:6] + [nid for nid in rows if nid not in leaves][:6]:
        child = node_data_lookup[child_id].tree_node
        for reference in references:
            # the identity holds for references outside the subtree of the child
            if reference in below(child):
                continue
            expected = deviation(reference, child_id) - deviation(
                reference, child.parentNode.nodeId
            )
            assert rows[child_id][reference] == pytest.approx(expected, abs=1e-9)
            checked += 1
    assert checked >= 10

    with pytest.raises(ValueError, match="bonsai_output_folder"):
        compute_delta_deviation_from_parent(node_data_lookup, references)


def test_accumulated_delta_deviation_recovers_the_deviation(rerooted_tree_and_lookup):
    tree, node_data_lookup = rerooted_tree_and_lookup
    # a leaf and a reference under different children of the root
    first, second = tree.root.childNodes[0], tree.root.childNodes[-1]

    def first_leaf_below(node) -> str:
        while not node.isLeaf:
            node = node.childNodes[0]
        return node.nodeId

    leaf, reference = first_leaf_below(first), first_leaf_below(second)
    compute_delta_deviation_from_parent(
        node_data_lookup,
        reference_node_ids=[reference],
        normalize_by_branch_length=False,
        bonsai_output_folder=PENJING_TREE,
    )
    n_steps, node = 0, node_data_lookup[leaf].tree_node
    while node.parentNode is not None:
        n_steps, node = n_steps + 1, node.parentNode
    accumulated, end_node_id = accumulate_delta_deviation_scores_along_lineage(
        node_data_lookup, leaf, reference, n_steps=n_steps, direction="up"
    )
    assert end_node_id == tree.root.nodeId
    assert accumulated == pytest.approx(
        deviation(reference, end_node_id) - deviation(reference, leaf), abs=1e-9
    )
    # one more step would leave the tree
    with pytest.raises(ValueError, match="reached root"):
        accumulate_delta_deviation_scores_along_lineage(
            node_data_lookup, leaf, reference, n_steps=n_steps + 1, direction="up"
        )
