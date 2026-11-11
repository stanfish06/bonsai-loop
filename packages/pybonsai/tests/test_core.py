import numpy as np
import pytest
from scipy.spatial.distance import squareform

from pybonsai import BonsaiOutput


def path_length_matrix(output):
    """Pairwise sum of branch lengths, from the parent links alone."""
    row_of_index = {int(index): row for row, index in enumerate(output.node_index)}
    parent_row = [row_of_index.get(int(parent), -1) for parent in output.parent_index]

    def chain_to_root(row):
        chain, depth = {}, 0.0
        while row != -1:
            chain[row] = depth
            depth += output.branch_length[row] if parent_row[row] != -1 else 0.0
            row = parent_row[row]
        return chain

    chains = [chain_to_root(row) for row in range(output.n_nodes)]
    matrix = np.zeros((output.n_nodes, output.n_nodes), dtype=np.float64)
    for a in range(output.n_nodes):
        for b in range(a + 1, output.n_nodes):
            # the first ancestor of b that is also above a is their lowest common ancestor
            row, climbed = b, 0.0
            while row not in chains[a]:
                climbed += output.branch_length[row]
                row = parent_row[row]
            matrix[a, b] = matrix[b, a] = chains[a][row] + climbed
    return matrix


def test_structure(output):
    assert output.n_features == 1131
    assert int(output.is_sample.sum()) == 64
    assert output.n_nodes == len(output.node_ids) == len(set(output.node_ids))
    assert len(output.feature_ids) == output.n_features
    assert int((output.parent_index == -1).sum()) == 1
    assert output.dialect in ("python", "cpp")


def test_tree_distances_match_parent_links(output):
    distances = output.tree_distances()
    assert distances.node_ids == output.node_ids
    np.testing.assert_allclose(
        squareform(distances.values), path_length_matrix(output), rtol=1e-10, atol=1e-12
    )


def test_selection_is_a_submatrix(output):
    full_tree = squareform(output.tree_distances().values)
    full_posterior = squareform(output.posterior_distances().values)
    leaf_rows = np.flatnonzero(output.is_sample)

    leaves = output.tree_distances(leaves_only=True)
    assert leaves.node_ids == [output.node_ids[row] for row in leaf_rows]
    np.testing.assert_array_equal(
        squareform(leaves.values), full_tree[np.ix_(leaf_rows, leaf_rows)]
    )

    # ids given out of order come back in ascending node order
    rows = np.array([5, 70, 2, 33])
    chosen = output.posterior_distances(nodes=[output.node_ids[row] for row in rows])
    ordered = np.sort(rows)
    assert chosen.node_ids == [output.node_ids[row] for row in ordered]
    np.testing.assert_array_equal(
        squareform(chosen.values), full_posterior[np.ix_(ordered, ordered)]
    )


def test_posterior_distance_within_covariance_bounds(output):
    """D = mean[(xa-xb)^2 + va + vb - 2c] with |c| <= sqrt(va*vb) bounds D from both sides."""
    rows = np.array([0, 1, 17, 40, 63, 64, 90, output.n_nodes - 1])
    ids = [output.node_ids[row] for row in rows]
    beliefs = [output.posterior_belief(node_id) for node_id in ids]
    result = output.posterior_distances(nodes=ids, with_uncertainty=True)
    distances = squareform(result.values)
    assert np.all(result.uncertainty > 0)

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            (mean_a, var_a), (mean_b, var_b) = beliefs[a], beliefs[b]
            gap = np.mean((mean_a - mean_b) ** 2)
            lower = gap + np.mean((np.sqrt(var_a) - np.sqrt(var_b)) ** 2)
            upper = gap + np.mean((np.sqrt(var_a) + np.sqrt(var_b)) ** 2)
            assert lower * (1 - 1e-9) <= distances[a, b] <= upper * (1 + 1e-9)
            # the variance term makes it larger than the plain distance between posterior means
            assert distances[a, b] > gap


def test_posterior_distance_of_one_pair_is_an_entry_of_all_pairs(output):
    result = output.posterior_distances(with_uncertainty=True)
    distances = squareform(result.values)
    spread = squareform(result.uncertainty)
    for a, b in [(0, 1), (17, 90), (5, output.n_nodes - 1)]:
        first, second = output.node_ids[a], output.node_ids[b]
        distance, standard_deviation = output.posterior_distance(
            first, second, with_uncertainty=True
        )
        assert distance == distances[a, b]
        assert standard_deviation == spread[a, b]
        # the order of the two nodes does not matter
        np.testing.assert_allclose(
            output.posterior_distance(second, first), distance, rtol=1e-8
        )

    # a node is at distance zero from itself
    node_id = output.node_ids[3]
    assert output.posterior_distance(node_id, node_id, with_uncertainty=True) == (0, 0)


def test_thread_count_does_not_change_results(output):
    one = output.posterior_distances(threads=1, with_uncertainty=True)
    many = output.posterior_distances(threads=4, with_uncertainty=True)
    np.testing.assert_array_equal(one.values, many.values)
    np.testing.assert_array_equal(one.uncertainty, many.uncertainty)
    without = output.posterior_distances(threads=1)
    assert without.uncertainty is None
    np.testing.assert_array_equal(one.values, without.values)


def test_convergence_candidates_agree_with_pair_arrays(output):
    tree = output.tree_distances(leaves_only=True)
    posterior = output.posterior_distances(leaves_only=True, with_uncertainty=True)
    z_all = (posterior.values - tree.values) / posterior.uncertainty
    n_leaves = len(tree.node_ids)
    top_k = 25

    found = output.detect_convergence(leaves_only=True, top_k=top_k)
    assert found["n_pairs_tested"] == n_leaves * (n_leaves - 1) // 2 == z_all.size
    assert len(found["first_node_id"]) == top_k

    # ranked by |z|, and the same set of extreme values as the pair arrays give
    magnitude = np.abs(found["z_score"])
    assert np.all(np.diff(magnitude) <= 0)
    np.testing.assert_allclose(
        magnitude, np.sort(np.abs(z_all))[::-1][:top_k], rtol=1e-9
    )

    # each reported pair carries the values stored at its condensed slot
    row_of = {node_id: row for row, node_id in enumerate(tree.node_ids)}
    for k in range(top_k):
        a, b = sorted(
            (row_of[found["first_node_id"][k]], row_of[found["second_node_id"][k]])
        )
        slot = a * (2 * n_leaves - a - 1) // 2 + (b - a - 1)
        np.testing.assert_allclose(
            found["tree_distance"][k], tree.values[slot], rtol=1e-9
        )
        np.testing.assert_allclose(
            found["posterior_distance"][k], posterior.values[slot], rtol=1e-9
        )
        np.testing.assert_allclose(
            found["standard_deviation"][k], posterior.uncertainty[slot], rtol=1e-9
        )
        np.testing.assert_allclose(
            found["residual"][k], posterior.values[slot] - tree.values[slot], atol=1e-12
        )


def test_topology_only_has_trees_but_no_posteriors(tree_folder, output):
    bare = BonsaiOutput(tree_folder, topology_only=True)
    np.testing.assert_array_equal(
        bare.tree_distances().values, output.tree_distances().values
    )
    with pytest.raises(RuntimeError, match="topology_only"):
        bare.posterior_distances()
    with pytest.raises(RuntimeError, match="topology_only"):
        bare.posterior_distance(bare.node_ids[0], bare.node_ids[1])
    with pytest.raises(RuntimeError, match="topology_only"):
        bare.detect_convergence()


def test_errors(output, tmp_path):
    with pytest.raises(RuntimeError):
        BonsaiOutput(tmp_path)  # not a Bonsai output folder
    with pytest.raises(KeyError, match="no_such_node"):
        output.tree_distances(nodes=[output.node_ids[0], "no_such_node"])
    with pytest.raises(KeyError, match="no_such_node"):
        output.posterior_distance(output.node_ids[0], "no_such_node")
    with pytest.raises(ValueError, match="at least two"):
        output.tree_distances(nodes=[output.node_ids[0]])
    with pytest.raises(ValueError, match="not both"):
        output.tree_distances(nodes=output.node_ids[:3], leaves_only=True)
    with pytest.raises(RuntimeError):
        output.detect_convergence(max_pairs=10)  # refuses more pairs than max_pairs
