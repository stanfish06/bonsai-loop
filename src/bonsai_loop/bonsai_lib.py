import sys
from pathlib import Path
from typing import Any, Callable, Literal, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform

_BONSAI = (
    Path(__file__).resolve().parents[2] / "third_party" / "Bonsai-data-representation"
)
if str(_BONSAI) not in sys.path:
    sys.path.insert(0, str(_BONSAI))

import bonsai
import bonsai_scout
import downstream_analyses
import paper_figure_scripts_and_notebooks as bonsai_paper
import paper_figure_scripts_and_notebooks.simulating_datasets as _bonsai_simulating_datasets
import paper_figure_scripts_and_notebooks.simulating_datasets.analyzing_simulated_datasets as _bonsai_simulated_datasets
import paper_figure_scripts_and_notebooks.simulating_datasets.analyzing_simulated_datasets.knn_recall_helpers as _bonsai_knn_recall_helpers

sys.modules[f"{__name__}.bonsai"] = bonsai
sys.modules[f"{__name__}.bonsai_scout"] = bonsai_scout
sys.modules[f"{__name__}.downstream_analyses"] = downstream_analyses
sys.modules[f"{__name__}.bonsai_paper"] = bonsai_paper
sys.modules[f"{__name__}.bonsai_paper.simulating_datasets"] = (
    _bonsai_simulating_datasets
)
sys.modules[
    f"{__name__}.bonsai_paper.simulating_datasets.analyzing_simulated_datasets"
] = _bonsai_simulated_datasets
sys.modules[
    f"{__name__}.bonsai_paper.simulating_datasets.analyzing_simulated_datasets.knn_recall_helpers"
] = _bonsai_knn_recall_helpers

# import relevant functions for distance computation
# pretend to be submodules of bonsai_lib
from .bonsai_lib.bonsai.bonsai_dataprocessing import (  # type: ignore[import]
    get_bonsai_euclidean_distances as _bonsai_euclidean_distances,
)
from .bonsai_lib.bonsai.bonsai_dataprocessing import (  # type: ignore[import]
    get_bonsai_posteriors as _bonsai_posteriors,
)
from .bonsai_lib.bonsai_scout.my_tree_layout import Layout_Tree  # type: ignore[import]
from .bonsai_lib.downstream_analyses import (  # type: ignore[import]
    get_clusters_max_diameter as bonsai_clustering,
)

WeightKind = Literal["t", "euclidean"]


def _layout_tree_from_nwk(nwk_file) -> Layout_Tree:
    tree = Layout_Tree()
    with open(nwk_file, "r") as f:
        nwk_str = f.readline()
    tree.from_newick(nwk_str=nwk_str)
    vert_ind_to_node, tree.nNodes = tree.root.renumber_verts(
        vertIndToNode={}, vert_count=0
    )
    tree.vert_ind_to_node = vert_ind_to_node
    tree.root.storeParent()
    return tree


def _tree_root(tree_or_layout_tree: Any) -> Any:
    return (
        tree_or_layout_tree.root
        if hasattr(tree_or_layout_tree, "root")
        else tree_or_layout_tree
    )


def _index_tree_nodes(root: Any) -> tuple[list[Any], dict[str, int], np.ndarray]:
    root.storeParent()
    nodes: list[Any] = []
    node_id_to_idx: dict[str, int] = {}
    parent_indices: list[int] = []

    def _visit(node: Any, parent_idx: int) -> None:
        idx = len(nodes)
        node_id_to_idx[node.nodeId] = idx
        nodes.append(node)
        parent_indices.append(parent_idx)
        for child in node.childNodes:
            _visit(child, idx)

    _visit(root, -1)
    return nodes, node_id_to_idx, np.asarray(parent_indices, dtype=np.int32)


def _edge_weight_t(_parent: Any, child: Any) -> float:
    return float(child.tParent or 0.0)


def _load_bonsai_output(bonsai_output_folder):
    try:
        from pybonsai import BonsaiOutput
    except ImportError as error:
        raise ImportError(
            "posterior distances need the pybonsai C++ extension "
            "(packages/pybonsai), which is not built in this environment."
        ) from error
    return BonsaiOutput(bonsai_output_folder)


def _edge_weight_euclidean_posterior(
    bonsai_output_folder,
) -> Callable[[Any, Any], float]:
    output = _load_bonsai_output(bonsai_output_folder)

    def _weight(parent: Any, child: Any) -> float:
        return output.posterior_distance(parent.nodeId, child.nodeId)

    return _weight


def _build_dist_to_root(
    nodes: list[Any],
    parent_indices: np.ndarray,
    edge_weight: Callable[[Any, Any], float],
) -> np.ndarray:
    n_nodes = len(nodes)
    dist_to_root = np.zeros(n_nodes, dtype=np.float64)
    for idx in range(1, n_nodes):
        parent_idx = int(parent_indices[idx])
        if parent_idx < 0:
            continue
        dist_to_root[idx] = dist_to_root[parent_idx] + edge_weight(
            nodes[parent_idx], nodes[idx]
        )
    return dist_to_root


def _build_lca_tables(
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_nodes = len(parent_indices)
    max_log = max(1, int(np.ceil(np.log2(max(n_nodes, 2)))))
    depth = np.zeros(n_nodes, dtype=np.int32)
    up = np.zeros((max_log, n_nodes), dtype=np.int32)

    for idx in range(n_nodes):
        parent_idx = int(parent_indices[idx])
        if parent_idx < 0:
            up[0, idx] = idx
            depth[idx] = 0
        else:
            up[0, idx] = parent_idx
            depth[idx] = depth[parent_idx] + 1

    for log_step in range(1, max_log):
        for idx in range(n_nodes):
            up[log_step, idx] = up[log_step - 1, up[log_step - 1, idx]]

    return depth, up, max_log


def _lca(
    left_idx: int,
    right_idx: int,
    depth: np.ndarray,
    up: np.ndarray,
    max_log: int,
) -> int:
    if left_idx == right_idx:
        return left_idx
    if depth[left_idx] < depth[right_idx]:
        left_idx, right_idx = right_idx, left_idx

    depth_diff = depth[left_idx] - depth[right_idx]
    for log_step in range(max_log):
        if depth_diff & (1 << log_step):
            left_idx = up[log_step, left_idx]

    if left_idx == right_idx:
        return left_idx

    for log_step in range(max_log - 1, -1, -1):
        if up[log_step, left_idx] != up[log_step, right_idx]:
            left_idx = up[log_step, left_idx]
            right_idx = up[log_step, right_idx]

    return int(up[0, left_idx])


def _patristic_pdist(
    root: Any,
    cell_ids: list[str],
    weight_kind: WeightKind = "t",
    *,
    bonsai_output_folder=None,
) -> np.ndarray:
    """Pairwise tree-path distances using dist-to-root and LCA (O(n^2) queries)."""
    nodes, node_id_to_idx, parent_indices = _index_tree_nodes(root)
    missing = [node_id for node_id in cell_ids if node_id not in node_id_to_idx]
    if missing:
        raise KeyError(f"cell_ids not found in tree (first missing: {missing[0]!r})")

    edge_weight: Callable[[Any, Any], float]
    if weight_kind == "t":
        edge_weight = _edge_weight_t
    elif weight_kind == "euclidean":
        if bonsai_output_folder is None:
            raise ValueError(
                "bonsai_output_folder is required for euclidean tree distances."
            )
        edge_weight = _edge_weight_euclidean_posterior(bonsai_output_folder)
    else:
        raise ValueError(f"unsupported weight_kind: {weight_kind!r}")

    dist_to_root = _build_dist_to_root(nodes, parent_indices, edge_weight)
    depth, up, max_log = _build_lca_tables(parent_indices)
    leaf_indices = np.array(
        [node_id_to_idx[node_id] for node_id in cell_ids], dtype=np.int32
    )
    n_cells = len(leaf_indices)

    condensed = np.empty(n_cells * (n_cells - 1) // 2, dtype=np.float64)
    write_idx = 0
    for i in range(n_cells):
        left_idx = int(leaf_indices[i])
        for j in range(i + 1, n_cells):
            right_idx = int(leaf_indices[j])
            ancestor = _lca(left_idx, right_idx, depth, up, max_log)
            condensed[write_idx] = (
                dist_to_root[left_idx]
                + dist_to_root[right_idx]
                - 2.0 * dist_to_root[ancestor]
            )
            write_idx += 1

    return condensed


def get_bonsai_posteriors(
    final_bonsai_folder,
    vert_ids,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Posterior ltqs and their variances for vert_ids.

    Parameters
    ----------
    final_bonsai_folder
        Final Bonsai output folder. Python Bonsai output is read by Bonsai's own
        ``get_bonsai_posteriors``. C++ Bonsai output is read from
        posterior_coords.npy, posterior_coords_variances.npy and vertex_list.tsv.
    vert_ids
        Node ids to include.

    Returns
    -------
    ``(ltqs, ltqs_vars)``, both genes x vert_ids.
    """
    folder = Path(final_bonsai_folder)
    if not (folder / "posterior_coords.npy").exists():
        return _bonsai_posteriors(final_bonsai_folder, vert_ids=vert_ids)

    node_id_to_vert_ind = {}
    with open(folder / "vertex_list.tsv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            node_id_to_vert_ind[parts[1]] = int(parts[0])
    missing = [vert_id for vert_id in vert_ids if vert_id not in node_id_to_vert_ind]
    if missing:
        raise KeyError(
            f"vert_ids not found in Bonsai output (first missing: {missing[0]!r})"
        )
    vert_inds = [node_id_to_vert_ind[vert_id] for vert_id in vert_ids]
    ltqs_vert_gene = np.load(folder / "posterior_coords.npy")
    ltqs_vars_vert_gene = np.load(folder / "posterior_coords_variances.npy")
    return ltqs_vert_gene[vert_inds].T, ltqs_vars_vert_gene[vert_inds].T


def get_bonsai_euclidean_distances(
    final_bonsai_folder,
    cell_ids,
) -> np.ndarray:
    """
    Pairwise squared-Euclidean distances between posterior ltqs of cell_ids, divided by
    the number of genes. (Old function; does not match bonsai t; use
    get_pdists_posterior instead)

    Parameters
    ----------
    final_bonsai_folder
        Final Bonsai output folder (Python or C++ Bonsai).
    cell_ids
        Node ids to include in the distance matrix.
    """
    if not (Path(final_bonsai_folder) / "posterior_coords.npy").exists():
        return _bonsai_euclidean_distances(final_bonsai_folder, cell_ids)
    ltqs_by_gene_cell, _ = get_bonsai_posteriors(
        final_bonsai_folder, vert_ids=list(cell_ids)
    )
    num_dims = ltqs_by_gene_cell.shape[0]
    return pdist(ltqs_by_gene_cell.T, metric="sqeuclidean") / num_dims


def get_pdists_on_tree(
    nwk_file,
    cell_ids,
    *,
    tree=None,
) -> np.ndarray:
    """
    Pairwise Bonsai t (tParent) path distances between cell_ids.

    Parameters
    ----------
    nwk_file
        Path to tree.nwk (used when ``tree`` is not provided).
    cell_ids
        Node ids to include in the distance matrix.
    tree
        Optional pre-loaded Bonsai ``Tree`` or ``Layout_Tree``. When provided,
        the Newick file is not re-parsed.
    """
    root = (
        _tree_root(tree) if tree is not None else _layout_tree_from_nwk(nwk_file).root
    )
    return _patristic_pdist(root, list(cell_ids), weight_kind="t")


def get_pdists_on_tree_euclidean(
    nwk_file,
    bonsai_output_folder,
    cell_ids,
    *,
    tree=None,
) -> np.ndarray:
    """
    Pairwise tree-path distances with squared-Euclidean edge weights.

    Each edge weighs the posterior distance between parent and child, the same
    per-pair quantity as ``get_pdists_posterior``:
    mean((ltqs_parent - ltqs_child)^2) plus the posterior variance of that
    difference. Only tree edges are evaluated (not an all-node pdist matrix).
    On a fitted tree each edge weight equals its branch length up to the fit
    tolerance, so the result stays close to ``get_pdists_on_tree``.

    Parameters
    ----------
    nwk_file
        Path to tree.nwk (used when ``tree`` is not provided).
    bonsai_output_folder
        Final Bonsai output folder (Python or C++ Bonsai), read by pybonsai.
    cell_ids
        Node ids to include in the distance matrix.
    tree
        Optional pre-loaded Bonsai ``Tree`` or ``Layout_Tree``. When provided,
        the Newick file is not re-parsed.
    """
    root = (
        _tree_root(tree) if tree is not None else _layout_tree_from_nwk(nwk_file).root
    )
    return _patristic_pdist(
        root,
        list(cell_ids),
        weight_kind="euclidean",
        bonsai_output_folder=bonsai_output_folder,
    )


def get_pdists_posterior(
    bonsai_output_folder,
    cell_ids,
    *,
    with_uncertainty: bool = False,
    threads: int = 0,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Pairwise posterior distances between cell_ids, in the same units as
    ``get_pdists_on_tree``.

    Per pair this is the posterior mean of the true mean squared distance,
    (1/G) * sum_g [(x_a - x_b)^2 + var_a + var_b - 2 cov_ab]: the squared
    distance between posterior means plus the posterior variance of their
    difference. ``get_bonsai_euclidean_distances`` keeps only the first term,
    which underestimates the distance because posterior means vary less than
    the positions they estimate. Subtracting ``get_pdists_on_tree`` gives the
    convergence residual; negative means closer than the tree allows.

    Computed by the pybonsai extension in packages/pybonsai.

    Parameters
    ----------
    bonsai_output_folder
        Final Bonsai output folder (Python or C++ Bonsai). For Python Bonsai
        output, ``processedDatafolder`` in its metadata.json must exist.
    cell_ids
        Node ids to include in the distance matrix.
    with_uncertainty
        Also return the standard deviation of each distance estimate.
    threads
        OpenMP threads; 0 uses the default.

    Returns
    -------
    Condensed distances in ``cell_ids`` order, or ``(distances, std)`` when
    ``with_uncertainty`` is set.
    """
    cell_ids = list(cell_ids)
    result = _load_bonsai_output(bonsai_output_folder).posterior_distances(
        nodes=cell_ids, threads=threads, with_uncertainty=with_uncertainty
    )
    # pybonsai reports nodes in its own order; put pairs back in cell_ids order
    row_of_id = {node_id: row for row, node_id in enumerate(result.node_ids)}
    order = np.array([row_of_id[node_id] for node_id in cell_ids], dtype=np.intp)

    def reorder(condensed: np.ndarray) -> np.ndarray:
        return squareform(squareform(condensed)[np.ix_(order, order)], checks=False)

    # uncertainty is None unless with_uncertainty was set
    if result.uncertainty is None:
        return reorder(result.values)
    return reorder(result.values), reorder(result.uncertainty)


__all__ = [
    "Layout_Tree",
    "bonsai_clustering",
    "get_bonsai_euclidean_distances",
    "get_pdists_on_tree",
    "get_pdists_on_tree_euclidean",
    "get_pdists_posterior",
]
