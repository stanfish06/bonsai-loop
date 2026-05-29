import sys
from typing import Any, Callable, Literal

import numpy as np
from pathlib import Path

_BONSAI = Path(__file__).resolve().parents[2] / "Bonsai-data-representation"
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
from .bonsai_lib.downstream_analyses import (  # type: ignore[import]
    get_clusters_max_diameter as bonsai_clustering,
)
from .bonsai_lib.bonsai.bonsai_dataprocessing import (  # type: ignore[import]
    get_bonsai_euclidean_distances,
    get_bonsai_posteriors,
)
from .bonsai_lib.bonsai_scout.my_tree_layout import Layout_Tree  # type: ignore[import]

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


def _edge_weight_euclidean_ltqs(parent: Any, child: Any) -> float:
    parent_ltqs = parent.ltqsAIRoot
    child_ltqs = child.ltqsAIRoot
    if parent_ltqs is None or child_ltqs is None:
        raise ValueError(
            "euclidean tree distances require ltqsAIRoot on nodes; "
            "load the tree with posterior ltqs or pass bonsai_output_folder."
        )
    return float(np.mean((parent_ltqs - child_ltqs) ** 2))


def _load_ltqs_by_node_id(
    bonsai_output_folder,
    node_ids: list[str],
) -> dict[str, np.ndarray]:
    ltqs_by_gene_cell, _ = get_bonsai_posteriors(
        bonsai_output_folder, vert_ids=list(node_ids)
    )
    return {node_id: ltqs_by_gene_cell[:, i] for i, node_id in enumerate(node_ids)}


def _edge_weight_euclidean_lookup(
    ltqs_by_node_id: dict[str, np.ndarray],
) -> Callable[[Any, Any], float]:
    def _weight(parent: Any, child: Any) -> float:
        parent_ltqs = ltqs_by_node_id[parent.nodeId]
        child_ltqs = ltqs_by_node_id[child.nodeId]
        return float(np.mean((parent_ltqs - child_ltqs) ** 2))

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


def _tree_has_ltqs(nodes: list[Any]) -> bool:
    return any(getattr(node, "ltqsAIRoot", None) is not None for node in nodes)


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
    """
    Pairwise tree-path distances using dist-to-root and LCA (O(n^2) queries).

    Much faster than repeated scipy shortest-path calls when many cell_ids are
    requested, because tree distance is patristic rather than general-graph.
    """
    nodes, node_id_to_idx, parent_indices = _index_tree_nodes(root)
    missing = [node_id for node_id in cell_ids if node_id not in node_id_to_idx]
    if missing:
        raise KeyError(f"cell_ids not found in tree (first missing: {missing[0]!r})")

    edge_weight: Callable[[Any, Any], float]
    if weight_kind == "t":
        edge_weight = _edge_weight_t
    elif weight_kind == "euclidean":
        if _tree_has_ltqs(nodes):
            edge_weight = _edge_weight_euclidean_ltqs
        else:
            if bonsai_output_folder is None:
                raise ValueError(
                    "bonsai_output_folder is required for euclidean tree distances "
                    "when ltqsAIRoot is not loaded on tree nodes."
                )
            ltqs_lookup = _load_ltqs_by_node_id(
                bonsai_output_folder, [node.nodeId for node in nodes]
            )
            edge_weight = _edge_weight_euclidean_lookup(ltqs_lookup)
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

    Edge weights match the parent-child term used elsewhere in bonsai-loop:
    mean((ltqs_parent - ltqs_child)^2). Only tree edges are evaluated (not an
    all-node pdist matrix).

    Parameters
    ----------
    nwk_file
        Path to tree.nwk (used when ``tree`` is not provided).
    bonsai_output_folder
        Bonsai output folder with posterior_ltqs_vertByGene.npy (used when
        ltqsAIRoot is not on tree nodes).
    cell_ids
        Node ids to include in the distance matrix.
    tree
        Optional pre-loaded Bonsai ``Tree`` with posterior ltqs on nodes.
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


__all__ = [
    "get_pdists_on_tree_euclidean",
    "get_bonsai_euclidean_distances",
    "get_pdists_on_tree",
    "bonsai_clustering",
    "Layout_Tree",
]
