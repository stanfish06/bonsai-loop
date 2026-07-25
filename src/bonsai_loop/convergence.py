from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .bonsai_lib.bonsai.bonsai_treeHelpers import Tree, TreeNode  # type: ignore[import]


@dataclass
class TreeNodeExtraData:
    """
    A container to store additional properties for each node (root, internal, or leaf)

    Attributes
    ----------
    tree_node : bonsai.bonsai_treeHelpers.TreeNode
        The Bonsai node associated with this extra data.
    topological_level : int | None
        Node level measured by topology, with leaves at level 0 and internal nodes increasing toward the root.
    geometric_level : float | None
        Node level measured by Bonsai edge length rather than edge count.
    identity : dict | None
        Label composition of annotated descendant leaves.
    n_leaves : int | None
        Number of annotated descendant leaves represented in identity.
    dendrogram_coords : tuple[float, float] | None
        2D coordinate in the ladderized dendrogram representation of the tree.
            - (x, y)
                - x: sum of tree edge lengths along the branch from root
                - y: vertical position in the dendrogram
    ordering_value : float | None
        The 1D global ordering value of the node in the Bonsai phylogenetic tree, with these potential versions:
            - bonsai tree distance to a specific node (e.g. root)
            - vertical distance in the dendrogram (e.g. nodes with fewer branches (more advanved) are placed higher)
            - computed from its descendents
    delta_deviation_from_parent : Mapping[str, float] | None
        Delta deviation scores for the incoming branch parent -> current node, indexed by reference node id
        (or by integer column index).
    branch_projection : pd.DataFrame | None
        Per-gene branch projection terms for the incoming branch parent (y) -> current node (z).
        Rows are indexed by reference node id (x), columns by gene (gene_names if provided, else integer
        gene index). Each cell holds the gene's contribution to the projection of (z - y) onto (x - z),
        i.e. (z_k - y_k)(x_k - z_k) / (x - z)^t(x - z), optionally divided by the branch length. Summing a
        row over genes recovers the full normalized projection scalar for that reference.
    other_props: dict | None
        Other non-essential properties
    """

    tree_node: TreeNode
    topological_level: int | None = None
    geometric_level: float | None = None
    identity: dict[str, float] | None = None
    n_leaves: int | None = None
    ordering_value: float | None = None
    dendrogram_coords: tuple[float, float] | None = None
    delta_deviation_from_parent: Mapping[str, float] | None = None
    delta_deviation_from_parent_smooth: Mapping[str, float] | None = None
    branch_projection: pd.DataFrame | None = None
    other_props: dict[str, Any] | None = None

    def __repr__(self) -> str:
        def _print_identity(identity: dict[str, float] | None, top_n: int = 3) -> str:
            if not identity:
                return "{}"
            top = sorted(identity.items(), key=lambda kv: -kv[1])[:top_n]
            parts = [f"({k},{round(v, 2)})" for k, v in top]
            if len(identity) > top_n:
                parts.append("...")
            return "[" + ", ".join(parts) + "]"

        def _print_dict_summary(value: Mapping | None) -> str:
            return "None" if value is None else f"dict(n={len(value)})"

        attrs = {
            "tree_node": f"TreeNode(nodeId={self.tree_node.nodeId!r})",
            "topological_level": self.topological_level,
            "geometric_level": self.geometric_level,
            "identity": _print_identity(self.identity, top_n=3),
            "n_leaves": self.n_leaves,
            "ordering_value": self.ordering_value,
            "dendrogram_coords": self.dendrogram_coords,
            "delta_deviation_from_parent": _print_dict_summary(
                self.delta_deviation_from_parent
            ),
            "delta_deviation_from_parent_smooth": _print_dict_summary(
                self.delta_deviation_from_parent
            ),
            "branch_projection": (
                "None"
                if self.branch_projection is None
                else f"DataFrame(shape={self.branch_projection.shape})"
            ),
            "other_props": self.other_props,
        }

        key_width = max(len(k) for k in attrs)
        value_width = max(len(str(v)) for v in attrs.values())
        top = f"┌{'─' * (key_width + 2)}┬{'─' * (value_width + 2)}┐"
        header = f"│ {'attribute'.ljust(key_width)} │ {'value'.ljust(value_width)} │"
        sep = f"├{'─' * (key_width + 2)}┼{'─' * (value_width + 2)}┤"
        rows = [
            f"│ {k.ljust(key_width)} │ {str(v).ljust(value_width)} │"
            for k, v in attrs.items()
        ]
        bottom = f"└{'─' * (key_width + 2)}┴{'─' * (value_width + 2)}┘"
        return "\n".join([top, header, sep, *rows, bottom])

    def compute_topological_level(
        self, node_data_children: list[TreeNodeExtraData]
    ) -> None:
        """
        Helper function to compute topological node level from the leaves (level = 0).
        Level increases toward the tree root. When child sub-trees have different
        heights, compute their root's level using the substree with max height.

        ┌── C
        A   ┌── E
        └── B
            └── D
        - Levels:
            - C = E = D = 0
            - B = 1
            - A = 2 = level(B) + 1
        """

        levels = [
            child_node_data.topological_level + 1
            for child_node_data in node_data_children
            if child_node_data.topological_level is not None
        ]
        if self.topological_level is not None:
            levels.append(self.topological_level)
        if levels:
            self.topological_level = max(levels)

    def compute_identity(self, node_data_children: list[TreeNodeExtraData]) -> None:
        """
        Helper function to compute node identity from its descendents (leaves).

        For the current node, the function aggregates its children's identity
        compositions, weighted by the number of annotated leaves associated with each
        child. This is equivalent to computing the identity composition of all annotated
        leaves associated with the current node.
        """
        if not node_data_children:
            if not self.tree_node.isLeaf:
                raise ValueError(
                    f"node {self.tree_node.nodeId} has no child and it is not a leaf"
                )
            return

        n_leaves: int | None = None
        identity_count: Counter | None = None
        for child_node_data in node_data_children:
            c_n_leaves = child_node_data.n_leaves
            c_identity = child_node_data.identity
            if c_identity is None or not c_identity:
                continue
            if c_n_leaves is None:
                raise ValueError(
                    f"node {child_node_data.tree_node.nodeId} has identity {c_identity}, but it is not associated with any leaves."
                )
            n_leaves = c_n_leaves if n_leaves is None else n_leaves + c_n_leaves
            c_identity_count = Counter(
                {k: v * c_n_leaves for k, v in c_identity.items()}
            )
            identity_count = (
                c_identity_count
                if identity_count is None
                else identity_count + c_identity_count
            )
        if identity_count is not None and n_leaves is not None:
            self.n_leaves = n_leaves
            self.identity = {k: v / n_leaves for k, v in identity_count.items()}
        else:
            self.n_leaves = None
            self.identity = None


def compute_bonsai_tree_dendrogram(
    tree: Tree,
    node_data_lookup: dict[str, TreeNodeExtraData],
    ladderize_by_annotated_leaves_only: bool = False,
) -> None:
    """
    Compute Bonsai dendrogram coordinates for every node.

    This replicates the dendrogram layout used by Bonsai Scout:
        - leaf y-values are evenly spaced after ladderized tree traversal
        - internal y-values are the mean y-values of their children
        - x-values are cumulative Bonsai tParent distances from the root, rescaled
          to the default dendrogram x-range

    Parameters
    ----------
    tree : bonsai.bonsai_treeHelpers.Tree
        Bonsai tree used to compute dendrogram coordinates.
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from TreeNode.nodeId to TreeNodeExtraData.
    ladderize_by_annotated_leaves_only : bool
        Whether to ladderize child branches by annotated descendant leaf counts
        stored in TreeNodeExtraData.n_leaves. If False, ladderize by all descendant
        leaves using TreeNode.dsLeafs.

    Returns
    -------
    None
        Dendrogram coordinates are stored in TreeNodeExtraData.dendrogram_coords.
    """
    # TODO: adapt Bonsai's branching flipping algorithms here
    xlims = (-0.95, 0.95)
    ylims = (-0.95, 0.95)

    if not ladderize_by_annotated_leaves_only:
        tree.root.get_ds_info_for_ladderize()

    def _get_child_weight(node: TreeNode) -> int:
        if ladderize_by_annotated_leaves_only:
            n_leaves = node_data_lookup[node.nodeId].n_leaves
            return n_leaves if n_leaves is not None else 0
        return int(node.dsLeafs)

    def _get_ladderized_leaves(node: TreeNode) -> list[TreeNode]:
        if node.isLeaf:
            return [node]
        leafs = []
        child_nodes = sorted(node.childNodes, key=_get_child_weight)
        for child_node in child_nodes:
            leafs += _get_ladderized_leaves(child_node)
        return leafs

    x_coords: dict[str, float] = {}
    y_coords: dict[str, float] = {}

    def _compute_x_coords(node: TreeNode, x: float = 0.0) -> None:
        x_coords[node.nodeId] = x
        for child_node in node.childNodes:
            _compute_x_coords(child_node, x + float(child_node.tParent))

    def _compute_y_coords(node: TreeNode) -> float:
        if node.isLeaf:
            return y_coords[node.nodeId]
        child_y_coords = [
            _compute_y_coords(child_node) for child_node in node.childNodes
        ]
        y_coords[node.nodeId] = float(np.mean(child_y_coords))
        return y_coords[node.nodeId]

    leafs = _get_ladderized_leaves(tree.root)
    leaf_y_coords = np.linspace(ylims[0], ylims[1], len(leafs))
    for leaf, y in zip(leafs, leaf_y_coords):
        y_coords[leaf.nodeId] = float(y)

    _compute_x_coords(tree.root)
    _compute_y_coords(tree.root)

    x_max = max(x_coords.values())
    for node_id, node_data in node_data_lookup.items():
        x = (
            x_coords[node_id] / (x_max / (xlims[1] - xlims[0])) + xlims[0]
            if x_max > 0
            else xlims[0]
        )
        node_data.dendrogram_coords = (x, y_coords[node_id])


def compute_node_ordering_value(
    tree: Tree,
    node_data_lookup: dict[str, TreeNodeExtraData],
    metric: Literal["bonsai_t_to_root", "dendrogram"] = "bonsai_t_to_root",
    aggregate_metric_from_leaves: bool = False,
) -> None:
    """
    Compute a 1D ordering value for each node.

    The ordering value is stored in TreeNodeExtraData.ordering_value and can be used
    later to order nodes within a level for visualization.

    Parameters
    ----------
    tree : bonsai.bonsai_treeHelpers.Tree
        Bonsai tree used to compute ordering values.
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from TreeNode.nodeId to TreeNodeExtraData.
    metric : {"bonsai_t_to_root", "dendrogram"}
        The metric used to compute ordering_value. Currently, "bonsai_t_to_root"
        computes Bonsai tree distance from the root using tParent edge lengths.
    aggregate_metric_from_leaves : bool
        Whether to compute internal node ordering values by aggregating leaf ordering
        values.
    """
    if aggregate_metric_from_leaves:
        raise NotImplementedError("no impl for subroutine aggregate metric from leaves")
    print(f"compute node ordering using metric {metric}")
    if metric == "bonsai_t_to_root":
        root_node: TreeNode = tree.root
        edge_df: pd.DataFrame = tree.get_edge_dataframe()
        G = nx.from_pandas_edgelist(
            df=edge_df, source="source", target="target", edge_attr="dist"
        )
        tree_dists_to_root = nx.shortest_path_length(
            G, source=root_node.nodeId, weight="dist"
        )
        for node_id, node_data in node_data_lookup.items():
            node_data.ordering_value = tree_dists_to_root.get(node_id, None)
    elif metric == "dendrogram":
        for node_id, node_data in node_data_lookup.items():
            node_data.ordering_value = (
                node_data.dendrogram_coords[1]
                if node_data.dendrogram_coords is not None
                else None
            )


def compute_node_ordering(
    node_data_lookup: dict[str, TreeNodeExtraData],
    level: int = -1,
    sort_by_identity_first: bool = True,
    ascending: bool = True,
) -> list[str]:
    """
    Compute node ordering globally or for one specific level based on
    TreeNodeExtraData.ordering_value.

    Parameters
    ----------
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from node id to TreeNodeExtraData with valid ordering_value
    level : int
        The level (from leaves) to compute the ordering. The default is -1, meaning all nodes in the tree.
    sort_by_identity_first : bool
        Whether to sort by the mean ordering_value of the identity before each node's ordering_value.
        Default is True, which might be helpful to group similar nodes. For example, let
                ┌── G (dog, n = 1)
            ┌── C (100% dog, 0% cat, n = 2)
            │   └── F (dog, n = 1)
            A (75% dog, 25% cat, n = 4)
            │   ┌── E (dog, n = 1)
            └── B (50% dog, 50% cat, n = 2)
                └── D (cat, n = 1)
            - Let v_X be the ordering value of node X
                - For the leaf level, if we sort by identity first, we first compute vmean_dog and vmean_cat
                    - vmean_dog = (v_G + v_F + v_E) / 3
                    - vmean_cat = v_D
                    - then suppose vmean_cat > vmean_dog, v_G > v_D > v_E > v_F, and we sort in ascending order
                        - the final order will be [v_F, v_E, v_G, v_D]
                - For other levels or all nodes, there will be fractional identity, so compute weighted identity mean.
                - For instance, for level 1
                    - vmean_dog = (v_C + 0.5 * v_B) / (1 + 0.5)
                    - vmean_cat = 0.5 * v_B / 0.5 = v_B

    ascending : bool
        Sort by increasing or decreasing ordering_value

    Returns
    -------
    node_ids_ordered : list[str]
        A list of ordered node ids
    """
    node_data_items = [
        (node_id, node_data)
        for node_id, node_data in node_data_lookup.items()
        if level == -1 or node_data.topological_level == level
    ]

    if sort_by_identity_first:
        identity_ordering_value_sum: defaultdict[str, float] = defaultdict(float)
        identity_weight_sum: defaultdict[str, float] = defaultdict(float)
        for _, node_data in node_data_items:
            if node_data.identity is None or node_data.ordering_value is None:
                continue
            for k, v in node_data.identity.items():
                identity_ordering_value_sum[k] += v * node_data.ordering_value
                identity_weight_sum[k] += v

        identity_ordering_value = {
            k: identity_ordering_value_sum[k] / identity_weight_sum[k]
            for k in identity_weight_sum
        }
        """
        Hierarchical node ordering:
        1. First sort by identity-weighted ordering value. Nodes with identity use
           (0, weighted_identity_ordering_value); nodes with no identity use (1, 0.0)
           and are pushed to the end when ascending=True.
        2. Within the same identity-ordering key, sort by node ordering_value. Nodes
           with no ordering_value use (1, 0.0) and are pushed after nodes with valid
           ordering_value.
        The resulting order is:
        [
            identity-ordered nodes with ordering_value,
            identity-ordered nodes without ordering_value,
            no-identity nodes with ordering_value,
            no-identity nodes without ordering_value,
        ]
        """
        node_data_items = sorted(
            node_data_items,
            key=lambda x: (
                (
                    0,
                    sum(
                        identity_ordering_value[k] * v
                        for k, v in x[1].identity.items()
                        if k in identity_ordering_value
                    ),
                )
                if x[1].identity is not None
                else (1, 0.0),
                (
                    0,
                    x[1].ordering_value,
                )
                if x[1].ordering_value is not None
                else (1, 0.0),
            ),
            reverse=not ascending,
        )
    else:
        node_data_items = sorted(
            node_data_items,
            key=lambda x: (
                (0, x[1].ordering_value)
                if x[1].ordering_value is not None
                else (1, 0.0)
            ),
            reverse=not ascending,
        )

    node_ids_ordered = [node_id for node_id, _ in node_data_items]

    return node_ids_ordered


def compute_tree_node_level_and_label(
    tree: Tree,
    node_level_type: Literal["topological", "geometric"],
    label_lookup_leaves: dict[str, str] | None = None,
) -> dict[str, TreeNodeExtraData]:
    """
    Compute the tree topology/geometric level and label of each node.
    - tree is likely imbalanced, so resolve level with the deepest substree.
        ┌── C
        A   ┌── E
        └── B
            └── D
        - Levels:
            - C = E = D = 0
            - B = 1
            - A = 2 = level(B) + 1
    - for label, assuming only leaves have labels, compute descendent identity composition for internal nodes, such as:
        A (75% dog, 25% cat, n = 4)
        ├── B (100% dog, n = 1) ── C (dog, n = 1)
        ├── D (dog, n = 1)
        │   ┌── F (dog, n = 1)
        └── E (50% dog, 50% cat, n = 2)
            └── G (cat, n = 1)
    - if label_lookup_leaves is provided, leaves missing from it are treated as unknown and are excluded from internal
      identity and n_leaves aggregation.
    - to be implemented: a better version might also consider edge length, make node level geometric
    Steps:
    1. do DFS to order the computation
        - for instance, for the example above, create a stack [D, E, B ,C, A]
    2. resolve the label and level of each node

    Parameters
    ----------
    tree : bonsai.bonsai_treeHelpers.Tree
        Bonsai tree defined in bonsai.bonsai_treeHelpers (e.g. reconstructed using loadReconstructedTreeAndData)

    Returns
    -------
    node_data_lookup : dict
    a map: TreeNode.nodeId → TreeNodeExtraData
    """
    if node_level_type == "geometric":
        raise NotImplementedError(f"no impl for subroutine {node_level_type}")
    node_data_lookup: dict[str, TreeNodeExtraData] = {}

    print("compute depth-first ordering of nodes")
    root_node: TreeNode = tree.root
    print(f"root node {root_node.nodeId}")
    stack: list[TreeNode] = [root_node]
    compute_order: list[TreeNode] = []
    while stack:
        node: TreeNode = stack.pop()
        compute_order.append(node)
        node_data: TreeNodeExtraData | None = None
        if node.isLeaf:
            # if is leaf, assign valid label and level
            node_label = (
                label_lookup_leaves[node.nodeId]
                if label_lookup_leaves is not None
                and node.nodeId in label_lookup_leaves
                else None
            )
            node_data = TreeNodeExtraData(
                tree_node=node,
                topological_level=0,
                geometric_level=0.0 if node_level_type == "geometric" else None,
                identity={node_label: 1.0} if node_label is not None else None,
                n_leaves=1,
            )
        else:
            node_data = TreeNodeExtraData(tree_node=node)
        node_data_lookup[node.nodeId] = node_data

        stack.extend(node.childNodes)

    print("compute node level and label")
    for node in tqdm(reversed(compute_order)):
        node_data = node_data_lookup[node.nodeId]
        assert node_data is not None, f"node {node.nodeId} has no associated data"
        node_data_children = [
            node_data_lookup[child_node.nodeId] for child_node in node.childNodes
        ]
        if label_lookup_leaves is not None:
            node_data.compute_identity(node_data_children=node_data_children)
        if node_level_type == "topological":
            node_data.compute_topological_level(node_data_children=node_data_children)

    return node_data_lookup


def get_pdists_on_tree_by_level(
    tree: Tree,
    node_data_lookup: dict[str, TreeNodeExtraData],
    dist_type: Literal["bonsai_t", "euclidean"] = "bonsai_t",
    level: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise distances on the Bonsai tree for nodes at one topological level.

    Parameters
    ----------
    tree : bonsai.bonsai_treeHelpers.Tree
        Bonsai tree used to compute shortest-path distances.
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from TreeNode.nodeId to TreeNodeExtraData with valid topological_level.
    dist_type : {"bonsai_t", "euclidean"}
        The edge weight used for shortest-path distances:
            - "bonsai_t": use Bonsai tParent edge lengths
            - "euclidean": use squared Euclidean distances between posterior node coordinates
    level : int
        The topological level from leaves used to select nodes.

    Returns
    -------
    dists : np.ndarray
        Pairwise distances in scipy condensed pdist format.
    node_ids : list[str]
        Node ids corresponding to the order used in dists.
    """
    node_ids = [
        node_id
        for node_id, node_data in node_data_lookup.items()
        if level == -1 or node_data.topological_level == level
    ]
    edge_df: pd.DataFrame = tree.get_edge_dataframe()
    node_id_to_ind = {node_id: i for i, node_id in enumerate(node_data_lookup)}
    edge_sources = [str(x) for x in edge_df["source"].to_list()]
    edge_targets = [str(x) for x in edge_df["target"].to_list()]
    cols = np.array([node_id_to_ind[node_id] for node_id in edge_sources], dtype=int)
    rows = np.array([node_id_to_ind[node_id] for node_id in edge_targets], dtype=int)

    if dist_type == "bonsai_t":
        weights = edge_df["dist"].to_numpy(dtype=float)
    elif dist_type == "euclidean":
        weights_list = []
        for src, tgt in zip(edge_sources, edge_targets):
            src_ltqs = node_data_lookup[src].tree_node.ltqsAIRoot
            tgt_ltqs = node_data_lookup[tgt].tree_node.ltqsAIRoot
            weights_list.append(np.mean((src_ltqs - tgt_ltqs) ** 2))
        weights = np.array(weights_list, dtype=float)

    cols_complete = np.concatenate((cols, rows))
    rows_complete = np.concatenate((rows, cols))
    weights_complete = np.concatenate((weights, weights))
    distance_csr = csr_matrix(
        (weights_complete, (rows_complete, cols_complete)),
        shape=(len(node_id_to_ind), len(node_id_to_ind)),
    )
    indices = [node_id_to_ind[node_id] for node_id in node_ids]
    dists = squareform(
        shortest_path(
            distance_csr,
            method="auto",
            directed=False,
            return_predecessors=False,
            unweighted=False,
            overwrite=False,
            indices=indices,
        )[:, indices],
        checks=False,
    )
    return dists, node_ids


def get_pdists_embedding_by_level(
    node_data_lookup: dict[str, TreeNodeExtraData],
    level: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute straight-line pairwise distances for nodes at one topological level.

    Distances are squared Euclidean distances between posterior node coordinates,
    normalized by the number of dimensions.

    Parameters
    ----------
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from TreeNode.nodeId to TreeNodeExtraData with valid topological_level.
    level : int
        The topological level from leaves used to select nodes.

    Returns
    -------
    dists : np.ndarray
        Pairwise distances in scipy condensed pdist format.
    node_ids : list[str]
        Node ids corresponding to the order used in dists.
    """
    node_ids = [
        node_id
        for node_id, node_data in node_data_lookup.items()
        if level == -1 or node_data.topological_level == level
    ]
    coords = np.array(
        [node_data_lookup[node_id].tree_node.ltqsAIRoot for node_id in node_ids]
    )
    dists = pdist(coords, metric="sqeuclidean") / coords.shape[1]
    return dists, node_ids


class _DeltaDeviationRow(Mapping[str, float]):
    """
    Read-only dict-like view over a single row of a shared ΔD matrix.

    Indexing accepts either:
        - str: reference node id  -> ΔD for that reference
        - int: positional column  -> ΔD for the i-th reference
    """

    __slots__ = ("_ref_ids", "_ref_index", "_row")

    _row: np.ndarray
    _ref_ids: list[str]
    _ref_index: dict[str, int]

    def __init__(
        self,
        row: np.ndarray,
        ref_ids: list[str],
        ref_index: dict[str, int],
    ) -> None:
        self._row = row
        self._ref_ids = ref_ids
        self._ref_index = ref_index

    def __getitem__(self, key: str | int) -> float:
        if isinstance(key, (int, np.integer)):
            try:
                return float(self._row[key])
            except IndexError as e:
                raise KeyError(key) from e
        return float(self._row[self._ref_index[key]])

    def __iter__(self) -> Iterator[str]:
        return iter(self._ref_ids)

    def __len__(self) -> int:
        return len(self._ref_ids)

    def to_array(self) -> np.ndarray:
        """Return the underlying row as a numpy view into the shared ΔD matrix."""
        return self._row


def _branch_lengths(branch_nodes: list[tuple[str, TreeNode, TreeNode]]) -> np.ndarray:
    """
    Bonsai tParent per parent->child branch, for normalizing per-branch quantities.

    A zero-length branch is NaN rather than 0.0: it cannot be normalized, and NaN keeps
    the outcome the same for every branch quantity instead of yielding +/-inf for a non-
    zero numerator and NaN for a zero one.
    """
    t_parents = np.asarray([float(c.tParent) for _, c, _ in branch_nodes], dtype=float)
    t_parents[t_parents == 0.0] = np.nan
    return t_parents


def compute_branch_projection(
    node_data_lookup: dict[str, TreeNodeExtraData],
    reference_node_ids: list[str] | None = None,
    normalize_by_branch_length: bool = True,
    gene_names: list[str] | None = None,
):
    """
    Compute branch projection towards reference cells.

    An example triplet of a parent (y), child (z), and reference node (x):
        y ─── z
         ⋱  ⋰
           x
    - Project (z - y) onto (x - z)
        - (z - y)^t(x - z) / (x - z)^t(x - z)
        - Optionally, normalize by branch length, (z - y)^t(x - z) / branch_length_yz / (x - z)^t(x - z)
        - For each projection, (z - y)^t(x - z) = ΔG1_yz * ΔG1_zx + ᳟ +  ΔGp_yz * ΔGp_zx,
          so a crude way to check if a gene contributes to convergence is to see if ΔGk_yz * ΔGk_zx is positive and large
    """

    # if reference node ids provided, use as is, otherwise use all node data map's keys
    ref_ids = (
        list(node_data_lookup)
        if reference_node_ids is None
        else list(reference_node_ids)
    )

    # from the node data map, we obtain tuple (child node id, child node object, parent node object)
    branch_nodes: list[tuple[str, TreeNode, TreeNode]] = [
        (nid, nd.tree_node, nd.tree_node.parentNode)
        for nid, nd in node_data_lookup.items()
        if not nd.tree_node.isRoot and nd.tree_node.parentNode is not None
    ]

    # reset branch_projection
    for nd in node_data_lookup.values():
        nd.branch_projection = None

    inv_sqrt_d = 1.0 / np.sqrt(len(branch_nodes[0][1].ltqsAIRoot))
    Z = (
        np.stack([np.asarray(z.ltqsAIRoot, dtype=float) for _, z, _ in branch_nodes])
        * inv_sqrt_d
    )
    Y = (
        np.stack([np.asarray(y.ltqsAIRoot, dtype=float) for _, _, y in branch_nodes])
        * inv_sqrt_d
    )
    X = (
        np.stack(
            [
                np.asarray(node_data_lookup[rid].tree_node.ltqsAIRoot, dtype=float)
                for rid in ref_ids
            ]
        )
        * inv_sqrt_d
    )

    V = Z - Y

    print(
        f"compute branch projection for {len(branch_nodes)} branches × {len(ref_ids)} refs"
    )
    t_parents: np.ndarray | None = None
    if normalize_by_branch_length:
        print("normalize branch projection by branch length")
        t_parents = _branch_lengths(branch_nodes)

    # built once and shared: Index is immutable, and this avoids rebuilding the labels
    # for every branch
    row_labels = pd.Index(ref_ids)
    col_labels = None if gene_names is None else pd.Index(gene_names)

    for i, (nid, _, _) in enumerate(branch_nodes):
        diff_xz = X - Z[i]
        numerator = diff_xz * V[i]
        denominator = np.einsum("ij,ij->i", diff_xz, diff_xz)

        zero_denominator = np.where(denominator == 0.0, np.nan, denominator)
        terms = numerator / zero_denominator[:, None]
        if t_parents is not None:
            terms = terms / t_parents[i]

        node_data_lookup[nid].branch_projection = pd.DataFrame(
            terms, index=row_labels, columns=col_labels
        )


# Node filter function, need to be refactor after having a better container of nodes in the future
def select_nodes(
    node_data_lookup: dict[str, TreeNodeExtraData],
    attribute: str | None = "branch_projection",
    identity: list[str] | None = None,
    min_identity_threshold: float = 0.0,
    min_global_mean_delta_deviation_threshold: float | None = None,
) -> set[str]:
    """
    Select the nodes matching one or more criteria and return their node ids.

    Parameters
    ----------
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from TreeNode.nodeId to TreeNodeExtraData.
    attribute : str | None
        Main filter attribute; only nodes where it is populated are considered. Pass
        None to consider every node.
    identity : list[str] | None
        Node identity (TreeNodeExtraData.identity) filter
    min_identity_threshold : float
        Nodes whose TreeNodeExtraData.identity summed over the requested labels is
        >= min_identity_threshold will be kept. The default 0.0 keeps every node
        carrying at least one of them.
    min_global_mean_delta_deviation_threshold : float | None
        Nodes with global delta deviation values >=
        min_global_mean_delta_deviation_threshold will be kept, the score being the
        mean of ΔD over all stored reference nodes.

    Returns
    -------
    keep : set[str]
        Node ids matching the criteria.
    """
    if not node_data_lookup:
        return set()

    if attribute is None:
        keep = set(node_data_lookup)
    else:
        keep = {
            nid
            for nid, nd in node_data_lookup.items()
            if getattr(nd, attribute, None) is not None
        }
        if not keep:
            raise ValueError(
                f"no node has {attribute!r} set; run the compute step that populates "
                "it, or pass attribute=None to select over all nodes"
            )

    if identity is not None:
        labelled = {
            nid: ident for nid in keep if (ident := node_data_lookup[nid].identity)
        }
        if not labelled:
            raise ValueError(
                "identity requires TreeNodeExtraData.identity; run "
                "compute_tree_node_level_and_label with label_lookup_leaves first"
            )
        available = {label for ident in labelled.values() for label in ident}
        unknown = sorted(set(identity) - available)
        if unknown:
            raise ValueError(
                f"unknown identity label(s) {unknown}; available: {sorted(available)}"
            )
        # summed over the requested labels, so a list is thresholded jointly: e.g.
        # ["pgc", "endo"] at 0.8 keeps nodes that are >= 80% pgc-or-endo combined
        keep &= {
            nid
            for nid, ident in labelled.items()
            if (total := sum(ident.get(label, 0.0) for label in identity)) > 0.0
            and total >= min_identity_threshold
        }

    if min_global_mean_delta_deviation_threshold is not None:
        if all(
            nd.delta_deviation_from_parent is None for nd in node_data_lookup.values()
        ):
            raise ValueError(
                "min_global_mean_delta_deviation_threshold requires "
                "delta_deviation_from_parent; run compute_delta_deviation_from_parent "
                "first"
            )
        scores = aggregate_delta_deviation_from_parent(node_data_lookup, method="mean")
        keep &= {
            nid
            for nid, score in scores.items()
            if score >= min_global_mean_delta_deviation_threshold
        }

    return keep


def compute_delta_deviation_from_parent(
    node_data_lookup: dict[str, TreeNodeExtraData],
    reference_node_ids: list[str] | None = None,
    normalize_by_branch_length: bool = True,
) -> None:
    """
    Compute delta deviation scores ΔD for every parent→child branch against every
    reference node.

    An example triplet of a parent (y), child (z), and reference node (x):
        y ─── z
         ⋱  ⋰
           x
    - one can show that D_{xz} - D_{xy} = 2(x - y)^t(z - y) ∝ cos(θ_xyz)
        - D_{xy} = tree_d(x, y) - d(x, y)
            - tree_d(x, y): summed squared euclidean distance along the tree path between node x and y (node x ⎯ closest common ancestor between x and y ⎯ node y)
            - d(x, y): squared euclidean distance between node x and y
        - D_{xz} - D_{xy} = [tree_d(x, z) - d(x, z)] - [tree_d(x, y) - d(x, y)]
                          = ‖z - y‖^2 (the difference between tree paths x⇔z and x⇔y) - ‖x - z‖^2 + ‖x - y‖^2
                          = (z - y)^t(z - y) - (x - z)^t(x - z) + (x - y)^t(x - y)
                          = z^tz - 2y^tz + y^ty - x^tx + 2x^tz - z^tz + x^tx - 2x^ty + y^ty
                          = 2(y^ty - y^tz + x^tz - x^ty)
                          = 2(x - y)^t(z - y)
        - This means a more positive value corresponds to a smaller angle between vector (x - y) and (z - y),
            indicating a stronger convergence between x and z.
    - optionally, normalize (D_{xz} - D_{xy}) by the branch length between y and z
    - this function computes all requested branch-reference pairs. For downstream
      score aggregation, we should probably mask reference nodes that are the parent y, the child z, descendants of z, or ancestors of y.
        - x = z → D_{xz} - D_{xy} = 2‖z - y‖^2, positive artifact
        - x = y → D_{xz} - D_{xy} = 0, 0 artifact
        - x being a descendant of z or an ancestor of y, which could potentially detect cycles (e.g. x loop back to z despite being descendant of z)
            - out of scope for now

    Steps:
        0. Matrices:
            - X_m×p: m reference nodes, each with p features/genes
            - Y_n×p: n parent nodes, each with p features/genes
            - Z_n×p: n child nodes, each with p features/genes
            - all matrices will be first divided by sqrt(p)
        1. V_n×p = Z_n×p - Y_n×p, where row pair i (Zi․, Yi․) represents branch/edge i between the parent yi and child zi
        2. then, ΔD[i, j] (delta deviation for branch i with respect to reference node j) = 2 (Xj․− Yi․)^t · Vi․ = 2 Xj․^t · Vi․ − 2 Yi․^t · Vi․
        3. to compute all 2 Yi․^t · Vi․, do row-wise dot product: 2.0 * np.einsum("ij,ij->i", Y, V), yielding 1d array with n elements
        4. to compute all 2 Xj․^t · Vi․, do regular dot product: 2.0 * V @ X^t, yielding a n by m matrix
        5. then subtract each column (each reference node) of 2.0 * V @ X^t with 2.0 * np.einsum("ij,ij->i", Y, V): 2.0 * V @ X^t - 2.0 * np.einsum("ij,ij->i", Y, V)[:, None]
        6. finally get a n by m matrix ΔD

    Parameters
    ----------
    node_data_lookup : dict[str, TreeNodeExtraData]
        Map from node id to TreeNodeExtraData. Each tree_node must expose
        ltqsAIRoot, parentNode, tParent.
    reference_node_ids : list[str] | None
        Reference nodes used as columns (the "x" in ΔD). If None, all nodes in
        node_data_lookup are used (asymmetric N × N matrix).
    normalize_by_branch_length : bool
        If True, divide row i by tParent of branch i.
    """
    # if reference node ids provided, use as is, otherwise use all node data map's keys
    ref_ids = (
        list(node_data_lookup)
        if reference_node_ids is None
        else list(reference_node_ids)
    )
    ref_index = {rid: i for i, rid in enumerate(ref_ids)}

    # from the node data map, we obtain tuple (child node id, child node object, parent node object)
    branch_nodes: list[tuple[str, TreeNode, TreeNode]] = [
        (nid, nd.tree_node, nd.tree_node.parentNode)
        for nid, nd in node_data_lookup.items()
        if not nd.tree_node.isRoot and nd.tree_node.parentNode is not None
    ]

    # reset delta deviation
    for nd in node_data_lookup.values():
        nd.delta_deviation_from_parent = None

    print(f"compute deviations for {len(branch_nodes)} branches × {len(ref_ids)} refs")
    if not branch_nodes:
        return

    inv_sqrt_d = 1.0 / np.sqrt(len(branch_nodes[0][1].ltqsAIRoot))
    Z = (
        np.stack([np.asarray(z.ltqsAIRoot, dtype=float) for _, z, _ in branch_nodes])
        * inv_sqrt_d
    )
    Y = (
        np.stack([np.asarray(y.ltqsAIRoot, dtype=float) for _, _, y in branch_nodes])
        * inv_sqrt_d
    )
    V = Z - Y
    X = (
        np.stack(
            [
                np.asarray(node_data_lookup[rid].tree_node.ltqsAIRoot, dtype=float)
                for rid in ref_ids
            ]
        )
        * inv_sqrt_d
    )

    delta_d = 2.0 * (V @ X.T) - 2.0 * np.einsum("ij,ij->i", Y, V)[:, None]

    if normalize_by_branch_length:
        print("normalize delta deviation by branch length")
        delta_d /= _branch_lengths(branch_nodes)[:, None]

    for i, (nid, _, _) in enumerate(branch_nodes):
        node_data_lookup[nid].delta_deviation_from_parent = _DeltaDeviationRow(
            delta_d[i], ref_ids, ref_index
        )


def smoothen_delta_deviation(
    node_data_lookup: dict[str, TreeNodeExtraData],
    reference_node_ids: list[str] | None = None,
    tau: float = 1.0,
    cutoff_multiplier: float = 3.0,
    normalize: bool = True,
) -> None:
    """
    Compute smoothed ΔD using expoential kernel.

    For branch b, child node j: i→j with parent i and children δ(i) = {descendents}:
        - Smoothed branch ΔD'(j) = ∑_parents ΔD(p)*exp(-d(p)/τ) + ∑_children ΔD(c)*exp(-d(c)/τ)
          for distance d from midpoint, c∈δ(i)
        - If normalize == True: divide each branch by ∑exp(-d(i)/τ) for all nodes i in the sum

    Parameters
    ----------
    node_data_lookup: dict[str, TreeNodeExtraData]
       Map from TreeNodel.nodeId to TreeNodeExtraData
    reference_node_ids: list[str] | None = None
       List of reference nodes used in computing delta-deviation scores (tyically one)
    tau: float = 1.0
       Exponential kernel length
    cutoff_multiplier: float = 3.0
       Cutoff distance for calculating exponential weights to reduce compute time
    normalize: bool = True
       Set to true to divide smoothed delta deviations by total weight (accounting
           for density of children

    Returns
    -------
    None
        Smoothed delta deviations are stored in TreeNodeExtraData.delta_deviation_from_parent_smooth
    """

    def expweight(t: float) -> float:
        return float(np.exp(-t / tau))

    branches = [
        (nid, nd)
        for nid, nd in node_data_lookup.items()
        if nd.delta_deviation_from_parent is not None
    ]
    for nd in node_data_lookup.values():
        nd.delta_deviation_from_parent_smooth = None
    if not branches:
        return

    stored_ref_ids = list(
        cast(_DeltaDeviationRow, branches[0][1].delta_deviation_from_parent)
    )
    if reference_node_ids is None:
        ref_ids = stored_ref_ids
        col_sel = np.arange(len(stored_ref_ids))
    else:
        pos = {r: i for i, r in enumerate(stored_ref_ids)}
        missing = [r for r in reference_node_ids if r not in pos]
        if missing:
            raise KeyError(
                f"references not in delta_deviation_from_parent: {missing[:5]}"
            )
        ref_ids = list(reference_node_ids)
        col_sel = np.array([pos[r] for r in ref_ids], dtype=int)
    ref_index = {r: i for i, r in enumerate(ref_ids)}

    branch_ids = [nid for nid, _ in branches]
    row_of = {nid: i for i, nid in enumerate(branch_ids)}
    delta_mat = np.stack(
        [
            np.asarray(
                cast(_DeltaDeviationRow, nd.delta_deviation_from_parent).to_array(),
                dtype=float,
            )[col_sel]
            for _, nd in branches
        ]
    )

    tparent = {
        nid: float(node_data_lookup[nid].tree_node.tParent or 0.0)
        for nid in node_data_lookup
    }
    parent_ = {
        nid: (
            node_data_lookup[nid].tree_node.parentNode.nodeId
            if node_data_lookup[nid].tree_node.parentNode is not None
            else None
        )
        for nid in node_data_lookup
    }
    children_ = {
        nid: [c.nodeId for c in node_data_lookup[nid].tree_node.childNodes]
        for nid in node_data_lookup
    }
    cutoff = cutoff_multiplier * tau

    def ancestor_chain(nid: str) -> list[tuple[str, float]]:
        cum = 0.5 * tparent[nid]
        chain: list[tuple[str, float]] = []
        cur = parent_[nid]
        while cur is not None and cum < cutoff:
            cum += 0.5 * tparent[cur]
            if cum >= cutoff:
                break
            chain.append((cur, cum))
            cum += 0.5 * tparent[cur]
            cur = parent_[cur]
        return chain

    def descendant_chain(nid: str) -> list[tuple[str, float]]:
        frontier = [(cid, 0.5 * tparent[nid]) for cid in children_[nid]]
        result: list[tuple[str, float]] = []
        while frontier:
            next_frontier: list[tuple[str, float]] = []
            for cid, dist_to_top in frontier:
                dist_to_mid = dist_to_top + 0.5 * tparent[cid]
                if dist_to_mid >= cutoff:
                    continue
                result.append((cid, dist_to_mid))
                dist_to_bottom = dist_to_top + tparent[cid]
                for gc in children_[cid]:
                    next_frontier.append((gc, dist_to_bottom))
            frontier = next_frontier
        return result

    # nan values are skipped
    valid = ~np.isnan(delta_mat)
    np.copyto(delta_mat, 0.0, where=~valid)

    smoothed = delta_mat.copy()
    weight_sum = valid.astype(float)
    for i, nid in enumerate(branch_ids):
        for neighbor_id, d in ancestor_chain(nid) + descendant_chain(nid):
            j = row_of.get(neighbor_id)
            if j is None:
                continue
            w = expweight(d)
            smoothed[i] += w * delta_mat[j]
            weight_sum[i] += w * valid[j]
    # nothing valid anywhere in the neighborhood, including the branch itself
    smoothed[weight_sum == 0.0] = np.nan
    if normalize:
        smoothed = np.divide(smoothed, weight_sum, out=smoothed, where=weight_sum > 0.0)

    for i, nid in enumerate(branch_ids):
        node_data_lookup[nid].delta_deviation_from_parent_smooth = _DeltaDeviationRow(
            smoothed[i], ref_ids, ref_index
        )


def aggregate_delta_deviation_from_parent(
    node_data_lookup: dict[str, TreeNodeExtraData],
    method: Literal["sum", "abs_sum", "mean", "abs_mean"] = "sum",
    axis: Literal["reference", "children", "parent"] = "reference",
    target_node_ids: list[str] | None = None,
    mask_irelevent_reference_nodes: bool = False,
    smoothed: bool = False,
) -> dict[str, float]:
    """
    Aggregate delta deviation scores ΔD for each branch.

    For a parent→child branch y→z, ΔD[y→z, x] (computed by
    compute_delta_deviation_from_parent) gives a score per reference node x.
    With axis="reference", this function collapses that per-reference row to a
    single score per branch, so each branch can be visualized as one color
    (e.g. on the Bonsai tree, by coloring the downstream/child node z).

    Parameters
    ----------
    node_data_lookup : dict[str, TreeNodeExtraData]
        A map from TreeNode.nodeId to TreeNodeExtraData with delta_deviation_from_parent
        populated by compute_delta_deviation_from_parent.
    method : {"sum", "abs_sum", "mean", "abs_mean"}
        Aggregation across reference nodes:
            - "sum": Σ ΔD (signed; captures directional convergence but positive
              and negative values can cancel)
            - "abs_sum": Σ |ΔD| (avoids cancellation)
            - "mean": mean of ΔD over reference nodes
            - "abs_mean": mean of |ΔD| over reference nodes
    axis : {"reference", "children", "parent"}
        Which dimension to aggregate over:
            - "reference": aggregate each branch across reference nodes x
            - "children": aggregate branches by child nodes z; not implemented yet
            - "parent": aggregate branches by parent nodes y; not implemented yet
    target_node_ids : list[str] | None
        Reference node ids to aggregate over when axis="reference". If None,
        aggregate over all reference nodes stored in delta_deviation_from_parent.
    mask_irelevent_reference_nodes : bool
        If True, exclude irrelevant references before aggregation:
            - x = z (the child) → ΔD = 2‖z - y‖^2 (positive artifact)
            - x = y (the parent) → ΔD = 0 (zero artifact)
            - x descendant of z or ancestor of y (potential cycle artifacts)
        Not implemented yet.
    smoothed: bool
        If True, agglomerate using smoothed ΔD scores

    Returns
    -------
    aggregated_scores : dict[str, float]
        A map from the branch's downstream/child node id to the aggregated ΔD
        score. Keys are the same as keys in node_data_lookup, restricted to
        non-root nodes whose delta_deviation_from_parent is set.
    """
    if smoothed and all(
        nd.delta_deviation_from_parent_smooth is None
        for nd in node_data_lookup.values()
    ):
        smoothed = False
        print("No smoothed values found. Using unsmoothed values")

    if mask_irelevent_reference_nodes:
        raise NotImplementedError(
            "no impl for subroutine mask irelevent reference nodes"
        )

    branches: list[tuple[str, TreeNodeExtraData]] = [
        (nid, nd)
        for nid, nd in node_data_lookup.items()
        if nd.delta_deviation_from_parent is not None
    ]
    if not branches:
        return {}

    def _get_delta_deviation_matrix() -> np.ndarray:

        if smoothed:
            return np.stack(
                [
                    cast(
                        _DeltaDeviationRow, nd.delta_deviation_from_parent_smooth
                    ).to_array()
                    for _, nd in branches
                ]
            )
        else:
            return np.stack(
                [
                    cast(_DeltaDeviationRow, nd.delta_deviation_from_parent).to_array()
                    for _, nd in branches
                ]
            )

    if axis in ("children", "parent"):
        _get_delta_deviation_matrix()
        raise NotImplementedError(
            f"no impl for subroutine aggregate delta deviation by {axis}"
        )

    aggregated_scores = {}
    for nid, nd in branches:
        delta_deviation_row = cast(
            _DeltaDeviationRow,
            nd.delta_deviation_from_parent_smooth
            if smoothed
            else nd.delta_deviation_from_parent,
        )
        values = (
            delta_deviation_row.to_array()
            if target_node_ids is None
            else np.array(
                [
                    delta_deviation_row[target_node_id]
                    for target_node_id in target_node_ids
                ],
                dtype=float,
            )
        )
        if method.startswith("abs"):
            values = np.abs(values)
        # nan values are skipped
        values = values[~np.isnan(values)]
        if values.size == 0:
            aggregated_scores[nid] = float("nan")
            continue
        score = values.sum() if method in ("sum", "abs_sum") else values.mean()
        aggregated_scores[nid] = float(score)

    return aggregated_scores


def delta_for_triplet(
    node_x: TreeNodeExtraData,
    node_y: TreeNodeExtraData,
    node_z: TreeNodeExtraData,
) -> float:
    """
    ΔD for a single triplet (reference x, parent y, child z):

    ΔD = 2 (x - y)·(z - y) / p

    y ─── z
     ⋱  ⋰
       x
    """
    x = np.asarray(node_x.tree_node.ltqsAIRoot, dtype=float)
    y = np.asarray(node_y.tree_node.ltqsAIRoot, dtype=float)
    z = np.asarray(node_z.tree_node.ltqsAIRoot, dtype=float)
    return 2.0 * float(np.dot(x - y, z - y)) / len(x)


def accumulate_delta_deviation_scores_along_lineage(
    node_data_lookup: dict[str, TreeNodeExtraData],
    source_node_id: str,
    reference_node_id: str,
    n_steps: int = 0,
    direction: Literal["up", "down"] = "up",
    target_node_id: str | None = None,
    smoothed: bool | None = False,
) -> tuple[float, str]:
    """
    Walk a lineage starting at source_node_id and accumulate signed ΔD scores, all taken
    against a single fixed reference node reference_node_id.

    Every step reads ΔD[branch, reference_node_id] (the same reference column),
    so by telescoping the accumulated value equals
        D(reference, end) - D(reference, source).
    (When reference_node_id == source_node_id this reduces to D(source, end),
    since D(source, source) = 0.)
        - child -> parent (toward root): accumulate -ΔD[branch, reference]
        - parent -> child (toward leaf): accumulate +ΔD[branch, reference]

    Two modes:
        - target_node_id is None: walk n_steps in `direction`.
        - target_node_id given: walk the lineage from source to target;
          n_steps and direction are ignored. (currently no check if lineage exists)

    Requires compute_delta_deviation_from_parent to have been run with
    reference_node_id among the reference nodes (e.g. reference_node_ids=None),
    and normalize_by_branch_length=False (normalized ΔD cannot be accumulated to D).

    Parameters
    ----------
    node_data_lookup : dict[str, TreeNodeExtraData]
        Map from node id to TreeNodeExtraData with delta_deviation_from_parent set.
    source_node_id : str
        Node where the walk starts.
    reference_node_id : str
        Reference node x held fixed for every step; the ΔD column accumulated.
    n_steps : int
        Number of branches to traverse when target_node_id is None.
    direction : {"up", "down"}
        Walk toward the root ("up") or the leaves ("down") when target_node_id
        is None. For "down", branch points follow childNodes[0].
    target_node_id : str | None
        If given, walk the unique lineage from source to this node instead;
        n_steps and direction are ignored.
    smoothed : bool | None
        If given (True), use smoothed delta-deviation values

    Returns
    -------
    accumulated_score : float
        Σ signed ΔD[·, reference] over traversed branches
        == D(reference, end_node_id) - D(reference, source_node_id).
    end_node_id : str
        Node id reached at the end of the walk.
    """
    if smoothed and all(
        nd.delta_deviation_from_parent_smooth is None
        for nd in node_data_lookup.values()
    ):
        smoothed = False
        print("No smoothed values found. Using unsmoothed values")

    if target_node_id is not None:
        # target pins a unique lineage, so n_steps and direction are ignored.
        # TODO: verify source and target lie on the same lineage (one an
        # ancestor of the other); no check for now.

        # Case 1: target is an ancestor of source -> walk up from source.
        node: TreeNode | None = node_data_lookup[source_node_id].tree_node
        upward_lineage_nodes: list[TreeNode] = []
        while node is not None:
            if node.nodeId == target_node_id:
                accumulated_score: float = 0.0
                for c in upward_lineage_nodes:
                    if smoothed:
                        row = node_data_lookup[
                            c.nodeId
                        ].delta_deviation_from_parent_smooth
                    else:
                        row = node_data_lookup[c.nodeId].delta_deviation_from_parent
                    assert row is not None, (
                        f"delta_deviation_from_parent is None for node {c.nodeId}"
                    )
                    accumulated_score -= row[reference_node_id]
                return accumulated_score, target_node_id
            upward_lineage_nodes.append(node)
            node = node.parentNode

        # Case 2: target is a descendant of source -> walk up from target.
        node = node_data_lookup[target_node_id].tree_node
        downward_lineage_nodes: list[TreeNode] = []
        while node is not None and node.nodeId != source_node_id:
            downward_lineage_nodes.append(node)
            node = node.parentNode
        accumulated_score = 0.0
        for c in downward_lineage_nodes:
            if smoothed:
                row = node_data_lookup[c.nodeId].delta_deviation_from_parent_smooth
            else:
                row = node_data_lookup[c.nodeId].delta_deviation_from_parent
            assert row is not None, (
                f"delta_deviation_from_parent is None for node {c.nodeId}"
            )
            accumulated_score += row[reference_node_id]
        return accumulated_score, target_node_id

    current: TreeNode = node_data_lookup[source_node_id].tree_node
    accumulated_score = 0.0
    for _ in range(n_steps):
        if direction == "up":
            parent: TreeNode | None = current.parentNode
            if current.isRoot or parent is None:
                raise ValueError(
                    f"reached root at {current.nodeId}; cannot take {n_steps} 'up' steps"
                )
            row = (
                node_data_lookup[current.nodeId].delta_deviation_from_parent_smooth
                if smoothed
                else node_data_lookup[current.nodeId].delta_deviation_from_parent
            )
            assert row is not None
            accumulated_score += -row[reference_node_id]
            current = parent
        else:  # "down"
            # childNodes[0] for now.
            if current.isLeaf or not current.childNodes:
                raise ValueError(
                    f"reached leaf at {current.nodeId}; cannot take {n_steps} 'down' steps"
                )
            child: TreeNode = current.childNodes[0]
            row = (
                node_data_lookup[child.nodeId].delta_deviation_from_parent_smooth
                if smoothed
                else node_data_lookup[child.nodeId].delta_deviation_from_parent
            )
            assert row is not None
            accumulated_score += row[reference_node_id]
            current = child
    return accumulated_score, current.nodeId
