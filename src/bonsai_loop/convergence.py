from __future__ import annotations
from tqdm import tqdm
from typing import Literal, cast
from dataclasses import dataclass
from collections import Counter
from collections.abc import Mapping
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
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
    other_props: dict | None
        Other non-essential properties
    """

    tree_node: TreeNode
    topological_level: int | None = None
    geometric_level: float | None = None
    identity: dict | None = None
    n_leaves: int | None = None
    ordering_value: float | None = None
    dendrogram_coords: tuple[float, float] | None = None
    delta_deviation_from_parent: Mapping[str, float] | None = None
    other_props: dict | None = None

    def __repr__(self) -> str:
        def _print_identity(identity: dict | None, top_n: int = 3) -> str:
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
        node_data.dendrogram_coords = (float(x), y_coords[node_id])


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
            node_data.ordering_value = (
                tree_dists_to_root[node_id] if node_id in tree_dists_to_root else None
            )
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
        identity_ordering_value_sum = Counter()
        identity_weight_sum = Counter()
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
    label_lookup_leaves: dict | None = None,
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
    node_data_lookup = {}

    print("compute depth-first ordering of nodes")
    root_node: TreeNode = tree.root
    print(f"root node {root_node.nodeId}")
    stack = [root_node]
    compute_order = []
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

        for child_node in node.childNodes:
            stack.append(child_node)

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

    __slots__ = ("_row", "_ref_ids", "_ref_index")

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

    def __iter__(self):
        return iter(self._ref_ids)

    def __len__(self) -> int:
        return len(self._ref_ids)

    def to_array(self) -> np.ndarray:
        """Return the underlying row as a numpy view into the shared ΔD matrix."""
        return self._row


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
        - D_{xz} - D_{xy} = ‖z - y‖^2 (the difference between tree paths x⇔z and x⇔y) - ‖x - z‖^2 + ‖x - y‖^2
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
        t_parents = np.asarray(
            [float(c.tParent) for _, c, _ in branch_nodes], dtype=float
        )
        delta_d /= t_parents[:, None]

    for i, (nid, _, _) in enumerate(branch_nodes):
        node_data_lookup[nid].delta_deviation_from_parent = _DeltaDeviationRow(
            delta_d[i], ref_ids, ref_index
        )


def aggregate_delta_deviation_from_parent(
    node_data_lookup: dict[str, TreeNodeExtraData],
    method: Literal["sum", "abs_sum", "mean", "abs_mean"] = "sum",
    axis: Literal["reference", "children", "parent"] = "reference",
    mask_irelevent_reference_nodes: bool = False,
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
    mask_irelevent_reference_nodes : bool
        If True, exclude irrelevant references before aggregation:
            - x = z (the child) → ΔD = 2‖z - y‖^2 (positive artifact)
            - x = y (the parent) → ΔD = 0 (zero artifact)
            - x descendant of z or ancestor of y (potential cycle artifacts)
        Not implemented yet.

    Returns
    -------
    aggregated_scores : dict[str, float]
        A map from the branch's downstream/child node id to the aggregated ΔD
        score. Keys are the same as keys in node_data_lookup, restricted to
        non-root nodes whose delta_deviation_from_parent is set.
    """
    if mask_irelevent_reference_nodes:
        raise NotImplementedError(
            "no impl for subroutine mask irelevent reference nodes"
        )

    branches = [
        (nid, nd)
        for nid, nd in node_data_lookup.items()
        if nd.delta_deviation_from_parent is not None
    ]
    if not branches:
        return {}

    def _get_delta_deviation_matrix() -> np.ndarray:
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
        values = cast(_DeltaDeviationRow, nd.delta_deviation_from_parent).to_array()
        if method.startswith("abs"):
            values = np.abs(values)
        if method in ("sum", "abs_sum"):
            score = values.sum()
        else:
            score = values.mean()
        aggregated_scores[nid] = float(score)

    return aggregated_scores
