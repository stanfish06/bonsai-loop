from __future__ import annotations
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass
from collections import Counter
from .bonsai_lib.bonsai.bonsai_treeHelpers import Tree, TreeNode


@dataclass
class TreeNodeExtraData:
    """A container to store additional properties for each node (root, internal, or
    leaf)

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
    ordering_value : float | None
        The 1D global ordering value of the node in the Bonsai phylogenetic tree, with these potential versions:
            - bonsai tree distance to a specific node (e.g. root)
            - vertical distance in the dendrogram (e.g. nodes with fewer branches (more advanved) are placed higher)
            - computed from its descendents
    """

    tree_node: TreeNode
    topological_level: int | None = None
    geometric_level: float | None = None
    identity: dict | None = None
    n_leaves: int | None = None
    ordering_value: float | None = None

    def compute_topological_level(
        self, node_data_children: list[TreeNodeExtraData]
    ) -> None:
        """Helper function to compute topological node level from the leaves (level =
        0). Level increases toward the tree root. When child sub-trees have different
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
        """Helper function to compute node identity from its descendents (leaves).

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
            if child_node_data is None:
                continue
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


def compute_node_ordering_value(
    tree: Tree,
    node_data_lookup: dict[str, TreeNodeExtraData],
    metric: Literal["bonsai_t_to_root", "dendrogram"],
    aggregate_metric_from_leaves: bool = False,
) -> None:
    pass


def compute_node_ordering(
    node_data_lookup: dict[str, TreeNodeExtraData],
) -> list[str]:
    return []


def compute_tree_node_level_and_label(
    tree: Tree,
    node_level_type: Literal["topological", "geometric"],
    label_lookup_leaves: dict | None = None,
) -> dict[str, TreeNodeExtraData]:
    """
    Compute the tree topology level and label of each node.
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
        raise NotImplementedError("not ready")
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
                geometric_level=0.0,
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


def get_pdists_on_tree_by_depth(type: Literal["bonsai_t", "euclidean"], depth: int = 0):
    pass


def get_pdists_embedding_by_depth(depth: int = 0):
    pass
