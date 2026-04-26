from __future__ import annotations
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass
from .bonsai_lib.bonsai.bonsai_treeHelpers import Tree, TreeNode


@dataclass
class TreeNodeExtraData:
    topological_level: int | None = None
    geometric_level: float | None = None
    identity: dict | None = None
    n_leaves: int | None = None

    def compute_topological_level(
        self, node_data_children: list[TreeNodeExtraData]
    ) -> None:
        pass

    def compute_identity(self, node_data_children: list[TreeNodeExtraData]) -> None:
        pass


def compute_tree_node_level_and_label(
    tree: Tree,
    node_level_type: Literal["topological", "geometric"],
    label_lookup_leaves: dict | None = None,
) -> dict:
    """
    Compute the tree topology level and label of each node.
    - tree is likely imbalanced, so resolve level with the deepest substree.
        ┌── C
        A   ┌── E
        └── B
            └── D
        - depths:
            - C = E = D = 0
            - B = 1
            - A = 2
    - for label, assuming only leaves have labels, compute descendent identity composition for internal nodes, such as:
        A (75% dog, 25% cat, n = 4)
        ├── B (100% dog, n = 1) ── C (dog, n = 1)
        ├── D (dog, n = 1)
        │   ┌── F (dog, n = 1)
        └── E (50% dog, 50% cat, n = 2)
            └── G (cat, n = 1)
    - to be implemented: a better version might also consider edge length, make node level geometric
    Steps:
    1. do DFS to order the computation
        - for instance, for the example above, create a stack [D, E, B ,C, A]
    2. resolve the label and level of each node

    Parameters
    ----------
    tree : Tree
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
                topological_level=0,
                geometric_level=0.0,
                identity={node_label: 1.0} if node_label is not None else None,
                n_leaves=1,
            )
        else:
            node_data = TreeNodeExtraData()
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
