from typing import Literal
from dataclasses import dataclass
from .bonsai_lib.bonsai.bonsai_treeHelpers import Tree


@dataclass
class TreeNodeExtraData:
    topological_level: int
    geometric_level: int
    identity: dict
    n_leaves: int


def compute_tree_node_level_and_label(
    tree: Tree,
    node_level_type: Literal["topological", "geometric"],
    label_lookup_leaves: dict,
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
    - for label, assuming only leaves have labels, compute average identity for internal nodes
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
    node_level_lookup : dict
    a map: TreeNode.nodeId → TreeNodeExtraData
    """
    if node_level_type == "geometric":
        raise NotImplementedError("not ready")
    node_level_lookup = {}
    return node_level_lookup


def get_pdists_on_tree_by_depth(type: Literal["bonsai_t", "euclidean"], depth: int = 0):
    pass


def get_pdists_embedding_by_depth(depth: int = 0):
    pass
