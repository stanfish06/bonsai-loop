from typing import Literal
from .bonsai_lib.bonsai.bonsai_treeHelpers import Tree

"""
compute the tree level of each internal node
- in case of imbalanced subtree, consider the deepest substree:
    ┌── C
    A   ┌── E
    └── B
        └── D
    - depths:
        - C = E = D = 0
        - B = 1
        - A = 2
steps:
1. do DFS to order the computation
2. then take max child depth
"""


def compute_tree_node_level(tree: Tree) -> dict:
    return {}


def get_pdists_on_tree_by_depth(type: Literal["bonsai_t", "euclidean"], depth: int = 0):
    pass


def get_pdists_embedding_by_depth(depth: int = 0):
    pass
