import sys
import numpy as np
from pathlib import Path

_BONSAI = Path(__file__).resolve().parents[2] / "Bonsai-data-representation"
if str(_BONSAI) not in sys.path:
    sys.path.insert(0, str(_BONSAI))

import bonsai
import bonsai_scout
import paper_figure_scripts_and_notebooks as bonsai_paper
import paper_figure_scripts_and_notebooks.simulating_datasets as _bonsai_simulating_datasets
import paper_figure_scripts_and_notebooks.simulating_datasets.analyzing_simulated_datasets as _bonsai_simulated_datasets
import paper_figure_scripts_and_notebooks.simulating_datasets.analyzing_simulated_datasets.knn_recall_helpers as _bonsai_knn_recall_helpers

sys.modules[f"{__name__}.bonsai"] = bonsai
sys.modules[f"{__name__}.bonsai_scout"] = bonsai_scout
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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform

# import relevant functions for distance computation
# pretend to be submodules of bonsai_lib
from .bonsai_lib.bonsai.bonsai_dataprocessing import get_bonsai_euclidean_distances
from .bonsai_lib.bonsai_scout.my_tree_layout import Layout_Tree
from .bonsai_lib.bonsai_paper.simulating_datasets.analyzing_simulated_datasets.knn_recall_helpers import (
    get_pdists_on_tree,
)


# euclidean distances based on tree path
def get_pdists_on_tree_euclidean(nwk_file, bonsai_output_folder, cell_ids):
    # Read in newick file
    tree = Layout_Tree()

    with open(nwk_file, "r") as f:
        nwk_str = f.readline()

    tree.from_newick(nwk_str=nwk_str)

    # Renumber vert_inds on tree such that they are in line with a depth-first search
    vertIndToNode, tree.nNodes = tree.root.renumber_verts(
        vertIndToNode={}, vert_count=0
    )
    tree.vert_ind_to_node = vertIndToNode
    tree.root.storeParent()
    # replace tree distance with euclidean distance
    # same as get_pairwise_dist_on_tree with only weights replaced by squared euclidean distances
    node_id_to_vert_ind = {
        node.nodeId: vert_ind for vert_ind, node in tree.vert_ind_to_node.items()
    }
    indices = [node_id_to_vert_ind[node_id] for node_id in cell_ids]
    edge_dict = tree.get_edge_dict(nodeIdToVertInd=node_id_to_vert_ind)
    cols = edge_dict["source"]
    rows = edge_dict["target"]
    # pairwise euclidean distances based on Bonsai posteriors
    # formula: \frac{1}{G} \sum_g (y_{gc} - y_{gc'})^2
    dist_mat = squareform(
        get_bonsai_euclidean_distances(
            bonsai_output_folder, list(node_id_to_vert_ind.keys())
        )
    )
    idx_lookup = {k: i for i, k in enumerate(node_id_to_vert_ind.keys())}
    weights = [dist_mat[idx_lookup[s], idx_lookup[t]] for s, t in zip(cols, rows)]
    cols = edge_dict["source_ind"]
    rows = edge_dict["target_ind"]

    colsComplete = np.concatenate((cols, rows))
    rowsComplete = np.concatenate((rows, cols))
    weightsComplete = np.concatenate((weights, weights))
    nVerts = np.max(colsComplete) + 1
    distance_csr = csr_matrix(
        (weightsComplete, (rowsComplete, colsComplete)), shape=(nVerts, nVerts)
    )

    print("Done preparing inputs. Starting shortest-path algorithm from Scipy.")
    distances = squareform(
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
    return distances


__all__ = [
    "get_pdists_on_tree_euclidean",
    "get_bonsai_euclidean_distances",
    "get_pdists_on_tree",
    "Layout_Tree",
]
