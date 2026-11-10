# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython layer over the C++ shim in _shim.hpp."""

import os
from collections import namedtuple

import numpy as np

from libc.stddef cimport size_t
from libcpp cimport bool as cbool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "_shim.hpp" namespace "pybonsai" nogil:
    cdef cppclass Candidates:
        vector[int] first_node
        vector[int] second_node
        vector[string] first_node_id
        vector[string] second_node_id
        vector[double] tree_distance
        vector[double] posterior_distance
        vector[double] residual
        vector[double] standard_deviation
        vector[double] z_score
        vector[double] posterior_probability
        vector[double] p_value
        vector[double] p_value_bonferroni
        size_t n_pairs_tested

    cdef cppclass Output:
        Output(const string& folder, cbool topology_only) except +
        string dialect() except +
        size_t n_features() except +
        vector[string] feature_ids() except +
        vector[int] node_indices() except +
        string node_id(int node) except +
        cbool is_sample(int node) except +
        int parent_index(int node) except +
        double branch_length(int node) except +
        void tree_distances(const vector[int]& nodes, int threads, double* out) except +
        void posterior_distances(const vector[int]& nodes, int threads, double* out, double* out_sd) except +
        void posterior_belief(int node, double* out_mean, double* out_variance) except +
        Candidates detect_convergence(const vector[int]& nodes, size_t top_k, size_t max_pairs,
                                      cbool allow_huge, int threads) except +


# values and uncertainty are condensed arrays in scipy.spatial.distance.squareform order
# over node_ids; uncertainty is None unless it was requested.
PairDistances = namedtuple("PairDistances", ["values", "node_ids", "uncertainty"])


cdef class BonsaiOutput:
    """A Bonsai output folder, written by the Python or the C++ implementation.

    Distances are in diffusion time, so tree and posterior distances compare directly.

    Args:
        folder: Final Bonsai output folder.
        topology_only: Read the tree without coordinates. Tree distances still
            work; posterior quantities raise.
    """

    cdef Output* _out
    cdef readonly str folder
    cdef readonly str dialect
    cdef readonly Py_ssize_t n_features
    cdef readonly list node_ids        # node id per live node, ascending node index
    cdef readonly object node_index    # int32 node index per live node
    cdef readonly object is_sample     # bool per live node; True for measured cells
    cdef readonly object parent_index  # int32 parent node index per live node, -1 at the root
    cdef readonly object branch_length # float64 edge length to the parent, in diffusion time
    cdef dict _row_of_id

    def __cinit__(self, folder, bint topology_only=False):
        self._out = NULL
        cdef string path = os.fsencode(folder)
        self._out = new Output(path, topology_only)

    def __dealloc__(self):
        del self._out

    def __init__(self, folder, bint topology_only=False):
        self.folder = os.fspath(folder)
        self.dialect = self._out.dialect().decode()
        self.n_features = self._out.n_features()

        cdef vector[int] indices = self._out.node_indices()
        cdef Py_ssize_t n = indices.size()
        cdef Py_ssize_t row
        node_index = np.empty(n, dtype=np.int32)
        is_sample = np.empty(n, dtype=np.bool_)
        parent_index = np.empty(n, dtype=np.int32)
        branch_length = np.empty(n, dtype=np.float64)
        ids = []
        for row in range(n):
            node_index[row] = indices[row]
            is_sample[row] = self._out.is_sample(indices[row])
            parent_index[row] = self._out.parent_index(indices[row])
            branch_length[row] = self._out.branch_length(indices[row])
            ids.append(self._out.node_id(indices[row]).decode())
        self.node_index = node_index
        self.is_sample = is_sample
        self.parent_index = parent_index
        self.branch_length = branch_length
        self.node_ids = ids
        self._row_of_id = {node_id: row for row, node_id in enumerate(ids)}
        if len(self._row_of_id) != n:
            raise ValueError(f"{self.folder}: node ids are not unique")

    @property
    def n_nodes(self):
        return len(self.node_ids)

    @property
    def feature_ids(self):
        return [feature.decode() for feature in self._out.feature_ids()]

    def _select(self, nodes, bint leaves_only):
        """Rows of the selected nodes, ascending. `nodes` is an iterable of node ids."""
        if nodes is not None and leaves_only:
            raise ValueError("pass either nodes or leaves_only, not both")
        if nodes is not None:
            try:
                rows = np.array([self._row_of_id[node_id] for node_id in nodes], dtype=np.intp)
            except KeyError as error:
                raise KeyError(f"unknown node id {error.args[0]!r}") from None
            rows = np.unique(rows)
        elif leaves_only:
            rows = np.flatnonzero(self.is_sample)
        else:
            rows = np.arange(len(self.node_ids), dtype=np.intp)
        if rows.size < 2:
            raise ValueError(f"pairwise quantities need at least two nodes, selected {rows.size}")
        return rows

    def tree_distances(self, nodes=None, bint leaves_only=False, int threads=0):
        """Sum of branch lengths between every pair of selected nodes.

        Args:
            nodes: Node ids to report; default all nodes. Reported in ascending node index.
            leaves_only: Report measured cells only.
            threads: OpenMP threads; 0 uses the default.

        Returns:
            PairDistances with float64 `values`.
        """
        rows = self._select(nodes, leaves_only)
        cdef vector[int] selected = self.node_index[rows].tolist()
        cdef Py_ssize_t m = rows.size
        cdef double[::1] out = np.empty(m * (m - 1) // 2, dtype=np.float64)
        with nogil:
            self._out.tree_distances(selected, threads, &out[0])
        return PairDistances(np.asarray(out), [self.node_ids[row] for row in rows], None)

    def posterior_distances(self, nodes=None, bint leaves_only=False, int threads=0,
                            bint with_uncertainty=False):
        """Posterior mean of the true mean squared distance between selected nodes.

        Per pair: (1/G) * sum_g [(x_a - x_b)^2 + var_a + var_b - 2 cov_ab], the squared
        distance between posterior means plus the posterior variance of their difference.

        Args:
            nodes: Node ids to report; default all nodes. Reported in ascending node index.
            leaves_only: Report measured cells only.
            threads: OpenMP threads; 0 uses the default.
            with_uncertainty: Also return each estimate's standard deviation.

        Returns:
            PairDistances with float64 `values` and, if requested, `uncertainty`.
        """
        rows = self._select(nodes, leaves_only)
        cdef vector[int] selected = self.node_index[rows].tolist()
        cdef Py_ssize_t m = rows.size
        cdef double[::1] out = np.empty(m * (m - 1) // 2, dtype=np.float64)
        cdef double[::1] out_sd
        cdef double* sd_ptr = NULL
        uncertainty = None
        if with_uncertainty:
            out_sd = np.empty(m * (m - 1) // 2, dtype=np.float64)
            sd_ptr = &out_sd[0]
            uncertainty = np.asarray(out_sd)
        with nogil:
            self._out.posterior_distances(selected, threads, &out[0], sd_ptr)
        return PairDistances(np.asarray(out), [self.node_ids[row] for row in rows], uncertainty)

    def posterior_belief(self, node_id):
        """Posterior mean and variance per feature of one node, in the diffusion basis.

        Returns:
            (mean, variance), each float64 of length n_features.
        """
        cdef int node = self.node_index[self._row_of_id[node_id]]
        cdef double[::1] mean = np.empty(self.n_features, dtype=np.float64)
        cdef double[::1] variance = np.empty(self.n_features, dtype=np.float64)
        self._out.posterior_belief(node, &mean[0], &variance[0])
        return np.asarray(mean), np.asarray(variance)

    def detect_convergence(self, nodes=None, bint leaves_only=False, Py_ssize_t top_k=50,
                           Py_ssize_t max_pairs=5_000_000, bint allow_huge=False, int threads=0):
        """Pairs whose posterior distance departs most from their tree distance.

        residual = posterior_distance - tree_distance; negative means the pair is closer
        than the tree allows (convergence). z_score = residual / standard_deviation.

        Args:
            nodes: Node ids to score; default all nodes.
            leaves_only: Score measured cells only.
            top_k: Number of pairs to keep, ranked by |z_score|.
            max_pairs: Refuse to score more pairs than this unless allow_huge.
            allow_huge: Score regardless of max_pairs.
            threads: OpenMP threads; 0 uses the default.

        Returns:
            dict of equal-length columns plus "n_pairs_tested" (int).
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_pairs < 0:
            raise ValueError("max_pairs must not be negative")
        rows = self._select(nodes, leaves_only)
        cdef vector[int] selected = self.node_index[rows].tolist()
        cdef size_t keep = top_k
        cdef size_t limit = max_pairs
        cdef Candidates found
        with nogil:
            found = self._out.detect_convergence(selected, keep, limit, allow_huge, threads)
        return {
            "first_node_id": [node_id.decode() for node_id in found.first_node_id],
            "second_node_id": [node_id.decode() for node_id in found.second_node_id],
            "first_node": np.array(found.first_node, dtype=np.int32),
            "second_node": np.array(found.second_node, dtype=np.int32),
            "tree_distance": np.array(found.tree_distance, dtype=np.float64),
            "posterior_distance": np.array(found.posterior_distance, dtype=np.float64),
            "residual": np.array(found.residual, dtype=np.float64),
            "standard_deviation": np.array(found.standard_deviation, dtype=np.float64),
            "z_score": np.array(found.z_score, dtype=np.float64),
            "posterior_probability": np.array(found.posterior_probability, dtype=np.float64),
            "p_value": np.array(found.p_value, dtype=np.float64),
            "p_value_bonferroni": np.array(found.p_value_bonferroni, dtype=np.float64),
            "n_pairs_tested": int(found.n_pairs_tested),
        }
