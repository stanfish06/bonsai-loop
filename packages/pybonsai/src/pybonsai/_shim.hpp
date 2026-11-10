#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/// Plain-type facade over penjing and bonsai_port for the Cython module.
/// No upstream header is included here, so Cython only ever sees these declarations.
namespace pybonsai {

/// Convergence candidates as columns, sorted by descending |z_score|.
struct Candidates {
    std::vector<int> first_node;
    std::vector<int> second_node;
    std::vector<std::string> first_node_id;
    std::vector<std::string> second_node_id;
    std::vector<double> tree_distance;
    std::vector<double> posterior_distance;
    std::vector<double> residual;  ///< posterior_distance - tree_distance
    std::vector<double> standard_deviation;
    std::vector<double> z_score;
    std::vector<double> posterior_probability;
    std::vector<double> p_value;
    std::vector<double> p_value_bonferroni;
    std::size_t n_pairs_tested = 0;
};

/// One loaded Bonsai output folder, from either the Python or the C++ implementation.
///
/// Pair results use scipy's condensed order: for selected nodes 0..m-1, pair (i, j) with
/// i < j sits at i*(2m-i-1)/2 + (j-i-1). Every `nodes` argument must be strictly
/// ascending node indices with at least two entries. `threads <= 0` uses the OpenMP default.
class Output {
public:
    /// `topology_only` skips the coordinate arrays: tree distances still work, posterior ones throw.
    Output(const std::string& folder, bool topology_only);
    ~Output();
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;

    std::string dialect() const;      ///< "python" or "cpp"
    std::size_t n_features() const;
    std::vector<std::string> feature_ids() const;

    std::vector<int> node_indices() const;  ///< live nodes, ascending
    std::string node_id(int node) const;
    bool is_sample(int node) const;
    int parent_index(int node) const;       ///< -1 for the root
    double branch_length(int node) const;   ///< edge length to the parent, in diffusion time

    /// Sum of branch lengths between every selected pair. `out` holds m*(m-1)/2 doubles.
    void tree_distances(const std::vector<int>& nodes, int threads, double* out) const;

    /// Posterior mean of the true mean squared distance for every selected pair, in diffusion time.
    /// `out_sd` may be null; otherwise it receives each estimate's standard deviation.
    void posterior_distances(const std::vector<int>& nodes, int threads, double* out, double* out_sd) const;

    /// Posterior mean and variance per feature for one node, in the diffusion basis.
    /// Each buffer holds n_features doubles.
    void posterior_belief(int node, double* out_mean, double* out_variance) const;

    /// Score every selected pair and keep the `top_k` with the largest |z_score|.
    /// Throws if the pair count exceeds `max_pairs` and `allow_huge` is false.
    Candidates detect_convergence(const std::vector<int>& nodes, std::size_t top_k, std::size_t max_pairs,
                                  bool allow_huge, int threads) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pybonsai
