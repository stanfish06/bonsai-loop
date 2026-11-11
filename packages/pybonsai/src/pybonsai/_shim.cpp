#include "_shim.hpp"

#include "bonsai/general_utils/parallel.hpp"
#include "bonsai_output_loader.hpp"
#include "convergence_detection.hpp"
#include "posterior_covariance.hpp"
#include "posterior_distances.hpp"
#include "tree_distances.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>

namespace pybonsai {

struct Output::Impl {
    penjing::LoadedOutput loaded;
    bool topology_only = false;

    // Built on first use: it needs both beliefs, which a topology-only load does not have.
    mutable std::mutex covariance_mutex;
    mutable std::optional<penjing::PosteriorCovariance> covariance_cache;

    const bonsai::TreeNode& node(int index) const {
        const auto& nodes = loaded.dataset.tree.nodes;
        if (index < 0 || static_cast<std::size_t>(index) >= nodes.size() ||
            nodes[static_cast<std::size_t>(index)].node_index < 0) {
            throw std::out_of_range("node index " + std::to_string(index) + " is not a live node");
        }
        return nodes[static_cast<std::size_t>(index)];
    }

    const penjing::PosteriorCovariance& covariance() const {
        if (topology_only) {
            throw std::runtime_error("posterior quantities need coordinates; reload without topology_only");
        }
        const std::lock_guard<std::mutex> lock(covariance_mutex);
        if (!covariance_cache) {
            covariance_cache.emplace(loaded.dataset);
        }
        return *covariance_cache;
    }

    void check_selection(const std::vector<int>& nodes) const {
        if (nodes.size() < 2) {
            throw std::invalid_argument("pairwise quantities need at least two nodes");
        }
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            node(nodes[i]);
            if (i > 0 && nodes[i] <= nodes[i - 1]) {
                throw std::invalid_argument("node indices must be strictly ascending");
            }
        }
    }
};

Output::Output(const std::string& folder, bool topology_only) : impl_(std::make_unique<Impl>()) {
    penjing::LoadOptions options;
    options.topology_only = topology_only;
    impl_->loaded = penjing::load_bonsai_output(folder, options);
    impl_->topology_only = topology_only;
}

Output::~Output() = default;

std::string Output::dialect() const { return penjing::to_string(impl_->loaded.dialect); }

std::size_t Output::n_features() const { return impl_->loaded.dataset.metadata.n_features; }

std::vector<std::string> Output::feature_ids() const { return impl_->loaded.dataset.metadata.feature_ids; }

std::vector<int> Output::node_indices() const { return impl_->loaded.node_indices; }

std::string Output::node_id(int node) const { return impl_->node(node).node_id; }

bool Output::is_sample(int node) const { return impl_->node(node).is_sample; }

int Output::parent_index(int node) const { return impl_->node(node).parent_index; }

double Output::branch_length(int node) const { return impl_->node(node).branch_length; }

void Output::tree_distances(const std::vector<int>& nodes, int threads, double* out) const {
    impl_->check_selection(nodes);
    penjing::TreeDistanceOptions options;
    options.threads = threads;
    const std::vector<double> distances =
        penjing::all_pairs_tree_distances(impl_->loaded.dataset, nodes, options);
    std::copy(distances.begin(), distances.end(), out);
}

void Output::posterior_distances(const std::vector<int>& nodes, int threads, double* out,
                                 double* out_sd) const {
    impl_->check_selection(nodes);
    const penjing::PosteriorCovariance& covariance = impl_->covariance();
    const int configured = bonsai::parallel::configure_threads(threads);
    const int n_threads = configured > 0 ? configured : 1;

    const auto n_selected = static_cast<std::ptrdiff_t>(nodes.size());
    const std::size_t n_features = covariance.n_features();

    // An exception must not leave an OpenMP region: keep the first one and rethrow after.
    std::exception_ptr failure;

#pragma omp parallel num_threads(n_threads)
    {
        std::vector<double> scratch(n_features);
#pragma omp for schedule(dynamic, 16)
        for (std::ptrdiff_t i = 0; i < n_selected - 1; ++i) {
            try {
                // Condensed offset of row i's first pair (i, i+1).
                const auto row = static_cast<std::size_t>(i);
                const auto count = static_cast<std::size_t>(n_selected);
                std::size_t slot = row * (2 * count - row - 1) / 2;
                for (std::ptrdiff_t j = i + 1; j < n_selected; ++j, ++slot) {
                    const penjing::PosteriorDistance estimate = penjing::posterior_squared_distance(
                        covariance, nodes[row], nodes[static_cast<std::size_t>(j)], scratch);
                    out[slot] = estimate.distance;
                    if (out_sd != nullptr) {
                        out_sd[slot] = estimate.standard_deviation;
                    }
                }
            } catch (...) {
#pragma omp critical(pybonsai_posterior_failure)
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        }
    }

    if (failure) {
        std::rethrow_exception(failure);
    }
}

double Output::posterior_distance(int first_node, int second_node, double* out_sd) const {
    impl_->node(first_node);
    impl_->node(second_node);
    const penjing::PosteriorCovariance& covariance = impl_->covariance();
    std::vector<double> scratch(covariance.n_features());
    const penjing::PosteriorDistance estimate =
        penjing::posterior_squared_distance(covariance, first_node, second_node, scratch);
    if (out_sd != nullptr) {
        *out_sd = estimate.standard_deviation;
    }
    return estimate.distance;
}

void Output::posterior_belief(int node, double* out_mean, double* out_variance) const {
    impl_->node(node);
    const penjing::PosteriorCovariance& covariance = impl_->covariance();
    const std::span<const double> mean = covariance.posterior_mean(node);
    const std::span<const double> variance = covariance.posterior_variance(node);
    std::copy(mean.begin(), mean.end(), out_mean);
    std::copy(variance.begin(), variance.end(), out_variance);
}

Candidates Output::detect_convergence(const std::vector<int>& nodes, std::size_t top_k,
                                      std::size_t max_pairs, bool allow_huge, int threads) const {
    impl_->check_selection(nodes);
    if (impl_->topology_only) {
        throw std::runtime_error("convergence detection needs coordinates; reload without topology_only");
    }
    penjing::check_convergence_pair_count(nodes.size(), max_pairs, allow_huge);
    bonsai::parallel::configure_threads(threads);

    const penjing::ConvergenceCandidateResult result =
        penjing::detect_convergence_candidates(impl_->loaded.dataset, nodes, top_k);

    Candidates columns;
    columns.n_pairs_tested = result.n_pairs_tested;
    for (const penjing::ConvergenceCandidate& candidate : result.candidates) {
        columns.first_node.push_back(candidate.first_node);
        columns.second_node.push_back(candidate.second_node);
        columns.first_node_id.push_back(candidate.first_node_id);
        columns.second_node_id.push_back(candidate.second_node_id);
        columns.tree_distance.push_back(candidate.tree_distance);
        columns.posterior_distance.push_back(candidate.posterior_distance);
        columns.residual.push_back(candidate.residual);
        columns.standard_deviation.push_back(candidate.standard_deviation);
        columns.z_score.push_back(candidate.z_score);
        columns.posterior_probability.push_back(candidate.posterior_probability);
        columns.p_value.push_back(candidate.p_value);
        columns.p_value_bonferroni.push_back(candidate.p_value_bonferroni);
    }
    return columns;
}

}  // namespace pybonsai
