from __future__ import annotations
import numpy as np
from numba import jit

__all__ = ["sample_farthest_points"]


@jit(nopython=True)
def _sample_impl(data: np.ndarray, seed_indices: np.ndarray, n: int) -> np.ndarray:
    indices = np.zeros(n, dtype=np.int64)
    min_dists = np.full(len(data), np.inf)

    num_seeds = len(seed_indices)
    indices[0] = seed_indices[0]

    for i in range(1, n):
        last_point = data[indices[i - 1]]
        dists = np.sum((data - last_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[indices[:i]] = -1
        if i >= num_seeds:
            next_idx = np.argmax(min_dists)
            indices[i] = next_idx
        else:
            indices[i] = seed_indices[i]

    return indices


def sample_farthest_points(
    embedding: np.ndarray,
    n: int,
    *,
    random_state: int | None = None,
) -> np.ndarray:
    embedding_arr = np.ascontiguousarray(embedding)
    n_points = embedding_arr.shape[0]

    rng = np.random if random_state is None else np.random.RandomState(random_state)
    seed_indices = np.array([rng.randint(n_points)], dtype=np.int64)

    weights = np.ones(n_points)

    return _sample_impl(embedding_arr, seed_indices, n, weights)
