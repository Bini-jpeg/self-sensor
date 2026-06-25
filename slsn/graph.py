"""Graph utilities: connectivity, shortest-path distance estimation."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

from .network import SensorNetwork


def connectivity_matrix(network: SensorNetwork) -> np.ndarray:
    """Dense adjacency matrix with measured distances; inf where no edge."""
    n = network.n_nodes
    M = np.full((n, n), np.inf)
    np.fill_diagonal(M, 0.0)
    for (i, j), d in network.edges.items():
        M[i, j] = d
        M[j, i] = d
    return M


def shortest_path_distances(network: SensorNetwork) -> np.ndarray:
    """All-pairs shortest path using measured edge weights.

    Used to estimate distances between non-adjacent nodes
    (multi-hop distance, which overestimates true Euclidean distance
    but is useful for anchor selection / initialization).
    """
    M = connectivity_matrix(network)
    # Replace inf off-diagonal with 0 for csgraph input → unreachable
    # csr_matrix with explicit zero weights would mislead Dijkstra,
    # so use the dense version directly.
    return shortest_path(M, method="D", directed=False)


def is_connected(network: SensorNetwork) -> bool:
    """True if the measured-link graph is a single connected component."""
    n = network.n_nodes
    rows, cols = [], []
    for (i, j) in network.edges.keys():
        rows.append(i); cols.append(j)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T
    n_components, _ = connected_components(A, directed=False)
    return n_components == 1
