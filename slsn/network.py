"""Sensor network simulation: deployment + noisy distance measurements."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class Node:
    """A single sensor node."""
    id: int
    true_pos: np.ndarray
    est_pos: Optional[np.ndarray] = None
    localized: bool = False


class SensorNetwork:
    """A deployment of N sensor nodes in a 2-D space.

    Nodes are dropped uniformly at random. Pairs within ``radio_range``
    measure their distance via signal Time-of-Flight, corrupted by
    Gaussian noise. A fraction ``link_dropout`` of viable links are
    missing (obstructions / packet loss).

    Parameters
    ----------
    n_nodes      : number of sensors to drop
    space_size   : side length (m) of the square deployment region
    radio_range  : max distance (m) at which two nodes can measure
    noise_sigma  : std dev (m) of Gaussian measurement noise
    link_dropout : probability that a viable link is missing
    seed         : RNG seed
    """

    def __init__(self,
                 n_nodes: int = 60,
                 space_size: float = 100.0,
                 radio_range: float = 30.0,
                 noise_sigma: float = 0.5,
                 link_dropout: float = 0.15,
                 seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        self.space_size = space_size
        self.radio_range = radio_range
        self.noise_sigma = noise_sigma
        self.link_dropout = link_dropout

        # Uniform random deployment
        self.true_positions = self.rng.uniform(
            0.0, space_size, size=(n_nodes, 2))
        self.nodes = [Node(i, self.true_positions[i]) for i in range(n_nodes)]

        # Measure inter-node distances
        self.edges: Dict[Tuple[int, int], float] = {}
        self._measure_distances()
        self._build_adjacency()

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------
    def _measure_distances(self) -> None:
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                d_true = float(np.linalg.norm(
                    self.true_positions[i] - self.true_positions[j]))
                if d_true > self.radio_range:
                    continue
                if self.rng.random() < self.link_dropout:
                    continue
                # Time-of-flight measurement with Gaussian noise
                d_meas = d_true + self.rng.normal(0.0, self.noise_sigma)
                d_meas = max(d_meas, 0.0)
                self.edges[(i, j)] = d_meas

    def _build_adjacency(self) -> None:
        self.adjacency: Dict[int, List[Tuple[int, float]]] = {
            i: [] for i in range(self.n_nodes)}
        for (i, j), d in self.edges.items():
            self.adjacency[i].append((j, d))
            self.adjacency[j].append((i, d))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def neighbors(self, i: int) -> List[int]:
        """IDs of nodes with a measured link to node ``i``."""
        return [j for j, _ in self.adjacency[i]]

    def measured_distance(self, i: int, j: int) -> Optional[float]:
        """Measured distance between i and j, or None if no direct link."""
        key = (i, j) if i < j else (j, i)
        return self.edges.get(key)

    def degree(self, i: int) -> int:
        return len(self.adjacency[i])

    def __repr__(self) -> str:
        return (f"SensorNetwork(n_nodes={self.n_nodes}, "
                f"edges={len(self.edges)}, "
                f"radio_range={self.radio_range}, "
                f"noise_sigma={self.noise_sigma})")
