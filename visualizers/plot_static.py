"""Static plots: ground-truth vs estimate, connectivity, distance matrix."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from slsn.localization import Localizer
from slsn.network import SensorNetwork


def plot_true_vs_estimated(network: SensorNetwork,
                           localizer: Localizer,
                           save_path: Optional[Path | str] = None):
    """Three-panel comparison: truth, estimate, and overlay with error lines."""
    est_aligned, true_c = localizer.align_to_ground_truth()
    errors = np.linalg.norm(est_aligned - true_c, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ----- Panel 1: Ground truth -----
    ax = axes[0]
    true = network.true_positions
    for (i, j), _ in network.edges.items():
        ax.plot([true[i, 0], true[j, 0]],
                [true[i, 1], true[j, 1]],
                color="steelblue", alpha=0.15, lw=0.6)
    ax.scatter(true[:, 0], true[:, 1], c="steelblue", s=45, zorder=3,
               edgecolors="k", linewidths=0.5)
    for k, a in enumerate(localizer.anchors):
        ax.scatter(true[a, 0], true[a, 1], c="red", s=260, marker="*",
                   zorder=5, edgecolors="k", linewidths=0.6,
                   label="Anchors" if k == 0 else None)
    ax.set_title("Ground Truth Deployment")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    # ----- Panel 2: Estimated -----
    ax = axes[1]
    for (i, j), _ in network.edges.items():
        ax.plot([est_aligned[i, 0], est_aligned[j, 0]],
                [est_aligned[i, 1], est_aligned[j, 1]],
                color="mediumseagreen", alpha=0.15, lw=0.6)
    ax.scatter(est_aligned[:, 0], est_aligned[:, 1], c="mediumseagreen",
               s=45, zorder=3, edgecolors="k", linewidths=0.5)
    for k, a in enumerate(localizer.anchors):
        ax.scatter(est_aligned[a, 0], est_aligned[a, 1], c="red", s=260,
                   marker="*", zorder=5, edgecolors="k", linewidths=0.6)
    ax.set_title("Estimated Positions (Procrustes-aligned)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # ----- Panel 3: Overlay -----
    ax = axes[2]
    for i in range(network.n_nodes):
        ax.plot([true_c[i, 0], est_aligned[i, 0]],
                [true_c[i, 1], est_aligned[i, 1]],
                color="k", alpha=0.35, lw=0.7)
    ax.scatter(true_c[:, 0], true_c[:, 1], c="steelblue", s=55, marker="o",
               edgecolors="k", linewidths=0.5, label="True", zorder=4)
    ax.scatter(est_aligned[:, 0], est_aligned[:, 1], c="mediumseagreen",
               s=55, marker="x", label="Estimated", zorder=5, linewidths=1.5)
    for k, a in enumerate(localizer.anchors):
        ax.scatter(true_c[a, 0], true_c[a, 1], c="red", s=260, marker="*",
                   zorder=6, edgecolors="k", linewidths=0.6,
                   label="Anchors" if k == 0 else None)
    ax.set_title(f"Overlay  (mean err = {errors.mean():.2f} m, "
                 f"max = {errors.max():.2f} m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_connectivity(network: SensorNetwork,
                      localizer: Optional[Localizer] = None,
                      save_path: Optional[Path | str] = None):
    """Network graph with edges colored by measured distance + adjacency matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Network graph -----
    ax = axes[0]
    true = network.true_positions
    edges = list(network.edges.items())
    dists = [d for _, d in edges]
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(dists), vmax=max(dists))
    for (i, j), d in edges:
        ax.plot([true[i, 0], true[j, 0]],
                [true[i, 1], true[j, 1]],
                color=cmap(norm(d)), alpha=0.55, lw=1.0)
    ax.scatter(true[:, 0], true[:, 1], c="steelblue", s=45, zorder=3,
               edgecolors="k", linewidths=0.5)
    if localizer is not None:
        for k, a in enumerate(localizer.anchors):
            ax.scatter(true[a, 0], true[a, 1], c="red", s=260, marker="*",
                       zorder=5, edgecolors="k", linewidths=0.6,
                       label="Anchors" if k == 0 else None)
        ax.legend(loc="upper right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Measured distance (m)")
    ax.set_title(f"Network Graph  ({len(edges)} links, "
                 f"radio range = {network.radio_range} m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # ----- Adjacency / distance matrix -----
    ax = axes[1]
    M = np.zeros((network.n_nodes, network.n_nodes))
    for (i, j), d in network.edges.items():
        M[i, j] = d
        M[j, i] = d
    im = ax.imshow(M, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="Measured distance (m)")
    ax.set_title("Pairwise Distance Matrix")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
