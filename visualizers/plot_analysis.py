"""Analysis plots: per-node error, histogram, stress curve, noise scatter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from slsn.localization import Localizer
from slsn.network import SensorNetwork


def plot_error_analysis(network: SensorNetwork,
                        localizer: Localizer,
                        save_path: Optional[Path | str] = None):
    """Four-panel analysis: errors, histogram, stress, measurement noise."""
    errors = localizer.per_node_errors()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ----- Per-node error bars -----
    ax = axes[0, 0]
    colors = ["crimson" if localizer.localized[i] else "lightgray"
              for i in range(network.n_nodes)]
    ax.bar(range(network.n_nodes), errors, color=colors, alpha=0.85,
           edgecolor="k", linewidth=0.4)
    ax.axhline(errors.mean(), color="k", ls="--", lw=1.2,
               label=f"Mean = {errors.mean():.2f} m")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Localization error (m)")
    ax.set_title("Per-node localization error\n(red = localized, gray = un-localized)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ----- Error histogram -----
    ax = axes[0, 1]
    ax.hist(errors, bins=20, color="steelblue", edgecolor="k", alpha=0.75)
    ax.axvline(errors.mean(), color="r", ls="--", lw=1.5,
               label=f"Mean = {errors.mean():.2f}")
    ax.axvline(np.median(errors), color="g", ls="--", lw=1.5,
               label=f"Median = {np.median(errors):.2f}")
    ax.set_xlabel("Localization error (m)"); ax.set_ylabel("Count")
    ax.set_title("Error distribution"); ax.legend(); ax.grid(True, alpha=0.3)

    # ----- Stress convergence -----
    ax = axes[1, 0]
    ax.plot(localizer.stress_history, "b.-", lw=1.4, ms=4)
    ax.set_xlabel("Iteration snapshot")
    ax.set_ylabel("Stress  (mean squared edge error, m²)")
    ax.set_title("Convergence of stress (trilateration → gradient descent)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")

    # ----- Measured vs true distance -----
    ax = axes[1, 1]
    true = network.true_positions
    d_true_list, d_meas_list = [], []
    for (i, j), d_meas in network.edges.items():
        d_true = float(np.linalg.norm(true[i] - true[j]))
        d_true_list.append(d_true); d_meas_list.append(d_meas)
    ax.scatter(d_true_list, d_meas_list, c="steelblue", s=18, alpha=0.6,
               edgecolors="none")
    lims = [0, max(max(d_true_list), max(d_meas_list)) * 1.05]
    ax.plot(lims, lims, "r--", lw=1.2, label="Ideal (y = x)")
    ax.set_xlabel("True distance (m)"); ax.set_ylabel("Measured distance (m)")
    ax.set_title(f"Measurement noise  (σ = {network.noise_sigma} m)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
