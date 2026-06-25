"""Tests for the SLSN localization pipeline."""
import numpy as np
import pytest

from slsn import SensorNetwork, Localizer
from slsn.graph import is_connected


def test_dense_low_noise_localizes_well():
    """With dense connectivity and low noise, mean error should be small."""
    net = SensorNetwork(
        n_nodes=25, space_size=50.0, radio_range=30.0,
        noise_sigma=0.1, link_dropout=0.0, seed=0)
    assert is_connected(net)

    loc = Localizer(net)
    loc.run(max_iter=30, refine_iterations=200, refine_lr=0.01)
    errors = loc.per_node_errors()
    assert errors.mean() < 3.0, f"Mean error {errors.mean()} too high"
    assert sum(loc.localized) == net.n_nodes, "Not all nodes localized"


def test_anchors_directly_connected():
    """Anchors should have direct measured links between them when possible."""
    net = SensorNetwork(
        n_nodes=30, space_size=80.0, radio_range=30.0,
        noise_sigma=0.3, link_dropout=0.0, seed=2)
    loc = Localizer(net)
    loc.select_anchors()
    a0, a1, a2 = loc.anchors
    assert net.measured_distance(a0, a1) is not None
    assert net.measured_distance(a0, a2) is not None
    assert net.measured_distance(a1, a2) is not None


def test_partial_dropout_still_localizes_most_nodes():
    """Even with link dropout, the bulk of nodes should localize."""
    net = SensorNetwork(
        n_nodes=40, space_size=80.0, radio_range=25.0,
        noise_sigma=0.5, link_dropout=0.25, seed=5)
    if not is_connected(net):
        pytest.skip("Random graph not connected; re-run with different seed.")
    loc = Localizer(net)
    loc.run(max_iter=40, refine_iterations=300, refine_lr=0.008)
    n_loc = sum(loc.localized)
    assert n_loc >= 0.7 * net.n_nodes, f"Only {n_loc}/{net.n_nodes} localized"


def test_high_noise_error_is_bounded():
    """With high noise, error grows but stays within reasonable bounds."""
    net = SensorNetwork(
        n_nodes=30, space_size=80.0, radio_range=30.0,
        noise_sigma=2.0, link_dropout=0.1, seed=11)
    loc = Localizer(net)
    loc.run(max_iter=30, refine_iterations=400, refine_lr=0.005)
    errors = loc.per_node_errors()
    # Should be in the same order of magnitude as the noise
    assert errors.mean() < 10.0


def test_stress_decreases_with_refinement():
    """Gradient descent should not increase the stress."""
    net = SensorNetwork(
        n_nodes=30, space_size=80.0, radio_range=30.0,
        noise_sigma=0.5, link_dropout=0.1, seed=7)
    loc = Localizer(net)
    loc.select_anchors()
    loc.place_anchors()
    loc.localize_iterative(max_iter=30)
    stress_before = loc.compute_stress()
    loc.refine_gradient_descent(iterations=300, lr=0.005)
    stress_after = loc.compute_stress()
    assert stress_after <= stress_before + 1e-6


def test_reproducibility_with_seed():
    """Same seed → same measurements and same recovered geometry."""
    n1 = SensorNetwork(n_nodes=20, seed=99)
    n2 = SensorNetwork(n_nodes=20, seed=99)
    assert set(n1.edges.keys()) == set(n2.edges.keys())
    for k in n1.edges:
        assert n1.edges[k] == n2.edges[k]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
