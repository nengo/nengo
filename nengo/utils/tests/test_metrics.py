import numpy as np
import pytest

import nengo
from nengo.utils.metrics import victor_purpura


def test_victor_purpura_basic():
    # Test addition/removal
    assert np.allclose(victor_purpura([0, 0, 0], [0, 0, 0]), 0)
    assert np.allclose(victor_purpura([0, 0, 1], [0, 0, 0]), 1)
    assert np.allclose(victor_purpura([0, 0, 0], [0, 0, 1]), 1)
    assert np.allclose(victor_purpura([0, 1, 0], [0, 0, 0]), 1)
    assert np.allclose(victor_purpura([0, 0, 0], [0, 1, 0]), 1)
    assert np.allclose(victor_purpura([1, 0, 1], [0, 0, 0]), 2)
    assert np.allclose(victor_purpura([0, 0, 0], [1, 0, 1]), 2)

    # Test multiple dimensions
    assert np.allclose(
        victor_purpura([[1, 0], [0, 1], [1, 0]], [1, 1, 1]),
        [[1], [2]])

    # Test shifting and matrix output
    assert np.allclose(
        victor_purpura([[1, 0],
                        [0, 1],
                        [1, 0]],
                       [[1, 1, 1],
                        [1, 1, 0],
                        [0, 1, 1]]),
        [[0.001, 1, 0], [1, 2, 1.001]])

    # Test increasing q/dt values, and see that they hit an upper-bound
    assert np.allclose(
        victor_purpura([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], 3, 0.005), 0.06)
    assert np.allclose(
        victor_purpura([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], 6, 0.01), 0.24)
    assert np.allclose(
        victor_purpura([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], 60, 0.005), 1.2)
    assert np.allclose(
        victor_purpura([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], 60, 0.01), 2)
    assert np.allclose(
        victor_purpura([1, 0, 0, 0, 0], [0, 0, 0, 0, 1], 1000), 2)


def test_victor_purpura_invalid():
    with pytest.raises(ValueError):  # length mismatch
        assert np.allclose(victor_purpura([0, 0], [0, 0, 0]), 0)

    with pytest.raises(ValueError):  # length mismatch
        assert np.allclose(victor_purpura([0, 0, 0], [0, 0]), 0)


def test_victor_purpura_large(seed):
    """Check that Victor-Purpura of constant spiking population is small."""
    n = 100
    T = 0.1

    m = nengo.Network(seed=seed)
    with m:
        u = nengo.Node(output=[0])
        x = nengo.Ensemble(n, 1)
        nengo.Connection(u, x, synapse=None)
        p_x = nengo.Probe(x.neurons, synapse=None)

    sim = nengo.Simulator(m)
    sim.run(T)

    k = len(sim.trange()) // 2

    dists = [victor_purpura(sim.data[p_x][:k, i], sim.data[p_x][k:2*k, i])
             for i in range(n)]
    dists = np.asarray(dists)

    assert np.any(dists > 0)
    assert np.all(dists < 3)
