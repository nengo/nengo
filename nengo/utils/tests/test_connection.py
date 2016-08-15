import numpy as np
import pytest
from matplotlib.mlab import griddata

import nengo
from nengo.builder.ensemble import get_activities
from nengo.dists import UniformHypersphere
from nengo.utils.connection import (
    target_function, eval_point_decoding, pes_learning_rate)
from nengo.utils.numpy import rms


@pytest.mark.parametrize("dimensions", [1, 4])
@pytest.mark.parametrize("radius", [1, 2.0])
def test_target_function(Simulator, nl_nodirect, plt, dimensions, radius,
                         seed, rng):
    eval_points = UniformHypersphere().sample(1000, dimensions, rng=rng)
    eval_points *= radius
    f = lambda x: x ** 2
    targets = f(eval_points)

    model = nengo.Network(seed=seed)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl_nodirect()
        inp = nengo.Node(lambda t: np.sin(t * 2 * np.pi) * radius)
        ens1 = nengo.Ensemble(40 * dimensions, dimensions, radius=radius)
        n1 = nengo.Node(size_in=dimensions)
        n2 = nengo.Node(size_in=dimensions)
        transform = np.linspace(1, -1, num=dimensions).reshape(-1, 1)
        nengo.Connection(inp, ens1, transform=transform)
        # pass in eval_points and targets
        nengo.Connection(ens1, n1, **target_function(eval_points, targets))
        # same, but let the builder apply f
        nengo.Connection(ens1, n2, function=f)
        probe1 = nengo.Probe(n1, synapse=0.03)
        probe2 = nengo.Probe(n2, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[probe1])
    plt.title('Square manually with target_function')
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(), sim.data[probe2])
    plt.title('Square by passing in function to connection')

    assert np.allclose(sim.data[probe1], sim.data[probe2], atol=0.2 * radius)


def test_eval_point_decoding(Simulator, nl_nodirect, plt, seed):
    with nengo.Network(seed=seed) as model:
        model.config[nengo.Ensemble].neuron_type = nl_nodirect()
        a = nengo.Ensemble(200, 2)
        b = nengo.Ensemble(100, 1)
        c = nengo.Connection(a, b, function=lambda x: x[0] * x[1])

    with Simulator(model) as sim:
        eval_points, targets, decoded = eval_point_decoding(c, sim)

    def contour(xy, z):
        xi = np.linspace(-1, 1, 101)
        yi = np.linspace(-1, 1, 101)
        zi = griddata(xy[:, 0], xy[:, 1], z.ravel(), xi, yi, interp='linear')
        plt.contourf(xi, yi, zi, cmap=plt.cm.seismic)
        plt.colorbar()

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    contour(eval_points, targets)
    plt.title("Target (desired decoding)")
    plt.subplot(132)
    plt.title("Actual decoding")
    contour(eval_points, decoded)
    plt.subplot(133)
    plt.title("Difference between actual and desired")
    contour(eval_points, decoded - targets)

    # Generous error check, just to make sure it's in the right ballpark.
    # Also make sure error is above zero, i.e. y != z
    error = rms(decoded - targets, axis=1).mean()
    assert error < 0.1 and error > 1e-8


def test_pes_learning_rate(Simulator, plt, seed):
    n = 50
    dt = 0.0005
    T = 1.0
    initial = 0.7
    desired = -0.9
    epsilon = 1e-3  # get to factor epsilon with T seconds

    # Get activity vector and initial decoders
    with nengo.Network(seed=seed) as model:
        x = nengo.Ensemble(
            n, 1, seed=seed, neuron_type=nengo.neurons.LIFRate())
        y = nengo.Node(size_in=1)
        conn = nengo.Connection(x, y, synapse=None)

    sim = Simulator(model, dt=dt)
    a = get_activities(sim.model, x, [initial])
    d = sim.data[conn].weights
    assert np.any(a > 0)

    # Use util function to calculate learning_rate
    init_error = float(desired - np.dot(d, a))
    learning_rate, gamma = pes_learning_rate(
        epsilon / abs(init_error), a, T, dt)

    # Build model with no filtering on any connections
    with model:
        stim = nengo.Node(output=initial)
        ystar = nengo.Node(output=desired)

        conn.learning_rule_type = nengo.PES(
            pre_tau=1e-15, learning_rate=learning_rate)

        nengo.Connection(stim, x, synapse=None)
        nengo.Connection(ystar, conn.learning_rule, synapse=0, transform=-1)
        nengo.Connection(y, conn.learning_rule, synapse=0)

        p = nengo.Probe(y, synapse=None)
        decoders = nengo.Probe(conn, 'weights', synapse=None)

    sim = Simulator(model, dt=dt)
    sim.run(T)

    # Check that the final error is exactly epsilon
    assert np.allclose(abs(desired - sim.data[p][-1]), epsilon)

    # Check that all of the errors are given exactly by gamma**k
    k = np.arange(len(sim.trange()))
    error = init_error * gamma ** k
    assert np.allclose(sim.data[p].flatten(), desired - error)

    # Check that all of the decoders are equal to their analytical solution
    dk = d.T + init_error * a.T[:, None] * (1 - gamma ** k) / np.dot(a, a.T)
    assert np.allclose(dk, np.squeeze(sim.data[decoders].T))

    plt.figure()
    plt.plot(sim.trange(), sim.data[p], lw=5, alpha=0.5)
    plt.plot(sim.trange(), desired - error, linestyle='--', lw=5, alpha=0.5)
