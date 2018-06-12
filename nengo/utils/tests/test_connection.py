import numpy as np
import pytest
import matplotlib.tri as tri

import nengo
from nengo.dists import UniformHypersphere
from nengo.utils.connection import target_function, eval_point_decoding
from nengo.utils.numpy import rms


@pytest.mark.parametrize("dimensions", [1, 4])
@pytest.mark.parametrize("radius", [1, 2.0])
@pytest.mark.filterwarnings("ignore:'targets' can be passed directly")
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
        xi, yi = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
        triang = tri.Triangulation(xy[:, 0], xy[:, 1])
        interp_lin = tri.LinearTriInterpolator(triang, z.ravel())
        zi = interp_lin(xi, yi)
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
