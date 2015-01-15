import logging

import numpy as np
import pytest

import nengo
from nengo.dists import UniformHypersphere
from nengo.utils.connection import target_function

logger = logging.getLogger(__name__)


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

    sim = Simulator(model)
    sim.run(0.5)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[probe1])
    plt.title('Square manually with target_function')
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(), sim.data[probe2])
    plt.title('Square by passing in function to connection')
    plt.saveas = ('utils.test_connection.test_target_function_%d_%g.pdf'
                  % (dimensions, radius))

    assert np.allclose(sim.data[probe1], sim.data[probe2], atol=0.2 * radius)
