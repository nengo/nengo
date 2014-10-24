import logging

import numpy as np
import pytest

import nengo
from nengo.utils.connection import target_function
from nengo.utils.distributions import UniformHypersphere

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
def test_target_function(Simulator, nl_nodirect, plt, dimension):

    model = nengo.Network("Connection Helper", seed=12)

    eval_points = UniformHypersphere().sample(
        1000, dimension, np.random.RandomState(seed=12))

    targets = eval_points ** 2

    with model:
        model.config[nengo.Ensemble].neuron_type = nl_nodirect()
        inp = nengo.Node(np.sin)
        ens1 = nengo.Ensemble(100, dimension)
        ens2 = nengo.Ensemble(100, dimension)
        ens3 = nengo.Ensemble(100, dimension)
        transform = [[1] if i % 2 == 0 else [-1]
                     for i in range(dimension)]
        nengo.Connection(inp, ens1, transform=transform)
        nengo.Connection(ens1, ens2,
                         **target_function(eval_points, targets))
        nengo.Connection(ens1, ens3, function=lambda x: x ** 2)
        probe1 = nengo.Probe(ens2, synapse=0.03)
        probe2 = nengo.Probe(ens3, synapse=0.03)

    sim = nengo.Simulator(model)
    sim.run(1)

    plt.plot(sim.trange(), sim.data[probe1])
    plt.plot(sim.trange(), sim.data[probe2], '--')

    assert np.allclose(sim.data[probe1], sim.data[probe2], atol=0.2)
