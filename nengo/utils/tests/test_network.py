import numpy as np
from numpy.testing import assert_allclose
import pytest

import nengo
from nengo.utils.network import activate_direct_mode, inhibit_net


def test_withself():
    model = nengo.Network(label='test_withself')
    with model:
        n1 = nengo.Node(0.5)
        assert n1 in model.nodes
        e1 = nengo.Ensemble(10, dimensions=1)
        assert e1 in model.ensembles
        c1 = nengo.Connection(n1, e1)
        assert c1 in model.connections
        ea1 = nengo.networks.EnsembleArray(10, n_ensembles=2)
        assert ea1 in model.networks
        assert len(ea1.ensembles) == 2
        n2 = ea1.add_output("out", None)
        assert n2 in ea1.nodes
        with ea1:
            e2 = nengo.Ensemble(10, dimensions=1)
            assert e2 in ea1.ensembles
    assert len(nengo.Network.context) == 0


def test_activate_direct_mode():
    with nengo.Network() as model:
        direct_mode_ens = [nengo.Ensemble(10, 1), nengo.Ensemble(10, 1)]
        non_direct_pre = nengo.Ensemble(10, 1)
        non_direct_post = nengo.Ensemble(10, 1)
        non_direct_probe = nengo.Ensemble(10, 1)
        non_direct_mode_ens = [
            non_direct_pre, non_direct_post, non_direct_probe]

        nengo.Connection(direct_mode_ens[0], direct_mode_ens[1])

        nengo.Connection(non_direct_pre.neurons[0], direct_mode_ens[0])
        nengo.Connection(direct_mode_ens[1], non_direct_post.neurons[0])
        nengo.Probe(non_direct_probe.neurons)

    activate_direct_mode(model)

    for ens in direct_mode_ens:
        assert type(ens.neuron_type) is nengo.Direct
    for ens in non_direct_mode_ens:
        assert type(ens.neuron_type) is not nengo.Direct


@pytest.mark.parametrize('learning_rule, weights', (
    (nengo.PES(), False),
    (nengo.BCM(), True),
    (nengo.Oja(), True),
    (nengo.Voja(), False)
))
def test_activate_direct_mode_learning(RefSimulator, learning_rule, weights):
    with nengo.Network() as model:
        pre = nengo.Ensemble(10, 1)
        post = nengo.Ensemble(10, 1)
        conn = nengo.Connection(
            pre, post, solver=nengo.solvers.LstsqL2(weights=weights))
        conn.learning_rule_type = learning_rule

    activate_direct_mode(model)

    with RefSimulator(model) as sim:
        sim.run(0.01)


def test_inhibit_net(RefSimulator, plt):
    with nengo.Network() as model:
        ea = nengo.networks.EnsembleArray(10, 10)
        node = nengo.Node(1)
        connections = inhibit_net(node, ea)

        nengo.Connection(node, ea.input,
                         transform=np.ones((ea.n_ensembles, 1)))
        p = nengo.Probe(ea.output, synapse=0.01)

    assert len(connections) == 10
    for c in connections:
        assert c.pre is node
        assert c.post.ensemble in ea.all_ensembles

    with RefSimulator(model) as sim:
        sim.run(0.3)

    plt.plot(sim.trange(), sim.data[p])
    plt.xlabel("Time [s]")

    print(np.max(np.abs(sim.data[p][sim.trange() > 0.1])))
    assert_allclose(sim.data[p][sim.trange() > 0.1], 0., atol=1e-4)
