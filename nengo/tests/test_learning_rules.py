import numpy as np
import pytest

import nengo
from nengo.learning_rules import LearningRuleParam, Oja
from nengo.solvers import LstsqL2nz


def test_pes_initial_weights(Simulator, nl_nodirect, plt, seed, rng):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=2)

        initial_weights = rng.uniform(size=(a.n_neurons, u_learned.n_neurons))
        nengo.Connection(u, a)
        err_conn = nengo.Connection(e, u_learned, modulatory=True)
        nengo.Connection(a.neurons, u_learned.neurons,
                         transform=initial_weights,
                         learning_rule=nengo.PES(err_conn, 10))

        nengo.Connection(u_learned, e, transform=-1)
        nengo.Connection(u, e)

        u_learned_p = nengo.Probe(u_learned, synapse=0.05)
        e_p = nengo.Probe(e, synapse=0.05)

    sim = Simulator(m)
    sim.run(1.)
    t = sim.trange()

    plt.plot(t, sim.data[u_learned_p])
    plt.plot(t, sim.data[e_p])

    tmask = t > 0.9
    assert np.allclose(sim.data[u_learned_p][tmask], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tmask], 0, atol=0.05)


def test_pes_nef_weights(Simulator, nl_nodirect, plt, seed):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=2)

        nengo.Connection(u, a)
        err_conn = nengo.Connection(e, u_learned, modulatory=True)
        nengo.Connection(a, u_learned,
                         learning_rule=nengo.PES(err_conn, 5),
                         solver=LstsqL2nz(weights=True))

        nengo.Connection(u_learned, e, transform=-1)
        nengo.Connection(u, e)

        u_learned_p = nengo.Probe(u_learned, synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)

    sim = Simulator(m)
    sim.run(1.)
    t = sim.trange()

    plt.plot(t, sim.data[u_learned_p])
    plt.plot(t, sim.data[e_p])

    tmask = t > 0.9
    assert np.allclose(sim.data[u_learned_p][tmask], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tmask], 0, atol=0.05)


def test_pes_decoders(Simulator, nl_nodirect, seed):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=2)

        nengo.Connection(u, a)
        nengo.Connection(u_learned, e, transform=-1)
        nengo.Connection(u, e)
        e_c = nengo.Connection(e, u_learned, modulatory=True)
        nengo.Connection(a, u_learned, learning_rule=nengo.PES(e_c))

        u_learned_p = nengo.Probe(u_learned, synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)

    sim = Simulator(m)
    sim.run(1.)

    tmask = sim.trange() > 0.9
    assert np.allclose(sim.data[u_learned_p][tmask], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tmask], 0, atol=0.05)


def test_pes_decoders_multidimensional(Simulator, nl_nodirect, seed):
    n = 200
    input_vector = [0.5, -0.5]
    learned_vector = [input_vector[0]**2 + input_vector[1]**2]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=input_vector)
        v = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=1)
        e = nengo.Ensemble(n, dimensions=1)

        nengo.Connection(u, a)
        err_conn = nengo.Connection(e, u_learned, modulatory=True)

        # initial decoded function is x[0] - x[1]
        nengo.Connection(a, u_learned, function=lambda x: x[0] - x[1],
                         learning_rule=nengo.PES(err_conn, 5))

        nengo.Connection(u_learned, e, transform=-1)

        # learned function is sum of squares
        nengo.Connection(v, e)

        u_learned_p = nengo.Probe(u_learned, synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)

    sim = Simulator(m)
    sim.run(1.)

    tmask = sim.trange() > 0.9
    assert np.allclose(sim.data[u_learned_p][tmask], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tmask], 0, atol=0.05)


@pytest.mark.parametrize('learning_rule', [
    nengo.BCM(), nengo.Oja(), [nengo.Oja(), nengo.BCM()]])
def test_unsupervised(Simulator, nl_nodirect, learning_rule, seed, rng):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=2)

        initial_weights = rng.normal(size=(a.n_neurons, u_learned.n_neurons))

        nengo.Connection(u, a)
        nengo.Connection(a.neurons, u_learned.neurons,
                         transform=initial_weights,
                         learning_rule=learning_rule)

    sim = Simulator(m)
    sim.run(1.)


def test_learningruleparam():
    """LearningRuleParam must be one or many learning rules."""
    class Test(object):
        lrp = LearningRuleParam(default=None)

    inst = Test()
    assert inst.lrp is None
    inst.lrp = Oja()
    assert isinstance(inst.lrp, Oja)
    inst.lrp = [Oja(), Oja()]
    for lr in inst.lrp:
        assert isinstance(lr, Oja)
    # Non-LR no good
    with pytest.raises(ValueError):
        inst.lrp = 'a'
    # All elements in list must be LR
    with pytest.raises(ValueError):
        inst.lrp = [Oja(), 'a', Oja()]


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
