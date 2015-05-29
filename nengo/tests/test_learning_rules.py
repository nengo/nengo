import numpy as np
import pytest

import nengo
from nengo.learning_rules import LearningRuleTypeParam, PES, BCM, Oja
from nengo.processes import WhiteSignal


def test_pes_weights(Simulator, nl_nodirect, plt, seed, rng):
    n = 200
    learned_vector = [0.5, -0.5]
    rate = 2e-3

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        b = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=2)

        initial_weights = rng.uniform(
            high=1e-3,
            size=(a.n_neurons, b.n_neurons))

        nengo.Connection(u, a)
        conn = nengo.Connection(a.neurons, b.neurons,
                                transform=initial_weights,
                                learning_rule_type=PES(rate))
        nengo.Connection(e, conn.learning_rule)

        nengo.Connection(b, e)
        nengo.Connection(u, e, transform=-1)

        b_p = nengo.Probe(b, synapse=0.05)
        e_p = nengo.Probe(e, synapse=0.05)

        # test probing rule itself
        se_p = nengo.Probe(conn.learning_rule, 'correction', synapse=0.05)

    sim = Simulator(m)
    sim.run(1.)
    t = sim.trange()

    plt.subplot(311)
    plt.plot(t, sim.data[b_p])
    plt.ylabel("Post decoded value")
    plt.subplot(312)
    plt.plot(t, sim.data[e_p])
    plt.ylabel("Error decoded value")
    plt.subplot(313)
    plt.plot(t, sim.data[se_p] / rate)
    plt.ylabel("PES correction")
    plt.xlabel("Time (s)")

    tend = t > 0.9
    assert np.allclose(sim.data[b_p][tend], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tend], 0, atol=0.05)
    assert np.allclose(sim.data[se_p][tend] / rate, 0, atol=0.05)


def test_pes_decoders(Simulator, nl_nodirect, seed, plt):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        b = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=2)

        nengo.Connection(u, a)
        nengo.Connection(b, e)
        nengo.Connection(u, e, transform=-1)
        conn = nengo.Connection(a, b, learning_rule_type=PES())
        nengo.Connection(e, conn.learning_rule)

        b_p = nengo.Probe(b, synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)
        dec_p = nengo.Probe(conn, 'decoders', sample_every=0.01)

    sim = Simulator(m)
    sim.run(0.5)
    t = sim.trange()

    plt.subplot(2, 1, 1)
    plt.plot(t, sim.data[b_p], label="Post")
    plt.plot(t, sim.data[e_p], label="Error")
    plt.legend(loc="best", fontsize="x-small")
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(dt=0.01), sim.data[dec_p][..., 0])
    plt.title("Change in one 2D decoder")
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding weight")

    tend = t > 0.4
    assert np.allclose(sim.data[b_p][tend], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tend], 0, atol=0.05)
    assert not np.all(sim.data[dec_p][0] == sim.data[dec_p][-1])


def test_pes_decoders_multidimensional(Simulator, seed, plt):
    n = 200
    input_vector = [0.5, -0.5]
    learned_vector = [input_vector[0]**2 + input_vector[1]**2]

    m = nengo.Network(seed=seed)
    with m:
        u = nengo.Node(output=input_vector)
        v = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        b = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=1)

        nengo.Connection(u, a)

        # initial decoded function is x[0] - x[1]
        conn = nengo.Connection(a, b[0], function=lambda x: x[0] - x[1],
                                learning_rule_type=PES(1e-3))
        nengo.Connection(e, conn.learning_rule)

        nengo.Connection(b[0], e)

        # learned function is sum of squares
        nengo.Connection(v, e, transform=-1)

        b_p = nengo.Probe(b[0], synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)
        dec_p = nengo.Probe(conn, 'decoders', sample_every=0.01)

    sim = Simulator(m)
    sim.run(0.5)
    t = sim.trange()

    plt.subplot(2, 1, 1)
    plt.plot(t, sim.data[b_p], label="Post")
    plt.plot(t, sim.data[e_p], label="Error")
    plt.legend(loc="best", fontsize="x-small")
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(dt=0.01), sim.data[dec_p][..., 0])
    plt.title("Change in one 1D decoder")
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding weight")

    tend = t > 0.4
    assert np.allclose(sim.data[b_p][tend], learned_vector, atol=0.05)
    assert np.allclose(sim.data[e_p][tend], 0, atol=0.05)


@pytest.mark.parametrize('learning_rule_type', [
    BCM(learning_rate=1e-8),
    Oja(learning_rate=1e-5),
    [Oja(learning_rate=1e-5), BCM(learning_rate=1e-8)]])
def test_unsupervised(Simulator, learning_rule_type, seed, rng, plt):
    n = 200

    m = nengo.Network(seed=seed)
    with m:
        u = nengo.Node(WhiteSignal(0.5, high=5), size_out=2)
        a = nengo.Ensemble(n, dimensions=2)
        b = nengo.Ensemble(n, dimensions=2)

        initial_weights = rng.uniform(
            high=1e-3,
            size=(a.n_neurons, b.n_neurons))

        nengo.Connection(u, a)
        conn = nengo.Connection(a.neurons, b.neurons,
                                transform=initial_weights,
                                learning_rule_type=learning_rule_type)
        inp_p = nengo.Probe(u)
        trans_p = nengo.Probe(conn, 'transform', sample_every=0.01)

        ap = nengo.Probe(a, synapse=0.03)
        up = nengo.Probe(b, synapse=0.03)

    sim = Simulator(m, seed=seed+1)
    sim.run(0.5)
    t = sim.trange()

    plt.subplot(2, 1, 1)
    plt.plot(t, sim.data[inp_p], label="Input")
    plt.plot(t, sim.data[ap], label="Pre")
    plt.plot(t, sim.data[up], label="Post")
    plt.legend(loc="best", fontsize="x-small")
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(dt=0.01), sim.data[trans_p][..., 4])
    plt.xlabel("Time (s)")
    plt.ylabel("Transform weight")

    assert not np.all(sim.data[trans_p][0] == sim.data[trans_p][-1])


def learning_net(learning_rule, net, rng):
    with net:
        u = nengo.Node(output=1.0)
        pre = nengo.Ensemble(10, dimensions=1)
        post = nengo.Ensemble(10, dimensions=1)
        initial_weights = rng.uniform(high=1e-3,
                                      size=(pre.n_neurons, post.n_neurons))
        conn = nengo.Connection(pre.neurons, post.neurons,
                                transform=initial_weights,
                                learning_rule_type=learning_rule())
        if learning_rule is nengo.PES:
            learning_rule.learning_rate = 1e-5
            err = nengo.Ensemble(10, dimensions=1)
            nengo.Connection(u, err)
            nengo.Connection(err, conn.learning_rule)

        activity_p = nengo.Probe(pre.neurons, synapse=0.01)
        trans_p = nengo.Probe(conn, 'transform', synapse=.01, sample_every=.01)
    return net, activity_p, trans_p


@pytest.mark.parametrize('learning_rule', [nengo.PES, nengo.BCM, nengo.Oja])
def test_dt_dependence(Simulator, plt, learning_rule, seed, rng):
    """Learning rules should work the same regardless of dt."""
    m, activity_p, trans_p = learning_net(
        learning_rule, nengo.Network(seed=seed), rng)

    trans_data = []
    # Using dts greater near tau_ref (0.002 by default) causes learning to
    # differ due to lowered presynaptic firing rate
    dts = (0.0001, 0.001)
    colors = ('b', 'g', 'r')
    for c, dt in zip(colors, dts):
        sim = Simulator(m, dt=dt)
        sim.run(0.1)
        trans_data.append(sim.data[trans_p])
        plt.subplot(2, 1, 1)
        plt.plot(sim.trange(dt=0.01), sim.data[trans_p][..., 0], c=c)
        plt.subplot(2, 1, 2)
        plt.plot(sim.trange(), sim.data[activity_p], c=c)

    plt.subplot(2, 1, 1)
    plt.xlim(right=sim.trange()[-1])
    plt.ylabel("Connection weight")
    plt.subplot(2, 1, 2)
    plt.xlim(right=sim.trange()[-1])
    plt.ylabel("Presynaptic activity")

    assert np.allclose(trans_data[0], trans_data[1], atol=3e-3)
    assert not np.all(sim.data[trans_p][0] == sim.data[trans_p][-1])


@pytest.mark.parametrize('learning_rule', [nengo.PES, nengo.BCM, nengo.Oja])
def test_reset(Simulator, learning_rule, plt, seed, rng):
    """Make sure resetting learning rules resets all state."""
    m, activity_p, trans_p = learning_net(
        learning_rule, nengo.Network(seed=seed), rng)

    sim = Simulator(m)
    sim.run(0.1)
    sim.run(0.2)

    first_t = sim.trange()
    first_t_trans = sim.trange(dt=0.01)
    first_activity_p = np.array(sim.data[activity_p], copy=True)
    first_trans_p = np.array(sim.data[trans_p], copy=True)

    sim.reset()
    sim.run(0.3)

    plt.subplot(2, 1, 1)
    plt.ylabel("Neural activity")
    plt.plot(first_t, first_activity_p, c='b')
    plt.plot(sim.trange(), sim.data[activity_p], c='g')
    plt.subplot(2, 1, 2)
    plt.ylabel("Connection weight")
    plt.plot(first_t_trans, first_trans_p[..., 0], c='b')
    plt.plot(sim.trange(dt=0.01), sim.data[trans_p][..., 0], c='g')

    assert np.all(sim.trange() == first_t)
    assert np.all(sim.trange(dt=0.01) == first_t_trans)
    assert np.all(sim.data[activity_p] == first_activity_p)
    assert np.all(sim.data[trans_p] == first_trans_p)


def test_learningruletypeparam():
    """LearningRuleTypeParam must be one or many learning rules."""
    class Test(object):
        lrp = LearningRuleTypeParam(default=None)

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


def test_learningrule_attr(seed):
    """Test learning_rule attribute on Connection"""
    def check_rule(rule, conn, rule_type):
        assert rule.connection is conn and rule.learning_rule_type is rule_type

    with nengo.Network(seed=seed):
        a, b, e = [nengo.Ensemble(10, 2) for i in range(3)]
        # nengo.Connection(e, b)  # dummy error connection

        r1 = PES()
        c1 = nengo.Connection(a.neurons, b.neurons, learning_rule_type=r1)
        check_rule(c1.learning_rule, c1, r1)

        r2 = [PES(), BCM()]
        c2 = nengo.Connection(a.neurons, b.neurons, learning_rule_type=r2)
        assert isinstance(c2.learning_rule, list)
        for rule, rule_type in zip(c2.learning_rule, r2):
            check_rule(rule, c2, rule_type)

        r3 = dict(oja=Oja(), bcm=BCM())
        c3 = nengo.Connection(a.neurons, b.neurons, learning_rule_type=r3)
        assert isinstance(c3.learning_rule, dict)
        assert set(c3.learning_rule) == set(r3)  # assert same keys
        for key in r3:
            check_rule(c3.learning_rule[key], c3, r3[key])
