import numpy as np
import pytest

import nengo
from nengo.dists import UniformHypersphere
from nengo.learning_rules import LearningRuleTypeParam, PES, BCM, Oja, Voja
from nengo.processes import WhiteSignal


def _test_pes(Simulator, nl, plt, seed,
              pre_neurons=False, post_neurons=False, weight_solver=False,
              vin=np.array([0.5, -0.5]), vout=None, n=200,
              function=None, transform=np.array(1.), rate=1e-3):

    vout = np.array(vin) if vout is None else vout

    model = nengo.Network(seed=seed)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl()

        u = nengo.Node(output=vin)
        v = nengo.Node(output=vout)
        a = nengo.Ensemble(n, dimensions=u.size_out)
        b = nengo.Ensemble(n, dimensions=u.size_out)
        e = nengo.Ensemble(n, dimensions=v.size_out)

        nengo.Connection(u, a)

        bslice = b[:v.size_out] if v.size_out < u.size_out else b
        pre = a.neurons if pre_neurons else a
        post = b.neurons if post_neurons else bslice

        conn = nengo.Connection(pre, post,
                                function=function, transform=transform,
                                learning_rule_type=PES(rate))
        if weight_solver:
            conn.solver = nengo.solvers.LstsqL2(weights=True)

        nengo.Connection(v, e, transform=-1)
        nengo.Connection(bslice, e)
        nengo.Connection(e, conn.learning_rule)

        b_p = nengo.Probe(bslice, synapse=0.03)
        e_p = nengo.Probe(e, synapse=0.03)

        weights_p = nengo.Probe(conn, 'weights', sample_every=0.01)
        corr_p = nengo.Probe(conn.learning_rule, 'correction', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.5)
    t = sim.trange()
    weights = sim.data[weights_p]

    plt.subplot(311)
    plt.plot(t, sim.data[b_p])
    plt.ylabel("Post decoded value")
    plt.subplot(312)
    plt.plot(t, sim.data[e_p])
    plt.ylabel("Error decoded value")
    plt.subplot(313)
    plt.plot(t, sim.data[corr_p] / rate)
    plt.ylabel("PES correction")
    plt.xlabel("Time (s)")

    tend = t > 0.4
    assert np.allclose(sim.data[b_p][tend], vout, atol=0.05)
    assert np.allclose(sim.data[e_p][tend], 0, atol=0.05)
    assert np.allclose(sim.data[corr_p][tend] / rate, 0, atol=0.05)
    assert not np.allclose(weights[0], weights[-1], atol=1e-5)


def test_pes_ens_ens(Simulator, nl_nodirect, plt, seed):
    function = lambda x: [x[1], x[0]]
    _test_pes(Simulator, nl_nodirect, plt, seed, function=function)


def test_pes_weight_solver(Simulator, plt, seed):
    function = lambda x: [x[1], x[0]]
    _test_pes(Simulator, nengo.LIF, plt, seed, function=function,
              weight_solver=True)


def test_pes_ens_slice(Simulator, plt, seed):
    vin = [0.5, -0.5]
    vout = [vin[0]**2 + vin[1]**2]
    function = lambda x: [x[0] - x[1]]
    _test_pes(Simulator, nengo.LIF, plt, seed, vin=vin, vout=vout,
              function=function)


def test_pes_neuron_neuron(Simulator, plt, seed, rng):
    n = 200
    initial_weights = rng.uniform(high=2e-4, size=(n, n))
    _test_pes(Simulator, nengo.LIF, plt, seed,
              pre_neurons=True, post_neurons=True,
              n=n, transform=initial_weights)


def test_pes_neuron_ens(Simulator, plt, seed, rng):
    n = 200
    initial_weights = rng.uniform(high=1e-4, size=(2, n))
    _test_pes(Simulator, nengo.LIF, plt, seed,
              pre_neurons=True, post_neurons=False,
              n=n, transform=initial_weights)


def test_pes_transform(Simulator, seed):
    """Test behaviour of PES when function and transform both defined."""
    n = 200
    # error must be with respect to transformed vector (conn.size_out)
    T = np.asarray([[0.5], [-0.5]])  # transform to output

    m = nengo.Network(seed=seed)
    with m:
        u = nengo.Node(output=[1])
        a = nengo.Ensemble(n, dimensions=1)
        b = nengo.Node(size_in=2)
        e = nengo.Node(size_in=1)

        nengo.Connection(u, a)
        learned_conn = nengo.Connection(
            a, b, function=lambda x: [0], transform=T,
            learning_rule_type=nengo.PES(learning_rate=1e-3))
        assert T.shape[0] == learned_conn.size_out
        assert T.shape[1] == learned_conn.size_mid

        nengo.Connection(b[0], e, synapse=None)
        nengo.Connection(nengo.Node(output=-1), e)
        nengo.Connection(
            e, learned_conn.learning_rule, transform=T, synapse=None)

        p_b = nengo.Probe(b, synapse=0.05)

    sim = nengo.Simulator(m)
    sim.run(1.0)
    tend = sim.trange() > 0.7

    assert np.allclose(sim.data[p_b][tend], [1, -1], atol=1e-2)


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
        b = nengo.Ensemble(n+1, dimensions=2)

        initial_weights = rng.uniform(
            high=1e-3, size=(b.n_neurons, a.n_neurons))

        nengo.Connection(u, a)
        conn = nengo.Connection(a.neurons, b.neurons,
                                transform=initial_weights,
                                learning_rule_type=learning_rule_type)
        inp_p = nengo.Probe(u)
        weights_p = nengo.Probe(conn, 'weights', sample_every=0.01)

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
    plt.plot(sim.trange(dt=0.01), sim.data[weights_p][..., 4])
    plt.xlabel("Time (s)")
    plt.ylabel("Weights")

    assert not np.all(sim.data[weights_p][0] == sim.data[weights_p][-1])


def learning_net(learning_rule, net, rng):
    with net:
        if learning_rule is nengo.PES:
            learning_rule_type = learning_rule(learning_rate=1e-5)
        else:
            learning_rule_type = learning_rule()

        u = nengo.Node(output=1.0)
        pre = nengo.Ensemble(10, dimensions=1)
        post = nengo.Ensemble(10, dimensions=1)
        initial_weights = rng.uniform(high=1e-3,
                                      size=(pre.n_neurons, post.n_neurons))
        conn = nengo.Connection(pre.neurons, post.neurons,
                                transform=initial_weights,
                                learning_rule_type=learning_rule_type)
        if learning_rule is nengo.PES:
            err = nengo.Ensemble(10, dimensions=1)
            nengo.Connection(u, err)
            nengo.Connection(err, conn.learning_rule)

        activity_p = nengo.Probe(pre.neurons, synapse=0.01)
        weights_p = nengo.Probe(conn, 'weights', synapse=.01, sample_every=.01)
    return net, activity_p, weights_p


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


def test_voja_encoders(Simulator, nl_nodirect, rng, seed, plt):
    """Tests that voja changes active encoders to the input."""
    n = 200
    learned_vector = np.asarray([0.3, -0.4, 0.6])
    learned_vector /= np.linalg.norm(learned_vector)
    n_change = n // 2  # modify first half of the encoders

    # Set the first half to always fire with random encoders, and the
    # remainder to never fire due to their encoder's dot product with the input
    intercepts = np.asarray([-1]*n_change + [0.99]*(n - n_change))
    rand_encoders = UniformHypersphere(surface=True).sample(
        n_change, len(learned_vector), rng=rng)
    encoders = np.append(
        rand_encoders, [-learned_vector] * (n - n_change), axis=0)

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        x = nengo.Ensemble(n, dimensions=len(learned_vector),
                           intercepts=intercepts, encoders=encoders,
                           radius=2.0)  # to test encoder scaling

        conn = nengo.Connection(
            u, x, synapse=None, learning_rule_type=Voja(learning_rate=1e-1))
        p_enc = nengo.Probe(conn.learning_rule, 'scaled_encoders')

    sim = Simulator(m)
    sim.run(1.0)
    t = sim.trange()
    tend = t > 0.5

    # Voja's rule relies on knowing exactly how the encoders were scaled
    # during the build process, because it modifies the scaled_encoders signal
    # proportional to this factor. Therefore, we should check that its
    # assumption actually holds.
    encoder_scale = (sim.model.params[x].gain / x.radius)[:, np.newaxis]
    assert np.allclose(sim.data[x].encoders,
                       sim.data[x].scaled_encoders / encoder_scale)

    # Check that the last half kept the same encoders throughout the simulation
    assert np.allclose(
        sim.data[p_enc][0, n_change:], sim.data[p_enc][:, n_change:])
    # and that they are also equal to their originally assigned value
    assert np.allclose(
        sim.data[p_enc][0, n_change:] / encoder_scale[n_change:],
        -learned_vector)

    # Check that the first half converged to the input
    assert np.allclose(
        sim.data[p_enc][tend, :n_change] / encoder_scale[:n_change],
        learned_vector, atol=0.01)


def test_voja_modulate(Simulator, nl_nodirect, seed, plt):
    """Tests that voja's rule can be modulated on/off."""
    n = 200
    learned_vector = np.asarray([0.5])

    def control_signal(t):
        """Modulates the learning on/off."""
        return 0 if t < 0.5 else -1

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        control = nengo.Node(output=control_signal)
        u = nengo.Node(output=learned_vector)
        x = nengo.Ensemble(n, dimensions=len(learned_vector))

        conn = nengo.Connection(
            u, x, synapse=None, learning_rule_type=Voja(None))
        nengo.Connection(control, conn.learning_rule, synapse=None)

        p_enc = nengo.Probe(conn.learning_rule, 'scaled_encoders')

    sim = Simulator(m)
    sim.run(1.0)
    tend = sim.trange() > 0.5

    # Check that encoders stop changing after 0.5s
    assert np.allclose(sim.data[p_enc][tend], sim.data[p_enc][-1])

    # Check that encoders changed during first 0.5s
    i = np.where(tend)[0][0]  # first time point after changeover
    assert not np.allclose(sim.data[p_enc][0], sim.data[p_enc][i])


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = PES(2e-3, 4e-3)
    b = PES(2e-3, 4e-3)
    c = PES(2e-3, 5e-3)

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely

    with pytest.raises((ValueError, RuntimeError)):
        a.learning_rate = 1e-1
