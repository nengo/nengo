import numpy as np
import pytest

import nengo
from nengo.builder import Builder
from nengo.builder.operator import Reset, Copy
from nengo.builder.signal import Signal
from nengo.dists import UniformHypersphere
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleTypeParam, PES, BCM, Oja, Voja
from nengo.processes import WhiteSignal
from nengo.synapses import Alpha, Lowpass


def best_weights(weight_data):
    return np.argmax(np.sum(np.var(weight_data, axis=0), axis=0))


def _test_pes(
    Simulator,
    nl,
    plt,
    seed,
    allclose,
    pre_neurons=False,
    post_neurons=False,
    weight_solver=False,
    vin=np.array([0.5, -0.5]),
    vout=None,
    n=200,
    function=None,
    transform=np.array(1.0),
    rate=1e-3,
):
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

        bslice = b[: v.size_out] if v.size_out < u.size_out else b
        pre = a.neurons if pre_neurons else a
        post = b.neurons if post_neurons else bslice

        conn = nengo.Connection(
            pre,
            post,
            function=function,
            transform=transform,
            learning_rule_type=PES(rate),
        )
        if weight_solver:
            conn.solver = nengo.solvers.LstsqL2(weights=True)

        nengo.Connection(v, e, transform=-1)
        nengo.Connection(bslice, e)
        nengo.Connection(e, conn.learning_rule)

        b_p = nengo.Probe(bslice, synapse=0.03)
        e_p = nengo.Probe(e, synapse=0.03)

        weights_p = nengo.Probe(conn, "weights", sample_every=0.01)

    with Simulator(model) as sim:
        sim.run(0.5)
    t = sim.trange()
    weights = sim.data[weights_p]

    plt.subplot(211)
    plt.plot(t, sim.data[b_p])
    plt.ylabel("Post decoded value")
    plt.subplot(212)
    plt.plot(t, sim.data[e_p])
    plt.ylabel("Error decoded value")
    plt.xlabel("Time (s)")

    tend = t > 0.4
    assert allclose(sim.data[b_p][tend], vout, atol=0.05)
    assert allclose(sim.data[e_p][tend], 0, atol=0.05)
    assert not allclose(weights[0], weights[-1], atol=1e-5, record_rmse=False)


def test_pes_ens_ens(Simulator, nl_nodirect, plt, seed, allclose):
    function = lambda x: [x[1], x[0]]
    _test_pes(Simulator, nl_nodirect, plt, seed, allclose, function=function)


def test_pes_weight_solver(Simulator, plt, seed, allclose):
    function = lambda x: [x[1], x[0]]
    _test_pes(
        Simulator, nengo.LIF, plt, seed, allclose, function=function, weight_solver=True
    )


def test_pes_ens_slice(Simulator, plt, seed, allclose):
    vin = [0.5, -0.5]
    vout = [vin[0] ** 2 + vin[1] ** 2]
    function = lambda x: [x[0] - x[1]]
    _test_pes(
        Simulator, nengo.LIF, plt, seed, allclose, vin=vin, vout=vout, function=function
    )


def test_pes_neuron_neuron(Simulator, plt, seed, rng, allclose):
    n = 200
    initial_weights = rng.uniform(high=4e-4, size=(n, n))
    _test_pes(
        Simulator,
        nengo.LIF,
        plt,
        seed,
        allclose,
        pre_neurons=True,
        post_neurons=True,
        n=n,
        transform=initial_weights,
    )


def test_pes_neuron_ens(Simulator, plt, seed, rng, allclose):
    n = 200
    initial_weights = rng.uniform(high=1e-4, size=(2, n))
    _test_pes(
        Simulator,
        nengo.LIF,
        plt,
        seed,
        allclose,
        pre_neurons=True,
        post_neurons=False,
        n=n,
        transform=initial_weights,
    )


def test_pes_transform(Simulator, seed, allclose):
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
            a,
            b,
            function=lambda x: [0],
            transform=T,
            learning_rule_type=nengo.PES(learning_rate=1e-3),
        )
        assert T.shape[0] == learned_conn.size_out
        assert T.shape[1] == learned_conn.size_mid

        nengo.Connection(b[0], e, synapse=None)
        nengo.Connection(nengo.Node(output=-1), e)
        nengo.Connection(e, learned_conn.learning_rule, transform=T, synapse=None)

        p_b = nengo.Probe(b, synapse=0.05)

    with Simulator(m) as sim:
        sim.run(1.0)
    tend = sim.trange() > 0.7

    assert allclose(sim.data[p_b][tend], [1, -1], atol=1e-2)


def test_pes_multidim_error(Simulator, seed):
    """Test that PES works on error connections mapping from N to 1 dims.

    Note that the transform is applied before the learning rule, so the error
    signal should be 1-dimensional.
    """

    with nengo.Network(seed=seed) as net:
        err = nengo.Node(output=[0])
        ens1 = nengo.Ensemble(20, 3)
        ens2 = nengo.Ensemble(10, 1)

        # Case 1: ens -> ens, weights=False
        conn = nengo.Connection(
            ens1,
            ens2,
            transform=np.ones((1, 3)),
            solver=nengo.solvers.LstsqL2(weights=False),
            learning_rule_type={"pes": nengo.PES()},
        )
        nengo.Connection(err, conn.learning_rule["pes"])
        # Case 2: ens -> ens, weights=True
        conn = nengo.Connection(
            ens1,
            ens2,
            transform=np.ones((1, 3)),
            solver=nengo.solvers.LstsqL2(weights=True),
            learning_rule_type={"pes": nengo.PES()},
        )
        nengo.Connection(err, conn.learning_rule["pes"])
        # Case 3: neurons -> ens
        conn = nengo.Connection(
            ens1.neurons,
            ens2,
            transform=np.ones((1, ens1.n_neurons)),
            learning_rule_type={"pes": nengo.PES()},
        )
        nengo.Connection(err, conn.learning_rule["pes"])

    with Simulator(net) as sim:
        sim.run(0.01)


@pytest.mark.parametrize("pre_synapse", [0, Lowpass(tau=0.05), Alpha(tau=0.005)])
def test_pes_synapse(Simulator, seed, pre_synapse, allclose):
    rule = PES(pre_synapse=pre_synapse)

    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(output=WhiteSignal(0.5, high=10))
        x = nengo.Ensemble(100, 1)

        nengo.Connection(stim, x, synapse=None)
        conn = nengo.Connection(x, x, learning_rule_type=rule)

        p_neurons = nengo.Probe(x.neurons, synapse=pre_synapse)
        p_pes = nengo.Probe(conn.learning_rule, "activities")

    with Simulator(model) as sim:
        sim.run(0.5)

    assert allclose(sim.data[p_neurons][1:, :], sim.data[p_pes][:-1, :])


@pytest.mark.parametrize("weights", [False, True])
def test_pes_recurrent_slice(Simulator, seed, weights):
    """Test that PES works on recurrent connections from N to 1 dims."""

    with nengo.Network(seed=seed) as net:
        err = nengo.Node(output=[-10])
        stim = nengo.Node(output=[0, 0])
        post = nengo.Ensemble(20, 2)
        nengo.Connection(stim, post)

        conn = nengo.Connection(
            post,
            post[1],
            function=lambda x: 0.0,
            solver=nengo.solvers.LstsqL2(weights=weights),
            learning_rule_type=nengo.PES(),
        )

        nengo.Connection(err, conn.learning_rule)
        p = nengo.Probe(post, synapse=0.025)

    with Simulator(net) as sim:
        sim.run(0.1)

    # Learning rule should drive second dimension high, but not first
    assert np.all(sim.data[p][-10:, 0] < 0.2)
    assert np.all(sim.data[p][-10:, 1] > 0.8)


def test_pes_cycle(Simulator):
    """Test that PES works when connection output feeds back into error."""

    with nengo.Network() as net:
        a = nengo.Ensemble(10, 1)
        b = nengo.Node(size_in=1)
        c = nengo.Connection(a, b, synapse=None, learning_rule_type=nengo.PES())
        nengo.Connection(b, c.learning_rule, synapse=None)

    with Simulator(net):
        # just checking that this builds without error
        pass


@pytest.mark.parametrize(
    "rule_type, solver",
    [
        (BCM(learning_rate=1e-8), False),
        (Oja(learning_rate=1e-5), False),
        ([Oja(learning_rate=1e-5), BCM(learning_rate=1e-8)], False),
        ([Oja(learning_rate=1e-5), BCM(learning_rate=1e-8)], True),
    ],
)
def test_unsupervised(Simulator, rule_type, solver, seed, rng, plt, allclose):
    n = 200

    m = nengo.Network(seed=seed)
    with m:
        u = nengo.Node(WhiteSignal(0.5, high=10), size_out=2)
        a = nengo.Ensemble(n, dimensions=2)
        b = nengo.Ensemble(n + 1, dimensions=2)
        nengo.Connection(u, a)

        if solver:
            conn = nengo.Connection(a, b, solver=nengo.solvers.LstsqL2(weights=True))
        else:
            initial_weights = rng.uniform(high=1e-3, size=(b.n_neurons, a.n_neurons))
            conn = nengo.Connection(a.neurons, b.neurons, transform=initial_weights)
        conn.learning_rule_type = rule_type

        inp_p = nengo.Probe(u)
        weights_p = nengo.Probe(conn, "weights", sample_every=0.01)

        ap = nengo.Probe(a, synapse=0.03)
        up = nengo.Probe(b, synapse=0.03)

    with Simulator(m, seed=seed + 1) as sim:
        sim.run(0.5)
    t = sim.trange()

    plt.subplot(2, 1, 1)
    plt.plot(t, sim.data[inp_p], label="Input")
    plt.plot(t, sim.data[ap], label="Pre")
    plt.plot(t, sim.data[up], label="Post")
    plt.legend(loc="best", fontsize="x-small")
    plt.subplot(2, 1, 2)
    best_ix = best_weights(sim.data[weights_p])
    plt.plot(sim.trange(sample_every=0.01), sim.data[weights_p][..., best_ix])
    plt.xlabel("Time (s)")
    plt.ylabel("Weights")

    assert not allclose(
        sim.data[weights_p][0], sim.data[weights_p][-1], record_rmse=False
    )


def learning_net(learning_rule=nengo.PES, net=None, rng=np.random):
    net = nengo.Network() if net is None else net
    with net:
        if learning_rule is nengo.PES:
            learning_rule_type = learning_rule(learning_rate=1e-5)
        else:
            learning_rule_type = learning_rule()

        u = nengo.Node(output=1.0)
        pre = nengo.Ensemble(10, dimensions=1)
        post = nengo.Ensemble(10, dimensions=1)
        initial_weights = rng.uniform(high=1e-3, size=(pre.n_neurons, post.n_neurons))
        conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            transform=initial_weights,
            learning_rule_type=learning_rule_type,
        )
        if learning_rule is nengo.PES:
            err = nengo.Ensemble(10, dimensions=1)
            nengo.Connection(u, err)
            nengo.Connection(err, conn.learning_rule)

        net.activity_p = nengo.Probe(pre.neurons, synapse=0.01)
        net.weights_p = nengo.Probe(conn, "weights", synapse=None, sample_every=0.01)
    return net


@pytest.mark.parametrize("learning_rule", [nengo.PES, nengo.BCM, nengo.Oja])
def test_dt_dependence(Simulator, plt, learning_rule, seed, rng, allclose):
    """Learning rules should work the same regardless of dt."""
    m = learning_net(learning_rule, nengo.Network(seed=seed), rng)

    trans_data = []
    # Using dts greater near tau_ref (0.002 by default) causes learning to
    # differ due to lowered presynaptic firing rate
    dts = (0.0001, 0.001)
    colors = ("b", "g", "r")
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    for c, dt in zip(colors, dts):
        with Simulator(m, dt=dt) as sim:
            sim.run(0.1)
        trans_data.append(sim.data[m.weights_p])
        best_ix = best_weights(sim.data[m.weights_p])
        ax1.plot(
            sim.trange(sample_every=0.01), sim.data[m.weights_p][..., best_ix], c=c
        )
        ax2.plot(sim.trange(), sim.data[m.activity_p], c=c)

    ax1.set_xlim(right=sim.trange()[-1])
    ax1.set_ylabel("Connection weight")
    ax2.set_xlim(right=sim.trange()[-1])
    ax2.set_ylabel("Presynaptic activity")

    assert allclose(trans_data[0], trans_data[1], atol=3e-3)
    assert not allclose(
        sim.data[m.weights_p][0], sim.data[m.weights_p][-1], record_rmse=False
    )


@pytest.mark.parametrize("learning_rule", [nengo.PES, nengo.BCM, nengo.Oja])
def test_reset(Simulator, learning_rule, plt, seed, rng, allclose):
    """Make sure resetting learning rules resets all state."""
    m = learning_net(learning_rule, nengo.Network(seed=seed), rng)

    with Simulator(m) as sim:
        sim.run(0.1)
        sim.run(0.2)

        first_t = sim.trange()
        first_t_trans = sim.trange(sample_every=0.01)
        first_activity_p = np.array(sim.data[m.activity_p], copy=True)
        first_weights_p = np.array(sim.data[m.weights_p], copy=True)

        sim.reset()
        sim.run(0.3)

    plt.subplot(2, 1, 1)
    plt.ylabel("Neural activity")
    plt.plot(first_t, first_activity_p, c="b")
    plt.plot(sim.trange(), sim.data[m.activity_p], c="g")
    plt.subplot(2, 1, 2)
    plt.ylabel("Connection weight")
    best_ix = best_weights(first_weights_p)
    plt.plot(first_t_trans, first_weights_p[..., best_ix], c="b")
    plt.plot(sim.trange(sample_every=0.01), sim.data[m.weights_p][..., best_ix], c="g")

    assert allclose(sim.trange(), first_t)
    assert allclose(sim.trange(sample_every=0.01), first_t_trans)
    assert allclose(sim.data[m.activity_p], first_activity_p)
    assert allclose(sim.data[m.weights_p], first_weights_p)


def test_learningruletypeparam():
    """LearningRuleTypeParam must be one or many learning rules."""

    class Test:
        lrp = LearningRuleTypeParam("lrp", default=None)

    inst = Test()
    assert inst.lrp is None
    inst.lrp = Oja()
    assert isinstance(inst.lrp, Oja)
    inst.lrp = [Oja(), Oja()]
    for lr in inst.lrp:
        assert isinstance(lr, Oja)
    # Non-LR no good
    with pytest.raises(ValueError):
        inst.lrp = "a"
    # All elements in list must be LR
    with pytest.raises(ValueError):
        inst.lrp = [Oja(), "a", Oja()]


def test_learningrule_attr(seed):
    """Test learning_rule attribute on Connection"""

    def check_rule(rule, conn, rule_type):
        assert rule.connection is conn and rule.learning_rule_type is rule_type

    with nengo.Network(seed=seed):
        a, b, e = [nengo.Ensemble(10, 2) for i in range(3)]
        T = np.ones((10, 10))

        r1 = PES()
        c1 = nengo.Connection(a.neurons, b.neurons, learning_rule_type=r1)
        check_rule(c1.learning_rule, c1, r1)

        r2 = [PES(), BCM()]
        c2 = nengo.Connection(a.neurons, b.neurons, learning_rule_type=r2, transform=T)
        assert isinstance(c2.learning_rule, list)
        for rule, rule_type in zip(c2.learning_rule, r2):
            check_rule(rule, c2, rule_type)

        r3 = dict(oja=Oja(), bcm=BCM())
        c3 = nengo.Connection(a.neurons, b.neurons, learning_rule_type=r3, transform=T)
        assert isinstance(c3.learning_rule, dict)
        assert set(c3.learning_rule) == set(r3)  # assert same keys
        for key in r3:
            check_rule(c3.learning_rule[key], c3, r3[key])


def test_voja_encoders(Simulator, nl_nodirect, rng, seed, allclose):
    """Tests that voja changes active encoders to the input."""
    n = 200
    learned_vector = np.asarray([0.3, -0.4, 0.6])
    learned_vector /= np.linalg.norm(learned_vector)
    n_change = n // 2  # modify first half of the encoders

    # Set the first half to always fire with random encoders, and the
    # remainder to never fire due to their encoder's dot product with the input
    intercepts = np.asarray([-1] * n_change + [0.99] * (n - n_change))
    rand_encoders = UniformHypersphere(surface=True).sample(
        n_change, len(learned_vector), rng=rng
    )
    encoders = np.append(rand_encoders, [-learned_vector] * (n - n_change), axis=0)

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        x = nengo.Ensemble(
            n,
            dimensions=len(learned_vector),
            intercepts=intercepts,
            encoders=encoders,
            max_rates=nengo.dists.Uniform(300.0, 400.0),
            radius=2.0,
        )  # to test encoder scaling

        conn = nengo.Connection(
            u, x, synapse=None, learning_rule_type=Voja(learning_rate=1e-1)
        )
        p_enc = nengo.Probe(conn.learning_rule, "scaled_encoders")
        p_enc_ens = nengo.Probe(x, "scaled_encoders")

    with Simulator(m) as sim:
        sim.run(1.0)
    t = sim.trange()
    tend = t > 0.5

    # Voja's rule relies on knowing exactly how the encoders were scaled
    # during the build process, because it modifies the scaled_encoders signal
    # proportional to this factor. Therefore, we should check that its
    # assumption actually holds.
    encoder_scale = (sim.data[x].gain / x.radius)[:, np.newaxis]
    assert allclose(sim.data[x].encoders, sim.data[x].scaled_encoders / encoder_scale)

    # Check that the last half kept the same encoders throughout the simulation
    assert allclose(sim.data[p_enc][0, n_change:], sim.data[p_enc][:, n_change:])
    # and that they are also equal to their originally assigned value
    assert allclose(
        sim.data[p_enc][0, n_change:] / encoder_scale[n_change:], -learned_vector
    )

    # Check that the first half converged to the input
    assert allclose(
        sim.data[p_enc][tend, :n_change] / encoder_scale[:n_change],
        learned_vector,
        atol=0.01,
    )
    # Check that encoders probed from ensemble equal encoders probed from Voja
    assert allclose(sim.data[p_enc], sim.data[p_enc_ens])


def test_voja_modulate(Simulator, nl_nodirect, seed, allclose):
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
            u, x, synapse=None, learning_rule_type=Voja(post_synapse=None)
        )
        nengo.Connection(control, conn.learning_rule, synapse=None)

        p_enc = nengo.Probe(conn.learning_rule, "scaled_encoders")

    with Simulator(m) as sim:
        sim.run(1.0)
    tend = sim.trange() > 0.5

    # Check that encoders stop changing after 0.5s
    assert allclose(sim.data[p_enc][tend], sim.data[p_enc][-1])

    # Check that encoders changed during first 0.5s
    i = np.where(tend)[0][0]  # first time point after changeover
    assert not allclose(sim.data[p_enc][0], sim.data[p_enc][i], record_rmse=False)


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = PES(learning_rate=2e-3, pre_synapse=4e-3)
    b = PES(learning_rate=2e-3, pre_synapse=4e-3)
    c = PES(learning_rate=2e-3, pre_synapse=5e-3)

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


def test_pes_direct_errors():
    """Test that applying a learning rule to a direct ensemble errors."""
    with nengo.Network():
        pre = nengo.Ensemble(10, 1, neuron_type=nengo.Direct())
        post = nengo.Ensemble(10, 1)
        conn = nengo.Connection(pre, post)
        with pytest.raises(ValidationError):
            conn.learning_rule_type = nengo.PES()


def test_custom_type(Simulator, allclose):
    """Test with custom learning rule type.

    A custom learning type may have ``size_in`` not equal to 0, 1, or None.
    """

    class TestRule(nengo.learning_rules.LearningRuleType):
        modifies = "decoders"

        def __init__(self):
            super().__init__(1.0, size_in=3)

    @Builder.register(TestRule)
    def build_test_rule(model, test_rule, rule):
        error = Signal(np.zeros(rule.connection.size_in))
        model.add_op(Reset(error))
        model.sig[rule]["in"] = error[: rule.size_in]

        model.add_op(Copy(error, model.sig[rule]["delta"]))

    with nengo.Network() as net:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        conn = nengo.Connection(
            a.neurons, b, transform=np.zeros((1, 10)), learning_rule_type=TestRule()
        )

        err = nengo.Node([1, 2, 3])
        nengo.Connection(err, conn.learning_rule, synapse=None)

        p = nengo.Probe(conn, "weights")

    with Simulator(net) as sim:
        sim.run(sim.dt * 5)

    assert allclose(sim.data[p][:, 0, :3], np.outer(np.arange(1, 6), np.arange(1, 4)))
    assert allclose(sim.data[p][:, :, 3:], 0)


@pytest.mark.parametrize("LearningRule", (nengo.PES, nengo.BCM, nengo.Voja, nengo.Oja))
def test_tau_deprecation(LearningRule):
    params = [
        ("pre_tau", "pre_synapse"),
        ("post_tau", "post_synapse"),
        ("theta_tau", "theta_synapse"),
    ]
    kwargs = {}
    for i, (p0, p1) in enumerate(params):
        if hasattr(LearningRule, p0):
            kwargs[p0] = i

    with pytest.warns(DeprecationWarning):
        l_rule = LearningRule(learning_rate=0, **kwargs)

    for i, (p0, p1) in enumerate(params):
        if hasattr(LearningRule, p0):
            assert getattr(l_rule, p0) == i
            assert getattr(l_rule, p1) == Lowpass(i)


def test_slicing(Simulator, seed, allclose):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(30, 1)
        b = nengo.Ensemble(30, 2)
        conn = nengo.Connection(
            a, b, learning_rule_type=PES(), function=lambda x: (0, 0)
        )
        nengo.Connection(nengo.Node(1.0), a)

        err1 = nengo.Node(lambda t, x: x - 0.75, size_in=1)
        nengo.Connection(b[0], err1)
        nengo.Connection(err1, conn.learning_rule[0])

        err2 = nengo.Node(lambda t, x: x + 0.5, size_in=1)
        nengo.Connection(b[1], err2)
        nengo.Connection(err2, conn.learning_rule[1])

        p = nengo.Probe(b, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange() > 0.8
    assert allclose(sim.data[p][t, 0], 0.75, atol=0.15)
    assert allclose(sim.data[p][t, 1], -0.5, atol=0.15)
