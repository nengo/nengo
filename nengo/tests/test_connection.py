from functools import partial

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.connection import ConnectionSolverParam
from nengo.dists import Choice, UniformHypersphere
from nengo.exceptions import BuildError, ValidationError
from nengo.processes import Piecewise
from nengo.solvers import LstsqL2
from nengo.transforms import Dense, NoTransform
from nengo.utils.testing import signals_allclose


def test_args(AnyNeuronType, seed, rng):
    N = 10
    d1, d2 = 3, 2

    with nengo.Network(seed=seed) as model:
        model.config[nengo.Ensemble].neuron_type = AnyNeuronType()
        A = nengo.Ensemble(N, dimensions=d1)
        B = nengo.Ensemble(N, dimensions=d2)
        nengo.Connection(
            A,
            B,
            eval_points=rng.normal(size=(500, d1)),
            synapse=0.01,
            function=np.sin,
            transform=rng.normal(size=(d2, d1)),
        )


def test_node_to_neurons(Simulator, PositiveNeuronType, plt, seed, allclose):
    N = 50

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = PositiveNeuronType()
        a = nengo.Ensemble(N, dimensions=1)
        inn = nengo.Node(output=np.sin)
        inh = nengo.Node(Piecewise({0: 0, 0.5: 1}))
        nengo.Connection(inn, a)
        nengo.Connection(inh, a.neurons, transform=[[-5]] * N)

        inn_p = nengo.Probe(inn, "output")
        a_p = nengo.Probe(a, "decoded_output", synapse=0.1)
        inh_p = nengo.Probe(inh, "output")

    with Simulator(m) as sim:
        sim.run(1.0)
    t = sim.trange()
    ideal = np.sin(t)
    ideal[t >= 0.5] = 0

    plt.plot(t, sim.data[inn_p], label="Input")
    plt.plot(t, sim.data[a_p], label="Neuron approx, synapse=0.1")
    plt.plot(t, sim.data[inh_p], label="Inhib signal")
    plt.plot(t, ideal, label="Ideal output")
    plt.legend(loc="best", fontsize="small")

    assert allclose(sim.data[a_p][-10:], 0, atol=0.1, rtol=0.01)


def test_ensemble_to_neurons(Simulator, PositiveNeuronType, plt, seed, allclose):
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Ensemble].neuron_type = PositiveNeuronType()
        ens = nengo.Ensemble(40, dimensions=1)
        inhibitor = nengo.Ensemble(40, dimensions=1)
        stim = nengo.Node(output=np.sin)
        inhibition = nengo.Node(Piecewise({0: 0, 0.5: 1}))
        nengo.Connection(stim, ens)
        nengo.Connection(inhibition, inhibitor)
        nengo.Connection(
            inhibitor, ens.neurons, transform=-10 * np.ones((ens.n_neurons, 1))
        )

        stim_p = nengo.Probe(stim, "output")
        ens_p = nengo.Probe(ens, "decoded_output", synapse=0.05)
        inhibitor_p = nengo.Probe(inhibitor, "decoded_output", synapse=0.05)
        inhibition_p = nengo.Probe(inhibition, "output")

    with Simulator(net) as sim:
        sim.run(1.0)
    t = sim.trange()
    ideal = np.sin(t)
    ideal[t >= 0.5] = 0

    plt.plot(t, sim.data[stim_p], label="Input")
    plt.plot(t, sim.data[ens_p], label="`ens` value, pstc=0.05")
    plt.plot(t, sim.data[inhibitor_p], label="`inhibitor` value, pstc=0.05")
    plt.plot(t, sim.data[inhibition_p], label="Inhibition signal")
    plt.plot(t, ideal, label="Ideal output")
    plt.legend(loc=0, prop={"size": 10})

    assert allclose(sim.data[ens_p][-10:], 0, atol=0.1, rtol=0.01)
    assert allclose(sim.data[inhibitor_p][-10:], 1, atol=0.1, rtol=0.01)


def test_node_to_ensemble(Simulator, NonDirectNeuronType, plt, seed, allclose):
    N = 50

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = NonDirectNeuronType()
        input_node = nengo.Node(output=lambda t: [np.sin(t * 3), np.cos(t * 3)])
        a = nengo.Ensemble(N * 1, dimensions=1)
        b = nengo.Ensemble(N * 1, dimensions=1)
        c = nengo.Ensemble(N * 2, dimensions=2)
        d = nengo.Ensemble(N, neuron_type=nengo.Direct(), dimensions=3)

        nengo.Connection(input_node, a, function=lambda x: -x[0])
        nengo.Connection(input_node[:1], b, function=lambda x: -x)
        nengo.Connection(input_node, c, function=lambda x: -(x ** 2))
        nengo.Connection(
            input_node, d, function=lambda x: [-x[0], -(x[0] ** 2), -(x[1] ** 2)]
        )

        a_p = nengo.Probe(a, "decoded_output", synapse=0.01)
        b_p = nengo.Probe(b, "decoded_output", synapse=0.01)
        c_p = nengo.Probe(c, "decoded_output", synapse=0.01)
        d_p = nengo.Probe(d, "decoded_output", synapse=0.01)

    with Simulator(m) as sim:
        sim.run(2.0)
    t = sim.trange()

    plt.plot(t, sim.data[a_p])
    plt.plot(t, sim.data[b_p])
    plt.plot(t, sim.data[c_p])
    plt.plot(t, sim.data[d_p])
    plt.legend(
        [
            "-sin",
            "-sin",
            "-(sin ** 2)",
            "-(cos ** 2)",
            "-sin",
            "-(sin ** 2)",
            "-(cos ** 2)",
        ],
        loc="best",
        fontsize="small",
    )

    assert allclose(sim.data[a_p][-10:], sim.data[d_p][-10:][:, 0], atol=0.1, rtol=0.01)
    assert allclose(sim.data[b_p][-10:], sim.data[d_p][-10:][:, 0], atol=0.1, rtol=0.01)
    assert allclose(
        sim.data[c_p][-10:], sim.data[d_p][-10:][:, 1:3], atol=0.1, rtol=0.01
    )


def test_neurons_to_ensemble(Simulator, PositiveNeuronType, plt, seed):
    N = 20

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = PositiveNeuronType()
        a = nengo.Ensemble(N * 2, dimensions=2)
        b = nengo.Ensemble(N, dimensions=1)
        c = nengo.Ensemble(N, dimensions=N * 2)
        nengo.Connection(a.neurons, b, transform=-5 * np.ones((1, N * 2)))
        nengo.Connection(a.neurons, c)

        b_p = nengo.Probe(b, "decoded_output", synapse=0.01)
        c_p = nengo.Probe(c, "decoded_output", synapse=0.01)

    with Simulator(m) as sim:
        sim.run(0.1)
    t = sim.trange()

    plt.plot(t, sim.data[b_p], c="b")
    plt.plot(t, sim.data[c_p], c="k")
    plt.legend(
        ["Negative weights", "Neurons -> Ensemble dimensions"],
        loc="best",
        fontsize="small",
    )
    plt.xlim(right=t[-1])

    assert np.all(sim.data[b_p][-10:] < 0)


def test_neurons_to_node(Simulator, NonDirectNeuronType, plt, seed, allclose):
    N = 5

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = NonDirectNeuronType()
        a = nengo.Ensemble(N, dimensions=1, encoders=np.ones((N, 1)))
        out = nengo.Node(lambda t, x: x, size_in=N)
        nengo.Connection(nengo.Node(1), a)
        nengo.Connection(a.neurons, out, synapse=None)

        a_spikes = nengo.Probe(a.neurons, synapse=0.005)
        out_p = nengo.Probe(out, synapse=0.005)

    with Simulator(m) as sim:
        sim.run(0.1)
    t = sim.trange()

    plt.subplot(2, 1, 1)
    plt.title("Activity filtered with $\\tau$ = 0.005")
    plt.ylabel("Neural activity")
    plt.plot(t, sim.data[a_spikes])
    plt.xlim(right=t[-1])
    plt.subplot(2, 1, 2)
    plt.ylabel("Node activity")
    plt.plot(t, sim.data[out_p])
    plt.xlim(right=t[-1])

    assert allclose(sim.data[a_spikes], sim.data[out_p])


def test_neurons_to_neurons(Simulator, PositiveNeuronType, plt, seed, allclose):
    N1, N2 = 50, 80

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = PositiveNeuronType()
        a = nengo.Ensemble(N1, dimensions=1)
        b = nengo.Ensemble(N2, dimensions=1)
        inp = nengo.Node(output=1)
        nengo.Connection(inp, a)
        nengo.Connection(a.neurons, b.neurons, transform=-1 * np.ones((N2, N1)))

        inp_p = nengo.Probe(inp, "output")
        a_p = nengo.Probe(a, "decoded_output", synapse=0.1)
        b_p = nengo.Probe(b, "decoded_output", synapse=0.1)

    with Simulator(m) as sim:
        sim.run(0.6)
    t = sim.trange()

    plt.plot(t, sim.data[inp_p], label="Input")
    plt.plot(t, sim.data[a_p], label="A, represents input")
    plt.plot(t, sim.data[b_p], label="B, should be 0")
    plt.ylim(top=1.1)
    plt.xlim(right=t[-1])
    plt.legend(loc="best")

    assert allclose(sim.data[a_p][-10:], 1, atol=0.1, rtol=0.01)
    assert allclose(sim.data[b_p][-10:], 0, atol=0.1, rtol=0.01)


def test_function_and_transform(Simulator, plt, seed, allclose):
    """Test using both a function and a transform"""

    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(output=lambda t: np.sin(6 * t))
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(200, 2, radius=1.5)
        nengo.Connection(u, a)
        nengo.Connection(a, b, function=np.square, transform=[[1.0], [-1.0]])
        ap = nengo.Probe(a, synapse=0.03)
        bp = nengo.Probe(b, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.8)
    x = np.dot(sim.data[ap] ** 2, [[1.0, -1]]).T
    y = sim.data[bp].T

    t = sim.trange()
    plt.plot(t, x[0], "b:", label="a**2")
    plt.plot(t, x[1], "g:", label="-a**2")
    plt.plot(t, y[0], "b", label="b[0]")
    plt.plot(t, y[1], "g", label="b[1]")
    plt.legend(loc=0, prop={"size": 10})
    plt.xlim(right=t[-1])

    assert allclose(x[0], y[0], atol=0.1, rtol=0.01)
    assert allclose(x[1], y[1], atol=0.1, rtol=0.01)


def test_dist_transform(Simulator, seed, allclose):
    """Using a distribution to initialize transform."""

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Connection].transform = nengo.dists.Gaussian(0.5, 1)
        n = 300
        a = nengo.Node(output=[0] * n)
        b = nengo.Node(size_in=n + 1)
        c = nengo.Ensemble(n + 2, 10)
        d = nengo.Ensemble(n + 3, 11)

        # make a couple different types of connections to make sure that a
        # correctly sized transform is being generated
        conn1 = nengo.Connection(a, b)
        conn2 = nengo.Connection(b, c)
        conn3 = nengo.Connection(b, c.neurons)
        conn4 = nengo.Connection(b[:2], c[2])
        conn5 = nengo.Connection(c, d, solver=nengo.solvers.LstsqL2(weights=True))

    assert isinstance(conn1.transform.init, nengo.dists.Gaussian)

    with Simulator(net) as sim:
        pass

    w = sim.data[conn1].weights
    assert allclose(np.mean(w), 0.5, atol=0.01)
    assert allclose(np.std(w), 1, atol=0.01)
    assert w.shape == (n + 1, n)

    assert sim.data[conn2].weights.shape == (10, n + 1)
    assert sim.data[conn3].weights.shape == (n + 2, n + 1)
    assert sim.data[conn4].weights.shape == (1, 2)
    assert sim.data[conn5].weights.shape == (n + 3, n + 2)

    # make sure the seed works (gives us the same transform)
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Connection].transform = nengo.dists.Gaussian(0.5, 1)
        a = nengo.Node(output=[0] * n)
        b = nengo.Node(size_in=n + 1)
        conn = nengo.Connection(a, b)

    with Simulator(net) as sim:
        pass
    assert allclose(w, sim.data[conn].weights)


def test_weights(Simulator, AnyNeuronType, plt, seed, allclose):
    """Tests connections using a solver with weights"""
    n1, n2 = 100, 50

    def func(t):
        return [np.sin(4 * t), np.cos(12 * t)]

    transform = np.array([[0.6, -0.4]])

    m = nengo.Network(label="test_weights", seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = AnyNeuronType()
        u = nengo.Node(output=func)
        a = nengo.Ensemble(n1, dimensions=2, radius=1.4)
        b = nengo.Ensemble(n2, dimensions=1)
        bp = nengo.Probe(b)

        nengo.Connection(u, a)
        nengo.Connection(
            a, b, synapse=0.01, transform=transform, solver=LstsqL2(weights=True)
        )

    with Simulator(m) as sim:
        sim.run(1.0)

    t = sim.trange()
    x = np.array(func(t)).T
    y = np.dot(x, transform.T)
    z = nengo.Lowpass(0.01).filt(sim.data[bp], dt=sim.dt)
    assert signals_allclose(
        t, y, z, atol=0.15, buf=0.1, delay=0.025, plt=plt, allclose=allclose
    )


@pytest.mark.filterwarnings(
    "ignore:For connections from.*setting the solver has no effect",
    "ignore:For connections to.*setting `weights=True` on a solver has no effect",
)
def test_configure_weight_solver(Simulator, seed, plt, allclose):
    """Ensures that connections that don't use the weight solver ignore it"""
    n1, n2 = 100, 101
    function = lambda x: x ** 2

    with nengo.Network(seed=seed) as net:
        net.config[nengo.Connection].solver = nengo.solvers.LstsqL2(weights=True)

        u = nengo.Node(lambda t: np.sin(8 * t))
        a = nengo.Ensemble(n1, 1)
        b = nengo.Ensemble(n2, 1)
        v = nengo.Node(size_in=1)
        up = nengo.Probe(u, synapse=nengo.Alpha(0.01))
        vp = nengo.Probe(v, synapse=nengo.Alpha(0.01))

        nengo.Connection(u, a)
        ens_conn = nengo.Connection(a, b, function=function)
        nengo.Connection(b, v)

    with nengo.Simulator(net) as sim:
        sim.run(1.0)

    t = sim.trange()
    x = sim.data[up]
    y = function(x)
    z = sim.data[vp]
    assert sim.data[ens_conn].weights.shape == (n2, n1)
    assert signals_allclose(
        t, y, z, buf=0.01, delay=0.015, atol=0.05, rtol=0.05, plt=plt, allclose=allclose
    )


def test_vector(Simulator, AnyNeuronType, plt, seed, allclose):
    N1, N2 = 50, 50
    transform = [-1, 0.5]

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = AnyNeuronType()
        u = nengo.Node(output=[0.5, 0.5])
        a = nengo.Ensemble(N1, dimensions=2)
        b = nengo.Ensemble(N2, dimensions=2)
        nengo.Connection(u, a)
        nengo.Connection(a, b, transform=transform)

        up = nengo.Probe(u, "output")
        bp = nengo.Probe(b, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()
    x = sim.data[up]
    y = x * transform
    yhat = sim.data[bp]

    plt.plot(t, y, "--")
    plt.plot(t, yhat)

    assert allclose(y[-10:], yhat[-10:], atol=0.1, rtol=0.01)


def test_dimensionality_errors(NonDirectNeuronType, seed, rng):
    N = 10
    with nengo.Network(seed=seed) as m:
        m.config[nengo.Ensemble].neuron_type = NonDirectNeuronType()
        n01 = nengo.Node(output=[1])
        n02 = nengo.Node(output=[1, 1])
        n21 = nengo.Node(output=lambda t, x: [1], size_in=2)
        e1 = nengo.Ensemble(N, 1)
        e2 = nengo.Ensemble(N, 2)

        # these should work
        nengo.Connection(n01, e1)
        nengo.Connection(n02, e2)
        nengo.Connection(e2, n21)
        nengo.Connection(n21, e1)
        nengo.Connection(e1.neurons, n21, transform=rng.randn(2, N))
        nengo.Connection(e2, e1, function=lambda x: x[0])
        nengo.Connection(e2, e2, transform=np.ones(2))

        # these should not work
        with pytest.raises(
            ValidationError, match="not equal to connection output size"
        ):
            nengo.Connection(n02, e1)
        with pytest.raises(
            ValidationError, match="not equal to connection output size"
        ):
            nengo.Connection(e1, e2)
        with pytest.raises(ValidationError, match="Transform input size"):
            nengo.Connection(
                e2.neurons, e1, transform=Dense((1, N + 1), init=Choice([1.0]))
            )
        with pytest.raises(ValidationError, match="Transform output size"):
            nengo.Connection(
                e2.neurons, e1, transform=Dense((2, N), init=Choice([1.0]))
            )
        with pytest.raises(ValidationError, match="Function output size"):
            nengo.Connection(e2, e1, function=lambda x: x, transform=Dense((1, 1)))
        with pytest.raises(ValidationError, match="function.*must accept a single"):
            nengo.Connection(e2, e1, function=lambda: 0, transform=Dense((1, 1)))
        with pytest.raises(ValidationError, match="Function output size"):
            nengo.Connection(n21, e2, transform=Dense((2, 2)))
        with pytest.raises(ValidationError, match="Shape of initial value"):
            nengo.Connection(e2, e2, transform=np.ones((2, 2, 2)))
        with pytest.raises(ValidationError, match="Function output size"):
            nengo.Connection(e1, e2, transform=Dense((3, 3), init=np.ones(3)))

        # these should not work because of indexing mismatches
        with pytest.raises(ValidationError, match="Function output size"):
            nengo.Connection(n02[0], e2, transform=Dense((2, 2)))
        with pytest.raises(ValidationError, match="Transform output size"):
            nengo.Connection(n02, e2[0], transform=Dense((2, 2)))
        with pytest.raises(ValidationError, match="Function output size"):
            nengo.Connection(n02[1], e2[0], transform=Dense((2, 2)))
        with pytest.raises(ValidationError, match="Transform input size"):
            nengo.Connection(n02, e2[0], transform=Dense((2, 1), init=Choice([1.0])))
        with pytest.raises(ValidationError, match="Transform input size"):
            nengo.Connection(e2[0], e2, transform=Dense((1, 2), init=Choice([1.0])))

        # these should not work because of repeated indices
        dense22 = Dense((2, 2), init=np.ones((2, 2)))
        with pytest.raises(ValidationError, match="Input.*repeated indices"):
            nengo.Connection(n02[[0, 0]], e2, transform=dense22)
        with pytest.raises(ValidationError, match="Output.*repeated indices"):
            nengo.Connection(e2, e2[[1, 1]], transform=dense22)


def test_slicing(Simulator, AnyNeuronType, plt, seed, allclose):
    N = 300

    x = np.array([-1, -0.25, 1])

    s1a = slice(1, None, -1)
    s1b = [2, 0]
    T1 = [[-1, 0.5], [2, 0.25]]
    y1 = np.zeros(3)
    y1[s1b] = np.dot(T1, x[s1a])

    s2a = [0, 2]
    s2b = slice(0, 2)
    T2 = [[-0.5, 0.25], [0.5, 0.75]]
    y2 = np.zeros(3)
    y2[s2b] = np.dot(T2, x[s2a])

    s3a = [2, 0]
    s3b = np.asarray([0, 2])  # test slicing with numpy array
    T3 = [0.5, 0.75]
    y3 = np.zeros(3)
    y3[s3b] = np.dot(np.diag(T3), x[s3a])

    sas = [s1a, s2a, s3a]
    sbs = [s1b, s2b, s3b]
    Ts = [T1, T2, T3]
    ys = [y1, y2, y3]

    weight_solver = nengo.solvers.LstsqL2(weights=True)

    with nengo.Network(seed=seed) as m:
        m.config[nengo.Ensemble].neuron_type = AnyNeuronType()

        u = nengo.Node(output=x)
        a = nengo.Ensemble(N, dimensions=3, radius=1.7)
        nengo.Connection(u, a)

        probes = []
        weight_probes = []
        for sa, sb, T in zip(sas, sbs, Ts):
            b = nengo.Ensemble(N, dimensions=3, radius=1.7)
            nengo.Connection(a[sa], b[sb], transform=T)
            probes.append(nengo.Probe(b, synapse=0.03))

            # also test on weight solver
            b = nengo.Ensemble(N, dimensions=3, radius=1.7)
            nengo.Connection(a[sa], b[sb], transform=T, solver=weight_solver)
            weight_probes.append(nengo.Probe(b, synapse=0.03))

    with Simulator(m) as sim:
        sim.run(0.25)
    t = sim.trange()

    for i, [y, p] in enumerate(zip(ys, probes)):
        plt.subplot(len(ys), 1, i + 1)
        plt.plot(t, np.tile(y, (len(t), 1)), "--")
        plt.plot(t, sim.data[p])

    atol = 0.01 if AnyNeuronType is nengo.Direct else 0.1
    for i, [y, p, wp] in enumerate(zip(ys, probes, weight_probes)):
        assert allclose(y, sim.data[p][-20:], atol=atol), "Failed %d" % i
        assert allclose(y, sim.data[wp][-20:], atol=atol), "Weights %d" % i


def test_neuron_slicing(Simulator, plt, seed, rng, allclose):
    N = 6
    sa = slice(None, None, 2)
    sb = slice(None, None, -2)

    x = np.array([-1, -0.25, 1])
    with nengo.Network(seed=seed) as m:
        m.config[nengo.Ensemble].neuron_type = nengo.LIFRate()

        u = nengo.Node(output=x)
        a = nengo.Ensemble(N, dimensions=3, radius=1.7)
        b = nengo.Ensemble(N, dimensions=3, radius=1.7)
        nengo.Connection(u, a)

        c = nengo.Connection(a.neurons[sa], b.neurons[sb])
        c.transform = rng.normal(scale=1e-3, size=(c.size_out, c.size_in))

        ap = nengo.Probe(a.neurons, synapse=0.03)
        bp = nengo.Probe(b.neurons, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()

    x = sim.data[ap]
    y = np.zeros((len(t), b.n_neurons))
    y[:, sb] = np.dot(x[:, sa], c.transform.init.T)
    y = b.neuron_type.rates(y, sim.data[b].gain, sim.data[b].bias)

    plt.plot(t, y, "k--")
    plt.plot(t, sim.data[bp])
    assert allclose(y[-10:], sim.data[bp][-10:], atol=3.0, rtol=0.0)


def test_shortfilter(Simulator, AnyNeuronType):
    # Testing the case where the connection filter is < dt
    m = nengo.Network()
    with m:
        m.config[nengo.Ensemble].neuron_type = AnyNeuronType()
        a = nengo.Ensemble(n_neurons=10, dimensions=1)
        nengo.Connection(a, a, synapse=0)

        b = nengo.Ensemble(n_neurons=10, dimensions=1)
        nengo.Connection(a, b, synapse=0)
        nengo.Connection(b, a, synapse=0)

    with Simulator(m, dt=0.01):
        # This test passes if there are no cycles in the op graph
        pass

    # We will still get a cycle if the user explicitly sets the
    # filter to None
    with m:
        d = nengo.Ensemble(1, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(d, d, synapse=None)
    with pytest.raises(ValueError):
        Simulator(m, dt=0.01)


def test_zerofilter(Simulator, seed):
    # Testing the case where the connection filter is zero
    m = nengo.Network(seed=seed)
    with m:
        # Ensure no cycles in the op graph.
        a = nengo.Ensemble(1, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(a, a, synapse=0)

        # Ensure that spikes are not filtered
        b = nengo.Ensemble(
            3, dimensions=1, intercepts=[-0.9, -0.8, -0.7], neuron_type=nengo.LIF()
        )
        bp = nengo.Probe(b.neurons)

    with Simulator(m) as sim:
        sim.run(1.0)
    # assert that we have spikes (binary)
    assert np.unique(sim.data[bp]).size == 2


def test_function_output_size(Simulator, plt, seed, allclose):
    """Try a function that outputs both 0-d and 1-d arrays"""

    def bad_function(x):
        return x if x > 0 else 0

    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(output=lambda t: (t - 0.1) * 10)
        a = nengo.Ensemble(n_neurons=100, dimensions=1)
        b = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(u, a, synapse=None)
        nengo.Connection(a, b, synapse=None, function=bad_function)
        up = nengo.Probe(u)
        bp = nengo.Probe(b, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)
    t = sim.trange()
    x = nengo.Lowpass(0.03).filt(np.maximum(sim.data[up], 0), dt=sim.dt)
    y = sim.data[bp]

    plt.plot(t, x, "k")
    plt.plot(t, y)

    assert allclose(x, y, atol=0.1)


def test_slicing_function(Simulator, plt, seed, allclose):
    """Test using a pre-slice and a function"""
    N = 300
    f_in = lambda t: [np.cos(3 * t), np.sin(3 * t)]
    f_x = lambda x: [x, -(x ** 2)]

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=f_in)
        a = nengo.Ensemble(N, 2, radius=np.sqrt(2))
        b = nengo.Ensemble(N, 2, radius=np.sqrt(2))
        nengo.Connection(u, a)
        nengo.Connection(a[1], b, function=f_x)

        up = nengo.Probe(u, synapse=0.05)
        bp = nengo.Probe(b, synapse=0.05)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    v = sim.data[up]
    w = np.column_stack(f_x(v[:, 1]))
    y = sim.data[bp]

    plt.plot(t, y)
    plt.plot(t, w, ":")

    assert allclose(w, y, atol=0.1)


@pytest.mark.parametrize("negative_indices", (True, False))
def test_list_indexing(Simulator, plt, seed, negative_indices, allclose):
    with nengo.Network(seed=seed) as model:
        u = nengo.Node([-1, 1])
        a = nengo.Ensemble(50, dimensions=1)
        b = nengo.Ensemble(50, dimensions=1, radius=2.2)
        c = nengo.Ensemble(100, dimensions=2, radius=1.3)
        d = nengo.Ensemble(100, dimensions=2, radius=1.3)
        if negative_indices:
            nengo.Connection(u[[0, 1]], a[[0, -1]])
            nengo.Connection(u[[1, -1]], b[[0, -1]])
            nengo.Connection(u[[0, 1]], c[[0, 1]])
            nengo.Connection(u[[1, -1]], d[[0, 1]])
        else:
            nengo.Connection(u[[0, 1]], a[[0, 0]])
            nengo.Connection(u[[1, 1]], b[[0, 0]])
            nengo.Connection(u[[0, 1]], c[[0, 1]])
            nengo.Connection(u[[1, 1]], d[[0, 1]])

        a_probe = nengo.Probe(a, synapse=0.03)
        b_probe = nengo.Probe(b, synapse=0.03)
        c_probe = nengo.Probe(c, synapse=0.03)
        d_probe = nengo.Probe(d, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.4)

    t = sim.trange()
    a_data = sim.data[a_probe]
    b_data = sim.data[b_probe]
    c_data = sim.data[c_probe]
    d_data = sim.data[d_probe]

    line = plt.plot(t, a_data)
    plt.axhline(0, color=line[0].get_color())
    line = plt.plot(t, b_data)
    plt.axhline(2, color=line[0].get_color())
    line = plt.plot(t, c_data)
    plt.axhline(-1, color=line[0].get_color())
    line = plt.plot(t, d_data)
    plt.axhline(1, color=line[1].get_color())

    assert allclose(a_data[t > 0.3], [0], atol=0.2)
    assert allclose(b_data[t > 0.3], [2], atol=0.2)
    assert allclose(c_data[t > 0.3], [-1, 1], atol=0.2)
    assert allclose(d_data[t > 0.3], [1, 1], atol=0.2)


@pytest.mark.filterwarnings("ignore:boolean index did not match")
def test_boolean_indexing(Simulator, rng, plt, allclose):
    D = 10
    mu = np.arange(D) % 2 == 0
    mv = np.arange(D) % 2 == 1
    x = rng.uniform(-1, 1, size=D)
    y = np.zeros(D)
    y[mv] = x[mu]

    with nengo.Network() as model:
        u = nengo.Node(x)
        v = nengo.Node(size_in=D)
        nengo.Connection(u[mu], v[mv], synapse=None)
        v_probe = nengo.Probe(v)

    with Simulator(model) as sim:
        sim.run(0.01)

    plt.plot(sim.trange(), sim.data[v_probe])
    assert allclose(sim.data[v_probe][1:], y, atol=1e-5, rtol=1e-3)


def test_set_weight_solver():
    with nengo.Network() as net:
        # set default Gaussian transform to avoid transform errors
        net.config[nengo.Connection].transform = nengo.dists.Gaussian(0, 1)

        ens = nengo.Ensemble(10, 2)
        node = nengo.Node(lambda t, x: x, size_in=2, size_out=2)

        nengo.Connection(ens, ens, solver=LstsqL2(weights=True))

        with pytest.warns(
            UserWarning, match="For connections from.*setting the solver has no effect"
        ):
            nengo.Connection(node, ens, solver=LstsqL2(weights=True))

        with pytest.warns(
            UserWarning, match="For connections to.*setting `weights=True`.*no effect"
        ):
            nengo.Connection(ens, node, solver=LstsqL2(weights=True))


def test_set_learning_rule():
    with nengo.Network():
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        nengo.Connection(a, b, learning_rule_type=nengo.PES())
        nengo.Connection(
            a, b, learning_rule_type=nengo.PES(), solver=LstsqL2(weights=True)
        )
        nengo.Connection(
            a.neurons, b.neurons, learning_rule_type=nengo.PES(), transform=1
        )
        nengo.Connection(
            a.neurons,
            b.neurons,
            learning_rule_type=nengo.Oja(),
            transform=np.ones((10, 10)),
        )

        n = nengo.Node(output=lambda t, x: t * x, size_in=2)
        with pytest.raises(
            ValidationError, match="'pre' must be of type 'Ensemble'.*PES"
        ):
            nengo.Connection(n, a, learning_rule_type=nengo.PES())


def test_set_function(Simulator):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        c = nengo.Ensemble(10, 1)
        d = nengo.Node(size_in=1)

        # Function only OK from node or ensemble
        with pytest.raises(
            ValidationError, match="function can only be set.*Ensemble or Node"
        ):
            nengo.Connection(a.neurons, b, function=lambda x: x)

        # Function and transform must match up
        with pytest.raises(
            ValidationError,
            match="Shape of initial value.*does not match expected shape",
        ):
            nengo.Connection(a, b, function=lambda x: x[0] * x[1], transform=np.eye(2))

        # No functions allowed on passthrough nodes
        with pytest.raises(
            ValidationError, match="Cannot apply functions to passthrough nodes"
        ):
            nengo.Connection(d, c, function=lambda x: [1])

        # These initial functions have correct dimensionality
        conn_2d = nengo.Connection(a, b)
        conn_1d = nengo.Connection(b, c, function=lambda x: x[0] * x[1])

    with Simulator(model):
        pass  # Builds fine

    with model:
        # Can change to another function with correct dimensionality
        conn_2d.function = lambda x: x ** 2
        conn_1d.function = lambda x: x[0] + x[1]

    with Simulator(model):
        pass  # Builds fine


def test_set_eval_points(Simulator):
    with nengo.Network() as model:
        a = nengo.Ensemble(
            10,
            2,
            encoders=nengo.dists.Choice([[1, 1]]),
            intercepts=nengo.dists.Uniform(-1, -0.1),
        )
        b = nengo.Ensemble(10, 2)
        # ConnEvalPoints parameter checks that pre is an ensemble
        nengo.Connection(a, b, eval_points=[[0, 0], [0.5, 1]])
        nengo.Connection(a, b, eval_points=nengo.dists.Uniform(0, 1))
        with pytest.raises(
            ValidationError,
            match="eval_points are only valid on connections from ensembles",
        ):
            nengo.Connection(a.neurons, b, eval_points=[[0, 0], [0.5, 1]])

    with Simulator(model):
        pass  # Builds fine


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("scale", [False, True])
def test_eval_points_scaling(Simulator, sample, radius, seed, rng, scale):
    eval_points = UniformHypersphere()
    if sample:
        eval_points = eval_points.sample(500, 3, rng=rng)

    model = nengo.Network(seed=seed)
    with model:
        a = nengo.Ensemble(
            1,
            3,
            encoders=np.ones((1, 3)),
            intercepts=nengo.dists.Choice([-1]),
            radius=radius,
        )
        b = nengo.Ensemble(1, 3)
        con = nengo.Connection(a, b, eval_points=eval_points, scale_eval_points=scale)

    with Simulator(model) as sim:
        dists = npext.norm(sim.data[con].eval_points, axis=1)
    limit = radius if scale else 1.0
    assert np.all(dists <= limit)
    assert np.any(dists >= 0.9 * limit)


def test_solverparam():
    """SolverParam must be a solver."""

    class Test:
        sp = ConnectionSolverParam("sp", default=None)

    inst = Test()
    assert inst.sp is None
    inst.sp = LstsqL2()
    assert isinstance(inst.sp, LstsqL2)
    assert not inst.sp.weights
    # Non-solver not OK
    with pytest.raises(ValidationError):
        inst.sp = "a"


def test_prepost_errors(Simulator):
    with nengo.Network():
        x = nengo.Ensemble(10, 1)
        y = nengo.Ensemble(10, 1)
        c = nengo.Connection(x, y)
        with pytest.raises(ValueError):
            nengo.Connection(3, x)  # from non-NengoObject
        with pytest.raises(ValueError):
            nengo.Connection(x, 3)  # to non-NengoObject
        with pytest.raises(ValueError):
            nengo.Connection(c, x)  # from connection
        with pytest.raises(ValueError):
            nengo.Connection(x, c)  # to connection

    # --- non-existent objects
    with nengo.Network():
        a = nengo.Ensemble(10, 1)

    # non-existent pre
    with nengo.Network() as model1:
        e1 = nengo.Ensemble(10, 1)
        nengo.Connection(a, e1)
    with pytest.raises(ValueError):
        Simulator(model1)

    # non-existent post
    with nengo.Network() as model2:
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e2, a)
    with pytest.raises(ValueError):
        Simulator(model2)

    # probe non-existent object
    with nengo.Network() as model3:
        nengo.Probe(a)
    with pytest.raises(ValueError):
        Simulator(model3)


def test_directneurons(NonDirectNeuronType):
    with nengo.Network():
        a = nengo.Ensemble(1, 1, neuron_type=NonDirectNeuronType())
        b = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # cannot connect to or from direct neurons
        with pytest.raises(ValidationError, match="must have size_in > 0"):
            nengo.Connection(a, b.neurons)
        with pytest.raises(ValidationError, match="must have size_out > 0"):
            nengo.Connection(b.neurons, a)


def test_decoder_probe(Simulator):
    """Ensures we can only probe decoders in connections from ensembles."""
    with nengo.Network() as net:
        pre = nengo.Ensemble(10, 10)
        post = nengo.Ensemble(10, 10)
        c_ens = nengo.Connection(pre, post)
        c_ens_neurons = nengo.Connection(pre, post.neurons)
        nengo.Probe(c_ens, "weights")
        nengo.Probe(c_ens_neurons, "weights")

    with Simulator(net) as sim:
        assert sim


def test_transform_probe(Simulator, allclose):
    """Ensures we can always probe transform in connections."""
    with nengo.Network() as net:
        pre = nengo.Ensemble(10, 10)
        post = nengo.Ensemble(10, 10)

        c_ens = nengo.Connection(
            pre, post, solver=nengo.solvers.NoSolver(np.ones((10, 10)))
        )
        c_ens_neurons = nengo.Connection(
            pre, post.neurons, solver=nengo.solvers.NoSolver(np.ones((10, 10)))
        )
        c_neurons_ens = nengo.Connection(pre.neurons, post, transform=1)
        c_neurons = nengo.Connection(pre.neurons, post.neurons, transform=1)

        p_neurons_ens = nengo.Probe(c_neurons_ens, "weights")
        p_neurons = nengo.Probe(c_neurons, "weights")
        p_ens = nengo.Probe(c_ens, "weights")
        p_ens_neurons = nengo.Probe(c_ens_neurons, "weights")

    with Simulator(net) as sim:
        sim.step()
        assert allclose(sim.data[p_neurons_ens], 1)
        assert allclose(sim.data[p_neurons], 1)
        assert allclose(sim.data[p_ens], 1)
        assert allclose(sim.data[p_ens_neurons], 1)

    with net:
        c_neurons_ens_none = nengo.Connection(pre.neurons, post, transform=None)
        nengo.Probe(c_neurons_ens_none, "weights")

    with pytest.raises(BuildError, match="cannot be probed"):
        Simulator(net)


def test_connectionlearningruletypeparam():
    with nengo.Network():
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(11, 1)

        with pytest.raises(
            ValidationError, match="can only be applied on connections to neurons"
        ):
            nengo.Connection(a, b, learning_rule_type=nengo.BCM())

        with pytest.raises(ValidationError, match="does not match expected shape"):
            nengo.Connection(
                a, b, transform=np.ones((10, 11)), learning_rule_type=nengo.BCM()
            )


def test_function_with_no_name(Simulator):
    def add(x, val):
        return x + val

    model = nengo.Network()

    with model:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(11, 1)
        nengo.Connection(a, b, function=partial(add, val=2))

    with Simulator(model) as sim:
        assert sim


def test_function_points(Simulator, seed, rng, plt, allclose):
    x = rng.uniform(-1, 1, size=(1000, 1))
    y = -x

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(nengo.processes.WhiteSignal(1.0, high=9, rms=0.3))
        a = nengo.Ensemble(100, 1)
        v = nengo.Node(size_in=1)
        nengo.Connection(u, a, synapse=None)
        nengo.Connection(a, v, eval_points=x, function=y)

        up = nengo.Probe(u, synapse=nengo.Alpha(0.01))
        vp = nengo.Probe(v, synapse=nengo.Alpha(0.01))

    with Simulator(model, seed=seed) as sim:
        sim.run(1.0)

    assert signals_allclose(
        sim.trange(),
        -sim.data[up],
        sim.data[vp],
        buf=0.01,
        delay=0.005,
        atol=5e-2,
        rtol=3e-2,
        plt=plt,
        allclose=allclose,
    )


def test_connectionfunctionparam_array(Simulator, seed):
    points_1d = np.zeros((100, 1))
    points_2d = np.zeros((100, 2))
    points_v = np.zeros((100,))
    points_50 = np.zeros((50, 1))

    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)

        # eval points not set raises error
        with pytest.raises(ValidationError, match="'eval_points' must also be set"):
            nengo.Connection(a, b, function=points_1d)

        # wrong ndims raises error
        with pytest.raises(ValidationError, match="array must be 2D"):
            nengo.Connection(a, b, eval_points=points_1d, function=points_v)

        # wrong number of points raises error
        with pytest.raises(
            ValidationError,
            match="Number of eval.*must match number of function points",
        ):
            nengo.Connection(a, b, eval_points=points_50, function=points_1d)

        # wrong output dims raises error
        with pytest.raises(
            ValidationError,
            match="Transform output size.*not equal to connection output",
        ):
            nengo.Connection(a, b, eval_points=points_1d, function=points_2d)

        nengo.Connection(a, b, eval_points=points_1d, function=points_1d)
        nengo.Connection(
            a, b, eval_points=points_1d, function=points_2d, transform=np.ones((1, 2))
        )

    with Simulator(model):
        pass


def test_function_names():
    def my_func(x):
        return x

    class MyFunc:
        def __call__(self, x):
            return x

    with nengo.Network():
        a = nengo.Ensemble(10, 1)
        b = nengo.Node(size_in=1)

        func_conn = nengo.Connection(a, b, function=my_func)
        assert str(func_conn).endswith("computing 'my_func'>")

        class_conn = nengo.Connection(a, b, function=MyFunc())
        assert str(class_conn).endswith("computing 'MyFunc'>")

        array_conn = nengo.Connection(
            a, b, eval_points=np.zeros((10, 1)), function=np.zeros((10, 1))
        )
        assert str(array_conn).endswith("computing 'ndarray'>")


@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.filterwarnings("ignore:Non-finite values detected")
def test_zero_activities_error(Simulator):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, 1)
        a.gain = np.zeros(10)
        a.bias = np.zeros(10)
        nengo.Connection(a, nengo.Node(size_in=1))

    with pytest.raises(BuildError, match="'activities' matrix is all zero"):
        with Simulator(model):
            pass


def test_function_returns_none_error(Simulator):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, 1)
        nengo.Connection(
            a,
            nengo.Node(size_in=1),
            eval_points=[[0], [1]],
            function=lambda x: x if x < 0.5 else None,
        )

    with pytest.raises(BuildError):
        with Simulator(model):
            pass


def test_transform_none():
    with nengo.Network():
        a = nengo.Node([0])
        b = nengo.Node(size_in=1)
        conn = nengo.Connection(a, b, transform=None)
        assert isinstance(conn.transform, NoTransform)


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_neuron_advanced_indexing(Simulator):
    with nengo.Network() as net:
        node = nengo.Node([1, 1, 1])
        ens = nengo.Ensemble(
            5, 1, bias=nengo.dists.Choice([0]), gain=nengo.dists.Choice([1])
        )
        nengo.Connection(node, ens.neurons[[0, 0, 2]], synapse=None)
        p = nengo.Probe(ens.neurons, "input")

    with Simulator(net) as sim:
        sim.run(0.001)
        assert np.allclose(sim.data[p], [2, 0, 1, 0, 0])


def test_learning_rule_equality():
    with nengo.Network():
        ens = nengo.Ensemble(10, 1)
        lr = nengo.PES()
        conn0 = nengo.Connection(ens, ens, learning_rule_type=lr)
        assert conn0.learning_rule == conn0.learning_rule
        assert conn0.learning_rule != lr

        conn1 = nengo.Connection(ens, ens, learning_rule_type=[lr, nengo.PES()])
        assert conn0.learning_rule[0] != conn1.learning_rule
        assert conn1.learning_rule_type[0] == conn0.learning_rule_type
        assert conn1.learning_rule[0] == conn1.learning_rule[1]


def test_learning_transform_shape_error(Simulator):
    with nengo.Network() as net:
        a = nengo.Ensemble(10, dimensions=2)
        b = nengo.Ensemble(10, dimensions=2)
        nengo.Connection(
            a.neurons, b.neurons, transform=1, learning_rule_type=nengo.BCM()
        )

    with pytest.raises(
        BuildError, match="'transform' must be a 2-dimensional array for learning"
    ):
        with Simulator(net):
            pass


def test_is_decoded_deprecation():
    with pytest.warns(DeprecationWarning, match="is_decoded is deprecated"):
        with nengo.Network():
            assert nengo.Connection(nengo.Node(0), nengo.Node(size_in=1)).is_decoded
