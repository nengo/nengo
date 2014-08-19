import logging

import numpy as np
import pytest

import nengo
from nengo.connection import ConnectionSolverParam
from nengo.solvers import LstsqL2
from nengo.utils.functions import piecewise
from nengo.utils.numpy import filtfilt
from nengo.utils.testing import Plotter, allclose

logger = logging.getLogger(__name__)


def test_args(nl):
    N = 10
    d1, d2 = 3, 2

    with nengo.Network(label='test_args') as model:
        model.config[nengo.Ensemble].neuron_type = nl()
        A = nengo.Ensemble(N, dimensions=d1)
        B = nengo.Ensemble(N, dimensions=d2)
        nengo.Connection(
            A, B,
            eval_points=np.random.normal(size=(500, d1)),
            synapse=0.01,
            function=np.sin,
            transform=np.random.normal(size=(d2, d1)))


def test_node_to_neurons(Simulator, nl_nodirect):
    name = 'node_to_neurons'
    N = 30

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        a = nengo.Ensemble(N, dimensions=1)
        inn = nengo.Node(output=np.sin)
        inh = nengo.Node(piecewise({0: 0, 2.5: 1}))
        nengo.Connection(inn, a)
        nengo.Connection(inh, a.neurons, transform=[[-2.5]]*N)

        inn_p = nengo.Probe(inn, 'output')
        a_p = nengo.Probe(a, 'decoded_output', synapse=0.1)
        inh_p = nengo.Probe(inh, 'output')

    sim = Simulator(m)
    sim.run(5.0)
    t = sim.trange()
    ideal = np.sin(t)
    ideal[t >= 2.5] = 0

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.plot(t, sim.data[inn_p], label='Input')
        plt.plot(t, sim.data[a_p], label='Neuron approx, synapse=0.1')
        plt.plot(t, sim.data[inh_p], label='Inhib signal')
        plt.plot(t, ideal, label='Ideal output')
        plt.legend(loc=0, prop={'size': 10})
        plt.savefig('test_connection.test_' + name + '.pdf')
        plt.close()

    assert np.allclose(sim.data[a_p][-10:], 0, atol=.1, rtol=.01)


def test_ensemble_to_neurons(Simulator, nl_nodirect):
    name = 'ensemble_to_neurons'
    N = 30

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        a = nengo.Ensemble(N, dimensions=1)
        b = nengo.Ensemble(N, dimensions=1)
        inn = nengo.Node(output=np.sin)
        inh = nengo.Node(piecewise({0: 0, 2.5: 1}))
        nengo.Connection(inn, a)
        nengo.Connection(inh, b)
        nengo.Connection(b, a.neurons, transform=[[-2.5]]*N)

        inn_p = nengo.Probe(inn, 'output')
        a_p = nengo.Probe(a, 'decoded_output', synapse=0.1)
        b_p = nengo.Probe(b, 'decoded_output', synapse=0.1)
        inh_p = nengo.Probe(inh, 'output')

    sim = Simulator(m)
    sim.run(5.0)
    t = sim.trange()
    ideal = np.sin(t)
    ideal[t >= 2.5] = 0

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.plot(t, sim.data[inn_p], label='Input')
        plt.plot(t, sim.data[a_p], label='Neuron approx, pstc=0.1')
        plt.plot(
            t, sim.data[b_p], label='Neuron approx of inhib sig, pstc=0.1')
        plt.plot(t, sim.data[inh_p], label='Inhib signal')
        plt.plot(t, ideal, label='Ideal output')
        plt.legend(loc=0, prop={'size': 10})
        plt.savefig('test_connection.test_' + name + '.pdf')
        plt.close()

    assert np.allclose(sim.data[a_p][-10:], 0, atol=.1, rtol=.01)
    assert np.allclose(sim.data[b_p][-10:], 1, atol=.1, rtol=.01)


def test_node_to_ensemble(Simulator, nl_nodirect):
    name = 'node_to_ensemble'
    N = 50

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        input_node = nengo.Node(output=lambda t: [np.sin(t), np.cos(t)])
        a = nengo.Ensemble(N * 1, dimensions=1)
        b = nengo.Ensemble(N * 2, dimensions=2)
        c = nengo.Ensemble(N, neuron_type=nengo.Direct(), dimensions=3)

        nengo.Connection(input_node, a, function=lambda x: -x[0])
        nengo.Connection(input_node, b, function=lambda x: -(x**2))
        nengo.Connection(input_node, c,
                         function=lambda x: [-x[0], -(x[0]**2), -(x[1]**2)])

        a_p = nengo.Probe(a, 'decoded_output', synapse=0.01)
        b_p = nengo.Probe(b, 'decoded_output', synapse=0.01)
        c_p = nengo.Probe(c, 'decoded_output', synapse=0.01)

    sim = Simulator(m)
    sim.run(5.0)
    t = sim.trange()

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.plot(t, sim.data[a_p], label='A')
        plt.plot(t, sim.data[b_p], label='B')
        plt.plot(t, sim.data[c_p], label='C')
        plt.savefig('test_connection.test_%s.pdf' % name)
        plt.close()

    assert np.allclose(sim.data[a_p][-10:], sim.data[c_p][-10:][:, 0],
                       atol=0.1, rtol=0.01)
    assert np.allclose(sim.data[b_p][-10:], sim.data[c_p][-10:][:, 1:3],
                       atol=0.1, rtol=0.01)


def test_neurons_to_ensemble(Simulator, nl_nodirect):
    name = 'neurons_to_ensemble'
    N = 20

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        a = nengo.Ensemble(N * 2, dimensions=2)
        b = nengo.Ensemble(N * 3, dimensions=3)
        c = nengo.Ensemble(N, dimensions=N*2)
        nengo.Connection(a.neurons, b, transform=-10 * np.ones((3, N*2)))
        nengo.Connection(a.neurons, c)

        a_p = nengo.Probe(a, 'decoded_output', synapse=0.01)
        b_p = nengo.Probe(b, 'decoded_output', synapse=0.01)
        c_p = nengo.Probe(c, 'decoded_output', synapse=0.01)

    sim = Simulator(m)
    sim.run(5.0)
    t = sim.trange()

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.plot(t, sim.data[a_p], label='A')
        plt.plot(t, sim.data[b_p], label='B')
        plt.plot(t, sim.data[c_p], label='C')
        plt.savefig('test_connection.test_' + name + '.pdf')
        plt.close()

    assert np.all(sim.data[b_p][-10:] < 0)


def test_neurons_to_node(Simulator, nl_nodirect):
    name = 'neurons_to_node'
    N = 30

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        a = nengo.Ensemble(N, dimensions=1)
        out = nengo.Node(lambda t, x: x, size_in=N)
        nengo.Connection(a.neurons, out, synapse=None)

        a_spikes = nengo.Probe(a, 'spikes')
        out_p = nengo.Probe(out, 'output')

    sim = Simulator(m)
    sim.run(0.6)
    t = sim.trange()

    with Plotter(Simulator, nl_nodirect) as plt:
        ax = plt.subplot(111)
        try:
            from nengo.matplotlib import rasterplot
            rasterplot(t, sim.data[a_spikes], ax=ax)
            rasterplot(t, sim.data[out_p], ax=ax)
        except ImportError:
            pass
        plt.savefig('test_connection.test_' + name + '.pdf')
        plt.close()

    assert np.allclose(sim.data[a_spikes][:-1], sim.data[out_p][1:])


def test_neurons_to_neurons(Simulator, nl_nodirect):
    name = 'neurons_to_neurons'
    N1, N2 = 30, 50

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        a = nengo.Ensemble(N1, dimensions=1)
        b = nengo.Ensemble(N2, dimensions=1)
        inp = nengo.Node(output=1)
        nengo.Connection(inp, a)
        nengo.Connection(
            a.neurons, b.neurons, transform=-1 * np.ones((N2, N1)))

        inp_p = nengo.Probe(inp, 'output')
        a_p = nengo.Probe(a, 'decoded_output', synapse=0.1)
        b_p = nengo.Probe(b, 'decoded_output', synapse=0.1)

    sim = Simulator(m)
    sim.run(5.0)
    t = sim.trange()

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.plot(t, sim.data[inp_p], label='Input')
        plt.plot(t, sim.data[a_p], label='A, represents input')
        plt.plot(t, sim.data[b_p], label='B, should be 0')
        plt.legend(loc=0, prop={'size': 10})
        plt.savefig('test_connection.test_' + name + '.pdf')
        plt.close()

    assert np.allclose(sim.data[a_p][-10:], 1, atol=.1, rtol=.01)
    assert np.allclose(sim.data[b_p][-10:], 0, atol=.1, rtol=.01)


def test_function_and_transform(Simulator, nl):
    """Test using both a function and a transform"""

    model = nengo.Network(seed=742)
    with model:
        u = nengo.Node(output=lambda t: np.sin(6 * t))
        a = nengo.Ensemble(100, 1, neuron_type=nl())
        b = nengo.Ensemble(200, 2, neuron_type=nl(), radius=1.5)
        nengo.Connection(u, a)
        nengo.Connection(a, b, function=np.square, transform=[[1.0], [-1.0]])
        ap = nengo.Probe(a, synapse=0.03)
        bp = nengo.Probe(b, synapse=0.03)

    sim = nengo.Simulator(model)
    sim.run(1.0)
    x0, x1 = np.dot(sim.data[ap]**2, [[1., -1]]).T
    y0, y1 = sim.data[bp].T

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, x0, 'b:', label='a**2')
        plt.plot(t, x1, 'g:', label='-a**2')
        plt.plot(t, y0, 'b', label='b[0]')
        plt.plot(t, y1, 'g', label='b[1]')
        plt.legend(loc=0, prop={'size': 10})
        plt.savefig('test_connection.test_function_and_transform.pdf')
        plt.close()

    assert np.allclose(x0, y0, atol=.1, rtol=.01)
    assert np.allclose(x1, y1, atol=.1, rtol=.01)


def test_weights(Simulator, nl):
    name = 'test_weights'
    n1, n2 = 100, 50

    def func(t):
        return np.array([np.sin(4 * t), np.cos(12 * t)])

    transform = np.array([[0.6, -0.4]])

    m = nengo.Network(label=name, seed=3902)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl()
        u = nengo.Node(output=func)
        a = nengo.Ensemble(n1, dimensions=2, radius=1.5)
        b = nengo.Ensemble(n2, dimensions=1)
        bp = nengo.Probe(b)

        nengo.Connection(u, a)
        nengo.Connection(a, b, transform=transform,
                         solver=LstsqL2(weights=True))

    sim = Simulator(m)
    sim.run(2.)

    t = sim.trange()
    x = func(t).T
    y = np.dot(x, transform.T)
    z = filtfilt(sim.data[bp], 10, axis=0)
    assert allclose(t, y.flatten(), z.flatten(),
                    plotter=Plotter(Simulator, nl),
                    filename='test_connection.' + name + '.pdf',
                    atol=0.1, rtol=0, buf=100, delay=10)


def test_vector(Simulator, nl):
    name = 'vector'
    N1, N2 = 50, 50
    transform = [-1, 0.5]

    m = nengo.Network(label=name, seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl()
        u = nengo.Node(output=[0.5, 0.5])
        a = nengo.Ensemble(N1, dimensions=2)
        b = nengo.Ensemble(N2, dimensions=2)
        nengo.Connection(u, a)
        nengo.Connection(a, b, transform=transform)

        up = nengo.Probe(u, 'output')
        bp = nengo.Probe(b, synapse=0.03)

    sim = Simulator(m)
    sim.run(1.0)
    t = sim.trange()
    x = sim.data[up]
    y = x * transform
    yhat = sim.data[bp]

    with Plotter(Simulator, nl) as plt:
        plt.plot(t, y, '--')
        plt.plot(t, yhat)
        plt.savefig('test_connection.test_' + name + '.pdf')
        plt.close()

    assert np.allclose(y[-10:], yhat[-10:], atol=.1, rtol=.01)


def test_dimensionality_errors(nl_nodirect):
    N = 10
    with nengo.Network(label="test_dimensionality_error") as m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
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
        nengo.Connection(e1.neurons, n21, transform=np.random.randn(2, N))
        nengo.Connection(e2, e1, function=lambda x: x[0])
        nengo.Connection(e2, e2, transform=np.ones(2))

        # these should not work
        with pytest.raises(ValueError):
            nengo.Connection(n02, e1)
        with pytest.raises(ValueError):
            nengo.Connection(e1, e2)
        with pytest.raises(ValueError):
            nengo.Connection(e2.neurons, e1, transform=np.random.randn(1, N+1))
        with pytest.raises(ValueError):
            nengo.Connection(e2.neurons, e1, transform=np.random.randn(2, N))
        with pytest.raises(ValueError):
            nengo.Connection(e2, e1, function=lambda x: x, transform=[[1]])
        with pytest.raises(TypeError):
            nengo.Connection(e2, e1, function=lambda: 0, transform=[[1]])
        with pytest.raises(ValueError):
            nengo.Connection(n21, e2, transform=np.ones((2, 2)))
        with pytest.raises(ValueError):
            nengo.Connection(e2, e2, transform=np.ones((2, 2, 2)))
        with pytest.raises(ValueError):
            nengo.Connection(e2, e2, transform=np.ones(3))

        # these should not work because of indexing mismatches
        with pytest.raises(ValueError):
            nengo.Connection(n02[0], e2)
        with pytest.raises(ValueError):
            nengo.Connection(n02, e2[0])
        with pytest.raises(ValueError):
            nengo.Connection(n02[1], e2[0], transform=[[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            nengo.Connection(n02, e2[0], transform=[[1], [2]])
        with pytest.raises(ValueError):
            nengo.Connection(e2[0], e2, transform=[[1, 2]])


def test_slicing(Simulator, nl):
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
    s3b = [0, 2]
    T3 = [0.5, 0.75]
    y3 = np.zeros(3)
    y3[s3b] = np.dot(np.diag(T3), x[s3a])

    sas = [s1a, s2a, s3a]
    sbs = [s1b, s2b, s3b]
    Ts = [T1, T2, T3]
    ys = [y1, y2, y3]

    with nengo.Network(seed=932) as m:
        m.config[nengo.Ensemble].neuron_type = nl()

        u = nengo.Node(output=x)
        a = nengo.Ensemble(N, dimensions=3, radius=1.7)
        nengo.Connection(u, a)

        probes = []
        for sa, sb, T in zip(sas, sbs, Ts):
            b = nengo.Ensemble(N, dimensions=3, radius=1.7)
            nengo.Connection(a[sa], b[sb], transform=T)
            probes.append(nengo.Probe(b, synapse=0.03))

    sim = nengo.Simulator(m)
    sim.run(0.2)
    t = sim.trange()

    with Plotter(Simulator, nl) as plt:
        for i, [y, p] in enumerate(zip(ys, probes)):
            plt.subplot(len(ys), 1, i)
            plt.plot(t, np.tile(y, (len(t), 1)), '--')
            plt.plot(t, sim.data[p])
        plt.savefig('test_connection.test_slicing.pdf')
        plt.close()

    atol = 0.01 if nl is nengo.Direct else 0.1
    for i, [y, p] in enumerate(zip(ys, probes)):
        assert np.allclose(y, sim.data[p][-20:], atol=atol), "Failed %d" % i


def test_shortfilter(Simulator, nl):
    # Testing the case where the connection filter is < dt
    m = nengo.Network()
    with m:
        m.config[nengo.Ensemble].neuron_type = nl()
        a = nengo.Ensemble(n_neurons=10, dimensions=1)
        nengo.Connection(a, a)

        b = nengo.Ensemble(n_neurons=10, dimensions=1)
        nengo.Connection(a, b)
        nengo.Connection(b, a)

    Simulator(m, dt=.01)
    # This test passes if there are no cycles in the op graph

    # We will still get a cycle if the user explicitly sets the
    # filter to None
    with m:
        d = nengo.Ensemble(1, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(d, d, synapse=None)
    with pytest.raises(ValueError):
        Simulator(m, dt=.01)


def test_zerofilter(Simulator):
    # Testing the case where the connection filter is zero
    m = nengo.Network(seed=8)
    with m:
        # Ensure no cycles in the op graph.
        a = nengo.Ensemble(1, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(a, a, synapse=0)

        # Ensure that spikes are not filtered
        b = nengo.Ensemble(3, dimensions=1, intercepts=[-.9, -.8, -.7],
                           neuron_type=nengo.LIF())
        bp = nengo.Probe(b, "neuron_output", synapse=0)

    sim = Simulator(m)
    sim.run(1.)
    # assert that we have spikes (binary)
    assert np.unique(sim.data[bp]).size == 2


def test_function_output_size(Simulator, nl_nodirect):
    """Try a function that outputs both 0-d and 1-d arrays"""
    def bad_function(x):
        return x if x > 0 else 0

    model = nengo.Network(seed=9)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=lambda t: t - 1)
        a = nengo.Ensemble(n_neurons=100, dimensions=1)
        b = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(u, a)
        nengo.Connection(a, b, function=bad_function)
        up = nengo.Probe(u, synapse=None)
        bp = nengo.Probe(b, synapse=0.03)

    sim = Simulator(model)
    sim.run(2.)
    t = sim.trange()
    x = nengo.utils.numpy.filt(sim.data[up].clip(0, np.inf), 0.03 / sim.dt)
    y = sim.data[bp]

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.plot(t, x, 'k')
        plt.plot(t, y)
        plt.savefig('test_connection.test_function_output_size.pdf')
        plt.close()

    assert np.allclose(x, y, atol=0.1)


def test_slicing_function(Simulator, nl):
    """Test using a pre-slice and a function"""
    N = 300
    f_in = lambda t: [np.cos(3*t), np.sin(3*t)]
    f_x = lambda x: [x, -x**2]

    with nengo.Network(seed=99) as model:
        model.config[nengo.Ensemble].neuron_type = nl()
        u = nengo.Node(output=f_in)
        a = nengo.Ensemble(N, 2, radius=1.5)
        b = nengo.Ensemble(N, 2, radius=1.5)
        nengo.Connection(u, a)
        nengo.Connection(a[1], b, function=f_x)

        up = nengo.Probe(u, synapse=0.03)
        bp = nengo.Probe(b, synapse=0.03)

    sim = Simulator(model)
    sim.run(1.)

    t = sim.trange()
    v = sim.data[up]
    w = np.column_stack(f_x(v[:, 1]))
    y = sim.data[bp]

    with Plotter(Simulator, nl) as plt:
        plt.plot(t, y)
        plt.plot(t, w, ':')
        plt.savefig('test_connection.test_slicing_function.pdf')
        plt.close()

    assert np.allclose(w, y, atol=0.1, rtol=0.0)


def test_set_weight_solver():
    with nengo.Network():
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        nengo.Connection(a, b, solver=LstsqL2(weights=True))
        with pytest.raises(ValueError):
            nengo.Connection(a.neurons, b, solver=LstsqL2(weights=True))
        with pytest.raises(ValueError):
            nengo.Connection(a, b.neurons, solver=LstsqL2(weights=True))
        with pytest.raises(ValueError):
            nengo.Connection(a.neurons, b.neurons,
                             solver=LstsqL2(weights=True))


def test_set_learning_rule():
    with nengo.Network():
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        err = nengo.Connection(a, b)
        n = nengo.Node(output=lambda t, x: t * x, size_in=2)
        nengo.Connection(a, b, learning_rule=nengo.PES(err))
        nengo.Connection(a, b, learning_rule=nengo.PES(err),
                         solver=LstsqL2(weights=True))
        nengo.Connection(a.neurons, b.neurons, learning_rule=nengo.PES(err))
        nengo.Connection(a.neurons, b.neurons, learning_rule=nengo.Oja())

        with pytest.raises(ValueError):
            nengo.Connection(n, a, learning_rule=nengo.PES(err))


def test_set_function(Simulator):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        c = nengo.Ensemble(10, 1)

        # Function only OK from node or ensemble
        with pytest.raises(ValueError):
            nengo.Connection(a.neurons, b, function=lambda x: x)

        # Function and transform must match up
        with pytest.raises(ValueError):
            nengo.Connection(a, b, function=lambda x: x[0] * x[1],
                             transform=np.eye(2))

        # These initial functions have correct dimensionality
        conn_2d = nengo.Connection(a, b)
        conn_1d = nengo.Connection(b, c, function=lambda x: x[0] * x[1])

    Simulator(model)  # Builds fine

    with model:
        # Can change to another function with correct dimensionality
        conn_2d.function = lambda x: x ** 2
        conn_1d.function = lambda x: x[0] + x[1]

    Simulator(model)  # Builds fine

    with model:
        # Cannot change to a function with different dimensionality
        # because that would require a change in transform
        with pytest.raises(ValueError):
            conn_2d.function = lambda x: x[0] * x[1]
        with pytest.raises(ValueError):
            conn_1d.function = None

    Simulator(model)  # Builds fine


def test_set_eval_points(Simulator):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        # ConnEvalPoints parameter checks that pre is an ensemble
        nengo.Connection(a, b, eval_points=[[0, 0], [0.5, 1]])
        with pytest.raises(ValueError):
            nengo.Connection(a.neurons, b, eval_points=[[0, 0], [0.5, 1]])

    Simulator(model)  # Builds fine


def test_solverparam():
    """SolverParam must be a solver."""
    class Test(object):
        sp = ConnectionSolverParam(default=None)

    inst = Test()
    assert inst.sp is None
    inst.sp = LstsqL2()
    assert isinstance(inst.sp, LstsqL2)
    assert not inst.sp.weights
    # Non-solver not OK
    with pytest.raises(ValueError):
        inst.sp = 'a'


def test_nonexistant_prepost(Simulator):
    with nengo.Network():
        a = nengo.Ensemble(100, 1)

    with nengo.Network() as model1:
        e1 = nengo.Ensemble(100, 1)
        nengo.Connection(a, e1)
    with pytest.raises(ValueError):
        nengo.Simulator(model1)

    with nengo.Network() as model2:
        e2 = nengo.Ensemble(100, 1)
        nengo.Connection(e2, a)
    with pytest.raises(ValueError):
        nengo.Simulator(model2)

    with nengo.Network() as model3:
        nengo.Probe(a)
    with pytest.raises(ValueError):
        nengo.Simulator(model3)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
