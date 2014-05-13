import logging

import numpy as np
import pytest

import nengo
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
                         solver=nengo.decoders.LstsqL2(weights=True))

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


def test_pes_learning_initial_weights(Simulator, nl_nodirect):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=3902)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=2)
        e = nengo.Ensemble(n, dimensions=2)

        initial_weights = np.random.random((a.n_neurons, u_learned.n_neurons))
        nengo.Connection(u, a)
        err_conn = nengo.Connection(e, u_learned, modulatory=True)
        nengo.Connection(a.neurons, u_learned.neurons,
                         transform=initial_weights,
                         learning_rule=nengo.PES(err_conn, 10))

        nengo.Connection(u_learned, e, transform=-1)
        nengo.Connection(u, e)

        u_learned_p = nengo.Probe(u_learned, synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)

    sim = Simulator(m)
    sim.run(1.)

    assert np.allclose(sim.data[u_learned_p][-1], learned_vector, atol=0.05)
    assert np.allclose(
        sim.data[e_p][-1], np.zeros(len(learned_vector)), atol=0.05)


def test_pes_learning_rule_nef_weights(Simulator, nl_nodirect):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=3902)
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
                         solver=nengo.decoders.LstsqL2nz(weights=True))

        nengo.Connection(u_learned, e, transform=-1)
        nengo.Connection(u, e)

        u_learned_p = nengo.Probe(u_learned, synapse=0.1)
        e_p = nengo.Probe(e, synapse=0.1)

    sim = Simulator(m)
    sim.run(1.)

    assert np.allclose(sim.data[u_learned_p][-1], learned_vector, atol=0.05)
    assert np.allclose(
        sim.data[e_p][-1], np.zeros(len(learned_vector)), atol=0.05)


def test_pes_learning_decoders(Simulator, nl_nodirect):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=3902)
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

    assert np.allclose(sim.data[u_learned_p][-1], learned_vector, atol=0.05)
    assert np.allclose(
        sim.data[e_p][-1], np.zeros(len(learned_vector)), atol=0.05)


def test_pes_learning_decoders_multidimensional(Simulator, nl_nodirect):
    n = 200
    input_vector = [0.5, -0.5]
    learned_vector = [input_vector[0]**2 + input_vector[1]**2]

    m = nengo.Network(seed=3902)
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

    assert np.allclose(sim.data[u_learned_p][-1], learned_vector, atol=0.05)
    assert np.allclose(
        sim.data[e_p][-1], np.zeros(len(learned_vector)), atol=0.05)


@pytest.mark.parametrize('learning_rule', [
    nengo.BCM(), nengo.Oja(), [nengo.Oja(), nengo.BCM()]])
def test_unsupervised_learning_rule(Simulator, nl_nodirect, learning_rule):
    n = 200
    learned_vector = [0.5, -0.5]

    m = nengo.Network(seed=3902)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=learned_vector)
        a = nengo.Ensemble(n, dimensions=2)
        u_learned = nengo.Ensemble(n, dimensions=2)

        initial_weights = np.random.random((a.n_neurons,
                                            u_learned.n_neurons))

        nengo.Connection(u, a)
        nengo.Connection(a.neurons, u_learned.neurons,
                         transform=initial_weights,
                         learning_rule=nengo.Oja())

    sim = Simulator(m)
    sim.run(1.)


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


def test_slicing(Simulator, nl_nodirect):
    name = 'connection_slicing'
    N = 30

    with nengo.Network(label=name) as m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        neurons3 = nengo.Ensemble(3, dimensions=1).neurons
        ens1 = nengo.Ensemble(N, dimensions=1)
        ens2 = nengo.Ensemble(N, dimensions=2)
        ens3 = nengo.Ensemble(N, dimensions=3)
        node1 = nengo.Node(output=[0])
        node2 = nengo.Node(output=[0, 0])
        node3 = nengo.Node(output=[0, 0, 0])

        # Pre slice with default transform -> 1x3 transform
        conn = nengo.Connection(node3[2], ens1)
        assert np.all(conn.transform == np.array(1))
        assert np.all(conn.transform_full == np.array([[0, 0, 1]]))

        # Post slice with 1x1 transform -> 1x2 transform
        conn = nengo.Connection(node2[0], ens1, transform=-2)
        assert np.all(conn.transform == np.array(-2))
        assert np.all(conn.transform_full == np.array([[-2, 0]]))

        # Post slice with 2x1 tranfsorm -> 3x1 transform
        conn = nengo.Connection(node1, ens3[::2], transform=[[1], [2]])
        assert np.all(conn.transform == np.array([[1], [2]]))
        assert np.all(conn.transform_full == np.array([[1], [0], [2]]))

        # Both slices with 2x1 transform -> 3x2 transform
        conn = nengo.Connection(ens2[-1], neurons3[1:], transform=[[1], [2]])
        assert np.all(conn.transform == np.array([[1], [2]]))
        assert np.all(conn.transform_full == np.array(
            [[0, 0], [0, 1], [0, 2]]))

        # Full slices that can be optimized away
        conn = nengo.Connection(ens3[:], ens3, transform=2)
        assert np.all(conn.transform == np.array(2))
        assert np.all(conn.transform_full == np.array(2))

        # Pre slice with 1x1 transform on 2x2 slices -> 2x3 transform
        conn = nengo.Connection(neurons3[:2], ens2, transform=-1)
        assert np.all(conn.transform == np.array(-1))
        assert np.all(conn.transform_full == np.array(
            [[-1, 0, 0], [0, -1, 0]]))

        # Both slices with 1x1 transform on 2x2 slices -> 3x3 transform
        conn = nengo.Connection(neurons3[1:], neurons3[::2], transform=-1)
        assert np.all(conn.transform == np.array(-1))
        assert np.all(conn.transform_full == np.array([[0, -1, 0],
                                                       [0, 0, 0],
                                                       [0, 0, -1]]))

        # Both slices with 2x2 transform -> 3x3 transform
        conn = nengo.Connection(node3[[0, 2]], neurons3[1:],
                                transform=[[1, 2], [3, 4]])
        assert np.all(conn.transform == np.array([[1, 2], [3, 4]]))
        assert np.all(conn.transform_full == np.array([[0, 0, 0],
                                                       [1, 0, 2],
                                                       [3, 0, 4]]))

        # Both slices with 2x3 transform -> 3x3 transform... IN REVERSE!
        conn = nengo.Connection(neurons3[::-1], neurons3[[2, 0]],
                                transform=[[1, 2, 3], [4, 5, 6]])
        assert np.all(conn.transform == np.array([[1, 2, 3], [4, 5, 6]]))
        assert np.all(conn.transform_full == np.array([[6, 5, 4],
                                                       [0, 0, 0],
                                                       [3, 2, 1]]))

        # Both slices using lists
        conn = nengo.Connection(neurons3[[1, 0, 2]], neurons3[[2, 1]],
                                transform=[[1, 2, 3], [4, 5, 6]])
        assert np.all(conn.transform == np.array([[1, 2, 3], [4, 5, 6]]))
        assert np.all(conn.transform_full == np.array([[0, 0, 0],
                                                       [5, 4, 6],
                                                       [2, 1, 3]]))


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


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
