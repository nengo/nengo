import logging

import numpy as np
import pytest

import nengo
from nengo.tests.helpers import Plotter


logger = logging.getLogger(__name__)


def test_simple(Simulator):
    m = nengo.Network('test_simple', seed=123)
    with m:
        input = nengo.Node(output=lambda t: np.sin(t))
        p = nengo.Probe(input, 'output')

    sim = Simulator(m)
    runtime = 0.5
    sim.run(runtime)

    with Plotter(Simulator) as plt:
        plt.plot(sim.trange(), sim.data[p], label='sin')
        plt.legend(loc='best')
        plt.savefig('test_node.test_simple.pdf')
        plt.close()

    sim_t = sim.trange()
    sim_in = sim.data[p].ravel()
    t = 0.001 * np.arange(len(sim_t))
    assert np.allclose(sim_t, t)
    # 1-step delay
    assert np.allclose(sim_in[1:], np.sin(t[:-1]))


def test_connected(Simulator):
    m = nengo.Network('test_connected', seed=123)
    with m:
        input = nengo.Node(output=lambda t: np.sin(t), label='input')
        output = nengo.Node(output=lambda t, x: np.square(x),
                            size_in=1,
                            label='output')
        nengo.Connection(input, output, filter=None)  # Direct connection
        p_in = nengo.Probe(input, 'output')
        p_out = nengo.Probe(output, 'output')

    sim = Simulator(m)
    runtime = 0.5
    sim.run(runtime)

    with Plotter(Simulator) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[p_in], label='sin')
        plt.plot(t, sim.data[p_out], label='sin squared')
        plt.plot(t, np.sin(t), label='ideal sin')
        plt.plot(t, np.sin(t) ** 2, label='ideal squared')
        plt.legend(loc='best')
        plt.savefig('test_node.test_connected.pdf')
        plt.close()

    sim_t = sim.trange()
    sim_sin = sim.data[p_in].ravel()
    sim_sq = sim.data[p_out].ravel()
    t = 0.001 * np.arange(len(sim_t))

    assert np.allclose(sim_t, t)
    # 1-step delay
    assert np.allclose(sim_sin[1:], np.sin(t[:-1]))
    assert np.allclose(sim_sq[1:], sim_sin[:-1] ** 2)


def test_passthrough(Simulator):
    m = nengo.Network("test_passthrough", seed=0)
    with m:
        in1 = nengo.Node(output=lambda t: np.sin(t))
        in2 = nengo.Node(output=lambda t: t)
        passthrough = nengo.Node(size_in=1)
        out = nengo.Node(output=lambda t, x: x, size_in=1)

        nengo.Connection(in1, passthrough, filter=None)
        nengo.Connection(in2, passthrough, filter=None)
        nengo.Connection(passthrough, out, filter=None)

        in1_p = nengo.Probe(in1, 'output')
        in2_p = nengo.Probe(in2, 'output')
        out_p = nengo.Probe(out, 'output')

    sim = Simulator(m)
    runtime = 0.5
    sim.run(runtime)

    with Plotter(Simulator) as plt:
        plt.plot(sim.trange(), sim.data[in1_p]+sim.data[in2_p], label='in+in2')
        plt.plot(sim.trange()[:-2], sim.data[out_p][2:], label='out')
        plt.legend(loc='best')
        plt.savefig('test_node.test_passthrough.pdf')
        plt.close()

    # One-step delay between first and second nonlinearity
    sim_in = sim.data[in1_p][:-1] + sim.data[in2_p][:-1]
    sim_out = sim.data[out_p][1:]
    assert np.allclose(sim_in, sim_out)


def test_circular(Simulator):
    m = nengo.Network("test_circular", seed=0)
    with m:
        a = nengo.Node(output=lambda t, x: x+1, size_in=1)
        b = nengo.Node(output=lambda t, x: x+1, size_in=1)
        nengo.Connection(a, b, filter=None)
        nengo.Connection(b, a, filter=None)

        a_p = nengo.Probe(a, 'output')
        b_p = nengo.Probe(b, 'output')

    sim = Simulator(m)
    runtime = 0.5
    sim.run(runtime)

    assert np.allclose(sim.data[a_p], sim.data[b_p])


def test_function_args_error(Simulator):
    with nengo.Network("test_function_args_error", seed=0):
        with pytest.raises(TypeError):
            nengo.Node(output=lambda t, x: x+1)
        with pytest.raises(TypeError):
            nengo.Node(output=lambda t: t+1, size_in=1)
        with pytest.raises(TypeError):
            nengo.Node(output=lambda t, x, y: t+1, size_in=2)


def test_output_shape_error(Simulator):
    with nengo.Network("test_output_shape_error", seed=0):
        with pytest.raises(ValueError):
            nengo.Node(output=[[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            nengo.Node(output=lambda t: [[t, t+1]])
        with pytest.raises(ValueError):
            nengo.Node(output=[[3, 1], [2, 9]], size_out=4)
        with pytest.raises(ValueError):
            nengo.Node(output=[1, 2, 3, 4, 5], size_out=4)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
