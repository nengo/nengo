import logging

import numpy as np
import pytest

import nengo
from nengo.builder import ShapeMismatch
from nengo.utils.numpy import rmse, norm
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("n_dimensions", [1, 200])
def test_encoders(n_dimensions, n_neurons=10, encoders=None):
    if encoders is None:
        encoders = np.random.normal(size=(n_neurons, n_dimensions))
        encoders /= norm(encoders, axis=-1, keepdims=True)

    model = nengo.Network(label="_test_encoders")
    with model:
        ens = nengo.Ensemble(neurons=nengo.LIF(n_neurons),
                             dimensions=n_dimensions,
                             encoders=encoders,
                             label="A")
    sim = nengo.Simulator(model)

    assert np.allclose(encoders, sim.data[ens].encoders)


def test_encoders_wrong_shape():
    n_dimensions = 3
    encoders = np.random.normal(size=n_dimensions)
    with pytest.raises(ShapeMismatch):
        test_encoders(n_dimensions, encoders=encoders)


def test_encoders_negative_neurons():
    with pytest.raises(ValueError):
        test_encoders(1, n_neurons=-1)


def test_encoders_no_dimensions():
    with pytest.raises(ValueError):
        test_encoders(0)


def test_constant_scalar(Simulator, nl):
    """A Network that represents a constant value."""
    N = 30
    val = 0.5

    m = nengo.Network(label='test_constant_scalar', seed=123)
    with m:
        input = nengo.Node(output=val, label='input')
        A = nengo.Ensemble(nl(N), 1)
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', filter=0.1)

    sim = Simulator(m, dt=0.001)
    sim.run(1.0)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.1')
        plt.legend(loc=0)
        plt.savefig('test_ensemble.test_constant_scalar.pdf')
        plt.close()

    assert np.allclose(sim.data[in_p].ravel(), val, atol=.1, rtol=.01)
    assert np.allclose(sim.data[A_p][-10:], val, atol=.1, rtol=.01)


def test_constant_vector(Simulator, nl):
    """A network that represents a constant 3D vector."""
    N = 30
    vals = [0.6, 0.1, -0.5]

    m = nengo.Network(label='test_constant_vector', seed=123)
    with m:
        input = nengo.Node(output=vals)
        A = nengo.Ensemble(nl(N * len(vals)), len(vals))
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', filter=0.1)

    sim = Simulator(m)
    sim.run(1.0)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.1')
        plt.legend(loc=0, prop={'size': 10})
        plt.savefig('test_ensemble.test_constant_vector.pdf')
        plt.close()

    assert np.allclose(sim.data[in_p][-10:], vals, atol=.1, rtol=.01)
    assert np.allclose(sim.data[A_p][-10:], vals, atol=.1, rtol=.01)


def test_scalar(Simulator, nl):
    """A network that represents sin(t)."""
    N = 30

    m = nengo.Network(label='test_scalar', seed=123)
    with m:
        input = nengo.Node(output=np.sin, label='input')
        A = nengo.Ensemble(nl(N), 1, label='A')
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', filter=0.02)

    sim = Simulator(m)
    sim.run(5.0)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.02')
        plt.legend(loc=0)
        plt.savefig('test_ensemble.test_scalar.pdf')
        plt.close()

    target = np.sin(np.arange(5000) / 1000.)
    target.shape = (-1, 1)
    logger.debug("[New API] input RMSE: %f", rmse(target, sim.data[in_p]))
    logger.debug("[New API] A RMSE: %f", rmse(target, sim.data[A_p]))
    assert rmse(target, sim.data[in_p]) < 0.001
    assert rmse(target, sim.data[A_p]) < 0.1


def test_vector(Simulator, nl):
    """A network that represents sin(t), cos(t), arctan(t)."""
    N = 40

    m = nengo.Network(label='test_vector', seed=123)
    with m:
        input = nengo.Node(
            output=lambda t: [np.sin(t), np.cos(t), np.arctan(t)])
        A = nengo.Ensemble(nl(N * 3), 3, radius=2)
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', filter=0.02)

    sim = Simulator(m)
    sim.run(5)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.02')
        plt.legend(loc='best', prop={'size': 10})
        plt.savefig('test_ensemble.test_vector.pdf')
        plt.close()

    target = np.vstack((np.sin(np.arange(5000) / 1000.),
                        np.cos(np.arange(5000) / 1000.),
                        np.arctan(np.arange(5000) / 1000.))).T
    logger.debug("In RMSE: %f", rmse(target, sim.data[in_p]))
    assert rmse(target, sim.data[in_p]) < 0.01
    assert rmse(target, sim.data[A_p]) < 0.1


def test_product(Simulator, nl):
    N = 80

    m = nengo.Network(label='test_product', seed=124)
    with m:
        sin = nengo.Node(output=np.sin)
        cons = nengo.Node(output=-.5)
        factors = nengo.Ensemble(nl(2 * N), dimensions=2, radius=1.5)
        if nl != nengo.Direct:
            factors.encoders = np.tile(
                [[1, 1], [-1, 1], [1, -1], [-1, -1]],
                (factors.n_neurons // 4, 1))
        product = nengo.Ensemble(nl(N), dimensions=1)
        nengo.Connection(sin, factors[0])
        nengo.Connection(cons, factors[1])
        nengo.Connection(
            factors, product, function=lambda x: x[0] * x[1], synapse=0.01)

        sin_p = nengo.Probe(sin, 'output', sample_every=.01)
        # TODO
        # m.probe(conn, sample_every=.01)
        factors_p = nengo.Probe(
            factors, 'decoded_output', sample_every=.01, filter=.01)
        product_p = nengo.Probe(
            product, 'decoded_output', sample_every=.01, filter=.01)

    sim = Simulator(m)
    sim.run(6)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange(dt=.01)
        plt.subplot(211)
        plt.plot(t, sim.data[factors_p])
        plt.plot(t, np.sin(np.arange(0, 6, .01)))
        plt.plot(t, sim.data[sin_p])
        plt.subplot(212)
        plt.plot(t, sim.data[product_p])
        # TODO
        # plt.plot(sim.data[conn])
        plt.plot(t, -.5 * np.sin(np.arange(0, 6, .01)))
        plt.savefig('test_ensemble.test_prod.pdf')
        plt.close()

    sin = np.sin(np.arange(0, 6, .01))
    assert rmse(sim.data[factors_p][:, 0], sin) < 0.1
    assert rmse(sim.data[factors_p][20:, 1], -0.5) < 0.1

    assert rmse(sim.data[product_p][:, 0], -0.5 * sin) < 0.1
    # assert rmse(sim.data[conn][:, 0], -0.5 * sin) < 0.1


@pytest.mark.parametrize('dims, points', [(1, 528), (2, 823), (3, 937)])
def test_eval_points_number(Simulator, nl, dims, points):
    model = nengo.Network(seed=123)
    with model:
        A = nengo.Ensemble(nl(5), dims, eval_points=points)

    sim = Simulator(model)
    assert sim.data[A].eval_points.shape == (points, dims)


@pytest.mark.parametrize('neurons, dims', [
    (10, 1), (392, 1), (2108, 1), (100, 2), (1290, 4), (20, 9)])
def test_eval_points_heuristic(Simulator, nl_nodirect, neurons, dims):
    def heuristic(neurons, dims):
        return max(np.clip(500 * dims, 750, 2500), 2 * neurons)

    model = nengo.Network(seed=123)
    with model:
        A = nengo.Ensemble(nl_nodirect(neurons), dims)

    sim = Simulator(model)
    points = sim.data[A].eval_points
    assert points.shape == (heuristic(neurons, dims), dims)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
