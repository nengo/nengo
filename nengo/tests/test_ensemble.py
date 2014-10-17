import logging

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.utils.distributions import Choice
from nengo.utils.testing import Plotter, warns

logger = logging.getLogger(__name__)


def test_missing_attribute():
    with nengo.Network():
        a = nengo.Ensemble(10, 1)

        with warns(SyntaxWarning):
            a.dne = 9


@pytest.mark.parametrize("n_dimensions", [1, 200])
def test_encoders(n_dimensions, n_neurons=10, encoders=None):
    if encoders is None:
        encoders = np.random.normal(size=(n_neurons, n_dimensions))
        encoders = npext.array(encoders, min_dims=2, dtype=np.float64)
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    model = nengo.Network(label="_test_encoders")
    with model:
        ens = nengo.Ensemble(n_neurons=n_neurons,
                             dimensions=n_dimensions,
                             encoders=encoders,
                             label="A")
    sim = nengo.Simulator(model)

    assert np.allclose(encoders, sim.data[ens].encoders)


def test_encoders_wrong_shape():
    n_dimensions = 3
    encoders = np.random.normal(size=n_dimensions)
    with pytest.raises(ValueError):
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
        m.config[nengo.Ensemble].neuron_type = nl()
        input = nengo.Node(output=val, label='input')
        A = nengo.Ensemble(N, 1)
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', synapse=0.1)

    sim = Simulator(m, dt=0.001)
    sim.run(1.0)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.1')
        plt.ylim([0, 1.05 * val])
        plt.legend(loc=0)
        plt.savefig('test_ensemble.test_constant_scalar.pdf')
        plt.close()

    assert np.allclose(sim.data[in_p], val, atol=.1, rtol=.01)
    assert np.allclose(sim.data[A_p][-10:], val, atol=.1, rtol=.01)


def test_constant_vector(Simulator, nl):
    """A network that represents a constant 3D vector."""
    N = 30
    vals = [0.6, 0.1, -0.5]

    m = nengo.Network(label='test_constant_vector', seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl()
        input = nengo.Node(output=vals)
        A = nengo.Ensemble(N * len(vals), len(vals))
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', synapse=0.1)

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
        m.config[nengo.Ensemble].neuron_type = nl()
        input = nengo.Node(output=np.sin, label='input')
        A = nengo.Ensemble(N, 1, label='A')
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', synapse=0.02)

    sim = Simulator(m)
    sim.run(5.0)
    t = sim.trange()

    with Plotter(Simulator, nl) as plt:
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.02')
        plt.legend(loc=0)
        plt.savefig('test_ensemble.test_scalar.pdf')
        plt.close()

    target = np.sin(t).reshape(-1, 1)
    logger.debug("Input RMSE: %f", npext.rmse(target, sim.data[in_p]))
    logger.debug("A RMSE: %f", npext.rmse(target, sim.data[A_p]))
    assert npext.rmse(target, sim.data[in_p]) < 0.001
    assert npext.rmse(target, sim.data[A_p]) < 0.1


def test_vector(Simulator, nl):
    """A network that represents sin(t), cos(t), arctan(t)."""
    N = 40

    m = nengo.Network(label='test_vector', seed=123)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl()
        input = nengo.Node(
            output=lambda t: [np.sin(t), np.cos(t), np.arctan(t)])
        A = nengo.Ensemble(N * 3, 3, radius=2)
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', synapse=0.02)

    sim = Simulator(m)
    sim.run(5)
    t = sim.trange()

    with Plotter(Simulator, nl) as plt:
        plt.plot(t, sim.data[in_p], label='Input')
        plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.02')
        plt.legend(loc='best', prop={'size': 10})
        plt.savefig('test_ensemble.test_vector.pdf')
        plt.close()

    target = np.vstack((np.sin(t), np.cos(t), np.arctan(t))).T
    logger.debug("In RMSE: %f", npext.rmse(target, sim.data[in_p]))
    assert npext.rmse(target, sim.data[in_p]) < 0.01
    assert npext.rmse(target, sim.data[A_p]) < 0.1


def test_product(Simulator, nl):
    N = 80
    dt2 = 0.002
    f = lambda t: np.sin(6*t)

    m = nengo.Network(label='test_product', seed=124)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl()
        sin = nengo.Node(output=f)
        cons = nengo.Node(output=-.5)
        factors = nengo.Ensemble(
            2 * N, 2, radius=1.5,
            encoders=Choice([[1, 1], [-1, 1], [1, -1], [-1, -1]]))
        product = nengo.Ensemble(N, dimensions=1)
        nengo.Connection(sin, factors[0])
        nengo.Connection(cons, factors[1])
        nengo.Connection(
            factors, product, function=lambda x: x[0] * x[1], synapse=0.01)

        sin_p = nengo.Probe(sin, sample_every=dt2)
        factors_p = nengo.Probe(factors, sample_every=dt2, synapse=0.01)
        product_p = nengo.Probe(product, sample_every=dt2, synapse=0.01)

    sim = Simulator(m)
    sim.run(1)
    t = sim.trange(dt=dt2)

    with Plotter(Simulator, nl) as plt:
        plt.subplot(211)
        plt.plot(t, sim.data[factors_p])
        plt.legend(['factor 1', 'factor 2'])
        plt.subplot(212)
        plt.plot(t, -.5 * f(t), 'k--')
        plt.plot(t, sim.data[product_p])
        plt.legend(['exact product', 'neural product'], loc=4)
        plt.savefig('test_ensemble.test_product.pdf')
        plt.close()

    assert npext.rmse(sim.data[factors_p][:, 0], f(t)) < 0.1
    assert npext.rmse(sim.data[factors_p][20:, 1], -0.5) < 0.1
    assert npext.rmse(sim.data[product_p][:, 0], -0.5 * f(t)) < 0.1


@pytest.mark.parametrize('dims, points', [(1, 528), (2, 823), (3, 937)])
def test_eval_points_number(Simulator, nl, dims, points):
    model = nengo.Network(seed=123)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl()
        A = nengo.Ensemble(5, dims, n_eval_points=points)

    sim = Simulator(model)
    assert sim.data[A].eval_points.shape == (points, dims)


def test_eval_points_number_warning(Simulator, recwarn):
    model = nengo.Network(seed=123)
    with model:
        A = nengo.Ensemble(5, 1, n_eval_points=10, eval_points=[[0.1], [0.2]])

    sim = Simulator(model)
    assert np.allclose(sim.data[A].eval_points, [[0.1], [0.2]])
    # n_eval_points doesn't match actual passed eval_points, which warns
    assert recwarn.pop() is not None


@pytest.mark.parametrize('neurons, dims', [
    (10, 1), (392, 1), (2108, 1), (100, 2), (1290, 4), (20, 9)])
def test_eval_points_heuristic(Simulator, nl_nodirect, neurons, dims):
    def heuristic(neurons, dims):
        return max(np.clip(500 * dims, 750, 2500), 2 * neurons)

    model = nengo.Network(seed=123)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl_nodirect()
        A = nengo.Ensemble(neurons, dims)

    sim = Simulator(model)
    points = sim.data[A].eval_points
    assert points.shape == (heuristic(neurons, dims), dims)


def test_len():
    """Make sure we can do len(ens) or len(ens.neurons)."""
    with nengo.Network():
        ens1 = nengo.Ensemble(10, dimensions=1)
        ens5 = nengo.Ensemble(100, dimensions=5)
    # Ensemble.__len__
    assert len(ens1) == 1
    assert len(ens5) == 5
    assert len(ens1[0]) == 1
    assert len(ens5[:3]) == 3

    # Neurons.__len__
    assert len(ens1.neurons) == 10
    assert len(ens5.neurons) == 100
    assert len(ens1.neurons[0]) == 1
    assert len(ens5.neurons[90:]) == 10


def test_invalid_rates(Simulator):
    with nengo.Network() as model:
        nengo.Ensemble(1, 1, max_rates=[200],
                       neuron_type=nengo.LIF(tau_ref=0.01))

    with pytest.raises(ValueError):
        Simulator(model)


def test_gain_bias(Simulator, nl_nodirect):

    N = 17
    D = 2

    gain = np.random.uniform(low=0.2, high=5, size=N)
    bias = np.random.uniform(low=0.2, high=1, size=N)

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(N, D)
        a.gain = gain
        a.bias = bias

    sim = Simulator(model)
    assert np.array_equal(gain, sim.data[a].gain)
    assert np.array_equal(bias, sim.data[a].bias)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
