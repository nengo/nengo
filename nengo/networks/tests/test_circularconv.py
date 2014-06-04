import logging

import numpy as np
import pytest

import nengo
from nengo.networks import EnsembleArray
from nengo.networks.circularconvolution import circconv
from nengo.utils.compat import range
from nengo.utils.numpy import rmse
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('invert_a', [True, False])
@pytest.mark.parametrize('invert_b', [True, False])
def test_circularconv_transforms(invert_a, invert_b):
    """Test the circular convolution transforms"""
    rng = np.random.RandomState(43232)

    dims = 100
    x = rng.randn(dims)
    y = rng.randn(dims)
    z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

    cconv = nengo.networks.CircularConvolution(
        1, dims, invert_a=invert_a, invert_b=invert_b)
    XY = np.dot(cconv.transformA, x) * np.dot(cconv.transformB, y)
    z1 = np.dot(cconv.transform_out, XY)

    assert np.allclose(z0, z1)


def test_circularconv(Simulator, nl, dims=4, neurons_per_product=128):
    rng = np.random.RandomState(4238)

    n_neurons = neurons_per_product
    n_neurons_d = 2 * neurons_per_product
    radius = 1

    a = rng.normal(scale=np.sqrt(1./dims), size=dims)
    b = rng.normal(scale=np.sqrt(1./dims), size=dims)
    result = circconv(a, b)
    assert np.abs(a).max() < radius
    assert np.abs(b).max() < radius
    assert np.abs(result).max() < radius

    # --- model
    model = nengo.Network(label="circular convolution")
    with model:
        model.config[nengo.Ensemble].neuron_type = nl()
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        A = EnsembleArray(n_neurons, dims, radius=radius)
        B = EnsembleArray(n_neurons, dims, radius=radius)
        cconv = nengo.networks.CircularConvolution(
            n_neurons_d, dimensions=dims)
        res = EnsembleArray(n_neurons, dims, radius=radius)

        nengo.Connection(inputA, A.input)
        nengo.Connection(inputB, B.input)
        nengo.Connection(A.output, cconv.A)
        nengo.Connection(B.output, cconv.B)
        nengo.Connection(cconv.output, res.input)

        A_p = nengo.Probe(A.output, synapse=0.03)
        B_p = nengo.Probe(B.output, synapse=0.03)
        res_p = nengo.Probe(res.output, synapse=0.03)

    # --- simulation
    sim = Simulator(model)
    sim.run(1.0)

    t = sim.trange()

    with Plotter(Simulator, nl) as plt:
        def plot(actual, probe, title=""):
            ref_y = np.tile(actual, (len(t), 1))
            sim_y = sim.data[probe]
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            for i in range(min(dims, len(colors))):
                plt.plot(t, ref_y[:, i], '--', color=colors[i])
                plt.plot(t, sim_y[:, i], '-', color=colors[i])
                plt.title(title)

        plt.subplot(311)
        plot(a, A_p, title="A")
        plt.subplot(312)
        plot(b, B_p, title="B")
        plt.subplot(313)
        plot(result, res_p, title="Result")
        plt.tight_layout()
        plt.savefig('test_circularconv.test_circularconv_%d.pdf' % dims)
        plt.close()

    # --- results
    tmask = t > (0.5 + sim.dt/2)
    assert sim.data[A_p][tmask].shape == (499, dims)
    a_sim = sim.data[A_p][tmask].mean(axis=0)
    b_sim = sim.data[B_p][tmask].mean(axis=0)
    res_sim = sim.data[res_p][tmask].mean(axis=0)

    rtol, atol = 0.1, 0.05
    assert np.allclose(a, a_sim, rtol=rtol, atol=atol)
    assert np.allclose(b, b_sim, rtol=rtol, atol=atol)
    assert rmse(result, res_sim) < 0.075


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
