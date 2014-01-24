import logging

import numpy as np
import pytest

import nengo
from nengo.networks import EnsembleArray
from nengo.networks.circularconvolution import circconv
from nengo.utils.numpy import rmse
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def test_circularconv_helpers():
    """Test the circular convolution helper functions in Numpy"""
    rng = np.random.RandomState(43232)

    dims = 1000
    invert_a = True
    invert_b = False
    x = rng.randn(dims)
    y = rng.randn(dims)
    z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

    dims2 = 2 * dims - (2 if dims % 2 == 0 else 1)
    inA = nengo.networks.CircularConvolution._input_transform(
        dims, first=True, invert=invert_a)
    inB = nengo.networks.CircularConvolution._input_transform(
        dims, first=False, invert=invert_b)
    outC = nengo.networks.CircularConvolution._output_transform(dims)

    XY = np.zeros((dims2, 2))
    XY += np.dot(inA.reshape(dims2, 2, dims), x)
    XY += np.dot(inB.reshape(dims2, 2, dims), y)

    C = XY[:, 0] * XY[:, 1]
    z1 = np.dot(outC, C)

    assert np.allclose(z0, z1)
    # assert_allclose(self, logger, z0, z1)


def test_circularconv(Simulator, nl, dims=4, neurons_per_product=128):
    rng = np.random.RandomState(42342)

    n_neurons = neurons_per_product
    n_neurons_d = 2 * neurons_per_product
    radius = 1

    a = rng.normal(scale=np.sqrt(1./dims), size=dims)
    b = rng.normal(scale=np.sqrt(1./dims), size=dims)
    c = circconv(a, b)
    assert np.abs(a).max() < radius
    assert np.abs(b).max() < radius
    assert np.abs(c).max() < radius

    ### model
    model = nengo.Model("circular convolution")

    inputA = nengo.Node(output=a)
    inputB = nengo.Node(output=b)
    A = EnsembleArray(nl(n_neurons), dims, radius=radius)
    B = EnsembleArray(nl(n_neurons), dims, radius=radius)
    C = EnsembleArray(nl(n_neurons), dims, radius=radius)
    D = nengo.networks.CircularConvolution(
        neurons=nl(n_neurons_d),
        dimensions=A.dimensions, radius=radius)

    nengo.Connection(inputA, A.input)
    nengo.Connection(inputB, B.input)
    nengo.Connection(A.output, D.A)
    nengo.Connection(B.output, D.B)
    nengo.Connection(D.output, C.input)

    A_p = nengo.Probe(A.output, 'output', filter=0.03)
    B_p = nengo.Probe(B.output, 'output', filter=0.03)
    C_p = nengo.Probe(C.output, 'output', filter=0.03)
    D_p = nengo.Probe(D.ensemble.output, 'output', filter=0.03)

    # check FFT magnitude
    d = np.dot(D.transformA, a) + np.dot(D.transformB, b)
    assert np.abs(d).max() < radius

    ### simulation
    sim = Simulator(model)
    sim.run(1.0)

    t = sim.trange()

    with Plotter(Simulator, nl) as plt:
        def plot(sim, a, A, title=""):
            a_ref = np.tile(a, (len(t), 1))
            a_sim = sim.data[A_p]
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            for i in range(min(dims, len(colors))):
                plt.plot(t, a_ref[:, i], '--', color=colors[i])
                plt.plot(t, a_sim[:, i], '-', color=colors[i])
                plt.title(title)

        plt.subplot(221)
        plot(sim, a, A, title="A")
        plt.subplot(222)
        plot(sim, b, B, title="B")
        plt.subplot(223)
        plot(sim, c, C, title="C")
        plt.subplot(224)
        plot(sim, d, D.ensemble, title="D")
        plt.savefig('test_circularconv.test_circularconv_%d.pdf' % dims)
        plt.close()

    ### results
    tmask = t > (0.5 + sim.dt/2)
    assert sim.data[A_p][tmask].shape == (499, dims)
    a_sim = sim.data[A_p][tmask].mean(axis=0)
    b_sim = sim.data[B_p][tmask].mean(axis=0)
    c_sim = sim.data[C_p][tmask].mean(axis=0)
    d_sim = sim.data[D_p][tmask].mean(axis=0)

    rtol, atol = 0.1, 0.05
    assert np.allclose(a, a_sim, rtol=rtol, atol=atol)
    assert np.allclose(b, b_sim, rtol=rtol, atol=atol)
    assert np.allclose(d, d_sim, rtol=rtol, atol=atol)
    assert rmse(c, c_sim) < 0.075


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
