import numpy as np
import pytest

import nengo
from nengo.networks.circularconvolution import (
    circconv, transform_in, transform_out)
from nengo.utils.numpy import rmse


@pytest.mark.parametrize('invert_a', [True, False])
@pytest.mark.parametrize('invert_b', [True, False])
def test_circularconv_transforms(invert_a, invert_b, rng):
    """Test the circular convolution transforms"""
    dims = 100
    x = rng.randn(dims)
    y = rng.randn(dims)
    z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

    tr_a = transform_in(dims, 'A', invert_a)
    tr_b = transform_in(dims, 'B', invert_b)
    tr_out = transform_out(dims)
    XY = np.dot(tr_a, x) * np.dot(tr_b, y)
    z1 = np.dot(tr_out, XY)

    assert np.allclose(z0, z1)


def test_input_magnitude(Simulator, seed, rng, dims=16, magnitude=10):
    """Test to make sure the magnitude scaling works.

    Builds two different CircularConvolution networks, one with the correct
    magnitude and one with 1.0 as the input_magnitude.
    """
    neurons_per_product = 128

    a = rng.normal(scale=np.sqrt(1./dims), size=dims) * magnitude
    b = rng.normal(scale=np.sqrt(1./dims), size=dims) * magnitude
    result = circconv(a, b)

    model = nengo.Network(label="circular conv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        cconv = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims,
            input_magnitude=magnitude)
        nengo.Connection(inputA, cconv.A, synapse=None)
        nengo.Connection(inputB, cconv.B, synapse=None)
        res_p = nengo.Probe(cconv.output)
        cconv_bad = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims,
            input_magnitude=1)  # incorrect magnitude
        nengo.Connection(inputA, cconv_bad.A, synapse=None)
        nengo.Connection(inputB, cconv_bad.B, synapse=None)
        res_p_bad = nengo.Probe(cconv_bad.output)
    with Simulator(model) as sim:
        sim.run(0.01)

    error = rmse(result, sim.data[res_p][-1]) / (magnitude ** 2)
    error_bad = rmse(result, sim.data[res_p_bad][-1]) / (magnitude ** 2)

    assert error < 0.1
    assert error_bad > 0.1


@pytest.mark.parametrize('dims', [4, 32])
def test_neural_accuracy(Simulator, seed, rng, dims, neurons_per_product=128):
    a = rng.normal(scale=np.sqrt(1./dims), size=dims)
    b = rng.normal(scale=np.sqrt(1./dims), size=dims)
    result = circconv(a, b)

    model = nengo.Network(label="circular conv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        cconv = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims)
        nengo.Connection(inputA, cconv.A, synapse=None)
        nengo.Connection(inputB, cconv.B, synapse=None)
        res_p = nengo.Probe(cconv.output)
    with Simulator(model) as sim:
        sim.run(0.01)

    error = rmse(result, sim.data[res_p][-1])

    assert error < 0.1
