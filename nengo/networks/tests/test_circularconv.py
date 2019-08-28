import numpy as np
import pytest

import nengo
from nengo.exceptions import ValidationError
from nengo.networks.circularconvolution import circconv, transform_in, transform_out
from nengo.npext import rms


@pytest.mark.parametrize("invert_a", [True, False])
@pytest.mark.parametrize("invert_b", [True, False])
def test_circularconv_transforms(invert_a, invert_b, rng, allclose):
    """Test the circular convolution transforms"""
    dims = 100
    x = rng.randn(dims)
    y = rng.randn(dims)
    z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

    tr_a = transform_in(dims, "A", invert_a)
    tr_b = transform_in(dims, "B", invert_b)
    tr_out = transform_out(dims)
    XY = np.dot(tr_a, x) * np.dot(tr_b, y)
    z1 = np.dot(tr_out, XY)

    assert allclose(z0, z1)


def test_input_magnitude(Simulator, seed, rng, dims=16, magnitude=10):
    """Test to make sure the magnitude scaling works.

    Builds two different CircularConvolution networks, one with the correct
    magnitude and one with 1.0 as the input_magnitude.
    """
    neurons_per_product = 128

    a = rng.normal(scale=np.sqrt(1.0 / dims), size=dims) * magnitude
    b = rng.normal(scale=np.sqrt(1.0 / dims), size=dims) * magnitude
    result = circconv(a, b)

    model = nengo.Network(label="circular conv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        input_a = nengo.Node(a)
        input_b = nengo.Node(b)
        cconv = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims, input_magnitude=magnitude
        )
        nengo.Connection(input_a, cconv.input_a, synapse=None)
        nengo.Connection(input_b, cconv.input_b, synapse=None)
        res_p = nengo.Probe(cconv.output)
        cconv_bad = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims, input_magnitude=1
        )  # incorrect magnitude
        nengo.Connection(input_a, cconv_bad.input_a, synapse=None)
        nengo.Connection(input_b, cconv_bad.input_b, synapse=None)
        res_p_bad = nengo.Probe(cconv_bad.output)
    with Simulator(model) as sim:
        sim.run(0.01)

    error = rms(result - sim.data[res_p][-1]) / (magnitude ** 2)
    error_bad = rms(result - sim.data[res_p_bad][-1]) / (magnitude ** 2)

    assert error < 0.1
    assert error_bad > 0.1


@pytest.mark.parametrize("dims", [16, 32])
def test_neural_accuracy(Simulator, seed, rng, dims, neurons_per_product=128):
    a = rng.normal(scale=np.sqrt(1.0 / dims), size=dims)
    b = rng.normal(scale=np.sqrt(1.0 / dims), size=dims)
    result = circconv(a, b)

    model = nengo.Network(label="circular conv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        input_a = nengo.Node(a)
        input_b = nengo.Node(b)
        cconv = nengo.networks.CircularConvolution(neurons_per_product, dimensions=dims)
        nengo.Connection(input_a, cconv.input_a, synapse=None)
        nengo.Connection(input_b, cconv.input_b, synapse=None)
        res_p = nengo.Probe(cconv.output)
    with Simulator(model) as sim:
        sim.run(0.01)

    error = rms(result - sim.data[res_p][-1])

    assert error < 0.1


def test_old_input_deprecation_warning():
    with nengo.Network():
        c = nengo.networks.CircularConvolution(n_neurons=10, dimensions=1)
        with pytest.warns(DeprecationWarning):
            assert c.A is c.input_a
        with pytest.warns(DeprecationWarning):
            assert c.B is c.input_b


def test_transform_in_align_error():
    with pytest.raises(ValidationError, match="'align' must be either"):
        transform_in(dims=3, align="badval", invert=False)
