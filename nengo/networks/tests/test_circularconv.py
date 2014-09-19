import logging

import numpy as np
import pytest

import nengo
from nengo.networks.circularconvolution import circconv
from nengo.utils.numpy import rmse

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


@pytest.mark.parametrize('dims', [4, 32])
def test_neural_accuracy(Simulator, dims, neurons_per_product=128):
    rng = np.random.RandomState(4238)
    a = rng.normal(scale=np.sqrt(1./dims), size=dims)
    b = rng.normal(scale=np.sqrt(1./dims), size=dims)
    result = circconv(a, b)

    model = nengo.Network(label="circular conv", seed=1)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        cconv = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims)
        nengo.Connection(inputA, cconv.A, synapse=None)
        nengo.Connection(inputB, cconv.B, synapse=None)
        res_p = nengo.Probe(cconv.output)
    sim = Simulator(model)
    sim.run(0.01)

    error = rmse(result, sim.data[res_p][-1])

    assert error < 0.1


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
