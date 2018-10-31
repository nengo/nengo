import numpy as np
import pytest

import nengo
from nengo.exceptions import ValidationError
from nengo._vendor.npconv2d import conv2d


@pytest.mark.parametrize("dimensions", (1, 2))
@pytest.mark.parametrize("padding", ("same", "valid"))
@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("fixed_kernel", (True, False))
def test_convolution(
        dimensions, padding, channels_last, fixed_kernel, Simulator, rng):
    input_d = 4
    input_channels = 2
    output_channels = 5
    kernel_d = 3
    kernel_size = (kernel_d,) if dimensions == 1 else (kernel_d, kernel_d)
    output_d = input_d - kernel_d // 2 * 2 if padding == "valid" else input_d

    input_shape = (input_d, input_channels)
    kernel_shape = (kernel_d, input_channels, output_channels)
    output_shape = (output_d, output_channels)

    if dimensions == 2:
        input_shape = (input_d,) + input_shape
        kernel_shape = (kernel_d,) + kernel_shape
        output_shape = (output_d,) + output_shape

    if not channels_last:
        input_shape = tuple(np.roll(input_shape, 1))
        output_shape = tuple(np.roll(output_shape, 1))

    with nengo.Network() as net:
        x = rng.randn(*input_shape)
        w = (rng.randn(*kernel_shape) if fixed_kernel
             else nengo.dists.Uniform(-0.1, 0.1))

        a = nengo.Node(np.ravel(x))
        b = nengo.Node(size_in=np.prod(output_shape))
        conn = nengo.Connection(
            a, b,
            synapse=None,
            transform=nengo.Convolution(
                output_channels,
                input_shape,
                init=w,
                padding=padding,
                kernel_size=kernel_size,
                strides=(1,) if dimensions == 1 else (1, 1),
                channels_last=channels_last))
        p = nengo.Probe(b)

        # check error handling
        bad_in = nengo.Node([0])
        bad_out = nengo.Node(size_in=5)
        with pytest.raises(ValidationError):
            nengo.Connection(bad_in, b, transform=conn.transform)
        with pytest.raises(ValidationError):
            nengo.Connection(a, bad_out, transform=conn.transform)

    assert conn.transform.output_shape.shape == output_shape
    assert conn.transform.kernel_shape == kernel_shape

    with Simulator(net) as sim:
        sim.step()

    weights = sim.data[conn].weights
    if not channels_last:
        x = np.moveaxis(x, 0, -1)
    if dimensions == 1:
        x = x[:, None, :]
        weights = weights[:, None, :, :]
    truth = conv2d.conv2d(x[None, ...], weights, pad=padding.upper())[0]
    if not channels_last:
        truth = np.moveaxis(truth, -1, 0)

    assert np.allclose(sim.data[p][0], np.ravel(truth))
