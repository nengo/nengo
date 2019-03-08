import numpy as np
import pytest

import nengo
from nengo.exceptions import ValidationError
from nengo.transforms import ChannelShape, ChannelShapeParam
from nengo._vendor.npconv2d import conv2d


@pytest.mark.parametrize("dimensions", (1, 2))
@pytest.mark.parametrize("padding", ("same", "valid"))
@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("fixed_kernel", (True, False))
def test_convolution(
        dimensions,
        padding,
        channels_last,
        fixed_kernel,
        Simulator,
        rng,
        seed):
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

    with nengo.Network(seed=seed) as net:
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


def test_argreprs():
    """Test repr() for each transform type."""
    assert repr(nengo.Dense((1, 2), init=[[1, 1]])) == "Dense(shape=(1, 2))"

    assert (repr(nengo.Convolution(3, (1, 2, 3)))
            == "Convolution(n_filters=3, input_shape=(1, 2, ch=3))")
    assert (repr(nengo.Convolution(3, (1, 2, 3), kernel_size=(3, 2)))
            == "Convolution(n_filters=3, input_shape=(1, 2, ch=3), "
               "kernel_size=(3, 2))")

    # repr uses the actual shape, str always shows shape with channels last
    conv = nengo.Convolution(3, (1, 2, 3), channels_last=False)
    assert repr(conv) == "Convolution(n_filters=3, input_shape=(ch=1, 2, 3))"


def test_channelshape_str():
    assert (repr(ChannelShape((1, 2, 3)))
            == "ChannelShape(shape=(1, 2, 3), channels_last=True)")
    assert (repr(ChannelShape((1, 2, 3), channels_last=False))
            == "ChannelShape(shape=(1, 2, 3), channels_last=False)")

    # `str` always has channels last
    assert str(ChannelShape((1, 2, 3))) == "(1, 2, ch=3)"
    assert str(ChannelShape((1, 2, 3), channels_last=False)) == "(ch=1, 2, 3)"


def test_channelshapeparam_last_persists():
    class Test:
        shape = ChannelShapeParam("shape", readonly=False)

        def __init__(self, shape, channels_last=None):
            self.shape = ChannelShape.cast(shape, channels_last=channels_last)

    a = Test((2, 3, 4), channels_last=False)
    assert a.shape.channels_last == False
    assert a.shape.n_channels == 2

    a.shape = (1, 2, 3)
    assert a.shape.channels_last == False
    assert a.shape.n_channels == 1

    a.shape = ChannelShape((1, 2, 3), channels_last=True)
    assert a.shape.channels_last == True
    assert a.shape.n_channels == 3
