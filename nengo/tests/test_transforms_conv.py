import numpy as np
import pytest

import nengo
from nengo._vendor.npconv2d import conv2d
from nengo.exceptions import BuildError, ValidationError


@pytest.mark.parametrize("x_mul", (1, 2, 3, 4))
@pytest.mark.parametrize("k_size", (1, 2, 3, 4))
@pytest.mark.parametrize("stride", (1, 2, 3, 4))
@pytest.mark.parametrize("padding", ("same", "valid"))
def test_convolution_shape(padding, stride, k_size, x_mul, rng, allclose):
    tf = pytest.importorskip("tensorflow")

    in_channels = 2
    out_channels = 3

    for i in range(2 * k_size):
        x_size = k_size + stride * (x_mul - 1) + i
        x_shape = (x_size, x_size, in_channels)
        k_shape = (k_size, k_size, in_channels, out_channels)

        x = rng.uniform(-1, 1, size=x_shape)
        kernel = rng.uniform(-1, 1, size=k_shape)
        y_tf = tf.nn.conv2d(
            x[None, ...], kernel, stride, padding=padding.upper()
        ).numpy()[0]

        y_np = conv2d.conv2d(
            x[None, ...], kernel, pad=padding.upper(), stride=(stride, stride)
        )[0]

        transform = nengo.Convolution(
            out_channels,
            x_shape,
            kernel_size=(k_size, k_size),
            strides=(stride, stride),
            padding=padding,
        )

        assert transform.output_shape.shape == y_tf.shape
        assert y_np.shape == y_tf.shape
        assert allclose(y_np, y_tf)


@pytest.mark.parametrize("dimensions", (1, 2))
@pytest.mark.parametrize("padding", ("same", "valid"))
@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("fixed_kernel", (True, False))
@pytest.mark.parametrize("transpose", (True, False))
def test_convolution(
    dimensions,
    padding,
    channels_last,
    fixed_kernel,
    transpose,
    Simulator,
    allclose,
    rng,
    seed,
):
    input_d = 4
    input_channels = 2
    output_channels = 5
    kernel_d = 3
    output_d = (
        input_d
        if padding == "same"
        else input_d + (1 if transpose else -1) * (kernel_d // 2 * 2)
    )

    kernel_size = (kernel_d,) if dimensions == 1 else (kernel_d, kernel_d)
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

    x = rng.randn(*input_shape)
    w = rng.randn(*kernel_shape) if fixed_kernel else nengo.dists.Uniform(-0.1, 0.1)

    if transpose:
        transform = nengo.transforms.ConvolutionTranspose(
            output_channels,
            input_shape,
            init=w,
            padding=padding,
            kernel_size=kernel_size,
            strides=(1,) if dimensions == 1 else (1, 1),
            channels_last=channels_last,
        )
    else:
        transform = nengo.Convolution(
            output_channels,
            input_shape,
            init=w,
            padding=padding,
            kernel_size=kernel_size,
            strides=(1,) if dimensions == 1 else (1, 1),
            channels_last=channels_last,
        )

    assert transform.output_shape.shape == output_shape

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(np.ravel(x))
        b = nengo.Node(size_in=np.prod(output_shape))
        conn = nengo.Connection(a, b, synapse=None, transform=transform)
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

    if transpose:
        outsize = (output_d, 1) if dimensions == 1 else (output_d, output_d)
        truth = conv2d.conv2d_gradx(
            weights, x[None, ...], xsize=outsize, pad=padding.upper()
        )[0]
    else:
        truth = conv2d.conv2d(x[None, ...], weights, pad=padding.upper())[0]

    if not channels_last:
        truth = np.moveaxis(truth, -1, 0)

    assert allclose(sim.data[p][0], np.ravel(truth))


@pytest.mark.parametrize("encoders", (True, False))
@pytest.mark.parametrize("decoders", (True, False))
def test_convolution_nef(encoders, decoders, Simulator):
    with nengo.Network() as net:
        transform = nengo.transforms.Convolution(n_filters=2, input_shape=(3, 3, 1))
        a = nengo.Ensemble(9, 9)
        b = nengo.Ensemble(2, 2)
        nengo.Connection(
            a if decoders else a.neurons,
            b if encoders else b.neurons,
            transform=transform,
        )

    if decoders:
        # error if decoders
        with pytest.raises(BuildError, match="decoded connection"):
            with Simulator(net):
                pass
    else:
        # no error
        with Simulator(net):
            pass


def test_convolution_validation_errors():
    # conflicting channels_last
    input_shape = nengo.transforms.ChannelShape((2, 3, 4), channels_last=True)
    with pytest.raises(ValidationError, match="transform has channels_l.*input shape"):
        nengo.Convolution(4, input_shape, channels_last=False)

    # kernel_size does not match dimensions (2)
    with pytest.raises(ValidationError, match=r"Kernel dimensions \(3\) does not mat"):
        nengo.Convolution(4, input_shape, kernel_size=(3, 3, 3))

    # strides does not match dimensions (2)
    with pytest.raises(ValidationError, match=r"Stride dimensions \(3\) does not mat"):
        nengo.Convolution(4, input_shape, strides=(1, 1, 1))

    # init shape does not match kernel shape
    input_shape = nengo.transforms.ChannelShape((5, 5, 4), channels_last=True)
    nengo.Convolution(4, input_shape, init=np.ones((3, 3, 4, 4)))  # this works
    with pytest.raises(ValidationError, match=r"Kernel shape \(9, 9, 4, 4\).*not mat"):
        nengo.Convolution(4, input_shape, init=np.ones((9, 9, 4, 4)))
    with pytest.raises(ValidationError, match=r"Kernel shape \(3, 3, 7, 4\).*not mat"):
        nengo.Convolution(4, input_shape, init=np.ones((3, 3, 7, 4)))
    with pytest.raises(ValidationError, match=r"Kernel shape \(3, 3, 4, 5\).*not mat"):
        nengo.Convolution(4, input_shape, init=np.ones((3, 3, 4, 5)))

    # test empty output
    with pytest.raises(ValidationError, match="exceeds the spatial size"):
        nengo.transforms.Convolution(n_filters=2, input_shape=(3, 2, 1))

    # valid output shape
    nengo.transforms.ConvolutionTranspose(
        n_filters=2, input_shape=(3, 2, 1), output_shape=(5, 4, 2)
    )
    with pytest.raises(ValidationError, match="number of dimensions"):
        # too many dims in output shape
        nengo.transforms.ConvolutionTranspose(
            n_filters=2, input_shape=(3, 2, 1), output_shape=(5, 4, 2, 1)
        )
    with pytest.raises(ValidationError, match="number of channels"):
        # too many channels in output shape
        nengo.transforms.ConvolutionTranspose(
            n_filters=2, input_shape=(3, 2, 1), output_shape=(5, 4, 3)
        )
    with pytest.raises(ValidationError, match="not a valid output shape"):
        # too many rows in output shape
        nengo.transforms.ConvolutionTranspose(
            n_filters=2, input_shape=(3, 2, 1), output_shape=(6, 4, 2)
        )
