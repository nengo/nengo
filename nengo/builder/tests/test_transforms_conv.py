import numpy as np
import pytest

from nengo._vendor.npconv2d import conv2d
from nengo.builder.signal import Signal
from nengo.builder.tests.test_operator import _test_operator_arg_attributes
from nengo.builder.transforms import ConvInc, ConvTransposeInc
from nengo.transforms import ChannelShape, Convolution, ConvolutionTranspose


@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("stride0", (1, 2))
@pytest.mark.parametrize("stride1", (1, 2))
@pytest.mark.parametrize("kernel0", (4, 5))
@pytest.mark.parametrize("kernel1", (4, 5))
@pytest.mark.parametrize("padding", ("same", "valid"))
def test_convinc_2d(
    channels_last, stride0, stride1, kernel0, kernel1, padding, rng, allclose
):
    correlate2d = pytest.importorskip("scipy.signal").correlate2d

    shape0 = 16
    shape1 = 17
    in_channels = 32
    out_channels = 64
    x_shape = (
        (shape0, shape1, in_channels)
        if channels_last
        else (in_channels, shape0, shape1)
    )
    x = Signal(rng.randn(*x_shape))
    w = Signal(rng.randn(kernel0, kernel1, in_channels, out_channels))

    conv = Convolution(
        out_channels,
        x_shape,
        kernel_size=(kernel0, kernel1),
        strides=(stride0, stride1),
        padding=padding,
        channels_last=channels_last,
    )

    y = Signal(np.zeros(conv.output_shape.shape))

    signals = {sig: np.array(sig.initial_value) for sig in (x, w, y)}
    step = ConvInc(w, x, y, conv).make_step(signals, None, None)

    step()

    x0 = x.initial_value

    if not channels_last:
        x0 = np.moveaxis(x0, 0, -1)

    if padding == "same":
        strides = np.asarray([stride0, stride1])
        padding = np.ceil(np.asarray([shape0, shape1]) / strides)
        padding = np.maximum(
            (padding - 1) * strides + (kernel0, kernel1) - (shape0, shape1), 0
        ).astype(np.int64)
        x0 = np.pad(
            x0,
            [
                (padding[0] // 2, padding[0] - padding[0] // 2),
                (padding[1] // 2, padding[1] - padding[1] // 2),
                (0, 0),
            ],
            "constant",
        )

    y0 = np.stack(
        [
            np.sum(
                [
                    correlate2d(x0[..., j], w.initial_value[..., j, i], mode="valid")
                    for j in range(in_channels)
                ],
                axis=0,
            )
            for i in range(out_channels)
        ],
        axis=-1,
    )
    y0 = y0[::stride0, ::stride1, :]
    if not channels_last:
        y0 = np.moveaxis(y0, -1, 0)

    assert allclose(signals[y], y0)


@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("strides", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("kernel_size", [(4, 5), (5, 4), (5, 5)])
@pytest.mark.parametrize("padding", ("same", "valid"))
def test_convtransposeinc_2d(
    channels_last, strides, kernel_size, padding, rng, allclose, plt
):
    """Test ConvTransposeInc by ensuring it is the transpose of ConvInc.

    Since convolution is a linear operator, it can be expressed as a matrix ``A``.
    We can therefore state that ``C.dot(A.dot(x)) == (A.T.dot(C.T)).T.dot(x)``,
    for an arbitrary vector ``x`` and arbitrary matrix ``C``.

    This test asserts this identity, and thereby tests the ``ConvTransposeInc`` operator
    against the ``ConvInc`` operator.
    """
    spatial_shape = (16, 17)
    in_channels = 32
    out_channels = 64

    x_shape = ChannelShape.from_space_and_channels(
        spatial_shape, in_channels, channels_last=channels_last
    )
    conv = Convolution(
        out_channels,
        x_shape,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        channels_last=channels_last,
    )

    nk = 10  # number of vectors we test ConvTransposeInc with
    C = rng.randn(nk, conv.output_shape.size)

    # compute ``conv_output = C.dot(A.dot(x))``, where ``A`` is the convolution operator
    x = Signal(rng.randn(*x_shape.shape))
    w = Signal(rng.randn(kernel_size[0], kernel_size[1], in_channels, out_channels))
    y = Signal(np.zeros(conv.output_shape.shape))

    signals = {sig: np.array(sig.initial_value) for sig in (x, w, y)}
    step_conv = ConvInc(w, x, y, conv).make_step(signals, None, None)
    step_conv()

    x_flat = signals[x].ravel()
    y_flat = signals[y].ravel()
    conv_output = C.dot(y_flat)

    # compute ``conv_transpose_output = (A.T.dot(C.T)).T.dot(x)``, where ``A.T`` is the
    # transpose convolution operator (applied to one column of ``C.T`` at a time)
    conv_transpose = ConvolutionTranspose(
        in_channels,
        conv.output_shape,
        output_shape=x_shape,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        channels_last=channels_last,
    )
    xt = Signal(np.zeros(conv_transpose.input_shape.shape))
    wt = Signal(np.transpose(w.initial_value, (0, 1, 3, 2)))
    yt = Signal(np.zeros(conv_transpose.output_shape.shape))

    signals_tr = {sig: np.array(sig.initial_value) for sig in (xt, wt, yt)}
    step_trans = ConvTransposeInc(wt, xt, yt, conv_transpose).make_step(
        signals_tr, None, None
    )

    AtCt = []
    for k in range(nk):
        signals_tr[xt][:] = C[k].reshape(conv_transpose.input_shape.shape)
        signals_tr[yt][:] = 0
        step_trans()
        AtCt.append(signals_tr[yt].copy().ravel())

    AtCt = np.array(AtCt)
    conv_transpose_output = AtCt.dot(x_flat)
    assert conv_transpose_output.shape == conv_output.shape

    success = allclose(conv_transpose_output, conv_output)
    if not success:
        debug_convtransposeinc_2d(w, wt, x, xt, y, yt, conv, conv_transpose, plt)
    assert success


def debug_convtransposeinc_2d(w, wt, x, xt, y, yt, conv, conv_transpose, plt):
    in_channels = w.shape[2]
    channels_last = conv.channels_last

    # transpose convolution should act as the derivative for forward convolution,
    # compute and compare to the derivative to see exactly where the difference is
    signals = {sig: np.array(sig.initial_value) for sig in (x, w, y)}
    step_conv = ConvInc(w, x, y, conv).make_step(signals, None, None)
    step_conv()

    dy_dx = []
    for j in range(conv.input_shape.size):
        xj = np.zeros(conv.input_shape.shape)
        xj.ravel()[j] = 1

        signals[x][:] = xj
        signals[y][:] = 0
        step_conv()

        dy_dx.append(signals[y].copy().ravel())

    dy_dx = np.column_stack(dy_dx)
    assert dy_dx.shape == (conv.output_shape.size, conv.input_shape.size)

    signals_tr = {sig: np.array(sig.initial_value) for sig in (xt, wt, yt)}
    step_trans = ConvTransposeInc(wt, xt, yt, conv_transpose).make_step(
        signals_tr, None, None
    )

    for i in range(conv.output_shape.size):
        xi = np.zeros(conv.output_shape.shape)
        xi.ravel()[i] = 1

        signals_tr[xt][:] = xi
        signals_tr[yt][:] = 0
        step_trans()

        actual = signals_tr[yt]
        expected = dy_dx[i, :].reshape(conv.input_shape.shape)

        assert actual.shape == expected.shape
        if not np.allclose(actual, expected):
            for k in range(in_channels):
                actual_k = actual[:, :, k] if channels_last else actual[k]
                expected_k = expected[:, :, k] if channels_last else expected[k]

                if not np.allclose(actual, expected):
                    plt.subplot(121)
                    plt.imshow(expected_k, vmin=expected_k.min(), vmax=expected_k.max())
                    plt.title(f"expected out={i} ch={k}")

                    plt.subplot(122)
                    plt.imshow(actual_k, vmin=expected_k.min(), vmax=expected_k.max())
                    plt.title(f"actual out={i} ch={k}")

                    assert False, f"Output {i} channel {k} differs"


@pytest.mark.parametrize("x_size", (1, 2, 3, 4))
@pytest.mark.parametrize("k_size", (1, 2, 3, 4))
@pytest.mark.parametrize("stride", (1, 2, 3, 4))
@pytest.mark.parametrize("padding", ("same", "valid"))
def test_conv2d_gradx(padding, stride, k_size, x_size, rng, plt, allclose):
    tf = pytest.importorskip("tensorflow")

    in_channels = 1
    out_channels = 1

    if padding == "same":
        y_min = (x_size - 1) * stride + 1
        y_max = x_size * stride
    else:
        y_min = (x_size - 1) * stride + k_size
        y_max = x_size * stride + k_size - 1

    bad_sizes = []
    for y_size in range(y_min, y_max + 1):
        x_shape = (1, x_size, x_size, in_channels)
        y_shape = (1, y_size, y_size, out_channels)
        k_shape = (k_size, k_size, out_channels, in_channels)

        x = rng.uniform(-1, 1, size=x_shape)
        kernel = rng.uniform(-1, 1, size=k_shape)
        y_tf = tf.nn.conv2d_transpose(
            x, kernel, y_shape, stride, padding=padding.upper()
        ).numpy()[0]

        kernel_t = np.transpose(kernel, (0, 1, 3, 2))
        y_np = conv2d.conv2d_gradx(
            kernel_t,
            x,
            xsize=y_shape[1:3],
            pad=padding.upper(),
            stride=(stride, stride),
        )[0]

        assert y_np.shape == y_tf.shape

        if not allclose(y_np, y_tf):
            plt.subplot(1, 2, 1)
            plt.imshow(y_tf[..., 0], vmin=y_tf.min(), vmax=y_tf.max())
            plt.subplot(1, 2, 2)
            plt.imshow(y_np[..., 0], vmin=y_tf.min(), vmax=y_tf.max())
            bad_sizes.append(y_size)

    assert len(bad_sizes) == 0, (y_min, y_max, bad_sizes)


def test_convinc_attrs_decstr():
    argnames = ["W", "X", "Y", "conv"]
    non_signals = ["conv"]

    conv = Convolution(4, (3, 5, 2))
    _, sim = _test_operator_arg_attributes(
        ConvInc, argnames, non_signals=non_signals, args={"conv": conv}
    )
    assert str(sim) == "ConvInc{conv2d(W, X) -> Y}"

    conv_transpose = ConvolutionTranspose(4, (3, 5, 2))
    _, sim = _test_operator_arg_attributes(
        ConvTransposeInc,
        argnames,
        non_signals=non_signals,
        args={"conv": conv_transpose},
    )
    assert str(sim) == "ConvTransposeInc{convtranspose2d(W, X) -> Y}"
