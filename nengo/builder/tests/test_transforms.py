import numpy as np
import pytest

from nengo.builder.signal import Signal
from nengo.builder.tests.test_operator import _test_operator_arg_attributes
from nengo.builder.transforms import ConvInc, ConvTransposeInc, multiply
from nengo.exceptions import BuildError
from nengo.transforms import Convolution, ConvolutionTranspose


def test_multiply():
    sca = np.array(4)
    vec = np.array([2, 3])
    mat = np.array([[1, 2], [3, 4]])

    assert np.array_equal(multiply(sca, sca), sca * sca)
    assert np.array_equal(multiply(sca, vec), sca * vec)
    assert np.array_equal(multiply(vec, sca), sca * vec)
    assert np.array_equal(multiply(sca, mat), sca * mat)
    assert np.array_equal(multiply(mat, sca), sca * mat)

    assert np.array_equal(multiply(vec, vec), vec * vec)
    assert np.array_equal(multiply(vec, mat), np.diag(vec).dot(mat))
    assert np.array_equal(multiply(mat, vec), mat.dot(np.diag(vec)))
    assert np.array_equal(multiply(mat, mat), mat.dot(mat))

    with pytest.raises(BuildError):
        ary3 = np.ones((2, 2, 2))
        multiply(ary3, mat)


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
@pytest.mark.parametrize("stride0", (1, 2))
@pytest.mark.parametrize("stride1", (1, 2))
@pytest.mark.parametrize("kernel0", (4, 5))
@pytest.mark.parametrize("kernel1", (4, 5))
@pytest.mark.parametrize("padding", ("same", "valid"))
def test_convtransposeinc_2d(
    channels_last, stride0, stride1, kernel0, kernel1, padding, rng, allclose, plt
):
    """Test ConvTransposeInc by ensuring it is the transpose of ConvInc.

    Since convolution is a linear operator, it can be expressed as a matrix ``A``.
    We can therefore state that ``C.dot(A.dot(x)) == (A.T.dot(C.T)).T.dot(x)``,
    for an arbitrary vector ``x`` and arbitrary matrix ``C``.

    This test asserts this identity, and thereby tests the ``ConvTransposeInc`` operator
    against the ``ConvInc`` operator.
    """
    shape0 = 16
    shape1 = 17
    in_channels = 32
    out_channels = 64

    x_shape = (
        (shape0, shape1, in_channels)
        if channels_last
        else (in_channels, shape0, shape1)
    )

    conv = Convolution(
        out_channels,
        x_shape,
        kernel_size=(kernel0, kernel1),
        strides=(stride0, stride1),
        padding=padding,
        channels_last=channels_last,
    )

    nk = 10  # number of vectors we test ConvTransposeInc with
    C = rng.randn(nk, conv.output_shape.size)

    # compute ``conv_output = C.dot(A.dot(x))``, where ``A`` is the convolution operator
    x = Signal(rng.randn(*x_shape))
    w = Signal(rng.randn(kernel0, kernel1, in_channels, out_channels))
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
        kernel_size=(kernel0, kernel1),
        strides=(stride0, stride1),
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

    # Check whether the outputs match. If they don't, print/plot some diagnostic output.
    success = allclose(conv_transpose_output, conv_output)
    if not success:
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
                        plt.imshow(
                            expected_k, vmin=expected_k.min(), vmax=expected_k.max()
                        )
                        plt.title("expected out=%d ch=%d" % (i, k))

                        plt.subplot(122)
                        plt.imshow(
                            actual_k, vmin=expected_k.min(), vmax=expected_k.max()
                        )
                        plt.title("actual out=%d ch=%d" % (i, k))

                        assert success, "Output %d channel %d differs" % (i, k)

    assert success


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
