import numpy as np
import pytest

from nengo.exceptions import BuildError
from nengo.transforms import Convolution
from nengo.builder.signal import Signal
from nengo.builder.transforms import ConvInc, multiply


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
        channels_last, stride0, stride1, kernel0, kernel1, padding, rng):
    correlate2d = pytest.importorskip("scipy.signal").correlate2d

    shape0 = 16
    shape1 = 17
    in_channels = 32
    out_channels = 64
    x_shape = (shape0, shape1, in_channels) if channels_last else (
        in_channels, shape0, shape1)
    x = Signal(rng.randn(*x_shape))
    w = Signal(rng.randn(kernel0, kernel1, in_channels, out_channels))

    conv = Convolution(out_channels,
                       x_shape,
                       kernel_size=(kernel0, kernel1),
                       strides=(stride0, stride1),
                       padding=padding,
                       channels_last=channels_last)

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
            (padding - 1) * strides + (kernel0, kernel1) - (shape0, shape1),
            0).astype(np.int64)
        x0 = np.pad(x0, [
            (padding[0] // 2, padding[0] - padding[0] // 2),
            (padding[1] // 2, padding[1] - padding[1] // 2),
            (0, 0),
        ], "constant")

    y0 = np.stack([
        np.sum([
            correlate2d(x0[..., j], w.initial_value[..., j, i], mode="valid")
            for j in range(in_channels)
        ], axis=0) for i in range(out_channels)
    ], axis=-1)
    y0 = y0[::stride0, ::stride1, :]
    if not channels_last:
        y0 = np.moveaxis(y0, -1, 0)

    assert np.allclose(signals[y], y0)
