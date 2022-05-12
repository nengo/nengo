import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import DotInc, ElementwiseInc, Operator, Reset, SparseDotInc
from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
from nengo.rc import rc
from nengo.transforms import (
    Convolution,
    ConvolutionTranspose,
    Dense,
    NoTransform,
    Sparse,
)
from nengo.utils.numpy import array_offset


def multiply(x, y):
    """Matrix-matrix multiply, interpreting vectors as diagonal matrices."""
    if x.ndim <= 2 and y.ndim < 2:
        return x * y
    elif x.ndim < 2 and y.ndim == 2:
        return x.reshape(-1, 1) * y
    elif x.ndim == 2 and y.ndim == 2:
        return np.dot(x, y)
    else:
        raise BuildError(f"Tensors not supported (x.ndim={x.ndim}, y.ndim={y.ndim})")


@Builder.register(Dense)
def build_dense(model, transform, sig_in, decoders=None, encoders=None, rng=np.random):
    """Build a `.Dense` transform object."""

    weights = transform.sample(rng=rng).astype(rc.float_dtype)

    if decoders is not None:
        weights = multiply(weights, decoders.astype(rc.float_dtype))
    if encoders is not None:
        weights = multiply(encoders.astype(rc.float_dtype).T, weights)

    # Add operator for applying weights
    weight_sig = Signal(weights, readonly=True, name=f"{transform}.weights")
    weighted = Signal(
        shape=transform.size_out if encoders is None else weights.shape[0],
        name=f"{transform}.weighted",
    )
    model.add_op(Reset(weighted))

    op = ElementwiseInc if weights.ndim < 2 else DotInc
    model.add_op(op(weight_sig, sig_in, weighted, tag=f"{transform}.apply_weights"))

    return weighted, weight_sig


@Builder.register(Sparse)
def build_sparse(model, transform, sig_in, decoders=None, encoders=None, rng=np.random):
    """Build a `.Sparse` transform object."""

    if decoders is not None:
        raise BuildError(
            "Applying a sparse transform to a decoded connection is not supported"
        )

    # Shouldn't be possible for encoders to be non-None, since that only
    # occurs for a connection solver with weights=True, and those can only
    # be applied to decoded connections (which are disallowed above)
    assert encoders is None

    # Add output signal
    weighted = Signal(shape=transform.size_out, name=f"{transform}.weighted")
    model.add_op(Reset(weighted))

    weights = transform.sample(rng=rng)
    assert weights.ndim == 2

    # Add operator for applying weights
    weight_sig = Signal(weights, name=f"{transform}.weights", readonly=True)
    model.add_op(
        SparseDotInc(weight_sig, sig_in, weighted, tag=f"{transform}.apply_weights")
    )

    return weighted, weight_sig


@Builder.register(Convolution)
@Builder.register(ConvolutionTranspose)
def build_convolution(
    model, transform, sig_in, decoders=None, encoders=None, rng=np.random
):
    """Build a `.Convolution` transform object."""

    if decoders is not None:
        raise BuildError(
            "Applying a convolution transform to a decoded "
            "connection is not supported"
        )

    # Shouldn't be possible for encoders to be non-None, since that only
    # occurs for a connection solver with weights=True, and those can only
    # be applied to decoded connections (which are disallowed above)
    assert encoders is None

    weights = transform.sample(rng=rng)
    weight_sig = Signal(weights, readonly=True, name=f"{transform}.weights")
    weighted = Signal(shape=transform.size_out, name=f"{transform}.weighted")
    model.add_op(Reset(weighted))

    op_class = (
        ConvTransposeInc if isinstance(transform, ConvolutionTranspose) else ConvInc
    )
    model.add_op(
        op_class(
            weight_sig, sig_in, weighted, transform, tag=f"{transform}.apply_weights"
        )
    )

    return weighted, weight_sig


def calc_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width.

    .. note:: Copied from MIT licensed https://github.com/renmengye/np-conv2d

    Arguments
    ---------
    pad : str or int
        Padding method, either "SAME", "VALID", or manually specified.
    ksize : int
        Kernel size [I, J].

    Returns
    -------
    pad_ : int
        Actual padding width.
    """
    if pad == "SAME":
        return max((out_siz - 1) * stride + ksize - in_siz, 0)
    elif pad == "VALID":
        return 0
    else:
        return pad


def calc_gradx_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width on a dilated image.

    .. note:: Copied from MIT licensed https://github.com/renmengye/np-conv2d

    Arguments
    --------
    pad : str or int
        Padding method, either "SAME", "VALID", or manually specified.
    ksize : int
        Kernel size [I, J].

    Returns
    -------
    pad_ : int
        Actual padding width.
    """
    if pad == "SAME":
        out_siz_min = (in_siz - 1) * stride + 1
        p = out_siz + ksize - 1 - out_siz_min
        p = max(p, 0)
        p = min(p, (ksize - 1) * 2)
        return p
    elif pad == "VALID":
        return (ksize - 1) * 2
    else:
        return pad


def calc_size(h, kh, pad, sh):
    """Calculate output image size on one dimension.

    .. note:: Copied from MIT licensed https://github.com/renmengye/np-conv2d

    Arguments
    ---------
    h : int
        Input image size.
    kh : int
        Kernel size.
    pad : str or int
        Padding strategy.
    sh : int
        Stride.

    Returns
    -------
    s : int
        Output size.
    """

    if pad == "VALID":
        return np.ceil((h - kh + 1) / sh)
    elif pad == "SAME":
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))


def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """Converts a tensor to sliding windows.

    .. note:: Copied from MIT licensed https://github.com/renmengye/np-conv2d

    Arguments
    ---------
    x : np.array
        Input with shape [N, H, W, C]
    ksize : Tuple
        KH, KW
    pad : str or int
        Padding strategy or [PH, PW].
    stride : int
        Stride, [SH, SW].

    Returns
    -------
    y : np.array
        Sliding window: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
    """
    n, h, w, c = x.shape
    kh, kw = ksize
    sh, sw = stride

    h2 = int(calc_size(h, kh, pad, sh))
    w2 = int(calc_size(w, kw, pad, sw))
    ph = int(calc_pad(pad, h, h2, sh, kh))
    pw = int(calc_pad(pad, w, w2, sw, kw))

    ph0 = int(np.floor(ph / 2))
    ph1 = int(np.ceil(ph / 2))
    pw0 = int(np.floor(pw / 2))
    pw1 = int(np.ceil(pw / 2))

    if floor_first:
        pph = (ph0, ph1)
        ppw = (pw0, pw1)
    else:
        pph = (ph1, ph0)
        ppw = (pw1, pw0)
    x = np.pad(x, ((0, 0), pph, ppw, (0, 0)), mode="constant", constant_values=(0.0,))

    x_sn, x_sh, x_sw, x_sc = x.strides  # pylint: disable=unpacking-non-sequence
    y_strides = (x_sn, sh * x_sh, sw * x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray(
        (n, h2, w2, kh, kw, c),
        dtype=x.dtype,
        buffer=x.data,
        offset=array_offset(x),
        strides=y_strides,
    )
    return y


def extract_sliding_windows_gradx(x, ksize, pad, stride, orig_size, floor_first=False):
    """Extracts windows on a dilated image.

    .. note:: Copied from MIT licensed https://github.com/renmengye/np-conv2d

    Arguments
    ---------
    x : np.array
        Input with shape [N, H', W', C] (usually dy)
    ksize : Tuple
        KH, KW
    pad : str or int
        Padding strategy or [PH, PW].
    stride : int
        Stride, [SH, SW].
    orig_size : Tuple
        H, W

    Returns
    -------
    y : np.array
        Sliding window: [N, H, W, KH, KW, C]
    """
    n, h, w, c = x.shape
    kh, kw = ksize
    sh, sw = stride
    ph, pw = pad
    sh, sw = stride
    h2, w2 = orig_size

    xs = np.zeros([n, h, sh, w, sw, c])
    xs[:, :, 0, :, 0, :] = x
    xss = xs.shape
    x = xs.reshape([xss[0], xss[1] * xss[2], xss[3] * xss[4], xss[5]])
    x = x[:, :h2, :w2, :]

    ph2 = int(np.ceil(ph / 2))
    ph3 = int(np.floor(ph / 2))
    pw2 = int(np.ceil(pw / 2))
    pw3 = int(np.floor(pw / 2))
    if floor_first:
        pph = (ph3, ph2)
        ppw = (pw3, pw2)
    else:
        pph = (ph2, ph3)
        ppw = (pw2, pw3)
    x = np.pad(x, ((0, 0), pph, ppw, (0, 0)), mode="constant", constant_values=(0.0,))

    # The following code extracts window without copying the data:
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         y[:, ii, jj, :, :, :] = x[:, ii:ii + kh, jj:jj + kw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides  # pylint: disable=unpacking-non-sequence
    y_strides = (x_sn, x_sh, x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray(
        (n, h2, w2, kh, kw, c),
        dtype=x.dtype,
        buffer=x.data,
        offset=array_offset(x),
        strides=y_strides,
    )
    return y


def conv2d(x, w, pad="SAME", stride=(1, 1)):
    """2D convolution (technically speaking, correlation).

    .. note:: Copied from MIT licensed https://github.com/renmengye/np-conv2d

    Arguments
    ---------
    x : np.array
        Input with shape [N, H, W, C]
    w : np.array
        Weights with shape [N, H, W, C]
    pad : str or int
        Padding strategy or [PH, PW].
    stride : int
        Stride, [SH, SW].

    Returns
    -------
    y : np.array
        Convolved result with shape [N, H', W', K]
    """
    ksize = w.shape[:2]
    x = extract_sliding_windows(x, ksize, pad, stride)
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    y = x.dot(w)
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y


def conv2d_gradx(w, dy, xsize, pad="SAME", stride=(1, 1)):
    """2D convolution gradient wrt. input.

    Arguments
    ---------
    dy : np.array
        Input array with shape ``N, H', W', K``
    w : np.array
        Weights with shape ``I, J, K, C``.
    xsize : Tuple
        Original image size, ``[H, W]``.

    Returns
    -------
    dx : np.array
        Output array with shape ``N, H, W, C``
    """
    assert w.shape[-2] == dy.shape[-1]

    dys = dy.shape[1:3]
    ksize = w.shape[:2]
    pad2 = (
        calc_gradx_pad(pad, dys[0], xsize[0], stride[0], ksize[0]),
        calc_gradx_pad(pad, dys[1], xsize[1], stride[1], ksize[1]),
    )

    dx = extract_sliding_windows_gradx(dy, ksize, pad2, stride, xsize)
    dxs = dx.shape
    dx = dx.reshape([dxs[0] * dxs[1] * dxs[2], -1])
    w = w[::-1, ::-1, :, :]
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    dx = dx.dot(w)
    return dx.reshape([dxs[0], dxs[1], dxs[2], -1])


class GeneralConvInc(Operator):
    """Apply convolutional weights to input signal.

    .. versionadded:: 3.2.0

    Parameters
    ----------
    W : Signal
        The convolutional weights (a.k.a. the kernel).
    X : Signal
        The input signal.
    Y : Signal
        Output signal to be incremented.
    conv : `~nengo.Convolution` or `~nengo.ConvolutionTranspose`
        The Convolution or ConvolutionTranspose transform being applied.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    W : Signal
        The convolutional weights.
    X : Signal
        The input signal.
    Y : Signal
        Output signal to be incremented.
    conv : `~nengo.Convolution` or `~nengo.ConvolutionTranspose`
        The Convolution or ConvolutionTranspose transform being applied.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[Y]``
    3. reads ``[W, X]``
    4. updates ``[]``
    """

    def __init__(self, W, X, Y, conv, tag=None):
        super().__init__(tag=tag)

        assert isinstance(conv, (Convolution, ConvolutionTranspose))

        self.conv = conv

        self.sets = []
        self.incs = [Y]
        self.reads = [W, X]
        self.updates = []

    @property
    def is_transpose(self):
        return isinstance(self.conv, ConvolutionTranspose)

    @property
    def W(self):
        return self.reads[0]

    @property
    def X(self):
        return self.reads[1]

    @property
    def Y(self):
        return self.incs[0]

    @property
    def _descstr(self):
        name = "convtranspose2d" if self.is_transpose else "conv2d"
        return f"{name}({self.W}, {self.X}) -> {self.Y}"

    def make_step(self, signals, dt, rng):
        if self.conv.dimensions > 2:
            # note: we raise the error here, rather than earlier, because
            # other backends might support different convolutions
            raise NotImplementedError("Convolution > 2D not supported")

        W = signals[self.W]
        X = signals[self.X]
        Y = signals[self.Y]
        pad = self.conv.padding.upper()
        stride = self.conv.strides
        output_spatial_shape = (
            self.conv.output_shape.spatial_shape if self.is_transpose else None
        )

        X = X.reshape(self.conv.input_shape.shape)
        Y = Y.reshape(self.conv.output_shape.shape)

        if not self.conv.channels_last:
            X = np.moveaxis(X, 0, -1)
            Y = np.moveaxis(Y, 0, -1)

        if self.conv.dimensions == 1:
            # add extra dimension to make it a 2D convolution
            X = X[None, :, :]
            W = W[None, :, :, :]
            Y = Y[None, :, :]
            stride = (1,) + stride
            if output_spatial_shape is not None:
                output_spatial_shape = (1,) + output_spatial_shape

        # add empty batch dimension
        X = X[None, ...]

        if self.is_transpose:

            def step_conv_transpose():
                Y[...] += conv2d_gradx(
                    W, X, xsize=output_spatial_shape, pad=pad, stride=stride
                )[0]

            return step_conv_transpose

        else:

            def step_conv():
                Y[...] += conv2d(X, W, pad=pad, stride=stride)[0]

            return step_conv


class ConvInc(GeneralConvInc):
    """Apply convolutional weights to input signal.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    W : Signal
        The convolutional weights (a.k.a. the kernel).
    X : Signal
        The input signal.
    Y : Signal
        Output signal to be incremented.
    conv : `~nengo.Convolution`
        The Convolution transform being applied.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    W : Signal
        The convolutional weights.
    X : Signal
        The input signal.
    Y : Signal
        Output signal to be incremented.
    conv : `~nengo.Convolution`
        The Convolution transform being applied.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[Y]``
    3. reads ``[W, X]``
    4. updates ``[]``
    """

    def __init__(self, W, X, Y, conv, tag=None):
        super().__init__(W, X, Y, conv, tag=tag)
        assert not self.is_transpose


class ConvTransposeInc(GeneralConvInc):
    """Apply transposed convolutional weights to input signal.

    .. versionadded:: 3.2.0

    Parameters
    ----------
    W : Signal
        The convolutional weights (a.k.a. the kernel).
    X : Signal
        The input signal.
    Y : Signal
        Output signal to be incremented.
    conv : `~nengo.ConvolutionTranspose`
        The ConvolutionTranspose transform being applied.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    W : Signal
        The convolutional weights.
    X : Signal
        The input signal.
    Y : Signal
        Output signal to be incremented.
    conv : `~nengo.ConvolutionTranspose`
        The ConvolutionTranspose transform being applied.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[Y]``
    3. reads ``[W, X]``
    4. updates ``[]``
    """

    def __init__(self, W, X, Y, conv, tag=None):
        super().__init__(W, X, Y, conv, tag=tag)
        assert self.is_transpose


@Builder.register(NoTransform)
def build_no_transform(
    model, transform, sig_in, decoders=None, encoders=None, rng=np.random
):
    """Build a `.NoTransform` transform object."""

    if decoders is not None or encoders is not None:
        return build_dense(
            model,
            Dense(shape=(transform.size_out, transform.size_in), init=1.0),
            sig_in,
            decoders=decoders,
            encoders=encoders,
            rng=rng,
        )

    return sig_in, None
