import numpy as np

from nengo._vendor.npconv2d import conv2d
from nengo.builder.builder import Builder
from nengo.builder.operator import DotInc, ElementwiseInc, Operator, Reset, SparseDotInc
from nengo.builder.signal import Signal
from nengo.exceptions import BuildError
from nengo.rc import rc
from nengo.transforms import Convolution, Dense, NoTransform, Sparse


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

    model.add_op(
        ConvInc(
            weight_sig, sig_in, weighted, transform, tag=f"{transform}.apply_weights"
        )
    )

    return weighted, weight_sig


class ConvInc(Operator):
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
        The Convolution object being applied.
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
        The Convolution object being applied.
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

        self.conv = conv

        self.sets = []
        self.incs = [Y]
        self.reads = [W, X]
        self.updates = []

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
        return f"conv2d({self.W}, {self.X}) -> {self.Y}"

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

        # add empty batch dimension
        X = X[None, ...]

        def step_conv():
            Y[...] += conv2d.conv2d(X, W, pad=pad, stride=stride)[0]

        return step_conv


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
