import numpy as np

from nengo.builder import Operator, Builder, Signal
from nengo.builder.operator import Reset, ElementwiseInc, DotInc
from nengo.exceptions import BuildError
from nengo.transforms import Dense, Convolution
from nengo._vendor.npconv2d import conv2d


def multiply(x, y):
    if x.ndim <= 2 and y.ndim < 2:
        return x * y
    elif x.ndim < 2 and y.ndim == 2:
        return x.reshape(-1, 1) * y
    elif x.ndim == 2 and y.ndim == 2:
        return np.dot(x, y)
    else:
        raise BuildError(
            "Tensors not supported (x.ndim=%d, y.ndim=%d)" % (x.ndim, y.ndim))


@Builder.register(Dense)
def build_dense(model, transform, sig_in,
                decoders=None, encoders=None, rng=np.random):
    weights = transform.sample(rng=rng)

    if decoders is not None:
        weights = multiply(weights, decoders)
    if encoders is not None:
        weights = multiply(encoders.T, weights)

    # Add operator for applying weights
    weight_sig = Signal(weights, name="%s.weights" % transform, readonly=True)
    weighted = Signal(
        np.zeros(transform.size_out if encoders is None else weights.shape[0]),
        name="%s.weighted" % transform)
    model.add_op(Reset(weighted))

    op = ElementwiseInc if weights.ndim < 2 else DotInc
    model.add_op(op(weight_sig, sig_in, weighted,
                    tag="%s.weights_elementwiseinc" % transform))

    return weighted, weight_sig


@Builder.register(Convolution)
def build_convolution(model, transform, sig_in,
                      decoders=None, encoders=None, rng=np.random):
    if decoders is not None:
        raise BuildError("Applying a convolution transform to a decoded "
                         "connection is not supported")
    if encoders is not None:
        raise BuildError(
            "Applying encoders to a convolution transform is not supported")

    weights = transform.sample(rng=rng)
    weight_sig = Signal(weights, name="%s.weights" % transform, readonly=True)
    weighted = Signal(
        np.zeros(transform.size_out), name="%s.weighted" % transform)
    model.add_op(Reset(weighted))

    model.add_op(ConvInc(weight_sig, sig_in, weighted, transform,
                         tag="%s.weights_convinc" % transform))

    return weighted, weight_sig


class ConvInc(Operator):
    def __init__(self, W, X, Y, conv, tag=None):
        super(ConvInc, self).__init__(tag=tag)

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

    def _descstr(self):
        return 'conv2d(%s, %s) -> %s' % (self.W, self.X, self.Y)

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
