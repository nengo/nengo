import numpy as np

from nengo.builder import Operator
from nengo._vendor.npconv2d import conv2d


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
