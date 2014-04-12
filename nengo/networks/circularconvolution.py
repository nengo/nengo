import numpy as np

import nengo
from nengo.networks.product import Product
from nengo.utils.decorators import memoize


def circconv(a, b, invert_a=False, invert_b=False, axis=-1):
    """A reference Numpy implementation of circular convolution"""
    A = np.fft.fft(a, axis=axis)
    B = np.fft.fft(b, axis=axis)
    if invert_a:
        A = A.conj()
    if invert_b:
        B = B.conj()
    return np.fft.ifft(A * B, axis=axis).real


class CircularConvolution(nengo.Network):
    """CircularConvolution docs XXX"""

    def __init__(self, neurons, dimensions, radius=1,
                 invert_a=False, invert_b=False):
        self.transformA = self._input_transform(
            dimensions, first=True, invert=invert_a)
        self.transformB = self._input_transform(
            dimensions, first=False, invert=invert_b)
        self.transformC = self._output_transform(dimensions)

        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.product = Product(neurons, self.transformC.shape[1], label="conv")
        self.output = nengo.Node(size_in=dimensions, label="output")

        # Connect into product.product.input rather than product.A and
        # product.B in order to avoid writing the transforms in terms ot the
        # partioned dimensions.
        nengo.Connection(
            self.A, self.product.product.input, transform=self.transformA,
            synapse=None)
        nengo.Connection(
            self.B, self.product.product.input, transform=self.transformB,
            synapse=None)
        nengo.Connection(
            self.product.output, self.output, transform=self.transformC,
            synapse=None)

    @classmethod
    def _input_transform(cls, dims, first, invert=False):
        dims2 = 4 * (dims // 2 + 1)
        T = np.zeros((dims2, 2, dims))
        dft = cls.dft_half(dims)

        for i in range(dims2):
            row = dft[i // 4] if not invert else dft[i // 4].conj()
            if first:
                T[i, 0] = row.real if i % 2 == 0 else row.imag
            else:
                T[i, 1] = row.real if i % 4 == 0 or i % 4 == 3 else row.imag

        # --- Throw away rows that we don't need (b/c they're zero)
        i = np.arange(dims2)
        if dims % 2 == 0:
            T = T[(i == 0) | (i > 3) & (i < len(i) - 3)]
        else:
            T = T[(i == 0) | (i > 3)]

        return T.reshape((-1, dims))

    @classmethod
    def _output_transform(cls, dims):
        dims2 = (dims // 2 + 1)
        T = np.zeros((dims2, 4, dims))
        idft = cls.dft_half(dims).conj()

        for i in range(dims2):
            row = idft[i] if i == 0 or 2*i == dims else 2*idft[i]
            T[i, 0] = row.real
            T[i, 1] = -row.real
            T[i, 2] = -row.imag
            T[i, 3] = -row.imag

        T = T.reshape(4*dims2, dims)

        # --- Throw away rows that we don't need (b/c they're zero)
        i = np.arange(4*dims2)
        if dims % 2 == 0:
            T = T[(i == 0) | (i > 3) & (i < len(i) - 3)]
        else:
            T = T[(i == 0) | (i > 3)]

        # scaling is needed since we have 1./sqrt(dims) in DFT
        T *= np.sqrt(dims)

        return T.T

    @classmethod
    @memoize
    def dft_half(cls, n):
        x = np.arange(n)
        w = np.arange(n // 2 + 1)
        return ((1. / np.sqrt(n)) *
                np.exp((-2.j * np.pi / n) * (w[:, None] * x[None, :])))
