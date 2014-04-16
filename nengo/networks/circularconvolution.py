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

    def __init__(self, neurons, dimensions, invert_a=False, invert_b=False):
        self.dimensions = dimensions
        self.invert_a = invert_a
        self.invert_b = invert_b

        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.product = Product(
            neurons, self.transform_out.shape[1], label="conv")
        self.output = nengo.Node(size_in=dimensions, label="output")

        nengo.Connection(self.A, self.product.A,
                         transform=self.transformA, synapse=None)
        nengo.Connection(self.B, self.product.B,
                         transform=self.transformB, synapse=None)
        nengo.Connection(
            self.product.output, self.output, transform=self.transform_out,
            synapse=None)

    @property
    def transformA(self):
        return self._input_transform(self.dimensions, 'A', self.invert_a)

    @property
    def transformB(self):
        return self._input_transform(self.dimensions, 'B', self.invert_b)

    @property
    def transform_out(self):
        dims = self.dimensions
        dims2 = (dims // 2 + 1)
        tr = np.zeros((dims2, 4, dims))
        idft = self.dft_half(dims).conj()

        for i in range(dims2):
            row = idft[i] if i == 0 or 2*i == dims else 2*idft[i]
            tr[i, 0] = row.real
            tr[i, 1] = -row.real
            tr[i, 2] = -row.imag
            tr[i, 3] = -row.imag

        tr = tr.reshape(4*dims2, dims)
        self._remove_imag_rows(tr)
        # scaling is needed since we have 1./sqrt(dims) in DFT
        tr *= np.sqrt(dims)

        return tr.T

    @staticmethod
    @memoize
    def dft_half(n):
        x = np.arange(n)
        w = np.arange(n // 2 + 1)
        return ((1. / np.sqrt(n)) *
                np.exp((-2.j * np.pi / n) * (w[:, None] * x[None, :])))

    @staticmethod
    @memoize
    def _input_transform(dims, align, invert):
        """Create a transform to map the input into the Fourier domain.

        See the class docstring for more details.

        Parameters
        ----------
        dims : int
            Input dimensions.
        align : 'A' or 'B'
            How to align the real and imaginary components; the alignment
            depends on whether we're doing transformA or transformB.
        invert : bool
            Whether to reverse the order of elements.
        """
        if align not in ('A', 'B'):
            raise ValueError("'align' must be either 'A' or 'B'")

        dims2 = 4 * (dims // 2 + 1)
        tr = np.zeros((dims2, dims))
        dft = CircularConvolution.dft_half(dims)

        for i in range(dims2):
            row = dft[i // 4] if not invert else dft[i // 4].conj()
            if align == 'A':
                tr[i] = row.real if i % 2 == 0 else row.imag
            else:  # align == 'B'
                tr[i] = row.real if i % 4 == 0 or i % 4 == 3 else row.imag

        CircularConvolution._remove_imag_rows(tr)
        return tr.reshape((-1, dims))

    @staticmethod
    def _remove_imag_rows(tr):
        """Throw away imaginary row we don't need (since they're zero)"""
        i = np.arange(tr.shape[0])
        if tr.shape[1] % 2 == 0:
            tr = tr[(i == 0) | (i > 3) & (i < len(i) - 3)]
        else:
            tr = tr[(i == 0) | (i > 3)]
