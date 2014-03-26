import numpy as np

import nengo
from .ensemblearray import EnsembleArray


def circconv(a, b, invert_a=False, invert_b=False, axis=-1):
    """A reference Numpy implementation of circular convolution"""
    A = np.fft.fft(a, axis=axis)
    B = np.fft.fft(b, axis=axis)
    if invert_a:
        A = A.conj()
    if invert_b:
        B = B.conj()
    return np.fft.ifft(A * B, axis=axis).real

_dft_half_cache = {}


def _dft_half_cached(n):
    if n not in _dft_half_cache:
        x = np.arange(n)
        w = np.arange(n // 2 + 1)
        D = ((1. / np.sqrt(n))
             * np.exp((-2.j * np.pi / n) * (w[:, None] * x[None, :])))
        _dft_half_cache[n] = D
    return _dft_half_cache[n]


class CircularConvolution(nengo.Network):
    """
    CircularConvolution docs XXX
    """

    def make(self, neurons, dimensions, radius=1,
             invert_a=False, invert_b=False):
        self.transformA = self._input_transform(
            dimensions, first=True, invert=invert_a)
        self.transformB = self._input_transform(
            dimensions, first=False, invert=invert_b)
        self.transformC = self._output_transform(dimensions)

        self.A = nengo.Node(size_in=dimensions)
        self.B = nengo.Node(size_in=dimensions)
        self.ensemble = EnsembleArray(neurons,
                                      self.transformC.shape[1],
                                      dimensions=2,
                                      radius=radius)
        self.output = nengo.Node(size_in=dimensions)

        for ens in self.ensemble.ensembles:
            if not isinstance(neurons, nengo.Direct):
                ens.encoders = np.tile(
                    [[1, 1], [-1, 1], [1, -1], [-1, -1]],
                    (ens.n_neurons // 4, 1))
        nengo.Connection(
            self.A, self.ensemble.input, transform=self.transformA)
        nengo.Connection(
            self.B, self.ensemble.input, transform=self.transformB)
        nengo.Connection(self.ensemble.add_output('product', self.product),
                         self.output,
                         filter=0.02,
                         transform=self.transformC)

    @staticmethod
    def _input_transform(dims, first, invert=False):
        dims2 = 4 * (dims // 2 + 1)
        T = np.zeros((dims2, 2, dims))
        dft = _dft_half_cached(dims)

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

    @staticmethod
    def _output_transform(dims):
        dims2 = (dims // 2 + 1)
        T = np.zeros((dims2, 4, dims))
        idft = _dft_half_cached(dims).conj()

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

    @staticmethod
    def product(x):
        return x[0] * x[1]
