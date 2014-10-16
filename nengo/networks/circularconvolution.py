import numpy as np

import nengo
from nengo.networks.product import Product
from nengo.utils.compat import range
from nengo.utils.magic import memoize


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
    """Compute the circular convolution of two vectors.

    The circular convolution `c` of vectors `a` and `b` is given by

        c[i] = sum_j a[j] * b[i - j]

    where the indices on `b` are assumed to wrap around as required.

    This computation can also be done in the Fourier domain,

        c = DFT^{-1}( DFT(a) * DFT(b) )

    where `DFT` is the Discrete Fourier Transform operator, and
    `DFT^{-1}` is its inverse. This network uses this method.

    Parameters
    ----------
    neurons : Neurons
        A Neurons object, defining both the number of neurons per
        dimension and the neuron model.
    dimensions : int
        The number of dimensions of the input and output vectors.
    invert_a, invert_b : bool
        Whether to reverse the order of elements in either
        the first input (`invert_a`) or the second input (`invert_b`).
        Flipping the second input will make the network perform circular
        correlation instead of circular convolution.

    Examples
    --------

        A = EnsembleArray(nengo.LIF(50), 10)
        B = EnsembleArray(nengo.LIF(50), 10)
        C = EnsembleArray(nengo.LIF(50), 10)
        cconv = nengo.networks.CircularConvolution(nengo.LIF(50), 10)

        nengo.Connection(A.output, cconv.A)
        nengo.Connection(B.output, cconv.B)
        nengo.Connection(cconv.output, C.input)

    Notes
    -----
    The network maps the input vectors `a` and `b` of length N into
    the Fourier domain and aligns them for complex multiplication.
    Letting `F = DFT(a)` and `G = DFT(b)`, this is given by:

        [ F[i].real ]     [ G[i].real ]     [ w[i] ]
        [ F[i].imag ]  *  [ G[i].imag ]  =  [ x[i] ]
        [ F[i].real ]     [ G[i].imag ]     [ y[i] ]
        [ F[i].imag ]     [ G[i].real ]     [ z[i] ]

    where `i` only ranges over the lower half of the spectrum, since
    the upper half of the spectrum is the flipped complex conjugate of
    the lower half, and therefore redundant. The input transforms are
    used to perform the DFT on the inputs and align them correctly for
    complex multiplication.

    The complex product `H = F * G` is then

        H[i] = (w[i] - x[i]) + (y[i] + z[i]) * I

    where `I = sqrt(-1)`. We can perform this addition along with the
    inverse DFT `c = DFT^{-1}(H)` in a single output transform, finding
    only the real part of `c` since the imaginary part is analytically zero.
    """

    def __init__(self, n_neurons, dimensions, invert_a=False, invert_b=False,
                 radius=1, encoders=nengo.Default, **ens_kwargs):
        self.dimensions = dimensions
        self.invert_a = invert_a
        self.invert_b = invert_b

        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.product = Product(n_neurons,
                               self.transform_out.shape[1],
                               radius=radius,
                               encoders=encoders,
                               label="conv",
                               **ens_kwargs)
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
        D = self.dft_half(dims).conj().T  # inverse DFT

        # scale middle columns to simulate the full DFT matrix
        i2 = (dims + 1) // 2
        D[:, 1:i2] *= 2

        T = np.zeros((dims, 3 * dims2))
        for i in xrange(dims):
            # Real part = (k1 - k3) * D.real - (k1 + k2) * D.imag
            Di = D[i]
            T[i, 0::3] = 2*D.real[i] - 2*D.imag[i]  # k1 * (D.real - D.imag)
            T[i, 1::3] = -2*D.imag[i]               # -k2 * Di.imag
            T[i, 2::3] = -2*D.real[i]               # -k3 * Di.real

        # scaling is needed since we have 1./sqrt(dims) in DFT
        T *= np.sqrt(dims)

        T = CircularConvolution._remove_imag_rows(T.T).T
        return T

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

        dims2 = dims // 2 + 1
        D = CircularConvolution.dft_half(dims)
        D = D.conj() if invert else D

        T = np.zeros((3 * dims2, dims))
        for i in xrange(dims2):
            Ti = T[3*i:3*(i+1)]
            if align == 'A':
                Ti[0] = 0.5 * (D.real[i] + D.imag[i])  # (a + b)
                Ti[1] = D.real[i]              # a
                Ti[2] = D.imag[i]              # b
            else:
                Ti[0] = D.real[i]              # c
                Ti[1] = 0.5 * (D.imag[i] - D.real[i])  # (d - c)
                Ti[2] = 0.5 * (D.real[i] + D.imag[i])  # (c + d)

        T = CircularConvolution._remove_imag_rows(T)
        return T

    @staticmethod
    def _remove_imag_rows(T):
        """Throw away imaginary row we don't need (since they're zero)"""
        i = np.arange(T.shape[0])
        if T.shape[1] % 2 == 0:
            return T[(i == 0) | (i > 2) & (i < len(i) - 2)]
        else:
            return T[(i == 0) | (i > 2)]
