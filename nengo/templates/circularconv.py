"""
circularconv.py: provides CircularConvolution template
"""

import numpy as np

from ..core import Decoder, Encoder, Filter, Signal, Transform
from ..core import Direct, LIF

def circconv(a, b, invert_a=False, invert_b=False, axis=-1):
    """A reference Numpy implementation of circular convolution"""
    A = np.fft.fft(a, axis=axis)
    B = np.fft.fft(b, axis=axis)
    if invert_a: A = A.conj()
    if invert_b: B = B.conj()
    return np.fft.ifft(A * B, axis=axis).real

_dft_half_cache = {}
def _dft_half_cached(n):
    if n not in _dft_half_cache:
        x = np.arange(n)
        w = np.arange(n/2+1)
        D = (1./np.sqrt(n))*np.exp((-2.j*np.pi/n)*(w[:,None]*x[None,:]))
        _dft_half_cache[n] = D
    return _dft_half_cache[n]

def _input_transform(dims, first, invert=False):
    dims2 = 4*(dims/2+1)
    T = np.zeros((dims2, 2, dims))
    dft = _dft_half_cached(dims)

    for i in xrange(dims2):
        row = dft[i/4] if not invert else dft[i/4].conj()
        if first:
            T[i,0] = row.real if i % 2 == 0 else row.imag
        else:
            T[i,1] = row.real if i % 4 == 0 or i % 4 == 3 else row.imag

    ### Throw away rows that we don't need (b/c they're zero)
    i = np.arange(dims2)
    if dims % 2 == 0: T = T[(i == 0) | (i > 3) & (i < len(i) - 3)]
    else:             T = T[(i == 0) | (i > 3)]

    return T

def _output_transform(dims):
    dims2 = (dims/2+1)
    T = np.zeros((dims2, 4, dims))
    idft = _dft_half_cached(dims).conj()

    for i in xrange(dims2):
        row = idft[i] if i == 0 or 2*i == dims else 2*idft[i]
        T[i,0] = row.real
        T[i,1] = -row.real
        T[i,2] = -row.imag
        T[i,3] = -row.imag

    T = T.reshape(4*dims2, dims)

    ### Throw away rows that we don't need (b/c they're zero)
    i = np.arange(4*dims2)
    if dims % 2 == 0: T = T[(i == 0) | (i > 3) & (i < len(i) - 3)]
    else:             T = T[(i == 0) | (i > 3)]

    T *= np.sqrt(dims) # scaling is needed since we have 1./sqrt(dims) in DFT

    return T.T

class DirectCircularConvolution(object):
    def __init__(self, model, A, B, C, invert_a=False, invert_b=False):

        # assert A.dimensions == B.dimensions
        # dims = A.dimensions
        dims = len(A)
        assert len(A) == len(B) and len(B) == len(C)

        TinA = _input_transform(dims, first=True, invert=invert_a)
        TinB = _input_transform(dims, first=False, invert=invert_b)
        Tout = _output_transform(dims)
        dims2 = TinA.shape[0]

        def fn(x):
            x = x.reshape((dims2, 2))
            return (x[:,0] * x[:,1]).reshape((-1,1))
        TinA = TinA.reshape(2*dims2, dims)
        TinB = TinB.reshape(2*dims2, dims)

        D = model.add(Direct(n_in=2*dims2, n_out=dims2, fn=fn))
        encA = model.add(Encoder(A, D, TinA))
        encB = model.add(Encoder(B, D, TinB))
        dec = model.add(Decoder(D, C, Tout))
        tf = model.add(Transform(1.0, C, C))


def weights(n_in, n_out):
    return np.random.randn(n_in, n_out)

class CircularConvolution(object):
    """
    CircularConvolution docs XXX
    """

    def __init__(self, model, A, B,
            neurons_per_product=128,
            neuron_type=None):

        if A.shape != B.shape:
            raise ValueError()
        if neuron_type != None:
            raise NotImplementedError()

        if len(A) % 2:
            raise NotImplementedError()

        D = len(A)

        # XXX compute correct matrices here
        fourier_matrix = np.random.randn(D, D)
        inverse_fourier_matrix = np.random.randn(D, D * 2)

        self.A = A
        self.B = B

        # -- the output signals
        self.out = model.add(Signal(D))

        # -- Fourier transforms of A, B, C
        self.A_Fourier = model.add(Signal(D)).reshape(D//2, 2)
        self.B_Fourier = model.add(Signal(D)).reshape(D//2, 2)
        self.AB_prods = model.add(Signal(D * 2)).reshape(D//2, 4)

        # -- compute the Fourier transform of A and B
        #    as Filters.
        #    N.B. delayed by one time step.
        model.add(Filter(fourier_matrix, A, self.A_Fourier.base))
        model.add(Filter(fourier_matrix, B, self.B_Fourier.base))

        # -- compute the complex elementwise product of A, B
        #    in Fourier domain
        self.pops = []
        for ii in range(D//2):
            # TODO: use neuron_type
            AA = model.add(LIF(neurons_per_product))
            AB = model.add(LIF(neurons_per_product))
            BA = model.add(LIF(neurons_per_product))
            BB = model.add(LIF(neurons_per_product))
            n_in = AA.n_in
            n_out = self.A_Fourier[ii, 0:1].size
            model.add(Encoder(self.A_Fourier[ii, 0:1], AA, weights(n_in,n_out)))
            model.add(Encoder(self.A_Fourier[ii, 1:2], AA, weights(n_in,n_out)))
            model.add(Encoder(self.A_Fourier[ii, 0:1], AB, weights(n_in,n_out)))
            model.add(Encoder(self.B_Fourier[ii, 1:2], AB, weights(n_in,n_out)))
            model.add(Encoder(self.B_Fourier[ii, 0:1], BA, weights(n_in,n_out)))
            model.add(Encoder(self.A_Fourier[ii, 1:2], BA, weights(n_in,n_out)))
            model.add(Encoder(self.B_Fourier[ii, 0:1], BB, weights(n_in,n_out)))
            model.add(Encoder(self.B_Fourier[ii, 1:2], BB, weights(n_in,n_out)))

            model.add(Decoder(AA, self.AB_prods[ii, 0:1], weights(n_out,n_in)))
            model.add(Decoder(AB, self.AB_prods[ii, 1:2], weights(n_out,n_in)))
            model.add(Decoder(BA, self.AB_prods[ii, 2:3], weights(n_out,n_in)))
            model.add(Decoder(BB, self.AB_prods[ii, 3:4], weights(n_out,n_in)))

        model.add(Transform(inverse_fourier_matrix,
                            self.AB_prods.base,
                            self.out.base))
