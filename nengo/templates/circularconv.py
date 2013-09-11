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


class CircularConvolution(object):
    """
    CircularConvolution docs XXX
    """

    def __init__(self, model, A, B, C, neurons=None, radius=1, pstc=0.005,
                 invert_a=False, invert_b=False, name=None):
        from ..objects import Ensemble
        from ..templates import EnsembleArray

        dims = A.dimensions
        assert A.dimensions == B.dimensions and B.dimensions == C.dimensions

        transform_inA = _input_transform(dims, first=True, invert=invert_a)
        transform_inB = _input_transform(dims, first=False, invert=invert_b)
        transform_out = _output_transform(dims)
        dims2 = transform_inA.shape[0]

        if name is None:
            name = "D"
        D = model.add(EnsembleArray(
                name, neurons, dims2, dimensions_per_ensemble=2, radius=radius))

        transform_inA = transform_inA.reshape(2*dims2, dims)
        transform_inB = transform_inB.reshape(2*dims2, dims)
        A.connect_to(D, transform=transform_inA, filter=pstc)
        B.connect_to(D, transform=transform_inB, filter=pstc)

        def product(x):
            return x[0] * x[1]
        D.connect_to(C, function=product, transform=transform_out, filter=pstc)

        self.ensemble = D
        self.transform_inA = transform_inA
        self.transform_inB = transform_inB
        self.transform_out = transform_out
