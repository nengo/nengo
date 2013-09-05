"""
circularconv.py: provides CircularConvolution template
"""

import numpy as np

from .. import core
from .. import objects

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

    return T.reshape((-1, dims))

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
        dims2 = Tout.shape[1]

        def fn(x):
            x = x.reshape((dims2, 2))
            return (x[:,0] * x[:,1]).reshape((-1,1))
        D = model.add(core.Direct(n_in=2*dims2, n_out=dims2, fn=fn))
        encA = model.add(core.Encoder(A, D, TinA))
        encB = model.add(core.Encoder(B, D, TinB))
        dec = model.add(core.Decoder(D, C, Tout))
        tf = model.add(core.Transform(1.0, C, C))


class CircularConvolution(object):
    """
    CircularConvolution docs XXX
    """

    def __init__(self, name, neurons, dimensions, radius=1,
                 invert_a=False, invert_b=False):
        from ..templates import EnsembleArray

        self.name = name
        self.dimensions = dimensions

        dims = dimensions
        self.transformA = _input_transform(dims, first=True, invert=invert_a)
        self.transformB = _input_transform(dims, first=False, invert=invert_b)
        self.transformC = _output_transform(dims)
        dims2 = self.transformC.shape[1]
        self.ensemble = EnsembleArray(
                name, neurons, dims2, dimensions_per_ensemble=2, radius=radius)

        self.connections_in = []
        self.connections_out = []
        self.probes = {'decoded_output': []}

    def connect_input_A(self, A, **kwargs):
        A.connect_to(self.ensemble, transform=self.transformA, **kwargs)

    def connect_input_B(self, B, **kwargs):
        B.connect_to(self.ensemble, transform=self.transformB, **kwargs)

    def connect_to(self, post, **kwargs):
        def product(x):
            return x[0] * x[1]
        c = self.ensemble.connect_to(
            post, function=product, transform=self.transformC, **kwargs)
        self.connections_out.append(c)

    def probe(self, to_probe='decoded_output',
              sample_every=0.001, filter=0.01, dt=0.001):
        if to_probe == 'decoded_output':
            probe = objects.Probe(self.name + ".decoded_output", sample_every)
            probe.dimensions = self.dimensions
            self.connect_to(probe, filter=filter)
            self.probes['decoded_output'].append(probe)
        return probe

    def build(self, model, dt):
        self.ensemble.build(model, dt)

