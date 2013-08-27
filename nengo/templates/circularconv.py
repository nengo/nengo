"""
circularconv.py: provides CircularConvolution template
"""

import numpy as np

from ..core import Decoder, Encoder, Filter, LIF, Signal, Transform


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
