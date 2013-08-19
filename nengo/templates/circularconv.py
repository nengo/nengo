"""
circularconv.py: provides CircularConvolution template
"""

import numpy as np

from ..objects import Decoder, Encoder, Filter, LIF, Signal, Transform


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
            model.add(Encoder(self.A_Fourier[ii, 0:1], AA))
            model.add(Encoder(self.A_Fourier[ii, 1:2], AA))
            model.add(Encoder(self.A_Fourier[ii, 0:1], AB))
            model.add(Encoder(self.B_Fourier[ii, 1:2], AB))
            model.add(Encoder(self.B_Fourier[ii, 0:1], BA))
            model.add(Encoder(self.A_Fourier[ii, 1:2], BA))
            model.add(Encoder(self.B_Fourier[ii, 0:1], BB))
            model.add(Encoder(self.B_Fourier[ii, 1:2], BB))

            model.add(Decoder(AA, self.AB_prods[ii, 0:1]))
            model.add(Decoder(AB, self.AB_prods[ii, 1:2]))
            model.add(Decoder(BA, self.AB_prods[ii, 2:3]))
            model.add(Decoder(BB, self.AB_prods[ii, 3:4]))

        model.add(Transform(inverse_fourier_matrix,
                            self.AB_prods.base,
                            self.out.base))
