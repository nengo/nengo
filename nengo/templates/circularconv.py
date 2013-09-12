"""
circularconv.py: provides CircularConvolution template
"""

import numpy as np

from ..core import LIF, Signal
import nengo.core as core
import nengo.simulator as simulator


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
        model._operators += [simulator.ProdUpdate(core.Constant(fourier_matrix), A, 
                                                  core.Constant(0), self.A_Fourier.base)]
        model._operators += [simulator.ProdUpdate(core.Constant(fourier_matrix), B, 
                                                  core.Constant(0), self.B_Fourier.base)]

        # -- compute the complex elementwise product of A, B
        #    in Fourier domain
        model._operators += [simulator.Reset(self.AB_prods)]
        
        self.pops = []
        for ii in range(D//2):
            # TODO: use neuron_type
            AA = model.add(LIF(neurons_per_product))
            AB = model.add(LIF(neurons_per_product))
            BA = model.add(LIF(neurons_per_product))
            BB = model.add(LIF(neurons_per_product))
            n_in = AA.n_in
            n_out = self.A_Fourier[ii, 0:1].size
            model._operators += [
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.A_Fourier[ii, 0:1], AA.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.A_Fourier[ii, 1:2], AA.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.A_Fourier[ii, 0:1], AB.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.B_Fourier[ii, 1:2], AB.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.B_Fourier[ii, 0:1], BA.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.A_Fourier[ii, 1:2], BA.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.B_Fourier[ii, 0:1], BB.input_signal),
                simulator.DotInc(core.Constant(weights(n_in,n_out)), self.B_Fourier[ii, 1:2], BB.input_signal)]

            model._operators += [simulator.DotInc(core.Constant(weights(n_out,n_in)), AA.output_signal,
                                     self.AB_prods[ii,0:1]),
                                 simulator.DotInc(core.Constant(weights(n_out,n_in)), AB.output_signal,
                                     self.AB_prods[ii,1:2]),
                                 simulator.DotInc(core.Constant(weights(n_out,n_in)), BA.output_signal,
                                     self.AB_prods[ii,2:3]),
                                 simulator.DotInc(core.Constant(weights(n_out,n_in)), BB.output_signal,
                                     self.AB_prods[ii,3:4])]

        model._operators += [simulator.ProdUpdate(core.Constant(inverse_fourier_matrix),
                                                  self.AB_prods.base,
                                                  core.Constant(0),
                                                  self.out.base)]
