# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:31:21 2015

@author: Paxon
"""


import nengo
import numpy as np

from nengo.solvers import Lstsq


class SpecifyDecoders(nengo.solvers.Solver):
    def __init__(self, decoders, weights=False):
        self.weights = weights
        self.decoders = np.array(decoders).T

    def __call__(self, A, Y, E=None, rng=None):
        return self.decoders, []


with nengo.Network() as m:
    n_neurons = 80
    dimensions = 4
    
    encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_neurons, dimensions) 
    
    pos_pop = nengo.Ensemble(n_neurons, dimensions, encoders=encoders)
    neg_pop = nengo.Ensemble(n_neurons, dimensions, encoders=-encoders)
    
        
    
    pos_encoders = encoders.copy()
    pos_encoders[pos_encoders<0] = 0
    neg_encoders = -1.0 * encoders.copy()
    neg_encoders[neg_encoders<0] = 0
    
    
    input_node = nengo.Node([0]*dimensions)
    
    
    nengo.Connection(neg_pop, pos_pop, transform=[-1]*dimensions)
    nengo.Connection(input_node, pos_pop, solver=SpecifyDecoders(pos_encoders))
    nengo.Connection(input_node, neg_pop, solver=SpecifyDecoders(neg_encoders))


if __name__=='__main__':
    import nengo_gui
    nengo_gui.Viz(__file__).start() 
    