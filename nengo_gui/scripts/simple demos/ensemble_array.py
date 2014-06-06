# # Nengo Network: Ensemble Array
# 
# An ensemble array is a group of ensembles that each represent a part of the overall signal.
# 
# Ensemble arrays are similar to normal ensembles, but expose a slightly different interface. Additionally, in an ensemble array, the components of the overall signal are not related. As a result, network arrays cannot be used to compute nonlinear functions that mix the dimensions they represent.

import nengo
import numpy as np

model = nengo.Network(label='Ensemble Array')
with model:
    # Make an input node
    sin = nengo.Node(output=lambda t: [np.cos(t), np.sin(t)], label="sin")
    
    # Make ensembles to connect
    A = nengo.networks.EnsembleArray(100, n_ensembles=2, label="A")
    B = nengo.Ensemble(100, dimensions=2, label="B")
    C = nengo.networks.EnsembleArray(100, n_ensembles=2, label="C")
    
    # Connect the model elements, just feedforward
    nengo.Connection(sin, A.input)
    nengo.Connection(A.output, B)
    nengo.Connection(B, C.input)
    
    # Setup the probes for plotting
    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A.output, synapse=0.02)
    B_probe = nengo.Probe(B, synapse=0.02)
    C_probe = nengo.Probe(C.output, synapse=0.02)
