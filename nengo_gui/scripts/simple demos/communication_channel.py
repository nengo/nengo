# # Communication Channel

# This example demonstrates how to create a connections from one neuronal ensemble to another that behaves like a communication channel (that is, it transmits information without changing it). 
#  

import numpy as np
import nengo

# Create a 'model' object to which we can add ensembles, connections, etc.  
model = nengo.Network(label="Communications Channel")
with model:
    # Create an abstract input signal that oscillates as sin(t)
    sin = nengo.Node(np.sin, label = "sin")
    
    # Create the neuronal ensembles
    A = nengo.Ensemble(100, dimensions=1, label = "A")
    B = nengo.Ensemble(100, dimensions=1, label = "B")
    
    # Connect the input to the first neuronal ensemble
    nengo.Connection(sin, A)
    
    # Connect the first neuronal ensemble to the second (this is the communication channel)
    nengo.Connection(A, B)

    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=.01)  # ensemble output 
    B_probe = nengo.Probe(B, synapse=.01)


