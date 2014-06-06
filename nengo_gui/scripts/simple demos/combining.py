# # Nengo Example: Combining
# 
# This example demonstrates how to create a neuronal ensemble that will combine two 1-D inputs into one 2-D representation.

import nengo
model = nengo.Network(label='Combining')
with model:
    # Our input ensembles consist of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1, label = "A")
    B = nengo.Ensemble(100, dimensions=1, label = "B")
    
    # The output ensemble consists of 200 leaky integrate-and-fire neurons,
    # representing a two-dimensional signal
    output = nengo.Ensemble(200, dimensions=2, label='2D Population')

import numpy as np
    # Create input nodes generating the sine and cosine
    sin = nengo.Node(output=np.sin, label = "sin")
    cos = nengo.Node(output=np.cos, label = "cos")

    nengo.Connection(sin, A)
    nengo.Connection(cos, B)
    
    # The square brackets define which dimension the input will project to
    nengo.Connection(A, output[1])
    nengo.Connection(B, output[0])

    sin_probe = nengo.Probe(sin)
    cos_probe = nengo.Probe(cos)
    A_probe = nengo.Probe(A, synapse=0.01)  # 10ms filter
    B_probe = nengo.Probe(B, synapse=0.01)  # 10ms filter
    out_probe = nengo.Probe(output, synapse=0.01)  # 10ms filter

