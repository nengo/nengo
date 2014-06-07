# # Nengo Example: Squaring the Input

# This demo shows you how to construct a network that squares the value encoded in a first population in the output of a second population. 

# Create the model object
import nengo
model = nengo.Network(label='Squaring')
with model:
    # Create two ensembles of 100 leaky-integrate-and-fire neurons
    A = nengo.Ensemble(100, dimensions=1, label = "A")
    B = nengo.Ensemble(100, dimensions=1, label = "B")

import numpy as np
with model:
    # Create an input node that represents a sine wave
    sin = nengo.Node(np.sin, label = "sin")
    
    # Connect the input node to ensemble A
    nengo.Connection(sin, A)
    
    # Define the squaring function
    def square(x):
        return x[0] * x[0]
    
    # Connection ensemble A to ensemble B
    nengo.Connection(A, B, function=square)

    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9991330790684605
gui[model].offset = 270.10435658240567,174.22430516380342
gui[A].pos = 150.000, 0.000
gui[A].scale = 1.000
gui[B].pos = 300.000, 0.000
gui[B].scale = 1.000
gui[sin].pos = 0.000, 0.000
gui[sin].scale = 1.000
