# # Nengo Example: Multiplication

# This example will show you how to multiply two values. The model architecture can be thought of as a combination of the combining demo and the squaring demo. Essentially, we project both inputs independently into a 2D space, and then decode a nonlinear transformation of that space (the product of the first and second vector elements).

# Create the model object
import nengo
model = nengo.Network(label='Multiplication')
with model:
    # Create 4 ensembles of leaky integrate-and-fire neurons
    A = nengo.Ensemble(100, dimensions=1, radius=10, label="A")
    B = nengo.Ensemble(100, dimensions=1, radius=10, label="B")
    combined = nengo.Ensemble(224, dimensions=2, radius=15, label="combined") # This radius is ~sqrt(10^2+10^2)
    prod = nengo.Ensemble(100, dimensions=1, radius=20, label="product")

# This next two lines make all of the encoders in the Combined population point at the 
# corners of the cube. This improves the quality of the computation.
# Note the number of neurons is assumed to be divisible by 4
import numpy as np
# Comment out the line below for 'normal' encoders
combined.encoders = np.tile([[1,1],[-1,1],[1,-1],[-1,-1]], (combined.n_neurons // 4, 1))

from nengo.utils.functions import piecewise
with model:
    # Create a piecewise step function for input
    inputA = nengo.Node(piecewise({0: 0, 2.5: 10, 4: -10}), label="input A")
    inputB = nengo.Node(piecewise({0: 10, 1.5: 2, 3: 0, 4.5: 2}), label="input B")
    
    # Connect the input nodes to the appropriate ensembles
    nengo.Connection(inputA, A)
    nengo.Connection(inputB, B)
    
    # Connect input ensembles A and B to the 2D combined ensemble
    nengo.Connection(A, combined[0])
    nengo.Connection(B, combined[1])
    
    # Define a function that computes the multiplication of two inputs
    def product(x):
        return x[0] * x[1]
    
    # Connect the combined ensemble to the output ensemble D
    nengo.Connection(combined, prod, function=product)

    inputA_probe = nengo.Probe(inputA)
    inputB_probe = nengo.Probe(inputB,)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    combined_probe = nengo.Probe(combined, synapse=0.01)
    prod_probe = nengo.Probe(prod, synapse=0.01)

