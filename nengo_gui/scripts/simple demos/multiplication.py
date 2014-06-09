# # Nengo Example: Multiplication

# This example will show you how to multiply two values. The model
# architecture can be thought of as a combination of the combining demo and
# the squaring demo. Essentially, we project both inputs independently into a
# 2D space, and then decode a nonlinear transformation of that space (the
# product of the first and second vector elements).

# Create the model object
import nengo
model = nengo.Network(label='Multiplication')
with model:
    # Create 4 ensembles of leaky integrate-and-fire neurons
    A = nengo.Ensemble(100, dimensions=1, radius=10, label="A")
    B = nengo.Ensemble(100, dimensions=1, radius=10, label="B")

    # Radius on this ensemble is ~sqrt(10^2+10^2)
    combined = nengo.Ensemble(224, dimensions=2, radius=15, label="combined")

    prod = nengo.Ensemble(100, dimensions=1, radius=20, label="product")

# These next two lines make all of the encoders in the Combined population
# point at the corners of the cube. This improves the quality of the
# computation. Note the number of neurons is assumed to be divisible by 4
import numpy as np
# Comment out the line below for 'normal' encoders
combined.encoders = np.tile(
    [[1,1],[-1,1],[1,-1],[-1,-1]],
    (combined.n_neurons // 4, 1))

with model:
    # Create a piecewise step function for input
    inputA = nengo.Node([0], label="input A")
    inputB = nengo.Node([0], label="input B")

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

    nengo.Probe(inputA)
    nengo.Probe(inputB,)
    nengo.Probe(A, synapse=0.01)
    nengo.Probe(B, synapse=0.01)
    nengo.Probe(combined, synapse=0.01)
    nengo.Probe(prod, synapse=0.01)
    nengo.Probe(combined, 'spikes')


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.7291542213316193
gui[model].offset = -12.174626639948599,130.63578585201708
gui[A].pos = 175.000, 50.000
gui[A].scale = 1.000
gui[B].pos = 175.000, 125.000
gui[B].scale = 1.000
gui[combined].pos = 300.000, 87.500
gui[combined].scale = 1.000
gui[prod].pos = 425.000, 87.500
gui[prod].scale = 1.000
gui[inputA].pos = 50.000, 50.000
gui[inputA].scale = 1.000
gui[inputB].pos = 50.000, 125.000
gui[inputB].scale = 1.000
