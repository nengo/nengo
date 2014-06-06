# # Nengo Example: Integrator
# 
# This demo implements a one-dimensional neural integrator.
# 
# This is the first example of a recurrent network in the demos. It shows how neurons can be used to implement stable dynamics. Such dynamics are important for memory, noise cleanup, statistical inference, and many other dynamic transformations.
# 
# When you run this demo, it will automatically put in some step functions on the input, so you can see that the output is integrating (i.e. summing over time) the input. You can also input your own values. Note that since the integrator constantly sums its input, it will saturate quickly if you leave the input non-zero. This makes it  clear that neurons have a finite range of representation. Such saturation effects can be exploited to perform useful computations (e.g. soft normalization).

import nengo
from nengo.utils.functions import piecewise

model = nengo.Network(label='Integrator')
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1, label="neurons")

# Create a piecewise step function for input
    input = nengo.Node(piecewise({0: 0, 0.2: 1, 1: 0, 2: -2, 3: 0, 4: 1, 5: 0}), label="input")

    # Connect the population to itself
    tau = 0.1
    nengo.Connection(A, A, transform=[[1]], synapse=tau) # Using a long time constant for stability
    
    # Connect the input
    nengo.Connection(input, A, transform=[[tau]], synapse=tau) # The same time constant as recurrent to make it more 'ideal'

    # Add probes
    input_probe = nengo.Probe(input)
    A_probe = nengo.Probe(A, synapse=0.01)
