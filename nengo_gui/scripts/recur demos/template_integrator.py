# # Nengo Example: Integrator
#
# This demo implements a one-dimensional neural integrator.
#
# This is the first example of a recurrent network in the demos. It shows how
# neurons can be used to implement stable dynamics. Such dynamics are important
# for memory, noise cleanup, statistical inference, and many other dynamic
# transformations.
#
# When you run this demo, it will automatically put in some step functions
# on the input, so you can see that the output is integrating (i.e.
# summing over time) the input. You can also input your own values. Note
# that since the integrator constantly sums its input, it will saturate
# quickly if you leave the input non-zero. This makes it  clear that neurons
# have a finite range of representation. Such saturation effects can be
# exploited to perform useful computations (e.g. soft normalization).

import nengo
from nengo.utils.functions import piecewise

tau = 0.1

model = nengo.Network(label="Integrator Template")

with model:
    integrator = nengo.networks.Integrator(
        tau, n_neurons=100, dimensions=1, label="integrator")
    input = nengo.Node(piecewise(
        {0: 0, 0.2: 1, 1: 0, 2: -2, 3: 0, 4: 1, 5: 0}), label="input")

    # Create a piecewise step function for input
    with integrator:

    # Connect the input
        nengo.Connection(input, integrator.input, synapse=tau)

        input_probe = nengo.Probe(input)
        integrator_probe = nengo.Probe(integrator.ensemble, synapse=0.01) # 10ms filter


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.1783041811384658
gui[model].offset = 186.82203414328256,120.37097172700416
gui[input].pos = 0.000, 0.000
gui[input].scale = 1.000
gui[integrator].pos = 225.000, 0.000
gui[integrator].scale = 1.000
gui[integrator].size = 230.000, 80.000
gui[integrator.ensemble].pos = 300.000, 0.000
gui[integrator.ensemble].scale = 1.000
gui[integrator.input].pos = 150.000, 0.000
gui[integrator.input].scale = 1.000
