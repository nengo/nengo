# # Nengo Example: Controlled Integrator
#
# A controlled integrator is a circuit that acts on two signals:
#
# 1. Input - the signal being integrated
# 2. Control - the control signal to the integrator
#
# A controlled integrator accumulates input, but its state can be directly
# manipulated by the control signal. We can write the dynamics of a simple
# controlled integrator like this:
#
# $$
# \dot{a}(t) = \mathrm{control}(t) \cdot a(t) + B \cdot \mathrm{input}(t)
# $$
#
# In this notebook, we will build a controlled intgrator with Leaky Integrate
# and Fire ([LIF](TODO)) neurons. The Neural Engineering Framework
# ([NEF](TODO)) equivalent equation for this integrator is:
#
# $$
# \dot{a}(t) = \mathrm{control}(t) \cdot a(t) + \tau \cdot \mathrm{input}(t).
# $$
#
# We call the coefficient $\tau$ here a *recurrent time constant* because its
# governs the rate of integration.
#
# Network behaviour:
# `A = tau * Input + Input * Control`
#

# ## Step 1: Create the network
#
# We can use standard network-creation commands to begin creating our
# controlled integrator. We create a Network, and then we create a population
# of neurons (called an *ensemble*). This population of neurons will
# represent the state of our integrator, and the connections between
# the neurons in the ensemble will define the dynamics of our integrator.

import nengo
from nengo.utils.functions import piecewise

model = nengo.Network(label='Controlled Integrator')
with model:
    # Make a population with 225 LIF neurons representing a 2 dimensional
    # signal, with a larger radius to accommodate large inputs
    A = nengo.Ensemble(225, dimensions=2, radius=1.5, label="A")

    # Create a piecewise step function for input
    input_func = piecewise(
        {0: 0, 0.2: 5, 0.3: 0, 0.44: -10, 0.54: 0, 0.8: 5, 0.9: 0})

    # Define an input signal within our model
    inp = nengo.Node(input_func, label="input")

    # Connect the Input signal to ensemble A.
    # The `transform` argument means "connect real-valued signal "Input" to the
    # first of the two input channels of A."
    tau = 0.1
    nengo.Connection(inp, A, transform=[[tau], [0]], synapse=tau)

    # Another piecewise step that changes half way through the run
    control_func = piecewise({0: 1, 0.6: 0.5})
    control = nengo.Node(output=control_func, label="control")

    # Connect the "Control" signal to the second of A's two input channels.
    nengo.Connection(control, A[1], synapse=0.005)

    # Create a recurrent connection that first takes the product
    # of both dimensions in A (i.e., the value times the control)
    # and then adds this back into the first dimension of A using
    # a transform
    nengo.Connection(A, A[0],
                     function=lambda x: x[0] * x[1],
                     synapse=tau)

    # Record both dimensions of A
    A_probe = nengo.Probe(A, 'decoded_output', synapse=0.01)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.6225966640653222
gui[model].offset = 234.9818132124747,150.67321922008716
gui[A].pos = 138.845, 0.859
gui[A].scale = 1.000
gui[inp].pos = 0.000, 0.000
gui[inp].scale = 1.000
gui[control].pos = 137.132, 115.402
gui[control].scale = 1.000
