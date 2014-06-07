# #Nengo Example: Controlled Integrator 2
# This demo implements a controlled one-dimensional neural integrator that is functionally the same as the controlled integrator in the previous example. However, the control signal is zero for integration, less than one for low-pass filtering, and greater than 1 for saturation. This behavior maps more directly to the differential equation used to describe an integrator: $\dot{x} = \mathrm{Ax}(t) + \mathrm{Bu}(t)$ The control in this circuit is A in that equation. This is also the controlled integrator described in the book How to build a brain.
# 

import nengo
from nengo.utils.functions import piecewise

model = nengo.Network(label='Controlled Integrator 2')
with model:
    # Make a population with 225 LIF neurons representing a 2 dimensional signal,
    # with a larger radius to accommodate large inputs
    A = nengo.Ensemble(225, dimensions=2, radius=1.5, label="A")

with model:
    # Create a piecewise step function for input
    input_func = piecewise({0.2: 5, 0.3: 0, 0.44: -10, 0.54: 0, 0.8: 5, 0.9: 0})
    inp = nengo.Node(output=input_func, label="input")
    
    # Connect the Input signal to ensemble A.
    tau = 0.1
    nengo.Connection(inp, A, transform=[[tau], [0]], synapse=0.1)

    #Another piecewise function that changes half way through the run
    control_func = piecewise({0: 0, 0.6: -0.5})
    control = nengo.Node(output=control_func, label="control")
    
    # -- Connect the "Control" signal to the second of A's two input channels
    nengo.Connection(control, A[1], synapse=0.005)

    # Note the changes from the previous example to the function being defined.
    nengo.Connection(A, A[0],
                     function=lambda x: x[0] * x[1] + x[0],
                     synapse=tau)
    
    # Record both dimensions of A
    A_probe = nengo.Probe(A, 'decoded_output', synapse=0.01)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.3901981214408314
gui[model].offset = 252.30568751828542,43.65994898682379
gui[A].pos = 150.000, 65.000
gui[A].scale = 1.000
gui[inp].pos = 3.747, 64.633
gui[inp].scale = 1.000
gui[control].pos = 148.368, 153.635
gui[control].scale = 1.000
