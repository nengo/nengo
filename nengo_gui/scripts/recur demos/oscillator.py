# # Nengo Example: A Simple Harmonic Oscillator
# This demo implements a simple harmonic oscillator in a 2D neural population. The oscillator is more visually interesting on its own than the integrator, but the principle at work is the same. Here, instead of having the recurrent input just integrate (i.e. feeding the full input value back to the population), we have two dimensions which interact. In Nengo there is a Linear System template which can also be used to quickly construct a harmonic oscillator (or any other linear system).

# Create the model object
import nengo
from nengo.utils.functions import piecewise

model = nengo.Network(label='Oscillator')
with model:
    # Create the ensemble for the oscillator
    neurons = nengo.Ensemble(200, dimensions=2, label="neurons")
    
    # Create an input signal
    input = nengo.Node(piecewise({0: [1, 0], 0.1: [0, 0]}), label="input")
    
    # Connect the input signal to the neural ensemble
    nengo.Connection(input, neurons)
    
    # Create the feedback connection
    nengo.Connection(neurons, neurons, transform=[[1, 1], [-1, 1]], synapse=0.1)

    input_probe = nengo.Probe(input, 'output')
    neuron_probe = nengo.Probe(neurons, 'decoded_output', synapse=0.1)
