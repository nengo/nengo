# # Nengo Example: Learning to square the input

# This demo shows you how to construct a network containing an ensemble which learns how to decode the square of its value.

# ## Step 1: Create the Model

# This network consists of an ensemble which represents the input, ('A'), an ensemble which learns to represent the square, ('A_squared'), and an ensemble which represents the error between A_squared and the actual square, ('error').

import nengo
import numpy as np

model = nengo.Network()
with model:
    # Create the ensemble to represent the input, the input squared (learned), and the error
    A = nengo.Ensemble(100, dimensions=1, label="A")
    A_squared = nengo.Ensemble(100, dimensions=1, label="A squared")
    error = nengo.Ensemble(100, dimensions=1, label="error")
    
    # Connect A and A_squared with a communication channel
    conn = nengo.Connection(A, A_squared)
    # This will provide an error signal to conn
    error_conn = nengo.Connection(error, A_squared, modulatory=True)
    # Apply the PES learning rule to conn using error_con as the error signal
    conn.learning_rule = nengo.PES(error_conn, learning_rate=2.0)

    # Compute the error signal
    nengo.Connection(A_squared, error, transform=-1)
    nengo.Connection(A, error, function=lambda x: x**2)  # This would normally come from some external system

    # Create an input node that steps between -1 and 1
    input_node = nengo.Node(output=lambda t: int(6*t/5)/3.0 % 2 - 1, label="input")
    
    # Connect the input node to ensemble A
    nengo.Connection(input_node, A)
    
    # Shut off learning by inhibiting the error population
    stop_learning = nengo.Node(output=lambda t: t >= 15, label="stop learning")
    nengo.Connection(stop_learning, error.neurons, transform=-20*np.ones((error.n_neurons, 1)))


    input_node_probe = nengo.Probe(input_node)
    A_probe = nengo.Probe(A, synapse=0.01)
    A_squared_probe = nengo.Probe(A_squared, synapse=0.01)
    error_probe = nengo.Probe(error, synapse=0.01)
    learn_probe = nengo.Probe(stop_learning, synapse=None)
