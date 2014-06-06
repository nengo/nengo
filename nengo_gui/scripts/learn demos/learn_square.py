
# coding: utf-8

# # Nengo Example: Learning to square the input

# This demo shows you how to construct a network containing an ensemble which learns how to decode the square of its value.

# ## Step 1: Create the Model

# This network consists of an ensemble which represents the input, ('A'), an ensemble which learns to represent the square, ('A_squared'), and an ensemble which represents the error between A_squared and the actual square, ('error').

# In[1]:

import nengo
model = nengo.Network()
with model:
    # Create the ensemble to represent the input, the input squared (learned), and the error
    A = nengo.Ensemble(100, dimensions=1)
    A_squared = nengo.Ensemble(100, dimensions=1)
    error = nengo.Ensemble(100, dimensions=1)
    
    # Connect A and A_squared with a communication channel
    conn = nengo.Connection(A, A_squared)
    # This will provide an error signal to conn
    error_conn = nengo.Connection(error, A_squared, modulatory=True)
    # Apply the PES learning rule to conn using error_con as the error signal
    conn.learning_rule = nengo.PES(error_conn, learning_rate=2.0)

    # Compute the error signal
    nengo.Connection(A_squared, error, transform=-1)
    nengo.Connection(A, error, function=lambda x: x**2)  # This would normally come from some external system


# ##Step 2: Provide Input to the Model

# A single input signal (a step function) will be used to drive the neural activity in ensemble A. An additonal node will inhibit the error signal after 15 seconds, to test the learning at the end.

# In[2]:

import numpy as np
with model:
    # Create an input node that steps between -1 and 1
    input_node = nengo.Node(output=lambda t: int(6*t/5)/3.0 % 2 - 1)
    
    # Connect the input node to ensemble A
    nengo.Connection(input_node, A)
    
    # Shut off learning by inhibiting the error population
    stop_learning = nengo.Node(output=lambda t: t >= 15)
    nengo.Connection(stop_learning, error.neurons, transform=-20*np.ones((error.n_neurons, 1)))


# ##Step 3: Probe the Output

# Let's collect output data from each ensemble and output.

# In[3]:

with model:
    input_node_probe = nengo.Probe(input_node)
    A_probe = nengo.Probe(A, synapse=0.01)
    A_squared_probe = nengo.Probe(A_squared, synapse=0.01)
    error_probe = nengo.Probe(error, synapse=0.01)
    learn_probe = nengo.Probe(stop_learning, synapse=None)


# ## Step 4: Run the Model

# In[4]:

# Create the simulator
sim = nengo.Simulator(model)
sim.run(20)


# In[5]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

# Plot the input signal
plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[input_node_probe], label='Input', color='k', linewidth=2.0)
plt.plot(sim.trange(), sim.data[learn_probe], label='Stop learning?', color='r', linewidth=2.0)
plt.legend(loc='lower right')
plt.ylim(-1.2, 1.2)

plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[input_node_probe] ** 2, label='Squared Input', linewidth=2.0)
plt.plot(sim.trange(), sim.data[A_squared_probe], label='Decoded Ensemble $A^2$')
plt.legend(loc='lower right')
plt.ylim(-1.2, 1.2)

plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[A_squared_probe] - sim.data[input_node_probe]**2, label='Error')
plt.legend(loc='lower right')
plt.tight_layout()


# We see that during the first three periods, the decoders quickly adjust to drive the error to zero. When learning is turned off for the fourth period, the error stays closer to zero, demonstrating that the learning has persisted in the connection.
