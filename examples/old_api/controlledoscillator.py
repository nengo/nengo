"""
Demonstrates how to create a neuronal ensemble that acts as an oscillator.

In this example, we examine a 2-D ring oscillator. The function an integrator
implements can be written in the following control theoretic equation:

  a_dot(t) = A_matrix * a(t)

where A_matrix = [ 0 w]
                 [-w 0]

The NEF equivalent A_matrix for this oscillator is: A_matrix = [ 1 1]
                                                               [-1 1]

The input to the network is needed to kick the system into
its stable oscillatory state.

Network diagram:
                    .----.
                    v    |
      [Input] ---> (A) --'

Network behaviour:
   A = A_matrix * A
"""

import numpy as np
import nengo.old_api as nengo
from nengo.old_api import compute_transform

# Define model parameters
dt = 1e-3
speed = 10                                      # Base frequency of oscillation
tau = 0.1                                       # Recurrent time constant

# Create the nengo model
net = nengo.Network('Controlled Oscillator')    # Create the network object

# Make controllable inputs
def start_input(t):
    if t < 0.15:
        return [1,0]
    else:
        return [0,0]

net.make_input('Input', start_input)

net.make_input('Speed', [1])

# Create the neuronal ensembles
net.make('A', 500, 3, radius=1.7)

# Create the connections within the model
# net.connect('Input', 'A', transform=np.array([[1,0],[0,1],[0,0]]))

# net.connect('Speed', 'A', transform=np.array([[0],[0],[1]]))

net.connect('Input', 'A', transform=[[1,0],[0,1],[0,0]])

net.connect('Speed', 'A', transform=[[0],[0],[1]])

def controlled_path(x):
    return [x[0] + x[2] * speed * tau * x[1],
            x[1] - x[2] * speed * tau * x[0],
            0]

net.connect('A', 'A', function=controlled_path)
# net.connect('A', 'A', function=controlled_path,
              # filter={'type': 'ExponentialPSC', 'pstc': tau})

# Build the model
net.build()

# Run the model
net.run(1)

