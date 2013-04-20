from pprint import pprint

import nengo.nef

net = nengo.nef.Network('2D Representation')  # Create the network

# Create a controllable 2-D input with a starting value of (0,0)
net.make_input('input', [0, 0])

# Create a population with 100 neurons representing 2 dimensions
net.make('neurons',100,2)

net.connect('input','neurons')  # Connect the input to the neurons

net.run(1)  # run for 1 second

pprint(net.network)
