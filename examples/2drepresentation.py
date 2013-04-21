##from pprint import pprint

from .. import nengo as nengo
from nengo.nef import nef as nef

# Create the network
model = nef.model.Network('2D Representation')  

# Create a controllable 2-D input with a starting value of (0,0)
model.make_node('input', [0, 0])

# Create a population with 100 neurons representing 2 dimensions
model.make_ensemble('neurons', 100, 2)

# Connect the input to the neurons
model.connect('input','neurons')  

# Run for 1 second
model.run(1)  

##pprint(net.network)
