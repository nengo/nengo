from .. import nengo as nengo
from ..nengo.networks import array

## This example demonstrates how to create a neuronal network array that can represent a 
##   two-dimensional signal. A neuronal network array is a collection of neuronal ensembles,
##   each representing a different dimension of the output. 
##
## Network diagram:
##
##      [Input] ---> {NetworkArray} ---> (Output)
##
##
## Network behaviour:
##   NetworkArray = Input
##

# Create the nengo model
model = nengo.Model('Network Array 2D Rep')

# Create the model inputs
model.make_node('Input', [0, 0])            # Create a controllable 2-D input with 
                                            #   a starting value of (0,0)

# Create the neuronal ensembles
array.make(model, 'NetworkArray', 200, 2)   # Create a population with 100 neurons 
                                            #   representing 2 dimensions
model.make_ensemble('B', 200, 2)            # Make a population with 200 neurons, 2 dimension

# Create the connections within the model
model.connect('Input', 'NetworkArray')      # Connect the input to the network array
model.connect('NetworkArray', 'B')          # Connect the output of the network array to the 
                                            #   'B' population

# Build the model
model.build()

# Run the model
model.run(1)                                # Run the model for 1 second
