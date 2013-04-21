from .. import nengo as nengo
from ..nengo.connection import gen_transform

## This example demonstrates how to create a neuronal ensemble that will combine two 1-D
##   inputs into one 2-D representation.
##
## Network diagram:
##
##      [Input A] ---> (A) --.
##                           v
##                          (C) 
##                           ^
##      [Input B] ---> (B) --'
##
##
## Network behaviour:
##   A = Input_A
##   B = Input_B
##   C = [A,B]
##

# Create the nengo model
model = nengo.Model('Combining')  

# Create the model inputs
model.make_node('Input A', [0])         # Create a controllable input function 
                                        #   with a starting value of 0
model.make_node('Input B', [0])         # Create another controllable input 
                                        #   function with a starting value of 0

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1)        # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B', 100, 1)        # Make a population with 100 neurons, 1 dimension 
model.make_ensemble('C', 100, 2,        # Make a population with 100 neurons, 2 dimensions, 
                    radius = 1.5)       #   and set a larger radius (so [1,1] input still 
                                        #   fits within the circle of that radius)

# Create the connections within the model
model.connect('Input A', 'A')           # Connect the inputs to the appropriate neuron
model.connect('Input B', 'B')           #   populations (default connection is identity)

model.connect('A', 'C', gen_transform(index_post = 0))    
                                        # Connect with A to the first dimension of C
model.connect('B', 'C', gen_transform(index_post = 1))
                                        # Connect with B to the second dimension of C

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second
