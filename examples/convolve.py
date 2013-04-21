from .. import nengo as nengo
from ..nengo.networks import convolution
from ..nengo.spa import hrr

## This example demonstrates how to create a nengo model that can bind 
##   (circular convolution) two semantic pointers together.
##
## Network diagram:
##
##      [Input A] ---> (A) --.
##                           v
##                          {*} ---> (C) 
##                           ^
##      [Input B] ---> (B) --'
##
##
## Network behaviour:
##   A = Input_A
##   B = Input_B
##   * = {Convolution network internal representation of A and B}
##   C = A (*) B
##

# Define model parameters
D = 10                                          # Number of dimensions of the semantic pointers
                                                #   being used
vocab = hrr.Vocabulary(D, include_pairs=True)   # Create and empty semantic pointer (of dimension D) 
                                                #   vocabulary
vocab.parse('a+b+c+d+e')                        # Add the semantic pointers A,B,C,D, and E (randomly
                                                #   generated) to the vocabulary

# Create the nengo model
model = nengo.Network('Convolution') 

# Create the model inputs
model.make_node('Input A', vocab.hrr['A'].v)    # Create a controllable input function with a starting
                                                #   value of the semantic pointer "A"
model.make_node('Input B', vocab.hrr['B'].v)    # Create another controllable input function with 
                                                #   a starting value of the semantic pointer "B"

# Create the neuronal ensembles
model.make_ensemble('A',300,D)                  # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B',300,D)                  # Make a population with 100 neurons, 1 dimension
model.make_ensemble('C',300,D)                  # Make a population with 100 neurons, 1 dimension

conv = convolution.make(model, '*', 'A', 'B', 'C', 100) 
                                                # Call the code to construct a convolution network using 
                                                # the created populations and 100 neurons per dimension

# Create the connections within the model
## No connections need to be made. The convolution template makes all of the necessary connections

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second
