from .. import nengo as nengo
from nengo.connection import gen_transform

## This example demonstrates how to create a neuronal ensemble that will combine two 1-D
##   inputs into one 2-D representations, and perform the multiplication of the two
##   input values.
##
## Network diagram:
##
##      [Input A] ---> (A) --.
##                           v
##                          (C) ---> (D)
##                           ^
##      [Input B] ---> (B) --'
##
##
## Network behaviour:
##   A = Input_A
##   B = Input_B
##   C = [A,B]
##   D = C[0] * C[1]
##

# Create the nengo model
model = nengo.Model('Multiplication') 

# Make controllable inputs
model.make_node('Input A', [8])                 # Create a controllable input function 
                                                #   with a starting value of 8
model.make_node('Input B', [5])                 # Create a controllable input function 
                                                #   with a starting value of 5

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1, radius = 10)   # Make a population with 100 neurons, 
                                                #   1 dimensions, a radius of 10
                                                #   (default is 1)
model.make_ensemble('B', 100, 1, radius = 10)   # Make a population with 100 neurons, 
                                                #   1 dimensions, a radius of 10 
model.make_ensemble('Combined', 225, 2,         # Make a population with 225 neurons, 
                    radius = 15)                #   2 dimensions, and set a larger  
                                                #   radius (so 10, 10 input still fits 
                                                #   within the circle of that radius)
model.make_ensemble('D', 100, 1, radius = 100)  # Make a population with 100 neurons, 
                                                #   1 dimensions, a radius of 100 (so 
                                                #   that it can represent the maximum 
                                                #   value of 10 * 10)
                      
# Create the connections within the model
model.connect('Input A', 'A')                   # Connect all the relevant objects
model.connect('Input B', 'B')
model.connect('A', 'C', transform = gen_transform(index_post = 0)) 
                                                # Connect with A to the first dimension of C
model.connect('B', 'C', transform = gen_transform(index_post = 1)) 
                                                # Connect with B to the second dimension of C

def product(x):                                 # Define the product function
    return x[0] * x[1]

model.connect('C', 'D', func = product)         # Create the output connection that maps the
                                                #   2D input to the appropriate 1D function 
                                                #   'product'

# Build the model
model.build()                             

# Run the model
model.run(1)                                    # Run for 1 second
