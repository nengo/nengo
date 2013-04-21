from .. import nengo as nengo

## This example demonstrates computing a nonlinear function (squaring) in neurons.
##
## Network diagram:
##
##
## Network behaviour:
##   A = Input
##   B = A^2

# Create the nengo model
model=nengo.Model('Squaring')                      

# Create the model inputs
model.make_node('Input', [0])                       # Create a controllable input function 
                                                    #   with a starting value of 0

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1)                    # Make 2 populations each with 
model.make_ensemble('B', 100, 1)                    #   100 neurons and 1 dimension 

# Create the connections within the model
model.connect('Input', 'A')                         # Connect the input to A

def square(x):
  return x[0] * x[0]
model.connect('A', 'B', func=square)                # Connect A to B with the 
                                                    #   squaring function approximated 
                                                    #   in that connection

# Build the model
model.build()

# Run the model
model.run(1)                                        # Run the model for 1 second
