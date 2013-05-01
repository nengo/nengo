import nengo

## This example demonstrates how to create a neuronal ensemble that behaves like
## a communication channel (leaves input unchanged)
##
## Network diagram:
##
##      [Input] ---> (A) ---> (B)
##
##
## Network behaviour:
##   A = Input
##   B = A
##

# Create the nengo model
model = nengo.Model('Communications Channel')

# Create the model inputs
model.make_node('Input', [0.5])         # Create a controllable input function 
                                        #   with a starting value of 0.5

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1)        # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B', 100, 1)        # Make a population with 100 neurons, 1 dimension

# Create the connections within the model
model.connect('Input','A')              # Connect the input to the first neuronal population
model.connect('A', 'B')                 # Connect the first neuronal population to the second
                                        #   neuronal population (this is the communication 
                                        #   channel)

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second
