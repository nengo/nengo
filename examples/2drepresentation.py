from .. import nengo as nengo

## This example demonstrates how to create a neuronal ensemble that can represent a 
##   two-dimensional signal.
##
## Network diagram:
##
##      [Input (2D)] ---> (Neurons) 
##
##
## Network behaviour:
##   Neurons = Input
##

# Create the nengo model
model = nengo.Model('2D Representation')

# Create the model inputs
model.make_node('Input', [0, 0])        # Create a controllable 2-D input with 
                                        #   a starting value of (0,0)

# Create the neuronal ensembles
model.make_ensemble('Neurons', 100, 2)  # Create a population with 100 neurons 
                                        #   representing 2 dimensions

# Create the connections within the model
model.connect('Input','Neurons')        # Connect the input to the neuronal population

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second
