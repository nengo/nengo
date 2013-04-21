from .. import nengo as nengo

## This example demonstrates how to create a neuronal ensemble that represents a one
##   dimensional signal. The population of neurons is configured to receive random
##   injected current noise with a variance of 1.
##
## Network diagram:
##
##      [Input] ---> (Neurons) 
##
##
## Network behaviour:
##   Neurons = Input
##

# Create the nengo model
model = nengo.Model('Many Neurons')

# Create the model inputs
model.make_node('Input', [-0.45])                   # Create a controllable input
                                                    #   with a starting value of -0.45

# Create the neuronal ensembles
neurons = model.make_ensemble('Neurons', 100, 1)    # Make a population of 100 neurons, 
neurons.noise = 1                                   #   representing 1 dimensions with random
                                                    #   injected input noise of variance 1

# Create the connections within the model
model.connect('Input','Neurons')                    # Connect the input to the neuronal 
                                                    #   population

# Build the model
model.build()

# Run the model
model.run(1)                                        # Run the model for 1 second
