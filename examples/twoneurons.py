from .. import nengo as nengo

## This example demonstrates how to create a neuronal ensemble containing a pair of neurons.
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
model = nengo.Model('Two Neurons')             # Create the network

# Create the model inputs
model.make_node('Input', [-0.45])              # Create a controllable input
                                               #   with a starting value of -.45

# Create the neuronal ensemble
model.make_ensemble('Neurons', 2, 1,           # Make 2 neurons representing
                    max_rate = (100, 100),     #   1 dimension, with a maximum
                    intercept = (-0.5, -0.5),  #   firing rate of 100, with a
                    encoders = [[1], [-1]]     #   tuning curve x-intercept of 
                                               #   -0.5, encoders of 1 and -1 
                                               #   (i.e. the first responds more
                                               #   to positive values and the
                                               #   second to negative values),
                                               #   and a noise of variance 3

model.noise = 3                                # Set the neural noise to have a
                                               #   variance of 3

# Create the connections within the model
model.connect('Input','Neurons')               # Connect the input to the neuron

# Build the model
model.build()                                  # Generate model parameters

# Run the model
model.run(1)                                   # Run for 1 second

