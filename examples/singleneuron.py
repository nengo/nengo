from .. import nengo as nengo

## This example demonstrates how to create a neuronal ensemble containing a single neuron.
##
## Network diagram:
##
##      [Input (1D)] ---> (Single Neuron) 
##
##

# Create the nengo model
model = nengo.Model('Single Neuron')           # Create the network

# Create the model inputs
model.make_node('Input', [-0.45])              # A controllable input with a 
                                               #   starting value of -0.45
# Create the neuronal ensemble
model.make_ensemble('Neuron', 1, 1,            # Make 1 neuron representing
                    max_rate = (100, 100),     #  1 dimension, with a maximum
                    intercept = (-0.5, -0.5),  #  firing rate of 100, a
                    encoders = [[1]])          #  tuning curve x-intercept of   
                                               #  -0.5, encoder of 1 (i.e. it
                                               #  responds more to positive
                                               #  values) 

model.noise = 3                                # Set the neural noise to have a
                                               #  variance of 3

# Create the connections within the model
model.connect('Input','Neuron')                # Connect the input to the neuron

# Build the model
model.build()                                  # Generate model parameters

# Run the model
model.run(1)                                   # Run for 1 second


