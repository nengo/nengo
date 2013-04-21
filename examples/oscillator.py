from .. import nengo as nengo
from nengo.connection import gen_transform

## This example demonstrates how to create a neuronal ensemble that acts as an oscillator
##
## Network diagram:
##
##      [Input (2D)] ---> (Neurons) --
##                              ^     |
##                              |     |
##                              -------         

# Create the nengo model
model = nengo.Model('Oscillator')         # Create the network object

# Make controllable inputs
def start_input(t):
    if t < 0.1:
        return [1,0]
    else:
        return [0,0]

model.make_node('Input', start_input)     # Create an input function that gets set to 
                                          #   to zero after .1s

# Create the neuronal ensembles
model.make_ensemble('Neurons', 200, 2)    # Make a population with 200 neurons, 2 dimensions

# Create the connections within the model
model.connect('Input','Neurons')

model.connect('Neurons','Neurons',        # Recurrently connect the population 
              transform=[[1,1], [-1,1]],  #   with the connection matrix for a 
              filter=ExponentialPSC(0.1)) #   simple harmonic oscillator mapped 
                                          #   to neurons with the NEF

# Build the model
model.build()                             # Generate model parameters

# Run the model
model.run(1)                              # Run for 1 second
