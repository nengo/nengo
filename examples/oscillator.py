from .. import nengo as nengo
from ..nengo.connection import gen_transform

## This example demonstrates how to create a neuronal ensemble that acts as an oscillator.
##   In this example, the oscillatory is a 2-D ring oscillator. The function an integrator 
##   implements can be written in the following control theoretic equation:
##     
##     a_dot(t) = A_matrix * a(t)
##
##     where A_matrix = [ 0 w]
##                      [-w 0]
##    
##     The NEF equivalent A_matrix for this oscillator is: A_matrix = [ 1 1]
##                                                                    [-1 1]
##
##   The input to the network is needed to kick the system into its stable oscillatory state.
##
## Network diagram:
##
##                    .----.
##                    v    | 
##      [Input] ---> (A) --'
##
##
## Network behaviour:
##   A = A_matrix * A
## 

# Create the nengo model
model = nengo.Model('Oscillator')           # Create the network object

# Make controllable inputs
def start_input(t):
    if t < 0.1:
        return [1,0]
    else:
        return [0,0]

model.make_node('Input', start_input)       # Create an input function that gets set to 
                                            #   to zero after 0.1s, just to get things going

# Create the neuronal ensembles
model.make_ensemble('A', 200, 2)            # Make a population with 200 neurons, 2 dimensions

# Create the connections within the model
model.connect('Input', 'A')                 # Connect the input population to the oscillator

model.connect('A', 'A',                     # Recurrently connect the population 
              transform = [[1,1], [-1,1]]   #   with the connection matrix for a 
              filter = ExponentialPSC(0.1)) #   simple harmonic oscillator mapped 
                                            #   to neurons with the NEF

# Build the model
model.build()                               # Generate model parameters

# Run the model
model.run(1)                                # Run for 1 second
