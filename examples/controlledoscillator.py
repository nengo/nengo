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

# Define model parameters
speed = 10                                      # Base frequency of oscillation
tau = 0.1                                       # Recurrent time constant

# Create the nengo model
model = nengo.Model('Controlled Oscillator')    # Create the network object

# Make controllable inputs
def start_input(t):
    if t < 0.15:
        return [1,0]
    else:
        return [0,0]

model.make_node('Input', start_input)           # Create an input function that gets set to 
                                                #   to zero after 0.15s, just to get things going
model.make_node('Speed', [1])                   # Create an input function that can be used to control 
                                                #   the speed interactively

# Create the neuronal ensembles
model.make_ensemble('A', 500, 3,                # Make a population with 500 neurons, 3 dimensions,
                    radius = 1.7)               # and a radius of 1.7

# Create the connections within the model
model.connect('Input', 'A', transform = gen_transform(index_post = [0,1]))
                                                # Connect the input population to the first 2
                                                #   dimensions of the oscillator
model.connect('Speed', 'A', transform = gen_transform(index_post = [2]))
                                                # Connect the speed control to the third dimension
                                                #   of the oscillator

def controlled_path(x):                         # Define the nonlinear interactions in the state 
    return [x[0] + x[2] * speed * tau * x[1],   #   space of the oscillator
            x[1] - x[2] * speed * tau * x[0], 
            0]

model.connect('A', 'A', func = controlled_path, filter = {'type': 'ExponentialPSC', 'pstc': tau})
                                                # Recurrently connect the population 
                                                #   with the defined function for controlling
                                                #   the oscillator

# Build the model
model.build()                                   # Generate model parameters

# Run the model
model.run(1)                                    # Run for 1 second



