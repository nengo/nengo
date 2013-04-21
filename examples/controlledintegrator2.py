from .. import nengo as nengo
from nengo.connection import gen_transfrom
from nengo.filter import ExponentialPSC

## This example demonstrates how to create a controlled integrator in neurons
##   as described in the book "How to build a brain".
##   The controlled integrator takes two inputs: 
##      Input - the input to the integrator
##      Control - the control signal to the integrator
##   The function the controlled integrator implements can be written in the 
##     following control theoretic equation:
##     
##     a_dot(t) = control(t) * a(t) + tau_feedback * input(t)
##
## Network diagram:
##
##                     .----.
##                     v    | 
##      [Input] ----> (A) --'
##                     ^ 
##      [Control] -----'
##
##
## Network behaviour:
##   A = tau_feedback * Input + Input * Control
##

# Create the nengo model
model = nengo.Model('Controlled Integrator 2')

# Create the model inputs
model.make_node('Input', {0.2:5, 0.3:0, 0.44:-10,   # Create a controllable input 
                          0.54:0, 0.8:5, 0.9:0} )   #   function with a default function
                                                    #   that goes to 5 at time 0.2s, to 0 
                                                    #   at time 0.3s and so on
model.make_node('Control', [1])                     # Create a controllable input function
                                                    #   with a starting value of 1

# Create the neuronal ensembles
model.make_ensemble('A', 225, 2,                    # Make a population with 225 neurons, 
                    radius = 1.5)                   #   2 dimensions, and a larger radius 
                                                    #   to accommodate large simulataneous 
                                                    #   inputs

# Create the connections within the model
tau_feedback = 0.1
model.connect('Input', 'A', transform = [[tau_feedback], [0]], 
              filter = ExponentialPSC(pstc = 0.1))  
                                                    # Connect all the input signals to the 
                                                    #   ensemble with the appropriate 1 x 2
                                                    #   mappings, postsynaptic time
                                                    #   constant is 10ms
model.connect('Control', 'A', transform = [[0], [1]], 
              filter = ExponentialPSC(pstc = 0.1))

def feedback(x):
    return x[0] * x[1] + x[0]                       # Note: This is different than the other c
                                                    #   controlled integrator
model.connect('A', 'A', transform = [[1], [0]], func = feedback, 
              filter = ExponentialPSC(pstc = feedback_pstc))  
                                                    # Create the recurrent
                                                    #   connection mapping the
                                                    #   1D function 'feedback'
                                                    #   into the 2D population
                                                    #   using the 1 x 2 transform

# Build the model
model.build()

# Run the model
model.run(1)                                        # Run the model for 1 second
