from .. import nengo as nengo
from nengo.connection import gen_transfrom
from nengo.filter import ExponentialPSC

## This example demonstrates how to create an integrator in neurons.
##   The function an integrator implements can be written in the 
##   following control theoretic equation:
##     
##     a_dot(t) = A * a(t) + B * input(t)
##
##   The NEF equivalent equation for this integrator is:
##
##     a_dot(t) = a(t) + tau * input(t)
##
##   where tau is the recurrent time constant.
##
## Network diagram:
##
##                     .----.
##                     v    | 
##      [Input] ----> (A) --'
##
##
## Network behaviour:
##   A = tau * Input + Input
##

# Define model parameters
tau = 0.1                                           # Recurrent time constant

# Create the nengo model
model = nengo.Model('Integrator')

# Create the model inputs
model.make_node('Input', {0.2:5, 0.3:0, 0.44:-10,   # Create a controllable input 
                          0.54:0, 0.8:5, 0.9:0} )   #   function with a default function
                                                    #   that goes to 5 at time 0.2s, to 0 
                                                    #   at time 0.3s and so on

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1)                    # Make a population with 100 neurons, 
                                                    #  1 dimension

# Create the connections within the model
model.connect('Input', 'A', transform = gen_transform(weight = tau), 
              filter = ExponentialPSC(pstc = 0.1))  
                                                    # Connect the input to the integrator, 
                                                    #   scaling the input by tau_feedback with 
                                                    #   a postsynaptic time constant of 10ms
model.connect('A', 'A', filter = ExponentialPSC(pstc = tau))  
                                                    # Connect the population to itself with the 
                                                    #   default weight of 1

# Build the model
model.build()

# Run the model
model.run(1)                                        # Run the model for 1 second
