from .. import nengo as nengo
from ..nengo.connection import gen_transfrom
from ..nengo.filter import ExponentialPSC

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
def input_func(t):                                  # Create a function that outputs
    if t < 0.2:                                     #   5 at time 0.2s, then 0 at time 0.3s,
        return [0]                                  #   -10 at time 0.44, then 0 at time 0.8,
    elif t < 0.3:                                   #   5 at time 0.8, then 0 at time 0.9
        return [5]
    elif t < 0.44:
        return [0]
    elif t < 0.54:
        return [-10]
    elif t < 0.8:
        return [0]
    elif t < 0.9:
        return [5]
    else:
        return [0]
model.make_node('Input', input_func)                # Create a controllable input function 
                                                    #   with the function above

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
