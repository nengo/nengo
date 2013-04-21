from .. import nengo as nengo
import nengo.nef as nef
from nef import *

model = nef.model.Network('Controlled Integrator')   # Create the network object
model.make_input('input', {0.2:5, 0.3:0, 0.44:-10,
                           0.54:0, 0.8:5, 0.9:0} )   # Create a controllable input 
                                                     #   function with a default function
                                                     #   that goes to 5 at time 0.2s, to 0 
                                                     #   at time 0.3s and so on

model.make_node('control', [1])         # Create a controllable input function
                                        #   with a starting value of 1
model.make_ensemble('A', 225, 2, radius = 1.5)   
                                        # Make a population with 225 neurons, 2 dimensions, and a 
                                        #   larger radius to accommodate large simulataneous inputs
                               
model.connect('input', 'A', transform = nef.gen_transform(index_post = 0, weight = 0.1), 
              filter = nef.ExponentialPSC(pstc = 0.1))  # Connect all the relevant
                                                        #   objects with the relevant 1x2
                                                        #   mappings, postsynaptic time
                                                        #   constant is 10ms
model.connect('control', 'A', transform = nef.gen_transform(index_post = 1), 
              filter = nef.ExponentialPSC(pstc = 0.1))
def feedback(x):
    return x[0]*x[1]
model.connect('A', 'A', transform = nef.gen_transform(index_post = 0), func = feedback, 
              filter = nef.ExponentialPSC(pstc = 0.1))  # Create the recurrent
                                                        #   connection mapping the
                                                        #   1D function 'feedback'
                                                        #   into the 2D population
                                                        #   using the 1x2 transform

model.run(1) # Run for 1 second
