from .. import nengo as nengo
import nengo.nef as nef
from nef import *

model = nef.model.Network('Integrator')             # Create the network object
model.make_node('input', {0.2:5, 0.3:0, 0.44:-10, 
                          0.54:0, 0.8:5, 0.9:0})    # Create a controllable input 
                                                    #   function with a default function
                                                    #   that goes to 5 at time 0.2s, to 0 
                                                    #   at time 0.3s and so on
                                                     
model.make_ensemble('A', 100, 1) # Make a population with 100 neurons, 1 dimension

model.connect('input', 'A', transform = nef.gen_transform(weight = 0.1), 
              filter = ExponentialPSC(pstc = 0.1))  # Connect the input to the integrator, 
                                                    #   scaling the input by .1; postsynaptic
                                                    #   time constant is 10ms
model.connect('A', 'A', filter = ExponentialPSC(pstc=0.1))
                                                    # Connect the population to itself with the 
                                                    #   default weight of 1

model.run(1) # Run for 1 second
