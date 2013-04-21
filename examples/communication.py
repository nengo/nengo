from .. import nengo as nengo
import nengo.nef as nef

model = nef.model.Network('Communications Channel') # Create the network object

model.make_node('input', [0.5])  # Create a controllable input function 
                                 #   with a starting value of 0.5
                              
model.make_ensemble('A', 100, 1) # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B', 100, 1) # Make a population with 100 neurons, 1 dimension

model.connect('input', 'A') # Connect all the relevant objects
model.connect('A', 'B')

model.run(1) # run for 1 second
