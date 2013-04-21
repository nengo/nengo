from .. import nengo as nengo
import nengo.nef as nef

model = nef.model.Network('Addition') # Create the network object

model.make_node('input A',[0])   # Create a controllable input function 
                                 #   with a starting value of 0
model.make_node('input B',[0])   # Create another controllable input 
                                 #   function with a starting value of 0
                               
model.make_ensemble('A', 100, 1) # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B', 100, 1) # Make a population with 100 neurons, 1 dimension
model.make_ensemble('C', 100, 1) # Make a population with 100 neurons, 1 dimension

model.connect('input A','A')     # Connect all the relevant objects
model.connect('input B','B')
model.connect('A','C')
model.connect('B','C')

model.run(1) # run for 1 second
