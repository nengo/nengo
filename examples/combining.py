from .. import nengo as nengo
import nengo.nef as nef

model = nef.model.Network('Combining')  # Create the network object

model.make_node('input A', [0])  # Create a controllable input function 
                                 #   with a starting value of 0
model.make_node('input B', [0])  # Create another controllable input 
                                 #   function with a starting value of 0
                              
model.make_ensemble('A', 100, 1) # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B', 100, 1) # Make a population with 100 neurons, 1 dimension 
model.make_ensemble('C', 100, 2, radius=1.5) 
                                 # Make a population with 100 neurons, 2 dimensions, and set a
                                 #   larger radius (so 1,1 input still fits within the circle of
                                 #   that radius)

model.connect('input A','A') # Connect all the relevant objects (default connection 
                             #   is identity)
model.connect('input B','B')
model.connect('A','C', index_post=0) # Connect with the given 1x2D mapping matrix
model.connect('B','C', index_post=1)

model.run(1) # run for 1 second
