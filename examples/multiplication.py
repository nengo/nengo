from .. import nengo as nengo

# Create the nengo model
model = nengo.Model('Multiplication') # Create the network object

# Make controllable inputs
model.make_node('Input A', [8]) # Create a controllable input function 
                                #   with a starting value of 8

model.make_node('Input B', [5]) # Create a controllable input function 
                                #   with a starting value of 5

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1, radius = 10)  # Make a population with 100 neurons, 
                                               #   1 dimensions, a radius of 10
                                               #   (default is 1)
model.make_ensemble('B', 100, 1, radius = 10)  # Make a population with 100 neurons, 1 dimensions, a
                                               #   radius of 10 (default is 1)                    
model.make_ensemble('Combined', 225, 2, radius = 15) 
                                               # Make a population with 225 neurons, 2 dimensions, and set a 
                                               #   larger radius (so 10, 10 input still fits within the 
                                               #   circle of that radius)
model.make_ensemble('D', 100, 1, radius = 100) # Make a population with 100 neurons, 1 dimensions, a 
                                               #   radius of 10 (default is 1)
                      
model.connect('Input A', 'A')                  # Connect all the relevant objects
model.connect('Input B', 'B')
model.connect('A', 'Combined', transform = nef.gen_transform(index_post = 0)) 
                                               # Connect with the given 1x2D mapping matrix
model.connect('B', 'Combined', transform = nef.gen_transform(index_post = 1)) 

def product(x):
    return x[0]*x[1]

model.connect('Combined', 'D', func = product) # Create the output connection mapping the 
                                               #   1D function 'product'

# Build the model
model.build()                             # Generate model parameters

# Run the model
model.run(1)                              # Run for 1 second
