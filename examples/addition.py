import nengo

## This example demonstrates how to create a neuronal ensemble that can add the values
##   represented in two preceding populations. 
##
## Network diagram:
##
##      [Input A] ---> (A) --.
##                           v
##                          (C) 
##                           ^
##      [Input B] ---> (B) --'
##
##
## Network behaviour:
##   A = Input_A
##   B = Input_B
##   C = A + B
##

# Create the nengo model
model = nengo.Model('Addition')  

# Create the model inputs
model.make_node('Input A',[0])          # Create a controllable input function 
                                        #   with a starting value of 0
model.make_node('Input B',[0])          # Create another controllable input 
                                        #   function with a starting value of 0

# Create the neuronal ensembles
model.make_ensemble('A', 100, 1)        # Make a population with 100 neurons, 1 dimension
model.make_ensemble('B', 100, 1)        # Make a population with 100 neurons, 1 dimension
model.make_ensemble('C', 100, 1)        # Make a population with 100 neurons, 1 dimension

# Create the connections within the model
model.connect('Input A', 'A')           # Connect the inputs to the revelant populations
model.connect('Input B', 'B')            
model.connect('A', 'C')                 # Connect the neuron populations together
model.connect('B', 'C')

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second
