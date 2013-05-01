import nengo

## This example demonstrates how to create a neuronal ensemble that can represent a 
##   one-dimensional signal.
##
## Network diagram:
##
##      [Input] ---> (Neurons) 
##
##
## Network behaviour:
##   Neurons = Input
##

# Create the nengo model
model = nengo.Model('1D Representation')

# Create the model inputs
model.make_node('Input', [1])        # Create a controllable 2-D input with 
                                        #   a starting value of (0,0)

# Create the neuronal ensembles
model.make_ensemble('Neurons', 100, 1)  # Create a population with 100 neurons 
                                        #   representing 2 dimensions

# Create the connections within the model
model.connect('Input','Neurons')        # Connect the input to the neuronal population

# Add probes to the model
model.probe('Input:output')
model.probe('Neurons:output')

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second

# Plot the results
probes = model.probes
import matplotlib.pyplot as plot
plot.figure()
plot.hold(True)
plot.plot(probes[0].get_data())
plot.plot(probes[1].get_data())
plot.ylim([-1.2,1.2])
plot.legend(['Input', 'Neurons'])
plot.show()