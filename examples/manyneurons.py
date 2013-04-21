from .. import nengo as nengo
import nengo.nef as nef

model = nef.model.Network('Many Neurons')   # Create the network
model.make_node('input',[-0.45])            # Create a controllable input
                                            #   with a starting value of -.45

neurons = model.make_ensemble('neurons', 100, 1)    # Make a population of 100 neurons, 
neurons.noise = 1.0                                 #   representing 1 dimensions with random
                                                    #   injected input noise of variance 1

model.connect('input','neurons')    # Connect the input to the neuron

model.run(1) # Run for 1 second
