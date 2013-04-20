import nef.nef_theano as nef

net=nef.Network('Many Neurons')       # Create the network
net.make_input('input',[-0.45])       # Create a controllable input
                                      #   with a starting value of -.45

net.make('neurons',neurons=100,       # Make a population of 100 neurons, 
           dimensions=1,noise=1)      #  representing 1 dimensions with random
                                      #  injected input noise of variance 1

net.connect('input','neurons')        # Connect the input to the neuron

net.run(1) # run for 1 second
