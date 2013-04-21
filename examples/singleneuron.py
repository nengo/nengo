from .. import nengo as nengo

model = nengo.Model('Single Neuron')           # Create the network

model.make_node('Input', [-0.45])              # Create a controllable input
                                               #   with a starting value of -.45

model.make_ensemble('Neuron', 1, 1,            # Make 1 neuron representing
                    max_rate = (100,100),      #  1 dimension, with a maximum
                    intercept = (-0.5,-0.5),   #  firing rate of 100, a
                    encoders = [[1]])          #  tuning curve x-intercept of   
                                               #  -0.5, encoder of 1 (i.e. it
                                               #  responds more to positive
                                               #  values) 

model.noise = 3                                # Set the neural noise to have a
                                               #  variance of 3

model.connect('input','neuron')                # Connect the input to the neuron

model.run(1)                                   # Run for 1 second
