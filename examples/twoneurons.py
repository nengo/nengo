import nef.nef_theano as nef

net=nef.Network('Two Neurons')        # Create the network
net.make_input('input',[-0.45])       # Create a controllable input
                                      #   with a starting value of -.45

net.make('neuron',neurons=2,dimensions=1,      # Make 2 neurons representing
    max_rate=(100,100),intercept=(-0.5,-0.5),  #  1 dimension, with a maximum
    encoders=[[1],[-1]],noise=3)               #  firing rate of 100, with a
                                               #  tuning curve x-intercept of 
                                               #  -0.5, encoders of 1 and -1 
                                               #  (i.e. the first responds more
                                               #  to positive values and the
                                               #  second to negative values),
                                               #  and a noise of variance 3

net.connect('input','neuron')         # Connect the input to the neuron

net.run(1) # run for 1 second
