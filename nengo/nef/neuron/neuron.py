import numpy as np


def accumulate(J, neurons, dt, time=1.0, init_time=0.05):
    """Accumulates neuron output over time.

    Take a neuron model, run it for the given amount of time with
    fixed input. Used to generate activity matrix when calculating
    origin decoders.
    
    Returns the accumulated output over that time.

    :param J: theano function object for the input current
    :param Neuron neuron: population of neurons from which to accumulate data
    :param float time: length of time to simulate population for (s)
    :param float init_time: run neurons for this long before collecting data
                            to get rid of startup transients (s)

    """
    # create internal state variable to keep track of number of spikes
    total = theano.shared(np.zeros(neurons.size).astype('float32'), 
                          name='neuron.total')
    
    ### make the standard neuron update function

    # updates is dictionary of variables returned by neuron.update
    updates = neurons.update(J.astype('float32'), dt)

    # update all internal state variables listed in updates
    tick = theano.function([], [], updates=updates)
    
    ### make a variant that also includes computing the total output
    # add another internal variable to change to updates dictionary
    updates[total] = total + neurons.output

    # create theano function that does it all
    accumulate_spikes = theano.function([], [], updates=updates)
    #, mode=theano.Mode(optimizer=None, linker='py'))

    # call the standard one a few times to avoid startup transients
    tick.fn(n_calls = int(init_time / dt))

    # call the accumulator version a bunch of times
    accumulate_spikes.fn(n_calls = int(time / dt))

    return total.get_value().astype('float32') / time


class Neuron(object):
    """Superclass for neuron models.

    All neurons must implement an update function,
    and should most likely define a more complicated reset function.

    """

    def __init__(self, size):
        """Constructor for neuron model superclass.

        :param int size: number of neurons in this population

        """
        self.output = np.zeros(size).astype('float32')

    @property
    def shape(self):
        return len(self.output)

    def __len__(self):
        return self.output.size

    def hashable_state(self):
        return ()


    def reset(self):
        """Reset the state of the neuron."""
        self.output[:] = 0


    def step(self, input_current):
        """All neuron subclasses must have an update function.

        The update function takes in input_current and returns
        activity information.

        """
        raise NotImplementedError()
