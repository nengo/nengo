from ..output import Output

class Neuron(object):
    """Superclass for neuron models.

    All neurons must implement an update function,
    and should most likely define a more complicated reset function.

    """

    def __init__(self, size):
        """Constructor for neuron model superclass.

        :param int size: number of neurons in this population

        """
        self.output = Output(dimensions=size)
        self.size = size

    @property
    def shape(self):
        return len(self.output)

    def __len__(self):
        return self.output.size

    """def hashable_state(self):
        return ()"""

    def reset(self):
        """Reset the state of the neuron."""
        self.output[:] = 0


    def step(self, input_current):
        """All neuron subclasses must have an update function.

        The update function takes in input_current and returns
        activity information.

        """
        raise NotImplementedError()
