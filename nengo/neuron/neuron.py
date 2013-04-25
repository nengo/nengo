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

    def __len__(self):
        return self.output.size

    def step(self, input_current):
        """All neuron subclasses must have an update function.

        The update function takes in input_current and returns
        activity information.

        """
        raise NotImplementedError()
