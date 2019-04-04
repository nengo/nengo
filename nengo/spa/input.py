import nengo
from nengo.spa.module import Module


def make_parse_func(func, vocab):
    """Create a function that calls func and parses the output in vocab."""

    def parse_func(t):
        return vocab.parse(func(t)).v

    return parse_func


class Input(Module):
    """A SPA module for providing external inputs to other modules.

    The parameters passed to this module indicate the module input name
    and the function to execute to generate inputs to that module.
    The functions should always return strings, which will then be parsed
    by the relevant vocabulary. For example::

        def input1(t):
            if t < 0.1:
                return 'A'
            else:
                return '0'

        spa_net.input = spa.Input(vision=input1, task='X')

    will create two inputs:

    1. an input to the ``vision`` module, which for the first 0.1 seconds
       is the value associated with the ``'A'`` semantic pointer and then
       a vector of all zeros, and
    2. an input to the ``task`` module which is always the value associated
       with the ``'X'`` semantic pointer.

    Parameters
    ----------
    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    def __init__(self, label=None, seed=None, add_to_container=None, **kwargs):
        super(Input, self).__init__(label, seed, add_to_container)
        self.kwargs = kwargs
        self.input_nodes = {}

    def on_add(self, spa):
        """Create the connections and nodes."""
        Module.on_add(self, spa)

        for name, value in self.kwargs.items():
            target, vocab = spa.get_module_input(name)
            if callable(value):
                val = make_parse_func(value, vocab)
            else:
                val = vocab.parse(value).v

            with self:
                node = nengo.Node(val, label='input_%s' % name)
            self.input_nodes[name] = node

            with spa:
                nengo.Connection(node, target, synapse=None)
