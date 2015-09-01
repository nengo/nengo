import nengo
from nengo.spa.module import Module
from nengo.utils.compat import iteritems


def make_parse_func(func, vocab):
    """Create a function that calls func and parses the output in vocab."""

    def parse_func(t):
        return vocab.parse(func(t)).v

    return parse_func


class Input(Module):
    """A SPA module for providing external inputs to other modules.

    The arguments are indicate the module input name and the function
    to execute to generate inputs to that module.  The functions should
    always return strings, which will then be parsed by the relevant
    Vocabulary.  For example::

        def input1(t):
            if t < 0.1:
                return 'A'
            else:
                return '0'

        Input(vision=input1, task='X')

    """

    def __init__(self, label=None, seed=None, add_to_container=None, **kwargs):
        super(Input, self).__init__(label, seed, add_to_container)
        self.kwargs = kwargs
        self.input_nodes = {}

    def on_add(self, spa):
        """Create the connections and nodes."""
        Module.on_add(self, spa)

        for name, value in iteritems(self.kwargs):
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
