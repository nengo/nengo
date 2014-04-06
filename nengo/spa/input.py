import nengo
from nengo.spa.module import Module
from nengo.utils.compat import iteritems


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
    def __init__(self, **kwargs):
        super(Input, self).__init__()
        self.kwargs = kwargs
        self.input_nodes = {}

    def on_add(self, spa):
        """Create the connections and nodes."""
        Module.on_add(self, spa)

        for name, value in iteritems(self.kwargs):
            target, vocab = spa.get_module_input(name)
            if callable(value):
                val = lambda t, value=value: vocab.parse(value(t)).v
            else:
                val = vocab.parse(value).v

            with self:
                node = nengo.Node(val, label='input_%s' % name)
            self.input_nodes[name] = node

            with spa:
                nengo.Connection(node, target, synapse=None)
