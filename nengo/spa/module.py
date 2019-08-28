import nengo


class Module(nengo.Network):
    """Base class for SPA Modules.

    Modules are networks that also have a list of inputs and outputs,
    each with an associated `.Vocabulary` (or a desired dimensionality for
    the vocabulary).

    The inputs and outputs are dictionaries that map a name to an
    (object, Vocabulary) pair. The object can be a `.Node` or an `.Ensemble`.
    """

    def __init__(self, label=None, seed=None, add_to_network=None):
        super().__init__(label, seed, add_to_network)
        self.inputs = {}
        self.outputs = {}

    def on_add(self, spa):
        """Called when this is assigned to a variable in the SPA network.

        Overload this when you want processing to be delayed until after
        the module is attached to the SPA network. This is usually for
        modules that connect to other things in the SPA model (such as
        the basal ganglia or thalamus).
        """
