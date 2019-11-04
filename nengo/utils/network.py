from .magic import decorator


@decorator
def with_self(method, network, args, kwargs):
    """Wraps a method with ``with network:``.

    This makes it easy to add methods to a network that create new
    Nengo objects. Instead of writing ``with self`` at the top of the method
    and indenting everything over, you can instead use this decorator.

    Examples
    --------
    The two methods in the following class do the same thing:

    .. testcode::

       from nengo.utils.network import with_self

       class MyNetwork(nengo.Network):
           def add_one_1(self):
               with self:
                   node = nengo.Node(output=1)

           @with_self
           def add_one_2(self):
               node = nengo.Node(output=1)
    """
    with network:
        return method(*args, **kwargs)


def activate_direct_mode(network):
    """Activates direct mode for a network.

    This sets the neuron type of all ensembles to a `nengo.Direct`
    instance unless:

    - there is a connection to or from the ensemble's neurons
    - there is a probe on an ensemble's neurons
    - the ensemble has a connection with a learning rule attached.

    Parameters
    ----------
    network : Network
        Network to activate direct mode for.
    """
    # pylint: disable=import-outside-toplevel
    # imported here to avoid circular imports
    from nengo.ensemble import Neurons
    from nengo.neurons import Direct

    requires_neurons = set()

    for c in network.all_connections:
        if isinstance(c.pre_obj, Neurons):
            requires_neurons.add(c.pre_obj.ensemble)
        if isinstance(c.post_obj, Neurons):
            requires_neurons.add(c.post_obj.ensemble)
        if c.learning_rule_type is not None:
            requires_neurons.add(c.pre_obj)
            requires_neurons.add(c.post_obj)

    for p in network.all_probes:
        if isinstance(p.obj, Neurons):
            requires_neurons.add(p.obj.ensemble)

    for e in network.all_ensembles:
        if e not in requires_neurons:
            e.neuron_type = Direct()
