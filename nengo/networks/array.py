
from ..nef import Model

def make(network, name, ensembles, neurons, dimensions, **ensemble_params):
    """Construct a network containing an array of ensembles.

    This template enables constructing networks full D (# of dimensions)
    independent populations of neurons.  These are faster to construct but
    cannot compute all the same nonlinear functions as a single large
    population with D dimensions.

    :param network: Parent network for the array
    :type network: str
    :param name: Name of the network array
    :type name: str
    :param ensembles: Number of ensembles in the array
    :type ensembles: int
    :param dimensions: Number of dimensions in each of the ensembles
    :type dimensions: int
    :param ensemble_params: Dictionary of parameters for the ensembles
    :type ensemble_params: dict
    """

    parent_net = check_parameters(
        network, name, ensembles, neurons, dimensions, **ensemble_params)

    net = parent_net.make_network(name)
    for i in xrange(ensembles):
        net.make_ensemble(str(i), neurons, dimensions, **ensemble_params)

    return net

def check_parameters(network, name, neurons, dimensions, ensemble_params):
    if isinstance(network, str):
        net = Model.get(network, None)
        if net is None:
            raise ValueError("The network \"%s\" does not exist" % network)
    else:
        net = network
    if not isinstance(network, Network):
        raise valueError("\"network\" is not a Network object")
    if network.get(name, None) is not None:
        raise ValueError(
            "The name \"%s\" is already taken in the network \"%s\""
            % (name, network.name))

    return net

