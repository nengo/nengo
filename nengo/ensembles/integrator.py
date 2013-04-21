
from ..networks import array

def make(network, name, neurons=100, dimensions=1, tau_feedback=0.1):
    """This constructs an integrator of the specified number of dimensions. 
    It requires an input of that number of dimensions after construction.
    
    :param str network:
        Name of the parent network
    :param str name:
        Name of the integrator
    :param int neurons:
        Number of neurons in the integrator
    :param int dimensions:
        Number of dimensions for the integrator
    :param float tau_feedback:
         Post-synaptic time constant of the integrative feedback, 
         in seconds (longer -> slower change but better value retention)
    """
    net = _check_parameters(network, name, neurons, dimensions, tau_feedback)
    
    if dimensions < 8:
        integrator = net.make(name, neurons, dimensions)
    else:
        integrator = array.make(name, dimensions, int(neurons/dimensions), 1)
    
    recurrent_connection = net.connect(integrator, integrator, 
                                       filter=tau_feedback)
    
    return integrator, recurrent_connection

def _check_parameters(network, name, neurons, dimensions, tau_feedback):
    if isinstance(network, str):
        net = Model.get(network, None)
        if net is None:
            raise ValueError("%s doesn't exist, can't add integrator" % network)
    else:
        net = network
    if not isinstance(network, Network):
        raise valueError("'Network' is not a Network object") 
    if network.get(name, None) is not None:
        raise ValueError("That name is already taken in this network")

    if neurons < dimensions:
        raise ValueError("Must have at least one neuron per dimension")
    
    return net
