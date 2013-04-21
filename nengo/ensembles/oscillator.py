    
def make(network, name, neurons=100, dimensions=2, frequency=5, 
         tau_feedback=0.1, controlled=False):
    """The typical oscillator needs an input (2D), which can be used to start it 
        off from rest.  Once it has started, the input can be set to zero. 
        The controlled oscillator also needs a 1D control input that changes 
        the frequency (and direction) of the oscillation. 
        http://nengo.ca/docs/html/demos/oscillator.html Tips & tricks.
    
    :param str network:
        Name of the parent network
    :param str name:
        Name of the oscillator
    :param int neurons:
        Number of neurons in the integrator
    :param int dimensions:
        Number of dimensions for the integrator
    :param float frequency:
        The frequence of the oscillator in Hz
    :param float tau_feedback:
         Post-synaptic time constant of the integrative feedback, 
         in seconds (longer -> slower change but better value retention)
    :param boolean controlled:
        Indication whether the freqency of the oscillator will be controlled
        by another input.  If so, the last dimension will be the control.
    """
    net = _check_parameters(network, name, neurons, dimensions, frequency, 
                            tau_feedback, controlled)

    frequency = frequency*2*math.pi;
    if (controlled):
        oscillator = net.make(name, neurons, dimensions=3)
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        connection = net.connect(oscillator, oscillator, feedback, A, 
                                 pstc=tau_feedback)
             
    else:
        oscillator = net.make(name, neurons, dimensions=2)
        A = [[1, -frequency*tau_feedback], [frequency*tau_feedback, 1]]
        connection = net.connect(oscillator, oscillator, A, pstc=tau_feedback)
    
    return oscillator, connection

def feedback(x):
    return x[0]+x[2]*x[1], x[1]-x[2]*x[0], 0

def _check_parameters(network, name, neurons, dimensions, frequency, 
         tau_feedback, controlled):
    if isinstance(network, str):
        net = Model.get(network, None)
        if net is None:
            raise ValueError("%s doesn't exist, can't add oscillator" % network)
    else:
        net = network
    if not isinstance(network, Network):
        raise valueError("'Network' is not a Network object") 
    if network.get(name, None) is not None:
        raise ValueError("That name is already taken in this network")

    if frequency<1: raise ValueError('Must have a positive frequency')
    
    return net
