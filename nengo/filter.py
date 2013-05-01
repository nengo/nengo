import collections

import numpy as np

def make_filter(parameters, dimensions):

    parameters = parameters.copy()
    filter_type = parameters.pop('type').lower() 

    if filter_type == 'exponentialpsc':
        return ExponentialPSC(pstc=parameters['pstc'], dimensions=dimensions)

class ExponentialPSC:
    """Filter an arbitrary value"""

    def __init__(self, pstc, name=None, dimensions=None):
        """
        :param float pstc:
        :param string name:
        :param source:
        :type source:
        :param tuple shape:
        """
        self.pstc = pstc
        self.value = np.zeros(dimensions)

    def filter(self, signal, dt):
        """
        :param float dt: the timestep of the update
        """
        
        if self.pstc >= dt:
            decay = np.exp(-dt / self.pstc)
            value_new = decay * self.value + (1 - decay) * signal
            self.value = value_new

        return self.value

