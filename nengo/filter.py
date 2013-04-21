import collections

import numpy as np

class Filter:
    """Filter an arbitrary value"""

    def __init__(self, pstc, name=None, source=None, dimension=None):
        """
        :param float pstc:
        :param string name:
        :param source:
        :type source:
        :param tuple shape:
        """
        self.pstc = pstc
        self.source = source
        if source == None:
            if dimension == None:
                raise Exception("Must pass a dimension to filter if no source given")
            self.value = np.array([0.0 for i in range(dimension)])
        else:
            self.value = np.array([0.0 for i in range(source.shape[0])])
        

    def filter(self, dt, source=None):
        """
        :param float dt: the timestep of the update
        """
        if self.source:
            source_input = self.source
        else:
            source_input = source
        
        if self.pstc >= dt:
            decay = np.cast(np.exp(-dt / self.pstc), self.value.dtype)
            value_new = decay * self.value + (1 - decay) * source_input
            return collections.OrderedDict([(self.value, value_new.astype('float32'))])
        else:
            ### no filtering (pstc = 0), so just make the value the source
            return collections.OrderedDict([(self.value, self.source.astype('float32'))])

