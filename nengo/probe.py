import numpy as np
import collections

from filter import make_filter
import nengo

def is_probe(obj):
    return isinstance(obj, (ListProbe, ArrayProbe))

class ListProbe(object):
    """A class to record from things (i.e., origins).

    """
    buffer_size = 1000

    def __init__(self, target, sample_every, static):
        self.target = target
        self.sample_every = sample_every
        self.static = static
        # XXX use sample_every !

    def _build(self, state, dt):
        self._reset(state)
        pass

    def _reset(self, state):
        self.data = []
        self.simtime = 0

    def _step(self, old_state, new_state, dt):
        self.data.append(old_state[self.target])
        self.simtime += dt

class ArrayProbe(object):
    """A class to record from Outputs

    """
    buffer_size = 1000

    def __init__(self, target, sample_every, static, filter=nengo.pstc(0.01)):
        """
        """
        self.target = target
        self.sample_every = sample_every
        self.static = static
        self.filter=make_filter(filter, dimensions=target.dimensions)

    def _build(self, state, dt):
        """
        """
        self._reset(state)
        pass

    def _reset(self, state):
        """
        """
        self.simtime = 0.0

        # create array to store the data over many time steps
        self.data = np.zeros((self.buffer_size, self.target.dimensions))
        # index of the last sample taken 
        self.i = -1 

    def _step(self, old_state, new_state, dt):
        """
        """
        i_samp = int(self.simtime / self.sample_every)
        if i_samp > self.i:
            # we're as close to a sample point as we're going to get,
            # so take a sample
            if i_samp >= len(self.data):
                # increase the buffer
                self.data = np.vstack(
                    [self.data, np.zeros((self.buffer_size,)
                                         + self.data.shape[1:])])

            # record the filtered value
            self.data[self.i+1:i_samp+1] = \
                self.filter.filter(dt=dt, signal=old_state[self.target]).flatten()
            self.i = i_samp
        self.simtime += dt

    def get_data(self):
        """
        """
        return self.data[:self.i+1]
