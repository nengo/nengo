import collections

import numpy as np
import collections

import numpy as np

from .filter import Filter

def is_probe(obj):
    return isinstance(obj, ListProbe)


class ListProbe(object):
    """A class to record from things (i.e., origins).

    """
    buffer_size = 1000

    def __init__(self, target, sample_every, static):
        self.target = target
        self.sample_every = sample_every
        self.static = static
        # XXX use sample_every !
        #self.filter = Filter(pstc, source=target)

    def _build(self, state, dt):
        #self.filter._build(state, dt)
        pass

    def _reset(self, *args):
        self.data = []
        self.simtime = 0

    def _step(self, state_t, state_tm1, dt):
        self.data.append(state_t[self.target])
        self.simtime += dt


if 0:
    def update(self):
        """
        """
        i_samp = int(self.t / self.dt_sample)
        if i_samp > self.i:
            # we're as close to a sample point as we're going to get,
            # so take a sample
            if i_samp >= len(self.data):
                # increase the buffer
                self.data = np.vstack(
                    [self.data, np.zeros((self.buffer_size,)
                                         + self.data.shape[1:])])

            # record the filtered value
            self.data[self.i+1:i_samp+1] = self.filter.value.get_value()
            self.i = i_samp

    def get_data(self):
        """
        """
        return self.data[:self.i+1]


        # create array to store the data over many time steps
        self.data = np.zeros((self.buffer_size,) + target.get_value().shape)
        self.i = -1 # index of the last sample taken

        # create a filter to filter the data
