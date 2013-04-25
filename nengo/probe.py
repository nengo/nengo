import numpy as np
import collections

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

    def _build(self, state, dt):
        self._reset()
        pass

    def _reset(self):
        self.data = []
        self.simtime = 0

    def _step(self, old_state, new_state, dt):
        self.data.append(old_state[self.target])
        self.simtime += dt
