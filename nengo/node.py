import numpy as np
import math

from output import Output

def is_node(obj):
    return isinstance(obj, (Node, TimeNode))

class Node(object):
    _default_output_idx = 0
    """
    A custom-node for feeding a function-of-time into the network.
    """

    @property
    def default_output(self):
        return self._outputs[self._default_output_idx]

    def __init__(self, name, output=None):
        self.name = name
        self._outputs = [Output()]
        self.output = output

    def _build(self, state, dt):
        for _output in self._outputs:
            state[output] = self.output

    def _reset(self, *args):
        for output in self._outputs:
            state_t[output] = self.output

    def _step(self, state_t, state_tm1, dt):
        for output in self._outputs:
            state_t[output] = self.step()


class TimeNode(object):
    """
    A custom-node for feeding a function-of-time into the network.
    """
    _default_output_idx = 0

    @property
    def default_output(self):
        return self._outputs[self._default_output_idx]
    def __init__(self, name, fn):
        self.name = name
        self._outputs = [Output()]
        self.t = 0.0
        self.fn = fn

    def _build(self, state, dt):
        self._reset(state)

    def _reset(self, state):
        self.t = 0
        v0 = self.fn(0)
        for output in self._outputs:
            state[output] = v0

    def _step(self, state_t, state_tm1, dt):
        self.t += dt
        vt = self.fn(self.t)
        for output in self._outputs:
            state_t[output] = vt

