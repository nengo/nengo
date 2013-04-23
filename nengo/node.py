import numpy as np
import math

from output import Output

def is_node(obj):
    return isinstance(obj, (Node, TimeNode))

class Node(object):
    """
    A custom-node for feeding a function into the network.
    """

    def __init__(self, name, output=None):
        self.name = name

        if not callable(output):
            outfunc = lambda : output
        else:
            outfunc = output
        self.outputs = {Output(dimensions=len(outfunc()), name="output"):outfunc}

    def _build(self, state, dt):
        for output in self.outputs:
            state[output] = self.outputs[output]()

    def _reset(self, state_t):
        for output in self.outputs:
            state_t[output] = self.outputs[output]()

    def _step(self, state_t, state_tm1, dt):
        for output in self.outputs:
            state_tm1[output] = self.outputs[output]()

class TimeNode(object):
    """
    A custom-node for feeding a function-of-time into the network.
    """
    def __init__(self, name, func):
        self.name = name
        self._outputs = {Output(dimensions=len(func(0.0))):func}
        self.t = 0.0

    def _build(self, state, dt):
        self._reset(state)

    def _reset(self, state):
        self.t = 0
        for output in self.outputs:
            state[output] = self.outputs[output](self.t)

    def _step(self, state_t, state_tm1, dt):
        self.t += dt
        for output in self.outputs:
            state_tm1[output] = self.outputs[output](self.t)

