import numpy as np
import math

from output import Output
from output import FunctionOutput

class Node(object):
    _default_output_idx = 0

    @property
    def default_output(self):
        return self._outputs[self._default_output_idx]

    def __init__(self, name, output):
        self.name = name
        if callable(output):
            self._outputs = [FunctionOutput(output)]
        else:
            self._outputs = [Output(output)]
        self.t = 0.0

    def _build(self, state, dt):
        for _output in self._outputs:
            state[output] = _output.build()

    def step(self):
        pass

    def _reset(self, *args):
        self.t = 0
        for output in self._outputs:
            state_t[output] = output.reset()

    def _step(self, state_t, state_tm1, dt):
        for output in self._outputs:
            state_t[output] = output.step()
        self.t += dt
        self.step()


class TimeNode(Node):
    def __init__(self, name, output):
        self.name = name
        if callable(output):
            dimensions = np.asarray(output(0)).shape
            self._outputs = [FunctionOutput(output, dimensions)]
        else:
            self._outputs = [Output(output)]
        self.t = 0.0

    def _build(self, state, dt):
        self._reset(state)

    def _reset(self, state):
        self.t = 0
        for output in self._outputs:
            state[output] = output.step(0)

    def _step(self, state_t, state_tm1, dt):
        for output in self._outputs:
            state_t[output] = output.step(self.t)
        self.t += dt
        self.step()
