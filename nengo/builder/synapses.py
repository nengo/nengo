import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.synapses import Synapse


class SimSynapse(Operator):
    """Simulate a Synapse object."""
    def __init__(self, input_sig, output_sig, synapse, tag=None):
        self.input = input_sig
        self.output = output_sig
        self.synapse = synapse
        self.tag = tag

        self.sets = []
        self.incs = []
        self.reads = [input_sig]
        self.updates = [output_sig]

    def __str__(self):
        return "SimSynapse(%s, %s -> %s%s)" % (
            self.synapse, self.input, self.output, self._tagstr)

    def make_step(self, signals, dt, rng):
        input_sig = signals[self.input]
        output_sig = signals[self.output]
        step_f = self.synapse.make_step(dt, output_sig)

        def step_simsynapse():
            step_f(input_sig)

        return step_simsynapse


@Builder.register(Synapse)
def build_synapse(model, synapse, input_sig, output=None):
    if output is None:
        output = Signal(np.zeros(input_sig.shape),
                        name="%s.%s" % (input_sig.name, synapse))

    model.add_op(SimSynapse(input_sig, output, synapse))
    return output
