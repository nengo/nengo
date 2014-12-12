import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.synapses import Lowpass, Synapse
from nengo.utils.compat import is_number


class SimSynapse(Operator):
    """Simulate a Synapse object."""
    def __init__(self, input, output, synapse):
        self.input = input
        self.output = output
        self.synapse = synapse

        self.sets = []
        self.incs = []
        self.reads = [input]
        self.updates = [output]

    def make_step(self, signals, dt, rng):
        input = signals[self.input]
        output = signals[self.output]
        step_f = self.synapse.make_step(dt, output)

        def step(input=input):
            step_f(input)

        return step


def filtered_signal(model, owner, sig, synapse):
    # Note: we add a filter here even if synapse < dt,
    # in order to avoid cycles in the op graph. If the filter
    # is explicitly set to None (e.g. for a passthrough node)
    # then cycles can still occur.
    if is_number(synapse):
        synapse = Lowpass(synapse)
    assert isinstance(synapse, Synapse)
    model.build(synapse, owner, sig)
    return model.sig[owner]['synapse_out']


@Builder.register(Synapse)
def build_synapse(model, synapse, owner, input_signal):
    model.sig[owner]['synapse_in'] = input_signal
    model.sig[owner]['synapse_out'] = Signal(
        np.zeros(input_signal.shape),
        name="%s.%s" % (input_signal.name, synapse))

    model.add_op(SimSynapse(input=model.sig[owner]['synapse_in'],
                            output=model.sig[owner]['synapse_out'],
                            synapse=synapse))
