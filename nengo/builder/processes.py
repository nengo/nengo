import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.processes import Process
from nengo.synapses import Synapse


class SimProcess(Operator):
    """Simulate a Process object."""
    def __init__(self, process, input, output, t, mode='set', tag=None):
        self.process = process
        self.input = input
        self.output = output
        self.t = t
        self.mode = mode
        self.tag = tag

        self.reads = [t, input] if input is not None else [t]
        self.sets = []
        self.incs = []
        self.updates = []
        if mode == 'update':
            self.updates = [output] if output is not None else []
        elif mode == 'inc':
            self.incs = [output] if output is not None else []
        elif mode == 'set':
            self.sets = [output] if output is not None else []
        else:
            raise ValueError("Unrecognized mode %r" % mode)

    def _descstr(self):
        return '%s, %s -> %s' % (self.process, self.input, self.output)

    def make_step(self, signals, dt, rng):
        t = signals[self.t]
        input = signals[self.input] if self.input is not None else None
        output = signals[self.output] if self.output is not None else None
        shape_in = input.shape if input is not None else (0,)
        shape_out = output.shape if output is not None else (0,)
        rng = self.process.get_rng(rng)
        step_f = self.process.make_step(shape_in, shape_out, dt, rng)
        inc = self.mode == 'inc'

        def step_simprocess():
            result = (step_f(t.item(), input) if input is not None else
                      step_f(t.item()))
            if output is not None:
                if inc:
                    output[...] += result
                else:
                    output[...] = result

        return step_simprocess


@Builder.register(Process)
def build_process(model, process, sig_in=None, sig_out=None, inc=False):
    model.add_op(SimProcess(
        process, sig_in, sig_out, model.time, mode='inc' if inc else 'set'))


@Builder.register(Synapse)
def build_synapse(model, synapse, sig_in, sig_out=None):
    if sig_out is None:
        sig_out = Signal(
            np.zeros(sig_in.shape), name="%s.%s" % (sig_in.name, synapse))

    model.add_op(SimProcess(
        synapse, sig_in, sig_out, model.time, mode='update'))
    return sig_out
