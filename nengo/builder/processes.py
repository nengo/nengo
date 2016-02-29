from nengo.builder.builder import Builder
from nengo.builder.operator import Operator
from nengo.processes import Process


class SimProcess(Operator):
    """Simulate a Process object."""
    def __init__(self, process, input, output, t, inc=False, tag=None):
        self.process = process
        self.input = input
        self.output = output
        self.t = t
        self.inc = inc
        self.tag = tag

        if inc:
            self.sets = []
            self.incs = [output] if output is not None else []
        else:
            self.sets = [output] if output is not None else []
            self.incs = []
        self.reads = [t, input] if input is not None else [t]
        self.updates = []

    def __str__(self):
        return 'SimProcess(%s, %s -> %s%s)' % (
            self.process, self.input, self.output, self._tagstr)

    def make_step(self, signals, dt, rng):
        t = signals[self.t]
        input = signals[self.input] if self.input is not None else None
        output = signals[self.output] if self.output is not None else None
        size_in = input.size if input is not None else 0
        size_out = output.size if output is not None else 0
        rng = self.process.get_rng(rng)
        step_f = self.process.make_step(size_in, size_out, dt, rng)
        inc = self.inc

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
    model.add_op(SimProcess(process, sig_in, sig_out, model.time, inc=inc))
