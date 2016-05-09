import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.processes import Process
from nengo.synapses import Synapse


class SimProcess(Operator):
    """Simulate a process.

    Parameters
    ----------
    process : Process
        The `.Process` to simulate.
    input : Signal or None
        Input to the process, or None if no input.
    output : Signal or None
        Output from the process, or None if no output.
    t : Signal
        The signal associated with the time (a float, in seconds).
    mode : str, optional (Default: ``'set'``)
        Denotes what type of update this operator performs.
        Must be one of ``'update'``, ``'inc'`` or ``'set'``.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    input : Signal or None
        Input to the process, or None if no input.
    mode : str
        Denotes what type of update this operator performs.
    output : Signal or None
        Output from the process, or None if no output.
    process : Process
        The `.Process` to simulate.
    t : Signal
        The signal associated with the time (a float, in seconds).
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[output] if output is not None and mode=='set' else []``
    2. incs ``[output] if output is not None and mode=='inc' else []``
    3. reads ``[t, input] if input is not None else [t]``
    4. updates ``[output] if output is not None and mode=='update' else []``
    """
    def __init__(self, process, input, output, t, mode='set', tag=None):
        super(SimProcess, self).__init__(tag=tag)
        self.process = process
        self.input = input
        self.output = output
        self.t = t
        self.mode = mode

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
    """Builds a `.Process` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    process : Process
        Process to build.
    sig_in : Signal, optional (Default: None)
        The input signal, or None if no input signal.
    sig_out : Signal, optional (Default: None)
        The output signal, or None if no output signal.
    inc : bool, optional (Default: False)
        Whether `.SimProcess` should be made with
        ``mode='inc'` (True) or ``mode='set'`` (False).

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Process` instance.
    """

    model.add_op(SimProcess(
        process, sig_in, sig_out, model.time, mode='inc' if inc else 'set'))


@Builder.register(Synapse)
def build_synapse(model, synapse, sig_in, sig_out=None):
    """Builds a `.Synapse` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    synapse : Synapse
        Synapse to build.
    sig_in : Signal
        The input signal.
    sig_out : Signal, optional (Default: None)
        The output signal. If None, a new output signal will be
        created and returned.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Synapse` instance.
    """
    if sig_out is None:
        sig_out = Signal(
            np.zeros(sig_in.shape), name="%s.%s" % (sig_in.name, synapse))

    model.add_op(SimProcess(
        synapse, sig_in, sig_out, model.time, mode='update'))
    return sig_out
