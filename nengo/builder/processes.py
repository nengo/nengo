from nengo.builder import Builder, Operator, Signal
from nengo.processes import Process
from nengo.rc import rc


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
    mode : str, optional
        Denotes what type of update this operator performs.
        Must be one of ``'update'``, ``'inc'`` or ``'set'``.
    tag : str, optional
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

    def __init__(self, process, input, output, t, mode="set", state=None, tag=None):
        super().__init__(tag=tag)
        self.process = process
        self.mode = mode

        assert output is not None

        self.reads = [t, input] if input is not None else [t]
        self.sets = []
        self.incs = []
        self.updates = []
        if mode == "update":
            self.updates.extend([output])
        elif mode == "inc":
            self.incs.extend([output])
        elif mode == "set":
            self.sets.extend([output])
        else:
            raise ValueError("Unrecognized mode %r" % mode)

        self.state = {} if state is None else state
        self.updates.extend(self.state.values())

    @property
    def input(self):
        return None if len(self.reads) == 1 else self.reads[1]

    @property
    def output(self):
        if len(self.updates) <= len(self.incs) <= len(self.sets) <= 0:
            return None
        elif self.mode == "update":
            return self.updates[0]
        elif self.mode == "inc":
            return self.incs[0]
        elif self.mode == "set":
            return self.sets[0]

    @property
    def t(self):
        return self.reads[0]

    def _descstr(self):
        return "%s, %s -> %s" % (self.process, self.input, self.output)

    def make_step(self, signals, dt, rng):
        t = signals[self.t]
        input = signals[self.input] if self.input is not None else None
        output = signals[self.output]
        shape_in = input.shape if input is not None else (0,)
        shape_out = output.shape
        rng = self.process.get_rng(rng)
        state = {name: signals[sig] for name, sig in self.state.items()}
        step_f = self.process.make_step(shape_in, shape_out, dt, rng, state)
        args = (t,) if input is None else (t, input)

        if self.mode == "inc":

            def step_simprocess():
                output[...] += step_f(args[0].item(), *args[1:])

        else:

            def step_simprocess():
                output[...] = step_f(args[0].item(), *args[1:])

        return step_simprocess


@Builder.register(Process)
def build_process(model, process, sig_in=None, sig_out=None, mode="set", **kwargs):
    """Builds a `.Process` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    process : Process
        Process to build.
    sig_in : Signal, optional
        The input signal, or None if no input signal.
    sig_out : Signal, optional
        The output signal, or None if no output signal.
    mode : "set" or "inc" or "update", optional
        The ``mode`` of the built `.SimProcess`.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Process` instance.
    """
    if sig_out is None:
        sig_out = Signal(shape=sig_in.shape, name="%s.%s" % (sig_in.name, process))

    shape_in = sig_in.shape if sig_in is not None else (0,)
    shape_out = sig_out.shape if sig_out is not None else (0,)
    dtype = (
        sig_out.dtype
        if sig_out is not None
        else sig_in.dtype
        if sig_in is not None
        else rc.float_dtype
    )
    state_init = process.make_state(
        shape_in, shape_out, model.dt, dtype=dtype, **kwargs
    )
    state = {}
    for name, value in state_init.items():
        state[name] = Signal(value)
        model.sig[process]["_state_" + name] = state[name]

    model.add_op(
        SimProcess(process, sig_in, sig_out, model.time, mode=mode, state=state)
    )

    return sig_out
