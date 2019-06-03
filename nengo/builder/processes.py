import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Reset, Copy, DotInc, ElementwiseInc
from nengo.processes import Process
from nengo.synapses import LinearFilter, Synapse


state_prefix = '_state_'


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
    def __init__(self, process, input, output, t, mode='set', state=None,
                 tag=None):
        super().__init__(tag=tag)
        self.process = process
        self.mode = mode

        self.reads = [t, input] if input is not None else [t]
        self.sets = []
        self.incs = []
        self.updates = []
        if mode == 'update':
            self.updates.extend([output] if output is not None else [])
        elif mode == 'inc':
            self.incs.extend([output] if output is not None else [])
        elif mode == 'set':
            self.sets.extend([output] if output is not None else [])
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
        elif self.mode == 'update':
            return self.updates[0]
        elif self.mode == 'inc':
            return self.incs[0]
        elif self.mode == 'set':
            return self.sets[0]

    @property
    def t(self):
        return self.reads[0]

    def _descstr(self):
        return '%s, %s -> %s' % (self.process, self.input, self.output)

    def make_step(self, signals, dt, rng):
        t = signals[self.t]
        input = signals[self.input] if self.input is not None else None
        output = signals[self.output] if self.output is not None else None
        shape_in = input.shape if input is not None else (0,)
        shape_out = output.shape if output is not None else (0,)
        rng = self.process.get_rng(rng)
        state = {name: signals[sig] for name, sig in self.state.items()}
        step_f = self.process.make_step(
            shape_in, shape_out, dt, rng, state=state)
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
    shape_in = sig_in.shape if sig_in is not None else (0,)
    shape_out = sig_out.shape if sig_out is not None else (0,)
    state_init = process.allocate(shape_in, shape_out, model.dt)
    state = {}
    for name, value in state_init.items():
        state[name] = Signal(value)
        model.sig[process][state_prefix + name] = state[name]

    model.add_op(SimProcess(
        process, sig_in, sig_out, model.time, state=state,
        mode='inc' if inc else 'set'))


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

    shape_in = sig_in.shape if sig_in is not None else (0,)
    shape_out = sig_out.shape if sig_out is not None else (0,)
    state_init = synapse.allocate(shape_in, shape_out, model.dt)
    state = {}
    for name, value in state_init.items():
        state[name] = Signal(value)
        model.sig[synapse][state_prefix + name] = state[name]

    model.add_op(SimProcess(
        synapse, sig_in, sig_out, model.time, state=state, mode='update'))
    return sig_out


class SimStateSpace(Operator):
    """Simulate ``x[t+1] = Ax[t] + Bu[t]`` update.

    Parameters
    ----------
    state : Signal
        State signal (``x[t]``).
    input : Signal
        Input signal (``u[t]``).
    next_state : Signal
        State at end of time-step (``x[t+1]``).
    A : np.ndarray
        A matrix for linear system.
    B : np.ndarray
        B matrix for linear system.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    state : Signal
        State signal (``x[t]``).
    input : Signal
        Input signal (``u[t]``).
    next_state : Signal
        State at end of time-step (``x[t+1]``).
    A : np.ndarray
        A matrix for linear system.
    B : np.ndarray
        B matrix for linear system.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[state, input]``
    4. updates ``[next_state]``
    """

    def __init__(self, state, input, next_state, A, B, tag=None):
        super().__init__(tag=tag)
        self.A = A
        self.B = B

        self.reads = [state, input]
        self.sets = []
        self.incs = []
        self.updates = [next_state]

    @property
    def state(self):
        return self.reads[0]

    @property
    def input(self):
        return self.reads[1]

    @property
    def next_state(self):
        return self.updates[0]

    def _descstr(self):
        return '%s = A%s + B%s' % (self.next_state, self.state, self.input)

    def make_step(self, signals, dt, rng):
        state = signals[self.state]
        input = signals[self.input]
        if input.ndim == 1:
            input = input[:, None]
        next_state = signals[self.next_state]
        AT = self.A.T  # transposed
        BT = self.B.T  # transposed

        def step_statespace():
            next_state[...] = np.dot(state, AT) + BT * input

        return step_statespace


@Builder.register(LinearFilter)
def build_state_space_filter(model, synapse, sig_in, sig_out=None):
    """Builds a `.LinearFilter` object into a state-space model.

    Uses separate operators to implement:

        ``x[t+1] = Ax[t] + Bu[t]``
        ``y[t] = Cx[t] + Du[t]``

    in the correct order (such that D is a true passthrough).

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

    shape_in = sig_in.shape if sig_in is not None else (0,)
    shape_out = sig_out.shape if sig_out is not None else (0,)
    state_init = synapse.allocate(shape_in, shape_out, model.dt)
    assert list(state_init.keys()) == ['X']  # this only supports state-space

    # TODO: refactor method='zoh'
    A, B, C, D = synapse._get_ss(dt=model.dt, method='zoh')

    # Handle stateless system first (only passthrough)
    # since 0-shaped matrices don't play nice with ops
    model.add_op(Reset(sig_out))
    if not np.allclose(D, 0):  # to avoid circular dependencies in op graph
        model.add_op(ElementwiseInc(
            Signal(D.squeeze(axis=(0, 1)), readonly=True),
            sig_in, sig_out, tag="LinearFilter:y += Du"))
    if len(A) == 0:
        return sig_out

    # We transpose X because DotInc only supports
    # matrix-vector multiplies because of OCL. But this is okay
    # because C is a vector.
    X = Signal(state_init['X'].T)  # keeps state at start of time-step
    model.sig[synapse][state_prefix + 'X'] = X

    Xnext = Signal(state_init['X'].T)  # remembers state at end of time-step
    model.sig[synapse][state_prefix + 'Xnext'] = Xnext

    # Set x based on update from last time-step
    # It is done this way because there are no primitive ops that
    # work in update mode
    model.add_op(Copy(Xnext, X))

    # Set y = Cx + Du
    model.add_op(DotInc(
        X, Signal(C.squeeze(axis=0), readonly=True),
        sig_out, tag="LinearFilter:y += Cx"))

    # Update next x = Ax + Bu
    # We need a custom operator here for two reasons:
    # (1) there are no primitive operators that do an 'update'
    # (2) dot(A, X) is a matrix-matrix multiply
    model.add_op(SimStateSpace(
        state=X, input=sig_in, next_state=Xnext,
        A=A, B=B, tag="LinearFilter:x = Ax + Bu"))

    return sig_out
